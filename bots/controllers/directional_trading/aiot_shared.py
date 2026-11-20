"""Shared-account controller for AIOT on one MEXC hedge long.

Each bot keeps its own base from its own fills, keyed by its own client
order ids. It does not adopt the exchange long and it does not touch another
bot's orders or stop plans. Exits and this bot's stop are reduce-only and
never larger than the smaller of its own base and the current exchange long.

MEXC allows several partial stop plans (volType 1) on one long. The sum of
those volumes must be less than the closable quantity, and a new plan at the
same price updates the existing plan instead of creating a second one. This
controller places volType 1, tracks only its own id, and will not adopt or
cancel an id that already belonged to another plan.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from hummingbot.core.data_type.common import PositionMode, TradeType
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import DirectionalTradingControllerConfigBase
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy
from pydantic import Field, model_validator

from . import aiot_research as research
from .aiot_engine import STRATEGIES, f3_l20_strategy

_SHARE_TOLERANCE = Decimal("0.0001")
_STOP_REJECTED = (
    "AIOT shared stop was not accepted. MEXC allows several partial stop plans "
    "on one long. Their volumes must sum to less than the closable quantity, "
    "and the same price updates the existing plan instead of adding one."
)


def strategy_for(key: str):
    if key == "f3l20":
        return f3_l20_strategy()
    spec = STRATEGIES.get(key)
    if spec is None:
        raise ValueError(f"unknown strategy {key}")
    return spec


def shares_match(left, right) -> bool:
    try:
        return abs(Decimal(str(left)) - Decimal(str(right))) <= _SHARE_TOLERANCE
    except (ArithmeticError, TypeError, ValueError):
        return False


def parse_fill_row(row, contract_size=Decimal("1")):
    """One fill as (client order id, base). `dealVol` is contracts."""
    if not isinstance(row, dict):
        return None
    client_id = row.get("client_order_id") or row.get("externalOid") or row.get("external_oid")
    if client_id in (None, ""):
        return None
    try:
        step = Decimal(str(row.get("_contract_size", contract_size)))
        if row.get("base") not in (None, ""):
            base = Decimal(str(row["base"]))
        elif row.get("dealVol") not in (None, ""):
            base = Decimal(str(row["dealVol"])) * step
        else:
            return None
    except (ArithmeticError, TypeError, ValueError):
        return None
    if base < 0:
        return None
    return str(client_id), base


def reconcile_share(orders, rows, contract_size=Decimal("1")):
    """Match this bot's stored fills to history.

    Returns (status, net base, reason). status is `ok` or `halt`. History rows
    for other client ids are ignored. A missing id or a different size halts.
    """
    found = {}
    for row in rows or []:
        parsed = parse_fill_row(row, contract_size)
        if parsed is None:
            continue
        client_id, base = parsed
        found[client_id] = base
    net = Decimal("0")
    for client_id, slot in orders.items():
        if client_id not in found:
            return (
                "halt",
                None,
                f"AIOT shared has no fill for its order {client_id}. " "Entries and exits are halted.",
            )
        seen = found[client_id]
        stored = slot.get("base")
        if stored in (None, ""):
            return (
                "halt",
                None,
                f"AIOT shared order {client_id} has no stored size. " "Entries and exits are halted.",
            )
        if not shares_match(stored, seen):
            return (
                "halt",
                None,
                f"AIOT shared order {client_id} was {stored} and the fill " f"history says {seen}. Entries and exits are halted.",
            )
        if slot.get("side") == "sell":
            net -= seen
        else:
            net += seen
    if net < 0:
        net = Decimal("0")
    return "ok", net, ""


def client_ids_of(executor) -> List[str]:
    info = getattr(executor, "custom_info", None) or {}
    found = []
    if isinstance(info, dict):
        if info.get("order_id"):
            found.append(str(info["order_id"]))
        for row in info.get("held_position_orders") or []:
            if isinstance(row, dict) and row.get("client_order_id"):
                found.append(str(row["client_order_id"]))
    for item in getattr(executor, "client_order_ids", None) or []:
        if item not in (None, ""):
            found.append(str(item))
    out = []
    seen = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _executor_owner(executor) -> str:
    owner = getattr(executor, "controller_id", None)
    if owner in (None, ""):
        owner = getattr(getattr(executor, "config", None), "controller_id", None)
    if owner in (None, ""):
        return ""
    return str(owner)


def _shared_rung(parts) -> Optional[int]:
    if len(parts) <= 3:
        return None
    text = str(parts[3])
    if text in ("", "-") or not text.lstrip("-").isdigit():
        return None
    return int(text)


def _amount_text(value) -> str:
    return f"{Decimal(str(value)):.8f}"


class AiotSharedConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "aiot_shared"
    connector_name: str = Field(default="mexc_perpetual")
    trading_pair: str = Field(default="AIOT-USDT")
    total_amount_quote: Decimal = Field(default=Decimal("2500"))
    leverage: int = Field(default=1)
    position_mode: PositionMode = Field(default=PositionMode.HEDGE)
    size_mode: str = Field(default="max")
    stop_loss: Optional[Decimal] = Field(default=None)
    take_profit: Optional[Decimal] = Field(default=None)
    time_limit: Optional[int] = Field(default=None)
    trailing_stop: Optional[str] = Field(default=None)
    candles_connector: str = Field(default="mexc")
    candles_trading_pair: str = Field(default="AIOT-USDT")
    strategy: str = Field(default="fin_b")

    @model_validator(mode="after")
    def shared_strategy_is_one_account_slice(self):
        spec = strategy_for(self.strategy)
        if self.size_mode not in ("max", "fixed"):
            raise ValueError(f"unknown size_mode {self.size_mode}")
        if self.total_amount_quote <= 0:
            raise ValueError("total_amount_quote must be positive")
        if int(self.leverage) != 1:
            raise ValueError("AIOT leverage stays at 1")
        mode = self.position_mode
        if isinstance(mode, str):
            mode = PositionMode[mode.upper()]
        if mode is not PositionMode.HEDGE:
            raise ValueError("AIOT position_mode stays HEDGE")
        if self.size_mode == "fixed" and self.total_amount_quote != Decimal(str(spec.total_quote)):
            raise ValueError(
                f"strategy {self.strategy} is fixed at {spec.total_quote} quote, "
                f"got total_amount_quote {self.total_amount_quote}"
            )
        barriers = (self.stop_loss, self.take_profit, self.time_limit, self.trailing_stop)
        if any(value is not None for value in barriers):
            raise ValueError(
                "leave stop_loss, take_profit, time_limit and trailing_stop empty; " "the AIOT engine owns those exits"
            )
        return self


class AiotSharedController(research.AiotResearchController):
    def __init__(self, config: AiotSharedConfig, *args, **kwargs):
        research.DirectionalTradingControllerBase.__init__(self, config, *args, **kwargs)
        self.config = config
        self._spec = strategy_for(config.strategy)
        self._sleeves = [research.Sleeve(spec) for spec in self._spec.sleeves]
        self._applied_base = {}
        self._last_bar = None
        self._limit_live = set()
        self._stop = {}
        self._stop_inflight = False
        self._stop_sync_inflight = False
        self._stop_sync_next = 0.0
        self._stop_missing = 0
        self._stop_blocked = None
        self._live_long = None
        self._flat_streak = 0
        self._positions_seen = False
        self._fetch_inflight = False
        self._fetch_next = 0.0
        self._exit_reserved = Decimal("0")
        self._drawdown_cap = None
        self._entries_halted = False
        self._atr_fallback = False
        self._warned = set()
        self._leverage_ready = False
        self._leverage_inflight = False
        self._leverage_next = 0.0
        self._own_orders = {}
        self._share_halt = None
        self._share_checked = False
        self._history_inflight = False
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in config.id)
        self._state_path = Path("data") / f"aiot_shared_{safe_id}.json"
        self._load_state()

    def get_custom_info(self) -> dict:
        info = super().get_custom_info()
        info["share_base"] = str(self._own_base())
        info["share_halt"] = self._share_halt
        info["stop_blocked"] = self._stop_blocked
        info["stop_id"] = (self._stop or {}).get("id")
        info["own_orders"] = len(self._own_orders)
        return info

    def to_format_status(self) -> List[str]:
        lines = super().to_format_status()
        if self._share_halt:
            lines.append(self._share_halt)
        elif self._stop_blocked:
            lines.append(self._stop_blocked)
        return lines

    def _reconcile(self) -> None:
        self.confirm_share()
        if self._share_halt:
            return
        for executor in list(self.executors_info or []):
            if not self._apply_executor(executor):
                return

    def confirm_share(self) -> bool:
        if self._share_halt:
            return False
        if self._share_checked:
            return True
        if not self._own_orders:
            self._share_checked = True
            return True
        connector = self._connector()
        if connector is None:
            self._halt("AIOT shared cannot read its own order history. Entries and exits are halted.")
            return False
        history = getattr(connector, "fill_history", None)
        if callable(history):
            try:
                rows = history()
            except Exception as exc:
                self._halt(f"AIOT shared order history failed ({exc}). Entries and exits are halted.")
                return False
            if hasattr(rows, "__await__"):
                self._schedule_history(rows)
                return False
            return self._apply_history(rows)
        if callable(getattr(connector, "_request_order_by_client_id", None)):
            self._schedule_exchange_history(connector)
            return False
        self._halt("AIOT shared cannot read its own order history. Entries and exits are halted.")
        return False

    def _apply_history(self, rows) -> bool:
        verdict, net, reason = reconcile_share(self._own_orders, rows)
        if verdict != "ok":
            self._halt(reason)
            return False
        own = self._own_base()
        if not shares_match(own, net):
            self._halt(f"AIOT shared share {own} does not match its own fills {net}. " "Entries and exits are halted.")
            return False
        if len(self._sleeves) == 1:
            self._sleeves[0].base = float(net)
        self._share_checked = True
        return True

    def _schedule_history(self, coro) -> None:
        if self._history_inflight:
            coro.close()
            return
        self._history_inflight = True
        safe_ensure_future(self._finish_history(coro))

    async def _finish_history(self, coro) -> None:
        rows = None
        try:
            rows = await coro
        except Exception as exc:
            self._halt(f"AIOT shared order history failed ({exc}). Entries and exits are halted.")
        finally:
            self._history_inflight = False
        if rows is not None and not self._share_halt:
            self._apply_history(rows)

    def _schedule_exchange_history(self, connector) -> None:
        if self._history_inflight:
            return
        self._history_inflight = True
        safe_ensure_future(self._load_exchange_history(connector))

    async def _load_exchange_history(self, connector) -> None:
        rows = None
        try:
            size = Decimal("1")
            getter = getattr(connector, "get_contract_size", None)
            if callable(getter):
                size = Decimal(str(getter(self.config.trading_pair)))
            rows = []
            for client_id in list(self._own_orders):
                data = await connector._request_order_by_client_id(self.config.trading_pair, client_id)
                if not isinstance(data, dict):
                    raise ValueError(f"order {client_id} was not a record")
                row = dict(data)
                row.setdefault("client_order_id", client_id)
                row["_contract_size"] = size
                rows.append(row)
        except Exception as exc:
            rows = None
            self._halt(f"AIOT shared order history failed ({exc}). Entries and exits are halted.")
        finally:
            self._history_inflight = False
        if rows is not None and not self._share_halt:
            self._apply_history(rows)

    def _apply_executor(self, executor) -> bool:
        owner = _executor_owner(executor)
        if owner and owner != self.config.id:
            return True
        level = getattr(executor.config, "level_id", None) or ""
        parts = level.split(":")
        if len(parts) < 3 or parts[0] != "shared":
            return True
        sleeve = self._sleeve(parts[1])
        if sleeve is None:
            return True
        role = parts[2]
        filled = research._float_field(executor, "executed_amount_base")
        price = research._float_field(executor, "average_executed_price")
        done = research._is_done(executor)
        if role == "limit" and research._is_active(executor):
            self._limit_live.add(sleeve.spec.name)
        recorded = self._applied_base.get(executor.id, 0.0)
        rung = _shared_rung(parts)
        if role == "buy" and not done:
            if sleeve.base <= 0.0 and research._is_active(executor):
                sleeve.phase = "pending_entry"
                if rung is not None and 0 <= rung < len(sleeve.rungs):
                    sleeve.rungs[rung] = True
                if rung == -1:
                    sleeve.added = True
            return True
        if filled > recorded + 1e-12 and price > 0.0:
            ids = client_ids_of(executor)
            if len(ids) != 1:
                self._halt("AIOT shared filled an order it cannot tie to one client order id. " "Entries and exits are halted.")
                return False
            delta = filled - recorded
            self._applied_base[executor.id] = filled
            now = int(self._now())
            if role == "buy":
                reason = "entry"
                if rung == -1:
                    reason = "add"
                elif rung is not None and rung >= 0:
                    reason = f"rung{rung + 1}"
                sleeve.apply_buy_fill(price, delta, delta * price, now, reason, -1)
            elif role == "sell":
                sleeve.exit_reason = sleeve.exit_reason or "exit"
                sleeve.apply_partial_sell(price, delta, now, -1)
            elif role == "limit":
                sleeve.apply_limit_fill(price, delta, now, -1)
            self._own_orders[ids[0]] = {
                "side": "buy" if role == "buy" else "sell",
                "level": level,
                "base": _amount_text(filled),
                "price": _amount_text(price),
            }
            if role == "buy" and sleeve.spec.stop_from_average:
                self._refresh_average_stop(sleeve)
            return True
        if done and filled <= 0.0 and executor.id not in self._applied_base:
            self._applied_base[executor.id] = 0.0
            if role == "buy":
                sleeve.abort_buy(rung, self._buys_still_live(sleeve.spec.name, executor.id))
            elif role == "sell":
                sleeve.abort_sell()
            return True
        if role == "sell" and not done and research._is_active(executor) and sleeve.phase == "open":
            sleeve.phase = "pending_exit"
        return True

    def _refresh_average_stop(self, sleeve) -> None:
        quote = 0.0
        base = 0.0
        prefix = f"shared:{sleeve.spec.name}:"
        for slot in self._own_orders.values():
            if slot.get("side") != "buy" or not str(slot.get("level", "")).startswith(prefix):
                continue
            try:
                part = float(slot.get("base") or 0)
                price = float(slot.get("price") or 0)
            except (TypeError, ValueError):
                continue
            if part > 0.0 and price > 0.0:
                base += part
                quote += part * price
        if base <= 0.0:
            return
        sleeve.quote_in = quote
        sleeve.stop_price = sleeve._stop_from(quote / base)

    def _buys_still_live(self, sleeve_name: str, exclude: str) -> bool:
        prefix = f"shared:{sleeve_name}:buy"
        for executor in getattr(self, "executors_info", None) or []:
            if executor.id == exclude or not research._is_active(executor):
                continue
            level = getattr(executor.config, "level_id", None) or ""
            if level.startswith(prefix):
                return True
        return False

    def _intent_actions(self, intent):
        if self._share_halt:
            self._warn_once("share", self._share_halt)
            return [], False
        if intent.kind == "market_sell" and self._own_orders and not self._share_checked:
            return [], True
        return super()._intent_actions(intent)

    def _buy_actions(self, intent):
        if self._share_halt:
            self._warn_once("share", self._share_halt)
            return [], False
        if self._stop_blocked:
            self._warn_once("stop-share", self._stop_blocked)
            return [], True
        if self._own_orders and not self._share_checked:
            return [], True
        return super()._buy_actions(intent)

    def _create(self, side, amount, price, level_id, strategy):
        if isinstance(level_id, str) and level_id.startswith("aiot:"):
            level_id = "shared:" + level_id[len("aiot:") :]
        return super()._create(side, amount, price, level_id, strategy)

    def _cancel_limits(self, sleeve_name: str):
        actions = []
        prefix = f"shared:{sleeve_name}:limit"
        for executor in self.executors_info or []:
            level = getattr(executor.config, "level_id", None) or ""
            if level == prefix and research._is_active(executor):
                actions.append(research.StopExecutorAction(controller_id=self.config.id, executor_id=executor.id))
        self._limit_live.discard(sleeve_name)
        return actions

    def _entries_allowed(self) -> bool:
        if self._share_halt or self._stop_blocked:
            return False
        if self._own_orders and not self._share_checked:
            return False
        connector = self._connector()
        confirmed = getattr(connector, "_position_mode_confirmed", None)
        mode = getattr(connector, "position_mode", None)
        if confirmed and mode is not None and mode is not PositionMode.HEDGE:
            self._warn_once("mode", "AIOT shared entries stay off. The account is not HEDGE.")
            return False
        return super()._entries_allowed()

    def _clip_sell(self, sleeve_base):
        if self._share_halt:
            return Decimal("0")
        if self._own_orders and not self._share_checked:
            return None
        own = Decimal("0") if sleeve_base is None else Decimal(str(sleeve_base))
        live = self._live_long
        if self._tracks_positions() and live is None:
            return None
        if live is not None and own > live:
            self._warn_once(
                "capped-exit",
                f"AIOT shared exit capped at the exchange long {live}. This bot's share is {own}.",
            )
        return super()._clip_sell(sleeve_base)

    def _tracks_positions(self) -> bool:
        return callable(getattr(self._connector(), "_update_positions", None))

    def _own_base(self) -> Decimal:
        total = Decimal("0")
        for sleeve in self._sleeves:
            total += Decimal(str(sleeve.base))
        return total

    def _halt(self, reason: str) -> None:
        if self._share_halt:
            return
        self._share_halt = reason
        self.logger().error(reason)

    def _apply_live_position(self) -> None:
        connector = self._connector()
        self._schedule_position_fetch(connector)
        mapped = None
        if connector is not None:
            mapped = research.live_long_base(connector, self.config.trading_pair)
        if mapped is not None and mapped > 0:
            self._positions_seen = True
            self._flat_streak = 0
            self._live_long = mapped
            own = self._own_base()
            if own > mapped:
                self._warn_once(
                    "capped-long",
                    f"AIOT shared owns {own} and the exchange long is {mapped}. " "Exits are capped at the exchange long.",
                )
            if own > 0:
                self._upgrade_fallback_stop()
            return
        if not self._positions_seen or self._flat_streak < 2:
            self._live_long = None
            return
        self._live_long = Decimal("0")

    def _sync_sleeves_to_long(self, live, connector) -> None:
        self._warn_once(
            "adopt",
            "AIOT shared does not adopt or resize its share from the exchange long.",
        )

    def _arm_adopted_stop(self, sleeve, connector) -> None:
        self._warn_once("adopt", "AIOT shared does not adopt an existing long.")

    def _stop_amount(self):
        own = self._own_base()
        live = self._live_long
        if live is not None and own > live:
            self._warn_once(
                "capped-stop",
                f"AIOT shared stop capped at the exchange long {live}. This bot's share is {own}.",
            )
            own = live
        if own < 0:
            own = Decimal("0")
        cap = self._order_cap()
        if cap is not None:
            own = min(own, cap)
        return own

    def protective_close_action(self):
        if self._share_halt:
            self.logger().error(self._share_halt)
            return None
        if self._tracks_positions() and self._live_long is None:
            self.logger().error("AIOT shared drawdown close is waiting for the exchange long.")
            return None
        self._entries_halted = True
        amount = self._stop_amount()
        if amount is None or amount <= 0:
            return None
        name = self._sleeves[0].spec.name if self._sleeves else "core"
        return self._create(TradeType.SELL, float(amount), None, f"aiot:{name}:sell", ExecutionStrategy.MARKET)

    def _schedule_stops(self) -> None:
        if self._share_halt:
            return
        if self._own_orders and not self._share_checked:
            return
        super()._schedule_stops()

    async def _sync_stops(self, connector) -> None:
        try:
            rows = await connector.list_open_stop_orders(self.config.trading_pair)
        except Exception as exc:
            self.logger().warning(f"AIOT shared could not list stop orders: {exc}")
            return
        finally:
            self._stop_sync_inflight = False
        if self._stop_inflight:
            return
        ids = research.stop_order_ids(rows)
        if ids is None:
            return
        current = (self._stop or {}).get("id")
        if not current:
            return
        if str(current) in ids:
            self._stop_missing = 0
            return
        self._stop_missing += 1
        if self._stop_missing < 2:
            return
        self.logger().error(f"AIOT shared stop {current} is no longer open. Another bot's plan was not adopted.")
        self._stop = {}
        if self._own_base() > 0:
            self._halt(
                f"AIOT shared stop {current} is gone and this bot's share was not changed. " "Entries and exits are halted."
            )

    async def _open_stop_ids(self, connector):
        list_open = getattr(connector, "list_open_stop_orders", None)
        if not callable(list_open):
            return None
        try:
            rows = await list_open(self.config.trading_pair)
        except Exception as exc:
            self.logger().warning(f"AIOT shared could not list stop orders: {exc}")
            return None
        ids = research.stop_order_ids(rows)
        if ids is None:
            return None
        return set(ids)

    async def _run_stop(self, connector, plan, base, stop_price) -> None:
        state = dict(self._stop or {})
        stop_id = state.get("id")
        cancelled = False
        try:
            if plan == "cancel":
                if stop_id:
                    await research._cancel_stop(connector, stop_id)
                self._stop = {}
                return
            position_id = research.long_position_id(connector, self.config.trading_pair)
            if position_id is None:
                self._warn_once("stop", "AIOT shared has no long position id, so its stop was not placed.")
                return
            before = await self._open_stop_ids(connector)
            if stop_id:
                await research._cancel_stop(connector, stop_id)
                cancelled = True
                self._stop = {}
                if before is not None:
                    before.discard(str(stop_id))
            new_id = await connector.place_position_stop_order(
                self.config.trading_pair,
                position_id,
                research._decimal(base),
                stop_loss_price=research._decimal(stop_price),
                vol_type=1,
                stop_loss_reverse=2,
            )
            if not new_id or str(new_id) in ("None", ""):
                self.logger().error(_STOP_REJECTED)
                return
            if before is not None and str(new_id) in before:
                self._stop_blocked = (
                    "MEXC merged this bot's stop into an existing plan at the same price "
                    f"({new_id}). This bot will not adopt or cancel that plan. Entries are halted."
                )
                self.logger().error(self._stop_blocked)
                return
            self._stop_blocked = None
            self._stop = {
                "id": str(new_id),
                "amount": str(research._decimal(base)),
                "price": str(research._decimal(stop_price)),
            }
        except Exception as exc:
            self.logger().error(f"{_STOP_REJECTED} {exc}")
            if state.get("id") and not cancelled:
                self._stop = state
        finally:
            self._stop_inflight = False

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._halt(f"AIOT shared state file is unreadable ({exc}). Entries and exits are halted.")
            return
        if payload.get("strategy") != self._spec.key:
            self._halt("AIOT shared state file is for a different strategy. Entries and exits are halted.")
            return
        saved = {item.get("name"): item for item in payload.get("sleeves") or []}
        for sleeve in self._sleeves:
            state = saved.get(sleeve.spec.name)
            if state:
                sleeve.import_state(state)
        self._applied_base = {key: float(value) for key, value in (payload.get("applied_base") or {}).items()}
        self._last_bar = payload.get("last_bar")
        stop = payload.get("stop")
        self._stop = stop if isinstance(stop, dict) else {}
        orders = payload.get("orders")
        self._own_orders = orders if isinstance(orders, dict) else {}

    def _save_state(self) -> None:
        payload = {
            "version": 1,
            "strategy": self._spec.key,
            "last_bar": self._last_bar,
            "applied_base": self._applied_base,
            "stop": self._stop,
            "orders": self._own_orders,
            "sleeves": [sleeve.export_state() for sleeve in self._sleeves],
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self.logger().warning(f"AIOT shared state file not written: {exc}")
