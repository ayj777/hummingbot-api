"""Shared-account controller for AIOT on one MEXC hedge long.

Each bot keeps its own base from its own fills, keyed by its own client
order ids. It does not adopt the exchange long and it does not touch another
bot's orders or stop plans. An exit is this bot's own base, and only when the
exchange long still covers that base.

MEXC allows several partial stop plans (volType 1) on one long. The sum of
those volumes must be less than the closable quantity, and a new plan at the
same price updates the existing plan instead of creating a second one. This
controller places volType 1 one tick away from a price another plan already
uses, tracks only its own id, and will not adopt or cancel that other id.
If a place still returns the other id, that plan's volume is put back.
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


def _stop_fill_executed(row) -> bool:
    """Only MEXC state 3 is an executed fill. 4 and 5 are not."""
    state = row.get("state", row.get("status"))
    if state in (3, "3"):
        return True
    if state not in (None, ""):
        return False
    for key in ("base", "dealVol", "realityVol"):
        if row.get(key) not in (None, ""):
            return True
    return False


def parse_stop_fill(row, contract_size=Decimal("1")):
    """A fill with no client order id, as (plan id, base, price).

    The plan id is empty when the row has size but nothing to attribute it to.
    None means the row is not one of these fills.
    """
    if not isinstance(row, dict) or not _stop_fill_executed(row):
        return None
    client_id = row.get("client_order_id") or row.get("externalOid") or row.get("external_oid")
    if client_id not in (None, ""):
        return None
    plan = None
    for key in ("stopPlanOrderId", "stop_plan_order_id", "planId", "plan_id"):
        if row.get(key) not in (None, ""):
            plan = str(row[key])
            break
    if plan is None and (row.get("state") not in (None, "") or row.get("stopLossPrice") not in (None, "")):
        plan = _stop_row_id(row)
    try:
        step = Decimal(str(row.get("_contract_size", contract_size)))
    except (ArithmeticError, TypeError, ValueError):
        return None
    base = None
    if row.get("base") not in (None, ""):
        base = _decimal_or_none(row.get("base"))
    elif row.get("dealVol") not in (None, ""):
        deal = _decimal_or_none(row.get("dealVol"))
        base = None if deal is None else deal * step
    elif row.get("realityVol") not in (None, ""):
        deal = _decimal_or_none(row.get("realityVol"))
        base = None if deal is None else deal * step
    if base is None or base <= 0:
        return None
    return plan or "", base, _stop_row_price(row)


def _fill_indexes(rows, contract_size=Decimal("1")):
    """Client fills, stop-plan fills, and whether a fill has neither id."""
    by_client = {}
    by_plan = {}
    bare = False
    for row in rows or []:
        parsed = parse_fill_row(row, contract_size)
        if parsed is not None:
            by_client[parsed[0]] = parsed[1]
            continue
        stop = parse_stop_fill(row, contract_size)
        if stop is None:
            continue
        plan, base, _price = stop
        if not plan:
            bare = True
            continue
        by_plan[plan] = base
    return by_client, by_plan, bare


def reconcile_share(orders, rows, contract_size=Decimal("1")):
    """Match this bot's stored fills to history.

    Returns (status, net base, reason). status is `ok` or `halt`. History rows
    for other client ids are ignored. A missing id or a different size halts.
    A stop fill with no client id is matched by its plan id. A fill with neither
    id halts.
    """
    found, plans, bare = _fill_indexes(rows, contract_size)
    if bare:
        return (
            "halt",
            None,
            "AIOT shared cannot attribute a fill with no client order id. " "Entries and exits are halted.",
        )
    net = Decimal("0")
    for client_id, slot in orders.items():
        plan = slot.get("plan_id")
        if plan:
            if plan not in plans:
                if shares_match(slot.get("base") or 0, 0):
                    continue
                # A client-order read has no plan fills. Keep a sell this bot already booked.
                if not plans:
                    seen = _decimal_or_none(slot.get("base")) or Decimal("0")
                    net -= seen
                    continue
                return (
                    "halt",
                    None,
                    f"AIOT shared has no fill for its stop plan {plan}. " "Entries and exits are halted.",
                )
            seen = plans[plan]
        elif client_id not in found:
            # An order that has not filled yet may be absent. A recorded fill may not.
            if shares_match(slot.get("base") or 0, 0):
                continue
            return (
                "halt",
                None,
                f"AIOT shared has no fill for its order {client_id}. " "Entries and exits are halted.",
            )
        else:
            seen = found[client_id]
        stored = slot.get("base")
        if stored in (None, ""):
            return (
                "halt",
                None,
                f"AIOT shared order {client_id} has no stored size. " "Entries and exits are halted.",
            )
        # A zero means the id was saved before the fill. History is that fill.
        if shares_match(stored, 0) and seen > 0:
            slot["base"] = _amount_text(seen)
        elif not shares_match(stored, seen):
            return (
                "halt",
                None,
                f"AIOT shared order {client_id} was {stored} and the fill " f"history says {seen}. Entries and exits are halted.",
            )
        if slot.get("side") == "sell" or plan:
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


def _decimal_or_none(value):
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return None


def fill_rows_of(executor):
    """Fills as (client id, base, price). None when a positive fill cannot be split.

    The executor total includes earlier partials. Those stay on their own client
    ids. The latest id gets only the remainder.
    """
    info = getattr(executor, "custom_info", None) or {}
    if not isinstance(info, dict):
        return None
    rows = []
    seen = set()
    held_sum = Decimal("0")
    held_quote = Decimal("0")
    held_quote_known = True
    for row in info.get("held_position_orders") or []:
        if not isinstance(row, dict):
            continue
        cid = row.get("client_order_id")
        base = _decimal_or_none(row.get("executed_amount_base"))
        if cid in (None, "") or base is None or str(cid) in seen:
            continue
        cid = str(cid)
        seen.add(cid)
        quote = _decimal_or_none(row.get("executed_amount_quote"))
        price = quote / base if quote is not None and base > 0 else None
        if price is None and base > 0:
            held_quote_known = False
        elif quote is not None:
            held_quote += quote
        rows.append((cid, base, price))
        held_sum += base
    total = _decimal_or_none(info.get("executed_amount_base"))
    current = info.get("order_id")
    current = None if current in (None, "") else str(current)
    if current and current not in seen:
        if total is None:
            rows.append((current, Decimal("0"), None))
        else:
            part = total - held_sum
            if part < -_SHARE_TOLERANCE:
                return None
            if part < 0:
                part = Decimal("0")
            price = None
            average = _decimal_or_none(info.get("average_executed_price"))
            if part > 0 and average is not None and average > 0 and held_quote_known:
                price = (total * average - held_quote) / part
            elif part > 0 and not rows:
                price = average
            rows.append((current, part, price))
    elif total is not None and total > 0 and not rows:
        return None
    return rows


def _stop_row_id(row) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for key in ("id", "orderId", "stopPlanOrderId"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _stop_row_price(row):
    if not isinstance(row, dict):
        return None
    for key in ("stopLossPrice", "stop_loss_price", "triggerPrice"):
        price = _decimal_or_none(row.get(key))
        if price is not None and price > 0:
            return price
    return None


def _stop_row_base(row, contract_size=Decimal("1")):
    """Plan size in base. `vol` on a MEXC stop row is contracts."""
    if not isinstance(row, dict):
        return None
    if row.get("base") not in (None, ""):
        return _decimal_or_none(row.get("base"))
    step = _decimal_or_none(contract_size) or Decimal("1")
    for key in ("vol", "volume", "stopLossVol"):
        vol = _decimal_or_none(row.get(key))
        if vol is not None:
            return vol * step
    return None


def _stop_rows(rows):
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("orders") or rows.get("result")
    if not isinstance(rows, list):
        return None
    return rows


def _is_plan_slot(client_id, slot) -> bool:
    if slot.get("plan_id"):
        return True
    return str(client_id).startswith("stop:")


def _booked_overlap(orders, rows) -> Decimal:
    """How much of these executor rows is already stored on their client ids."""
    total = Decimal("0")
    for client_id, base, _price in rows or []:
        if base <= 0:
            continue
        slot = orders.get(client_id)
        if slot is None:
            continue
        have = _decimal_or_none(slot.get("base")) or Decimal("0")
        if have > 0:
            total += min(have, base)
    return total


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
    total_amount_quote: Decimal = Field(default=Decimal("2200"))
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
        self._entry_halt = None
        self._share_checked = False
        self._checked_long = None
        self._checked_at = 0.0
        self._history_inflight = False
        self._stop_cancel_requested = False
        self._loading_state = False
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
        if self._share_checked and not self._needs_share_recheck():
            return True
        self._share_checked = False
        if not self._own_orders and not (self._stop or {}).get("id"):
            connector = self._connector()
            history = getattr(connector, "fill_history", None) if connector is not None else None
            if not callable(history):
                self._mark_share_checked()
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

    def _needs_share_recheck(self) -> bool:
        if not self._share_checked:
            return True
        if self._live_long != getattr(self, "_checked_long", None):
            return True
        checked_at = getattr(self, "_checked_at", None)
        if checked_at is None:
            return False
        return self._now() - float(checked_at) >= 30

    def _mark_share_checked(self) -> None:
        self._share_checked = True
        self._checked_long = self._live_long
        self._checked_at = self._now()

    def _apply_history(self, rows) -> bool:
        unknown = self._unknown_client_id(rows)
        if unknown:
            self._halt(f"AIOT shared fill {unknown} was not saved for this bot. " "Entries and exits are halted.")
            return False
        if not self._absorb_stop_fills(rows):
            return False
        verdict, net, reason = reconcile_share(self._own_orders, rows)
        if verdict != "ok":
            if self._history_is_lagging(reason):
                return False
            self._halt(reason)
            return False
        stop_id = (self._stop or {}).get("id")
        plan_slot = self._own_orders.get(f"stop:{stop_id}") if stop_id else None
        plan_filled = bool(plan_slot) and not shares_match(plan_slot.get("base") or 0, 0)
        if len(self._sleeves) == 1:
            sleeve = self._sleeves[0]
            if net <= _SHARE_TOLERANCE:
                if sleeve.phase != "flat" or sleeve.base > 0:
                    sleeve.exit_reason = "stop"
                    sleeve.apply_sell_fill(float(sleeve.stop_price or 0.0), int(self._now()), -1)
            else:
                sleeve.base = float(net)
                if sleeve.phase == "flat":
                    sleeve.phase = "open"
        if plan_filled:
            self._stop = {}
        elif not shares_match(self._own_base(), net):
            self._halt(
                f"AIOT shared share {self._own_base()} does not match its own fills {net}. " "Entries and exits are halted."
            )
            return False
        self._mismatch_seen = False
        self._mark_share_checked()
        self._save_state()
        return True

    def _history_is_lagging(self, reason: str) -> bool:
        """A recheck may see a dealVol that has not caught the ledger yet.

        The first read after a successful check waits. The next one still
        behind is a real mismatch and halts.
        """
        if "fill history says" not in (reason or ""):
            return False
        if not getattr(self, "_checked_at", None):
            return False
        if getattr(self, "_mismatch_seen", False):
            return False
        self._mismatch_seen = True
        return True

    def _unknown_client_id(self, rows):
        known = set(self._own_orders) | self._sibling_order_ids()
        for row in rows or []:
            parsed = parse_fill_row(row)
            if parsed is None:
                continue
            client_id, base = parsed
            if base > 0 and client_id not in known:
                return client_id
        return None

    def _sibling_order_ids(self):
        found = set()
        folder = self._state_path.parent
        if not folder.is_dir():
            return found
        for path in folder.glob("aiot_shared_*.json"):
            try:
                if path.resolve() == self._state_path.resolve():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            orders = payload.get("orders") if isinstance(payload, dict) else None
            if isinstance(orders, dict):
                found.update(str(key) for key in orders)
        return found

    def _our_plan_ids(self):
        found = set()
        stop_id = (self._stop or {}).get("id")
        if stop_id:
            found.add(str(stop_id))
        for slot in self._own_orders.values():
            if slot.get("plan_id"):
                found.add(str(slot["plan_id"]))
        return found

    def _absorb_stop_fills(self, rows) -> bool:
        """Book a stop-plan fill on this bot. Another plan's fill is left alone."""
        _clients, plans, bare = _fill_indexes(rows)
        if bare:
            self._halt("AIOT shared cannot attribute a fill with no client order id. " "Entries and exits are halted.")
            return False
        ours = self._our_plan_ids()
        for row in rows or []:
            parsed = parse_stop_fill(row)
            if parsed is None:
                continue
            plan, base, price = parsed
            if not plan or plan not in ours or plan not in plans:
                continue
            current = self._own_orders.get(f"stop:{plan}") or {}
            used = price if price is not None else _decimal_or_none(current.get("price"))
            self._own_orders[f"stop:{plan}"] = {
                "side": "sell",
                "level": current.get("level") or "shared:stop:sell",
                "plan_id": plan,
                "base": _amount_text(base),
                "price": _amount_text(used or 0),
            }
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
            for client_id, slot in list(self._own_orders.items()):
                if _is_plan_slot(client_id, slot):
                    continue
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
        side = "buy" if role == "buy" else "sell"
        rows = fill_rows_of(executor)
        if role == "buy" and not done:
            self._note_open_orders(rows, level, side)
            if sleeve.base <= 0.0 and research._is_active(executor):
                sleeve.phase = "pending_entry"
                if rung is not None and 0 <= rung < len(sleeve.rungs):
                    sleeve.rungs[rung] = True
                if rung == -1:
                    sleeve.added = True
            return True
        if filled > recorded + 1e-12 and price > 0.0:
            if not rows:
                self._halt("AIOT shared filled an order it cannot tie to one client order id. " "Entries and exits are halted.")
                return False
            # History may already have put this fill on the sleeve. Add only the new part.
            already = _booked_overlap(self._own_orders, rows)
            delta = filled - max(float(recorded), float(already))
            if delta <= float(_SHARE_TOLERANCE):
                self._applied_base[executor.id] = filled
                return True
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
            self._store_fill_rows(rows, level, side, price)
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

    def _note_open_orders(self, rows, level: str, side: str) -> None:
        for client_id, _base, _price in rows or []:
            self._own_orders.setdefault(
                client_id,
                {"side": side, "level": level, "base": "0.00000000", "price": "0"},
            )

    def _store_fill_rows(self, rows, level: str, side: str, fallback_price: float) -> None:
        for client_id, base, price in rows:
            if base <= 0:
                self._note_open_orders([(client_id, base, price)], level, side)
                continue
            used = price if price is not None and price > 0 else Decimal(str(fallback_price))
            self._own_orders[client_id] = {
                "side": side,
                "level": level,
                "base": _amount_text(base),
                "price": _amount_text(used),
            }

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
        if intent.kind == "market_sell" and self._own_exceeds_long():
            self._halt(self._residual_reason())
            return [], False
        if intent.kind == "market_sell" and intent.reason == "stop" and self._has_exchange_stop():
            # The exchange plan is the close. A second market sell would take the other share.
            sleeve = self._sleeve(intent.sleeve)
            if sleeve is not None:
                sleeve.abort_sell()
            return [], False
        if intent.kind == "market_sell" and self._has_exchange_stop():
            self._request_stop_cancel()
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
        if self._share_halt or self._stop_blocked or getattr(self, "_entry_halt", None):
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
            self._halt(self._residual_reason())
            return Decimal("0")
        return super()._clip_sell(sleeve_base)

    def _tracks_positions(self) -> bool:
        return callable(getattr(self._connector(), "_update_positions", None))

    def _has_exchange_stop(self) -> bool:
        if not (self._stop or {}).get("id"):
            return False
        return callable(getattr(self._connector(), "place_position_stop_order", None))

    def _own_exceeds_long(self) -> bool:
        live = self._live_long
        return live is not None and self._own_base() > live

    def _residual_reason(self) -> str:
        return (
            f"AIOT shared owns {self._own_base()} and the exchange long is {self._live_long}. "
            "The residual was not closed, because it may be another bot's share. "
            "Entries and exits are halted."
        )

    def _request_stop_cancel(self) -> None:
        if self._stop_inflight or not (self._stop or {}).get("id"):
            return
        connector = self._connector()
        if connector is None:
            return
        self._stop_inflight = True
        self._stop_cancel_requested = True
        safe_ensure_future(self._run_stop(connector, "cancel", 0, None))

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
        if getattr(self, "_loading_state", False):
            return
        # The other bot's book is in its own instance directory. A missing,
        # stale, or unreadable file is not an empty book, so a halt never
        # places a stop on the long that is left. Cancel this plan when it
        # is larger than that long.
        if self._own_exceeds_long():
            self._request_stop_cancel()
        self._save_state()

    def _read_finished_plan(self) -> None:
        """Read this plan's executed size before any halt can cover the residual."""
        if getattr(self, "_attribute_inflight", False) or not (self._stop or {}).get("id"):
            return
        connector = self._connector()
        if connector is None:
            return
        self._attribute_inflight = True
        safe_ensure_future(self._attribute_finished_plan(connector))

    async def _attribute_finished_plan(self, connector) -> None:
        try:
            current = (self._stop or {}).get("id")
            if not current:
                return
            plans = await self._stop_plan_rows(connector)
            if self._book_exchange_stop_fill(str(current), plans):
                return
            if self._own_base() > 0 and self._own_exceeds_long():
                self._halt(self._residual_reason())
        finally:
            self._attribute_inflight = False

    def _apply_live_position(self) -> None:
        connector = self._connector()
        self._schedule_position_fetch(connector)
        mapped = None
        if connector is not None:
            mapped = research.live_long_base(connector, self.config.trading_pair)
        if mapped is not None and mapped > 0:
            self._positions_seen = True
            self._flat_streak = 0
            changed = self._live_long != mapped
            self._live_long = mapped
            if changed:
                self.confirm_share()
            own = self._own_base()
            if own > mapped and not self._history_inflight:
                if (self._stop or {}).get("id"):
                    self._read_finished_plan()
                else:
                    self._halt(self._residual_reason())
            elif own > 0 and not self._share_halt:
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
        # Our plan only. Never the whole long, and never the other bot's residual.
        own = self._own_base()
        if self._own_exceeds_long():
            return Decimal("0")
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
        if self._share_halt or self._stop_blocked:
            return
        if self._own_orders and not self._share_checked:
            return
        connector = self._connector()
        if connector is None or not callable(getattr(connector, "place_position_stop_order", None)):
            return
        self._schedule_stop_list(connector)
        if self._stop_inflight:
            return
        if self._own_exceeds_long():
            if (self._stop or {}).get("id"):
                self._read_finished_plan()
            else:
                self._halt(self._residual_reason())
            return
        live = self._live_long
        price = self._hard_stop_price()
        own = self._stop_amount()
        known = live is not None or not self._tracks_positions()
        state = self._stop or {}
        plan = research.protective_stop_plan(
            state.get("id"),
            state.get("amount"),
            state.get("price"),
            own,
            price,
            live_known=known,
        )
        if plan == "place" and self._nudged_stop_holds(state, own, price):
            return
        if plan == "hold":
            return
        if plan == "place" and (own is None or own <= 0 or not price):
            return
        self._stop_inflight = True
        safe_ensure_future(self._run_stop(connector, plan, own, price))

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
            listed = None
            for row in _stop_rows(rows) or []:
                if _stop_row_id(row) == str(current):
                    listed = row
                    break
            tracked = (self._stop or {}).get("amount")
            volume = _stop_row_base(listed, self._contract_size(connector)) if listed is not None else None
            if volume is not None and tracked not in (None, "") and not shares_match(volume, tracked):
                self.logger().error(f"AIOT shared stop {current} volume is {volume}, not {tracked}. The plan will be replaced.")
                self._stop["amount"] = None
            self._stop_missing = 0
            return
        if getattr(self, "_stop_cancel_requested", False):
            return
        plans = await self._stop_plan_rows(connector)
        if self._book_exchange_stop_fill(str(current), plans):
            return
        history = self._history_rows(connector)
        if history is not None and self._book_exchange_stop_fill(str(current), history):
            return
        self._stop_missing += 1
        if self._stop_missing < 2:
            return
        self.logger().error(f"AIOT shared stop {current} is no longer open. Another bot's plan was not adopted.")
        if self._own_base() > 0:
            self._stop = {}
            self._halt(
                f"AIOT shared stop {current} is gone and its fill could not be attributed. " "Entries and exits are halted."
            )

    async def _open_stop_rows(self, connector):
        list_open = getattr(connector, "list_open_stop_orders", None)
        if not callable(list_open):
            return None
        try:
            rows = await list_open(self.config.trading_pair)
        except Exception as exc:
            self.logger().warning(f"AIOT shared could not list stop orders: {exc}")
            return None
        return _stop_rows(rows)

    def _contract_size(self, connector) -> Decimal:
        getter = getattr(connector, "get_contract_size", None)
        if not callable(getter):
            return Decimal("1")
        try:
            size = Decimal(str(getter(self.config.trading_pair)))
        except (ArithmeticError, TypeError, ValueError):
            return Decimal("1")
        return size if size > 0 else Decimal("1")

    def _price_tick(self):
        connector = self._connector()
        rules = getattr(connector, "trading_rules", None) if connector is not None else None
        rule = rules.get(self.config.trading_pair) if isinstance(rules, dict) else None
        tick = _decimal_or_none(getattr(rule, "min_price_increment", None)) if rule is not None else None
        if tick is None or tick <= 0:
            return None
        return tick

    def _free_stop_price(self, rows, our_id, stop_price):
        """Our stop price, moved up by one tick when another plan already uses it."""
        try:
            wanted = research._decimal(float(stop_price))
        except (ArithmeticError, TypeError, ValueError):
            return None
        tick = self._price_tick()
        if tick is not None:
            wanted = (wanted // tick) * tick
        taken = set()
        for row in rows or []:
            rid = _stop_row_id(row)
            if rid is None or rid == str(our_id or ""):
                continue
            seen = _stop_row_price(row)
            if seen is not None:
                taken.add(research._decimal(float(seen)))
        if wanted not in taken:
            return wanted
        if tick is None:
            return None
        for step in range(1, 9):
            candidate = wanted + tick * step
            if candidate not in taken:
                return candidate
        return None

    def _nudged_stop_holds(self, state, own, price) -> bool:
        """A stop one to eight ticks tighter than the signal is still the right plan."""
        if not state.get("id") or price in (None, 0) or own is None:
            return False
        try:
            wanted_amount = research._decimal(float(own))
            wanted = research._decimal(float(price))
        except (ArithmeticError, TypeError, ValueError):
            return False
        if not research._same_text(state.get("amount"), wanted_amount):
            return False
        tracked = _decimal_or_none(state.get("price"))
        if tracked is None or tracked == wanted:
            return tracked is not None
        tick = self._price_tick()
        if tick is None:
            return False
        floored = (wanted // tick) * tick
        if tracked == floored:
            return True
        return floored < tracked <= floored + tick * 8

    def _history_rows(self, connector):
        history = getattr(connector, "fill_history", None)
        if not callable(history):
            return None
        try:
            rows = history()
        except Exception as exc:
            self.logger().warning(f"AIOT shared order history failed ({exc})")
            return None
        if hasattr(rows, "__await__"):
            rows.close()
            return None
        return list(rows or [])

    async def _stop_plan_rows(self, connector):
        """Finished stop plans from GET /api/v1/private/stoporder/list/orders.

        `id` is the plan, `state` 3 is executed, `realityVol` is the contracts
        that filled. `triggerSide` 2 is a stop-loss trigger.
        """
        getter = getattr(connector, "list_stop_orders", None)
        try:
            if callable(getter):
                rows = getter(self.config.trading_pair)
                if hasattr(rows, "__await__"):
                    rows = await rows
                return _stop_rows(rows) or []
            api = getattr(connector, "_api_get", None)
            symbol = getattr(connector, "exchange_symbol_associated_to_pair", None)
            if not callable(api) or not callable(symbol):
                return []
            pair = symbol(self.config.trading_pair)
            if hasattr(pair, "__await__"):
                pair = await pair
            return await self._page_stop_plans(api, pair)
        except Exception as exc:
            self.logger().warning(f"AIOT shared could not list stop plans: {exc}")
            return []

    def _stop_windows(self):
        """Finished-list windows. A start and end cannot span more than 90 days."""
        start = (self._stop or {}).get("created")
        if start in (None, ""):
            return [(None, None)]
        span = 89 * 24 * 60 * 60 * 1000
        now_ms = int(self._now() * 1000)
        start_ms = int(start)
        if now_ms < start_ms:
            now_ms = start_ms
        recent = start_ms if now_ms - start_ms <= span else now_ms - span
        windows = [(recent, now_ms)]
        if start_ms < recent:
            windows.append((start_ms, min(now_ms, start_ms + span)))
        return windows

    async def _page_stop_plans(self, api, pair):
        """Page the finished list from this plan's create time until its id shows up."""
        want = str((self._stop or {}).get("id") or "")
        found = []
        pages = 0
        for start_ms, end_ms in self._stop_windows():
            page = 1
            while page <= 20 and pages < 20:
                params = {"symbol": pair, "is_finished": 1, "page_num": page, "page_size": 100}
                if start_ms is not None:
                    params["start_time"] = int(start_ms)
                    params["end_time"] = int(end_ms)
                response = await api(
                    path_url="/api/v1/private/stoporder/list/orders",
                    params=params,
                    is_auth_required=True,
                )
                pages += 1
                data = response.get("data") if isinstance(response, dict) else response
                if isinstance(data, dict):
                    data = data.get("resultList") or data.get("orders") or data.get("list")
                if not isinstance(data, list) or not data:
                    break
                found.extend(data)
                if want and any(isinstance(row, dict) and _stop_row_id(row) == want for row in data):
                    return found
                if len(data) < 100:
                    break
                page += 1
        return found

    def _book_exchange_stop_fill(self, plan_id, rows) -> bool:
        """Book this plan's executed contracts. Other plans in the list are ignored."""
        size = self._contract_size(self._connector())
        parsed = None
        for row in rows or []:
            found = parse_stop_fill(row, size)
            if found is not None and found[0] == str(plan_id) and found[1] > 0:
                parsed = found
                break
        if parsed is None:
            return False
        plan, base, price = parsed
        key = f"stop:{plan}"
        current = self._own_orders.get(key) or {}
        if shares_match(current.get("base") or 0, base):
            self._stop = {}
            self._stop_missing = 0
            return True
        used = price if price is not None else _decimal_or_none((self._stop or {}).get("price"))
        self._own_orders[key] = {
            "side": "sell",
            "level": current.get("level") or "shared:stop:sell",
            "plan_id": plan,
            "base": _amount_text(base),
            "price": _amount_text(used or 0),
        }
        if len(self._sleeves) == 1:
            sleeve = self._sleeves[0]
            fill_price = float(used or sleeve.stop_price or 0.0)
            if sleeve.base > 0 and Decimal(str(sleeve.base)) <= base + _SHARE_TOLERANCE:
                sleeve.exit_reason = "stop"
                sleeve.apply_sell_fill(fill_price, int(self._now()), -1)
            elif sleeve.base > 0:
                sleeve.apply_partial_sell(fill_price, float(base), int(self._now()), -1)
        self._stop = {}
        self._stop_missing = 0
        self._stop_cancel_requested = False
        if self._own_base() <= _SHARE_TOLERANCE and self._share_halt:
            if "exchange long" in self._share_halt or "could not be attributed" in self._share_halt:
                self._share_halt = None
        self._save_state()
        return True

    def _drop_unfilled_stop(self, stop_id) -> None:
        if not stop_id:
            return
        key = f"stop:{stop_id}"
        slot = self._own_orders.get(key)
        if slot and shares_match(slot.get("base") or 0, 0):
            self._own_orders.pop(key, None)

    def _remember_stop(self, new_id, base, price) -> None:
        placed = research._decimal(price)
        self._stop_blocked = None
        self._entry_halt = None
        self._stop_cancel_requested = False
        self._stop = {
            "id": str(new_id),
            "amount": str(research._decimal(base)),
            "price": str(placed),
            "created": int(self._now() * 1000),
        }
        key = f"stop:{new_id}"
        current = self._own_orders.get(key)
        if current and not shares_match(current.get("base") or 0, 0):
            return
        self._own_orders[key] = {
            "side": "sell",
            "level": "shared:stop:sell",
            "plan_id": str(new_id),
            "base": "0.00000000",
            "price": str(placed),
        }
        self._save_state()

    async def _restore_other_plan(self, connector, position_id, row) -> None:
        price = _stop_row_price(row)
        base = _stop_row_base(row, self._contract_size(connector))
        if price is None or base is None or base <= 0:
            return
        try:
            await connector.place_position_stop_order(
                self.config.trading_pair,
                position_id,
                research._decimal(base),
                stop_loss_price=research._decimal(price),
                vol_type=1,
                stop_loss_reverse=2,
            )
        except Exception as exc:
            self.logger().error(f"AIOT shared could not restore the other stop plan's volume: {exc}")

    async def _run_stop(self, connector, plan, base, stop_price) -> None:
        state = dict(self._stop or {})
        stop_id = state.get("id")
        cancelled = False
        try:
            if plan == "cancel":
                if stop_id:
                    await research._cancel_stop(connector, stop_id)
                self._drop_unfilled_stop(stop_id)
                self._stop = {}
                self._stop_cancel_requested = False
                self._save_state()
                return
            position_id = research.long_position_id(connector, self.config.trading_pair)
            if position_id is None:
                self._warn_once("stop", "AIOT shared has no long position id, so its stop was not placed.")
                return
            rows = await self._open_stop_rows(connector)
            if rows is None:
                self.logger().warning("AIOT shared did not place a stop because open stop plans could not be listed.")
                return
            price = self._free_stop_price(rows, stop_id, stop_price)
            if price is None:
                self._stop_blocked = (
                    "MEXC already has a stop plan at this price. " "This bot will not update that plan. Entries are halted."
                )
                self.logger().error(self._stop_blocked)
                self._save_state()
                return
            before = {}
            for row in rows:
                rid = _stop_row_id(row)
                if rid:
                    before[rid] = row
            if stop_id:
                await research._cancel_stop(connector, stop_id)
                cancelled = True
                self._stop_cancel_requested = True
                self._drop_unfilled_stop(stop_id)
                self._stop = {}
                before.pop(str(stop_id), None)
            new_id = await connector.place_position_stop_order(
                self.config.trading_pair,
                position_id,
                research._decimal(base),
                stop_loss_price=research._decimal(price),
                vol_type=1,
                stop_loss_reverse=2,
            )
            if not new_id or str(new_id) in ("None", ""):
                self._entry_halt = _STOP_REJECTED + " Entries stay off until this bot has its own stop."
                self.logger().error(self._entry_halt)
                self._save_state()
                return
            if str(new_id) in before:
                await self._restore_other_plan(connector, position_id, before[str(new_id)])
                self._stop_blocked = (
                    "MEXC merged this bot's stop into an existing plan at the same price "
                    f"({new_id}). This bot will not adopt or cancel that plan. Entries are halted."
                )
                self.logger().error(self._stop_blocked)
                self._save_state()
                return
            self._remember_stop(new_id, base, price)
        except Exception as exc:
            if not cancelled:
                self._stop_cancel_requested = False
            kept = bool(state.get("id")) and not cancelled
            if kept:
                self._stop = state
            else:
                self._entry_halt = f"{_STOP_REJECTED} Entries stay off until this bot has its own stop. {exc}"
                self._save_state()
            self.logger().error(f"{_STOP_REJECTED} {exc}")
        finally:
            self._stop_inflight = False

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        self._loading_state = True
        try:
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
            if payload.get("halt"):
                self._share_halt = str(payload["halt"])
            self._entry_halt = payload.get("entry_halt") or None
            if payload.get("stop_blocked"):
                self._stop_blocked = str(payload["stop_blocked"])
        finally:
            self._loading_state = False

    def _save_state(self) -> None:
        payload = {
            "version": 1,
            "strategy": self._spec.key,
            "last_bar": self._last_bar,
            "applied_base": self._applied_base,
            "stop": self._stop,
            "orders": self._own_orders,
            "halt": self._share_halt,
            "entry_halt": getattr(self, "_entry_halt", None),
            "stop_blocked": self._stop_blocked,
            "sleeves": [sleeve.export_state() for sleeve in self._sleeves],
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self.logger().warning(f"AIOT shared state file not written: {exc}")
