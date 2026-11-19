"""Paper-trade controller for the AIOT research strategies.

One module, seven configs. `strategy` selects the sleeve rules in `aiot_engine`.
Orders go to `mexc_paper_trade`. Candles and the higher-timeframe filter come
from the public `mexc` spot feed, and only from candles that have closed.

The engine decides. This class turns those decisions into OrderExecutor
actions: a market order for entries and for exits that the engine queues off a
closed bar, a resting limit for a partial take-profit or the S2326 take-profit,
and a market sell when the live mid trades through the hard stop. The stock
triple barrier is not used. Its stop is net of fees, its trail follows the live
price rather than the highest closed bar, and it cannot express a partial, a
ladder, a stop measured from the first fill, or the re-arm rule.

A buy uses PositionAction.OPEN. A sell uses PositionAction.CLOSE so hedge mode
closes the long. size_mode max caps each entry at the configured ceiling and at
the available balance times leverage times 0.95, then splits that base at the
exchange max order size. After a fill, a reduce-only stop rests at the hard
stop. The trail stays in the engine. The exchange stop is replaced when a
partial changes the size and cancelled when the position is flat.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from hummingbot.core.data_type.common import (
    PositionAction,
    PositionMode,
    PositionSide,
    PriceType,
    TradeType,
)
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy, OrderExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType
from pydantic import Field, model_validator

from .aiot_engine import INTERVAL_SECONDS, STRATEGIES, Bar, Intent, Sleeve, build_contexts


class AiotResearchConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "aiot_research"
    connector_name: str = Field(default="mexc_paper_trade")
    trading_pair: str = Field(default="AIOT-USDT")
    total_amount_quote: Decimal = Field(default=Decimal("15000"))
    leverage: int = Field(default=1)
    position_mode: PositionMode = Field(default=PositionMode.HEDGE)
    # max: ceiling is total_amount_quote, then the balance cap. fixed: the ceiling only.
    size_mode: str = Field(default="max")
    # Empty on purpose. The engine owns exits; a stock barrier would not match.
    stop_loss: Optional[Decimal] = Field(default=None)
    take_profit: Optional[Decimal] = Field(default=None)
    time_limit: Optional[int] = Field(default=None)
    trailing_stop: Optional[str] = Field(default=None)
    candles_connector: str = Field(default="mexc")
    candles_trading_pair: str = Field(default="AIOT-USDT")
    strategy: str = Field(default="f1")

    @model_validator(mode="after")
    def strategy_has_its_notional(self):
        spec = STRATEGIES.get(self.strategy)
        if spec is None:
            raise ValueError(f"unknown strategy {self.strategy}")
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
        expected = Decimal(str(spec.total_quote))
        if self.size_mode == "fixed" and self.total_amount_quote != expected:
            raise ValueError(
                f"strategy {self.strategy} is fixed at {expected} quote, "
                f"got total_amount_quote {self.total_amount_quote}"
            )
        barriers = (self.stop_loss, self.take_profit, self.time_limit, self.trailing_stop)
        if any(value is not None for value in barriers):
            raise ValueError(
                "leave stop_loss, take_profit, time_limit and trailing_stop empty; "
                "the AIOT engine owns those exits"
            )
        return self


class AiotResearchController(DirectionalTradingControllerBase):
    def __init__(self, config: AiotResearchConfig, *args, **kwargs):
        self.config = config
        super().__init__(config, *args, **kwargs)
        self._spec = STRATEGIES[config.strategy]
        self._sleeves = [Sleeve(spec) for spec in self._spec.sleeves]
        self._applied_base = {}
        self._last_bar = None
        self._limit_live = set()
        self._stops = {}
        self._stop_inflight = set()
        self._warned = set()
        self._leverage_ready = False
        self._leverage_inflight = False
        self._leverage_next = 0.0
        safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in config.id)
        self._state_path = Path("data") / f"aiot_research_{safe_id}.json"
        self._load_state()

    def get_candles_config(self) -> List[CandlesConfig]:
        intervals = [self._spec.signal_interval]
        if self._spec.htf_interval:
            intervals.append(self._spec.htf_interval)
        return [
            CandlesConfig(
                connector=self.config.candles_connector,
                trading_pair=self.config.candles_trading_pair,
                interval=interval,
                max_records=400,
            )
            for interval in intervals
        ]

    async def update_processed_data(self):
        frames = {}
        now = self._now()
        for interval in self._intervals():
            try:
                frame = self.market_data_provider.get_candles_df(
                    connector_name=self.config.candles_connector,
                    trading_pair=self.config.candles_trading_pair,
                    interval=interval,
                    max_records=400,
                )
            except Exception as exc:
                self.logger().warning(f"AIOT candles {interval} unavailable: {exc}")
                frame = None
            frames[interval] = self._closed_bars(frame, interval, now)
        self.processed_data["frames"] = frames
        self.processed_data["now"] = now

    def determine_executor_actions(self):
        actions = []
        self._schedule_leverage()
        self._reconcile()
        mid = self._mid_price()
        if mid is not None:
            for sleeve in self._sleeves:
                sleeve.on_price(mid)
        frames = self.processed_data.get("frames") or {}
        signal_bars = frames.get(self._spec.signal_interval) or []
        if signal_bars and signal_bars[-1].open_time != self._last_bar:
            self._last_bar = signal_bars[-1].open_time
            htf_bars = frames.get(self._spec.htf_interval) if self._spec.htf_interval else None
            contexts = build_contexts(
                signal_bars,
                self._spec.signal_interval,
                htf_bars,
                self._spec.htf_interval,
                self._spec.htf_mode,
            )
            bar = signal_bars[-1]
            for sleeve in self._sleeves:
                sleeve.on_close(bar, contexts[-1], len(signal_bars) - 1, INTERVAL_SECONDS[self._spec.signal_interval])
        for sleeve in self._sleeves:
            if sleeve.cancel_limit:
                actions.extend(self._cancel_limits(sleeve.spec.name))
                sleeve.cancel_limit = False
            for intent in sleeve.drain():
                built, keep = self._intent_actions(intent)
                if keep:
                    sleeve.requeue(intent)
                    continue
                actions.extend(built)
                if intent.kind == "market_buy" and not built:
                    sleeve.abort_buy(intent.rung, self._buys_still_live(sleeve.spec.name, ""))
            if sleeve.limit_price and sleeve.limit_base > 0 and sleeve.spec.name not in self._limit_live:
                limit = self._limit_action(sleeve)
                if limit is not None:
                    actions.append(limit)
                    self._limit_live.add(sleeve.spec.name)
            if sleeve.limit_price is None:
                self._limit_live.discard(sleeve.spec.name)
        self._schedule_stops()
        self._save_state()
        return actions

    def get_custom_info(self) -> dict:
        return {
            "strategy": self._spec.key,
            "connector": self.config.connector_name,
            "candles_connector": self.config.candles_connector,
            "pair": self.config.trading_pair,
            "sleeves": [sleeve.snapshot() for sleeve in self._sleeves],
        }

    def to_format_status(self) -> List[str]:
        lines = [f"AIOT {self._spec.key} {self.config.trading_pair} on {self.config.connector_name}"]
        for sleeve in self._sleeves:
            lines.append(
                f"{sleeve.spec.name}: {sleeve.phase} base={sleeve.base:.8f} "
                f"fill={sleeve.first_fill_price} stop={sleeve.stop_price} "
                f"peak={sleeve.highest_close} blocked={sleeve.blocked}"
            )
        return lines

    def _intervals(self):
        intervals = [self._spec.signal_interval]
        if self._spec.htf_interval:
            intervals.append(self._spec.htf_interval)
        return intervals

    def _now(self) -> float:
        try:
            return float(self.market_data_provider.time())
        except Exception:
            return 0.0

    def _mid_price(self) -> Optional[float]:
        try:
            price = self.market_data_provider.get_price_by_type(
                self.config.connector_name,
                self.config.trading_pair,
                PriceType.MidPrice,
            )
        except Exception:
            return None
        if price is None:
            return None
        price = float(price)
        if price <= 0.0 or price != price:
            return None
        return price

    def _closed_bars(self, frame, interval: str, now: float) -> List[Bar]:
        if frame is None or len(frame) == 0:
            return []
        seconds = INTERVAL_SECONDS[interval]
        bars = []
        seen = set()
        for row in frame.itertuples(index=False):
            open_time = int(row.timestamp)
            if open_time + seconds > now or open_time in seen:
                continue
            seen.add(open_time)
            bars.append(Bar(open_time, float(row.open), float(row.high), float(row.low), float(row.close)))
        bars.sort(key=lambda bar: bar.open_time)
        return bars

    def _reconcile(self) -> None:
        for executor in list(self.executors_info or []):
            level = getattr(executor.config, "level_id", None) or ""
            parts = level.split(":")
            if len(parts) < 3 or parts[0] != "aiot":
                continue
            sleeve = self._sleeve(parts[1])
            if sleeve is None:
                continue
            role = parts[2]
            filled = _float_field(executor, "executed_amount_base")
            price = _float_field(executor, "average_executed_price")
            done = _is_done(executor)
            if role == "limit" and _is_active(executor):
                self._limit_live.add(sleeve.spec.name)
            recorded = self._applied_base.get(executor.id, 0.0)
            if role == "buy" and not done:
                # A restart reloads an unfilled entry as flat. Hold that sleeve
                # until this order resolves so the next close cannot buy again.
                if sleeve.base <= 0.0 and _is_active(executor):
                    sleeve.phase = "pending_entry"
                    rung = _rung_index(parts)
                    if rung is not None and 0 <= rung < len(sleeve.rungs):
                        sleeve.rungs[rung] = True
                continue
            if filled > recorded + 1e-12 and price > 0.0:
                delta = filled - recorded
                self._applied_base[executor.id] = filled
                now = int(self._now())
                if role == "buy":
                    notional = delta * price
                    rung = _rung_index(parts)
                    rung_reason = f"rung{rung + 1}" if rung is not None else "entry"
                    sleeve.apply_buy_fill(price, delta, notional, now, rung_reason, -1)
                elif role == "sell":
                    sleeve.exit_reason = sleeve.exit_reason or "exit"
                    sleeve.apply_sell_fill(price, now, -1)
                elif role == "limit":
                    sleeve.apply_limit_fill(price, delta, now, -1)
            elif done and filled <= 0.0 and executor.id not in self._applied_base:
                self._applied_base[executor.id] = 0.0
                if role == "buy":
                    sleeve.abort_buy(
                        _rung_index(parts),
                        self._buys_still_live(sleeve.spec.name, executor.id),
                    )
                elif role == "sell":
                    sleeve.abort_sell()
            elif role == "sell" and not done and _is_active(executor) and sleeve.phase == "open":
                sleeve.phase = "pending_exit"

    def _buys_still_live(self, sleeve_name: str, exclude: str) -> bool:
        prefix = f"aiot:{sleeve_name}:buy"
        for executor in self.executors_info or []:
            if executor.id == exclude or not _is_active(executor):
                continue
            level = getattr(executor.config, "level_id", None) or ""
            if level.startswith(prefix):
                return True
        return False

    def _sleeve(self, name: str) -> Optional[Sleeve]:
        for sleeve in self._sleeves:
            if sleeve.spec.name == name:
                return sleeve
        return None

    def _intent_actions(self, intent: Intent):
        if intent.kind == "market_sell":
            if intent.base <= 0.0:
                return [], False
            action = self._create(
                TradeType.SELL,
                intent.base,
                None,
                f"aiot:{intent.sleeve}:sell",
                ExecutionStrategy.MARKET,
            )
            return ([action] if action is not None else []), False
        if intent.kind != "market_buy":
            return [], False
        return self._buy_actions(intent)

    def _buy_actions(self, intent: Intent):
        if not self._entries_allowed():
            self._schedule_leverage()
            self._warn_once(
                "leverage",
                "AIOT held an entry until leverage 1 isolated is confirmed.",
            )
            return [], True
        budget = self._entry_budget()
        if budget is None:
            self._warn_once(
                "balance",
                "AIOT skipped an entry because the quote balance could not be read.",
            )
            return [], True
        limits = self._order_limits()
        if limits is None:
            self._warn_once("rules", "AIOT skipped an entry because trading rules are not loaded.")
            return [], True
        min_size, max_size, step = limits
        price = float(intent.price or 0.0)
        if price <= 0.0:
            return [], False
        base = scaled_base(intent.notional, self._spec.total_quote, budget, price)
        chunks = split_base(base, min_size, max_size, step)
        if not chunks:
            self._warn_once("min", "AIOT skipped an entry below the minimum order size.")
            return [], False
        role = "buy" if intent.rung is None else f"buy:{intent.rung}"
        actions = []
        for index, chunk in enumerate(chunks):
            level = f"aiot:{intent.sleeve}:{role}"
            if len(chunks) > 1:
                level = f"{level}:c{index}"
            action = self._create(
                TradeType.BUY, float(chunk), None, level, ExecutionStrategy.MARKET
            )
            if action is not None:
                actions.append(action)
        return actions, False

    def _limit_action(self, sleeve: Sleeve) -> Optional[CreateExecutorAction]:
        if sleeve.phase != "open" or sleeve.limit_price is None or sleeve.limit_base <= 0.0:
            return None
        return self._create(
            TradeType.SELL,
            sleeve.limit_base,
            sleeve.limit_price,
            f"aiot:{sleeve.spec.name}:limit",
            ExecutionStrategy.LIMIT,
        )

    def _create(self, side, amount, price, level_id, strategy):
        if amount <= 0.0:
            return None
        config = OrderExecutorConfig(
            controller_id=self.config.id,
            timestamp=self._now(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            side=side,
            amount=_decimal(amount),
            price=_decimal(price) if price is not None else None,
            execution_strategy=strategy,
            position_action=position_action_for(side),
            level_id=level_id,
            leverage=1,
        )
        return CreateExecutorAction(controller_id=self.config.id, executor_config=config)

    def _cancel_limits(self, sleeve_name: str):
        actions = []
        prefix = f"aiot:{sleeve_name}:limit"
        for executor in self.executors_info or []:
            level = getattr(executor.config, "level_id", None) or ""
            if level == prefix and _is_active(executor):
                actions.append(StopExecutorAction(controller_id=self.config.id, executor_id=executor.id))
        self._limit_live.discard(sleeve_name)
        return actions

    def _connector(self):
        provider = getattr(self, "market_data_provider", None)
        connectors = getattr(provider, "connectors", None) if provider is not None else None
        if not isinstance(connectors, dict):
            return None
        return connectors.get(self.config.connector_name)

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        self.logger().warning(message)

    def _entries_allowed(self) -> bool:
        connector = self._connector()
        if connector is None:
            return False
        if not callable(getattr(connector, "confirm_leverage_and_margin", None)):
            return True
        return self._leverage_ready

    def _schedule_leverage(self) -> None:
        connector = self._connector()
        if connector is None or self._leverage_ready or self._leverage_inflight:
            return
        if not callable(getattr(connector, "confirm_leverage_and_margin", None)):
            return
        now = self._now()
        if self._leverage_next and now < self._leverage_next:
            return
        self._leverage_inflight = True
        self._leverage_next = now + 30.0
        safe_ensure_future(self._confirm_leverage(connector))

    async def _confirm_leverage(self, connector) -> None:
        try:
            self._leverage_ready = await confirm_aiot_leverage(
                connector,
                self.config.trading_pair,
                int(self.config.leverage),
            )
            if not self._leverage_ready:
                self._warn_once(
                    "leverage",
                    "AIOT leverage 1 isolated was not confirmed. Entries stay off.",
                )
        except Exception as exc:
            self._leverage_ready = False
            self.logger().warning(f"AIOT leverage confirmation failed: {exc}")
        finally:
            self._leverage_inflight = False

    def _entry_budget(self):
        ceiling = Decimal(str(self.config.total_amount_quote))
        if self.config.size_mode == "fixed":
            return ceiling
        connector = self._connector()
        if connector is None:
            return None
        getter = getattr(connector, "get_available_balance", None)
        if not callable(getter):
            return None
        quote = self.config.trading_pair.split("-")[-1]
        try:
            balance = getter(quote)
        except Exception:
            return None
        return max_entry_quote(ceiling, balance, int(self.config.leverage))

    def _order_limits(self):
        connector = self._connector()
        rules = getattr(connector, "trading_rules", None) if connector is not None else None
        if not isinstance(rules, dict):
            return None
        rule = rules.get(self.config.trading_pair)
        if rule is None:
            return None
        try:
            low = Decimal(str(rule.min_order_size))
            high = Decimal(str(rule.max_order_size))
        except (ArithmeticError, TypeError, ValueError):
            return None
        return low, high, getattr(rule, "min_base_amount_increment", None)

    def _schedule_stops(self) -> None:
        connector = self._connector()
        if connector is None or not callable(getattr(connector, "place_position_stop_order", None)):
            return
        for sleeve in self._sleeves:
            name = sleeve.spec.name
            if name in self._stop_inflight:
                continue
            state = self._stops.get(name) or {}
            plan = protective_stop_plan(
                state.get("id"),
                state.get("amount"),
                state.get("price"),
                sleeve.base,
                sleeve.stop_price,
            )
            if plan == "hold":
                continue
            self._stop_inflight.add(name)
            safe_ensure_future(
                self._run_stop(connector, name, plan, sleeve.base, sleeve.stop_price)
            )

    async def _run_stop(self, connector, name, plan, base, stop_price) -> None:
        try:
            state = self._stops.get(name) or {}
            stop_id = state.get("id")
            if plan == "cancel":
                await _cancel_stop(connector, stop_id)
                self._stops.pop(name, None)
                return
            position_id = long_position_id(connector, self.config.trading_pair)
            if position_id is None:
                self._warn_once(
                    f"stop-{name}",
                    f"AIOT {name} has no long position id, so the stop was not placed.",
                )
                return
            if stop_id:
                await _cancel_stop(connector, stop_id)
                self._stops.pop(name, None)
            new_id = await connector.place_position_stop_order(
                self.config.trading_pair,
                position_id,
                _decimal(base),
                stop_loss_price=_decimal(stop_price),
                vol_type=1,
            )
            if not new_id:
                self.logger().error(f"AIOT {name} stop was not accepted.")
                return
            self._stops[name] = {
                "id": str(new_id),
                "amount": str(_decimal(base)),
                "price": str(_decimal(stop_price)),
            }
        except Exception as exc:
            self.logger().error(f"AIOT {name} stop update failed: {exc}")
        finally:
            self._stop_inflight.discard(name)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.logger().warning(f"AIOT state file unreadable, starting flat: {exc}")
            return
        if payload.get("strategy") != self._spec.key:
            return
        saved = {item.get("name"): item for item in payload.get("sleeves") or []}
        for sleeve in self._sleeves:
            state = saved.get(sleeve.spec.name)
            if state:
                sleeve.import_state(state)
        self._applied_base = {key: float(value) for key, value in (payload.get("applied_base") or {}).items()}
        self._last_bar = payload.get("last_bar")
        self._stops = payload.get("stops") or {}

    def _save_state(self) -> None:
        payload = {
            "version": 1,
            "strategy": self._spec.key,
            "last_bar": self._last_bar,
            "applied_base": self._applied_base,
            "stops": self._stops,
            "sleeves": [sleeve.export_state() for sleeve in self._sleeves],
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self.logger().warning(f"AIOT state file not written: {exc}")


def position_action_for(side) -> PositionAction:
    if side == TradeType.SELL:
        return PositionAction.CLOSE
    return PositionAction.OPEN


def max_entry_quote(ceiling, balance, leverage, buffer=Decimal("0.95")):
    if balance is None:
        return None
    try:
        cash = Decimal(str(balance))
        cap = Decimal(str(ceiling))
        room = cash * Decimal(int(leverage)) * Decimal(str(buffer))
    except (ArithmeticError, TypeError, ValueError):
        return None
    if cash <= 0 or cap <= 0 or room <= 0:
        return None
    return min(cap, room)


def scaled_base(notional, strategy_total, budget, price) -> Decimal:
    try:
        quote = Decimal(str(notional)) * Decimal(str(budget)) / Decimal(str(strategy_total))
        base = quote / Decimal(str(price))
    except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
        return Decimal("0")
    if base <= 0:
        return Decimal("0")
    return base


def split_base(amount, min_size, max_size, step=None):
    try:
        left = Decimal(str(amount))
        low = Decimal(str(min_size))
        high = Decimal(str(max_size))
        increment = None if step is None else Decimal(str(step))
    except (ArithmeticError, TypeError, ValueError):
        return []
    if left <= 0 or low <= 0 or high <= 0 or high < low:
        return []
    chunks = []
    while left >= low and len(chunks) < 50:
        chunk = high if left > high else left
        if increment is not None and increment > 0:
            chunk = (chunk // increment) * increment
        if chunk < low:
            break
        chunks.append(chunk)
        left -= chunk
    return chunks


def protective_stop_plan(stop_id, tracked_amount, tracked_price, base, stop_price) -> str:
    """Exchange stop stays on the hard stop. The trail is not an argument."""
    amount = Decimal("0") if base is None else _decimal(float(base))
    price = None if stop_price is None else _decimal(float(stop_price))
    if amount <= 0 or price is None or price <= 0:
        return "cancel" if stop_id else "hold"
    if stop_id and _same_text(tracked_amount, amount) and _same_text(tracked_price, price):
        return "hold"
    return "place"


def _same_text(tracked, current: Decimal) -> bool:
    if tracked is None:
        return False
    try:
        return Decimal(str(tracked)) == current
    except (ArithmeticError, TypeError, ValueError):
        return False


async def confirm_aiot_leverage(connector, trading_pair: str, leverage: int) -> bool:
    """Ask the connector for 1x isolated on the long side.

    MEXC rejects a leverage change while orders are open. The connector accepts
    that rejection only when a live read already shows this leverage and margin.
    """
    confirm = getattr(connector, "confirm_leverage_and_margin", None)
    if not callable(confirm):
        return False
    result = confirm(trading_pair, int(leverage), "ISOLATED", PositionSide.LONG)
    if hasattr(result, "__await__"):
        result = await result
    return bool(result)


def long_position_id(connector, trading_pair: str):
    ids = getattr(connector, "_position_ids", None)
    if isinstance(ids, dict):
        found = ids.get((trading_pair, PositionSide.LONG))
        if found is not None:
            return found
    finder = getattr(connector, "_position_id_for_order", None)
    if not callable(finder):
        return None
    try:
        return finder(trading_pair, TradeType.SELL, PositionAction.CLOSE, True)
    except Exception:
        return None


async def _cancel_stop(connector, stop_id) -> None:
    if not stop_id:
        return
    await connector.cancel_stop_orders([stop_id])


def _rung_index(parts) -> Optional[int]:
    if len(parts) > 3 and str(parts[3]).isdigit():
        return int(parts[3])
    return None


def _decimal(value: float) -> Decimal:
    return Decimal(f"{value:.8f}")


def _float_field(executor, name: str) -> float:
    info = getattr(executor, "custom_info", None) or {}
    value = info.get(name) if isinstance(info, dict) else None
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_active(executor) -> bool:
    if getattr(executor, "is_active", False):
        return True
    return False


def _is_done(executor) -> bool:
    if getattr(executor, "is_done", False):
        return True
    close_type = getattr(executor, "close_type", None)
    return close_type not in (None, CloseType.POSITION_HOLD) and not _is_active(executor)
