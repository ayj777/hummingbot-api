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
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from hummingbot.core.data_type.common import PositionAction, PositionMode, PriceType, TradeType
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
    position_mode: PositionMode = Field(default="ONEWAY")
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
        expected = Decimal(str(spec.total_quote))
        if self.total_amount_quote != expected:
            raise ValueError(
                f"strategy {self.strategy} is fixed at {expected} quote, "
                f"got total_amount_quote {self.total_amount_quote}"
            )
        if any(value is not None for value in (self.stop_loss, self.take_profit, self.time_limit, self.trailing_stop)):
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
                action = self._intent_action(intent)
                if action is not None:
                    actions.append(action)
            if sleeve.limit_price and sleeve.limit_base > 0 and sleeve.spec.name not in self._limit_live:
                limit = self._limit_action(sleeve)
                if limit is not None:
                    actions.append(limit)
                    self._limit_live.add(sleeve.spec.name)
            if sleeve.limit_price is None:
                self._limit_live.discard(sleeve.spec.name)
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
                    if len(parts) > 3:
                        rung = int(parts[3])
                        if 0 <= rung < len(sleeve.rungs):
                            sleeve.rungs[rung] = True
                continue
            if filled > recorded + 1e-12 and price > 0.0:
                delta = filled - recorded
                self._applied_base[executor.id] = filled
                now = int(self._now())
                if role == "buy":
                    notional = delta * price
                    rung_reason = f"rung{int(parts[3]) + 1}" if len(parts) > 3 else "entry"
                    sleeve.apply_buy_fill(price, delta, notional, now, rung_reason, -1)
                elif role == "sell":
                    sleeve.exit_reason = sleeve.exit_reason or "exit"
                    sleeve.apply_sell_fill(price, now, -1)
                elif role == "limit":
                    sleeve.apply_limit_fill(price, delta, now, -1)
            elif done and filled <= 0.0 and executor.id not in self._applied_base:
                self._applied_base[executor.id] = 0.0
                if role == "buy":
                    rung = int(parts[3]) if len(parts) > 3 else None
                    sleeve.abort_buy(rung, self._buys_still_live(sleeve.spec.name, executor.id))
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

    def _intent_action(self, intent: Intent) -> Optional[CreateExecutorAction]:
        if intent.base <= 0.0:
            return None
        if intent.kind == "market_buy":
            side = TradeType.BUY
            role = "buy" if intent.rung is None else f"buy:{intent.rung}"
        elif intent.kind == "market_sell":
            side = TradeType.SELL
            role = "sell"
        else:
            return None
        return self._create(side, intent.base, None, f"aiot:{intent.sleeve}:{role}", ExecutionStrategy.MARKET)

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
            position_action=PositionAction.OPEN,
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

    def _save_state(self) -> None:
        payload = {
            "version": 1,
            "strategy": self._spec.key,
            "last_bar": self._last_bar,
            "applied_base": self._applied_base,
            "sleeves": [sleeve.export_state() for sleeve in self._sleeves],
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            self.logger().warning(f"AIOT state file not written: {exc}")


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
