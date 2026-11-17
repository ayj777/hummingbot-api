"""Signal and exit engine for the AIOT research strategies.

The paper controller and the unit tests both drive this module. It does not
import hummingbot.

Timing. A signal is read on a closed bar. A market order is queued there and
filled at the next bar's open. In the paper bot that queue is a market order
sent as soon as the bar closes, which is the same moment.

A hard stop is a price. When a bar trades through it, the simulated fill is the
stop, or the open if the bar opens through the stop. The trailing stop is
different: the peak is the highest signal-bar close since the fill, and it
fires only when a later close is at or below that peak times (1 - trail). A
wick through the trail does not fire it.

A partial take-profit and the S2326 take-profit are resting limits. They fill
when a bar's high trades them, at the limit, or at the open if the bar gaps
through the limit.

When one bar trades both a resting limit and the stop, a bar that closes at or
above its open is treated as low-then-high, and a bar that closes below its
open as high-then-low.

Indicators. EMA is the standard recursive form with adjust=False: the first
seed is the first close, then value = alpha * close + (1 - alpha) * previous,
alpha = 2 / (span + 1). Values before `span` closes are withheld so a one-bar
EMA cannot trade. RSI14 and ATR14 are Wilder's: an SMA seed over the first
period, then (previous * (period - 1) + value) / period. The z-score uses the
SMA and the population standard deviation (divide by N, not N - 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

RUNG_THRESHOLDS = (32.0, 26.0, 20.0)
RUNG_NOTIONAL = 5000.0
DAY_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class Bar:
    """One closed candle. `open_time` is seconds."""

    open_time: int
    open: float
    high: float
    low: float
    close: float

    def close_time(self, interval_seconds: int) -> int:
        return self.open_time + interval_seconds


@dataclass(frozen=True)
class Context:
    """Indicators known at a signal bar's close. None means not warmed up."""

    rsi: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    zscore: Optional[float] = None
    regime_on: Optional[bool] = None
    atr: Optional[float] = None


@dataclass(frozen=True)
class SleeveSpec:
    name: str
    kind: str
    notional: float
    rsi_entry: Optional[float] = None
    rsi_exit: Optional[float] = None
    stop_pct: Optional[float] = None
    stop_atr_mult: Optional[float] = None
    trail_pct: Optional[float] = None
    time_limit_seconds: Optional[int] = None
    time_limit_bars: Optional[int] = None
    partial_gain: Optional[float] = None
    partial_fraction: Optional[float] = None
    take_profit_pct: Optional[float] = None
    z_entry: Optional[float] = None
    rearm: bool = False
    rung_notionals: Tuple[float, ...] = ()
    rung_thresholds: Tuple[float, ...] = ()


@dataclass(frozen=True)
class StrategySpec:
    key: str
    signal_interval: str
    htf_interval: Optional[str]
    htf_mode: Optional[str]
    total_quote: float
    sleeves: Tuple[SleeveSpec, ...]


def _trend(name: str, notional: float, **kwargs) -> SleeveSpec:
    return SleeveSpec(name=name, kind="trend", notional=notional, **kwargs)


def _rsi(name: str, notional: float, entry: float, exit_level: float) -> SleeveSpec:
    return SleeveSpec(
        name=name,
        kind="rsi",
        notional=notional,
        rsi_entry=entry,
        rsi_exit=exit_level,
        stop_pct=0.10,
        time_limit_seconds=DAY_SECONDS,
    )


STRATEGIES = {
    "f1": StrategySpec(
        key="f1",
        signal_interval="15m",
        htf_interval="1h",
        htf_mode="ema_cross",
        total_quote=15000.0,
        sleeves=(_trend("core", 15000.0, stop_pct=0.15, trail_pct=0.30, rearm=True),),
    ),
    "f3": StrategySpec(
        key="f3",
        signal_interval="15m",
        htf_interval=None,
        htf_mode=None,
        total_quote=15000.0,
        sleeves=(_rsi("core", 15000.0, 28.0, 70.0),),
    ),
    "rsi32": StrategySpec(
        key="rsi32",
        signal_interval="15m",
        htf_interval=None,
        htf_mode=None,
        total_quote=15000.0,
        sleeves=(_rsi("core", 15000.0, 32.0, 70.0),),
    ),
    "ladder": StrategySpec(
        key="ladder",
        signal_interval="15m",
        htf_interval=None,
        htf_mode=None,
        total_quote=15000.0,
        sleeves=(
            SleeveSpec(
                name="core",
                kind="ladder",
                notional=15000.0,
                rsi_exit=70.0,
                stop_pct=0.10,
                time_limit_seconds=DAY_SECONDS,
                rung_notionals=(RUNG_NOTIONAL, RUNG_NOTIONAL, RUNG_NOTIONAL),
                rung_thresholds=RUNG_THRESHOLDS,
            ),
        ),
    ),
    "fin_a": StrategySpec(
        key="fin_a",
        signal_interval="15m",
        htf_interval="1h",
        htf_mode="ema_cross",
        total_quote=15000.0,
        sleeves=(
            SleeveSpec(
                name="rsi",
                kind="rsi",
                notional=12000.0,
                rsi_entry=28.0,
                rsi_exit=75.0,
                stop_pct=0.10,
                time_limit_seconds=DAY_SECONDS,
            ),
            _trend(
                "trend",
                3000.0,
                stop_atr_mult=5.0,
                trail_pct=0.30,
                partial_gain=1.0,
                partial_fraction=0.50,
                rearm=True,
            ),
        ),
    ),
    "fin_b": StrategySpec(
        key="fin_b",
        signal_interval="15m",
        htf_interval="1h",
        htf_mode="ema_cross",
        total_quote=15000.0,
        sleeves=(
            _trend(
                "core",
                15000.0,
                stop_atr_mult=5.0,
                trail_pct=0.30,
                partial_gain=1.5,
                partial_fraction=0.67,
                rearm=True,
            ),
        ),
    ),
    "s2326": StrategySpec(
        key="s2326",
        signal_interval="5m",
        htf_interval="15m",
        htf_mode="ema_trend",
        total_quote=5000.0,
        sleeves=(
            SleeveSpec(
                name="core",
                kind="zscore",
                notional=5000.0,
                z_entry=-2.5,
                stop_pct=0.10,
                time_limit_bars=288,
                take_profit_pct=0.03,
            ),
        ),
    ),
}


@dataclass(frozen=True)
class Fill:
    sleeve: str
    side: str
    price: float
    base: float
    notional: float
    reason: str
    bar_index: int
    time: int


@dataclass
class Intent:
    sleeve: str
    kind: str
    base: float
    notional: float
    reason: str
    bar_index: int
    rung: Optional[int] = None
    price: Optional[float] = None


def ema_adjust_false(values: Sequence[float], span: int) -> List[float]:
    """Recursive EMA, adjust=False, including the warmup values."""
    if span <= 0:
        raise ValueError("span must be positive")
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


def ema(values: Sequence[float], span: int) -> List[Optional[float]]:
    raw = ema_adjust_false(values, span)
    return [value if index >= span - 1 else None for index, value in enumerate(raw)]


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def wilder_rsi(closes: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """Wilder RSI. The first value sits on the close that completes `period` changes."""
    out: List[Optional[float]] = [None] * len(closes)
    if period <= 0 or len(closes) <= period:
        return out
    gains: List[float] = []
    losses: List[float] = []
    for index in range(1, len(closes)):
        delta = closes[index] - closes[index - 1]
        gains.append(delta if delta > 0.0 else 0.0)
        losses.append(-delta if delta < 0.0 else 0.0)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = _rsi_from_averages(avg_gain, avg_loss)
    for index in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[index]) / period
        avg_loss = (avg_loss * (period - 1) + losses[index]) / period
        out[index + 1] = _rsi_from_averages(avg_gain, avg_loss)
    return out


def true_ranges(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> List[float]:
    if not closes:
        return []
    ranges = [highs[0] - lows[0]]
    for index in range(1, len(closes)):
        ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
        )
    return ranges


def wilder_atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> List[Optional[float]]:
    """Wilder ATR. The seed is the SMA of the first `period` true ranges."""
    ranges = true_ranges(highs, lows, closes)
    out: List[Optional[float]] = [None] * len(ranges)
    if period <= 0 or len(ranges) < period:
        return out
    average = sum(ranges[:period]) / period
    out[period - 1] = average
    for index in range(period, len(ranges)):
        average = (average * (period - 1) + ranges[index]) / period
        out[index] = average
    return out


def population_zscore(closes: Sequence[float], window: int = 48) -> List[Optional[float]]:
    """(close - SMA) / population stdev. A zero-deviation window is not a signal."""
    out: List[Optional[float]] = [None] * len(closes)
    if window <= 1:
        raise ValueError("window must be at least 2")
    for index in range(window - 1, len(closes)):
        sample = closes[index - window + 1:index + 1]
        mean = sum(sample) / window
        variance = sum((value - mean) ** 2 for value in sample) / window
        if variance <= 0.0:
            continue
        out[index] = (closes[index] - mean) / math.sqrt(variance)
    return out


def last_closed_index(bars: Sequence[Bar], interval_seconds: int, as_of: int) -> Optional[int]:
    """Latest bar whose close is at or before `as_of`. Bars must be in time order."""
    found = None
    for index, bar in enumerate(bars):
        if bar.open_time + interval_seconds <= as_of:
            found = index
        elif bar.open_time + interval_seconds > as_of:
            break
    return found


def build_contexts(
    signal_bars: Sequence[Bar],
    signal_interval: str,
    htf_bars: Optional[Sequence[Bar]] = None,
    htf_interval: Optional[str] = None,
    htf_mode: Optional[str] = None,
) -> List[Context]:
    """Indicator context for every signal bar, using closed higher-timeframe bars only."""
    closes = [bar.close for bar in signal_bars]
    rsi = wilder_rsi(closes, 14)
    ema50 = ema(closes, 50)
    ema200 = ema(closes, 200)
    zscores = population_zscore(closes, 48)
    htf_ema_fast: List[Optional[float]] = []
    htf_ema_slow: List[Optional[float]] = []
    htf_atr: List[Optional[float]] = []
    if htf_bars and htf_mode:
        htf_closes = [bar.close for bar in htf_bars]
        if htf_mode == "ema_cross":
            htf_ema_fast = ema(htf_closes, 20)
            htf_ema_slow = ema(htf_closes, 50)
            htf_atr = wilder_atr(
                [bar.high for bar in htf_bars],
                [bar.low for bar in htf_bars],
                htf_closes,
                14,
            )
        elif htf_mode == "ema_trend":
            htf_ema_fast = ema(htf_closes, 50)
            htf_ema_slow = ema(htf_closes, 200)
        else:
            raise ValueError(f"unknown htf_mode {htf_mode}")
    signal_seconds = INTERVAL_SECONDS[signal_interval]
    htf_seconds = INTERVAL_SECONDS[htf_interval] if htf_interval else None
    contexts: List[Context] = []
    for index, bar in enumerate(signal_bars):
        regime = None
        atr_value = None
        if htf_bars and htf_mode and htf_seconds is not None:
            closed = last_closed_index(htf_bars, htf_seconds, bar.close_time(signal_seconds))
            if closed is not None and htf_ema_fast[closed] is not None and htf_ema_slow[closed] is not None:
                regime = bool(htf_ema_fast[closed] > htf_ema_slow[closed])
            if closed is not None and htf_atr:
                atr_value = htf_atr[closed]
        contexts.append(
            Context(
                rsi=rsi[index],
                ema50=ema50[index],
                ema200=ema200[index],
                zscore=zscores[index],
                regime_on=regime,
                atr=atr_value,
            )
        )
    return contexts


class Sleeve:
    """One position state machine. Market orders wait for the next open."""

    def __init__(self, spec: SleeveSpec):
        self.spec = spec
        self.phase = "flat"
        self.base = 0.0
        self.rungs = [False, False, False]
        self.first_fill_price: Optional[float] = None
        self.first_fill_time: Optional[int] = None
        self.highest_close: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.limit_price: Optional[float] = None
        self.limit_base = 0.0
        self.entry_atr: Optional[float] = None
        self.bars_held = 0
        self.blocked = False
        self.seen_regime_off = False
        self.exit_reason: Optional[str] = None
        self.pending_saw_regime_off = False
        self.cancel_limit = False
        self._queued: List[Intent] = []

    def drain(self) -> List[Intent]:
        queued = self._queued
        self._queued = []
        return queued

    def on_price(self, price: float) -> None:
        """Live stop. Queues one market sell when price trades at or below the stop.

        The trail is not checked here. A wick through the trail level does nothing;
        only `on_close` can trail out, and only from the closed close.
        """
        if self.phase != "open" or self.stop_price is None or price > self.stop_price:
            return
        self.pending_saw_regime_off = False
        self._queue_sell("stop", bar_index=-1)

    def consume_open(self, bar: Bar, index: int) -> List[Fill]:
        fills = []
        for intent in self.drain():
            if intent.kind == "market_buy":
                fills.append(self.apply_buy_fill(bar.open, intent.base, intent.notional, bar.open_time, intent.reason, index))
            elif intent.kind == "market_sell":
                fills.append(self.apply_sell_fill(bar.open, bar.open_time, index))
        return fills

    def consume_range(self, bar: Bar, index: int) -> List[Fill]:
        """Resting stop and resting limit, with the open-to-close path rule."""
        if self.phase != "open" or self.base <= 0.0:
            return []
        stop = self.stop_price
        limit = self.limit_price if self.limit_base > 0.0 else None
        hit_stop = stop is not None and bar.low <= stop
        hit_limit = limit is not None and bar.high >= limit
        if hit_stop and hit_limit:
            if bar.open <= stop:
                return [self._exit_now(bar.open if bar.open < stop else stop, "stop", bar, index)]
            if bar.open >= limit:
                return self._limit_then_stop(bar.open, bar, index, stop)
            if bar.close >= bar.open:
                return [self._exit_now(stop, "stop", bar, index)]
            return self._limit_then_stop(limit, bar, index, stop)
        if hit_stop:
            price = bar.open if bar.open < stop else stop
            return [self._exit_now(price, "stop", bar, index)]
        if hit_limit:
            price = bar.open if bar.open > limit else limit
            return [self._fill_limit(price, bar, index)]
        return []

    def on_close(self, bar: Bar, ctx: Context, index: int, interval_seconds: int) -> None:
        if self.phase == "pending_entry" or self.phase == "pending_exit":
            return
        if self.phase != "open":
            self._enter_if_allowed(bar, ctx, index)
            return
        self.bars_held += 1
        self._trail(bar, ctx, index)
        if self.phase != "open":
            return
        if self._signal_exit(bar, ctx, index):
            return
        if self._time_exit(bar, index, interval_seconds):
            return
        if self.spec.kind == "ladder":
            self._queue_rungs(bar, ctx, index, entry=False)

    def apply_buy_fill(
        self,
        price: float,
        base: float,
        notional: float,
        time: int,
        reason: str,
        bar_index: int,
    ) -> Fill:
        self.base += base
        if reason.startswith("rung") and reason[-1].isdigit():
            rung_index = int(reason[-1]) - 1
            if 0 <= rung_index < len(self.rungs):
                self.rungs[rung_index] = True
        if self.first_fill_price is None:
            self.first_fill_price = price
            self.first_fill_time = time
            self.stop_price = self._stop_from(price)
            self._arm_limit(base)
        self.phase = "open"
        self.exit_reason = None
        return Fill(self.spec.name, "buy", price, base, notional, reason, bar_index, time)

    def apply_sell_fill(self, price: float, time: int, bar_index: int) -> Fill:
        reason = self.exit_reason or "exit"
        saw_off = self.pending_saw_regime_off
        base = self.base
        notional = base * price
        rearm = reason in ("stop", "trail") and self.spec.rearm
        self._flatten()
        if rearm:
            self.blocked = True
            self.seen_regime_off = saw_off
        return Fill(self.spec.name, "sell", price, base, notional, reason, bar_index, time)

    def apply_limit_fill(self, price: float, base: float, time: int, bar_index: int) -> Fill:
        return self._fill_limit_amount(price, base, time, bar_index)

    def abort_buy(self, rung: Optional[int], any_buys_left: bool) -> None:
        if rung is not None:
            self.rungs[rung] = False
        if not any_buys_left and self.base <= 0.0:
            self.phase = "flat"

    def abort_sell(self) -> None:
        if self.phase == "pending_exit" and self.base > 0.0:
            self.phase = "open"
            self.exit_reason = None

    def snapshot(self) -> dict:
        return {
            "name": self.spec.name,
            "phase": self.phase,
            "base": self.base,
            "first_fill_price": self.first_fill_price,
            "first_fill_time": self.first_fill_time,
            "stop_price": self.stop_price,
            "limit_price": self.limit_price,
            "limit_base": self.limit_base,
            "highest_close": self.highest_close,
            "entry_atr": self.entry_atr,
            "bars_held": self.bars_held,
            "blocked": self.blocked,
            "seen_regime_off": self.seen_regime_off,
            "exit_reason": self.exit_reason,
            "pending_saw_regime_off": self.pending_saw_regime_off,
            "rungs": list(self.rungs),
        }

    def export_state(self) -> dict:
        """Position state worth keeping across a process restart. In-flight orders are not."""
        state = self.snapshot()
        if state["phase"] == "pending_entry":
            state["phase"] = "flat"
            state["rungs"] = [False, False, False]
        elif state["phase"] == "pending_exit":
            state["phase"] = "open" if state["base"] > 0.0 else "flat"
        return state

    def import_state(self, state: dict) -> None:
        self.phase = state.get("phase", "flat")
        self.base = float(state.get("base") or 0.0)
        self.first_fill_price = state.get("first_fill_price")
        self.first_fill_time = state.get("first_fill_time")
        self.stop_price = state.get("stop_price")
        self.limit_price = state.get("limit_price")
        self.limit_base = float(state.get("limit_base") or 0.0)
        self.highest_close = state.get("highest_close")
        self.entry_atr = state.get("entry_atr")
        self.bars_held = int(state.get("bars_held") or 0)
        self.blocked = bool(state.get("blocked"))
        self.seen_regime_off = bool(state.get("seen_regime_off"))
        self.exit_reason = state.get("exit_reason")
        self.pending_saw_regime_off = bool(state.get("pending_saw_regime_off"))
        rungs = state.get("rungs") or [False, False, False]
        self.rungs = [bool(flag) for flag in rungs[:3]]
        while len(self.rungs) < 3:
            self.rungs.append(False)

    def _enter_if_allowed(self, bar: Bar, ctx: Context, index: int) -> None:
        if not self._rearm_allows(ctx.regime_on):
            return
        if self.spec.kind == "trend":
            if ctx.regime_on is True and self._atr_ready(ctx):
                self.entry_atr = ctx.atr
                self._queue_buy(self.spec.notional, bar.close, "entry", index, None)
        elif self.spec.kind == "rsi":
            if self._rsi_trend(ctx) and ctx.rsi is not None and ctx.rsi <= self.spec.rsi_entry:
                self._queue_buy(self.spec.notional, bar.close, "entry", index, None)
        elif self.spec.kind == "ladder":
            self._queue_rungs(bar, ctx, index, entry=True)
        elif self.spec.kind == "zscore":
            if ctx.zscore is not None and ctx.zscore <= self.spec.z_entry and ctx.regime_on is True:
                self._queue_buy(self.spec.notional, bar.close, "entry", index, None)

    def _rearm_allows(self, regime_on: Optional[bool]) -> bool:
        if not self.blocked:
            return True
        if regime_on is not True:
            self.seen_regime_off = True
            return False
        if self.seen_regime_off:
            self.blocked = False
            self.seen_regime_off = False
            return True
        return False

    def _atr_ready(self, ctx: Context) -> bool:
        if self.spec.stop_atr_mult is None:
            return True
        return ctx.atr is not None and ctx.atr > 0.0

    def _rsi_trend(self, ctx: Context) -> bool:
        return ctx.ema50 is not None and ctx.ema200 is not None and ctx.ema50 > ctx.ema200

    def _queue_rungs(self, bar: Bar, ctx: Context, index: int, entry: bool) -> None:
        if ctx.rsi is None:
            return
        thresholds = self.spec.rung_thresholds or RUNG_THRESHOLDS
        notionals = self.spec.rung_notionals or (RUNG_NOTIONAL, RUNG_NOTIONAL, RUNG_NOTIONAL)
        for rung, threshold in enumerate(thresholds):
            if self.rungs[rung] or ctx.rsi > threshold:
                continue
            if rung == 0:
                if not entry or not self._rsi_trend(ctx):
                    return
            elif not self.rungs[0]:
                return
            self.rungs[rung] = True
            self._queue_buy(notionals[rung], bar.close, f"rung{rung + 1}", index, rung)

    def _queue_buy(self, notional: float, price_ref: float, reason: str, index: int, rung: Optional[int]) -> None:
        if price_ref <= 0.0 or notional <= 0.0:
            if rung is not None:
                self.rungs[rung] = False
            return
        base = notional / price_ref
        self._queued.append(Intent(self.spec.name, "market_buy", base, notional, reason, index, rung, price_ref))
        if self.phase == "flat":
            self.phase = "pending_entry"

    def _trail(self, bar: Bar, ctx: Context, index: int) -> None:
        if self.spec.trail_pct is None:
            return
        self.highest_close = bar.close if self.highest_close is None else max(self.highest_close, bar.close)
        if bar.close <= self.highest_close * (1.0 - self.spec.trail_pct):
            self.pending_saw_regime_off = ctx.regime_on is False
            self._queue_sell("trail", index)

    def _signal_exit(self, bar: Bar, ctx: Context, index: int) -> bool:
        if self.spec.kind == "trend" and ctx.regime_on is False:
            self._queue_sell("cross", index)
            return True
        if self.spec.rsi_exit is not None and ctx.rsi is not None and ctx.rsi >= self.spec.rsi_exit:
            self._queue_sell("rsi", index)
            return True
        return False

    def _time_exit(self, bar: Bar, index: int, interval_seconds: int) -> bool:
        if self.first_fill_time is None:
            return False
        if self.spec.time_limit_bars is not None and self.bars_held >= self.spec.time_limit_bars:
            self._queue_sell("time", index)
            return True
        if self.spec.time_limit_seconds is not None:
            if bar.close_time(interval_seconds) - self.first_fill_time >= self.spec.time_limit_seconds:
                self._queue_sell("time", index)
                return True
        return False

    def _queue_sell(self, reason: str, bar_index: int) -> None:
        if self.phase != "open" or self.base <= 0.0:
            return
        self.phase = "pending_exit"
        self.exit_reason = reason
        if self.limit_price is not None:
            self.cancel_limit = True
            self.limit_price = None
            self.limit_base = 0.0
        self._queued.append(Intent(self.spec.name, "market_sell", self.base, self.base, reason, bar_index))

    def _stop_from(self, price: float) -> Optional[float]:
        if self.spec.stop_pct is not None:
            return price * (1.0 - self.spec.stop_pct)
        if self.spec.stop_atr_mult is not None and self.entry_atr is not None:
            return price - self.spec.stop_atr_mult * self.entry_atr
        return None

    def _arm_limit(self, base: float) -> None:
        if self.first_fill_price is None:
            return
        if self.spec.take_profit_pct is not None:
            self.limit_price = self.first_fill_price * (1.0 + self.spec.take_profit_pct)
            self.limit_base = self.base
        elif self.spec.partial_gain is not None and self.spec.partial_fraction is not None:
            self.limit_price = self.first_fill_price * (1.0 + self.spec.partial_gain)
            self.limit_base = base * self.spec.partial_fraction

    def _limit_then_stop(self, limit_price: float, bar: Bar, index: int, stop: float) -> List[Fill]:
        fills = [self._fill_limit(limit_price, bar, index)]
        if self.phase == "open" and self.base > 0.0 and bar.low <= stop:
            fills.append(self._exit_now(stop, "stop", bar, index))
        return fills

    def _fill_limit(self, price: float, bar: Bar, index: int) -> Fill:
        return self._fill_limit_amount(price, self.limit_base, bar.open_time, index)

    def _fill_limit_amount(self, price: float, base: float, time: int, bar_index: int) -> Fill:
        sold = min(base, self.base)
        full = sold >= self.base - 1e-12
        self.base -= sold
        self.limit_price = None
        self.limit_base = 0.0
        if full or self.base <= 1e-12:
            reason = "take_profit"
            self.base = 0.0
            self._flatten()
        else:
            reason = "partial"
        return Fill(self.spec.name, "sell", price, sold, sold * price, reason, bar_index, time)

    def _exit_now(self, price: float, reason: str, bar: Bar, index: int) -> Fill:
        base = self.base
        self.exit_reason = reason
        rearm = reason in ("stop", "trail") and self.spec.rearm
        self._flatten()
        if rearm:
            self.blocked = True
            self.seen_regime_off = False
        return Fill(self.spec.name, "sell", price, base, base * price, reason, index, bar.open_time)

    def _flatten(self) -> None:
        self.phase = "flat"
        self.base = 0.0
        self.rungs = [False, False, False]
        self.first_fill_price = None
        self.first_fill_time = None
        self.highest_close = None
        self.stop_price = None
        self.limit_price = None
        self.limit_base = 0.0
        self.entry_atr = None
        self.bars_held = 0
        self.exit_reason = None
        self.cancel_limit = False


@dataclass
class Simulation:
    fills: List[Fill] = field(default_factory=list)
    sleeves: List[Sleeve] = field(default_factory=list)


def simulate_contexts(
    specs: Sequence[SleeveSpec],
    bars: Sequence[Bar],
    contexts: Sequence[Context],
    interval_seconds: int,
) -> Simulation:
    sleeves = [Sleeve(spec) for spec in specs]
    fills: List[Fill] = []
    for index, bar in enumerate(bars):
        for sleeve in sleeves:
            fills.extend(sleeve.consume_open(bar, index))
            fills.extend(sleeve.consume_range(bar, index))
            sleeve.on_close(bar, contexts[index], index, interval_seconds)
    return Simulation(fills, sleeves)


def simulate(
    strategy: StrategySpec,
    signal_bars: Sequence[Bar],
    htf_bars: Optional[Sequence[Bar]] = None,
) -> Simulation:
    contexts = build_contexts(
        signal_bars,
        strategy.signal_interval,
        htf_bars,
        strategy.htf_interval,
        strategy.htf_mode,
    )
    return simulate_contexts(
        strategy.sleeves,
        signal_bars,
        contexts,
        INTERVAL_SECONDS[strategy.signal_interval],
    )
