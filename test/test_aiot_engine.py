"""Signal and exit rules for the AIOT paper strategies.

The cases below drive the pure engine on synthetic candles. A signal is taken
on a bar's close and a market order fills at the next bar's open. A hard stop
fills inside the bar that trades it. A trail fires only on a close.
"""

from pathlib import Path

import pytest
import yaml

from bots.controllers.directional_trading.aiot_engine import (
    STRATEGIES,
    Bar,
    Context,
    SleeveSpec,
    build_contexts,
    ema_adjust_false,
    population_zscore,
    simulate_contexts,
    true_ranges,
    wilder_atr,
    wilder_rsi,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {
    "f1": "aiot_f1",
    "f3": "aiot_f3",
    "rsi32": "aiot_rsi32",
    "ladder": "aiot_ladder",
    "fin_a": "aiot_fin_a",
    "fin_b": "aiot_fin_b",
    "s2326": "aiot_s2326",
}


def _bars(rows, step=900, start=1_700_000_000):
    return [Bar(start + index * step, row[0], row[1], row[2], row[3]) for index, row in enumerate(rows)]


def _flat(price, count):
    return [(price, price, price, price) for _ in range(count)]


def _run(strategy, rows, contexts, step=900):
    return simulate_contexts(strategy.sleeves if hasattr(strategy, "sleeves") else strategy, _bars(rows, step), contexts, step)


def _buys(result, sleeve=None):
    return [fill for fill in result.fills if fill.side == "buy" and (sleeve is None or fill.sleeve == sleeve)]


def _sells(result, sleeve=None):
    return [fill for fill in result.fills if fill.side == "sell" and (sleeve is None or fill.sleeve == sleeve)]


def _uptrend():
    return Context(ema50=2.0, ema200=1.0)


def _downtrend():
    return Context(ema50=1.0, ema200=2.0)


# --- indicators -------------------------------------------------------------


def test_ema_adjust_false_matches_the_recursive_definition():
    values = ema_adjust_false([10.0, 12.0, 11.0, 13.0], 3)
    # alpha = 2 / 4 = 0.5
    assert values[0] == 10.0
    assert values[1] == pytest.approx(11.0)
    assert values[2] == pytest.approx(11.0)
    assert values[3] == pytest.approx(12.0)


def test_wilder_rsi_uses_the_sma_seed_then_wilder_smoothing():
    closes = [10.0, 11.0, 12.0, 11.0, 10.0, 9.0, 10.0]
    rsi = wilder_rsi(closes, period=3)
    # Seed averages: gain 2/3, loss 1/3, RS = 2, RSI = 100 - 100/3.
    assert rsi[3] == pytest.approx(100.0 - 100.0 / 3.0)
    # Next loss: avg_gain = 4/9, avg_loss = 5/9, RS = 0.8.
    assert rsi[4] == pytest.approx(100.0 - 100.0 / 1.8)
    assert rsi[0] is None
    assert wilder_rsi([1.0, 2.0, 3.0, 4.0], period=3)[3] == pytest.approx(100.0)


def test_wilder_atr_seeds_with_the_sma_of_the_first_period():
    highs = [10.0, 12.0, 11.0, 13.0]
    lows = [8.0, 9.0, 9.0, 10.0]
    closes = [9.0, 11.0, 10.0, 12.0]
    ranges = true_ranges(highs, lows, closes)
    assert ranges[0] == pytest.approx(2.0)
    assert ranges[1] == pytest.approx(max(3.0, abs(12.0 - 9.0), abs(9.0 - 9.0)))
    atr = wilder_atr(highs, lows, closes, period=3)
    assert atr[2] == pytest.approx(sum(ranges[:3]) / 3.0)
    assert atr[3] == pytest.approx((atr[2] * 2.0 + ranges[3]) / 3.0)
    assert atr[1] is None


def test_zscore_uses_the_population_standard_deviation():
    closes = [1.0, 2.0, 3.0, 4.0, 5.0]
    score = population_zscore(closes, window=5)[4]
    mean = 3.0
    stdev = (2.0) ** 0.5  # variance (4+1+0+1+4) / 5 = 2
    assert score == pytest.approx((5.0 - mean) / stdev)
    assert population_zscore([5.0] * 5, window=5)[4] is None


def test_flat_hours_are_not_a_bullish_regime_and_a_forming_hour_cannot_flip_one():
    hours = [Bar(index * 3600, 100.0, 101.0, 99.0, 100.0) for index in range(60)]
    signal = [Bar(59 * 3600, 100.0, 100.0, 100.0, 100.0)]
    context = build_contexts(signal, "15m", hours, "1h", "ema_cross")[0]
    assert context.regime_on is False

    rising = [Bar(index * 3600, 100.0 + index, 101.0 + index, 99.0 + index, 100.0 + index) for index in range(70)]
    crash = []
    for index in range(70, 90):
        price = 1000.0 - (index - 69) * 80.0
        crash.append(Bar(index * 3600, price, price + 1.0, price - 1.0, price))
    hours = rising + crash
    contexts = build_contexts(
        [Bar(bar.open_time + 3600, 1, 1, 1, 1) for bar in hours],
        "15m",
        hours,
        "1h",
        "ema_cross",
    )
    # A signal bar whose close equals an hour's open has NOT closed that hour.
    # Its regime is the previous hour. The bar that closes with the hour sees the new one.
    flipped = next(index for index, ctx in enumerate(contexts) if ctx.regime_on is False and index > 60)
    before = build_contexts(
        [Bar(hours[flipped].open_time - 900, 1, 1, 1, 1)],
        "15m",
        hours,
        "1h",
        "ema_cross",
    )[0]
    after = build_contexts(
        [Bar(hours[flipped].open_time + 3600 - 900, 1, 1, 1, 1)],
        "15m",
        hours,
        "1h",
        "ema_cross",
    )[0]
    assert before.regime_on is True
    assert after.regime_on is False


# --- F1 --------------------------------------------------------------------


def test_f1_buys_the_next_open_while_the_hourly_regime_is_on_and_flat():
    rows = [(100, 100, 100, 100), (110, 112, 108, 111), (111, 111, 111, 111)]
    contexts = [Context(regime_on=True, atr=1.0)] * 3
    result = _run(STRATEGIES["f1"], rows, contexts)
    buys = _buys(result)
    assert len(buys) == 1
    assert buys[0].price == pytest.approx(110.0)
    assert buys[0].notional == pytest.approx(15000.0)
    assert buys[0].base == pytest.approx(15000.0 / 100.0)
    assert result.sleeves[0].stop_price == pytest.approx(110.0 * 0.85)
    assert _sells(result) == []


def test_f1_does_not_buy_when_the_regime_is_off():
    rows = _flat(100, 4)
    contexts = [Context(regime_on=False, atr=1.0)] * 4
    assert _buys(_run(STRATEGIES["f1"], rows, contexts)) == []


def test_f1_sells_on_the_hourly_cross_and_can_reenter_when_it_turns_back_on():
    rows = _flat(100, 8)
    contexts = [
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=False),
        Context(regime_on=False),
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=True),
    ]
    result = _run(STRATEGIES["f1"], rows, contexts)
    sells = _sells(result)
    buys = _buys(result)
    assert [fill.reason for fill in sells] == ["cross"]
    assert sells[0].bar_index == 3
    assert sells[0].price == pytest.approx(100.0)
    assert [fill.bar_index for fill in buys] == [1, 5]


def test_f1_hard_stop_fills_at_the_stop_when_a_wick_trades_it_and_uses_the_fill():
    rows = [
        (100, 100, 100, 100),
        (110, 112, 108, 111),
        (111, 112, 90, 105),
    ]
    contexts = [Context(regime_on=True)] * 3
    result = _run(STRATEGIES["f1"], rows, contexts)
    sell = _sells(result)[0]
    assert sell.reason == "stop"
    assert sell.price == pytest.approx(110.0 * 0.85)
    assert sell.bar_index == 2
    # 90 is below a stop measured from the 110 fill, and above a stop measured
    # from the signal close of 100 (85). The fill is the reference.


def test_f1_stop_gaps_through_at_the_open():
    rows = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (70, 72, 68, 71),
    ]
    contexts = [Context(regime_on=True)] * 3
    sell = _sells(_run(STRATEGIES["f1"], rows, contexts))[0]
    assert sell.reason == "stop"
    assert sell.price == pytest.approx(70.0)


def test_f1_trail_uses_the_highest_close_and_ignores_wicks_and_the_signal_close():
    rows = [
        (1000, 1000, 1000, 1000),  # signal close must not become the peak
        (100, 200, 99, 130),  # fill 100, close 130 is the peak, high 200 is not
        (130, 140, 110, 120),  # 120 is above 70% of 130, so the trail holds
        (120, 125, 90, 90),  # close 90 breaks 91, and the low stays above the 85 stop
        (100, 100, 100, 100),
    ]
    contexts = [Context(regime_on=True)] * 5
    result = _run(STRATEGIES["f1"], rows, contexts)
    sells = _sells(result)
    assert len(sells) == 1
    assert sells[0].reason == "trail"
    assert sells[0].bar_index == 4
    assert sells[0].price == pytest.approx(100.0)
    assert result.sleeves[0].blocked is True


def test_f1_live_price_check_does_not_use_the_trail():
    """The paper bot's mid-price check is the hard stop only."""
    sleeve = simulate_contexts(
        STRATEGIES["f1"].sleeves,
        _bars([(100, 100, 100, 100), (100, 101, 99, 100)]),
        [Context(regime_on=True), Context(regime_on=True)],
        900,
    ).sleeves[0]
    # Hard stop sits at 85. A mid above it must not sell, even though a trail
    # computed from the 100 close would sit at 70.
    sleeve.on_price(90.0)
    assert sleeve.drain() == []
    sleeve.on_price(85.0)
    sold = sleeve.drain()
    assert [intent.reason for intent in sold] == ["stop"]
    sleeve.on_price(80.0)
    assert sleeve.drain() == []


def test_f1_rearm_requires_the_regime_to_turn_off_and_on_after_a_stop():
    rows = _flat(100, 9)
    rows[1] = (100, 100, 80, 100)  # wick through the 85 stop, close back above it
    contexts = [
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=False),
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=True),
        Context(regime_on=True),
    ]
    buys = _buys(_run(STRATEGIES["f1"], rows, contexts))
    assert [fill.bar_index for fill in buys] == [1, 6]


def test_f1_has_no_time_stop():
    rows = _flat(100, 40)
    contexts = [Context(regime_on=True)] * 40
    result = _run(STRATEGIES["f1"], rows, contexts)
    assert len(_buys(result)) == 1
    assert _sells(result) == []
    assert result.sleeves[0].phase == "open"


# --- F3 and RSI32 -----------------------------------------------------------


def test_f3_requires_rsi_and_the_15m_trend_and_exits_at_70():
    rows = _flat(100, 6)
    contexts = [
        Context(rsi=28.0, ema50=2.0, ema200=1.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0),
        Context(rsi=69.9, ema50=2.0, ema200=1.0),
        Context(rsi=70.0, ema50=2.0, ema200=1.0),
        Context(rsi=80.0, ema50=2.0, ema200=1.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0),
    ]
    result = _run(STRATEGIES["f3"], rows, contexts)
    assert _buys(result)[0].notional == pytest.approx(15000.0)
    assert _buys(result)[0].price == pytest.approx(100.0)
    sell = _sells(result)[0]
    assert sell.reason == "rsi"
    assert sell.bar_index == 4


def test_f3_does_not_enter_without_the_trend_or_above_28():
    rows = _flat(100, 3)
    no_trend = _run(STRATEGIES["f3"], rows, [Context(rsi=20.0, ema50=1.0, ema200=2.0)] * 3)
    too_high = _run(STRATEGIES["f3"], rows, [Context(rsi=28.01, ema50=2.0, ema200=1.0)] * 3)
    assert _buys(no_trend) == []
    assert _buys(too_high) == []


def test_rsi32_enters_between_28_and_32_and_f3_does_not():
    rows = _flat(100, 3)
    contexts = [Context(rsi=30.0, ema50=2.0, ema200=1.0)] * 3
    assert _buys(_run(STRATEGIES["rsi32"], rows, contexts))[0].notional == pytest.approx(15000.0)
    assert _buys(_run(STRATEGIES["f3"], rows, contexts)) == []
    assert _buys(_run(STRATEGIES["rsi32"], rows, [Context(rsi=32.01, ema50=2.0, ema200=1.0)] * 3)) == []


def test_f3_stop_is_10_percent_under_the_fill_and_beats_the_24h_clock():
    rows = _flat(100, 6)
    rows[2] = (100, 101, 90, 99)
    contexts = [Context(rsi=28.0, ema50=2.0, ema200=1.0)] + [Context(rsi=40.0, ema50=2.0, ema200=1.0)] * 5
    sells = _sells(_run(STRATEGIES["f3"], rows, contexts))
    assert len(sells) == 1
    assert sells[0].reason == "stop"
    assert sells[0].price == pytest.approx(90.0)


def test_f3_time_stop_is_24h_after_the_fill_when_nothing_else_hits():
    rows = _flat(100, 100)
    contexts = [Context(rsi=28.0, ema50=2.0, ema200=1.0)] + [Context(rsi=40.0, ema50=2.0, ema200=1.0)] * 99
    result = _run(STRATEGIES["f3"], rows, contexts)
    sell = _sells(result)[0]
    buy = _buys(result)[0]
    assert sell.reason == "time"
    assert sell.time - buy.time == 24 * 60 * 60


def test_f3_wiring_enters_from_real_rsi_on_a_synthetic_uptrend_then_dump():
    prices = [100.0 + index * 0.4 for index in range(260)]
    last = prices[-1]
    for _ in range(6):
        last *= 0.90
        prices.append(last)
    bars = [Bar(index * 900, price, price * 1.001, price * 0.999, price) for index, price in enumerate(prices)]
    contexts = build_contexts(bars, "15m")
    result = simulate_contexts(STRATEGIES["f3"].sleeves, bars, contexts, 900)
    buys = _buys(result)
    assert buys, "the dump should push RSI through 28 while EMA50 is still above EMA200"
    signal = buys[0].bar_index - 1
    assert contexts[signal].rsi <= 28.0
    assert contexts[signal].ema50 > contexts[signal].ema200
    assert buys[0].notional == pytest.approx(15000.0)
    assert buys[0].price == pytest.approx(bars[buys[0].bar_index].open)


# --- ladder -----------------------------------------------------------------


def test_ladder_buys_one_rung_at_32_and_adds_without_rechecking_trend():
    rows = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (80, 80, 80, 80),
        (80, 80, 80, 80),
    ]
    contexts = [
        Context(rsi=30.0, ema50=2.0, ema200=1.0),
        Context(rsi=25.0, ema50=1.0, ema200=2.0),
        Context(rsi=40.0, ema50=1.0, ema200=2.0),
        Context(rsi=40.0, ema50=1.0, ema200=2.0),
    ]
    buys = _buys(_run(STRATEGIES["ladder"], rows, contexts))
    assert [fill.reason for fill in buys] == ["rung1", "rung2"]
    assert [fill.notional for fill in buys] == pytest.approx([5000.0, 5000.0])
    assert buys[0].price == pytest.approx(100.0)
    assert buys[1].price == pytest.approx(80.0)


def test_ladder_takes_every_unfilled_rung_on_a_single_close_and_then_caps():
    rows = _flat(100, 4)
    contexts = [
        Context(rsi=19.0, ema50=2.0, ema200=1.0),
        Context(rsi=10.0, ema50=2.0, ema200=1.0),
        Context(rsi=10.0, ema50=2.0, ema200=1.0),
        Context(rsi=10.0, ema50=2.0, ema200=1.0),
    ]
    buys = _buys(_run(STRATEGIES["ladder"], rows, contexts))
    assert [fill.reason for fill in buys] == ["rung1", "rung2", "rung3"]
    assert sum(fill.notional for fill in buys) == pytest.approx(15000.0)


def test_ladder_does_not_start_when_the_trend_is_off():
    rows = _flat(100, 3)
    contexts = [Context(rsi=10.0, ema50=1.0, ema200=2.0)] * 3
    assert _buys(_run(STRATEGIES["ladder"], rows, contexts)) == []


def test_ladder_stop_is_ten_percent_under_the_first_fill_and_closes_every_rung():
    rows = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (110, 112, 108, 111),  # second rung fills above the stop
        (100, 101, 91, 95),  # 91 stays above the 90 stop
        (95, 96, 90, 94),
    ]
    contexts = [
        Context(rsi=30.0, ema50=2.0, ema200=1.0),
        Context(rsi=25.0, ema50=1.0, ema200=2.0),
        Context(rsi=40.0, ema50=1.0, ema200=2.0),
        Context(rsi=40.0, ema50=1.0, ema200=2.0),
        Context(rsi=40.0, ema50=1.0, ema200=2.0),
    ]
    result = _run(STRATEGIES["ladder"], rows, contexts)
    sells = _sells(result)
    assert len(sells) == 1
    assert sells[0].reason == "stop"
    assert sells[0].price == pytest.approx(90.0)
    assert sells[0].base == pytest.approx(5000.0 / 100.0 + 5000.0 / 100.0)
    assert _buys(result)[1].price == pytest.approx(110.0)


def test_ladder_exits_the_whole_book_at_rsi_70_or_24h_and_then_resets():
    rows = _flat(100, 8)
    contexts = [
        Context(rsi=19.0, ema50=2.0, ema200=1.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0),
        Context(rsi=70.0, ema50=2.0, ema200=1.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0),
        Context(rsi=30.0, ema50=2.0, ema200=1.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0),
    ]
    result = _run(STRATEGIES["ladder"], rows, contexts)
    assert _sells(result)[0].reason == "rsi"
    assert _sells(result)[0].base == pytest.approx(150.0)
    assert [fill.reason for fill in _buys(result)] == ["rung1", "rung2", "rung3", "rung1"]

    rows = _flat(100, 100)
    contexts = [Context(rsi=28.0, ema50=2.0, ema200=1.0)] + [Context(rsi=40.0, ema50=2.0, ema200=1.0)] * 99
    timed = _run(STRATEGIES["ladder"], rows, contexts)
    sell = _sells(timed)[0]
    assert sell.reason == "time"
    assert sell.time - _buys(timed)[0].time == 24 * 60 * 60


# --- FIN-A and FIN-B --------------------------------------------------------


def test_fin_a_runs_both_sleeves_and_the_rsi_sleeve_exits_at_75():
    rows = _flat(100, 6)
    contexts = [
        Context(rsi=28.0, ema50=2.0, ema200=1.0, regime_on=True, atr=2.0),
        Context(rsi=72.0, ema50=2.0, ema200=1.0, regime_on=True, atr=2.0),
        Context(rsi=75.0, ema50=2.0, ema200=1.0, regime_on=True, atr=2.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0, regime_on=True, atr=2.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0, regime_on=True, atr=2.0),
        Context(rsi=40.0, ema50=2.0, ema200=1.0, regime_on=True, atr=2.0),
    ]
    result = _run(STRATEGIES["fin_a"], rows, contexts)
    assert _buys(result, "rsi")[0].notional == pytest.approx(12000.0)
    assert _buys(result, "trend")[0].notional == pytest.approx(3000.0)
    assert _sells(result, "rsi")[0].reason == "rsi"
    assert _sells(result, "rsi")[0].bar_index == 3
    assert _sells(result, "trend") == []


def test_fin_a_trend_sleeve_stops_at_five_atrs_from_the_signal_bar_and_takes_half_at_double():
    rows = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 219, 96, 150),  # 219 trades the 200 limit; 96 stays above the 90 stop
        (150, 220, 140, 180),
        (180, 180, 180, 180),
    ]
    contexts = [
        Context(regime_on=True, atr=2.0, rsi=50.0, ema50=1.0, ema200=2.0),
        Context(regime_on=True, atr=0.5, rsi=50.0, ema50=1.0, ema200=2.0),
        Context(regime_on=True, atr=0.5, rsi=50.0, ema50=1.0, ema200=2.0),
        Context(regime_on=True, atr=0.5, rsi=50.0, ema50=1.0, ema200=2.0),
        Context(regime_on=True, atr=0.5, rsi=50.0, ema50=1.0, ema200=2.0),
    ]
    # Keep the RSI sleeve out of the trade so the sells belong to the trend sleeve.
    result = _run(STRATEGIES["fin_a"], rows, contexts)
    trend_sells = _sells(result, "trend")
    assert [fill.reason for fill in trend_sells] == ["partial"]
    assert trend_sells[0].price == pytest.approx(200.0)
    assert trend_sells[0].base == pytest.approx(0.5 * 3000.0 / 100.0)
    assert result.sleeves[1].base == pytest.approx(0.5 * 3000.0 / 100.0)
    assert result.sleeves[1].stop_price == pytest.approx(90.0)

    rows[4] = (180, 180, 90, 170)
    stopped = _run(STRATEGIES["fin_a"], rows, contexts)
    reasons = [fill.reason for fill in _sells(stopped, "trend")]
    assert reasons == ["partial", "stop"]
    assert _sells(stopped, "trend")[1].price == pytest.approx(90.0)
    assert stopped.sleeves[1].blocked is True


def test_same_bar_stop_and_partial_follow_the_bar_direction():
    spec = SleeveSpec(
        name="core",
        kind="trend",
        notional=100.0,
        stop_pct=0.15,
        partial_gain=1.0,
        partial_fraction=0.5,
        rearm=True,
    )
    bullish = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 210, 80, 150),
    ]
    bearish = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 210, 80, 90),
    ]
    contexts = [Context(regime_on=True)] * 3
    bull = simulate_contexts([spec], _bars(bullish), contexts, 900)
    bear = simulate_contexts([spec], _bars(bearish), contexts, 900)
    assert [(fill.reason, fill.price) for fill in _sells(bull)] == [("stop", pytest.approx(85.0))]
    assert [(fill.reason, round(fill.price, 6)) for fill in _sells(bear)] == [("partial", 200.0), ("stop", 85.0)]
    assert _sells(bear)[0].base == pytest.approx(0.5)
    assert _sells(bear)[1].base == pytest.approx(0.5)


def test_fin_b_uses_the_atr_stop_the_150_percent_partial_and_the_rearm_rule():
    rows = [
        (100, 100, 100, 100),
        (100, 100, 100, 100),
        (100, 260, 92, 180),
        (180, 180, 180, 180),
        (180, 180, 90, 170),
        (170, 170, 170, 170),
        (170, 170, 170, 170),
        (170, 170, 170, 170),
        (170, 170, 170, 170),
    ]
    contexts = [
        Context(regime_on=True, atr=2.0),
        Context(regime_on=True, atr=2.0),
        Context(regime_on=True, atr=2.0),
        Context(regime_on=True, atr=2.0),
        Context(regime_on=True, atr=2.0),
        Context(regime_on=True, atr=2.0),
        Context(regime_on=False, atr=2.0),
        Context(regime_on=True, atr=2.0),
        Context(regime_on=True, atr=2.0),
    ]
    result = _run(STRATEGIES["fin_b"], rows, contexts)
    sells = _sells(result)
    assert sells[0].reason == "partial"
    assert sells[0].price == pytest.approx(250.0)
    assert sells[0].base == pytest.approx(0.67 * 150.0)
    assert sells[1].reason == "stop"
    # 92 is above the frozen ATR stop at 90, so the earlier bar did not stop us.
    # 90 does. A 15% stop would have sat at 85 and would not be the price used here.
    assert sells[1].price == pytest.approx(90.0)
    # The bar after the stop is still regime-on, so it cannot re-enter. The next
    # buy fills only after the regime has been off and then on again.
    assert [fill.bar_index for fill in _buys(result)] == [1, 8]


def test_fin_b_trail_and_cross_still_apply_and_rearm_blocks_a_stop_reentry():
    rows = _flat(100, 8)
    rows[2] = (100, 100, 99, 140)  # new peak close
    rows[3] = (140, 140, 100, 98)  # 98 <= 0.7 * 140, low stays above the ATR stop
    contexts = [Context(regime_on=True, atr=1.0)] * 8
    # stop = 100 - 5 = 95. Trail level after the 140 close is 98.
    result = _run(STRATEGIES["fin_b"], rows, contexts)
    assert _sells(result)[0].reason == "trail"
    assert result.sleeves[0].blocked is True
    assert len(_buys(result)) == 1


# --- S2326 ------------------------------------------------------------------


def test_s2326_enters_on_z_and_the_closed_15m_trend_for_5000_quote():
    rows = _flat(100, 4)
    entered = [
        Context(zscore=-2.5, regime_on=True),
        Context(zscore=0.0, regime_on=True),
        Context(zscore=0.0, regime_on=True),
        Context(zscore=0.0, regime_on=True),
    ]
    buy = _buys(_run(STRATEGIES["s2326"], rows, entered, step=300))[0]
    assert buy.notional == pytest.approx(5000.0)
    assert buy.price == pytest.approx(100.0)
    assert _buys(_run(STRATEGIES["s2326"], rows, [Context(zscore=-2.49, regime_on=True)] * 4, step=300)) == []
    assert _buys(_run(STRATEGIES["s2326"], rows, [Context(zscore=-3.0, regime_on=False)] * 4, step=300)) == []


def test_s2326_resting_limit_is_the_fill_times_1_03_and_rsi_does_not_exit():
    rows = _flat(100, 5)
    rows[2] = (100, 102.9, 99, 101)
    rows[3] = (101, 103, 100, 102)
    contexts = [Context(zscore=-3.0, regime_on=True, rsi=95.0)] + [Context(zscore=0.0, regime_on=True, rsi=95.0)] * 4
    sells = _sells(_run(STRATEGIES["s2326"], rows, contexts, step=300))
    assert len(sells) == 1
    assert sells[0].reason == "take_profit"
    assert sells[0].price == pytest.approx(103.0)
    assert sells[0].bar_index == 3
    assert sells[0].base == pytest.approx(50.0)


def test_s2326_stop_and_the_288_bar_time_stop():
    rows = _flat(100, 4)
    rows[2] = (100, 101, 90, 99)
    contexts = [Context(zscore=-3.0, regime_on=True)] + [Context(zscore=0.0, regime_on=True)] * 3
    sell = _sells(_run(STRATEGIES["s2326"], rows, contexts, step=300))[0]
    assert sell.reason == "stop"
    assert sell.price == pytest.approx(90.0)

    rows = _flat(100, 292)
    contexts = [Context(zscore=-3.0, regime_on=True, rsi=90.0)] + [
        Context(zscore=0.0, regime_on=True, rsi=90.0) for _ in range(291)
    ]
    result = _run(STRATEGIES["s2326"], rows, contexts, step=300)
    timed = _sells(result)[0]
    assert timed.reason == "time"
    assert timed.time - _buys(result)[0].time == 288 * 300
    assert all(fill.reason != "rsi" for fill in result.fills)


def test_s2326_wiring_uses_population_z_and_the_last_closed_15m_trend():
    hours_15m = [Bar(index * 900, 100.0 + index, 101.0 + index, 99.0 + index, 100.0 + index) for index in range(220)]
    last_15m_close = hours_15m[-1].open_time + 900
    fives = []
    price = 100.0
    for index in range(48):
        open_time = last_15m_close - (48 - index) * 300
        price += 0.2
        fives.append(Bar(open_time, price, price + 0.1, price - 0.1, price))
    cliff = price * 0.8
    fives.append(Bar(last_15m_close, cliff, cliff, cliff * 0.99, cliff))
    fives.append(Bar(last_15m_close + 300, cliff, cliff, cliff, cliff))
    contexts = build_contexts(fives, "5m", hours_15m, "15m", "ema_trend")
    result = simulate_contexts(STRATEGIES["s2326"].sleeves, fives, contexts, 300)
    buys = _buys(result)
    assert buys, "the cliff should print z <= -2.5 while the last closed 15m trend is still up"
    signal = buys[0].bar_index - 1
    assert contexts[signal].zscore <= -2.5
    assert contexts[signal].regime_on is True
    assert fives[signal].close_time(300) > last_15m_close
    assert buys[0].notional == pytest.approx(5000.0)


# --- configs and the paper profile ------------------------------------------


def test_each_shipped_config_matches_its_strategy_and_stays_on_paper():
    assert set(CONFIGS) == set(STRATEGIES)
    for key, filename in CONFIGS.items():
        payload = yaml.safe_load((ROOT / "bots" / "conf" / "controllers" / f"{filename}.yml").read_text())
        spec = STRATEGIES[key]
        assert payload["strategy"] == key
        assert float(payload["total_amount_quote"]) == spec.total_quote
        assert payload["controller_name"] == "aiot_research"
        assert payload["connector_name"] == "mexc_paper_trade"
        assert payload["candles_connector"] == "mexc"
        assert payload["trading_pair"] == "AIOT-USDT"
        assert payload["candles_trading_pair"] == "AIOT-USDT"
        assert payload["leverage"] == 1
        assert payload["position_mode"] == "HEDGE"
        assert payload["size_mode"] == "max"
        assert payload["stop_loss"] is None
        assert payload["take_profit"] is None
        assert payload["time_limit"] is None
        assert payload["trailing_stop"] is None


def test_paper_profile_includes_mexc_and_can_fund_every_sleeve():
    payload = yaml.safe_load((ROOT / "bots" / "credentials" / "master_account" / "conf_client.yml").read_text())
    assert "mexc" in payload["paper_trade"]["paper_trade_exchanges"]
    balance = payload["paper_trade"]["paper_trade_account_balance"]["USDT"]
    assert balance >= sum(spec.total_quote for spec in STRATEGIES.values())
    connectors = ROOT / "bots" / "credentials" / "master_account" / "connectors"
    assert list(connectors.glob("mexc*.yml")) == []


def test_state_round_trip_keeps_a_rearm_block():
    rows = _flat(100, 3)
    rows[1] = (100, 100, 80, 100)
    result = _run(STRATEGIES["f1"], rows, [Context(regime_on=True)] * 3)
    sleeve = result.sleeves[0]
    restored = simulate_contexts(STRATEGIES["f1"].sleeves, [], [], 900).sleeves[0]
    restored.import_state(sleeve.export_state())
    assert restored.blocked is True
    assert restored.phase == "flat"
    assert restored.base == 0.0
