"""Live-safety checks for the AIOT controller.

A hedge-mode sell must close the long. Max mode keeps the position inside one
AIOT_USDT order. The exchange stop is placed before the old one is cancelled.
"""

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest
from hummingbot.core.data_type.common import (
    PositionAction,
    PositionMode,
    PositionSide,
    TradeType,
)
from hummingbot.strategy_v2.executors.order_executor.data_types import ExecutionStrategy
from hummingbot.strategy_v2.models.base import RunnableStatus

from bots.controllers.directional_trading.aiot_engine import STRATEGIES, Intent, Sleeve
from bots.controllers.directional_trading.aiot_research import (
    AiotResearchConfig,
    AiotResearchController,
    cap_bases,
    confirm_aiot_leverage,
    live_long_base,
    max_entry_quote,
    position_action_for,
    protective_stop_plan,
    scaled_base,
)
from bots.scripts.v2_with_controllers import V2WithControllers, drawdown_quote_from_equity

# Live AIOT_USDT rules: contractSize 10, maxVol 6300, so one order is 63000 base.
PRICE = Decimal("0.08223")
LOW_PRICE = Decimal("0.07934")
MAX_BASE = Decimal("6300") * Decimal("10")
ORDER_QUOTE = MAX_BASE * PRICE * Decimal("0.98")


class _Rule:
    min_order_size = Decimal("10")
    max_order_size = MAX_BASE
    min_base_amount_increment = Decimal("10")


class _Log:
    def warning(self, message):
        return None

    def error(self, message):
        return None

    def info(self, message):
        return None


def _controller(connector, strategy="f1", **overrides):
    controller = AiotResearchController.__new__(AiotResearchController)
    fields = {
        "id": "t",
        "strategy": strategy,
        "connector_name": "mexc_paper_trade",
        "trading_pair": "AIOT-USDT",
    }
    fields.update(overrides)
    controller.config = AiotResearchConfig(**fields)
    controller._spec = STRATEGIES[strategy]
    controller._warned = set()
    controller._leverage_ready = True
    controller._leverage_inflight = False
    controller._leverage_next = 0.0
    controller._stop = {}
    controller._stop_inflight = False
    controller._live_long = None
    controller._exit_reserved = Decimal("0")
    controller._drawdown_cap = None
    controller._entries_halted = False
    controller._sleeves = []
    controller.market_data_provider = SimpleNamespace(
        connectors={"mexc_paper_trade": connector},
        time=lambda: 100.0,
    )
    controller._now = lambda: 100.0
    controller.logger = lambda: _Log()
    return controller


def test_sell_closes_the_long_and_a_buy_stays_open():
    assert position_action_for(TradeType.SELL) is PositionAction.CLOSE
    assert position_action_for(TradeType.BUY) is PositionAction.OPEN
    controller = _controller(SimpleNamespace())
    sell = Intent("core", "market_sell", 4.0, 40.0, "stop", 1)
    actions, keep = controller._intent_actions(sell)
    assert keep is False
    assert actions[0].executor_config.side is TradeType.SELL
    assert actions[0].executor_config.position_action is PositionAction.CLOSE
    limit = controller._create(
        TradeType.SELL, 1.0, 0.2, "aiot:core:limit", ExecutionStrategy.LIMIT
    )
    assert limit.executor_config.position_action is PositionAction.CLOSE


def test_max_budget_stays_inside_one_aiot_order_and_keeps_weights():
    balance = Decimal("5925")
    room = balance * Decimal("0.95")
    assert room / PRICE / Decimal("10") > Decimal("6300")
    budget = max_entry_quote(Decimal("15000"), balance, 1, ORDER_QUOTE)
    assert budget == ORDER_QUOTE
    assert budget / PRICE == MAX_BASE * Decimal("0.98")
    assert budget / PRICE / Decimal("10") <= Decimal("6300") * Decimal("0.98")
    fin = STRATEGIES["fin_a"]
    quotes = [
        Decimal(str(sleeve.notional)) * budget / Decimal(str(fin.total_quote))
        for sleeve in fin.sleeves
    ]
    assert sum(quotes) == budget
    assert quotes[0] == budget * Decimal("12000") / Decimal("15000")
    assert quotes[1] == budget * Decimal("3000") / Decimal("15000")
    ladder = STRATEGIES["ladder"].sleeves[0]
    scaled = [
        Decimal(str(rung)) * budget / Decimal("15000")
        for rung in ladder.rung_notionals
    ]
    assert scaled[0] == scaled[1] == scaled[2]
    assert sum(scaled) == budget
    low_order = MAX_BASE * LOW_PRICE * Decimal("0.98")
    assert max_entry_quote(Decimal("5000"), balance, 1, low_order) == low_order
    assert low_order / LOW_PRICE / Decimal("10") <= Decimal("6300") * Decimal("0.98")
    assert max_entry_quote(Decimal("20"), balance, 1, ORDER_QUOTE) == Decimal("20")
    assert max_entry_quote(Decimal("15000"), None, 1, ORDER_QUOTE) is None
    assert max_entry_quote(Decimal("15000"), Decimal("0"), 1) is None


def test_scaled_base_uses_the_budget_fraction():
    base = scaled_base(12000, 15000, Decimal("5628.75"), Decimal("0.1"))
    assert base == Decimal("12000") * Decimal("5628.75") / Decimal("15000") / Decimal("0.1")


def test_buy_is_one_order_inside_the_aiot_contract_cap():
    connector = SimpleNamespace(
        get_available_balance=lambda asset: Decimal("5925"),
        trading_rules={"AIOT-USDT": _Rule()},
    )
    controller = _controller(connector)
    intent = Intent("trend", "market_buy", 1.0, 15000.0, "entry", 1, None, float(PRICE))
    actions, keep = controller._buy_actions(intent)
    assert keep is False
    assert len(actions) == 1
    amount = actions[0].executor_config.amount
    contracts = amount / Decimal("10")
    assert contracts <= Decimal("6300") * Decimal("0.98")
    assert amount == Decimal("61740.00000000")
    assert actions[0].executor_config.position_action is PositionAction.OPEN
    assert actions[0].executor_config.side is TradeType.BUY
    assert ":c" not in actions[0].executor_config.level_id


def test_an_unconfirmed_leverage_holds_the_entry():
    connector = SimpleNamespace(
        confirm_leverage_and_margin=lambda *args: False,
        get_available_balance=lambda asset: Decimal("5925"),
        trading_rules={"AIOT-USDT": _Rule()},
    )
    controller = _controller(connector)
    controller._leverage_ready = False
    controller._leverage_inflight = True
    intent = Intent("trend", "market_buy", 1.0, 15000.0, "entry", 1, None, 0.1)
    actions, keep = controller._buy_actions(intent)
    assert actions == []
    assert keep is True


def test_stop_plan_holds_the_hard_stop_and_an_unknown_position():
    amount = "100.00000000"
    price = "0.08500000"
    assert protective_stop_plan(None, None, None, 100, 0.085) == "place"
    assert protective_stop_plan("s1", amount, price, 100, 0.085) == "hold"
    assert protective_stop_plan("s1", amount, price, 40, 0.085) == "place"
    assert protective_stop_plan("s1", amount, price, 0, 0.12) == "cancel"
    assert protective_stop_plan(None, None, None, 0, None) == "hold"
    assert protective_stop_plan("s1", amount, price, None, 0.085, live_known=False) == "hold"
    assert protective_stop_plan("s1", amount, price, 100, None) == "hold"


class _Stops:
    def __init__(self, reject_after=None):
        self.calls = []
        self._position_ids = {("AIOT-USDT", PositionSide.LONG): 42}
        self._reject_after = reject_after
        self._placed = 0

    async def cancel_stop_orders(self, ids):
        self.calls.append(("cancel", list(ids)))

    async def place_position_stop_order(
        self,
        pair,
        position_id,
        amount,
        stop_loss_price=None,
        vol_type=None,
        stop_loss_reverse=None,
    ):
        self.calls.append(
            ("place", pair, position_id, amount, stop_loss_price, vol_type, stop_loss_reverse)
        )
        self._placed += 1
        if self._reject_after is not None and self._placed > self._reject_after:
            raise RuntimeError("maxVol")
        return f"stop-{self._placed}"


@pytest.mark.asyncio
async def test_exchange_stop_is_placed_before_the_old_one_is_cancelled():
    book = _Stops()
    controller = _controller(book)
    await controller._run_stop(book, "place", 100.0, 0.085)
    kind, pair, position_id, amount, price, vol_type, reverse = book.calls[0]
    assert kind == "place"
    assert pair == "AIOT-USDT"
    assert position_id == 42
    assert amount == Decimal("100.00000000")
    assert price == Decimal("0.08500000")
    assert vol_type == 1
    assert reverse == 2
    await controller._run_stop(book, "place", 40.0, 0.085)
    assert book.calls[1][0] == "place"
    assert book.calls[1][3] == Decimal("40.00000000")
    assert book.calls[2] == ("cancel", ["stop-1"])
    await controller._run_stop(book, "cancel", 0.0, None)
    assert book.calls[3] == ("cancel", ["stop-2"])
    assert controller._stop == {}


@pytest.mark.asyncio
async def test_a_rejected_stop_leaves_the_previous_stop_in_place():
    book = _Stops(reject_after=1)
    controller = _controller(book)
    await controller._run_stop(book, "place", 100.0, 0.085)
    await controller._run_stop(book, "place", Decimal("68451"), 0.085)
    assert [call[0] for call in book.calls] == ["place", "place"]
    assert controller._stop["id"] == "stop-1"
    assert controller._stop["amount"] == "100.00000000"


@pytest.mark.asyncio
async def test_leverage_confirmation_asks_for_isolated_long():
    seen = {}

    class Connector:
        async def confirm_leverage_and_margin(self, pair, leverage, margin, side):
            seen["args"] = (pair, leverage, margin, side)
            return True

    assert await confirm_aiot_leverage(Connector(), "AIOT-USDT", 1) is True
    assert seen["args"] == ("AIOT-USDT", 1, "ISOLATED", PositionSide.LONG)

    class Rejected:
        async def confirm_leverage_and_margin(self, pair, leverage, margin, side):
            return False

    assert await confirm_aiot_leverage(Rejected(), "AIOT-USDT", 1) is False


@pytest.mark.asyncio
async def test_startup_uses_the_read_back_and_leaves_other_bots_alone():
    seen = {}

    class AiotConnector:
        def confirm_leverage_and_margin(self, pair, leverage, margin, side):
            seen["aiot"] = (pair, leverage, margin, side)

            async def _done():
                return True

            return _done()

        def set_leverage(self, **kwargs):
            seen["stock"] = kwargs

    class Other:
        def set_leverage(self, **kwargs):
            seen["other"] = kwargs

    class Missing:
        def set_leverage(self, **kwargs):
            seen["stock"] = kwargs

    script = V2WithControllers.__new__(V2WithControllers)
    script.logger = lambda: _Log()
    script.connectors = {"mexc_perpetual": AiotConnector(), "other": Other()}
    script._apply_leverage(
        "mexc_perpetual",
        {"controller_name": "aiot_research", "trading_pair": "AIOT-USDT", "leverage": 1},
    )
    await asyncio.sleep(0)
    assert seen["aiot"] == ("AIOT-USDT", 1, "ISOLATED", PositionSide.LONG)
    assert "stock" not in seen
    script.connectors["mexc_perpetual"] = Missing()
    script._apply_leverage(
        "mexc_perpetual",
        {"controller_name": "aiot_research", "trading_pair": "AIOT-USDT", "leverage": 1},
    )
    assert "stock" not in seen
    script._apply_leverage(
        "other",
        {"controller_name": "pmm", "trading_pair": "AIOT-USDT", "leverage": 2},
    )
    assert seen["other"] == {"leverage": 2, "trading_pair": "AIOT-USDT"}


def test_drawdown_default_is_twenty_percent_of_quote_balance():
    assert drawdown_quote_from_equity(Decimal("5925")) == Decimal("1185.00")
    assert drawdown_quote_from_equity(None) is None
    assert drawdown_quote_from_equity(Decimal("0")) is None

    class Config:
        def model_dump(self):
            return {
                "controller_name": "aiot_research",
                "connector_name": "mexc_perpetual",
                "trading_pair": "AIOT-USDT",
            }

    class Controller:
        config = Config()

    class Connector:
        def get_available_balance(self, asset):
            assert asset == "USDT"
            return Decimal("5925")

    script = V2WithControllers.__new__(V2WithControllers)
    script.logger = lambda: _Log()
    script.controllers = {"aiot": Controller()}
    script.connectors = {"mexc_perpetual": Connector()}
    script.config = SimpleNamespace(max_controller_drawdown_quote=None)
    script._apply_aiot_drawdown()
    assert script.config.max_controller_drawdown_quote == 1185.0
    script.config.max_controller_drawdown_quote = 25.0
    script._apply_aiot_drawdown()
    assert script.config.max_controller_drawdown_quote == 25.0


def test_max_mode_accepts_a_small_ceiling_and_rejects_other_leverage():
    config = AiotResearchConfig(
        id="trial",
        strategy="s2326",
        total_amount_quote=Decimal("20"),
        size_mode="max",
    )
    assert config.total_amount_quote == Decimal("20")
    assert config.position_mode is PositionMode.HEDGE
    assert config.model_dump()["position_mode"] is PositionMode.HEDGE
    with pytest.raises(ValueError):
        AiotResearchConfig(
            id="trial",
            strategy="s2326",
            total_amount_quote=Decimal("20"),
            size_mode="fixed",
        )
    with pytest.raises(ValueError):
        AiotResearchConfig(id="trial", strategy="f1", leverage=2)
    with pytest.raises(ValueError):
        AiotResearchConfig(id="trial", strategy="f1", position_mode="ONEWAY")


def test_a_partial_sell_does_not_flatten_the_sleeve():
    sleeve = Sleeve(STRATEGIES["f1"].sleeves[0])
    sleeve.base = 100.0
    sleeve.phase = "open"
    sleeve.stop_price = 0.07
    sleeve.apply_partial_sell(0.08, 40.0, 1, 1)
    assert sleeve.base == 60.0
    assert sleeve.phase == "open"
    assert sleeve.stop_price == 0.07
    sleeve.apply_partial_sell(0.08, 60.0, 1, 1)
    assert sleeve.phase == "flat"
    assert sleeve.base == 0.0


def test_two_sleeves_share_one_live_long():
    connector = SimpleNamespace(trading_rules={"AIOT-USDT": _Rule()})
    controller = _controller(connector, strategy="fin_a")
    controller._live_long = Decimal("60000")
    first, _kept = controller._intent_actions(
        Intent("rsi", "market_sell", 48000.0, 1.0, "rsi", 1)
    )
    second, _kept = controller._intent_actions(
        Intent("trend", "market_sell", 20000.0, 1.0, "cross", 1)
    )
    assert first[0].executor_config.amount == Decimal("48000.00000000")
    assert second[0].executor_config.amount == Decimal("12000.00000000")
    assert first[0].executor_config.position_action is PositionAction.CLOSE
    alone = _controller(connector)
    alone._live_long = Decimal("68451")
    oversized, _kept = alone._intent_actions(
        Intent("core", "market_sell", 68451.0, 1.0, "stop", 1)
    )
    assert oversized[0].executor_config.amount == Decimal("61740.00000000")


def test_a_flat_live_long_clears_a_phantom_sleeve():
    connector = SimpleNamespace(account_positions={}, ready=True)
    controller = _controller(connector)
    sleeve = Sleeve(STRATEGIES["f1"].sleeves[0])
    sleeve.base = 1000.0
    sleeve.phase = "open"
    sleeve.stop_price = 0.07
    controller._sleeves = [sleeve]
    controller._stop = {"id": "dead", "amount": "1000", "price": "0.07000000"}
    controller._apply_live_position()
    assert sleeve.phase == "flat"
    assert sleeve.base == 0.0
    assert controller._live_long == Decimal("0")
    assert protective_stop_plan("dead", "1000", "0.07", Decimal("0"), 0.07) == "cancel"
    unread = SimpleNamespace(account_positions={}, ready=False)
    assert live_long_base(unread, "AIOT-USDT") is None


def test_an_open_live_long_is_adopted_when_the_sleeve_is_flat():
    position = SimpleNamespace(
        trading_pair="AIOT-USDT",
        position_side=PositionSide.LONG,
        amount=Decimal("1000"),
    )
    connector = SimpleNamespace(account_positions={"k": position}, ready=True)
    controller = _controller(connector)
    sleeve = Sleeve(STRATEGIES["f1"].sleeves[0])
    controller._sleeves = [sleeve]
    controller._apply_live_position()
    assert sleeve.phase == "open"
    assert sleeve.base == 1000.0
    rsi = Decimal("48000")
    trend = Decimal("20000")
    capped = cap_bases([rsi, trend], Decimal("50000"))
    assert sum(capped) == Decimal("50000")
    assert capped[0] < rsi
    assert capped[1] < trend


def test_a_missing_balance_holds_the_entry():
    connector = SimpleNamespace(
        get_available_balance=lambda asset: Decimal("0"),
        trading_rules={"AIOT-USDT": _Rule()},
        ready=True,
    )
    controller = _controller(connector)
    intent = Intent("trend", "market_buy", 1.0, 15000.0, "entry", 1, None, float(PRICE))
    actions, keep = controller._buy_actions(intent)
    assert actions == []
    assert keep is True


def test_drawdown_closes_the_long_and_stops_entries():
    connector = SimpleNamespace(
        trading_rules={"AIOT-USDT": _Rule()},
        get_available_balance=lambda asset: Decimal("5925"),
    )
    controller = _controller(connector)
    controller._live_long = Decimal("1000")
    controller._stop = {"id": "s1", "amount": "1000.00000000", "price": "0.07000000"}
    action = controller.protective_close_action()
    assert action.executor_config.side is TradeType.SELL
    assert action.executor_config.position_action is PositionAction.CLOSE
    assert action.executor_config.amount == Decimal("1000.00000000")
    assert controller._entries_halted is True
    assert controller._stop["id"] == "s1"
    held = protective_stop_plan(
        "s1", "1000.00000000", "0.07000000", Decimal("1000"), 0.07
    )
    assert held == "hold"
    intent = Intent("trend", "market_buy", 1.0, 15000.0, "entry", 1, None, float(PRICE))
    actions, keep = controller._buy_actions(intent)
    assert actions == []
    assert keep is True


def test_drawdown_check_submits_the_close_before_stopping():
    connector = SimpleNamespace(trading_rules={"AIOT-USDT": _Rule()})
    controller = _controller(connector)
    controller._live_long = Decimal("1000")
    close = controller.protective_close_action()
    stopped = []

    class Running:
        status = RunnableStatus.RUNNING

        def protective_close_action(self):
            return close

        def stop(self):
            stopped.append(True)

    seen = []
    script = V2WithControllers.__new__(V2WithControllers)
    script.logger = lambda: _Log()
    script.config = SimpleNamespace(max_controller_drawdown_quote=100.0)
    script.max_pnl_by_controller = {"t": Decimal("0")}
    script.controllers = {"t": Running()}
    script.drawdown_exited_controllers = []
    script.get_performance_report = lambda cid: SimpleNamespace(global_pnl_quote=Decimal("-150"))
    script.executor_orchestrator = SimpleNamespace(
        execute_actions=lambda actions: seen.extend(actions)
    )
    script.filter_executors = lambda **kwargs: []
    script.get_executors_by_controller = lambda cid: []
    script.check_max_controller_drawdown()
    assert seen[0].executor_config.position_action is PositionAction.CLOSE
    assert stopped == [True]
    script.get_performance_report = lambda cid: None
    script.check_max_controller_drawdown()


def test_a_ready_account_with_no_balance_refuses_entries():
    connector = SimpleNamespace(ready=True, get_available_balance=lambda asset: None)
    controller = _controller(connector)
    script = V2WithControllers.__new__(V2WithControllers)
    script.logger = lambda: _Log()
    script.controllers = {"t": controller}
    script.connectors = {"mexc_paper_trade": connector}
    script.config = SimpleNamespace(max_controller_drawdown_quote=None)
    script._apply_aiot_drawdown()
    assert controller._entries_halted is True
    assert script.config.max_controller_drawdown_quote is None


@pytest.mark.asyncio
async def test_startup_adopts_a_live_stop_and_drops_a_dead_one():
    class Listed:
        async def list_open_stop_orders(self, pair):
            return [{"id": "live-stop"}]

    controller = _controller(Listed())
    controller._stop = {"id": "dead", "amount": "10", "price": "0.07000000"}
    controller._live_long = Decimal("100")
    await controller._sync_stops(Listed())
    assert controller._stop["id"] == "live-stop"

    class Empty:
        async def list_open_stop_orders(self, pair):
            return []

    controller._stop = {"id": "dead", "amount": "10", "price": "0.07000000"}
    controller._live_long = Decimal("100")
    await controller._sync_stops(Empty())
    assert controller._stop == {}
