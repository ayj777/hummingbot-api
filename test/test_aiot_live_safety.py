"""Live-safety checks for the AIOT controller.

A hedge-mode sell must close the long. Max mode sizes from the balance, keeps
the strategy weights, and splits at the exchange max. The exchange stop is the
hard stop: placed, resized, then cancelled. The trail is not sent to it.
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

from bots.controllers.directional_trading.aiot_engine import STRATEGIES, Intent
from bots.controllers.directional_trading.aiot_research import (
    AiotResearchConfig,
    AiotResearchController,
    confirm_aiot_leverage,
    max_entry_quote,
    position_action_for,
    protective_stop_plan,
    scaled_base,
    split_base,
)
from bots.scripts.v2_with_controllers import V2WithControllers, drawdown_quote_from_equity


class _Rule:
    min_order_size = Decimal("1")
    max_order_size = Decimal("6300")
    min_base_amount_increment = Decimal("1")


class _Log:
    def warning(self, message):
        return None

    def error(self, message):
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
    controller._stops = {}
    controller._stop_inflight = set()
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


def test_max_budget_keeps_sleeve_weights_and_the_configured_ceiling():
    balance = Decimal("5925")
    budget = max_entry_quote(Decimal("15000"), balance, 1)
    assert budget == balance * Decimal("0.95")
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
    assert max_entry_quote(Decimal("5000"), balance, 1) == Decimal("5000")
    assert max_entry_quote(Decimal("20"), balance, 1) == Decimal("20")
    assert max_entry_quote(Decimal("15000"), None, 1) is None
    assert max_entry_quote(Decimal("15000"), Decimal("0"), 1) is None


def test_scaled_base_uses_the_budget_fraction():
    base = scaled_base(12000, 15000, Decimal("5628.75"), Decimal("0.1"))
    assert base == Decimal("12000") * Decimal("5628.75") / Decimal("15000") / Decimal("0.1")


def test_orders_split_at_the_exchange_max_and_drop_dust():
    assert split_base(Decimal("10000"), Decimal("1"), Decimal("6300")) == [
        Decimal("6300"),
        Decimal("3700"),
    ]
    stepped = split_base(Decimal("10.4"), Decimal("1"), Decimal("6300"), Decimal("1"))
    assert stepped == [Decimal("10")]
    assert split_base(Decimal("0.4"), Decimal("1"), Decimal("6300")) == []
    assert split_base(Decimal("50"), Decimal("1"), Decimal("6300")) == [Decimal("50")]


def test_buy_orders_open_the_long_and_split():
    connector = SimpleNamespace(
        get_available_balance=lambda asset: Decimal("5925"),
        trading_rules={"AIOT-USDT": _Rule()},
    )
    controller = _controller(connector)
    intent = Intent("trend", "market_buy", 1.0, 15000.0, "entry", 1, None, 0.1)
    actions, keep = controller._buy_actions(intent)
    assert keep is False
    amounts = [item.executor_config.amount for item in actions]
    assert amounts[0] == Decimal("6300.00000000")
    assert all(amount <= Decimal("6300") for amount in amounts)
    assert sum(amounts) == Decimal("56287.00000000")
    assert all(item.executor_config.position_action is PositionAction.OPEN for item in actions)
    assert all(item.executor_config.side is TradeType.BUY for item in actions)
    assert actions[0].executor_config.level_id.endswith(":c0")


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


def test_stop_plan_holds_the_hard_stop_and_moves_only_when_size_changes():
    amount = "100.00000000"
    price = "0.08500000"
    assert protective_stop_plan(None, None, None, 100, 0.085) == "place"
    assert protective_stop_plan("s1", amount, price, 100, 0.085) == "hold"
    assert protective_stop_plan("s1", amount, price, 40, 0.085) == "place"
    assert protective_stop_plan("s1", amount, price, 0, 0.12) == "cancel"
    assert protective_stop_plan(None, None, None, 0, None) == "hold"


class _Stops:
    def __init__(self):
        self.calls = []
        self._position_ids = {("AIOT-USDT", PositionSide.LONG): 42}

    async def cancel_stop_orders(self, ids):
        self.calls.append(("cancel", list(ids)))

    async def place_position_stop_order(
        self, pair, position_id, amount, stop_loss_price=None, vol_type=None
    ):
        self.calls.append(("place", pair, position_id, amount, stop_loss_price, vol_type))
        return f"stop-{len(self.calls)}"


@pytest.mark.asyncio
async def test_exchange_stop_is_placed_resized_and_cancelled():
    book = _Stops()
    controller = _controller(book)
    await controller._run_stop(book, "core", "place", 100.0, 0.085)
    kind, pair, position_id, amount, price, vol_type = book.calls[0]
    assert kind == "place"
    assert pair == "AIOT-USDT"
    assert position_id == 42
    assert amount == Decimal("100.00000000")
    assert price == Decimal("0.08500000")
    assert vol_type == 1
    await controller._run_stop(book, "core", "place", 40.0, 0.085)
    assert book.calls[1][0] == "cancel"
    assert book.calls[1][1] == ["stop-1"]
    assert book.calls[2][3] == Decimal("40.00000000")
    assert book.calls[2][4] == Decimal("0.08500000")
    await controller._run_stop(book, "core", "cancel", 0.0, None)
    assert book.calls[3] == ("cancel", ["stop-3"])
    assert "core" not in controller._stops


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
