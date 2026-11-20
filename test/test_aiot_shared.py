"""Two AIOT bots can share one MEXC hedge long, and each keeps its own share."""

import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hummingbot.core.data_type.common import PositionAction, PositionMode, PositionSide, TradeType

from bots.controllers.directional_trading.aiot_engine import STRATEGIES, Intent
from bots.controllers.directional_trading.aiot_shared import (
    AiotSharedConfig,
    AiotSharedController,
    parse_fill_row,
    reconcile_share,
)
from bots.scripts.v2_with_controllers import V2WithControllers

ROOT = Path(__file__).resolve().parent.parent
PRICE = Decimal("0.08223")
MAX_BASE = Decimal("6300") * Decimal("10")


class _Rule:
    min_order_size = Decimal("10")
    max_order_size = MAX_BASE
    min_base_amount_increment = Decimal("10")


class _Log:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, message):
        self.warnings.append(message)

    def error(self, message):
        self.errors.append(message)

    def info(self, message):
        return None


class _Book:
    def __init__(self, stops=None):
        self.calls = []
        self._position_ids = {("AIOT-USDT", PositionSide.LONG): 42}
        self.open_stops = list(stops or [])
        self._n = 0

    async def cancel_stop_orders(self, ids):
        self.calls.append(("cancel", list(ids)))

    async def place_position_stop_order(
        self, pair, position_id, amount, stop_loss_price=None, vol_type=None, stop_loss_reverse=None
    ):
        self.calls.append(("place", pair, position_id, amount, stop_loss_price, vol_type, stop_loss_reverse))
        self._n += 1
        new_id = f"stop-new-{self._n}"
        self.open_stops.append({"id": new_id})
        return new_id

    async def list_open_stop_orders(self, pair):
        return list(self.open_stops)


def _long(amount):
    return SimpleNamespace(
        trading_pair="AIOT-USDT",
        position_side=PositionSide.LONG,
        amount=Decimal(str(amount)),
        entry_price=Decimal("0.10"),
    )


def _bot(strategy="fin_b", quote="2500", connector=None, log=None):
    from bots.controllers.directional_trading.aiot_engine import Sleeve, f3_l20_strategy

    if connector is None:
        connector = SimpleNamespace()
    if log is None:
        log = _Log()
    bot = AiotSharedController.__new__(AiotSharedController)
    bot.config = AiotSharedConfig(
        id="bot-a",
        strategy=strategy,
        connector_name="mexc_perpetual",
        total_amount_quote=Decimal(quote),
        size_mode="max",
    )
    bot._spec = f3_l20_strategy() if strategy == "f3l20" else STRATEGIES[strategy]
    bot._sleeves = [Sleeve(spec) for spec in bot._spec.sleeves]
    bot._applied_base = {}
    bot._last_bar = None
    bot._limit_live = set()
    bot._stop = {}
    bot._stop_inflight = False
    bot._stop_sync_inflight = False
    bot._stop_sync_next = 0.0
    bot._stop_missing = 0
    bot._stop_blocked = None
    bot._live_long = None
    bot._flat_streak = 0
    bot._positions_seen = False
    bot._fetch_inflight = False
    bot._fetch_next = 0.0
    bot._exit_reserved = Decimal("0")
    bot._drawdown_cap = None
    bot._entries_halted = False
    bot._atr_fallback = False
    bot._warned = set()
    bot._leverage_ready = True
    bot._leverage_inflight = False
    bot._leverage_next = 0.0
    bot._own_orders = {}
    bot._share_halt = None
    bot._share_checked = True
    bot._history_inflight = False
    bot.executors_info = []
    bot._state_path = Path("data") / "aiot_shared_bot-a.json"
    bot.market_data_provider = SimpleNamespace(
        connectors={"mexc_perpetual": connector},
        time=lambda: 100.0,
    )
    bot._now = lambda: 100.0
    bot._log = log
    bot.logger = lambda: log
    return bot


def _executor(executor_id, level, base, price, client_id, controller_id="bot-a", active=False):
    return SimpleNamespace(
        id=executor_id,
        controller_id=controller_id,
        config=SimpleNamespace(level_id=level, controller_id=controller_id),
        custom_info={
            "executed_amount_base": base,
            "average_executed_price": price,
            "order_id": client_id,
        },
        is_done=not active,
        is_active=active,
    )


def test_shared_configs_are_live_slices_and_leave_the_paper_configs_alone():
    fin = yaml.safe_load((ROOT / "bots/conf/controllers/aiot_fin_b_shared.yml").read_text())
    f3 = yaml.safe_load((ROOT / "bots/conf/controllers/aiot_f3l20_shared.yml").read_text())
    assert fin["strategy"] == "fin_b"
    assert fin["total_amount_quote"] == 2500
    assert f3["strategy"] == "f3l20"
    assert f3["total_amount_quote"] == 1500
    for payload in (fin, f3):
        assert payload["controller_name"] == "aiot_shared"
        assert payload["connector_name"] == "mexc_perpetual"
        assert payload["trading_pair"] == "AIOT-USDT"
        assert payload["leverage"] == 1
        assert payload["position_mode"] == "HEDGE"
        assert payload["size_mode"] == "max"
        assert payload["stop_loss"] is None
    paper = yaml.safe_load((ROOT / "bots/conf/controllers/aiot_fin_b.yml").read_text())
    assert paper["controller_name"] == "aiot_research"
    assert paper["connector_name"] == "mexc_paper_trade"
    assert paper["total_amount_quote"] == 15000


def test_the_shared_module_resolves_its_own_classes():
    from unittest.mock import MagicMock

    from utils.file_system import fs_util

    cls = fs_util.load_controller_config_class("directional_trading", "aiot_shared")
    # The loader reloads the module, so the class object is not the one imported above.
    assert cls.__name__ == "AiotSharedConfig"
    config = cls(id="t", strategy="f3l20", total_amount_quote=Decimal("1500"))
    controller_cls = config.get_controller_class()
    assert controller_cls.__name__ == "AiotSharedController"
    controller = controller_cls(config, MagicMock(), MagicMock())
    assert controller._spec.key == "f3l20"
    assert controller._state_path.name == "aiot_shared_t.json"


def test_shared_config_stays_at_1x_hedge():
    with pytest.raises(ValueError):
        AiotSharedConfig(id="t", strategy="f3l20", leverage=2)
    with pytest.raises(ValueError):
        AiotSharedConfig(id="t", strategy="fin_b", position_mode="ONEWAY")
    with pytest.raises(ValueError):
        AiotSharedConfig(id="t", strategy="nope")


def test_parse_fill_row_converts_mexc_contracts_and_ignores_other_ids():
    assert parse_fill_row({"externalOid": "cid", "dealVol": "10"}, Decimal("10")) == (
        "cid",
        Decimal("100"),
    )
    verdict, net, reason = reconcile_share(
        {"cid-a": {"side": "buy", "base": "100"}},
        [
            {"client_order_id": "cid-a", "base": "100"},
            {"client_order_id": "cid-b", "base": "500"},
        ],
    )
    assert verdict == "ok"
    assert net == Decimal("100")
    assert reason == ""
    verdict, _net, reason = reconcile_share(
        {"cid-a": {"side": "buy", "base": "100"}},
        [{"client_order_id": "cid-a", "base": "40"}],
    )
    assert verdict == "halt"
    assert "halted" in reason


def test_one_bot_exits_its_share_and_the_other_keeps_its_stop():
    book = _Book([{"id": "stop-a"}, {"id": "stop-b"}])
    log = _Log()
    bot_a = _bot(connector=book, log=log)
    bot_b = _bot(connector=book, log=_Log())
    bot_b.config = AiotSharedConfig(
        id="bot-b", strategy="fin_b", connector_name="mexc_perpetual", total_amount_quote=Decimal("2500")
    )
    for bot, base, stop_id in ((bot_a, 100, "stop-a"), (bot_b, 80, "stop-b")):
        sleeve = bot._sleeves[0]
        sleeve.base = float(base)
        sleeve.phase = "open"
        sleeve.stop_price = 0.07
        bot._live_long = Decimal("180")
        bot._stop = {"id": stop_id, "amount": f"{base}.00000000", "price": "0.07000000"}
    actions, keep = bot_a._intent_actions(Intent("core", "market_sell", 100.0, 100.0, "cross", 1))
    assert keep is False
    assert actions[0].executor_config.amount == Decimal("100.00000000")
    assert actions[0].executor_config.position_action is PositionAction.CLOSE
    assert actions[0].executor_config.side is TradeType.SELL
    assert actions[0].executor_config.level_id.startswith("shared:")
    bot_a._sleeves[0].apply_partial_sell(0.08, 100.0, 1, 1)
    assert bot_a._sleeves[0].phase == "flat"

    async def _cancel():
        await bot_a._run_stop(book, "cancel", 0, None)

    asyncio.run(_cancel())
    assert ("cancel", ["stop-a"]) in book.calls
    assert all(call[1] != ["stop-b"] for call in book.calls if call[0] == "cancel")
    assert bot_b._stop["id"] == "stop-b"
    assert bot_b._sleeves[0].base == 80.0
    assert bot_b._sleeves[0].phase == "open"


def test_a_partial_exit_replaces_only_that_bots_stop():
    book = _Book([{"id": "stop-a"}, {"id": "stop-b"}])
    bot = _bot(connector=book)
    sleeve = bot._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    sleeve.stop_price = 0.07
    sleeve.apply_partial_sell(0.09, 40.0, 1, 1)
    bot._live_long = Decimal("140")
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    assert sleeve.base == 60.0

    async def _replace():
        await bot._run_stop(book, "place", bot._stop_amount(), sleeve.stop_price)

    asyncio.run(_replace())
    kinds = [call[0] for call in book.calls]
    assert kinds == ["cancel", "place"]
    assert book.calls[0] == ("cancel", ["stop-a"])
    _kind, pair, position_id, amount, price, vol_type, reverse = book.calls[1]
    assert pair == "AIOT-USDT"
    assert position_id == 42
    assert amount == Decimal("60.00000000")
    assert price == Decimal("0.07000000")
    assert vol_type == 1
    assert reverse == 2
    assert bot._stop["id"] == "stop-new-1"
    assert all(call[1] != ["stop-b"] for call in book.calls if call[0] == "cancel")


def test_restart_restores_this_bots_share_and_does_not_take_the_other_long(tmp_path):
    saved = _bot()
    saved._state_path = tmp_path / "state.json"
    sleeve = saved._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    sleeve.first_fill_price = 0.1
    sleeve.stop_price = 0.07
    saved._own_orders = {
        "cid-a": {
            "side": "buy",
            "base": "100.00000000",
            "price": "0.10000000",
            "level": "shared:core:buy",
        }
    }
    saved._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    saved._save_state()

    connector = SimpleNamespace(
        account_positions={"k": _long(600)},
        fill_history=lambda: [
            {"client_order_id": "cid-a", "base": "100"},
            {"client_order_id": "cid-other", "base": "500"},
        ],
    )
    restored = _bot(connector=connector)
    restored._state_path = saved._state_path
    restored._share_checked = False
    restored._load_state()
    assert restored.confirm_share() is True
    assert restored._share_halt is None
    assert restored._sleeves[0].base == 100.0
    assert restored._sleeves[0].first_fill_price == 0.1
    assert restored._stop["id"] == "stop-a"
    restored._apply_live_position()
    assert restored._live_long == Decimal("600")
    assert restored._sleeves[0].base == 100.0
    assert restored._sleeves[0].phase == "open"


def test_an_unreconcilable_share_halts_entries_and_exits():
    connector = SimpleNamespace(fill_history=lambda: [{"client_order_id": "cid-a", "base": "40"}])
    bot = _bot(connector=connector)
    bot._share_checked = False
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._own_orders = {"cid-a": {"side": "buy", "base": "100", "price": "0.1", "level": "shared:core:buy"}}
    assert bot.confirm_share() is False
    assert "halted" in bot._share_halt
    sell, _keep = bot._intent_actions(Intent("core", "market_sell", 100.0, 1.0, "cross", 1))
    buy, _keep = bot._buy_actions(Intent("core", "market_buy", 1.0, 15000.0, "entry", 1, None, 0.1))
    assert sell == []
    assert buy == []

    missing = _bot(connector=SimpleNamespace(fill_history=lambda: []))
    missing._share_checked = False
    missing._sleeves[0].base = 10.0
    missing._own_orders = {"cid-a": {"side": "buy", "base": "10", "price": "0.1", "level": "shared:core:buy"}}
    assert missing.confirm_share() is False
    assert "no fill" in missing._share_halt
    gone, _keep = missing._intent_actions(Intent("core", "market_sell", 10.0, 1.0, "cross", 1))
    assert gone == []


def test_an_existing_long_is_not_adopted_and_another_stop_is_not_taken():
    connector = SimpleNamespace(account_positions={"k": _long(1000)})

    async def _list(_pair):
        return [{"id": "other-bot"}]

    connector.list_open_stop_orders = _list
    bot = _bot(connector=connector)
    bot._apply_live_position()
    assert bot._sleeves[0].phase == "flat"
    assert bot._sleeves[0].base == 0.0
    assert bot._live_long == Decimal("1000")
    assert bot._stop == {}

    async def _sync():
        await bot._sync_stops(connector)

    asyncio.run(_sync())
    assert bot._stop == {}
    assert bot._share_halt is None


def test_an_exit_is_capped_at_the_exchange_long():
    log = _Log()
    bot = _bot(log=log)
    sleeve = bot._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    bot._live_long = Decimal("40")
    actions, keep = bot._intent_actions(Intent("core", "market_sell", 100.0, 1.0, "cross", 1))
    assert keep is False
    assert actions[0].executor_config.amount == Decimal("40.00000000")
    assert actions[0].executor_config.position_action is PositionAction.CLOSE
    assert any("capped at the exchange long 40" in message for message in log.warnings)


def test_reconcile_ignores_another_bots_orders_and_the_research_prefix():
    bot = _bot()
    bot.executors_info = [
        _executor("other", "shared:core:buy", 500, 0.1, "cid-b", controller_id="bot-b"),
        _executor("old", "aiot:core:buy", 700, 0.1, "cid-old", controller_id="bot-a"),
        _executor("mine", "shared:core:buy", 100, 0.1, "cid-a", controller_id="bot-a"),
    ]
    bot._reconcile()
    assert bot._sleeves[0].base == pytest.approx(100.0)
    assert set(bot._own_orders) == {"cid-a"}
    assert bot._own_orders["cid-a"]["side"] == "buy"


def test_max_size_uses_this_bots_ceiling_and_the_available_balance():
    cash = {"value": Decimal("100000")}

    def balance(_asset):
        return cash["value"]

    connector = SimpleNamespace(
        get_available_balance=balance,
        trading_rules={"AIOT-USDT": _Rule()},
    )
    fin = _bot(strategy="fin_b", quote="2500", connector=connector)
    actions, keep = fin._buy_actions(Intent("core", "market_buy", 1.0, 15000.0, "entry", 1, None, float(PRICE)))
    assert keep is False
    quote = actions[0].executor_config.amount * PRICE
    assert quote <= Decimal("2500")
    assert quote > Decimal("2490")
    f3 = _bot(strategy="f3l20", quote="1500", connector=connector)
    first, _keep = f3._buy_actions(Intent("core", "market_buy", 1.0, 1500.0, "entry", 1, None, float(PRICE)))
    cash["value"] = Decimal("1000")
    second, _keep = f3._buy_actions(Intent("core", "market_buy", 1.0, 1500.0, "add", 2, -1, float(PRICE)))
    second_quote = second[0].executor_config.amount * PRICE
    assert first[0].executor_config.amount * PRICE > Decimal("1490")
    assert Decimal("900") < second_quote < Decimal("950")
    assert second[0].executor_config.position_action is PositionAction.OPEN


def test_f3_l20_leg_two_updates_the_average_stop_and_replaces_only_that_stop():
    book = _Book([{"id": "s1"}])
    bot = _bot(strategy="f3l20", quote="1500", connector=book)
    sleeve = bot._sleeves[0]
    sleeve.apply_buy_fill(100.0, 15.0, 1500.0, 1, "entry", 0)
    assert sleeve.stop_price == pytest.approx(90.0)
    bot._stop = {"id": "s1", "amount": "15.00000000", "price": "90.00000000"}
    sleeve.apply_buy_fill(140.0, 15.0, 2100.0, 2, "add", 1)
    assert sleeve.added is True
    assert sleeve.base == pytest.approx(30.0)
    assert sleeve.stop_price == pytest.approx(108.0)
    assert sleeve.first_fill_time == 1

    async def _replace():
        await bot._run_stop(book, "place", bot._stop_amount(), sleeve.stop_price)

    asyncio.run(_replace())
    assert book.calls[0] == ("cancel", ["s1"])
    assert book.calls[1][0] == "place"
    assert book.calls[1][3] == Decimal("30.00000000")
    assert book.calls[1][4] == Decimal("108.00000000")
    assert book.calls[1][5] == 1
    assert bot._stop["id"] == "stop-new-1"


def test_a_same_price_plan_is_not_adopted_or_cancelled():
    book = _Book([{"id": "stop-b"}])

    async def place(*_args, **_kwargs):
        book.calls.append(("place", _args))
        return "stop-b"

    book.place_position_stop_order = place
    log = _Log()
    bot = _bot(connector=book, log=log)
    bot._sleeves[0].base = 10.0
    bot._sleeves[0].phase = "open"

    async def _place():
        await bot._run_stop(book, "place", Decimal("10"), 0.09)

    asyncio.run(_place())
    assert bot._stop == {}
    assert bot._stop_blocked
    assert "same price" in bot._stop_blocked
    assert not any(call[0] == "cancel" for call in book.calls)
    actions, keep = bot._buy_actions(Intent("core", "market_buy", 1.0, 15000.0, "entry", 1, None, 0.1))
    assert actions == []
    assert keep is True


def test_a_rejected_stop_leaves_the_previous_plan_when_it_was_not_cancelled():
    book = _Book()

    async def explode(*_args, **_kwargs):
        raise RuntimeError("closable")

    book.place_position_stop_order = explode
    log = _Log()
    bot = _bot(connector=book, log=log)
    bot._stop = {"id": "s1", "amount": "10.00000000", "price": "0.08000000"}

    async def _cancel_fails():
        await bot._run_stop(book, "place", Decimal("10"), 0.08)

    book.cancel_stop_orders = explode
    asyncio.run(_cancel_fails())
    assert bot._stop["id"] == "s1"
    assert any("less than the closable" in message for message in log.errors)


def test_entries_wait_for_1x_isolated_and_for_hedge():
    connector = SimpleNamespace(
        confirm_leverage_and_margin=lambda *_args: False,
        get_available_balance=lambda _asset: Decimal("100000"),
        trading_rules={"AIOT-USDT": _Rule()},
        position_mode=PositionMode.HEDGE,
        _position_mode_confirmed=True,
    )
    bot = _bot(connector=connector)
    bot._leverage_ready = False
    bot._leverage_inflight = True
    actions, keep = bot._buy_actions(Intent("core", "market_buy", 1.0, 15000.0, "entry", 1, None, 0.1))
    assert actions == []
    assert keep is True
    bot._leverage_ready = True
    connector.position_mode = PositionMode.ONEWAY
    assert bot._entries_allowed() is False


@pytest.mark.asyncio
async def test_startup_confirms_isolated_leverage_for_the_shared_controller():
    seen = {}

    class AiotConnector:
        def confirm_leverage_and_margin(self, pair, leverage, margin, side):
            seen["aiot"] = (pair, leverage, margin, side)

            async def _done():
                return True

            return _done()

        def set_leverage(self, **kwargs):
            seen["stock"] = kwargs

    script = V2WithControllers.__new__(V2WithControllers)
    script.logger = lambda: _Log()
    script.connectors = {"mexc_perpetual": AiotConnector()}
    script._apply_leverage(
        "mexc_perpetual",
        {"controller_name": "aiot_shared", "trading_pair": "AIOT-USDT", "leverage": 1},
    )
    await asyncio.sleep(0)
    assert seen["aiot"] == ("AIOT-USDT", 1, "ISOLATED", PositionSide.LONG)
    assert "stock" not in seen


@pytest.mark.asyncio
async def test_restart_reads_mexc_order_history_by_client_id(tmp_path):
    class Connector:
        def __init__(self):
            self.account_positions = {"k": _long(600)}

        def get_contract_size(self, _pair):
            return Decimal("10")

        async def _request_order_by_client_id(self, pair, client_id):
            assert pair == "AIOT-USDT"
            assert client_id == "cid-a"
            return {"externalOid": "cid-a", "dealVol": "10", "dealAvgPrice": "0.1"}

    saved = _bot()
    saved._state_path = tmp_path / "state.json"
    saved._sleeves[0].base = 100.0
    saved._sleeves[0].phase = "open"
    saved._sleeves[0].stop_price = 0.09
    saved._own_orders = {"cid-a": {"side": "buy", "base": "100.00000000", "price": "0.10000000", "level": "shared:core:buy"}}
    saved._save_state()
    restored = _bot(connector=Connector())
    restored._state_path = saved._state_path
    restored._share_checked = False
    restored._load_state()
    assert restored.confirm_share() is False
    await asyncio.sleep(0)
    assert restored._share_halt is None
    assert restored._share_checked is True
    assert restored._sleeves[0].base == 100.0
    restored._apply_live_position()
    assert restored._live_long == Decimal("600")
    assert restored._sleeves[0].base == 100.0


def test_a_missing_client_id_does_not_guess_a_fill():
    bot = _bot()
    bot.executors_info = [
        SimpleNamespace(
            id="e1",
            controller_id="bot-a",
            config=SimpleNamespace(level_id="shared:core:buy", controller_id="bot-a"),
            custom_info={"executed_amount_base": 25, "average_executed_price": 0.1},
            is_done=True,
            is_active=False,
        )
    ]
    bot._reconcile()
    assert bot._sleeves[0].base == 0.0
    assert bot._share_halt
    assert "client order id" in bot._share_halt
