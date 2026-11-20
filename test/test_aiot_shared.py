"""Two AIOT bots can share one MEXC hedge long, and each keeps its own share."""

import asyncio
import json
import tempfile
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
    parse_stop_fill,
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


def _bot(strategy="fin_b", quote="2200", connector=None, log=None):
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
    bot._entry_halt = None
    bot._share_checked = True
    bot._checked_long = None
    bot._checked_at = None
    bot._history_inflight = False
    bot._stop_cancel_requested = False
    bot._loading_state = False
    bot.executors_info = []
    bot._state_path = Path(tempfile.mkdtemp()) / "aiot_shared_bot-a.json"
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
    assert fin["total_amount_quote"] == 2200
    assert f3["strategy"] == "f3l20"
    assert f3["total_amount_quote"] == 1650
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
    config = cls(id="t", strategy="f3l20", total_amount_quote=Decimal("1650"))
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


@pytest.mark.asyncio
async def test_one_bot_exits_its_share_and_the_other_keeps_its_stop():
    book = _Book([{"id": "stop-a"}, {"id": "stop-b"}])
    log = _Log()
    bot_a = _bot(connector=book, log=log)
    bot_b = _bot(connector=book, log=_Log())
    bot_b.config = AiotSharedConfig(
        id="bot-b", strategy="fin_b", connector_name="mexc_perpetual", total_amount_quote=Decimal("2200")
    )
    for bot, base, stop_id in ((bot_a, 100, "stop-a"), (bot_b, 80, "stop-b")):
        sleeve = bot._sleeves[0]
        sleeve.base = float(base)
        sleeve.phase = "open"
        sleeve.stop_price = 0.07
        bot._live_long = Decimal("180")
        bot._stop = {"id": stop_id, "amount": f"{base}.00000000", "price": "0.07000000"}
    held, keep = bot_a._intent_actions(Intent("core", "market_sell", 100.0, 100.0, "cross", 1))
    assert held == []
    assert keep is True
    await asyncio.sleep(0)
    assert ("cancel", ["stop-a"]) in book.calls
    assert all(call[1] != ["stop-b"] for call in book.calls if call[0] == "cancel")
    actions, keep = bot_a._intent_actions(Intent("core", "market_sell", 100.0, 100.0, "cross", 1))
    assert keep is False
    assert actions[0].executor_config.amount == Decimal("100.00000000")
    assert actions[0].executor_config.position_action is PositionAction.CLOSE
    assert actions[0].executor_config.side is TradeType.SELL
    assert actions[0].executor_config.level_id.startswith("shared:")
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
    (tmp_path / "aiot_shared_bot-b.json").write_text(
        json.dumps({"orders": {"cid-other": {"side": "buy", "base": "500.00000000"}}}),
        encoding="utf-8",
    )

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


def test_an_exit_does_not_take_the_other_bots_residual():
    log = _Log()
    bot = _bot(log=log)
    sleeve = bot._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    bot._live_long = Decimal("80")
    actions, keep = bot._intent_actions(Intent("core", "market_sell", 100.0, 1.0, "cross", 1))
    assert actions == []
    assert keep is False
    assert bot._share_halt
    assert "was not closed" in bot._share_halt
    smaller = _bot()
    smaller._sleeves[0].base = 30.0
    smaller._sleeves[0].phase = "open"
    smaller._live_long = Decimal("80")
    sold, keep = smaller._intent_actions(Intent("core", "market_sell", 30.0, 1.0, "cross", 1))
    assert keep is False
    assert sold[0].executor_config.amount == Decimal("30.00000000")
    assert sold[0].executor_config.position_action is PositionAction.CLOSE


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
    fin = _bot(strategy="fin_b", quote="2200", connector=connector)
    actions, keep = fin._buy_actions(Intent("core", "market_buy", 1.0, 15000.0, "entry", 1, None, float(PRICE)))
    assert keep is False
    quote = actions[0].executor_config.amount * PRICE
    assert quote <= Decimal("2200")
    assert quote > Decimal("2190")
    f3 = _bot(strategy="f3l20", quote="1650", connector=connector)
    first, _keep = f3._buy_actions(Intent("core", "market_buy", 1.0, 1500.0, "entry", 1, None, float(PRICE)))
    cash["value"] = Decimal("1000")
    second, _keep = f3._buy_actions(Intent("core", "market_buy", 1.0, 1500.0, "add", 2, -1, float(PRICE)))
    second_quote = second[0].executor_config.amount * PRICE
    assert first[0].executor_config.amount * PRICE > Decimal("1640")
    assert Decimal("900") < second_quote < Decimal("950")
    assert second[0].executor_config.position_action is PositionAction.OPEN


def test_f3_l20_leg_two_updates_the_average_stop_and_replaces_only_that_stop():
    book = _Book([{"id": "s1"}])
    bot = _bot(strategy="f3l20", quote="1650", connector=book)
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


@pytest.mark.asyncio
async def test_a_matching_stop_is_not_replaced_on_the_shared_long():
    book = _Book([{"id": "stop-a", "stopLossPrice": "0.07"}])
    bot = _bot(connector=book)
    sleeve = bot._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    sleeve.stop_price = 0.07
    bot._live_long = Decimal("180")
    bot._positions_seen = True
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    for _ in range(3):
        bot._stop_inflight = False
        bot._stop_sync_inflight = False
        bot._stop_sync_next = 0.0
        bot._schedule_stops()
        await asyncio.sleep(0)
    assert [call for call in book.calls if call[0] in ("cancel", "place")] == []
    assert bot._stop["id"] == "stop-a"
    assert bot._share_halt is None


def test_a_live_exchange_stop_is_not_joined_by_a_market_stop_sell():
    book = _Book([{"id": "stop-a"}])
    bot = _bot(connector=book)
    sleeve = bot._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    sleeve.stop_price = 0.07
    bot._live_long = Decimal("180")
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    sleeve.on_price(0.01)
    assert sleeve.phase == "pending_exit"
    actions = []
    for intent in sleeve.drain():
        built, _keep = bot._intent_actions(intent)
        actions.extend(built)
    assert actions == []
    assert sleeve.phase == "open"
    assert bot._stop["id"] == "stop-a"


@pytest.mark.asyncio
async def test_a_rejected_place_after_cancel_stops_new_entries():
    book = _Book([{"id": "s1"}])

    async def explode(*_args, **_kwargs):
        book.calls.append("place")
        raise RuntimeError("closable")

    book.place_position_stop_order = explode
    bot = _bot(connector=book)
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._stop = {"id": "s1", "amount": "100.00000000", "price": "0.08000000"}
    await bot._run_stop(book, "place", Decimal("60"), 0.08)
    assert bot._stop == {}
    assert bot._entry_halt
    assert bot._entries_allowed() is False
    assert ("cancel", ["s1"]) in book.calls


def test_a_renewed_order_keeps_each_client_id_fill():
    bot = _bot()
    bot.executors_info = [
        SimpleNamespace(
            id="e1",
            controller_id="bot-a",
            config=SimpleNamespace(level_id="shared:core:buy", controller_id="bot-a"),
            custom_info={
                "executed_amount_base": 100,
                "average_executed_price": 0.1,
                "order_id": "cid-b",
                "held_position_orders": [
                    {
                        "client_order_id": "cid-a",
                        "executed_amount_base": "40",
                        "executed_amount_quote": "4",
                    }
                ],
            },
            is_done=True,
            is_active=False,
        )
    ]
    bot._reconcile()
    assert bot._share_halt is None
    assert bot._sleeves[0].base == pytest.approx(100.0)
    assert bot._own_orders["cid-a"]["base"] == "40.00000000"
    assert bot._own_orders["cid-b"]["base"] == "60.00000000"


def _sibling(bot, orders):
    path = bot._state_path.parent / "aiot_shared_bot-b.json"
    path.write_text(json.dumps({"orders": orders}), encoding="utf-8")


def test_a_stop_plan_fill_is_booked_on_that_bot_and_reloaded(tmp_path):
    bot = _bot()
    bot._state_path = tmp_path / "aiot_shared_bot-a.json"
    sleeve = bot._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    sleeve.stop_price = 0.07
    bot._own_orders = {"cid-a": {"side": "buy", "base": "100.00000000", "price": "0.10000000", "level": "shared:core:buy"}}
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    bot._share_checked = False
    _sibling(bot, {"cid-other": {"side": "buy", "base": "80.00000000"}})
    bot.market_data_provider.connectors["mexc_perpetual"] = SimpleNamespace(
        fill_history=lambda: [
            {"client_order_id": "cid-a", "base": "100"},
            {"client_order_id": "cid-other", "base": "80"},
            {"stopPlanOrderId": "stop-a", "base": "100", "stopLossPrice": "0.07"},
            {"stopPlanOrderId": "stop-b", "base": "80", "stopLossPrice": "0.05"},
        ]
    )
    assert bot.confirm_share() is True
    assert bot._share_halt is None
    assert bot._sleeves[0].base == 0.0
    assert bot._sleeves[0].phase == "flat"
    assert bot._own_orders["stop:stop-a"]["side"] == "sell"
    assert bot._own_orders["stop:stop-a"]["base"] == "100.00000000"
    assert "cid-other" not in bot._own_orders
    assert "stop:stop-b" not in bot._own_orders
    reloaded = _bot()
    reloaded._state_path = bot._state_path
    reloaded._load_state()
    assert reloaded._share_halt is None
    assert reloaded._sleeves[0].base == 0.0
    assert reloaded._own_orders["stop:stop-a"]["base"] == "100.00000000"
    assert reloaded._stop == {}


def test_an_unattributed_fill_halts_and_the_halt_is_reloaded(tmp_path):
    bot = _bot()
    bot._state_path = tmp_path / "aiot_shared_bot-a.json"
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._own_orders = {"cid-a": {"side": "buy", "base": "100", "price": "0.1", "level": "shared:core:buy"}}
    bot._share_checked = False
    bot.market_data_provider.connectors["mexc_perpetual"] = SimpleNamespace(
        fill_history=lambda: [
            {"client_order_id": "cid-a", "base": "100"},
            {"base": "20", "dealVol": "2"},
        ]
    )
    assert bot.confirm_share() is False
    assert "no client order id" in bot._share_halt
    assert bot._entries_allowed() is False
    sold, _keep = bot._intent_actions(Intent("core", "market_sell", 100.0, 1.0, "cross", 1))
    assert sold == []
    reloaded = _bot()
    reloaded._state_path = bot._state_path
    reloaded._load_state()
    assert reloaded._share_halt == bot._share_halt
    assert reloaded._entries_allowed() is False


def test_an_unsaved_client_id_halts_and_another_bots_id_does_not(tmp_path):
    bot = _bot()
    bot._state_path = tmp_path / "aiot_shared_bot-a.json"
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._own_orders = {"cid-a": {"side": "buy", "base": "100.00000000", "price": "0.1", "level": "shared:core:buy"}}
    bot._share_checked = False
    bot.market_data_provider.connectors["mexc_perpetual"] = SimpleNamespace(
        fill_history=lambda: [
            {"client_order_id": "cid-a", "base": "100"},
            {"client_order_id": "cid-missed", "base": "25"},
        ]
    )
    assert bot.confirm_share() is False
    assert "cid-missed" in bot._share_halt
    assert "not saved" in bot._share_halt

    other = _bot()
    other._state_path = tmp_path / "aiot_shared_bot-a2.json"
    other._sleeves[0].base = 100.0
    other._sleeves[0].phase = "open"
    other._own_orders = {"cid-a": {"side": "buy", "base": "100.00000000", "price": "0.1", "level": "shared:core:buy"}}
    other._share_checked = False
    _sibling(other, {"cid-other": {"side": "buy", "base": "80.00000000"}})
    other.market_data_provider.connectors["mexc_perpetual"] = SimpleNamespace(
        fill_history=lambda: [
            {"client_order_id": "cid-a", "base": "100"},
            {"client_order_id": "cid-other", "base": "80"},
        ]
    )
    assert other.confirm_share() is True
    assert other._share_halt is None
    assert other._sleeves[0].base == pytest.approx(100.0)


def test_the_share_check_runs_again_after_the_first_success():
    box = {"rows": [{"client_order_id": "cid-a", "base": "100"}]}
    bot = _bot(connector=SimpleNamespace(fill_history=lambda: box["rows"]))
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._own_orders = {"cid-a": {"side": "buy", "base": "100.00000000", "price": "0.1", "level": "shared:core:buy"}}
    bot._share_checked = False
    bot._now = lambda: 100.0
    assert bot.confirm_share() is True
    box["rows"] = [{"client_order_id": "cid-a", "base": "40"}]
    assert bot.confirm_share() is True
    bot._live_long = Decimal("100")
    assert bot.confirm_share() is False
    assert bot._share_halt is None
    assert bot.confirm_share() is False
    assert "halted" in bot._share_halt


def test_a_same_price_plan_keeps_the_other_bots_volume():
    tick = Decimal("0.0001")
    book = _Book([{"id": "stop-b", "stopLossPrice": "0.07000000", "base": "80"}])
    book.trading_rules = {"AIOT-USDT": SimpleNamespace(min_price_increment=tick)}
    bot = _bot(connector=book)
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._sleeves[0].stop_price = 0.07

    async def _place():
        await bot._run_stop(book, "place", Decimal("100"), 0.07)

    asyncio.run(_place())
    places = [call for call in book.calls if call[0] == "place"]
    assert len(places) == 1
    assert places[0][4] == Decimal("0.07010000")
    assert book.open_stops[0]["base"] == "80"
    assert bot._stop["id"] == "stop-new-1"
    assert bot._stop_blocked is None
    assert not any(call[0] == "cancel" for call in book.calls)

    merged = _Book([{"id": "stop-b", "stopLossPrice": "0.09000000", "base": "80"}])
    merged.trading_rules = {"AIOT-USDT": SimpleNamespace(min_price_increment=tick)}

    async def place(pair, position_id, amount, stop_loss_price=None, vol_type=None, stop_loss_reverse=None):
        merged.calls.append(("place", pair, position_id, amount, stop_loss_price, vol_type, stop_loss_reverse))
        return "stop-b"

    merged.place_position_stop_order = place
    victim = _bot(connector=merged)
    victim._sleeves[0].base = 10.0
    victim._sleeves[0].phase = "open"

    async def _merge():
        await victim._run_stop(merged, "place", Decimal("10"), 0.09)

    asyncio.run(_merge())
    assert victim._stop == {}
    assert victim._stop_blocked
    assert "same price" in victim._stop_blocked
    restored = [call for call in merged.calls if call[0] == "place" and call[3] == Decimal("80.00000000")]
    assert restored
    assert restored[0][4] == Decimal("0.09000000")
    assert not any(call[0] == "cancel" for call in merged.calls)


def test_a_stop_whose_volume_changed_is_not_left_in_place():
    book = _Book([{"id": "stop-a", "stopLossPrice": "0.07000000", "base": "40"}])
    bot = _bot(connector=book)
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._sleeves[0].stop_price = 0.07
    bot._live_long = Decimal("180")
    bot._positions_seen = True
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}

    async def _sync():
        await bot._sync_stops(book)

    asyncio.run(_sync())
    assert bot._stop.get("amount") in (None, "")
    assert bot._stop["id"] == "stop-a"
    bot._stop_inflight = False

    async def _replace():
        bot._schedule_stops()
        await asyncio.sleep(0)

    asyncio.run(_replace())
    places = [call for call in book.calls if call[0] == "place"]
    assert places
    assert places[0][3] == Decimal("100.00000000")


@pytest.mark.asyncio
async def test_a_stop_rounded_down_to_the_tick_is_not_replaced():
    tick = Decimal("0.0001")
    book = _Book([{"id": "stop-a", "stopLossPrice": "0.07000000"}])
    book.trading_rules = {
        "AIOT-USDT": SimpleNamespace(
            min_order_size=Decimal("10"),
            max_order_size=MAX_BASE,
            min_base_amount_increment=Decimal("10"),
            min_price_increment=tick,
        )
    }
    bot = _bot(connector=book)
    sleeve = bot._sleeves[0]
    sleeve.base = 100.0
    sleeve.phase = "open"
    sleeve.stop_price = 0.07003
    bot._live_long = Decimal("180")
    bot._positions_seen = True
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    for _ in range(3):
        bot._stop_inflight = False
        bot._stop_sync_inflight = False
        bot._stop_sync_next = 0.0
        bot._schedule_stops()
        await asyncio.sleep(0)
    assert [call for call in book.calls if call[0] in ("cancel", "place")] == []
    assert bot._stop["id"] == "stop-a"


@pytest.mark.asyncio
async def test_a_stop_plan_key_is_not_looked_up_as_a_client_order(tmp_path):
    seen = []

    class Connector:
        async def _request_order_by_client_id(self, pair, client_id):
            seen.append(str(client_id))
            if str(client_id).startswith("stop:"):
                raise ValueError(f"Order {client_id} not found")
            return {"externalOid": client_id, "dealVol": "10", "dealAvgPrice": "0.1"}

        def get_contract_size(self, _pair):
            return Decimal("10")

    bot = _bot(connector=Connector())
    bot._state_path = tmp_path / "aiot_shared_bot-a.json"
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._own_orders = {
        "cid-a": {"side": "buy", "base": "100.00000000", "price": "0.1", "level": "shared:core:buy"},
        "stop:77": {
            "side": "sell",
            "level": "shared:stop:sell",
            "plan_id": "77",
            "base": "0.00000000",
            "price": "0.07",
        },
    }
    bot._stop = {"id": "77", "amount": "100.00000000", "price": "0.07000000"}
    bot._share_checked = False
    assert bot.confirm_share() is False
    await asyncio.sleep(0)
    assert seen == ["cid-a"]
    assert bot._share_halt is None
    assert bot._sleeves[0].base == pytest.approx(100.0)


def test_history_and_the_executor_do_not_double_the_share():
    bot = _bot(connector=SimpleNamespace(fill_history=lambda: [{"client_order_id": "cid-a", "base": "40"}]))
    bot._own_orders = {"cid-a": {"side": "buy", "base": "0.00000000", "price": "0", "level": "shared:core:buy"}}
    bot._sleeves[0].base = 0.0
    bot._share_checked = False
    bot._live_long = Decimal("40")
    bot.executors_info = [_executor("e1", "shared:core:buy", 40, 0.1, "cid-a")]
    bot._reconcile()
    assert bot._share_halt is None
    assert bot._sleeves[0].base == pytest.approx(40.0)
    assert bot._own_orders["cid-a"]["base"] == "40.00000000"


def _plans():
    """MEXC stoporder list rows. realityVol is contracts. state 3 is executed."""
    return [
        {
            "id": "stop-a",
            "state": 3,
            "triggerSide": 2,
            "realityVol": "10",
            "vol": "10",
            "stopLossPrice": "0.07",
            "isFinished": 1,
        },
        {
            "id": "stop-b",
            "state": 3,
            "triggerSide": 2,
            "realityVol": "8",
            "vol": "8",
            "stopLossPrice": "0.05",
            "isFinished": 1,
        },
    ]


def _armed(bot, base, stop_id, price):
    sleeve = bot._sleeves[0]
    sleeve.base = float(base)
    sleeve.phase = "open"
    sleeve.stop_price = price
    bot._own_orders = {"cid-" + stop_id: {"side": "buy", "base": f"{base}.00000000", "price": "0.1", "level": "shared:core:buy"}}
    bot._stop = {"id": stop_id, "amount": f"{base}.00000000", "price": f"{price:.8f}"}
    bot._live_long = Decimal("180")


@pytest.mark.asyncio
async def test_one_stop_fill_books_only_that_plans_volume():
    book = _Book([{"id": "stop-b", "stopLossPrice": "0.05", "base": "80"}])
    book.get_contract_size = lambda _pair: Decimal("10")

    async def list_stop_orders(_pair):
        rows = _plans()
        rows[1] = {"id": "stop-b", "state": 1, "triggerSide": 0, "realityVol": "0", "vol": "8", "stopLossPrice": "0.05"}
        return rows

    book.list_stop_orders = list_stop_orders
    bot_a = _bot(connector=book)
    bot_b = _bot(connector=book)
    bot_b.config = AiotSharedConfig(
        id="bot-b", strategy="fin_b", connector_name="mexc_perpetual", total_amount_quote=Decimal("2200")
    )
    _armed(bot_a, 100, "stop-a", 0.07)
    _armed(bot_b, 80, "stop-b", 0.05)
    await bot_a._sync_stops(book)
    await bot_b._sync_stops(book)
    assert bot_a._share_halt is None
    assert bot_a._sleeves[0].base == 0.0
    assert bot_a._own_orders["stop:stop-a"]["base"] == "100.00000000"
    assert "stop:stop-b" not in bot_a._own_orders
    assert bot_b._share_halt is None
    assert bot_b._sleeves[0].base == 80.0
    assert bot_b._stop["id"] == "stop-b"
    assert not any(call[0] == "cancel" and call[1] == ["stop-b"] for call in book.calls)


@pytest.mark.asyncio
async def test_both_stop_fills_are_booked_from_each_plan():
    book = _Book([])
    book.get_contract_size = lambda _pair: Decimal("10")

    async def _async_plans(_pair):
        return _plans()

    book.list_stop_orders = _async_plans
    bot_a = _bot(connector=book)
    bot_b = _bot(connector=book)
    _armed(bot_a, 100, "stop-a", 0.07)
    _armed(bot_b, 80, "stop-b", 0.05)
    await bot_a._sync_stops(book)
    await bot_b._sync_stops(book)
    assert bot_a._sleeves[0].base == 0.0
    assert bot_a._own_orders["stop:stop-a"]["base"] == "100.00000000"
    assert "stop:stop-b" not in bot_a._own_orders
    assert bot_b._sleeves[0].base == 0.0
    assert bot_b._own_orders["stop:stop-b"]["base"] == "80.00000000"
    assert "stop:stop-a" not in bot_b._own_orders
    assert bot_a._share_halt is None and bot_b._share_halt is None


@pytest.mark.asyncio
async def test_a_close_with_no_triggered_plan_halts_and_does_not_invent_a_fill():
    book = _Book([])

    async def list_stop_orders(_pair):
        return [{"id": "stop-a", "state": 2, "triggerSide": 0, "realityVol": "0", "vol": "10", "stopLossPrice": "0.07"}]

    book.list_stop_orders = list_stop_orders
    bot = _bot(connector=book)
    _armed(bot, 100, "stop-a", 0.07)
    bot._live_long = Decimal("0")
    await bot._sync_stops(book)
    await bot._sync_stops(book)
    assert bot._share_halt
    assert "could not be attributed" in bot._share_halt
    assert bot._sleeves[0].base == 100.0
    assert [call for call in book.calls if call[0] == "place"] == []
    assert bot._state_path.read_text(encoding="utf-8").find(bot._share_halt) > 0


@pytest.mark.asyncio
async def test_a_shrink_reads_the_finished_plan_before_covering_the_residual():
    book = _Book([{"id": "stop-b", "stopLossPrice": "0.05", "base": "80"}])
    book.get_contract_size = lambda _pair: Decimal("10")
    book.account_positions = {"k": _long(80)}
    book.fill_history = lambda: [{"client_order_id": "cid-stop-a", "base": "100"}]
    seen = {"n": 0}

    async def list_stop_orders(_pair):
        seen["n"] += 1
        rows = _plans()
        rows[1] = {"id": "stop-b", "state": 1, "triggerSide": 0, "realityVol": "0", "vol": "8", "stopLossPrice": "0.05"}
        return rows

    book.list_stop_orders = list_stop_orders
    bot = _bot(connector=book)
    _armed(bot, 100, "stop-a", 0.07)
    bot._apply_live_position()
    bot._schedule_stops()
    await asyncio.sleep(0)
    assert seen["n"] >= 1
    assert bot._share_halt is None
    assert bot._sleeves[0].base == 0.0
    assert bot._own_orders["stop:stop-a"]["base"] == "100.00000000"
    assert [call for call in book.calls if call[0] == "place"] == []
    assert not any(call[0] == "cancel" and call[1] == ["stop-b"] for call in book.calls)


@pytest.mark.asyncio
async def test_a_failed_cancel_does_not_ignore_a_missing_plan():
    book = _Book([])

    async def boom(*_args, **_kwargs):
        raise RuntimeError("timeout")

    book.cancel_stop_orders = boom
    bot = _bot(connector=book)
    bot._stop_cancel_requested = True
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._live_long = Decimal("80")
    bot._stop = {"id": "s1", "amount": "100.00000000", "price": "0.07000000"}
    await bot._run_stop(book, "cancel", 0, None)
    assert bot._stop_cancel_requested is False
    assert bot._stop.get("id") == "s1"


@pytest.mark.asyncio
async def test_an_oversized_stop_is_cancelled_when_the_long_is_smaller():
    book = _Book([{"id": "stop-a"}])
    book.account_positions = {"k": _long(80)}
    bot = _bot(connector=book)
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    bot._sleeves[0].stop_price = 0.07
    bot._apply_live_position()
    for _ in range(4):
        await asyncio.sleep(0)
    assert bot._share_halt
    assert bot._sleeves[0].base == 100.0
    assert [call for call in book.calls if call[0] == "place"] == []
    assert ("cancel", ["stop-a"]) in book.calls
    assert bot._stop == {}


@pytest.mark.asyncio
async def test_a_cancel_is_saved_before_the_sell(tmp_path):
    book = _Book([])
    bot = _bot(connector=book)
    bot._state_path = tmp_path / "aiot_shared_bot-a.json"
    bot._sleeves[0].base = 100.0
    bot._sleeves[0].phase = "open"
    bot._stop = {"id": "stop-a", "amount": "100.00000000", "price": "0.07000000"}
    bot._own_orders = {"cid-a": {"side": "buy", "base": "100.00000000", "price": "0.1", "level": "shared:core:buy"}}
    bot._stop_cancel_requested = True
    await bot._run_stop(book, "cancel", 0, None)
    saved = json.loads(bot._state_path.read_text(encoding="utf-8"))
    assert saved.get("stop") == {}
    assert bot._stop == {}
    assert bot._stop_cancel_requested is False
    sold, keep = bot._intent_actions(Intent("core", "market_sell", 100.0, 100.0, "cross", 1))
    assert keep is False
    assert sold[0].executor_config.amount == Decimal("100.00000000")
    assert sold[0].executor_config.position_action is PositionAction.CLOSE


def test_states_4_and_5_are_not_stop_fills():
    failed = {"id": "stop-a", "state": 5, "realityVol": "10", "isFinished": 1, "stopLossPrice": "0.07"}
    invalid = {"id": "stop-a", "state": 4, "realityVol": "10", "isFinished": 1, "stopLossPrice": "0.07"}
    assert parse_stop_fill(failed, Decimal("10")) is None
    assert parse_stop_fill(invalid, Decimal("10")) is None
    executed = parse_stop_fill(
        {"id": "stop-a", "state": 3, "realityVol": "10", "stopLossPrice": "0.07"},
        Decimal("10"),
    )
    assert executed[0] == "stop-a"
    assert executed[1] == Decimal("100")


@pytest.mark.asyncio
async def test_a_cancelled_plan_does_not_stop_the_other_bots_long(tmp_path):
    book = _Book([])
    book.account_positions = {"k": _long(80)}

    async def list_stop_orders(_pair):
        return [{"id": "stop-a", "state": 2, "triggerSide": 0, "realityVol": "0", "vol": "10", "stopLossPrice": "0.07"}]

    book.list_stop_orders = list_stop_orders
    bot = _bot(connector=book)
    bot._state_path = tmp_path / "aiot_shared_bot-a.json"
    (tmp_path / "aiot_shared_bot-b.json").write_text(
        json.dumps({"sleeves": [{"name": "core", "base": 80}]}),
        encoding="utf-8",
    )
    _armed(bot, 100, "stop-a", 0.07)
    bot._apply_live_position()
    bot._schedule_stops()
    for _ in range(4):
        await asyncio.sleep(0)
    assert bot._share_halt
    assert bot._sleeves[0].base == 100.0
    assert [call for call in book.calls if call[0] == "place"] == []
    assert ("cancel", ["stop-a"]) in book.calls
    assert bot._stop == {}


@pytest.mark.asyncio
async def test_finished_stop_list_uses_the_plan_time_and_pages_past_100():
    calls = []

    class Connector:
        def exchange_symbol_associated_to_pair(self, _pair):
            return "AIOT_USDT"

        async def _api_get(self, path_url, params, is_auth_required):
            calls.append(dict(params))
            assert path_url == "/api/v1/private/stoporder/list/orders"
            assert is_auth_required is True
            if params["page_num"] == 1:
                return {"data": [{"id": "old-%s" % i, "state": 2} for i in range(100)]}
            return {"data": [{"id": "stop-a", "state": 3, "realityVol": "1", "stopLossPrice": "0.07"}]}

    bot = _bot(connector=Connector())
    bot._stop = {"id": "stop-a", "amount": "10", "price": "0.07", "created": 100000}
    rows = await bot._stop_plan_rows(Connector())
    assert [row["id"] for row in rows if row["id"] == "stop-a"] == ["stop-a"]
    assert len(calls) == 2
    assert calls[0]["start_time"] == 100000
    assert calls[0]["page_num"] == 1
    assert calls[1]["page_num"] == 2
    assert calls[0]["is_finished"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sibling", ["absent", "zero", "broken", "eighty", "high"])
async def test_a_smaller_long_does_not_get_a_stop_from_the_other_file(tmp_path, sibling):
    book = _Book([{"id": "stop-a", "stopLossPrice": "0.07", "base": "100"}])
    book.account_positions = {"k": _long(80)}
    book.fill_history = lambda: [{"client_order_id": "cid-stop-a", "base": "100"}]

    async def list_stop_orders(_pair):
        return [{"id": "stop-a", "state": 2, "realityVol": "0", "vol": "10", "stopLossPrice": "0.07"}]

    book.list_stop_orders = list_stop_orders
    bot = _bot(connector=book)
    bot._state_path = tmp_path / "aiot_shared_bot-a.json"
    other = tmp_path / "aiot_shared_bot-b.json"
    if sibling == "zero":
        other.write_text(json.dumps({"sleeves": [{"base": 0}]}), encoding="utf-8")
    elif sibling == "broken":
        other.write_text('{"sleeves": [{"base": 80', encoding="utf-8")
    elif sibling == "eighty":
        other.write_text(json.dumps({"sleeves": [{"base": 80}]}), encoding="utf-8")
    elif sibling == "high":
        other.write_text(json.dumps({"sleeves": [{"base": 200}]}), encoding="utf-8")
    _armed(bot, 100, "stop-a", 0.07)
    bot._apply_live_position()
    bot._schedule_stops()
    for _ in range(4):
        await asyncio.sleep(0)
    assert bot._share_halt
    assert bot._sleeves[0].base == 100.0
    assert [call for call in book.calls if call[0] == "place"] == []
    assert ("cancel", ["stop-a"]) in book.calls
    assert bot._stop == {}


def test_an_old_stop_query_stays_inside_ninety_days():
    bot = _bot()
    bot._now = lambda: 200 * 24 * 60 * 60
    bot._stop = {"id": "stop-a", "created": 1000}
    windows = bot._stop_windows()
    span = 89 * 24 * 60 * 60 * 1000
    assert len(windows) == 2
    assert windows[0][0] > 1000
    assert windows[1][0] == 1000
    for start, end in windows:
        assert end >= start
        assert end - start <= span


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
