"""Open executor orders are cancelled on the way down, and again on the way back up.

`ExecutorService.stop` used to call `executor.stop()`, which drops the task and leaves
working orders on the exchange. The next boot then marked every leftover RUNNING row
SYSTEM_CLEANUP and never looked at those orders again. Graceful shutdown now asks each
executor for `early_stop(keep_position=True)` (cancel working orders, do not close a
fill) and waits, up to a bound, for cancel confirmation before trading interfaces are
torn down. An unclean shutdown leaves the row RUNNING; startup cancels child orders
still open on the exchange, using the persisted executor_orders rows and custom_info,
and only then marks the row SYSTEM_CLEANUP. A range iceberg saved with
resume_on_startup and exit_state ARMED, RESTING, or CLOSING is resumed before
that cleanup. Resuming continues exits only, on the exchange position, and
does not continue an unfinished entry. Cleanup still cancels that executor's
entry and exit client order ids. Only its exchange stop stays up.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from database.models import ExecutorOrder, ExecutorPerformanceSnapshot, ExecutorRecord, Order
from database.repositories.executor_repository import ExecutorRepository
from database.repositories.order_repository import OrderRepository
from services.executor_service import (
    ExecutorService,
    _exchange_stop_ids,
    _exit_only_custom_info,
    child_order_ids_from_custom_info,
    range_iceberg_entry_pauses,
    range_iceberg_held_base,
    range_iceberg_side_held_base,
    range_iceberg_sibling_hints,
)


class _AsyncSessionAdapter:
    """The async surface the repositories use, over a real synchronous Session."""

    def __init__(self, session: Session):
        self._session = session

    def add(self, obj):
        self._session.add(obj)

    def add_all(self, objs):
        self._session.add_all(objs)

    async def execute(self, statement):
        return self._session.execute(statement)

    async def flush(self):
        self._session.flush()

    async def refresh(self, obj):
        self._session.refresh(obj)

    async def commit(self):
        self._session.commit()

    async def rollback(self):
        self._session.rollback()

    async def close(self):
        self._session.close()

    def begin_nested(self):
        nested = self._session.begin_nested()

        @asynccontextmanager
        async def _savepoint():
            try:
                yield nested
                if nested.is_active:
                    nested.commit()
            except Exception:
                if nested.is_active:
                    nested.rollback()
                raise

        return _savepoint()


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    ExecutorRecord.__table__.create(engine)
    ExecutorOrder.__table__.create(engine)
    Order.__table__.create(engine)
    ExecutorPerformanceSnapshot.__table__.create(engine)

    @asynccontextmanager
    async def session_context():
        session = Session(engine)
        adapter = _AsyncSessionAdapter(session)
        try:
            yield adapter
            await adapter.commit()
        except Exception:
            await adapter.rollback()
            raise
        finally:
            await adapter.close()

    def query(model):
        with Session(engine) as session:
            return session.query(model).all()

    try:
        yield SimpleNamespace(session_context=session_context, query=query)
    finally:
        engine.dispose()


class _Interface:
    def __init__(self, events):
        self.events = events

    async def cleanup(self):
        self.events.append("cleanup")


class _Connector:
    def __init__(self, confirmed=False):
        self.in_flight_orders = {}
        self.cancels = []
        self.stop_cancels = []
        self.closes = []
        self.confirmed = confirmed
        self._confirm_events = None
        self.stop_result = None

    def is_cancel_confirmed(self, order_id):
        confirmed = self.confirmed is True or (isinstance(self.confirmed, set) and order_id in self.confirmed)
        if confirmed and self._confirm_events is not None:
            self._confirm_events.append("confirmed")
            self._confirm_events = None
        return confirmed

    async def cancel(self, trading_pair=None, client_order_id=None):
        self.cancels.append((trading_pair, client_order_id))

    async def cancel_stop_orders(self, stop_ids):
        self.stop_cancels.extend(list(stop_ids))
        if self.stop_result is not None:
            return self.stop_result
        for stop_id in stop_ids:
            self.in_flight_orders.pop(stop_id, None)
        return {"success": True, "code": 0, "data": [{"stopPlanOrderId": stop_id, "errorCode": 0} for stop_id in stop_ids]}

    async def list_open_stop_orders(self, trading_pair=None):
        return [{"id": order_id} for order_id in self.in_flight_orders if "stop" in str(order_id)]

    async def buy(self, *args, **kwargs):
        self.closes.append(("buy", args, kwargs))

    async def sell(self, *args, **kwargs):
        self.closes.append(("sell", args, kwargs))


class _Executor:
    def __init__(self, events, custom_info=None, unconfirmed=None):
        self.events = events
        self.is_closed = False
        self.status = SimpleNamespace(name="RUNNING")
        self.close_type = None
        self._shutdown_pending = False
        self._unconfirmed_cancels = unconfirmed
        self.custom_info = custom_info or {}
        self.connectors = {}
        self.early_kwargs = []
        self.closes = []

    def get_custom_info(self):
        if isinstance(self.custom_info, Exception):
            raise self.custom_info
        return self.custom_info

    def early_stop(self, keep_position=True):
        self.early_kwargs.append(keep_position)
        self.events.append("early_stop")
        self.close_type = SimpleNamespace(name="POSITION_HOLD")
        self._shutdown_pending = True
        self.status = SimpleNamespace(name="SHUTTING_DOWN")

    def stop(self):
        self.events.append("stop")

    def close_position(self):
        self.closes.append("close_position")


def _service(db=None, connectors=None):
    service = ExecutorService.__new__(ExecutorService)
    service.db_manager = None if db is None else SimpleNamespace(get_session_context=db.session_context)
    service._active_executors = {}
    service._executor_metadata = {}
    service._trading_interfaces = {}
    service._persisted_child_order_ids = {}
    service._control_loop_task = None
    service._is_running = True
    service._log_capture = MagicMock()
    service._positions_held = {}
    service._entry_pause_sent = {}
    service._entry_pause_streak = {}
    service._lp_position_addresses = {}
    service._lp_rent_recorded = set()
    service._lp_rent_retry_after = {}
    service.default_account = "master"
    service.SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 2.0
    service.SHUTDOWN_CANCEL_POLL_SECONDS = 0.01
    books = connectors or {}

    def get_account_connectors(account_name):
        return books.get(account_name, {})

    service._trading_service = SimpleNamespace(
        connector_service=SimpleNamespace(get_account_connectors=get_account_connectors),
        update_all_timestamps=lambda: None,
    )
    return service


def _track_completion(service, events):
    async def _complete(executor_id):
        executor = service._active_executors[executor_id]
        events.append(("completion", getattr(executor.close_type, "name", None)))
        service._active_executors.pop(executor_id, None)
        service._executor_metadata.pop(executor_id, None)

    service._handle_executor_completion = _complete


def _open_order():
    return SimpleNamespace(
        is_failure=False,
        is_cancelled=False,
        is_filled=False,
        is_done=False,
        current_state=SimpleNamespace(name="OPEN"),
    )


def _cancelled_order():
    return SimpleNamespace(
        is_failure=False,
        is_cancelled=True,
        is_filled=False,
        is_done=True,
        current_state=SimpleNamespace(name="CANCELED"),
    )


async def _seed_executor(db, executor_id, account, connector, pair, final_state, child_orders):
    async with db.session_context() as session:
        repo = ExecutorRepository(session)
        await repo.create_executor(
            executor_id=executor_id,
            executor_type="range_iceberg_executor",
            account_name=account,
            connector_name=connector,
            trading_pair=pair,
        )
        if final_state is not None:
            await repo.update_executor(executor_id, final_state=json.dumps(final_state))
        for client_order_id, status in child_orders:
            await repo.create_executor_order(
                executor_id=executor_id,
                client_order_id=client_order_id,
                order_type="child",
                trade_type="BUY",
                amount=Decimal("1"),
                status=status,
            )


async def _seed_exchange_order(db, client_order_id, status, account="master", connector="mexc_perpetual", pair="BTC-USDT"):
    async with db.session_context() as session:
        await OrderRepository(session).create_order(
            {
                "client_order_id": client_order_id,
                "account_name": account,
                "connector_name": connector,
                "trading_pair": pair,
                "trade_type": "BUY",
                "order_type": "LIMIT",
                "amount": Decimal("1"),
                "status": status,
            }
        )


def test_child_order_ids_keep_fills_out_of_the_cancel_set():
    working, filled = child_order_ids_from_custom_info(
        {
            "child_order_ids": ["a", "b", "skipped", ""],
            "levels": [
                {"order_id": "b", "filled": True},
                {"order_id": "a", "filled": False},
            ],
            "held_position_orders": [{"client_order_id": "c"}],
            "order_ids": ["a", {"order_id": "d", "is_filled": True}],
            "order_id": "e",
        }
    )

    assert working == ["a", "e"]
    assert filled == ["c", "b", "d"]
    assert child_order_ids_from_custom_info(None) == ([], [])
    assert child_order_ids_from_custom_info(["a"]) == ([], [])


@pytest.mark.asyncio
async def test_graceful_stop_waits_for_cancel_confirmation_before_teardown():
    """keep_position stays on: the service cancels through early_stop and does not close the fill."""
    events = []
    connector = _Connector(confirmed=False)
    connector._confirm_events = events
    executor = _Executor(
        events,
        custom_info={"child_order_ids": ["oid-working"], "side": "BUY"},
        unconfirmed=["oid-working"],
    )
    executor.connectors = {"mexc_perpetual": connector}
    connector.in_flight_orders["oid-working"] = _open_order()

    service = _service()
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = {
        "connector_name": "mexc_perpetual",
        "trading_pair": "BTC-USDT",
        "account_name": "master",
    }
    _track_completion(service, events)

    async def confirm():
        await asyncio.sleep(0.05)
        connector.confirmed = True

    task = asyncio.create_task(confirm())
    await service.stop()
    await task

    assert executor.early_kwargs == [True]
    assert executor.closes == []
    assert connector.closes == []
    assert connector.cancels == []
    assert events == ["early_stop", "confirmed", "stop", ("completion", "POSITION_HOLD"), "cleanup"]
    assert "ex-1" not in service._active_executors


@pytest.mark.asyncio
async def test_graceful_stop_timeout_leaves_the_executor_running(db, caplog):
    connector = _Connector(confirmed=False)
    events = []
    executor = _Executor(
        events,
        custom_info={"child_order_ids": ["oid-working"], "side": "BUY"},
        unconfirmed=["oid-working"],
    )
    executor.connectors = {"mexc_perpetual": connector}
    connector.in_flight_orders["oid-working"] = _open_order()

    await _seed_executor(
        db,
        "ex-1",
        "master",
        "mexc_perpetual",
        "BTC-USDT",
        None,
        [],
    )
    await _seed_exchange_order(db, "oid-working", "OPEN")

    service = _service(db, {"master": {"mexc_perpetual": connector}})
    service.SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = {
        "connector_name": "mexc_perpetual",
        "trading_pair": "BTC-USDT",
        "account_name": "master",
    }

    async def fail_completion(executor_id):
        raise AssertionError(f"completion ran for {executor_id}")

    service._handle_executor_completion = fail_completion

    with caplog.at_level(logging.ERROR):
        await service.stop()

    assert executor.early_kwargs == [True]
    assert executor.closes == []
    assert events == ["early_stop", "stop", "cleanup"]
    assert "oid-working" in caplog.text
    assert "Leaving the executor RUNNING" in caplog.text
    assert connector.cancels == []

    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "ex-1")
    assert row.status == "RUNNING"
    assert row.close_type is None
    child = next(row for row in db.query(ExecutorOrder) if row.client_order_id == "oid-working")
    assert child.status == "OPEN"

    # The same leftover is what the next boot has to cancel. No executor is resumed.
    service._is_running = False
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()

    assert connector.cancels == [("BTC-USDT", "oid-working")]
    assert connector.closes == []
    assert (
        "Cancelled open child order oid-working for executor ex-1 on mexc_perpetual BTC-USDT before SYSTEM_CLEANUP" in caplog.text
    )
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "ex-1")
    assert row.status == "TERMINATED"
    assert row.close_type == "SYSTEM_CLEANUP"
    exchange = next(row for row in db.query(Order) if row.client_order_id == "oid-working")
    assert exchange.status == "OPEN"


@pytest.mark.asyncio
@pytest.mark.parametrize("already", ["status", "pending"])
async def test_graceful_stop_does_not_call_early_stop_again(already):
    events = []
    connector = _Connector(confirmed=True)
    executor = _Executor(events, unconfirmed=[])
    executor.connectors = {"mexc_perpetual": connector}
    executor.close_type = SimpleNamespace(name="POSITION_HOLD")
    if already == "status":
        executor.status = SimpleNamespace(name="SHUTTING_DOWN")
    else:
        executor._shutdown_pending = True

    service = _service()
    service.SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = {"connector_name": "mexc_perpetual"}
    _track_completion(service, events)

    await service.stop()

    assert executor.early_kwargs == []
    assert "early_stop" not in events
    assert events[-1] == "cleanup"
    assert ("completion", "POSITION_HOLD") in events


@pytest.mark.asyncio
async def test_empty_unconfirmed_set_does_not_wait_on_custom_info():
    """Range iceberg always owns this set. Empty means that executor has nothing left to confirm."""
    events = []
    connector = _Connector(confirmed=False)
    executor = _Executor(
        events,
        custom_info={"child_order_ids": ["still-open"]},
        unconfirmed=[],
    )
    executor.connectors = {"mexc_perpetual": connector}
    connector.in_flight_orders["still-open"] = _open_order()

    service = _service()
    service.SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = {"connector_name": "mexc_perpetual"}
    _track_completion(service, events)

    await service.stop()

    assert ("completion", "POSITION_HOLD") in events
    assert "Leaving the executor RUNNING" not in "".join(str(item) for item in events)


@pytest.mark.asyncio
async def test_graceful_stop_waits_until_the_executor_closes_without_a_checker():
    events = []

    class _NoChecker:
        def __init__(self):
            self.in_flight_orders = {}
            self.closes = []

        async def buy(self, *args, **kwargs):
            self.closes.append("buy")

    connector = _NoChecker()
    executor = _Executor(events, custom_info={"order_ids": ["oid-1"]})
    executor.connectors = {"mexc_perpetual": connector}

    service = _service()
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = {"connector_name": "mexc_perpetual"}
    _track_completion(service, events)

    async def close_later():
        await asyncio.sleep(0.05)
        executor.is_closed = True
        events.append("closed")

    task = asyncio.create_task(close_later())
    await service.stop()
    await task

    assert executor.early_kwargs == [True]
    assert connector.closes == []
    assert events == ["early_stop", "closed", ("completion", "POSITION_HOLD"), "cleanup"]


@pytest.mark.asyncio
async def test_early_stop_falls_back_when_keep_position_is_unsupported():
    events = []

    class _Legacy:
        def __init__(self):
            self.is_closed = False
            self.status = SimpleNamespace(name="RUNNING")
            self.close_type = SimpleNamespace(name="POSITION_HOLD")
            self.connectors = {}
            self.calls = []

        def early_stop(self):
            self.calls.append("plain")
            events.append("early_stop")
            self.is_closed = True

        def stop(self):
            events.append("stop")

        def get_custom_info(self):
            return {}

    executor = _Legacy()
    service = _service()
    service.SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["ex-1"] = executor
    _track_completion(service, events)

    await service.stop()

    assert executor.calls == ["plain"]
    assert ("completion", "POSITION_HOLD") in events


@pytest.mark.asyncio
async def test_startup_cancels_only_child_orders_still_open(db, caplog):
    custom = {
        "child_order_ids": ["open-book", "filled-level", "book-only", "unknown-id", "skipped"],
        "levels": [
            {"order_id": "filled-level", "filled": True},
            {"order_id": "open-book", "filled": False},
        ],
        "held_position_orders": [{"client_order_id": "held-1"}],
    }
    await _seed_executor(
        db,
        "orphan-1",
        "master",
        "mexc_perpetual",
        "BTC-USDT",
        custom,
        [
            ("open-book", "OPEN"),
            ("filled-level", "OPEN"),
            ("extra-open", "OPEN"),
            ("stale-open", "OPEN"),
            ("already-filled", "FILLED"),
            ("held-1", "OPEN"),
            ("already-cancelled", "OPEN"),
        ],
    )
    await _seed_executor(
        db,
        "no-conn",
        "other",
        "mexc_perpetual",
        "ETH-USDT",
        {"child_order_ids": ["no-conn-order"]},
        [("no-conn-order", "OPEN")],
    )
    await _seed_executor(
        db,
        "live-1",
        "master",
        "mexc_perpetual",
        "BTC-USDT",
        {"child_order_ids": ["live-order"]},
        [("live-order", "OPEN")],
    )
    for client_order_id, status in (
        ("open-book", "OPEN"),
        ("extra-open", "OPEN"),
        ("stale-open", "FILLED"),
        ("filled-level", "OPEN"),
        ("no-conn-order", "OPEN"),
        ("live-order", "OPEN"),
        ("already-cancelled", "OPEN"),
    ):
        account = "other" if client_order_id == "no-conn-order" else "master"
        pair = "ETH-USDT" if client_order_id == "no-conn-order" else "BTC-USDT"
        await _seed_exchange_order(db, client_order_id, status, account=account, pair=pair)

    connector = _Connector()
    connector.in_flight_orders.update(
        {
            "open-book": _open_order(),
            "book-only": _open_order(),
            "filled-level": _open_order(),
            "live-order": _open_order(),
            "already-cancelled": _cancelled_order(),
        }
    )
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    service._active_executors["live-1"] = _Executor([])

    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()

    assert set(connector.cancels) == {
        ("BTC-USDT", "open-book"),
        ("BTC-USDT", "book-only"),
        ("BTC-USDT", "extra-open"),
    }
    assert connector.closes == []
    assert (
        "Cancelled open child order open-book for executor orphan-1 on mexc_perpetual BTC-USDT before SYSTEM_CLEANUP"
        in caplog.text
    )
    assert (
        "Cancelled open child order book-only for executor orphan-1 on mexc_perpetual BTC-USDT before SYSTEM_CLEANUP"
        in caplog.text
    )
    assert (
        "Cancelled open child order extra-open for executor orphan-1 on mexc_perpetual BTC-USDT before SYSTEM_CLEANUP"
        in caplog.text
    )
    assert "Failed to cancel open child order no-conn-order for executor no-conn on mexc_perpetual ETH-USDT" in caplog.text
    assert "live-order" not in {order_id for _pair, order_id in connector.cancels}

    rows = {row.executor_id: row for row in db.query(ExecutorRecord)}
    assert rows["orphan-1"].status == "TERMINATED"
    assert rows["orphan-1"].close_type == "SYSTEM_CLEANUP"
    assert rows["no-conn"].status == "TERMINATED"
    assert rows["no-conn"].close_type == "SYSTEM_CLEANUP"
    assert rows["live-1"].status == "RUNNING"
    assert rows["live-1"].close_type is None
    assert "live-1" in service._active_executors

    exchange = {row.client_order_id: row.status for row in db.query(Order)}
    assert exchange["open-book"] == "OPEN"
    assert exchange["extra-open"] == "OPEN"
    assert exchange["filled-level"] == "OPEN"


@pytest.mark.asyncio
async def test_running_child_orders_are_persisted_for_the_next_startup(db, caplog):
    await _seed_executor(db, "ex-1", "master", "mexc_perpetual", "BTC-USDT", None, [])
    service = _service(db)

    class Side(Enum):
        SELL = 2

    executor = _Executor(
        [],
        custom_info={
            "child_order_ids": ["keep-1", "drop-1"],
            "side": Side.SELL,
        },
    )
    await service._persist_running_child_orders("ex-1", executor)
    await service._persist_running_child_orders("ex-1", executor)

    children = [row for row in db.query(ExecutorOrder) if row.executor_id == "ex-1"]
    assert sorted(row.client_order_id for row in children) == ["drop-1", "keep-1"]
    assert {row.status for row in children} == {"OPEN"}
    assert {row.trade_type for row in children} == {"SELL"}
    stored = json.loads(next(row.final_state for row in db.query(ExecutorRecord) if row.executor_id == "ex-1"))
    assert stored["child_order_ids"] == ["keep-1", "drop-1"]

    executor.custom_info = RuntimeError("custom_info blew up")
    await service._persist_running_child_orders("ex-1", executor)
    assert {row.status for row in db.query(ExecutorOrder) if row.executor_id == "ex-1"} == {"OPEN"}

    executor.custom_info = {"child_order_ids": ["keep-1"], "side": "SELL"}
    await service._persist_running_child_orders("ex-1", executor)
    statuses = {row.client_order_id: row.status for row in db.query(ExecutorOrder)}
    assert statuses["keep-1"] == "OPEN"
    assert statuses["drop-1"] == "DONE"

    connector = _Connector()
    connector.in_flight_orders["keep-1"] = _open_order()
    await _seed_exchange_order(db, "keep-1", "OPEN")
    service._trading_service.connector_service = SimpleNamespace(
        get_account_connectors=lambda account: {"mexc_perpetual": connector} if account == "master" else {}
    )
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()

    assert connector.cancels == [("BTC-USDT", "keep-1")]
    assert connector.closes == []
    assert "drop-1" not in {order_id for _pair, order_id in connector.cancels}
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "ex-1")
    assert row.status == "TERMINATED"
    assert row.close_type == "SYSTEM_CLEANUP"


@pytest.mark.asyncio
async def test_control_loop_persists_child_orders_while_the_executor_is_open(db):
    await _seed_executor(db, "ex-1", "master", "mexc_perpetual", "BTC-USDT", None, [])
    service = _service(db)
    service.update_interval = 0
    service.performance_snapshot_interval = 10**9
    service.performance_retention_days = 0
    service._last_snapshot_at = 10**12
    executor = _Executor([], custom_info={"child_order_ids": ["loop-1"], "side": "BUY"})
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = {"executor_type": "range_iceberg_executor"}

    def stop_after_this_tick():
        service._is_running = False

    service._trading_service.update_all_timestamps = stop_after_this_tick
    await service._control_loop()

    children = [row for row in db.query(ExecutorOrder) if row.executor_id == "ex-1"]
    assert [row.client_order_id for row in children] == ["loop-1"]
    assert children[0].status == "OPEN"


def test_exit_order_ids_join_the_working_set():
    working, filled = child_order_ids_from_custom_info(
        {
            "child_order_ids": ["entry-1"],
            "exit_order_ids": ["tp-1", "sl-1", "entry-1"],
            "levels": [{"order_id": "sl-1", "filled": True}],
        }
    )
    assert working == ["entry-1", "tp-1"]
    assert filled == ["sl-1"]


def test_range_iceberg_held_base_prefers_open_base():
    assert range_iceberg_held_base(None, {"open_base": "1.25", "filled_base": "9"}) == Decimal("1.25")
    assert range_iceberg_held_base(None, {"filled_base": "2", "closed_base": "0.5"}) == Decimal("1.5")
    assert range_iceberg_held_base(None, {"filled_base": "2"}) == Decimal("2")
    assert range_iceberg_held_base(None, {"open_base": "0", "filled_base": "5"}) == Decimal("0")
    assert range_iceberg_held_base(None, {"filled_amount_base": "4"}) is None
    assert range_iceberg_held_base(None, {}) is None
    assert range_iceberg_held_base(
        None, {"open_base": "5", "exchange_position_base": "0.4"},
    ) == Decimal("0.4")
    assert range_iceberg_held_base(
        None,
        {"open_base": "5", "exchange_position_base": "0", "external_close": True},
    ) == Decimal("0")
    # The close type is enough. A false flag and a stale book must not keep base.
    closed = SimpleNamespace(close_type=SimpleNamespace(name="POSITION_CLOSED_EXTERNALLY"))
    assert range_iceberg_held_base(
        closed,
        {
            "open_base": "5",
            "exchange_position_base": "5",
            "external_close": False,
            "reconcile_error": "position read failed: timeout",
        },
    ) == Decimal("0")
    # A failed read is not a close. The last reported book still counts.
    assert range_iceberg_held_base(
        None,
        {
            "open_base": "5",
            "exchange_position_base": "4",
            "external_close": False,
            "reconcile_error": "position read failed: timeout",
        },
    ) == Decimal("4")
    assert range_iceberg_held_base(
        None,
        {"open_base": "5", "external_close": False, "reconcile_error": "position read failed: timeout"},
    ) == Decimal("5")


class _CancellingExecutor(_Executor):
    """early_stop cancels software entry and exit orders and leaves the exchange stop."""

    def early_stop(self, keep_position=True):
        super().early_stop(keep_position=keep_position)
        connector = next(iter(self.connectors.values()), None)
        if connector is None:
            return
        info = self.get_custom_info()
        stop_id = info.get("exchange_stop_order_id")
        seen = set()
        for oid in list(info.get("child_order_ids") or []) + list(info.get("exit_order_ids") or []):
            if oid and oid != stop_id and oid not in seen:
                seen.add(oid)
                connector.cancels.append(("BTC-USDT", oid))


def _iceberg_custom(exit_state, **extra):
    info = {
        "child_order_ids": ["entry-1"],
        "exit_order_ids": ["tp-1"],
        "exchange_stop_order_id": "stop-1",
        "open_base": "1.5",
        "filled_base": "1.5",
        "closed_base": "0",
        "side": "BUY",
    }
    if exit_state is not None:
        info["exit_state"] = exit_state
    info.update(extra)
    return info


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_state,park",
    [
        ("ARMED", True),
        ("RESTING", True),
        ("CLOSING", True),
        ("resting", True),
        ("TRIGGERED", False),
        ("DONE", False),
        ("NONE", False),
        (None, False),
    ],
)
async def test_graceful_shutdown_persists_resumable_icebergs(db, caplog, exit_state, park):
    events = []
    connector = _Connector(confirmed=True)
    info = _iceberg_custom(exit_state)
    executor = _CancellingExecutor(events, custom_info=info, unconfirmed=[])
    executor.connectors = {"mexc_perpetual": connector}
    connector.in_flight_orders["stop-1"] = _open_order()
    connector.in_flight_orders["tp-1"] = _open_order()

    await _seed_executor(db, "ex-1", "master", "mexc_perpetual", "BTC-USDT", None, [])
    service = _service(db)
    service.SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = {
        "connector_name": "mexc_perpetual",
        "trading_pair": "BTC-USDT",
        "account_name": "master",
        "executor_type": "range_iceberg_executor",
    }
    _track_completion(service, events)

    with caplog.at_level(logging.INFO):
        await service.stop()

    stored = json.loads(next(row.final_state for row in db.query(ExecutorRecord) if row.executor_id == "ex-1"))
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "ex-1")
    if park:
        assert executor.early_kwargs == [True]
        assert ("BTC-USDT", "entry-1") in connector.cancels
        assert ("BTC-USDT", "tp-1") in connector.cancels
        assert ("BTC-USDT", "stop-1") not in connector.cancels
        assert "early_stop" in events
        assert ("completion", "POSITION_HOLD") not in events
        assert stored["resume_on_startup"] is True
        assert stored["exit_state"] == exit_state
        assert stored["exchange_stop_order_id"] == "stop-1"
        assert stored["exit_order_ids"] == ["tp-1"]
        assert row.status == "RUNNING"
        assert "Persisted range iceberg executor ex-1" in caplog.text
        assert "stop-1" in caplog.text
        assert "ex-1" not in service._active_executors
        exit_rows = [row for row in db.query(ExecutorOrder) if row.client_order_id == "tp-1"]
        assert exit_rows and exit_rows[0].status == "OPEN"
    else:
        assert executor.early_kwargs == [True]
        assert ("BTC-USDT", "entry-1") in connector.cancels
        assert ("BTC-USDT", "tp-1") in connector.cancels
        assert ("BTC-USDT", "stop-1") not in connector.cancels
        assert stored.get("resume_on_startup") is not True
        assert ("completion", "POSITION_HOLD") in events


class _ResumeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _ResumeExecutor:
    def __init__(self, strategy, config, update_interval, max_retries):
        self.strategy = strategy
        self.config = config
        self.is_closed = False
        self.status = SimpleNamespace(name="RUNNING")
        self.close_type = None
        self.restored = None
        self.started = False

    def restore_from_custom_info(self, info):
        self.restored = info

    def start(self):
        self.started = True

    def stop(self):
        pass

    def get_custom_info(self):
        return {}


class _BrokenResumeExecutor(_ResumeExecutor):
    def restore_from_custom_info(self, info):
        raise RuntimeError("restore blew up")


def _install_resume_factory(service, executor_cls=_ResumeExecutor):
    service.EXECUTOR_REGISTRY = dict(ExecutorService.EXECUTOR_REGISTRY)
    service.EXECUTOR_REGISTRY["range_iceberg_executor"] = (executor_cls, _ResumeConfig)
    service._get_trading_interface = lambda account: SimpleNamespace(current_timestamp=1.0)

    async def _prepare(*_args, **_kwargs):
        return None

    service._prepare_market = _prepare


async def _seed_iceberg_row(db, executor_id, final_state, config=None):
    async with db.session_context() as session:
        repo = ExecutorRepository(session)
        await repo.create_executor(
            executor_id=executor_id,
            executor_type="range_iceberg_executor",
            account_name="master",
            connector_name="mexc_perpetual",
            trading_pair="BTC-USDT",
            config=json.dumps(config or {
                "type": "range_iceberg_executor",
                "connector_name": "mexc_perpetual",
                "trading_pair": "BTC-USDT",
            }),
        )
        await repo.update_executor(executor_id, final_state=json.dumps(final_state))


@pytest.mark.asyncio
async def test_startup_resumes_saved_iceberg_and_cancels_the_rest(db, caplog):
    await _seed_iceberg_row(
        db,
        "resume-1",
        {
            "resume_on_startup": True,
            "exit_state": "RESTING",
            "side": "BUY",
            "child_order_ids": ["entry-1"],
            "exit_order_ids": ["tp-child"],
            "exchange_stop_order_id": "stop-1",
            "open_base": "1.5",
            "filled_base": "1.5",
            "closed_base": "0",
        },
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "BUY",
            "protective_exchange_stop": True,
            "stop_loss_fallback_seconds": 30,
            "allow_adverse_take_profit": False,
            "protective_stop_buffer_pct": "0.1",
            "exchange_stop_trigger_price": "100",
        },
    )
    await _seed_iceberg_row(
        db,
        "crash-1",
        {
            "exit_state": "RESTING",
            "child_order_ids": ["crash-entry"],
            "exit_order_ids": ["crash-tp"],
            "exchange_stop_order_id": "crash-stop",
            "open_base": "2",
        },
    )
    await _seed_iceberg_row(
        db,
        "closed-1",
        {
            "exit_state": "DONE",
            "child_order_ids": [],
            "exit_order_ids": [],
            "exchange_stop_order_id": "closed-stop",
            "open_base": "0",
            "filled_base": "1",
            "closed_base": "1",
        },
    )
    await _seed_iceberg_row(
        db,
        "open-stop-1",
        {
            "child_order_ids": ["naked-stop"],
            "exchange_stop_order_id": "naked-stop",
            "open_base": "0.4",
        },
    )
    connector = _Connector()
    connector.account_positions = {
        "BTC-USDT:LONG": SimpleNamespace(
            trading_pair="BTC-USDT",
            position_side=SimpleNamespace(name="LONG"),
            amount=Decimal("1.5"),
        )
    }
    for order_id in ("entry-1", "tp-child", "stop-1", "crash-entry", "crash-tp", "crash-stop", "closed-stop", "naked-stop"):
        connector.in_flight_orders[order_id] = _open_order()
        await _seed_exchange_order(db, order_id, "OPEN")

    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)

    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()

    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "entry-1" in cancelled
    assert "tp-child" in cancelled
    assert "stop-1" not in cancelled
    assert "crash-tp" in cancelled
    assert "crash-entry" in cancelled
    assert "crash-stop" not in cancelled
    assert "closed-stop" not in cancelled
    assert "closed-stop" in connector.stop_cancels
    assert "stop-1" not in connector.stop_cancels
    assert "crash-stop" not in connector.stop_cancels
    assert "naked-stop" not in cancelled
    assert (
        "Cancelled exchange stop closed-stop for executor closed-1 "
        "on mexc_perpetual BTC-USDT because open_base is 0"
    ) in caplog.text
    assert "Resumed range iceberg executors before orphan cleanup: resume-1" in caplog.text

    resumed = service._active_executors["resume-1"]
    assert resumed.started is True
    assert resumed.restored["exit_state"] == "RESTING"
    assert resumed.restored["resume_entries"] is False
    assert resumed.restored["exit_order_ids"] == []
    assert resumed.restored["child_order_ids"] == []
    assert resumed.restored["open_base"] == "1.5"
    assert "resume_on_startup" not in resumed.restored
    assert resumed.config.id == "resume-1"
    assert resumed.config.total_amount == "1.5"
    assert resumed.config.protective_exchange_stop is True
    assert resumed.config.stop_loss_fallback_seconds == 30
    assert resumed.config.allow_adverse_take_profit is False
    assert resumed.config.protective_stop_buffer_pct == "0.1"
    assert resumed.config.exchange_stop_trigger_price == "100"

    rows = {row.executor_id: row for row in db.query(ExecutorRecord)}
    assert rows["resume-1"].status == "RUNNING"
    assert rows["crash-1"].status == "TERMINATED"
    assert rows["crash-1"].close_type == "SYSTEM_CLEANUP"
    assert rows["closed-1"].close_type == "SYSTEM_CLEANUP"
    assert rows["open-stop-1"].close_type == "SYSTEM_CLEANUP"


@pytest.mark.asyncio
async def test_failed_resume_leaves_the_exchange_stop_and_marks_the_row(db, caplog):
    await _seed_iceberg_row(
        db,
        "fail-1",
        {
            "resume_on_startup": True,
            "exit_state": "ARMED",
            "exit_order_ids": ["tp-fail"],
            "side": "BUY",
            "exchange_stop_order_id": "stop-fail",
            "open_base": "1",
        },
    )
    connector = _Connector()
    connector.account_positions = {
        "BTC-USDT:LONG": SimpleNamespace(
            trading_pair="BTC-USDT",
            position_side=SimpleNamespace(name="LONG"),
            amount=Decimal("1"),
        )
    }
    connector.in_flight_orders["tp-fail"] = _open_order()
    connector.in_flight_orders["stop-fail"] = _open_order()
    await _seed_exchange_order(db, "tp-fail", "OPEN")
    await _seed_exchange_order(db, "stop-fail", "OPEN")

    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service, _BrokenResumeExecutor)

    with caplog.at_level(logging.ERROR):
        await service.cleanup_orphaned_executors()

    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "stop-fail" not in cancelled
    assert "tp-fail" in cancelled
    assert "Failed to resume range iceberg executor fail-1" in caplog.text
    assert "stop-fail" in caplog.text
    assert "Left exchange stop" in caplog.text

    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "fail-1")
    assert row.status == "TERMINATED"
    assert row.close_type == "SYSTEM_CLEANUP"
    stored = json.loads(row.final_state)
    assert stored["resume_failed"] is True
    assert "restore blew up" in stored["resume_error"]
    assert stored["exchange_stop_order_id"] == "stop-fail"
    errors = json.loads(row.error_log)
    assert "Left exchange stop stop-fail" in errors[-1]["message"]
    assert "fail-1" not in service._active_executors


class _HoldExecutor:
    def __init__(self, info, filled_amount_base=None):
        self._info = info
        self.cum_fees_quote = Decimal("0")
        if filled_amount_base is not None:
            self.filled_amount_base = filled_amount_base

    @property
    def executor_info(self):
        raise RuntimeError("no executor_info")

    def get_custom_info(self):
        return self._info


def _hold_metadata(executor_type):
    return {
        "account_name": "master",
        "connector_name": "mexc_perpetual",
        "trading_pair": "BTC-USDT",
        "controller_id": "main",
        "executor_type": executor_type,
        "config": {"side": "BUY"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "executor_type,info,attribute,expected",
    [
        ("range_iceberg_executor", {"open_base": "1.25", "filled_base": "9", "side": "BUY"}, None, Decimal("1.25")),
        ("range_iceberg_executor", {"filled_base": "2", "closed_base": "0.5", "side": "BUY"}, None, Decimal("1.5")),
        ("range_iceberg_executor", {"filled_base": "2", "side": "BUY"}, None, Decimal("2")),
        ("range_iceberg_executor", {"open_base": "0", "filled_base": "5", "side": "BUY"}, Decimal("5"), Decimal("0")),
        ("position_executor", {"filled_base": "4", "side": "BUY"}, Decimal("3"), Decimal("3")),
        ("position_executor", {"filled_base": "4", "side": "BUY"}, None, Decimal("0")),
    ],
)
async def test_keep_position_uses_iceberg_base_fields(executor_type, info, attribute, expected):
    service = _service()
    metadata = _hold_metadata(executor_type)
    executor = _HoldExecutor(info, filled_amount_base=attribute)
    await service._aggregate_position_hold("ex-1", executor, metadata)
    position = next(iter(service._positions_held.values()))
    assert position.buy_amount_base == expected
    if expected > 0:
        assert position.executor_ids == ["ex-1"]
    else:
        assert position.executor_ids == []


def test_search_record_keeps_range_iceberg_custom_info_fields():
    service = _service()
    payload = {
        "exit_state": "RESTING",
        "exit_order_ids": ["tp-1", "sl-child"],
        "exchange_stop_order_id": "stop-1",
        "filled_base": "1.5",
        "closed_base": "0.25",
        "open_base": "1.25",
        "exchange_position_base": "0",
        "external_close": True,
        "reconcile_error": "position read failed: timeout",
        "group_id": "aiot-long",
        "sibling_base": None,
        "entry_paused": False,
        "pause_reason": None,
    }
    record = SimpleNamespace(
        executor_id="ex-1",
        executor_type="range_iceberg_executor",
        account_name="master",
        connector_name="mexc_perpetual",
        trading_pair="BTC-USDT",
        status="RUNNING",
        close_type=None,
        created_at=None,
        closed_at=None,
        controller_id="main",
        net_pnl_quote=0,
        net_pnl_pct=0,
        cum_fees_quote=0,
        filled_amount_quote=0,
        config=json.dumps({"exchange_stop": True, "exit_retry_budget": 4}),
        final_state=json.dumps(payload),
        error_log=None,
    )
    formatted = service._format_db_record(record)
    for key, value in payload.items():
        assert formatted["custom_info"][key] == value


def test_validate_forwards_protective_stop_fields():
    service = _service()

    class _Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    service.EXECUTOR_REGISTRY = dict(ExecutorService.EXECUTOR_REGISTRY)
    service.EXECUTOR_REGISTRY["range_iceberg_executor"] = (object, _Config)
    _executor_class, _config_class, typed = service._validate_executor_config(
        {
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "protective_exchange_stop": True,
            "stop_loss_fallback_seconds": 30,
            "allow_adverse_take_profit": False,
            "protective_stop_buffer_pct": "0.1",
            "exchange_stop_trigger_price": "100",
            "timestamp": 10,
        }
    )
    assert typed.protective_exchange_stop is True
    assert typed.stop_loss_fallback_seconds == 30
    assert typed.allow_adverse_take_profit is False
    assert typed.protective_stop_buffer_pct == "0.1"
    assert typed.exchange_stop_trigger_price == "100"
    assert typed.connector_name == "mexc_perpetual"


def _long_position(amount):
    return {
        "BTC-USDT:LONG": SimpleNamespace(
            trading_pair="BTC-USDT",
            position_side=SimpleNamespace(name="LONG"),
            amount=Decimal(str(amount)),
        )
    }


async def _resume_with_position(db, executor_id, final_state, amount, caplog):
    await _seed_iceberg_row(db, executor_id, final_state)
    connector = _Connector()
    connector.account_positions = _long_position(amount)
    for order_id in ("entry-x", "tp-x", "stop-x"):
        connector.in_flight_orders[order_id] = _open_order()
        await _seed_exchange_order(db, order_id, "OPEN")
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    return service, connector


@pytest.mark.asyncio
async def test_flat_exchange_position_is_not_resumed_and_its_stop_is_cancelled(db, caplog):
    final_state = {
        "resume_on_startup": True,
        "exit_state": "RESTING",
        "side": "BUY",
        "child_order_ids": ["entry-x"],
        "exit_order_ids": ["tp-x"],
        "exchange_stop_order_id": "stop-x",
        "open_base": "1.5",
    }
    service, connector = await _resume_with_position(db, "flat-1", final_state, "0", caplog)
    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "entry-x" in cancelled
    assert "tp-x" in cancelled
    assert "stop-x" not in cancelled
    assert "stop-x" in connector.stop_cancels
    assert "flat-1" not in service._active_executors
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "flat-1")
    assert row.status == "TERMINATED"
    assert row.close_type == "CLOSED_WHILE_DOWN"
    stored = json.loads(row.final_state)
    assert stored["closed_while_down"] is True
    assert "flat" in caplog.text
    assert "was flat while the API was down" in caplog.text


@pytest.mark.asyncio
async def test_resume_uses_the_smaller_exchange_position(db, caplog):
    final_state = {
        "resume_on_startup": True,
        "exit_state": "ARMED",
        "side": "BUY",
        "child_order_ids": ["entry-x"],
        "exit_order_ids": ["tp-x"],
        "exchange_stop_order_id": "stop-x",
        "open_base": "2",
        "filled_base": "2",
        "closed_base": "0",
    }
    service, connector = await _resume_with_position(db, "small-1", final_state, "0.5", caplog)
    resumed = service._active_executors["small-1"]
    assert resumed.restored["open_base"] == "0.5"
    assert resumed.restored["resume_entries"] is False
    assert resumed.restored["exit_order_ids"] == []
    assert resumed.config.total_amount == "0.5"
    assert "stop-x" not in {order_id for _pair, order_id in connector.cancels}
    assert "entry-x" in {order_id for _pair, order_id in connector.cancels}
    assert "Allocated 0.5 of exchange position 0.5" in caplog.text


@pytest.mark.asyncio
async def test_resume_caps_at_saved_open_base_when_the_exchange_is_larger(db, caplog):
    final_state = {
        "resume_on_startup": True,
        "exit_state": "CLOSING",
        "side": "BUY",
        "exit_order_ids": ["tp-x"],
        "exchange_stop_order_id": "stop-x",
        "open_base": "1",
    }
    service, connector = await _resume_with_position(db, "large-1", final_state, "4", caplog)
    resumed = service._active_executors["large-1"]
    assert resumed.restored["open_base"] == "1"
    assert resumed.config.total_amount == "1"
    assert "stop-x" not in {order_id for _pair, order_id in connector.cancels}
    assert "larger than saved open_base" in caplog.text


@pytest.mark.asyncio
async def test_unreadable_exchange_position_leaves_the_stop(db, caplog):
    await _seed_iceberg_row(
        db,
        "blind-1",
        {
            "resume_on_startup": True,
            "exit_state": "RESTING",
            "side": "BUY",
            "exchange_stop_order_id": "stop-blind",
            "exit_order_ids": ["tp-blind"],
            "open_base": "1",
        },
    )
    connector = _Connector()
    connector.in_flight_orders["stop-blind"] = _open_order()
    connector.in_flight_orders["tp-blind"] = _open_order()
    await _seed_exchange_order(db, "stop-blind", "OPEN")
    await _seed_exchange_order(db, "tp-blind", "OPEN")
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    with caplog.at_level(logging.ERROR):
        await service.cleanup_orphaned_executors()
    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "stop-blind" not in cancelled
    assert "tp-blind" in cancelled
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "blind-1")
    stored = json.loads(row.final_state)
    assert stored["resume_failed"] is True
    assert "could not read the exchange position" in stored["resume_error"]
    assert "blind-1" not in service._active_executors


@pytest.mark.asyncio
async def test_stop_without_keep_position_cancels_the_exchange_stop():
    connector = _Connector()
    connector.in_flight_orders["stop-1"] = _open_order()
    executor = _Executor([], custom_info={
        "exchange_stop_order_id": "stop-1",
        "exit_order_ids": ["tp-1"],
        "open_base": "1",
        "side": "BUY",
    })
    service = _service(connectors={"master": {"mexc_perpetual": connector}})
    metadata = {
        "account_name": "master",
        "connector_name": "mexc_perpetual",
        "trading_pair": "BTC-USDT",
        "executor_type": "range_iceberg_executor",
    }
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = metadata
    await service.stop_executor("ex-1", keep_position=False)
    assert executor.early_kwargs == [False]
    assert connector.cancels == []
    assert connector.stop_cancels == ["stop-1"]

    connector.cancels.clear()
    kept = _Executor([], custom_info={"exchange_stop_order_id": "stop-1", "open_base": "1"})
    service._active_executors["ex-2"] = kept
    service._executor_metadata["ex-2"] = metadata
    await service.stop_executor("ex-2", keep_position=True)
    assert kept.early_kwargs == [True]
    assert connector.cancels == []


@pytest.mark.asyncio
async def test_startup_cancels_exchange_stop_on_a_finished_executor(db, caplog):
    await _seed_iceberg_row(
        db,
        "failed-stop",
        {
            "exit_state": "DONE",
            "exchange_stop_order_id": "stop-failed",
            "open_base": "1",
            "side": "BUY",
        },
    )
    await _seed_iceberg_row(
        db,
        "held-stop",
        {
            "exit_state": "RESTING",
            "exchange_stop_order_id": "stop-held",
            "open_base": "1",
            "side": "BUY",
        },
    )
    async with db.session_context() as session:
        repo = ExecutorRepository(session)
        await repo.update_executor("failed-stop", status="TERMINATED", close_type="FAILED")
        await repo.update_executor("held-stop", status="TERMINATED", close_type="POSITION_HOLD")
    connector = _Connector()
    connector.in_flight_orders["stop-failed"] = _open_order()
    connector.in_flight_orders["stop-held"] = _open_order()
    await _seed_exchange_order(db, "stop-failed", "OPEN")
    await _seed_exchange_order(db, "stop-held", "OPEN")
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "stop-failed" not in cancelled
    assert connector.stop_cancels == ["stop-failed"]
    assert "stop-held" not in cancelled
    assert "stop-held" not in connector.stop_cancels
    assert "not RUNNING and was not parked for resume" in caplog.text


@pytest.mark.asyncio
async def test_keep_position_quote_matches_open_base():
    service = _service()
    info = {
        "open_base": "1",
        "filled_base": "4",
        "filled_quote": "400",
        "average_fill_price": "90",
        "side": "BUY",
    }
    executor = _HoldExecutor(info, filled_amount_base=Decimal("4"))
    executor.filled_amount_quote = Decimal("400")
    await service._aggregate_position_hold("ex-1", executor, _hold_metadata("range_iceberg_executor"))
    position = next(iter(service._positions_held.values()))
    assert position.buy_amount_base == Decimal("1")
    assert position.buy_amount_quote == Decimal("90")
    assert position.executor_ids == ["ex-1"]


@pytest.mark.asyncio
async def test_quote_sized_executor_resumes_in_base(db, caplog):
    """A QUOTE total_amount must not be reused as the open base."""
    await _seed_iceberg_row(
        db,
        "quote-1",
        {
            "resume_on_startup": True,
            "exit_state": "ARMED",
            "side": "BUY",
            "open_base": "50",
            "filled_base": "50",
            "filled_quote": "8.5",
            "exchange_stop_order_id": "stop-x",
            "total_amount": "8.5",
            "target_amount": "8.5",
            "price_low": "0.1",
            "price_high": "0.2",
            "take_profit": {"percent": "0.01"},
            "stop_loss": {"percent": "0.02"},
        },
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "BUY",
            "size_unit": "QUOTE",
            "total_amount": "8.5",
        },
    )
    connector = _Connector()
    connector.account_positions = {
        "BTC-USDT:LONG": SimpleNamespace(
            trading_pair="BTC-USDT",
            position_side=SimpleNamespace(name="LONG"),
            amount=Decimal("50"),
            entry_price=Decimal("0.17"),
        )
    }
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    resumed = service._active_executors["quote-1"]
    assert resumed.config.size_unit == "BASE"
    assert resumed.config.total_amount == "50"
    assert getattr(resumed.config, "price_low", None) is None
    assert resumed.config.resume_entries is False
    assert resumed.restored.get("total_amount") is None
    assert resumed.restored["size_unit"] == "BASE"
    assert resumed.restored["resume_entries"] is False
    assert Decimal(resumed.restored["open_base"]) == Decimal("50")
    assert Decimal(resumed.restored["filled_quote"]) == Decimal("8.5")
    stored_config = json.loads(
        next(row for row in db.query(ExecutorRecord) if row.executor_id == "quote-1").config
    )
    assert stored_config["size_unit"] == "BASE"
    assert stored_config["total_amount"] == "50"


@pytest.mark.asyncio
async def test_partial_take_profit_resumes_on_the_entry_average(db, caplog):
    await _seed_iceberg_row(
        db,
        "tp-1",
        {
            "resume_on_startup": True,
            "exit_state": "RESTING",
            "side": "BUY",
            "open_base": "20",
            "filled_base": "60",
            "closed_base": "40",
            "filled_quote": "5.1",
            "average_fill_price": "0.085",
            "exchange_stop_order_id": "stop-x",
            "exit_order_ids": ["tp-x"],
            "exit": {
                "active": "take_profit",
                "status": "resting",
                "entry_filled_base": "60",
                "entry_filled_quote": "5.1",
                "filled_base": "40",
                "filled_quote": "5.668",
                "filled_by_order": {"tp-x": {"base": "40", "quote": "5.668"}},
                "order_ids": ["tp-x"],
            },
        },
    )
    connector = _Connector()
    connector.account_positions = {
        "BTC-USDT:LONG": SimpleNamespace(
            trading_pair="BTC-USDT",
            position_side=SimpleNamespace(name="LONG"),
            amount=Decimal("20"),
            entry_price=Decimal("0.085"),
        )
    }
    connector.in_flight_orders["tp-x"] = _open_order()
    await _seed_exchange_order(db, "tp-x", "OPEN")
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    resumed = service._active_executors["tp-1"]
    assert Decimal(resumed.restored["filled_base"]) == Decimal("20")
    assert Decimal(resumed.restored["closed_base"]) == Decimal("0")
    assert Decimal(resumed.restored["open_base"]) == Decimal("20")
    assert Decimal(resumed.restored["average_fill_price"]) == Decimal("0.085")
    assert Decimal(resumed.restored["filled_quote"]) == Decimal("1.7")
    assert resumed.restored["exit"]["filled_by_order"] == {}
    assert Decimal(resumed.restored["exit"]["entry_filled_base"]) == Decimal("20")
    assert Decimal(resumed.restored["exit"]["entry_filled_quote"]) == Decimal("1.7")
    assert Decimal(resumed.restored["exit"]["filled_base"]) == Decimal("0")
    assert Decimal(resumed.restored["exit"]["filled_quote"]) == Decimal("0")
    assert resumed.restored["exit_order_ids"] == []
    assert resumed.config.size_unit == "BASE"
    assert resumed.config.resume_entries is False
    # 5.668 / 40 is the stale take-profit price. It must not become the entry.
    assert Decimal(resumed.restored["average_fill_price"]) != Decimal("0.1417")


@pytest.mark.asyncio
async def test_two_parked_executors_share_one_position(db, caplog):
    first = {
        "resume_on_startup": True,
        "exit_state": "ARMED",
        "side": "BUY",
        "open_base": "2",
        "filled_base": "2",
        "filled_quote": "2",
        "exchange_stop_order_id": "stop-x",
    }
    second = dict(first)
    second["exchange_stop_order_id"] = "stop-y"
    await _seed_iceberg_row(db, "share-1", first)
    await _seed_iceberg_row(db, "share-2", second)
    connector = _Connector()
    connector.account_positions = {
        "BTC-USDT:LONG": SimpleNamespace(
            trading_pair="BTC-USDT",
            position_side=SimpleNamespace(name="LONG"),
            amount=Decimal("3"),
            entry_price=Decimal("1"),
        )
    }
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    first_exec = service._active_executors["share-1"]
    second_exec = service._active_executors["share-2"]
    assert Decimal(first_exec.restored["open_base"]) == Decimal("2")
    assert Decimal(second_exec.restored["open_base"]) == Decimal("1")
    total = Decimal(first_exec.config.total_amount) + Decimal(second_exec.config.total_amount)
    assert total == Decimal("3")
    assert "Allocated exchange position 3" in caplog.text
    assert "share-1=2" in caplog.text
    assert "share-2=1" in caplog.text


@pytest.mark.asyncio
async def test_untracked_software_order_is_cancelled_by_exchange_id(db, caplog):
    await _seed_executor(
        db,
        "orphan-direct",
        "master",
        "mexc_perpetual",
        "BTC-USDT",
        {"child_order_ids": ["orphan-oid"], "exit_order_ids": ["exit-oid"]},
        [("orphan-oid", "OPEN"), ("exit-oid", "OPEN")],
    )
    await _seed_exchange_order(db, "orphan-oid", "OPEN")
    await _seed_exchange_order(db, "exit-oid", "OPEN")
    async with db.session_context() as session:
        repo = OrderRepository(session)
        await repo.update_order_fill("orphan-oid", Decimal("0"), Decimal("0"), exchange_order_id="9001")
        await repo.update_order_fill("exit-oid", Decimal("0"), Decimal("0"), exchange_order_id="9002")

    class _FreshConnector(_Connector):
        def __init__(self):
            super().__init__()
            self.direct = []
            self.open_orders = [
                {"externalOid": "orphan-oid", "orderId": "9001"},
                {"externalOid": "exit-oid", "orderId": "9002"},
            ]

        async def cancel(self, trading_pair=None, client_order_id=None):
            # The fresh tracker does not know these ids, so this call is a no-op.
            self.cancels.append((trading_pair, client_order_id))

        async def _place_cancel(self, order_id, tracked):
            self.direct.append((order_id, tracked.exchange_order_id))
            self.open_orders = [
                row for row in self.open_orders
                if row.get("externalOid") != order_id
                and str(row.get("orderId")) != str(tracked.exchange_order_id)
            ]
            return True

        async def list_open_orders(self, trading_pair=None):
            return list(self.open_orders)

    connector = _FreshConnector()
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    assert connector.cancels == []
    assert ("orphan-oid", "9001") in connector.direct
    assert ("exit-oid", "9002") in connector.direct
    assert connector.open_orders == []
    assert "Cancelled open child order orphan-oid" in caplog.text
    assert "Cancelled open child order exit-oid" in caplog.text


@pytest.mark.asyncio
async def test_rejected_stop_cancel_is_logged_as_a_failure(db, caplog):
    await _seed_iceberg_row(
        db,
        "bad-stop",
        {
            "exit_state": "DONE",
            "exchange_stop_order_id": "stop-bad",
            "open_base": "1",
            "side": "BUY",
        },
    )
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            "bad-stop", status="TERMINATED", close_type="FAILED"
        )
    connector = _Connector()
    connector.stop_result = {"success": False, "code": 500, "message": "rejected"}
    connector.in_flight_orders["stop-bad"] = _open_order()
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    assert connector.stop_cancels == ["stop-bad"]
    assert connector.cancels == []
    assert "Failed to cancel exchange stop stop-bad" in caplog.text
    assert "Cancelled exchange stop stop-bad" not in caplog.text
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "bad-stop")
    stored = json.loads(row.final_state)
    assert "rejected" in stored["exchange_stop_error"]
    assert stored.get("exchange_stop_cancelled") is not True


@pytest.mark.asyncio
async def test_search_exposes_resume_and_stop_errors(db):
    await _seed_iceberg_row(
        db,
        "shown-1",
        {
            "resume_failed": True,
            "resume_error": "could not read the exchange position",
            "exchange_stop_error": "stop still open",
            "exit_state": "RESTING",
        },
    )
    service = _service(db)
    rows = await service.get_executors()
    shown = next(row for row in rows if row["executor_id"] == "shown-1")
    assert shown["resume_failed"] is True
    assert shown["resume_error"] == "could not read the exchange position"
    assert shown["exchange_stop_error"] == "stop still open"


def test_exchange_stop_ids_fall_back_to_the_primary():
    assert _exchange_stop_ids({"exchange_stop_order_id": "stop-1"}) == ["stop-1"]
    assert _exchange_stop_ids({"exchange_stop_order_ids": ["stop-2", "stop-3"]}) == ["stop-2", "stop-3"]
    assert _exchange_stop_ids({
        "exchange_stop_order_id": "stop-1",
        "exchange_stop_order_ids": ["stop-2", "stop-1"],
    }) == ["stop-2", "stop-1"]
    assert _exchange_stop_ids({
        "exchange_stop_order_id": "stop-1",
        "exchange_stop_order_ids": ["stop-2"],
    }) == ["stop-2", "stop-1"]
    assert _exchange_stop_ids({}) == []


@pytest.mark.asyncio
async def test_resume_re_adopts_every_exchange_stop_id(db, caplog):
    await _seed_iceberg_row(
        db,
        "stops-1",
        {
            "resume_on_startup": True,
            "exit_state": "RESTING",
            "side": "BUY",
            "child_order_ids": ["entry-list"],
            "exchange_stop_order_id": "stop-a",
            "exchange_stop_order_ids": ["stop-b", "stop-a"],
            "open_base": "1.5",
            "filled_base": "1.5",
            "filled_quote": "1.5",
        },
    )
    connector = _Connector()
    connector.account_positions = _long_position("1.5")
    for order_id in ("entry-list", "stop-a", "stop-b"):
        connector.in_flight_orders[order_id] = _open_order()
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    resumed = service._active_executors["stops-1"]
    assert resumed.restored["exchange_stop_order_id"] == "stop-a"
    assert resumed.restored["exchange_stop_order_ids"] == ["stop-b", "stop-a"]
    assert resumed.restored["exit"]["exchange_stop_order_ids"] == ["stop-b", "stop-a"]
    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "entry-list" in cancelled
    assert "stop-a" not in cancelled
    assert "stop-b" not in cancelled
    assert connector.stop_cancels == []
    assert "Protected exchange stop stop-b, stop-a" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_persists_every_exchange_stop_id(db, caplog):
    events = []
    connector = _Connector(confirmed=True)
    info = _iceberg_custom("RESTING", exchange_stop_order_ids=["stop-1", "stop-2"])
    executor = _CancellingExecutor(events, custom_info=info, unconfirmed=[])
    executor.connectors = {"mexc_perpetual": connector}
    connector.in_flight_orders["stop-1"] = _open_order()
    connector.in_flight_orders["stop-2"] = _open_order()
    await _seed_executor(db, "park-stops", "master", "mexc_perpetual", "BTC-USDT", None, [])
    service = _service(db)
    service.SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 0
    service._trading_interfaces["master"] = _Interface(events)
    service._active_executors["park-stops"] = executor
    service._executor_metadata["park-stops"] = {
        "connector_name": "mexc_perpetual",
        "trading_pair": "BTC-USDT",
        "account_name": "master",
        "executor_type": "range_iceberg_executor",
    }
    _track_completion(service, events)
    with caplog.at_level(logging.INFO):
        await service.stop()
    stored = json.loads(next(row.final_state for row in db.query(ExecutorRecord) if row.executor_id == "park-stops"))
    assert stored["resume_on_startup"] is True
    assert stored["exchange_stop_order_id"] == "stop-1"
    assert stored["exchange_stop_order_ids"] == ["stop-1", "stop-2"]
    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "stop-1" not in cancelled
    assert "stop-2" not in cancelled
    assert connector.stop_cancels == []
    assert "stop-1, stop-2" in caplog.text


@pytest.mark.asyncio
async def test_cleanup_cancels_every_exchange_stop_id(db, caplog):
    await _seed_iceberg_row(
        db,
        "flat-list",
        {
            "resume_on_startup": True,
            "exit_state": "RESTING",
            "side": "BUY",
            "child_order_ids": ["entry-flat"],
            "exchange_stop_order_id": "stop-flat-a",
            "exchange_stop_order_ids": ["stop-flat-a", "stop-flat-b"],
            "open_base": "1",
        },
    )
    await _seed_iceberg_row(
        db,
        "zero-list",
        {
            "exit_state": "DONE",
            "exchange_stop_order_id": "stop-zero-a",
            "exchange_stop_order_ids": ["stop-zero-a", "stop-zero-b"],
            "open_base": "0",
        },
    )
    await _seed_iceberg_row(
        db,
        "done-list",
        {
            "exit_state": "DONE",
            "exchange_stop_order_ids": ["stop-done-a", "stop-done-b"],
            "open_base": "1",
        },
    )
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            "done-list", status="TERMINATED", close_type="FAILED"
        )
    connector = _Connector()
    connector.account_positions = _long_position("0")
    connector.in_flight_orders["entry-flat"] = _open_order()
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "entry-flat" in cancelled
    for stop_id in (
        "stop-flat-a",
        "stop-flat-b",
        "stop-zero-a",
        "stop-zero-b",
        "stop-done-a",
        "stop-done-b",
    ):
        assert stop_id not in cancelled
        assert stop_id in connector.stop_cancels
    assert "open_base is 0" in caplog.text
    assert "not RUNNING and was not parked for resume" in caplog.text
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "flat-list")
    assert row.close_type == "CLOSED_WHILE_DOWN"


@pytest.mark.asyncio
async def test_stop_route_cancels_every_exchange_stop_id():
    connector = _Connector()
    executor = _Executor([], custom_info={
        "exchange_stop_order_id": "stop-1",
        "exchange_stop_order_ids": ["stop-1", "stop-2"],
        "open_base": "1",
        "side": "BUY",
    })
    service = _service(connectors={"master": {"mexc_perpetual": connector}})
    metadata = {
        "account_name": "master",
        "connector_name": "mexc_perpetual",
        "trading_pair": "BTC-USDT",
        "executor_type": "range_iceberg_executor",
    }
    service._active_executors["ex-1"] = executor
    service._executor_metadata["ex-1"] = metadata
    await service.stop_executor("ex-1", keep_position=False)
    assert connector.cancels == []
    assert connector.stop_cancels == ["stop-1", "stop-2"]

    connector.stop_cancels.clear()
    kept = _Executor([], custom_info={
        "exchange_stop_order_id": "stop-1",
        "exchange_stop_order_ids": ["stop-1", "stop-2"],
        "open_base": "1",
    })
    service._active_executors["ex-2"] = kept
    service._executor_metadata["ex-2"] = metadata
    await service.stop_executor("ex-2", keep_position=True)
    assert connector.stop_cancels == []
    assert connector.cancels == []


@pytest.mark.asyncio
async def test_direct_cancel_false_and_gone_is_already_gone(db, caplog):
    await _seed_executor(
        db,
        "gone-exec",
        "master",
        "mexc_perpetual",
        "BTC-USDT",
        {"child_order_ids": ["gone-oid", "stuck-oid"]},
        [("gone-oid", "OPEN"), ("stuck-oid", "OPEN")],
    )
    await _seed_exchange_order(db, "gone-oid", "OPEN")
    await _seed_exchange_order(db, "stuck-oid", "OPEN")

    class _DecliningConnector(_Connector):
        def __init__(self):
            super().__init__()
            self.direct = []
            self.open_orders = [{"externalOid": "stuck-oid", "orderId": "9002"}]

        async def cancel(self, trading_pair=None, client_order_id=None):
            self.cancels.append((trading_pair, client_order_id))

        async def _place_cancel(self, order_id, tracked):
            self.direct.append(order_id)
            return False

        async def list_open_orders(self, trading_pair=None):
            return list(self.open_orders)

    connector = _DecliningConnector()
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()
    assert connector.cancels == []
    assert connector.direct == ["gone-oid", "stuck-oid"]
    assert "Software order gone-oid for executor gone-exec" in caplog.text
    assert "was already gone" in caplog.text
    assert "Cancelled open child order gone-oid" not in caplog.text
    assert "Failed to cancel open child order stuck-oid" in caplog.text
    assert "was rejected" in caplog.text
    assert "Software order stuck-oid" not in caplog.text


def test_resume_payload_keeps_exchange_stop_released():
    released = _exit_only_custom_info(
        {
            "side": "BUY",
            "exit_state": "CLOSING",
            "exchange_stop_released": True,
            "exit": {"exchange_stop_released": True, "active": "stop_loss"},
            "filled_base": "1",
            "filled_quote": "1",
        },
        Decimal("1"),
        Decimal("1"),
    )
    assert released["exchange_stop_released"] is True
    assert released["exit"]["exchange_stop_released"] is True

    from_exit_only = _exit_only_custom_info(
        {
            "side": "BUY",
            "exit_state": "CLOSING",
            "exit": {"exchange_stop_released": True},
            "filled_base": "1",
            "filled_quote": "1",
        },
        Decimal("1"),
        Decimal("1"),
    )
    assert from_exit_only["exchange_stop_released"] is True
    assert from_exit_only["exit"]["exchange_stop_released"] is True

    absent = _exit_only_custom_info(
        {
            "side": "BUY",
            "exit_state": "RESTING",
            "exit": {"active": "take_profit"},
            "filled_base": "1",
            "filled_quote": "1",
        },
        Decimal("1"),
        Decimal("1"),
    )
    assert absent.get("exchange_stop_released") in (None, False)
    assert absent["exit"].get("exchange_stop_released") in (None, False)


_OPEN_IDS = iter(range(1, 10_000))


class _OpenConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not getattr(self, "id", None):
            self.id = f"open-{next(_OPEN_IDS)}"


class _OpenExecutor:
    def __init__(self, strategy, config, update_interval, max_retries):
        self.config = config
        self.is_closed = False
        self.status = SimpleNamespace(name="RUNNING")
        self.close_type = None
        self._kept = None
        self.applied = {}
        self.reject_edit = None
        self.edit_calls = 0
        self.reported = {}
        self.sibling_calls = []
        self.entry_calls = []

    def start(self):
        return None

    def update_config(self, changes):
        self.edit_calls += 1
        if self.reject_edit:
            return {"accepted": False, "reason": self.reject_edit}
        unknown = [key for key in changes if key not in {
            "total_amount", "price_low", "price_high", "take_profit", "stop_loss",
        }]
        if unknown:
            return {"accepted": False, "reason": f"unsupported config field: {', '.join(sorted(unknown))}"}
        self.applied.update(changes)
        return {"accepted": True, "reason": ""}

    def early_stop(self, keep_position=False):
        self.is_closed = True
        self.status = SimpleNamespace(name="TERMINATED")
        name = "POSITION_HOLD" if keep_position else "EARLY_STOP"
        self.close_type = SimpleNamespace(name=name)
        if keep_position:
            self._kept = {
                "side": getattr(self.config, "side", "BUY"),
                "open_base": "1",
                "exchange_stop_order_id": "stop-kept",
            }

    def set_sibling_base(self, amount):
        self.sibling_calls.append(amount)
        if amount is None:
            self.reported["sibling_base"] = None
        else:
            self.reported["sibling_base"] = format(Decimal(str(amount)), "f")

    def set_entry_paused(self, paused, reason):
        self.entry_calls.append((bool(paused), reason))
        self.reported["entry_paused"] = bool(paused)
        self.reported["pause_reason"] = reason

    def get_custom_info(self):
        if self._kept is not None:
            return dict(self._kept)
        info = {"side": getattr(self.config, "side", "BUY")}
        info.update(self.reported)
        return info


def _open_service(db):
    service = _service(db)
    service.update_interval = 1.0
    service.max_retries = 10
    service.EXECUTOR_REGISTRY = dict(ExecutorService.EXECUTOR_REGISTRY)
    service.EXECUTOR_REGISTRY["range_iceberg_executor"] = (_OpenExecutor, _OpenConfig)
    service._get_trading_interface = lambda account: SimpleNamespace(current_timestamp=1.0)

    async def _prepare(*_args, **_kwargs):
        return None

    service._prepare_market = _prepare
    service._log_capture.get_error_count.return_value = 0
    return service


def _open_config(side, pair="BTC-USDT", connector="mexc_perpetual", **extra):
    config = {
        "type": "range_iceberg_executor",
        "connector_name": connector,
        "trading_pair": pair,
        "side": side,
        "total_amount": "1",
    }
    config.update(extra)
    return config


def _cap_detail():
    return "10 range icebergs are already open on BTC-USDT BUY for master. Close one first."


async def _open_ids(service, side="BUY"):
    rows = await service._range_iceberg_open_rows(("master", "mexc_perpetual", "BTC-USDT", side))
    return {row["executor_id"] for row in rows}


@pytest.mark.asyncio
async def test_a_parked_executor_counts_toward_the_cap(db):
    await _seed_iceberg_row(
        db,
        "parked-long",
        {"side": "LONG", "exit_state": "RESTING", "resume_on_startup": True, "open_base": "1"},
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "LONG",
        },
    )
    service = _open_service(db)
    assert "parked-long" not in service._active_executors
    assert "parked-long" in await _open_ids(service)
    for _ in range(9):
        await service.create_executor(_open_config("BUY"), account_name="master")
    with pytest.raises(HTTPException) as caught:
        await service.create_executor(_open_config("BUY"), account_name="master")
    assert caught.value.status_code == 409
    assert caught.value.detail == _cap_detail()


@pytest.mark.asyncio
async def test_long_and_short_on_the_same_pair_are_both_allowed(db):
    service = _open_service(db)
    buy = await service.create_executor(_open_config("BUY"), account_name="master")
    sell = await service.create_executor(_open_config("SELL"), account_name="master")
    assert buy["executor_id"] != sell["executor_id"]
    assert buy["executor_id"] in service._active_executors
    assert sell["executor_id"] in service._active_executors


@pytest.mark.asyncio
async def test_range_iceberg_is_allowed_after_the_first_terminates(db):
    service = _open_service(db)
    first = await service.create_executor(_open_config("BUY"), account_name="master")
    service._active_executors.pop(first["executor_id"], None)
    service._executor_metadata.pop(first["executor_id"], None)
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            first["executor_id"], status="TERMINATED", close_type="EARLY_STOP"
        )
    second = await service.create_executor(_open_config("BUY"), account_name="master")
    assert second["executor_id"] != first["executor_id"]


@pytest.mark.asyncio
async def test_external_close_frees_the_slot_even_when_open_base_remains(db):
    """POSITION_CLOSED_EXTERNALLY is finished. open_base may still be the old size."""
    custom = {
        "side": "BUY",
        "open_base": "5",
        "filled_base": "5",
        "exchange_position_base": "0",
        "external_close": True,
        "reconcile_error": "",
        "status_reason": "position closed outside the bot",
    }
    await _seed_iceberg_row(
        db,
        "closed-outside",
        custom,
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "BUY",
        },
    )
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            "closed-outside",
            status="TERMINATED",
            close_type="POSITION_CLOSED_EXTERNALLY",
        )
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "closed-outside"
    shown = await service.get_executor("closed-outside")
    assert shown["close_type"] == "POSITION_CLOSED_EXTERNALLY"
    info = shown["custom_info"]
    assert info["open_base"] == "5"
    assert info["exchange_position_base"] == "0"
    assert info["external_close"] is True
    assert info["reconcile_error"] == ""
    assert info["status_reason"] == "position closed outside the bot"
    assert service.get_positions_held() == []


@pytest.mark.asyncio
async def test_a_running_external_close_does_not_hold_the_slot(db):
    """A row still marked RUNNING is finished once external_close is set."""
    await _seed_iceberg_row(
        db,
        "closing-outside",
        {
            "side": "BUY",
            "open_base": "5",
            "exchange_position_base": "0",
            "external_close": True,
            "reconcile_error": "",
            "resume_on_startup": True,
            "exit_state": "RESTING",
        },
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "BUY",
        },
    )
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "closing-outside"


@pytest.mark.asyncio
async def test_external_close_does_not_keep_a_phantom_position():
    service = _service()
    info = {
        "open_base": "5",
        "filled_base": "5",
        "exchange_position_base": "0",
        "external_close": True,
        "side": "BUY",
        "average_fill_price": "0.1",
    }
    await service._aggregate_position_hold(
        "ex-1", _HoldExecutor(info), _hold_metadata("range_iceberg_executor"),
    )
    position = next(iter(service._positions_held.values()))
    assert position.buy_amount_base == 0
    assert position.executor_ids == []


@pytest.mark.asyncio
async def test_close_type_frees_a_running_row_when_external_close_is_false(db):
    """The column is enough. A false flag and a stale error must not keep the slot."""
    await _seed_iceberg_row(
        db,
        "typed-outside",
        {
            "side": "BUY",
            "open_base": "5",
            "exchange_position_base": "5",
            "external_close": False,
            "reconcile_error": "position read failed: timeout",
            "resume_on_startup": True,
        },
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "BUY",
        },
    )
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            "typed-outside",
            status="RUNNING",
            close_type="POSITION_CLOSED_EXTERNALLY",
        )
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "typed-outside"
    candidates = await service._load_resume_candidates()
    assert [row["executor_id"] for row in candidates] == []


@pytest.mark.asyncio
async def test_a_live_close_type_frees_the_slot_before_the_flag_is_stored(db):
    service = _open_service(db)
    first = await service.create_executor(_open_config("BUY"), account_name="master")
    executor = service._active_executors[first["executor_id"]]
    executor.close_type = SimpleNamespace(name="POSITION_CLOSED_EXTERNALLY")
    executor._kept = {
        "side": "BUY",
        "open_base": "5",
        "exchange_position_base": "5",
        "external_close": False,
        "reconcile_error": "position read failed: timeout",
    }
    second = await service.create_executor(_open_config("BUY"), account_name="master")
    assert second["executor_id"] != first["executor_id"]


@pytest.mark.asyncio
async def test_a_stale_reconcile_error_does_not_free_the_slot(db):
    """A failed position read leaves the error set while the position is still open."""
    await _seed_iceberg_row(
        db,
        "stale-read",
        {
            "side": "BUY",
            "open_base": "5",
            "exchange_position_base": "5",
            "external_close": False,
            "reconcile_error": "position read failed: timeout",
            "resume_on_startup": True,
            "exit_state": "RESTING",
        },
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "BUY",
        },
    )
    service = _open_service(db)
    assert "stale-read" in await _open_ids(service)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "stale-read"
    assert "stale-read" in await _open_ids(service)


@pytest.mark.asyncio
async def test_range_iceberg_on_another_account_is_allowed(db):
    service = _open_service(db)
    first = await service.create_executor(_open_config("BUY"), account_name="master")
    other = await service.create_executor(_open_config("BUY"), account_name="other")
    assert other["executor_id"] != first["executor_id"]
    assert service._executor_metadata[other["executor_id"]]["account_name"] == "other"


@pytest.mark.asyncio
async def test_concurrent_range_iceberg_creates_both_succeed(db):
    service = _open_service(db)
    results = await asyncio.gather(
        service.create_executor(_open_config("BUY"), account_name="master"),
        service.create_executor(_open_config("BUY"), account_name="master"),
        return_exceptions=True,
    )
    created = [row for row in results if isinstance(row, dict)]
    rejected = [row for row in results if isinstance(row, HTTPException)]
    assert len(created) == 2
    assert rejected == []
    assert created[0]["executor_id"] != created[1]["executor_id"]
    rows = [row for row in db.query(ExecutorRecord) if row.trading_pair == "BTC-USDT" and row.status == "RUNNING"]
    assert len(rows) == 2


def _position_connector(amount, stop_ids=None):
    connector = _Connector()
    connector.account_positions = {
        "p": {
            "trading_pair": "BTC-USDT",
            "position_side": "LONG",
            "amount": amount,
            "entry_price": "1",
        }
    }
    for stop_id in stop_ids or []:
        connector.in_flight_orders[stop_id] = object()
    return connector


def _use_book(service, connector):
    books = {"master": {"mexc_perpetual": connector}}

    def get_account_connectors(account_name):
        return books.get(account_name, {})

    service._trading_service.connector_service.get_account_connectors = get_account_connectors


async def _park_kept_position(service, db):
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    await service.stop_executor(executor_id, keep_position=True)
    await service._handle_executor_completion(executor_id)
    rows = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id]
    assert len(rows) == 1
    assert rows[0].status == "TERMINATED"
    assert rows[0].close_type == "POSITION_HOLD"
    return executor_id


@pytest.mark.asyncio
async def test_keep_position_still_holds_the_slot(db):
    service = _open_service(db)
    parked = await _park_kept_position(service, db)
    assert parked in await _open_ids(service)
    second = await service.create_executor(_open_config("BUY"), account_name="master")
    assert second["executor_id"] != parked
    assert parked in await _open_ids(service)


@pytest.mark.asyncio
async def test_flat_position_frees_a_position_hold_slot(db):
    service = _open_service(db)
    parked = await _park_kept_position(service, db)
    _use_book(service, _position_connector("0", ["stop-kept"]))
    second = await service.create_executor(_open_config("BUY"), account_name="master")
    assert second["executor_id"] != parked


@pytest.mark.asyncio
async def test_release_slot_lets_a_new_range_iceberg_start(db):
    from models.executors import ReleaseRangeIcebergSlotRequest
    from routers.executors import release_range_iceberg_slot

    service = _open_service(db)
    with pytest.raises(HTTPException) as missing:
        await service.release_range_iceberg_slot(
            "master", "mexc_perpetual", "BTC-USDT", "BUY"
        )
    assert missing.value.status_code == 404

    parked = await _park_kept_position(service, db)
    assert parked in await _open_ids(service)

    released = await release_range_iceberg_slot(
        ReleaseRangeIcebergSlotRequest(
            account_name="master",
            connector_name="mexc_perpetual",
            trading_pair="btc_usdt",
            side="LONG",
            executor_id=parked,
        ),
        executor_service=service,
    )
    assert released.executor_id == parked
    assert released.released is True
    assert released.trading_pair == "BTC-USDT"
    assert released.side == "BUY"
    assert released.position_closed is False
    assert released.exchange_stop_cancelled is False
    assert released.warning == (
        "Release does not close the position or cancel the parked exchange stop. "
        "A new same-side executor runs on the software stop loss only. "
        "Its exchange stop is refused until the parked stop is gone."
    )
    assert parked not in await _open_ids(service)
    stored = [row for row in db.query(ExecutorRecord) if row.executor_id == parked]
    assert json.loads(stored[0].final_state)["slot_released"] is True

    second = await service.create_executor(_open_config("BUY"), account_name="master")
    assert second["executor_id"] != parked
    with pytest.raises(HTTPException) as gone:
        await service.release_range_iceberg_slot(
            "master", "mexc_perpetual", "BTC-USDT", "BUY", executor_id=parked
        )
    assert gone.value.status_code == 404


@pytest.mark.asyncio
async def test_stale_running_flat_with_no_live_stop_does_not_block(db):
    await _seed_iceberg_row(
        db,
        "stale-flat",
        {
            "side": "BUY",
            "open_base": "0",
            "exchange_stop_order_id": "stop-gone",
        },
    )
    service = _open_service(db)
    _use_book(service, _position_connector("0"))
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "stale-flat"


@pytest.mark.asyncio
async def test_stale_running_blocks_when_the_position_or_stop_is_live(db):
    await _seed_iceberg_row(db, "stale-open", {"side": "BUY", "open_base": "1"})
    service = _open_service(db)
    _use_book(service, _position_connector("1"))
    assert "stale-open" in await _open_ids(service)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "stale-open"
    assert "stale-open" in await _open_ids(service)

    await _seed_iceberg_row(
        db,
        "stale-stop",
        {"side": "SELL", "open_base": "0", "exchange_stop_order_id": "stop-stale"},
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "SELL",
        },
    )
    short_book = _position_connector("0", ["stop-stale"])
    short_book.account_positions["p"]["position_side"] = "SHORT"
    short_book.account_positions["p"]["amount"] = "0"
    _use_book(service, short_book)
    assert "stale-stop" in await _open_ids(service, "SELL")
    created = await service.create_executor(_open_config("SELL"), account_name="master")
    assert created["executor_id"] != "stale-stop"
    assert "stale-stop" in await _open_ids(service, "SELL")


@pytest.mark.asyncio
async def test_stale_running_blocks_when_the_book_cannot_be_read(db):
    await _seed_iceberg_row(db, "stale-unread", {"side": "BUY", "open_base": "0"})
    service = _open_service(db)
    assert "stale-unread" in await _open_ids(service)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "stale-unread"
    assert "stale-unread" in await _open_ids(service)


@pytest.mark.asyncio
async def test_position_hold_without_base_or_stop_does_not_hold_the_slot(db):
    service = _open_service(db)
    first = await service.create_executor(_open_config("BUY"), account_name="master")
    executor = service._active_executors[first["executor_id"]]
    executor.early_stop(keep_position=True)
    executor._kept = {"side": "BUY", "open_base": "0"}
    await service._handle_executor_completion(first["executor_id"])
    second = await service.create_executor(_open_config("BUY"), account_name="master")
    assert second["executor_id"] != first["executor_id"]


@pytest.mark.asyncio
async def test_trading_pair_case_and_separators_share_one_slot(db):
    service = _open_service(db)
    first = await service.create_executor(_open_config("BUY", pair="BTC-USDT"), account_name="master")
    second = await service.create_executor(_open_config("BUY", pair="btc-usdt"), account_name="master")
    ids = await _open_ids(service)
    assert first["executor_id"] in ids
    assert second["executor_id"] in ids
    extra = 0
    rejected = None
    for _ in range(12):
        try:
            await service.create_executor(_open_config("BUY", pair="btc/usdt"), account_name="master")
            extra += 1
        except HTTPException as exc:
            rejected = exc
            break
    assert extra == 8
    assert rejected is not None
    assert rejected.status_code == 409
    assert rejected.detail == _cap_detail()
    service._active_executors.pop(first["executor_id"], None)
    assert first["executor_id"] in await _open_ids(service)


class _FailingSession:
    async def __aenter__(self):
        raise RuntimeError("completion write failed")

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_completion_write_retries_once(db, caplog):
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    executor = service._active_executors[executor_id]
    executor.status = SimpleNamespace(name="TERMINATED")
    executor.close_type = SimpleNamespace(name="EARLY_STOP")
    real_context = service.db_manager.get_session_context
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _FailingSession()
        return real_context()

    service.db_manager.get_session_context = flaky
    with caplog.at_level(logging.ERROR):
        await service._handle_executor_completion(executor_id)
    assert attempts["n"] == 2
    assert "attempt 1 of 2" in caplog.text
    assert "failed after one retry" not in caplog.text
    rows = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id]
    assert rows[0].status == "TERMINATED"
    assert rows[0].close_type == "EARLY_STOP"


@pytest.mark.asyncio
async def test_completion_write_logs_when_the_retry_also_fails(db, caplog):
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    executor = service._active_executors[executor_id]
    executor.status = SimpleNamespace(name="TERMINATED")
    executor.close_type = SimpleNamespace(name="EARLY_STOP")

    def always_fail():
        return _FailingSession()

    service.db_manager.get_session_context = always_fail
    with caplog.at_level(logging.ERROR):
        await service._handle_executor_completion(executor_id)
    assert "attempt 1 of 2" in caplog.text
    assert "attempt 2 of 2" in caplog.text
    assert "failed after one retry" in caplog.text
    assert "The row may still be RUNNING." in caplog.text
    rows = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id]
    assert rows[0].status == "RUNNING"


async def _mark_position_hold(db, executor_id):
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            executor_id, status="TERMINATED", close_type="POSITION_HOLD"
        )


@pytest.mark.asyncio
async def test_release_refuses_a_running_row_and_resume_skips_released(db):
    await _seed_iceberg_row(
        db,
        "still-running",
        {
            "side": "BUY",
            "resume_on_startup": True,
            "exit_state": "RESTING",
            "open_base": "1",
        },
    )
    service = _open_service(db)
    with pytest.raises(HTTPException) as unnamed:
        await service.release_range_iceberg_slot(
            "master", "mexc_perpetual", "BTC-USDT", "BUY"
        )
    assert unnamed.value.status_code == 409
    assert unnamed.value.detail == (
        "This range iceberg will resume on restart (executor still-running). "
        "Stop it first."
    )
    with pytest.raises(HTTPException) as named:
        await service.release_range_iceberg_slot(
            "master",
            "mexc_perpetual",
            "BTC-USDT",
            "BUY",
            executor_id="still-running",
        )
    assert named.value.status_code == 409
    stored_rows = [row for row in db.query(ExecutorRecord) if row.executor_id == "still-running"]
    stored = json.loads(stored_rows[0].final_state)
    assert stored.get("slot_released") is not True
    assert stored["resume_on_startup"] is True

    stored["slot_released"] = True
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            "still-running", final_state=json.dumps(stored)
        )
    await _seed_iceberg_row(
        db,
        "will-resume",
        {
            "side": "SELL",
            "resume_on_startup": True,
            "exit_state": "RESTING",
            "open_base": "1",
        },
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "SELL",
        },
    )
    found = await service._load_resume_candidates()
    ids = {item["executor_id"] for item in found}
    assert "still-running" not in ids
    assert "will-resume" in ids


@pytest.mark.asyncio
async def test_position_book_is_refreshed_once_per_create(db):
    await _seed_iceberg_row(db, "hold-a", {"side": "BUY", "open_base": "1"})
    await _seed_iceberg_row(db, "hold-b", {"side": "BUY", "open_base": "2"})
    await _mark_position_hold(db, "hold-a")
    await _mark_position_hold(db, "hold-b")
    service = _open_service(db)
    connector = _position_connector("0")
    connector.refresh_calls = 0

    async def _update_positions():
        connector.refresh_calls += 1

    connector._update_positions = _update_positions
    _use_book(service, connector)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] not in {"hold-a", "hold-b"}
    assert connector.refresh_calls == 1


@pytest.mark.asyncio
async def test_unreadable_target_position_is_not_flat(db):
    await _seed_iceberg_row(
        db,
        "hold-bad",
        {"side": "BUY", "open_base": "1", "exchange_stop_order_id": "stop-kept"},
    )
    await _mark_position_hold(db, "hold-bad")
    service = _open_service(db)
    connector = _position_connector("1")
    connector.account_positions["p"]["amount"] = "not-a-number"
    connector.account_positions["other"] = {
        "trading_pair": "ETH-USDT",
        "position_side": "LONG",
        "amount": "also-bad",
        "entry_price": "1",
    }
    _use_book(service, connector)
    assert "hold-bad" in await _open_ids(service)
    allowed = await service.create_executor(_open_config("BUY"), account_name="master")
    assert allowed["executor_id"] != "hold-bad"
    assert "hold-bad" in await _open_ids(service)

    connector.account_positions["p"]["amount"] = "0"
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    assert created["executor_id"] != "hold-bad"


@pytest.mark.asyncio
async def test_accepted_range_iceberg_edit_survives_resume(db):
    from routers.executors import update_range_iceberg
    from models.executors import UpdateRangeIcebergRequest

    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    body = UpdateRangeIcebergRequest(
        total_amount="3",
        price_low="90",
        price_high="110",
        take_profit={"percent": "0.02", "execution": "MARKET"},
        stop_loss=None,
    )
    result = await update_range_iceberg(executor_id, body, executor_service=service)
    assert result.accepted is True
    assert result.reason == ""
    assert result.executor_id == executor_id

    rows = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id]
    saved = json.loads(rows[0].config)
    assert saved["total_amount"] == "1"
    assert "price_low" not in saved
    assert saved["side"] == "BUY"
    state = json.loads(rows[0].final_state)
    assert state["pending_config_edit"] is True
    assert "target_amount" not in state
    assert state["config_edits"]["total_amount"] == "3"
    assert state["config_edits"]["price_low"] == "90"
    assert state["config_edits"]["price_high"] == "110"
    assert state["config_edits"]["take_profit"] == {"percent": "0.02", "execution": "MARKET"}
    assert state["config_edits"]["stop_loss"] is None

    state["resume_on_startup"] = True
    state["exit_state"] = "RESTING"
    state["open_base"] = "1"
    state["side"] = "BUY"
    state["exchange_stop_order_id"] = "stop-kept"
    state["total_amount"] = "8.5"
    state["price_low"] = "1"
    async with db.session_context() as session:
        await ExecutorRepository(session).update_executor(
            executor_id, final_state=json.dumps(state)
        )
    service._active_executors.pop(executor_id, None)
    service._executor_metadata.pop(executor_id, None)
    _install_resume_factory(service)
    connector = _position_connector("1")
    _use_book(service, connector)
    await service.cleanup_orphaned_executors()
    resumed = service._active_executors[executor_id]
    assert resumed.config.total_amount == "3"
    assert resumed.config.price_low == "90"
    assert resumed.config.price_high == "110"
    assert resumed.config.take_profit == {"percent": "0.02", "execution": "MARKET"}
    assert resumed.config.stop_loss is None
    assert resumed.config.size_unit == "BASE"
    assert resumed.config.resume_entries is False
    assert resumed.restored["total_amount"] == "3"
    assert resumed.restored["target_amount"] == "3"
    assert resumed.restored["price_low"] == "90"
    assert resumed.restored["price_high"] == "110"
    assert resumed.restored["take_profit"]["percent"] == "0.02"
    assert resumed.restored["stop_loss"] is None
    assert resumed.restored["open_base"] == "1"


@pytest.mark.asyncio
async def test_rejected_range_iceberg_edit_changes_nothing(db):
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    before = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    before_config = before.config
    before_state = before.final_state
    service._active_executors[executor_id].reject_edit = "a stop loss close is in progress"
    with pytest.raises(HTTPException) as caught:
        await service.update_range_iceberg_config(executor_id, {"total_amount": "9"})
    assert caught.value.status_code == 409
    assert caught.value.detail["accepted"] is False
    assert caught.value.detail["reason"] == "a stop loss close is in progress"
    assert caught.value.detail["executor_id"] == executor_id
    after = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    assert after.config == before_config
    assert after.final_state == before_state
    assert service._active_executors[executor_id].applied == {}


@pytest.mark.asyncio
async def test_range_iceberg_edit_404_does_not_touch_a_parked_row(db):
    await _seed_iceberg_row(
        db,
        "parked-only",
        {"side": "BUY", "resume_on_startup": True, "open_base": "1", "exit_state": "RESTING"},
    )
    before = [row for row in db.query(ExecutorRecord) if row.executor_id == "parked-only"][0]
    service = _open_service(db)
    with pytest.raises(HTTPException) as missing:
        await service.update_range_iceberg_config("missing", {"total_amount": "2"})
    assert missing.value.status_code == 404
    assert missing.value.detail == "Executor missing not found"

    service._active_executors["other"] = SimpleNamespace()
    service._executor_metadata["other"] = {"executor_type": "position_executor"}
    with pytest.raises(HTTPException) as wrong_type:
        await service.update_range_iceberg_config("other", {"total_amount": "2"})
    assert wrong_type.value.status_code == 404
    assert wrong_type.value.detail == "Executor other is not a range iceberg"

    with pytest.raises(HTTPException) as parked:
        await service.update_range_iceberg_config("parked-only", {"total_amount": "4"})
    assert parked.value.status_code == 404
    after = [row for row in db.query(ExecutorRecord) if row.executor_id == "parked-only"][0]
    assert after.config == before.config
    assert after.final_state == before.final_state


@pytest.mark.asyncio
async def test_concurrent_range_iceberg_edits_do_not_interleave(db):
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    executor = service._active_executors[executor_id]
    executor.inside = False
    executor.overlapped = False

    async def update_config(changes):
        if executor.inside:
            executor.overlapped = True
        executor.inside = True
        await asyncio.sleep(0.05)
        executor.inside = False
        executor.applied.update(changes)
        return {"accepted": True, "reason": ""}

    executor.update_config = update_config
    first, second = await asyncio.gather(
        service.update_range_iceberg_config(executor_id, {"total_amount": "4"}),
        service.update_range_iceberg_config(executor_id, {"price_low": "80", "price_high": "120"}),
    )
    assert first["accepted"] is True
    assert second["accepted"] is True
    assert executor.overlapped is False
    row = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    saved = json.loads(row.config)
    assert saved["total_amount"] == "1"
    state = json.loads(row.final_state)
    assert state["pending_config_edit"] is True
    assert state["config_edits"]["total_amount"] == "4"
    assert state["config_edits"]["price_low"] == "80"
    assert state["config_edits"]["price_high"] == "120"


@pytest.mark.asyncio
async def test_editing_a_range_iceberg_does_not_free_its_slot(db):
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    await service.update_range_iceberg_config(
        executor_id,
        {"total_amount": "2", "price_low": "10", "price_high": "20"},
    )
    second = await service.create_executor(_open_config("BUY"), account_name="master")
    assert second["executor_id"] != executor_id
    assert executor_id in await _open_ids(service)
    other = await service.create_executor(_open_config("SELL"), account_name="master")
    assert other["executor_id"] != executor_id
    assert executor_id in service._active_executors


async def _resume_sized_executor(
    db,
    executor_id,
    size_unit,
    config_total,
    extra_custom=None,
    contract_size=None,
    position="50",
    filled_quote="8.5",
    entry_price="0.17",
    open_base="50",
):
    custom = {
        "resume_on_startup": True,
        "exit_state": "RESTING",
        "side": "BUY",
        "open_base": open_base,
        "filled_base": open_base,
        "exchange_stop_order_id": "stop-x",
        "total_amount": config_total,
        "target_amount": config_total,
        "price_low": "9",
        "price_high": "11",
        "take_profit": {"percent": "0.01"},
        "stop_loss": {"percent": "0.02"},
    }
    if filled_quote is not None:
        custom["filled_quote"] = filled_quote
    if extra_custom:
        custom.update(extra_custom)
    await _seed_iceberg_row(
        db,
        executor_id,
        custom,
        config={
            "type": "range_iceberg_executor",
            "connector_name": "mexc_perpetual",
            "trading_pair": "BTC-USDT",
            "side": "BUY",
            "size_unit": size_unit,
            "total_amount": config_total,
        },
    )
    connector = _Connector()
    connector.account_positions = {
        "BTC-USDT:LONG": SimpleNamespace(
            trading_pair="BTC-USDT",
            position_side=SimpleNamespace(name="LONG"),
            amount=Decimal(position),
            entry_price=Decimal(entry_price),
        )
    }
    if contract_size is not None:
        connector.get_contract_size = lambda _pair: contract_size
    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)
    await service.cleanup_orphaned_executors()
    return service, connector


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "size_unit", "config_total", "edits", "contract_size", "expected"),
    [
        ("quote", "QUOTE", "8.5", None, None, "50"),
        ("base", "BASE", "80", None, None, "50"),
        ("contracts", "CONTRACTS", "10", None, "2", "50"),
        ("quote-edit", "QUOTE", "8.5", {"total_amount": "17", "price_low": "0.15"}, None, "100"),
        ("base-edit", "BASE", "80", {"total_amount": "60"}, None, "60"),
        ("contracts-edit", "CONTRACTS", "10", {"total_amount": "20", "price_high": "3"}, "2", "40"),
    ],
)
async def test_resume_size_uses_open_base_unless_config_edits_sets_total(
    db, name, size_unit, config_total, edits, contract_size, expected,
):
    """8.5 USDT with 50 base filled resumes as base 50 unless config_edits says otherwise."""
    extra = {"config_edits": edits} if edits else None
    service, connector = await _resume_sized_executor(
        db,
        f"sized-{name}",
        size_unit,
        config_total,
        extra_custom=extra,
        contract_size=contract_size,
    )
    resumed = service._active_executors[f"sized-{name}"]
    assert resumed.config.size_unit == "BASE"
    assert Decimal(resumed.config.total_amount) == Decimal(expected)
    assert resumed.config.resume_entries is False
    assert connector.stop_cancels == []
    if edits and "total_amount" in edits:
        assert Decimal(resumed.restored["total_amount"]) == Decimal(expected)
        assert Decimal(resumed.restored["target_amount"]) == Decimal(expected)
    else:
        assert resumed.restored.get("total_amount") is None
    if edits and "price_low" in edits:
        assert resumed.config.price_low == edits["price_low"]
        assert resumed.restored["price_low"] == edits["price_low"]
    else:
        assert getattr(resumed.config, "price_low", None) is None
    if edits and "price_high" in edits:
        assert resumed.config.price_high == edits["price_high"]
    else:
        assert getattr(resumed.config, "price_high", None) is None


@pytest.mark.asyncio
async def test_unconvertible_quote_edit_keeps_the_open_base(db, caplog):
    with caplog.at_level(logging.ERROR):
        service, connector = await _resume_sized_executor(
            db,
            "quote-blind",
            "QUOTE",
            "8.5",
            extra_custom={"config_edits": {"total_amount": "17"}},
            filled_quote=None,
            entry_price="0",
        )
    resumed = service._active_executors["quote-blind"]
    assert Decimal(resumed.config.total_amount) == Decimal("50")
    assert connector.stop_cancels == []
    assert "Could not convert edited total_amount" in caplog.text


@pytest.mark.asyncio
async def test_dropped_config_edit_is_not_resumed_and_is_visible(db):
    service, _connector = await _resume_sized_executor(
        db,
        "dropped-1",
        "QUOTE",
        "8.5",
        extra_custom={
            "config_edits": {"total_amount": "17"},
            "config_edit_error": "the edit was dropped",
            "pending_config_edit": False,
        },
    )
    resumed = service._active_executors["dropped-1"]
    assert Decimal(resumed.config.total_amount) == Decimal("50")
    service._active_executors.pop("dropped-1", None)
    rows = await service.get_executors()
    shown = next(row for row in rows if row["executor_id"] == "dropped-1")
    assert shown["pending_config_edit"] is False
    assert shown["config_edit_error"] == "the edit was dropped"


@pytest.mark.asyncio
async def test_pending_edit_stays_pending_until_the_executor_applies_it(db):
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    await service.update_range_iceberg_config(
        executor_id, {"total_amount": "4", "price_low": "90"},
    )
    row = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    assert json.loads(row.config)["total_amount"] == "1"

    executor = service._active_executors[executor_id]
    executor.get_custom_info = lambda: {
        "side": "BUY",
        "total_amount": "8.5",
        "price_low": "1",
        "child_order_ids": ["c1"],
    }
    await service._persist_running_child_orders(executor_id, executor)
    row = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    saved = json.loads(row.config)
    state = json.loads(row.final_state)
    assert saved["total_amount"] == "1"
    assert "price_low" not in saved
    assert state["pending_config_edit"] is True
    assert state["config_edits"]["total_amount"] == "4"
    assert state["total_amount"] == "8.5"

    executor.get_custom_info = lambda: {
        "side": "BUY",
        "total_amount": "8.5",
        "child_order_ids": ["c1", "c2"],
        "pending_config_edit": False,
        "config_edits": {"total_amount": "4", "price_low": "90"},
    }
    await service._persist_running_child_orders(executor_id, executor)
    row = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    saved = json.loads(row.config)
    state = json.loads(row.final_state)
    assert saved["total_amount"] == "4"
    assert saved["price_low"] == "90"
    assert state["pending_config_edit"] is False
    assert state["config_edits"]["total_amount"] == "4"
    assert state["total_amount"] == "8.5"

    service._active_executors.pop(executor_id, None)
    shown = await service.get_executor(executor_id)
    assert shown["pending_config_edit"] is False
    assert shown["config_edit_error"] is None


@pytest.mark.asyncio
async def test_a_reported_drop_clears_the_pending_edit(db):
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    await service.update_range_iceberg_config(executor_id, {"total_amount": "4"})
    await service._write_final_state(executor_id, {
        "side": "BUY",
        "total_amount": "1",
        "pending_config_edit": False,
        "config_edit_error": "the edit was dropped",
    })
    row = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    saved = json.loads(row.config)
    state = json.loads(row.final_state)
    assert saved["total_amount"] == "1"
    assert "config_edits" not in state
    assert state["pending_config_edit"] is False
    assert state["config_edit_error"] == "the edit was dropped"
    service._active_executors.pop(executor_id, None)
    shown = await service.get_executor(executor_id)
    assert shown["pending_config_edit"] is False
    assert shown["config_edit_error"] == "the edit was dropped"


def test_live_executor_response_surfaces_the_config_edit():
    service = _service()
    info = SimpleNamespace(
        side=SimpleNamespace(name="BUY"),
        model_dump=lambda: {
            "custom_info": {
                "pending_config_edit": True,
                "config_edit_error": "still queued",
                "total_amount": "8.5",
                "size_unit": "BASE",
            },
        },
    )
    executor = SimpleNamespace(
        executor_info=info,
        status=SimpleNamespace(name="RUNNING"),
        close_type=None,
        is_closed=False,
    )
    service._executor_metadata["live-1"] = {
        "executor_type": "range_iceberg_executor",
        "account_name": "master",
    }
    shown = service._format_executor_info("live-1", executor)
    assert shown["pending_config_edit"] is True
    assert shown["config_edit_error"] == "still queued"
    assert shown["config_edit_note"] is None
    assert shown["size_unit"] == "BASE"
    info.model_dump = lambda: {
        "custom_info": {
            "pending_config_edit": False,
            "config_edit_note": "clamped to filled",
            "size_unit": "BASE",
        },
    }
    noted = service._format_executor_info("live-1", executor)
    assert noted["pending_config_edit"] is False
    assert noted["config_edit_error"] is None
    assert noted["config_edit_note"] == "clamped to filled"


@pytest.mark.asyncio
async def test_an_applied_edit_with_a_note_is_stored(db):
    """config_edit_note means the edit was applied. The note is not a drop."""
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    await service.update_range_iceberg_config(executor_id, {"total_amount": "4"})
    await service._write_final_state(executor_id, {
        "side": "BUY",
        "total_amount": "3",
        "pending_config_edit": False,
        "config_edit_note": "clamped to filled",
        "config_edits": {"total_amount": "3", "size_unit": "BASE"},
    })
    row = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    saved = json.loads(row.config)
    state = json.loads(row.final_state)
    assert saved["total_amount"] == "3"
    assert saved["size_unit"] == "BASE"
    assert state["pending_config_edit"] is False
    assert state["config_edits"]["total_amount"] == "3"
    assert state["config_edits"]["size_unit"] == "BASE"
    assert state["config_edit_note"] == "clamped to filled"
    assert "config_edit_error" not in state
    service._active_executors.pop(executor_id, None)
    shown = await service.get_executor(executor_id)
    assert shown["pending_config_edit"] is False
    assert shown["config_edit_error"] is None
    assert shown["config_edit_note"] == "clamped to filled"


@pytest.mark.asyncio
async def test_a_note_does_not_drop_a_pending_edit(db):
    """While pending_config_edit is true, a note leaves the queued edit in place."""
    service = _open_service(db)
    created = await service.create_executor(_open_config("BUY"), account_name="master")
    executor_id = created["executor_id"]
    await service.update_range_iceberg_config(
        executor_id, {"total_amount": "4", "price_low": "90"},
    )
    executor = service._active_executors[executor_id]
    executor.get_custom_info = lambda: {
        "side": "BUY",
        "total_amount": "1",
        "pending_config_edit": True,
        "config_edit_note": "cancels still pending",
    }
    await service._persist_running_child_orders(executor_id, executor)
    row = [row for row in db.query(ExecutorRecord) if row.executor_id == executor_id][0]
    saved = json.loads(row.config)
    state = json.loads(row.final_state)
    assert saved["total_amount"] == "1"
    assert "price_low" not in saved
    assert state["pending_config_edit"] is True
    assert state["config_edits"]["total_amount"] == "4"
    assert state["config_edits"]["price_low"] == "90"
    assert state["config_edit_note"] == "cancels still pending"
    assert "config_edit_error" not in state
    service._active_executors.pop(executor_id, None)
    shown = await service.get_executor(executor_id)
    assert shown["pending_config_edit"] is True
    assert shown["config_edit_error"] is None
    assert shown["config_edit_note"] == "cancels still pending"


@pytest.mark.asyncio
async def test_u1b_quote_edit_resumes_as_200_base(db):
    """20 USDT at 0.1 is 200 base. The config column is still QUOTE."""
    service, connector = await _resume_sized_executor(
        db,
        "u1b",
        "QUOTE",
        "10",
        extra_custom={"config_edits": {"total_amount": "20", "size_unit": "QUOTE"}},
        position="100",
        filled_quote="10",
        entry_price="0.1",
        open_base="100",
    )
    resumed = service._active_executors["u1b"]
    assert resumed.config.size_unit == "BASE"
    assert Decimal(resumed.config.total_amount) == Decimal("200")
    assert connector.stop_cancels == []


@pytest.mark.asyncio
async def test_u2_base_edit_on_a_quote_column_stays_200(db):
    """config_edits.size_unit BASE wins over a config column that still says QUOTE."""
    service, connector = await _resume_sized_executor(
        db,
        "u2",
        "QUOTE",
        "10",
        extra_custom={"config_edits": {"total_amount": "200", "size_unit": "BASE"}},
        position="100",
        filled_quote="10",
        entry_price="0.1",
        open_base="100",
    )
    resumed = service._active_executors["u2"]
    assert resumed.config.size_unit == "BASE"
    assert Decimal(resumed.config.total_amount) == Decimal("200")
    assert connector.stop_cancels == []
    stored = json.loads(
        next(row for row in db.query(ExecutorRecord) if row.executor_id == "u2").config
    )
    assert stored["size_unit"] == "BASE"
    assert stored["total_amount"] == "200"


@pytest.mark.asyncio
async def test_u4_edit_after_resume_is_not_converted_again(db):
    """QUOTE 10 USDT, 100 base at 0.1, then a base edit of 300, stays 300."""
    service, connector = await _resume_sized_executor(
        db,
        "u4",
        "QUOTE",
        "10",
        position="100",
        filled_quote="10",
        entry_price="0.1",
        open_base="100",
    )
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "u4")
    stored = json.loads(row.config)
    assert stored["size_unit"] == "BASE"
    assert stored["total_amount"] == "100"

    resumed = service._active_executors["u4"]
    resumed.update_config = lambda changes: {"accepted": True, "reason": ""}
    accepted = await service.update_range_iceberg_config("u4", {"total_amount": "300"})
    assert accepted["accepted"] is True
    await service._write_final_state("u4", {
        "side": "BUY",
        "size_unit": "BASE",
        "total_amount": "300",
        "open_base": "100",
        "filled_base": "100",
        "filled_quote": "10",
        "exit_state": "RESTING",
        "resume_on_startup": True,
        "exchange_stop_order_id": "stop-x",
        "pending_config_edit": False,
        "config_edits": {"total_amount": "300", "size_unit": "BASE"},
    })
    row = next(row for row in db.query(ExecutorRecord) if row.executor_id == "u4")
    stored = json.loads(row.config)
    state = json.loads(row.final_state)
    assert stored["size_unit"] == "BASE"
    assert stored["total_amount"] == "300"
    assert state["config_edits"]["size_unit"] == "BASE"
    assert state["config_edits"]["total_amount"] == "300"
    assert state["size_unit"] == "BASE"

    service._active_executors.pop("u4", None)
    service._executor_metadata.pop("u4", None)
    await service.cleanup_orphaned_executors()
    again = service._active_executors["u4"]
    assert again.config.size_unit == "BASE"
    assert Decimal(again.config.total_amount) == Decimal("300")
    assert connector.stop_cancels == []
    shown = service._format_db_record(row)
    assert shown["size_unit"] == "BASE"


def test_sibling_hints_shrink_only_when_the_live_book_is_short():
    """LIFO: the newest tier gives up size first. A sibling fill does not."""
    oldest = {
        "executor_id": "old",
        "created_at": 1,
        "open_base": Decimal("5"),
        "exchange_position_base": Decimal("7"),
    }
    mid = {
        "executor_id": "mid",
        "created_at": 2,
        "open_base": Decimal("3"),
        "exchange_position_base": Decimal("7"),
    }
    newest = {
        "executor_id": "new",
        "created_at": 3,
        "open_base": Decimal("2"),
        "exchange_position_base": Decimal("7"),
    }
    hints = range_iceberg_sibling_hints([newest, oldest, mid])
    assert hints == {
        "old": Decimal("2"),
        "mid": Decimal("5"),
        "new": Decimal("7"),
    }
    newest["open_base"] = Decimal("0")
    for row in (oldest, mid, newest):
        row["exchange_position_base"] = Decimal("8")
    after_fill = range_iceberg_sibling_hints([oldest, mid, newest])
    assert after_fill["old"] == Decimal("3")
    assert after_fill["mid"] == Decimal("5")
    assert after_fill["new"] == Decimal("8")
    assert range_iceberg_sibling_hints([oldest]) == {"old": None}
    stale_book = range_iceberg_sibling_hints([
        {
            "executor_id": "old",
            "created_at": 1,
            "open_base": Decimal("6"),
            "exchange_position_base": Decimal("10"),
        },
        {
            "executor_id": "new",
            "created_at": 2,
            "open_base": Decimal("1"),
            "exchange_position_base": Decimal("10"),
        },
    ])
    assert stale_book == {"old": Decimal("1"), "new": Decimal("6")}
    assert range_iceberg_sibling_hints([
        {
            "executor_id": "old",
            "created_at": 1,
            "open_base": Decimal("10"),
            "exchange_position_base": Decimal("8"),
        },
        {
            "executor_id": "new",
            "created_at": 2,
            "open_base": Decimal("10"),
            "exchange_position_base": Decimal("15"),
        },
    ]) == {}
    missing_open = dict(oldest)
    missing_open["open_base"] = None
    assert range_iceberg_sibling_hints([missing_open, mid]) == {}
    flat = dict(oldest)
    flat["exchange_position_base"] = Decimal("0")
    other = dict(mid)
    other["exchange_position_base"] = Decimal("0")
    assert range_iceberg_sibling_hints([flat, other]) == {}


def test_side_held_base_counts_the_shared_book_once():
    assert range_iceberg_side_held_base([
        {"open_base": "4", "exchange_position_base": "10"},
        {"open_base": "6", "exchange_position_base": "10"},
    ]) == Decimal("10")
    assert range_iceberg_side_held_base([
        {"open_base": "5", "exchange_position_base": "7"},
        {"open_base": "5", "exchange_position_base": "7"},
    ]) == Decimal("7")
    assert range_iceberg_held_base(
        None, {"open_base": "5", "exchange_position_base": "0.4"},
    ) == Decimal("0.4")


def _limit_tier(executor_id, limit, unspent=True):
    return {
        "executor_id": executor_id,
        "mode": "UP_TO",
        "limit_price": Decimal(limit),
        "unspent": unspent,
    }


def test_entry_pause_keeps_only_the_deepest_tier_the_price_qualifies_for():
    tiers = [_limit_tier("t1", "0.1"), _limit_tier("t2", "0.09"), _limit_tier("t3", "0.08")]
    assert range_iceberg_entry_pauses(tiers, "BUY", Decimal("0.095")) == {
        "t1": None, "t2": None, "t3": None,
    }
    at_second = range_iceberg_entry_pauses(tiers, "BUY", Decimal("0.089"))
    assert at_second == {
        "t1": "deeper tier t2 active",
        "t2": None,
        "t3": None,
    }
    at_third = range_iceberg_entry_pauses(tiers, "BUY", Decimal("0.079"))
    assert at_third["t1"] == "deeper tier t2 active"
    assert at_third["t2"] == "deeper tier t3 active"
    assert at_third["t3"] is None
    assert range_iceberg_entry_pauses(tiers, "BUY", Decimal("0.095"))["t1"] is None
    spent = [_limit_tier("t1", "0.1"), _limit_tier("t2", "0.09", False), _limit_tier("t3", "0.08")]
    assert range_iceberg_entry_pauses(spent, "BUY", Decimal("0.089")) == {
        "t1": None, "t2": None, "t3": None,
    }
    partial = [_limit_tier("t1", "0.1"), _limit_tier("t2", "0.09"), _limit_tier("t3", "0.08")]
    assert range_iceberg_entry_pauses(partial, "BUY", Decimal("0.089"))["t1"] == (
        "deeper tier t2 active"
    )
    held = [_limit_tier("t1", "0.1"), _limit_tier("t2", "0.09"), _limit_tier("t3", "0.08")]
    held[2]["eligible"] = False
    assert range_iceberg_entry_pauses(held, "BUY", Decimal("0.079")) == {
        "t1": "deeper tier t2 active",
        "t2": None,
    }
    assert range_iceberg_entry_pauses([_limit_tier("only", "0.1")], "BUY", Decimal("0.05")) == {
        "only": None,
    }
    assert range_iceberg_entry_pauses(tiers, "BUY", None) == {}

    sells = [_limit_tier("s1", "0.08"), _limit_tier("s2", "0.09"), _limit_tier("s3", "0.10")]
    mirrored = range_iceberg_entry_pauses(sells, "SELL", Decimal("0.095"))
    assert mirrored["s1"] == "deeper tier s2 active"
    assert mirrored["s2"] is None
    assert mirrored["s3"] is None
    above = range_iceberg_entry_pauses(sells, "SELL", Decimal("0.105"))
    assert above["s2"] == "deeper tier s3 active"
    assert above["s3"] is None

    ranges = [
        {
            "executor_id": "r1",
            "mode": "RANGE",
            "price_low": Decimal("0.05"),
            "price_high": Decimal("0.10"),
            "unspent": True,
        },
        {
            "executor_id": "r2",
            "mode": "RANGE",
            "price_low": Decimal("0.04"),
            "price_high": Decimal("0.09"),
            "unspent": True,
        },
    ]
    assert range_iceberg_entry_pauses(ranges, "BUY", Decimal("0.089"))["r1"] == (
        "deeper tier r2 active"
    )


def test_reference_price_uses_the_mid_then_the_mark():
    from services.executor_service import _connector_reference_price

    class Book:
        def get_mid_price(self, _pair):
            return Decimal("0.095")

    assert _connector_reference_price(Book(), "AIOT-USDT") == Decimal("0.095")

    class Mark:
        def get_mid_price(self, _pair):
            raise RuntimeError("no book")

        def get_funding_info(self, _pair):
            return SimpleNamespace(mark_price="0.09")

    assert _connector_reference_price(Mark(), "AIOT-USDT") == Decimal("0.09")
    assert _connector_reference_price(None, "AIOT-USDT") is None


@pytest.mark.asyncio
async def test_held_tiers_do_not_book_the_shared_position_twice():
    service = _service()
    infos = [
        {
            "open_base": "5",
            "exchange_position_base": "7",
            "side": "BUY",
            "average_fill_price": "2",
        },
        {
            "open_base": "5",
            "exchange_position_base": "7",
            "side": "BUY",
            "average_fill_price": "2",
        },
    ]
    metadata = _hold_metadata("range_iceberg_executor")
    for index, info in enumerate(infos):
        executor_id = f"tier-{index}"
        service._active_executors[executor_id] = _HoldExecutor(info)
        service._executor_metadata[executor_id] = metadata
    service._active_executors.pop("tier-0")
    await service._aggregate_position_hold("tier-0", _HoldExecutor(infos[0]), metadata)
    service._active_executors.pop("tier-1")
    await service._aggregate_position_hold("tier-1", _HoldExecutor(infos[1]), metadata)
    position = next(iter(service._positions_held.values()))
    assert position.buy_amount_base == Decimal("7")


@pytest.mark.asyncio
async def test_three_tiers_are_allowed_and_a_shared_stop_is_refused(db):
    service = _open_service(db)
    first = await service.create_executor(
        _open_config("BUY", stop_loss={"price": "0.07"}, group_id="aiot-long"),
        account_name="master",
    )
    assert service._executor_metadata[first["executor_id"]]["config"]["group_id"] == "aiot-long"
    second = await service.create_executor(
        _open_config("BUY", stop_loss={"price": "0.06"}),
        account_name="master",
    )
    third = await service.create_executor(
        _open_config("BUY", stop_loss={"price": "0.05"}),
        account_name="master",
    )
    assert len({first["executor_id"], second["executor_id"], third["executor_id"]}) == 3
    with pytest.raises(HTTPException) as duplicate:
        await service.create_executor(
            _open_config("BUY", stop_loss={"price": "0.070"}),
            account_name="master",
        )
    assert duplicate.value.status_code == 409
    assert "MEXC would overwrite" in duplicate.value.detail
    assert first["executor_id"] in duplicate.value.detail
    with pytest.raises(HTTPException) as patched:
        await service.update_range_iceberg_config(
            second["executor_id"],
            {"stop_loss": {"price": "0.07"}},
        )
    assert patched.value.status_code == 409
    assert "MEXC would overwrite" in patched.value.detail
    cleared = await service.update_range_iceberg_config(
        second["executor_id"],
        {"stop_loss": None},
    )
    assert cleared["accepted"] is True
    percent = await service.create_executor(
        _open_config("BUY", stop_loss={"percent": "0.02"}),
        account_name="master",
    )
    assert percent["executor_id"] not in {
        first["executor_id"], second["executor_id"], third["executor_id"],
    }


@pytest.mark.asyncio
async def test_the_eleventh_tier_is_refused_until_an_external_close(db):
    service = _open_service(db)
    ids = []
    for _ in range(10):
        created = await service.create_executor(_open_config("BUY"), account_name="master")
        ids.append(created["executor_id"])
    service._active_executors[ids[0]].reported["paused"] = True
    service._active_executors[ids[1]].reported["entry_paused"] = True
    with pytest.raises(HTTPException) as caught:
        await service.create_executor(_open_config("BUY"), account_name="master")
    assert caught.value.status_code == 409
    assert caught.value.detail == _cap_detail()
    executor = service._active_executors[ids[0]]
    executor.close_type = SimpleNamespace(name="POSITION_CLOSED_EXTERNALLY")
    executor.reported["external_close"] = True
    executor.reported["open_base"] = "5"
    opened = await service.create_executor(_open_config("BUY"), account_name="master")
    assert opened["executor_id"] not in ids
    assert ids[0] not in await _open_ids(service)


@pytest.mark.asyncio
async def test_sibling_hint_follows_a_partial_close_and_not_a_sibling_fill(db):
    service = _open_service(db)
    ids = []
    for _ in range(3):
        created = await service.create_executor(_open_config("BUY"), account_name="master")
        ids.append(created["executor_id"])
    for executor_id, opened in zip(ids, ("5", "3", "2")):
        executor = service._active_executors[executor_id]
        executor.reported["open_base"] = opened
        executor.reported["exchange_position_base"] = "7"
    service._sync_range_iceberg_sibling_hints()
    sent = [service._active_executors[executor_id].sibling_calls[-1] for executor_id in ids]
    assert sent == [Decimal("2"), Decimal("5"), Decimal("7")]
    lengths = [len(service._active_executors[executor_id].sibling_calls) for executor_id in ids]
    service._sync_range_iceberg_sibling_hints()
    assert [len(service._active_executors[executor_id].sibling_calls) for executor_id in ids] == lengths

    newest = service._active_executors[ids[2]]
    newest.reported["open_base"] = "0"
    for executor_id in ids:
        service._active_executors[executor_id].reported["exchange_position_base"] = "8"
    service._sync_range_iceberg_sibling_hints()
    sent = [service._active_executors[executor_id].sibling_calls[-1] for executor_id in ids]
    assert sent == [Decimal("3"), Decimal("5"), Decimal("8")]


@pytest.mark.asyncio
async def test_a_dropped_book_resends_an_unchanged_sibling_hint(db):
    service = _open_service(db)
    ids = []
    for _ in range(2):
        created = await service.create_executor(_open_config("BUY"), account_name="master")
        ids.append(created["executor_id"])
    for executor_id, opened in zip(ids, ("6", "4")):
        executor = service._active_executors[executor_id]
        executor.reported["open_base"] = opened
        executor.reported["exchange_position_base"] = "10"
    service._sync_range_iceberg_sibling_hints()
    oldest = service._active_executors[ids[0]]
    newest = service._active_executors[ids[1]]
    assert oldest.sibling_calls == [Decimal("4")]
    assert newest.sibling_calls == [Decimal("6")]
    service._sync_range_iceberg_sibling_hints()
    assert newest.sibling_calls == [Decimal("6")]

    newest.reported["open_base"] = "1"
    service._sync_range_iceberg_sibling_hints()
    assert oldest.sibling_calls[-1] == Decimal("1")
    assert newest.sibling_calls == [Decimal("6")]

    for executor_id in ids:
        service._active_executors[executor_id].reported["exchange_position_base"] = "7"
    service._sync_range_iceberg_sibling_hints()
    assert oldest.sibling_calls[-1] == Decimal("1")
    assert newest.sibling_calls[-1] == Decimal("6")
    assert len(newest.sibling_calls) == 2


async def _limit_executors(db, side, limits):
    service = _open_service(db)
    ids = []
    for limit in limits:
        created = await service.create_executor(
            _open_config(side, mode="UP_TO", limit_price=limit, total_amount="10"),
            account_name="master",
        )
        executor_id = created["executor_id"]
        ids.append(executor_id)
        executor = service._active_executors[executor_id]
        executor.reported["unfilled_amount"] = "10"
        executor.reported["mode"] = "UP_TO"
        executor.reported["limit_price"] = limit
    return service, ids


@pytest.mark.asyncio
async def test_entry_pause_dwells_and_leaves_an_unknown_price_alone(db):
    service, ids = await _limit_executors(db, "BUY", ("0.1", "0.09", "0.08"))
    prices = {"value": Decimal("0.095")}
    service._range_iceberg_reference_price = lambda *_args: prices["value"]

    def calls(index):
        return service._active_executors[ids[index]].entry_calls

    service._sync_range_iceberg_entry_pauses()
    assert calls(0) == []
    service._sync_range_iceberg_entry_pauses()
    assert calls(0) == [(False, None)]
    assert calls(1) == [(False, None)]

    prices["value"] = Decimal("0.089")
    service._sync_range_iceberg_entry_pauses()
    assert calls(0) == [(False, None)]
    service._sync_range_iceberg_entry_pauses()
    assert calls(0)[-1] == (True, f"deeper tier {ids[1]} active")
    service._active_executors[ids[0]].reported["entry_paused"] = False
    service._active_executors[ids[0]].reported["pause_reason"] = None
    service._sync_range_iceberg_entry_pauses()
    assert len(calls(0)) == 2
    service._sync_range_iceberg_entry_pauses()
    assert calls(0)[-1] == (True, f"deeper tier {ids[1]} active")
    assert len(calls(0)) == 3

    prices["value"] = None
    frozen = list(calls(0))
    service._sync_range_iceberg_entry_pauses()
    assert calls(0) == frozen

    prices["value"] = Decimal("0.095")
    service._sync_range_iceberg_entry_pauses()
    assert calls(0) == frozen
    prices["value"] = Decimal("0.089")
    service._sync_range_iceberg_entry_pauses()
    assert calls(0) == frozen

    service._active_executors[ids[1]].reported["unfilled_amount"] = "0"
    prices["value"] = Decimal("0.089")
    service._sync_range_iceberg_entry_pauses()
    service._sync_range_iceberg_entry_pauses()
    assert calls(0)[-1] == (False, None)

    service._active_executors[ids[1]].reported["unfilled_amount"] = "10"
    prices["value"] = Decimal("0.079")
    service._sync_range_iceberg_entry_pauses()
    service._sync_range_iceberg_entry_pauses()
    assert calls(0)[-1] == (True, f"deeper tier {ids[1]} active")
    assert calls(1)[-1] == (True, f"deeper tier {ids[2]} active")
    assert calls(2)[-1] == (False, None)

    prices["value"] = Decimal("0.095")
    service._sync_range_iceberg_entry_pauses()
    service._sync_range_iceberg_entry_pauses()
    assert calls(0)[-1] == (False, None)
    assert calls(1)[-1] == (False, None)


@pytest.mark.asyncio
async def test_entry_pause_mirrors_sell_limits(db):
    service, ids = await _limit_executors(db, "SELL", ("0.08", "0.09", "0.10"))
    prices = {"value": Decimal("0.095")}
    service._range_iceberg_reference_price = lambda *_args: prices["value"]
    service._sync_range_iceberg_entry_pauses()
    service._sync_range_iceberg_entry_pauses()
    calls = service._active_executors[ids[0]].entry_calls
    assert calls[-1] == (True, f"deeper tier {ids[1]} active")
    assert service._active_executors[ids[1]].entry_calls[-1] == (False, None)


@pytest.mark.asyncio
async def test_a_paused_or_halted_deeper_tier_does_not_silence_the_side(db):
    service, ids = await _limit_executors(db, "BUY", ("0.1", "0.09", "0.08"))
    deeper = service._active_executors[ids[2]]
    deeper.reported["paused"] = True
    deeper.reported["unknown_order_halt"] = True
    service._active_executors[ids[1]].reported["unfilled_amount"] = "4"
    prices = {"value": Decimal("0.079")}
    service._range_iceberg_reference_price = lambda *_args: prices["value"]
    service._sync_range_iceberg_entry_pauses()
    service._sync_range_iceberg_entry_pauses()
    assert service._active_executors[ids[0]].entry_calls[-1] == (
        True, f"deeper tier {ids[1]} active",
    )
    assert service._active_executors[ids[1]].entry_calls[-1] == (False, None)
    assert deeper.entry_calls == []

    service._entry_pause_sent.clear()
    service._entry_pause_streak.clear()
    before = list(service._active_executors[ids[0]].entry_calls)
    service._sync_range_iceberg_entry_pauses()
    assert service._active_executors[ids[0]].entry_calls == before
    service._sync_range_iceberg_entry_pauses()
    assert service._active_executors[ids[0]].entry_calls[-1] == before[-1]


@pytest.mark.asyncio
async def test_an_exit_only_tier_does_not_block_a_shallower_entry(db):
    service, ids = await _limit_executors(db, "BUY", ("0.1", "0.09"))
    service._executor_metadata[ids[1]]["config"]["resume_entries"] = False
    prices = {"value": Decimal("0.089")}
    service._range_iceberg_reference_price = lambda *_args: prices["value"]
    service._sync_range_iceberg_entry_pauses()
    service._sync_range_iceberg_entry_pauses()
    assert service._active_executors[ids[0]].entry_calls[-1] == (False, None)
    assert service._active_executors[ids[1]].entry_calls == []


def test_a_failed_mid_does_not_switch_to_the_mark_and_a_stale_mid_is_unknown():
    service = _service()

    class Conn:
        def __init__(self):
            self.mid = Decimal("0.095")
            self.mark = Decimal("0.095")
            self.raise_mid = False

        def get_mid_price(self, _pair):
            if self.raise_mid:
                raise RuntimeError("book down")
            return self.mid

        def get_funding_info(self, _pair):
            return SimpleNamespace(mark_price=self.mark)

    conn = Conn()
    _use_book(service, conn)
    assert service._range_iceberg_reference_price(
        "master", "mexc_perpetual", "BTC-USDT",
    ) == Decimal("0.095")
    conn.mark = Decimal("0.08")
    assert service._range_iceberg_reference_price(
        "master", "mexc_perpetual", "BTC-USDT",
    ) is None
    conn.raise_mid = True
    conn.mark = Decimal("0.089")
    assert service._range_iceberg_reference_price(
        "master", "mexc_perpetual", "BTC-USDT",
    ) is None

    fresh = _service()
    mark_only = Conn()
    mark_only.raise_mid = True
    mark_only.mark = Decimal("0.09")
    _use_book(fresh, mark_only)
    assert fresh._range_iceberg_reference_price(
        "master", "mexc_perpetual", "BTC-USDT",
    ) == Decimal("0.09")
