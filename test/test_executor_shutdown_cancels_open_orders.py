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
that cleanup. Every other leftover is still cancelled.
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
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from database.models import ExecutorOrder, ExecutorPerformanceSnapshot, ExecutorRecord, Order
from database.repositories.executor_repository import ExecutorRepository
from database.repositories.order_repository import OrderRepository
from services.executor_service import (
    ExecutorService,
    child_order_ids_from_custom_info,
    range_iceberg_held_base,
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
        self.closes = []
        self.confirmed = confirmed
        self._confirm_events = None

    def is_cancel_confirmed(self, order_id):
        confirmed = self.confirmed is True or (isinstance(self.confirmed, set) and order_id in self.confirmed)
        if confirmed and self._confirm_events is not None:
            self._confirm_events.append("confirmed")
            self._confirm_events = None
        return confirmed

    async def cancel(self, trading_pair=None, client_order_id=None):
        self.cancels.append((trading_pair, client_order_id))

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


class _CancellingExecutor(_Executor):
    """early_stop stands in for the real call that pulls the exchange stop."""

    def early_stop(self, keep_position=True):
        super().early_stop(keep_position=keep_position)
        connector = next(iter(self.connectors.values()), None)
        if connector is not None:
            connector.cancels.append(("BTC-USDT", "stop-1"))


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
        assert executor.early_kwargs == []
        assert connector.cancels == []
        assert "early_stop" not in events
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
        assert connector.cancels == [("BTC-USDT", "stop-1")]
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
            "exchange_stop": True,
            "exit_retry_budget": 4,
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
    for order_id in ("entry-1", "tp-child", "stop-1", "crash-entry", "crash-tp", "crash-stop", "closed-stop", "naked-stop"):
        connector.in_flight_orders[order_id] = _open_order()
        await _seed_exchange_order(db, order_id, "OPEN")

    service = _service(db, {"master": {"mexc_perpetual": connector}})
    _install_resume_factory(service)

    with caplog.at_level(logging.INFO):
        await service.cleanup_orphaned_executors()

    cancelled = {order_id for _pair, order_id in connector.cancels}
    assert "entry-1" not in cancelled
    assert "tp-child" not in cancelled
    assert "stop-1" not in cancelled
    assert "crash-tp" in cancelled
    assert "crash-entry" in cancelled
    assert "crash-stop" not in cancelled
    assert "closed-stop" in cancelled
    assert "naked-stop" not in cancelled
    assert (
        "Cancelled exchange stop closed-stop for executor closed-1 "
        "on mexc_perpetual BTC-USDT because open_base is 0"
    ) in caplog.text
    assert "Resumed range iceberg executors before orphan cleanup: resume-1" in caplog.text

    resumed = service._active_executors["resume-1"]
    assert resumed.started is True
    assert resumed.restored["exit_state"] == "RESTING"
    assert resumed.restored["exit_order_ids"] == ["tp-child"]
    assert "resume_on_startup" not in resumed.restored
    assert resumed.config.id == "resume-1"
    assert resumed.config.exchange_stop is True
    assert resumed.config.exit_retry_budget == 4

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
            "exchange_stop_order_id": "stop-fail",
            "open_base": "1",
        },
    )
    connector = _Connector()
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


def test_validate_forwards_exchange_stop_and_exit_retry_budget():
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
            "exchange_stop": True,
            "exit_retry_budget": 4,
            "timestamp": 10,
        }
    )
    assert typed.exchange_stop is True
    assert typed.exit_retry_budget == 4
    assert typed.connector_name == "mexc_perpetual"
