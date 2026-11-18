"""Open executor orders are cancelled on the way down, and again on the way back up.

`ExecutorService.stop` used to call `executor.stop()`, which drops the task and leaves
working orders on the exchange. The next boot then marked every leftover RUNNING row
SYSTEM_CLEANUP and never looked at those orders again. Graceful shutdown now asks each
executor for `early_stop(keep_position=True)` (cancel working orders, do not close a
fill) and waits, up to a bound, for cancel confirmation before trading interfaces are
torn down. An unclean shutdown leaves the row RUNNING; startup cancels child orders
still open on the exchange, using the persisted executor_orders rows and custom_info,
and only then marks the row SYSTEM_CLEANUP. Nothing here resumes an executor.
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
from services.executor_service import ExecutorService, child_order_ids_from_custom_info


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
