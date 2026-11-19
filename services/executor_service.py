"""
ExecutorService manages executor lifecycle and orchestration.
This service enables running Hummingbot executors directly via API
without Docker containers or full strategy setup.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Type

from fastapi import HTTPException
from hummingbot.strategy_v2.executors.arbitrage_executor.arbitrage_executor import ArbitrageExecutor
from hummingbot.strategy_v2.executors.arbitrage_executor.data_types import ArbitrageExecutorConfig
from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.strategy_v2.executors.dca_executor.dca_executor import DCAExecutor
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.grid_executor.data_types import GridExecutorConfig
from hummingbot.strategy_v2.executors.grid_executor.grid_executor import GridExecutor
from hummingbot.strategy_v2.executors.lp_executor.data_types import LPExecutorConfig
from hummingbot.strategy_v2.executors.lp_executor.lp_executor import LPExecutor
from hummingbot.strategy_v2.executors.order_executor.data_types import OrderExecutorConfig
from hummingbot.strategy_v2.executors.order_executor.order_executor import OrderExecutor
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.position_executor import PositionExecutor
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import RangeIcebergExecutorConfig
from hummingbot.strategy_v2.executors.range_iceberg_executor.range_iceberg_executor import RangeIcebergExecutor
from hummingbot.strategy_v2.executors.twap_executor.data_types import TWAPExecutorConfig
from hummingbot.strategy_v2.executors.twap_executor.twap_executor import TWAPExecutor
from hummingbot.strategy_v2.executors.xemm_executor.data_types import XEMMExecutorConfig
from hummingbot.strategy_v2.executors.xemm_executor.xemm_executor import XEMMExecutor
from hummingbot.strategy_v2.models.executors import CloseType, TrackedOrder
from sqlalchemy.exc import IntegrityError

from database import (
    AsyncDatabaseManager,
    ExecutorPerformanceRepository,
    ExecutorRepository,
    GatewayCLMMRepository,
    GatewaySwapRepository,
    OrderRepository,
)
from models.executors import PositionHold
from services.gateway_client import get_native_gas_token
from services.trading_service import AccountTradingInterface, TradingService
from utils.executor_log_capture import ExecutorLogCapture, current_executor_id
from utils.trading_pair import InvalidTradingPair, split_trading_pair

logger = logging.getLogger(__name__)

# Order rows that no longer need a cancel. DONE is the marker this service writes when a
# child id drops out of the executor's working set between snapshots.
_TERMINAL_ORDER_STATUSES = {
    "FILLED",
    "CANCELED",
    "CANCELLED",
    "FAILED",
    "COMPLETED",
    "DONE",
}
_SKIP_ORDER_IDS = {"", "skipped"}
_ORDER_LIST_KEYS = (
    "child_order_ids",
    "order_ids",
    "unknown_order_ids",
    "active_order_ids",
    # Take-profit and stop-loss children, including iceberg exit slices.
    "exit_order_ids",
)
_LEVEL_KEYS = ("levels", "orders")
# Graceful shutdown persists these so startup can restore the executor and its
# exchange-side stop. TRIGGERED, DONE, and NONE stay on the ordinary cancel path.
_RESUMABLE_EXIT_STATES = frozenset({"ARMED", "RESTING", "CLOSING"})
_RESUME_ON_STARTUP = "resume_on_startup"
_RESUME_FAILED = "resume_failed"
_RESUME_ERROR = "resume_error"
_RESUME_ENTRIES = "resume_entries"
_CLOSED_WHILE_DOWN = "CLOSED_WHILE_DOWN"
_EXCHANGE_STOP_CANCELLED = "exchange_stop_cancelled"
_API_RESUME_KEYS = frozenset({_RESUME_ON_STARTUP, _RESUME_FAILED, _RESUME_ERROR})


def _order_id_text(value: Any) -> Optional[str]:
    """A client order id worth tracking, or None for blanks and placeholders."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in _SKIP_ORDER_IDS:
        return None
    return text


def _dict_order_id(item: Dict[str, Any]) -> Optional[str]:
    return _order_id_text(item.get("order_id") or item.get("client_order_id"))


def _is_filled_marker(item: Dict[str, Any]) -> bool:
    return item.get("filled") is True or item.get("is_filled") is True


def child_order_ids_from_custom_info(custom_info: Any) -> tuple[List[str], List[str]]:
    """Split custom_info into working child order ids and filled ones.

    Range iceberg reports ``child_order_ids`` plus ``exit_order_ids`` (every exit
    client order id, including iceberg children) and per-level ``filled`` flags.
    Position, DCA and TWAP report ``order_ids``. Order executor reports a single
    ``order_id``. Filled ids and ``held_position_orders`` are the exposure to keep,
    not orders to cancel. ``exchange_stop_order_id`` and ``exchange_stop_order_ids``
    are exchange stops, not software orders to cancel here.
    """
    if not isinstance(custom_info, dict):
        return [], []

    filled: List[str] = []
    seen_filled: set = set()
    working: List[str] = []
    seen_working: set = set()

    def mark_filled(value: Any) -> None:
        oid = _order_id_text(value) if not isinstance(value, dict) else _dict_order_id(value)
        if oid and oid not in seen_filled:
            seen_filled.add(oid)
            filled.append(oid)

    def mark_working(value: Any) -> None:
        oid = _order_id_text(value) if not isinstance(value, dict) else _dict_order_id(value)
        if oid and oid not in seen_filled and oid not in seen_working:
            seen_working.add(oid)
            working.append(oid)

    for item in custom_info.get("held_position_orders") or []:
        mark_filled(item)

    for key in _LEVEL_KEYS:
        for item in custom_info.get(key) or []:
            if not isinstance(item, dict):
                continue
            if _is_filled_marker(item):
                mark_filled(item)
            else:
                mark_working(item)

    for key in _ORDER_LIST_KEYS:
        value = custom_info.get(key)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict) and _is_filled_marker(item):
                mark_filled(item)
            else:
                mark_working(item)

    single = custom_info.get("order_id")
    if isinstance(single, str):
        mark_working(single)

    working = [oid for oid in working if oid not in seen_filled]
    return working, filled


def _parse_stored_custom_info(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _exit_state_name(custom_info: Any) -> Optional[str]:
    """``exit_state`` uppercased, or None when the field is absent."""
    if not isinstance(custom_info, dict) or "exit_state" not in custom_info:
        return None
    value = custom_info.get("exit_state")
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        text = name
    else:
        raw = getattr(value, "value", value)
        text = raw if isinstance(raw, str) else str(raw)
    text = text.strip().upper()
    return text or None


def _exchange_stop_ids(custom_info: Any) -> List[str]:
    """Every live exchange stop id for this executor.

    ``exchange_stop_order_ids`` is the full list. The primary
    ``exchange_stop_order_id`` is included when that list is absent or does
    not already contain it.
    """
    if not isinstance(custom_info, dict):
        return []
    ids: List[str] = []
    seen: set = set()

    def add(value: Any) -> None:
        oid = _order_id_text(value) if not isinstance(value, dict) else _dict_order_id(value)
        if oid and oid not in seen:
            seen.add(oid)
            ids.append(oid)

    listed = custom_info.get("exchange_stop_order_ids")
    if isinstance(listed, list):
        for item in listed:
            add(item)
    else:
        add(custom_info.get("exchange_stop_order_id"))
        return ids
    add(custom_info.get("exchange_stop_order_id"))
    return ids


def _exchange_stop_id(custom_info: Any) -> Optional[str]:
    """Primary exchange stop id, or the first id when only the list is set."""
    if isinstance(custom_info, dict):
        primary = _order_id_text(custom_info.get("exchange_stop_order_id"))
        if primary:
            return primary
    ids = _exchange_stop_ids(custom_info)
    return ids[0] if ids else None


def _exchange_stop_text(custom_info: Any) -> str:
    ids = _exchange_stop_ids(custom_info)
    return ", ".join(ids) if ids else "none"


def _range_iceberg_side_name(value: Any) -> str:
    """Position side for the one-open-iceberg slot.

    BUY and LONG are one side. SELL and SHORT are the other. A missing
    side counts as BUY, which is the executor default.
    """
    name = getattr(value, "name", value)
    text = str(name or "").strip().upper()
    if text in {"SELL", "SHORT"}:
        return "SELL"
    return "BUY"


def _normalize_trading_pair(value: Any) -> str:
    """Upper case, with '_' and '/' written as '-'.

    ``btc-usdt``, ``btc_usdt`` and ``BTC/USDT`` are one pair. An empty
    value stays empty.
    """
    return str(value or "").strip().upper().replace("_", "-").replace("/", "-")


def _open_base_is_zero(custom_info: Any) -> bool:
    """True when ``open_base`` is present and zero.

    A missing field is not treated as a closed position.
    """
    if not isinstance(custom_info, dict) or "open_base" not in custom_info:
        return False
    value = custom_info.get("open_base")
    if value is None or value == "":
        return False
    try:
        amount = Decimal(str(value))
    except Exception:
        return False
    return amount == 0


def _reported_decimal(executor: Any, custom_info: Any, key: str) -> Optional[Decimal]:
    if isinstance(custom_info, dict) and key in custom_info and custom_info.get(key) is not None:
        value = custom_info.get(key)
    elif executor is not None and hasattr(executor, key):
        value = getattr(executor, key)
        if callable(value):
            return None
    else:
        return None
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def range_iceberg_held_base(executor: Any, custom_info: Any) -> Optional[Decimal]:
    """Base still held by a range iceberg.

    ``open_base`` when that field is present. Otherwise ``filled_base`` minus
    ``closed_base`` when both are present. Otherwise ``filled_base``. None when
    none of those fields were reported, so the caller can use the generic fill.
    """
    open_base = _reported_decimal(executor, custom_info, "open_base")
    if open_base is not None:
        return open_base if open_base > 0 else Decimal("0")
    filled = _reported_decimal(executor, custom_info, "filled_base")
    closed = _reported_decimal(executor, custom_info, "closed_base")
    if filled is not None and closed is not None:
        net = filled - closed
        return net if net > 0 else Decimal("0")
    if filled is not None:
        return filled if filled > 0 else Decimal("0")
    return None


def _stored_position_hold_keeps_slot(custom_info: Any) -> bool:
    """A parked hold still has base, or it still names an exchange stop."""
    if not isinstance(custom_info, dict):
        return False
    held = range_iceberg_held_base(None, custom_info)
    if held is not None and held > 0:
        return True
    return bool(_exchange_stop_ids(custom_info))


def range_iceberg_held_quote(executor: Any, custom_info: Any, held_base: Optional[Decimal]) -> Optional[Decimal]:
    """Quote still held, sized to ``held_base`` times the average entry.

    The executor's filled quote is the whole entry. A partly closed iceberg
    keeps only the open base, so the quote is that base times the average
    entry price. None when the average was not reported.
    """
    if held_base is None:
        return None
    if held_base <= 0:
        return Decimal("0")
    average = _average_entry_price(executor, custom_info)
    if average is None or average <= 0:
        return None
    return held_base * average


def _average_entry_price(executor: Any, custom_info: Any) -> Optional[Decimal]:
    for key in ("average_fill_price", "average_entry_price"):
        price = _reported_decimal(executor, custom_info, key)
        if price is not None and price > 0:
            return price
    filled = _reported_decimal(executor, custom_info, "filled_base")
    if filled is None or filled <= 0:
        filled = _reported_decimal(executor, custom_info, "filled_amount_base")
    quote = _reported_decimal(executor, custom_info, "filled_quote")
    if quote is None:
        quote = _reported_decimal(executor, custom_info, "filled_amount_quote")
    if quote is None:
        quote = _reported_decimal(executor, custom_info, "entry_filled_quote")
    if filled is not None and filled > 0 and quote is not None and quote > 0:
        return quote / filled
    return None


def _entry_average_from_saved(custom: Dict[str, Any]) -> Optional[Decimal]:
    """Average price of the entry fills, ignoring exit fills.

    A partial take profit stores its own quote on the exit. Dividing that
    quote by the remaining base is not the entry average.
    """
    filled = _reported_decimal(None, custom, "filled_base")
    if filled is None or filled <= 0:
        filled = _reported_decimal(None, custom, "filled_amount_base")
    quote = _reported_decimal(None, custom, "filled_quote")
    if quote is None or quote <= 0:
        quote = _reported_decimal(None, custom, "filled_amount_quote")
    exit_blob = custom.get("exit") if isinstance(custom.get("exit"), dict) else {}
    if (filled is None or filled <= 0) and exit_blob:
        filled = _reported_decimal(None, exit_blob, "entry_filled_base")
    if (quote is None or quote <= 0) and exit_blob:
        quote = _reported_decimal(None, exit_blob, "entry_filled_quote")
    if filled is not None and filled > 0 and quote is not None and quote > 0:
        return quote / filled
    for key in ("average_fill_price", "average_entry_price"):
        price = _reported_decimal(None, custom, key)
        if price is not None and price > 0:
            return price
    return None


def _prices_are_consistent(left: Decimal, right: Decimal) -> bool:
    scale = max(abs(left), abs(right))
    if scale <= 0:
        return False
    return abs(left - right) / scale <= Decimal("0.01")


def _resume_entry_average(
    custom: Dict[str, Any],
    exchange_entry_price: Optional[Decimal],
) -> Optional[Decimal]:
    """Entry average for the restored position.

    Prefer the saved entry fills. Use the exchange position's entry price when
    it is the only price we have, or when it agrees with those fills. A price
    that disagrees is another position on the same side, so the saved fills win.
    """
    saved = _entry_average_from_saved(custom)
    exchange = exchange_entry_price if exchange_entry_price is not None and exchange_entry_price > 0 else None
    if saved is not None and exchange is not None and not _prices_are_consistent(saved, exchange):
        logger.warning(
            "Exchange entry price %s does not match saved entry average %s. "
            "Resuming on the saved entry average.",
            exchange,
            saved,
        )
        return saved
    if saved is not None:
        return saved
    return exchange


def _exit_phase_for_state(exit_state: Any) -> str:
    name = str(exit_state or "").strip().upper()
    return {
        "ARMED": "armed",
        "RESTING": "resting",
        "TRIGGERED": "cancelling_entry",
        "CLOSING": "executing",
        "DONE": "done",
    }.get(name, "armed")


def _exit_only_custom_info(
    custom: Dict[str, Any],
    open_base: Decimal,
    average: Optional[Decimal],
) -> Dict[str, Any]:
    """Clean restore payload that continues exits only.

    A resumed range iceberg does not continue an unfinished entry. The payload
    is not a copy of the saved custom_info: a partial take profit's
    filled_by_order and exit quote would make the remaining base look like it
    was entered at the wrong price, and the stop would fire at once.

    ``size_unit`` is BASE. ``open_base`` is the position the exit may close.
    Entry quote is that base times the entry average. Startup still cancels
    the old client order ids from the saved row. ``exchange_stop_released``,
    when the saved info has it, is copied onto the top level and the exit
    section so a resumed stop-loss close does not place a new exchange stop.
    """
    amount = format(open_base, "f")
    quote = "0" if average is None or average <= 0 else format(open_base * average, "f")
    average_text = None if average is None or average <= 0 else format(average, "f")
    exit_blob = custom.get("exit") if isinstance(custom.get("exit"), dict) else {}
    exit_state = custom.get("exit_state") or exit_blob.get("exit_state")
    active = custom.get("exit_active") or exit_blob.get("active")
    stop_ids = _exchange_stop_ids(custom)
    payload: Dict[str, Any] = {
        _RESUME_ENTRIES: False,
        "size_unit": "BASE",
        "side": custom.get("side"),
        "mode": custom.get("mode"),
        "exit_state": exit_state,
        "exit_active": active,
        "open_base": amount,
        "filled_base": amount,
        "closed_base": "0",
        "filled_quote": quote,
        "average_fill_price": average_text,
        "exchange_stop_order_id": _exchange_stop_id(custom),
        "exchange_stop_order_ids": stop_ids,
        "child_order_ids": [],
        "exit_order_ids": [],
    }
    if "schedule_clock" in custom:
        payload["schedule_clock"] = custom.get("schedule_clock")
    if "schedule_started_at" in custom:
        payload["schedule_started_at"] = custom.get("schedule_started_at")
    payload["exit"] = {
        "active": active,
        "status": _exit_phase_for_state(exit_state),
        "exit_state": exit_state,
        "exchange_stop_order_id": _exchange_stop_id(custom),
        "exchange_stop_order_ids": stop_ids,
        "trigger_price": exit_blob.get("trigger_price"),
        "entry_filled_base": amount,
        "entry_filled_quote": quote,
        "filled_base": "0",
        "filled_quote": "0",
        "average_price": average_text,
        "order_ids": [],
        "filled_by_order": {},
        "started_at": exit_blob.get("started_at"),
        "paused_seconds": exit_blob.get("paused_seconds") or 0,
    }
    if "exchange_stop_released" in custom:
        released = custom.get("exchange_stop_released")
    elif "exchange_stop_released" in exit_blob:
        released = exit_blob.get("exchange_stop_released")
    else:
        released = None
    if "exchange_stop_released" in custom or "exchange_stop_released" in exit_blob:
        payload["exchange_stop_released"] = released
        payload["exit"]["exchange_stop_released"] = released
    return payload


def _resume_search_fields(custom_info: Any) -> Dict[str, Any]:
    """Fields Condor reads from an executor search row."""
    info = custom_info if isinstance(custom_info, dict) else {}
    error = info.get("exchange_stop_error")
    resume_error = info.get(_RESUME_ERROR)
    return {
        "resume_failed": bool(info.get(_RESUME_FAILED)),
        "resume_error": None if resume_error in (None, "") else str(resume_error),
        "exchange_stop_error": None if error in (None, "") else str(error),
    }


def _live_order_ids_from_custom(custom_info: Any) -> List[str]:
    working, _filled = child_order_ids_from_custom_info(custom_info)
    ids = list(working)
    for stop_id in _exchange_stop_ids(custom_info):
        if stop_id not in ids:
            ids.append(stop_id)
    return ids


def _status_name(executor: Any) -> str:
    status = getattr(executor, "status", None)
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    return ""


def _inflight_state_name(order: Any) -> str:
    state = getattr(order, "current_state", None)
    name = getattr(state, "name", None)
    return name if isinstance(name, str) else ""


def _inflight_is_open(order: Any) -> bool:
    """True when an in-flight order can still be working on the exchange."""
    if order is None:
        return False
    if getattr(order, "is_failure", False):
        return False
    if getattr(order, "is_cancelled", False) or getattr(order, "is_canceled", False):
        return False
    state_name = _inflight_state_name(order)
    if state_name in _TERMINAL_ORDER_STATUSES:
        return False
    if getattr(order, "is_filled", False) and getattr(order, "is_done", False):
        return False
    return True


def _order_status_is_open(status: Optional[str]) -> bool:
    if not status:
        return False
    return status.upper() not in _TERMINAL_ORDER_STATUSES


def _json_default(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.name
    if isinstance(obj, TrackedOrder):
        return {
            "order_id": obj.order_id,
            "price": float(obj.price) if obj.price else None,
            "executed_amount_base": float(obj.executed_amount_base) if obj.executed_amount_base else 0.0,
            "executed_amount_quote": float(obj.executed_amount_quote) if obj.executed_amount_quote else 0.0,
            "is_filled": obj.is_filled if hasattr(obj, 'is_filled') else False,
            "is_open": obj.is_open if hasattr(obj, 'is_open') else False,
        }
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        return obj.model_dump(mode='json')
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _coerce_json_compatible(obj):
    """Recursively coerce a value into JSON-compatible primitives.

    Mirrors the result of ``json.loads(json.dumps(obj, default=_json_default))``
    without the string round-trip: containers are walked recursively and any
    object handled by ``_json_default`` is coerced to the same output type.
    """
    # JSON-native primitives are returned as-is.
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        # json.dumps coerces non-string scalar keys (int/float/bool/None) to
        # strings; replicate that so the output shape is identical.
        coerced = {}
        for key, value in obj.items():
            if isinstance(key, str):
                str_key = key
            elif isinstance(key, bool):
                str_key = "true" if key else "false"
            elif key is None:
                str_key = "null"
            elif isinstance(key, (int, float)):
                str_key = json.dumps(key)
            else:
                raise TypeError(
                    f"keys must be str, int, float, bool or None, not {type(key).__name__}"
                )
            coerced[str_key] = _coerce_json_compatible(value)
        return coerced
    if isinstance(obj, (list, tuple)):
        # json.dumps serializes tuples as JSON arrays (-> lists on decode).
        return [_coerce_json_compatible(item) for item in obj]
    # Non-native types: route through the same coercion as the JSON encoder,
    # then recurse into the (possibly nested) replacement value.
    return _coerce_json_compatible(_json_default(obj))


class ExecutorService:
    """
    Service for managing trading executors without Docker containers.

    This service provides:
    - Dynamic executor creation for any market/connector
    - Executor lifecycle management (start, stop, cleanup)
    - Real-time executor status monitoring
    - Database persistence of executor state and history
    """

    # How long to wait before looking again for a position row that was not there. The
    # poller creates it on discovery, which is far slower than the control loop's tick.
    LP_RENT_RETRY_SECONDS = 30.0

    # How often the retention sweep runs, once it is enabled at all. Deleting rows older
    # than N days does not get more correct for being done every minute, and the sweep is
    # a DELETE over two tables sharing the control loop's tick.
    PERFORMANCE_PRUNE_INTERVAL_SECONDS = 3600.0

    # Graceful shutdown asks each executor to cancel working orders, then waits until the
    # connector reports those cancels confirmed. The cap bounds stall time so a hung
    # exchange cannot hold connector teardown open. MEXC perpetual exposes
    # is_cancel_confirmed; connectors without that hook are not waited on.
    SHUTDOWN_CANCEL_TIMEOUT_SECONDS = 30.0
    SHUTDOWN_CANCEL_POLL_SECONDS = 0.25

    # Mapping of executor type strings to (executor_class, config_class)
    EXECUTOR_REGISTRY: Dict[str, tuple[Type[ExecutorBase], Type[ExecutorConfigBase]]] = {
        "position_executor": (PositionExecutor, PositionExecutorConfig),
        "grid_executor": (GridExecutor, GridExecutorConfig),
        "dca_executor": (DCAExecutor, DCAExecutorConfig),
        "arbitrage_executor": (ArbitrageExecutor, ArbitrageExecutorConfig),
        "twap_executor": (TWAPExecutor, TWAPExecutorConfig),
        "xemm_executor": (XEMMExecutor, XEMMExecutorConfig),
        "order_executor": (OrderExecutor, OrderExecutorConfig),
        "lp_executor": (LPExecutor, LPExecutorConfig),
        "range_iceberg_executor": (RangeIcebergExecutor, RangeIcebergExecutorConfig),
    }

    def __init__(
        self,
        trading_service: TradingService,
        db_manager: AsyncDatabaseManager,
        default_account: str = "master_account",
        update_interval: float = 1.0,
        max_retries: int = 10,
        performance_snapshot_interval: float = 60.0,
        performance_retention_days: int = 0
    ):
        """
        Initialize ExecutorService.

        Args:
            trading_service: TradingService for trading operations and interfaces
            db_manager: AsyncDatabaseManager for persistence
            default_account: Default account to use
            update_interval: Executor update interval in seconds
            max_retries: Maximum retries for executor operations
            performance_snapshot_interval: How often a live executor's performance is
                written to executor_performance_snapshots, in seconds
            performance_retention_days: Delete performance snapshots older than this many
                days; 0 (the default) keeps everything
        """
        self._trading_service = trading_service
        self.db_manager = db_manager
        self.default_account = default_account
        self.update_interval = update_interval
        self.max_retries = max_retries
        self.performance_snapshot_interval = performance_snapshot_interval
        self.performance_retention_days = performance_retention_days

        # Trading interfaces per account (lazy initialized via TradingService)
        self._trading_interfaces: Dict[str, AccountTradingInterface] = {}

        # Active executors: executor_id -> executor instance
        self._active_executors: Dict[str, ExecutorBase] = {}

        # Executor metadata: executor_id -> metadata dict
        self._executor_metadata: Dict[str, Dict[str, Any]] = {}

        # Position holds: key = "account_name|connector_name|trading_pair"
        # Tracks aggregated positions from executors stopped with keep_position=True
        self._positions_held: Dict[str, PositionHold] = {}

        # Executor log capture
        self._log_capture = ExecutorLogCapture()
        self._log_capture.install()

        # An LP executor's position address, learned while the executor is live. A
        # successful close clears it from custom_info before the executor terminates, so
        # by completion — which is when the rent refund is finally known — there is
        # nothing left to file it under. See _record_lp_position_rent.
        self._lp_position_addresses: Dict[str, str] = {}
        # Executors whose locked rent is already stored, so the control loop stops
        # re-reading and re-writing a figure that cannot change.
        self._lp_rent_recorded: set = set()
        # Earliest monotonic time to retry an executor whose position row did not exist
        # yet. Discovery runs on its own schedule, so retrying at the control loop's 1 Hz
        # would be a query a second against a row that appears about once a minute.
        self._lp_rent_retry_after: Dict[str, float] = {}

        # Last child-order id set written for each live executor. The control loop
        # compares against this and skips the database when nothing changed.
        self._persisted_child_order_ids: Dict[str, tuple] = {}

        # Order ids startup must not cancel: live ids of a resumed range iceberg,
        # and the exchange stop of one whose resume failed.
        self._resume_protected_order_ids: set = set()

        # Performance snapshot cadence, on the control loop's own clock (monotonic, so a
        # system clock step cannot stall the series or flood it).
        self._last_snapshot_at: float = 0.0
        self._last_prune_at: float = 0.0

        # Control loop task
        self._control_loop_task: Optional[asyncio.Task] = None
        self._is_running = False

        # One lock per (account, connector, pair, side) so two creates of a
        # range iceberg on that slot cannot both pass the open check.
        # The lock is per process. A single uvicorn worker is assumed.
        # Another worker has its own dict and would not share this lock.
        self._range_iceberg_slot_locks: Dict[tuple, asyncio.Lock] = {}

    def start(self):
        """Start the executor service control loop."""
        if not self._is_running:
            self._is_running = True
            self._control_loop_task = asyncio.create_task(self._control_loop())
            logger.info("ExecutorService started")

    async def recover_positions_from_db(self):
        """
        Recover position holds from the dedicated position_holds table on startup.
        """
        if not self.db_manager:
            return

        try:
            async with self.db_manager.get_session_context() as session:
                repo = ExecutorRepository(session)

                records = await repo.get_active_position_holds()

                for record in records:
                    controller_id = record.controller_id or "main"
                    position_key = self._get_position_key(
                        record.account_name,
                        record.connector_name,
                        record.trading_pair,
                        controller_id
                    )

                    executor_ids = []
                    if record.executor_ids:
                        try:
                            executor_ids = json.loads(record.executor_ids)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    position = PositionHold(
                        trading_pair=record.trading_pair,
                        connector_name=record.connector_name,
                        account_name=record.account_name,
                        controller_id=controller_id,
                        buy_amount_base=Decimal(str(record.buy_amount_base or 0)),
                        buy_amount_quote=Decimal(str(record.buy_amount_quote or 0)),
                        sell_amount_base=Decimal(str(record.sell_amount_base or 0)),
                        sell_amount_quote=Decimal(str(record.sell_amount_quote or 0)),
                        realized_pnl_quote=Decimal(str(record.realized_pnl_quote or 0)),
                        cum_fees_quote=Decimal(str(record.cum_fees_quote or 0)),
                        executor_ids=executor_ids,
                        last_updated=record.last_updated,
                    )
                    # Settle any matched volume from legacy unsettled data
                    position._calculate_realized_pnl()
                    self._positions_held[position_key] = position

                if self._positions_held:
                    logger.info(f"Recovered {len(self._positions_held)} position holds from database")

        except Exception as e:
            logger.error(f"Error recovering positions from database: {e}", exc_info=True)

    async def cleanup_orphaned_executors(self):
        """
        Clean up orphaned executors from database on startup.

        Identifies executors marked as RUNNING in the database but not present
        in memory (i.e., from previous API sessions that were terminated).

        Range iceberg executors that graceful shutdown persisted for resume
        (exit_state ARMED, RESTING, or CLOSING) are started from that saved state
        before anything else. A resumed executor does not continue an unfinished
        entry. It resumes exits only, sized to the exchange position. Every
        exchange stop id stays up. Entry and exit client order ids are cancelled
        for every executor, resumed ones included.

        Exchange stops on an executor that is not RUNNING and was not parked
        with resume_on_startup are cancelled here as well, except a POSITION_HOLD
        row, which kept the position on purpose. A failed resume leaves its
        exchange stops in place and records the failure on the row. A flat
        exchange position is not resumed: the row is marked CLOSED_WHILE_DOWN
        and its exchange stops are cancelled.
        """
        if not self.db_manager:
            logger.debug("No database manager available, skipping orphaned executor cleanup")
            return

        try:
            try:
                await self._resume_range_iceberg_executors()
            except Exception as e:
                logger.error(f"Error resuming range iceberg executors: {e}", exc_info=True)

            # Resumed executors are in memory. Their software order ids are
            # still cancelled. Only the exchange stop ids in `protected` stay.
            active_executor_ids = list(self._active_executors.keys())
            protected = getattr(self, "_resume_protected_order_ids", None) or set()

            await self._cancel_orphaned_child_orders(
                active_executor_ids,
                protected_order_ids=protected,
            )
            await self._cancel_finished_exchange_stops(protected)

            async with self.db_manager.get_session_context() as session:
                repo = ExecutorRepository(session)

                # Clean up orphaned executors
                cleaned_count = await repo.cleanup_orphaned_executors(
                    active_executor_ids=active_executor_ids,
                    close_type="SYSTEM_CLEANUP"
                )

                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} orphaned executors from database")
                else:
                    logger.debug("No orphaned executors found in database")

            for item in list(getattr(self, "_flat_while_down", None) or []):
                await self._mark_closed_while_down(item)

        except Exception as e:
            logger.error(f"Error cleaning up orphaned executors: {e}", exc_info=True)

    async def stop(self):
        """Stop the executor service and all active executors.

        Each executor still running is asked to early_stop with keep_position, which
        cancels software entry and exit orders and does not close a filled position.
        A range iceberg whose exit_state is ARMED, RESTING, or CLOSING is persisted
        after those cancels are confirmed. Every exchange stop id stays on the
        exchange. Shutdown waits, up to SHUTDOWN_CANCEL_TIMEOUT_SECONDS, for the
        software cancels. Connector teardown happens only after this method returns.
        """
        self._is_running = False

        if self._control_loop_task:
            self._control_loop_task.cancel()
            try:
                await self._control_loop_task
            except asyncio.CancelledError:
                pass
            self._control_loop_task = None

        # The API control loop is down, so it cannot persist a completion while a
        # cancel is still unconfirmed. Each executor's own loop keeps running and
        # is what observes the exchange ack. Resumable range icebergs take the
        # same early_stop, so entry and exit orders are cancelled, and are persisted
        # afterwards with the exchange stop still in place.
        resume_ids: List[str] = []
        for executor_id in list(self._active_executors.keys()):
            executor = self._active_executors.get(executor_id)
            if executor is None or self._executor_is_closed(executor):
                continue
            if (
                self._should_persist_for_resume(executor_id, executor)
                and not self._shutdown_already_started(executor)
            ):
                resume_ids.append(executor_id)
                info = self._custom_info_for_snapshot(executor)
                if info is not None:
                    snapshots = getattr(self, "_shutdown_resume_snapshots", None)
                    if snapshots is None:
                        snapshots = {}
                        self._shutdown_resume_snapshots = snapshots
                    snapshots[executor_id] = info
            await self._persist_running_child_orders(executor_id, executor)
            if self._shutdown_already_started(executor):
                # close_type was already chosen. Calling early_stop again would
                # replace it and could skip a close that was in progress.
                continue
            try:
                self._request_early_stop(executor)
            except Exception as e:
                logger.error(f"Error stopping executor {executor_id}: {e}")

        unconfirmed = await self._wait_for_cancel_confirmations(
            list(self._active_executors.keys())
        )

        for executor_id in list(self._active_executors.keys()):
            executor = self._active_executors.get(executor_id)
            if executor is None:
                continue
            if executor_id in resume_ids:
                if executor_id in unconfirmed:
                    logger.error(
                        "Software-order cancel was not confirmed for range iceberg %s before "
                        "shutdown. Startup will cancel leftover entry and exit orders. "
                        "Left exchange stop %s on the exchange.",
                        executor_id,
                        _exchange_stop_text(self._safe_custom_info(executor)),
                    )
                await self._persist_range_iceberg_for_resume(executor_id, executor)
                continue
            if executor_id in unconfirmed:
                pending = [oid for oid in unconfirmed[executor_id] if oid]
                if pending:
                    logger.error(
                        "Cancel was not confirmed for executor %s orders %s before connector "
                        "shutdown. Leaving the executor RUNNING so startup can cancel them.",
                        executor_id,
                        ", ".join(pending),
                    )
                else:
                    logger.error(
                        "Executor %s did not finish cancelling working orders before connector "
                        "shutdown. Leaving the executor RUNNING so startup can cancel them.",
                        executor_id,
                    )
                self._stop_executor_task(executor_id, executor)
                self._drop_executor_memory(executor_id)
                continue
            if not self._executor_is_closed(executor):
                self._stop_executor_task(executor_id, executor)
            await self._handle_executor_completion(executor_id)

        # Clear anything completion did not claim.
        self._active_executors.clear()
        self._executor_metadata.clear()

        # Cleanup trading interfaces only after cancel confirmations (or the timeout).
        for trading_interface in self._trading_interfaces.values():
            await trading_interface.cleanup()
        self._trading_interfaces.clear()

        logger.info("ExecutorService stopped")

    @staticmethod
    def _executor_is_closed(executor: Any) -> bool:
        if getattr(executor, "is_closed", False):
            return True
        return _status_name(executor) == "TERMINATED"

    @staticmethod
    def _shutdown_already_started(executor: Any) -> bool:
        if getattr(executor, "_shutdown_pending", False):
            return True
        return _status_name(executor) == "SHUTTING_DOWN"

    @staticmethod
    def _request_early_stop(executor: Any) -> None:
        """Cancel working orders and keep any fill. Do not unwind the position."""
        try:
            executor.early_stop(keep_position=True)
        except TypeError as exc:
            if "keep_position" not in str(exc):
                raise
            executor.early_stop()

    def _should_persist_for_resume(self, executor_id: str, executor: Any) -> bool:
        meta = self._executor_metadata.get(executor_id, {})
        if meta.get("executor_type") != "range_iceberg_executor":
            return False
        return _exit_state_name(self._safe_custom_info(executor)) in _RESUMABLE_EXIT_STATES

    def _custom_info_for_snapshot(self, executor: Any) -> Optional[Dict[str, Any]]:
        getter = getattr(executor, "get_custom_info", None)
        if not callable(getter):
            return None
        try:
            info = getter()
        except Exception:
            logger.error("Could not read custom_info for a resume snapshot", exc_info=True)
            return None
        if not isinstance(info, dict):
            return None
        return info

    async def _write_final_state(self, executor_id: str, payload: Dict[str, Any]) -> None:
        if not self.db_manager:
            return
        try:
            encoded = json.dumps(payload, default=_json_default)
        except Exception as exc:
            logger.error("Could not serialize resume state for executor %s: %s", executor_id, exc)
            return
        try:
            async with self.db_manager.get_session_context() as session:
                updated = await ExecutorRepository(session).update_executor(
                    executor_id, final_state=encoded
                )
            if updated is None:
                logger.error(
                    "No executor row to store resume state for %s. Left exchange stop %s on the exchange.",
                    executor_id,
                    _exchange_stop_text(payload),
                )
        except Exception as exc:
            logger.error("Could not store resume state for executor %s: %s", executor_id, exc)

    async def _persist_range_iceberg_for_resume(self, executor_id: str, executor: Any) -> None:
        """Save the pre-cancel snapshot after early_stop, and drop the in-process task.

        early_stop has already been asked to cancel software entry and exit orders.
        The snapshot is the custom_info from before that call, so exit_state and the
        software order ids are still present for startup. The exchange stop is not
        part of that cancel. Completion is not run, so the row stays RUNNING.
        """
        snapshots = getattr(self, "_shutdown_resume_snapshots", None) or {}
        info = snapshots.get(executor_id)
        if info is None:
            info = self._custom_info_for_snapshot(executor)
        state = _exit_state_name(info) if info is not None else None
        stop_text = _exchange_stop_text(info)
        if info is None:
            logger.error(
                "Could not read custom_info while persisting range iceberg executor %s for resume. "
                "Left exchange stop %s on the exchange.",
                executor_id,
                stop_text,
            )
        else:
            payload = dict(info)
            payload[_RESUME_ON_STARTUP] = True
            cache = getattr(self, "_persisted_child_order_ids", None)
            if isinstance(cache, dict):
                cache.pop(executor_id, None)
            await self._persist_running_child_orders(executor_id, executor)
            await self._write_final_state(executor_id, payload)
        self._stop_executor_task(executor_id, executor)
        self._drop_executor_memory(executor_id)
        logger.info(
            "Persisted range iceberg executor %s for resume on startup (exit_state=%s). "
            "Software entry and exit orders were cancelled. Left exchange stop %s on the exchange.",
            executor_id,
            state or "unknown",
            stop_text,
        )

    async def _load_resume_candidates(self) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        async with self.db_manager.get_session_context() as session:
            repo = ExecutorRepository(session)
            running = await repo.get_active_executors()
            for record in running:
                if record.executor_id in self._active_executors:
                    continue
                if record.executor_type != "range_iceberg_executor":
                    continue
                custom = _parse_stored_custom_info(record.final_state)
                if custom.get(_RESUME_ON_STARTUP) is not True:
                    continue
                open_order_ids = []
                for row in await repo.get_executor_orders(record.executor_id):
                    oid = row.client_order_id
                    if not oid or oid in _SKIP_ORDER_IDS:
                        continue
                    if (row.status or "").upper() in _TERMINAL_ORDER_STATUSES:
                        continue
                    open_order_ids.append(oid)
                candidates.append({
                    "executor_id": record.executor_id,
                    "account_name": record.account_name,
                    "connector_name": record.connector_name,
                    "trading_pair": record.trading_pair,
                    "controller_id": record.controller_id or "main",
                    "config_json": record.config,
                    "created_at": record.created_at,
                    "custom": custom,
                    "open_order_ids": open_order_ids,
                })
        return candidates

    async def _mark_resume_failed(self, item: Dict[str, Any], message: str) -> None:
        if not self.db_manager:
            return
        custom = dict(item.get("custom") or {})
        stop_text = _exchange_stop_text(custom)
        custom[_RESUME_FAILED] = True
        custom[_RESUME_ERROR] = message
        custom[_RESUME_ON_STARTUP] = False
        text = (
            f"Resume failed for range iceberg executor {item['executor_id']}: {message}. "
            f"Left exchange stop {stop_text} on the exchange."
        )
        try:
            final_state = json.dumps(custom, default=_json_default)
            error_log = json.dumps([{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": text,
                "exc_info": None,
            }])
        except Exception:
            logger.error(
                "Could not serialize resume failure for %s", item["executor_id"], exc_info=True
            )
            return
        try:
            async with self.db_manager.get_session_context() as session:
                await ExecutorRepository(session).update_executor(
                    item["executor_id"],
                    final_state=final_state,
                    error_log=error_log,
                )
        except Exception:
            logger.error(
                "Could not mark resume failure for %s", item["executor_id"], exc_info=True
            )

    async def _resume_one_range_iceberg(self, item: Dict[str, Any]) -> List[str]:
        executor_id = item["executor_id"]
        custom = item["custom"]
        if not item.get("config_json"):
            raise ValueError("saved executor config is missing")
        try:
            config = json.loads(item["config_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("saved executor config is not valid JSON") from exc
        if not isinstance(config, dict):
            raise ValueError("saved executor config is not an object")
        config = dict(config)
        config["type"] = "range_iceberg_executor"
        config["id"] = executor_id
        # A resumed executor does not continue an unfinished entry. size_unit
        # is forced to BASE. A QUOTE total_amount equal to the open base would
        # be read as quote and place a much larger new entry. resume_entries
        # false is the same instruction on the config the executor reads.
        open_base = _reported_decimal(None, custom, "open_base")
        if open_base is not None:
            config["total_amount"] = format(open_base, "f")
            config["size_unit"] = "BASE"
            config[_RESUME_ENTRIES] = False
        config.setdefault("controller_id", item["controller_id"])
        if item.get("connector_name"):
            config.setdefault("connector_name", item["connector_name"])
        if item.get("trading_pair"):
            config.setdefault("trading_pair", item["trading_pair"])

        account = item.get("account_name") or self.default_account
        trading = self._get_trading_interface(account)
        _executor_class, _config_class, typed_config = self._validate_executor_config(
            config, default_timestamp=getattr(trading, "current_timestamp", None)
        )
        await self._prepare_market(account, config.get("connector_name"), config.get("trading_pair"))

        executor = _executor_class(
            strategy=trading,
            config=typed_config,
            update_interval=getattr(self, "update_interval", 1.0),
            max_retries=getattr(self, "max_retries", 10),
        )
        restore = getattr(executor, "restore_from_custom_info", None)
        if not callable(restore):
            raise RuntimeError("executor has no restore_from_custom_info")
        # resume_entries false tells the executor, once it honors the flag, not
        # to place another entry rung. The trimmed order lists keep restore from
        # adopting software orders that startup is about to cancel.
        saved = {key: value for key, value in custom.items() if key not in _API_RESUME_KEYS}
        saved[_RESUME_ENTRIES] = False
        result = restore(saved)
        if asyncio.iscoroutine(result):
            await result

        created_at = item.get("created_at") or datetime.now(timezone.utc)
        self._active_executors[executor_id] = executor
        self._executor_metadata[executor_id] = {
            "account_name": account,
            "connector_name": config.get("connector_name"),
            "trading_pair": config.get("trading_pair"),
            "executor_type": "range_iceberg_executor",
            "controller_id": item["controller_id"],
            "created_at": created_at,
            "config": config,
        }
        try:
            token = current_executor_id.set(executor_id)
            try:
                executor.start()
            finally:
                current_executor_id.reset(token)
        except Exception:
            self._drop_executor_memory(executor_id)
            raise

        # Every exchange stop is protected. Entry and exit client order ids
        # are leftover software orders and startup cancels them, including for
        # this resumed executor.
        return _exchange_stop_ids(custom)

    def _saved_side_name(self, item: Dict[str, Any]) -> str:
        custom = item.get("custom") or {}
        side = custom.get("side")
        if side in (None, ""):
            try:
                config = json.loads(item.get("config_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                config = {}
            if isinstance(config, dict):
                side = config.get("side")
        if side in (None, ""):
            return "BUY"
        name = getattr(side, "name", side)
        return str(name).strip().upper() or "BUY"

    async def _exchange_open_base(
        self,
        account_name: str,
        connector_name: str,
        trading_pair: str,
        side_name: str,
    ) -> Optional[tuple]:
        """Absolute base and entry price on this pair and side.

        None when the connector position book cannot be read. The base is zero
        when the book was read and that side is flat. The entry price is None
        when the book has no price. Hedge mode matches LONG to BUY and SHORT
        to SELL. A signed one-way amount uses its sign the same way.
        """
        connector = self._account_connector(account_name, connector_name)
        if connector is None or not hasattr(connector, "account_positions"):
            return None
        updater = getattr(connector, "_update_positions", None)
        if callable(updater):
            try:
                result = updater()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.error(
                    "Could not refresh positions for %s %s",
                    connector_name,
                    trading_pair,
                    exc_info=True,
                )
                return None
        book = getattr(connector, "account_positions", None)
        if not isinstance(book, dict):
            return None
        want_long = side_name in {"BUY", "LONG"}
        total = Decimal("0")
        entry_base = Decimal("0")
        entry_notional = Decimal("0")
        for position in book.values():
            pair = position.get("trading_pair") if isinstance(position, dict) else getattr(position, "trading_pair", None)
            if _normalize_trading_pair(pair) != _normalize_trading_pair(trading_pair):
                continue
            raw_side = position.get("position_side") if isinstance(position, dict) else getattr(position, "position_side", None)
            side_text = getattr(raw_side, "name", raw_side)
            side_text = side_text.strip().upper() if isinstance(side_text, str) else ""
            raw_amount = position.get("amount") if isinstance(position, dict) else getattr(position, "amount", 0)
            raw_entry = position.get("entry_price") if isinstance(position, dict) else getattr(position, "entry_price", None)
            try:
                amount = Decimal(str(raw_amount or 0))
            except Exception:
                continue
            if side_text in {"LONG", "SHORT"}:
                matches = (side_text == "LONG") == want_long
            elif amount == 0:
                matches = False
            else:
                matches = (amount > 0) == want_long
            if not matches:
                continue
            base = abs(amount)
            total += base
            try:
                entry_price = Decimal(str(raw_entry)) if raw_entry not in (None, "") else None
            except Exception:
                entry_price = None
            if entry_price is not None and entry_price > 0 and base > 0:
                entry_base += base
                entry_notional += entry_price * base
        average_entry = entry_notional / entry_base if entry_base > 0 else None
        return total, average_entry

    async def _mark_closed_while_down(self, item: Dict[str, Any]) -> None:
        """The exchange position is flat, so this parked executor is not resumed."""
        if not self.db_manager:
            return
        custom = dict(item.get("custom") or {})
        stop_text = _exchange_stop_text(custom)
        custom[_RESUME_ON_STARTUP] = False
        custom["closed_while_down"] = True
        message = (
            f"Position for range iceberg executor {item['executor_id']} was flat on the exchange "
            f"while the API was down. Cancelled exchange stop {stop_text}."
        )
        custom[_RESUME_ERROR] = message
        try:
            final_state = json.dumps(custom, default=_json_default)
            error_log = json.dumps([{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": message,
                "exc_info": None,
            }])
        except Exception:
            logger.error(
                "Could not serialize flat-position close for %s", item["executor_id"], exc_info=True
            )
            return
        try:
            async with self.db_manager.get_session_context() as session:
                await ExecutorRepository(session).update_executor(
                    item["executor_id"],
                    status="TERMINATED",
                    close_type=_CLOSED_WHILE_DOWN,
                    final_state=final_state,
                    error_log=error_log,
                )
        except Exception:
            logger.error(
                "Could not mark %s closed while the API was down", item["executor_id"], exc_info=True
            )

    async def _cancel_exchange_stop_now(
        self,
        account_name: str,
        connector_name: str,
        trading_pair: str,
        executor_id: str,
        stop_id: Optional[str],
        reason: str,
    ) -> Optional[str]:
        """Cancel one exchange stop. None when it is confirmed, otherwise the error.

        ``connector.cancel`` does not cancel a MEXC stop id. The stop cancel
        API is ``cancel_stop_orders``. A response that does not accept the id,
        or an open-stop query that still lists it, is a failure.
        """
        if not stop_id:
            return None
        connector = self._account_connector(account_name, connector_name)
        try:
            error = await self._cancel_stop_on_connector(connector, trading_pair, stop_id)
        except Exception as exc:
            error = str(exc)
        if error:
            logger.error(
                "Failed to cancel exchange stop %s for executor %s on %s %s: %s",
                stop_id,
                executor_id,
                connector_name,
                trading_pair,
                error,
            )
            return error
        logger.info(
            "Cancelled exchange stop %s for executor %s on %s %s because %s",
            stop_id,
            executor_id,
            connector_name,
            trading_pair,
            reason,
        )
        return None

    async def _cancel_exchange_stops(
        self,
        account_name: str,
        connector_name: str,
        trading_pair: str,
        executor_id: str,
        stop_ids: List[str],
        reason: str,
    ) -> Optional[str]:
        """Cancel every exchange stop. None when each one is confirmed."""
        errors: List[str] = []
        for stop_id in stop_ids:
            error = await self._cancel_exchange_stop_now(
                account_name,
                connector_name,
                trading_pair,
                executor_id,
                stop_id,
                reason,
            )
            if error:
                errors.append(error)
        if errors:
            return "; ".join(errors)
        return None

    async def _park_flat(self, item: Dict[str, Any], reason: str) -> None:
        """Leave the row RUNNING so software orders are still cancelled, then close it."""
        flats = getattr(self, "_flat_while_down", None)
        if flats is None:
            flats = []
            self._flat_while_down = flats
        flats.append(item)
        stop_ids = _exchange_stop_ids(item.get("custom"))
        error = await self._cancel_exchange_stops(
            item.get("account_name") or self.default_account,
            item.get("connector_name") or "",
            item.get("trading_pair") or "",
            item["executor_id"],
            stop_ids,
            reason,
        )
        if error:
            custom = item.get("custom")
            if isinstance(custom, dict):
                custom["exchange_stop_error"] = error
        logger.info(
            "Range iceberg executor %s was not resumed on %s %s. %s Cancelled exchange stop %s.",
            item["executor_id"],
            item.get("connector_name"),
            item.get("trading_pair"),
            reason,
            ", ".join(stop_ids) if stop_ids else "none",
        )

    async def _resume_allocated_positions(
        self,
        ready: List[Dict[str, Any]],
        protected: set,
        resumed: List[str],
    ) -> None:
        """Give each parked executor a share of the one exchange position.

        Executors on the same account, connector, pair, and side share one
        position. Shares are handed out in created_at order. Each share is at
        most that executor's saved open base, and the shares sum to at most
        the exchange position. A later executor that is left with nothing is
        closed instead of resumed.
        """
        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        for item in ready:
            key = (
                item.get("account_name") or self.default_account,
                item.get("connector_name") or "",
                item.get("trading_pair") or "",
                item.get("_side_name") or "BUY",
            )
            groups.setdefault(key, []).append(item)
        for key, items in groups.items():
            items.sort(key=lambda row: (row.get("created_at") is None, row.get("created_at"), row["executor_id"]))
            actual = items[0].get("_actual") or Decimal("0")
            if actual == 0:
                for item in items:
                    await self._park_flat(item, "the position was flat while the API was down")
                continue
            remaining = actual
            shares: List[str] = []
            for item in items:
                saved_open = item.get("_saved_open")
                if saved_open is None:
                    share = remaining
                else:
                    share = min(saved_open, remaining)
                if share <= 0:
                    await self._park_flat(
                        item,
                        "no exchange position remained after earlier executors on this side were allocated",
                    )
                    shares.append(f"{item['executor_id']}=0")
                    continue
                remaining -= share
                if saved_open is not None and share < saved_open:
                    logger.info(
                        "Allocated %s of exchange position %s to range iceberg %s "
                        "(saved open_base %s) on %s %s. Later executors share what remains.",
                        share,
                        actual,
                        item["executor_id"],
                        saved_open,
                        key[2],
                        key[3],
                    )
                elif saved_open is not None and actual > saved_open and len(items) == 1:
                    logger.warning(
                        "Exchange position %s is larger than saved open_base %s for range iceberg %s "
                        "on %s. Another position may share this side. Resuming exits on %s.",
                        actual,
                        saved_open,
                        item["executor_id"],
                        key[2],
                        share,
                    )
                shares.append(f"{item['executor_id']}={format(share, 'f')}")
                custom = item["custom"]
                average = _resume_entry_average(custom, item.get("_entry_price"))
                attempt = dict(item)
                attempt["custom"] = _exit_only_custom_info(custom, share, average)
                state = item.get("_state")
                stop_ids = _exchange_stop_ids(custom)
                try:
                    live_ids = await self._resume_one_range_iceberg(attempt)
                except Exception as exc:
                    await self._mark_resume_failed(item, str(exc))
                    protected.update(stop_ids)
                    logger.error(
                        "Failed to resume range iceberg executor %s (exit_state=%s). "
                        "Left exchange stop %s on the exchange. %s",
                        item["executor_id"],
                        state,
                        ", ".join(stop_ids) if stop_ids else "none",
                        exc,
                    )
                    continue
                protected.update(live_ids)
                self._resumed_executor_ids.add(item["executor_id"])
                resumed.append(item["executor_id"])
                logger.info(
                    "Resumed range iceberg executor %s from saved state (exit_state=%s) "
                    "before orphan cleanup. Resuming exits only on open_base %s. "
                    "Protected exchange stop %s. Software order ids stay eligible for cleanup.",
                    item["executor_id"],
                    state,
                    format(share, "f"),
                    ", ".join(stop_ids) if stop_ids else "none",
                )
            if len(items) > 1:
                logger.info(
                    "Allocated exchange position %s on %s %s across parked executors: %s. "
                    "The shares sum to %s, which does not exceed the position.",
                    actual,
                    key[2],
                    key[3],
                    ", ".join(shares),
                    format(actual - remaining, "f"),
                )

    async def _resume_range_iceberg_executors(self) -> None:
        """Start range icebergs that graceful shutdown saved, before orphan cleanup.

        A resumed executor does not continue an unfinished entry. The restore
        payload sets resume_entries false and sizes the exit to the reconciled
        open base. Every exchange stop id is protected from the cleanup that
        follows. Entry and exit client order ids are cancelled with every
        other leftover software order.
        """
        protected: set = set()
        self._resume_protected_order_ids = protected
        self._resumed_executor_ids = set()
        self._flat_while_down = []
        if not self.db_manager:
            return
        candidates = await self._load_resume_candidates()
        resumed: List[str] = []
        ready: List[Dict[str, Any]] = []
        for item in candidates:
            executor_id = item["executor_id"]
            custom = item["custom"]
            state = _exit_state_name(custom)
            stop_ids = _exchange_stop_ids(custom)
            stop_text = ", ".join(stop_ids) if stop_ids else "none"
            if state not in _RESUMABLE_EXIT_STATES:
                await self._mark_resume_failed(
                    item,
                    f"saved exit_state {state or 'missing'} cannot be resumed",
                )
                protected.update(stop_ids)
                logger.error(
                    "Failed to resume range iceberg executor %s (exit_state=%s). "
                    "Left exchange stop %s on the exchange. saved exit_state cannot be resumed",
                    executor_id,
                    state or "missing",
                    stop_text,
                )
                continue
            side_name = self._saved_side_name(item)
            try:
                snapshot = await self._exchange_open_base(
                    item.get("account_name") or self.default_account,
                    item.get("connector_name") or "",
                    item.get("trading_pair") or "",
                    side_name,
                )
            except Exception as exc:
                snapshot = None
                logger.error(
                    "Could not read the exchange position for range iceberg %s: %s",
                    executor_id,
                    exc,
                    exc_info=True,
                )
            if snapshot is None:
                await self._mark_resume_failed(
                    item,
                    "could not read the exchange position before resume",
                )
                protected.update(stop_ids)
                logger.error(
                    "Failed to resume range iceberg executor %s (exit_state=%s). "
                    "Left exchange stop %s on the exchange. could not read the exchange position",
                    executor_id,
                    state,
                    stop_text,
                )
                continue
            actual, entry_price = snapshot
            item["_saved_open"] = range_iceberg_held_base(None, custom)
            item["_actual"] = actual
            item["_entry_price"] = entry_price
            item["_state"] = state
            item["_side_name"] = side_name
            ready.append(item)
        await self._resume_allocated_positions(ready, protected, resumed)
        if resumed:
            logger.info(
                "Resumed range iceberg executors before orphan cleanup: %s",
                ", ".join(resumed),
            )

    def _stop_executor_task(self, executor_id: str, executor: Any) -> None:
        try:
            executor.stop()
        except Exception as e:
            logger.error(f"Error stopping executor task {executor_id}: {e}")

    def _drop_executor_memory(self, executor_id: str) -> None:
        """Forget an executor locally without writing a terminal database row."""
        self._active_executors.pop(executor_id, None)
        self._executor_metadata.pop(executor_id, None)
        getattr(self, "_persisted_child_order_ids", {}).pop(executor_id, None)
        log_capture = getattr(self, "_log_capture", None)
        if log_capture is not None:
            try:
                log_capture.clear(executor_id)
            except Exception:
                logger.debug(f"Could not clear logs for executor {executor_id}", exc_info=True)

    def _executor_connector(self, executor_id: str, executor: Any) -> Any:
        connectors = getattr(executor, "connectors", None)
        if not isinstance(connectors, dict) or not connectors:
            return None
        meta = self._executor_metadata.get(executor_id, {})
        name = meta.get("connector_name") or getattr(
            getattr(executor, "config", None), "connector_name", None
        )
        if name and name in connectors:
            return connectors[name]
        if len(connectors) == 1:
            return next(iter(connectors.values()))
        return None

    def _cancel_checker(self, connector: Any):
        if connector is None:
            return None
        checker = getattr(connector, "is_cancel_confirmed", None)
        if checker is None or not callable(checker):
            return None
        return checker

    def _orders_awaiting_cancel_confirmation(self, executor_id: str, executor: Any) -> List[str]:
        """Child orders whose cancel the connector has not confirmed.

        Connectors without is_cancel_confirmed cannot be waited on. Range iceberg
        treats that as confirmed, and so does this wait.
        """
        connector = self._executor_connector(executor_id, executor)
        checker = self._cancel_checker(connector)
        if checker is None:
            return []

        explicit = getattr(executor, "_unconfirmed_cancels", None)
        if isinstance(explicit, (set, list, tuple)):
            candidates = list(explicit)
        else:
            info = self._safe_custom_info(executor)
            working, _filled = child_order_ids_from_custom_info(info)
            book = getattr(connector, "in_flight_orders", None)
            if isinstance(book, dict):
                candidates = [
                    oid for oid in working
                    if oid in book and _inflight_is_open(book[oid])
                ]
            else:
                candidates = list(working)

        pending: List[str] = []
        seen: set = set()
        for oid in candidates:
            if not isinstance(oid, str) or oid in _SKIP_ORDER_IDS or oid in seen:
                continue
            seen.add(oid)
            try:
                confirmed = bool(checker(oid))
            except Exception:
                logger.debug(
                    f"Cancel confirmation check failed for {oid} on executor {executor_id}",
                    exc_info=True,
                )
                confirmed = False
            if not confirmed:
                pending.append(oid)
        return pending

    async def _wait_for_cancel_confirmations(self, executor_ids: List[str]) -> Dict[str, List[str]]:
        """Poll until every shutdown cancel is confirmed, or the timeout elapses."""
        deadline = time.monotonic() + self.SHUTDOWN_CANCEL_TIMEOUT_SECONDS
        still: Dict[str, List[str]] = {}
        while True:
            still = {}
            for executor_id in executor_ids:
                executor = self._active_executors.get(executor_id)
                if executor is None:
                    continue
                pending = self._orders_awaiting_cancel_confirmation(executor_id, executor)
                if pending:
                    still[executor_id] = pending
                elif (
                    self._cancel_checker(self._executor_connector(executor_id, executor)) is None
                    and not self._executor_is_closed(executor)
                ):
                    # No confirmation hook. Keep waiting so the executor's own loop can
                    # send the cancels. An empty list means "not finished", not "no orders".
                    still[executor_id] = []
            if not still or time.monotonic() >= deadline:
                return still
            await asyncio.sleep(self.SHUTDOWN_CANCEL_POLL_SECONDS)

    def _account_connector(self, account_name: str, connector_name: str) -> Any:
        trading = self._trading_service
        service = getattr(trading, "connector_service", None) if trading is not None else None
        if service is None:
            return None
        getter = getattr(service, "get_account_connectors", None)
        if getter is None:
            return None
        try:
            connectors = getter(account_name) or {}
        except Exception:
            logger.error(
                f"Could not read connectors for account {account_name}",
                exc_info=True,
            )
            return None
        if not isinstance(connectors, dict):
            return None
        return connectors.get(connector_name)

    @staticmethod
    def _safe_custom_info(executor: Any) -> Dict[str, Any]:
        getter = getattr(executor, "get_custom_info", None)
        if not callable(getter):
            return {}
        try:
            info = getter()
        except Exception:
            logger.debug("Could not read executor custom_info", exc_info=True)
            return {}
        return info if isinstance(info, dict) else {}

    async def _persist_running_child_orders(self, executor_id: str, executor: Any) -> None:
        """Store the working child order ids and the custom_info that lists them.

        An unclean shutdown reads this back. The snapshot is skipped when the id
        set has not changed since the last successful write.
        """
        if not self.db_manager:
            return
        cache = getattr(self, "_persisted_child_order_ids", None)
        if cache is None:
            cache = {}
            self._persisted_child_order_ids = cache

        # A failed read is not an empty book. Treating it as one would mark every
        # stored child DONE, and the next startup would leave those orders open.
        getter = getattr(executor, "get_custom_info", None)
        if not callable(getter):
            return
        try:
            info = getter()
        except Exception:
            logger.error(
                f"Could not read custom_info for executor {executor_id}; keeping the last child-order snapshot",
                exc_info=True,
            )
            return
        if not isinstance(info, dict):
            return
        working, _filled = child_order_ids_from_custom_info(info)
        signature = tuple(working)
        if cache.get(executor_id) == signature:
            return

        side = info.get("side") or "BUY"
        trade_type = side.name if hasattr(side, "name") else str(side)
        try:
            final_state_json = json.dumps(info, default=_json_default)
        except Exception as e:
            logger.warning(f"Could not serialize custom_info for {executor_id}: {e}")
            final_state_json = None

        try:
            async with self.db_manager.get_session_context() as session:
                repo = ExecutorRepository(session)
                existing = {
                    row.client_order_id: row
                    for row in await repo.get_executor_orders(executor_id)
                }
                for oid in working:
                    row = existing.get(oid)
                    if row is None:
                        await repo.create_executor_order(
                            executor_id=executor_id,
                            client_order_id=oid,
                            order_type="child",
                            trade_type=trade_type,
                            amount=Decimal("0"),
                            status="OPEN",
                        )
                    elif (row.status or "").upper() in _TERMINAL_ORDER_STATUSES:
                        await repo.update_executor_order(oid, status="OPEN")
                for oid, row in existing.items():
                    if oid not in signature and (row.status or "").upper() not in _TERMINAL_ORDER_STATUSES:
                        await repo.update_executor_order(oid, status="DONE")
                if final_state_json is not None:
                    await repo.update_executor(executor_id, final_state=final_state_json)
            cache[executor_id] = signature
        except Exception as e:
            logger.error(f"Error persisting child orders for executor {executor_id}: {e}")

    async def _cancel_orphaned_child_orders(
        self,
        active_executor_ids: List[str],
        protected_order_ids: Optional[set] = None,
    ) -> None:
        """Cancel leftover software orders, including those of a resumed executor.

        Order ids come from executor_orders and from the custom_info snapshot stored
        on the executor row, including exit_order_ids. Filled orders are left alone.
        A resumed executor is in active_executor_ids. Its entry and exit client
        order ids are still cancelled. Every exchange stop id of a resumed
        executor is in protected_order_ids. A non-resumed exchange stop is
        cancelled when open_base is 0. A failed resume puts its stops in
        protected_order_ids so they stay up.
        """
        if not self.db_manager:
            return

        active = set(active_executor_ids)
        resumed_ids = set(getattr(self, "_resumed_executor_ids", None) or ())
        protected = set(protected_order_ids or ())
        pending: List[Dict[str, Any]] = []
        async with self.db_manager.get_session_context() as session:
            repo = ExecutorRepository(session)
            order_repo = OrderRepository(session)
            running = await repo.get_active_executors()
            for record in running:
                # An executor already running in this process keeps its orders.
                # A resumed range iceberg is in memory too, and its software
                # orders are still cancelled.
                if record.executor_id in active and record.executor_id not in resumed_ids:
                    continue
                custom = _parse_stored_custom_info(record.final_state)
                working, filled = child_order_ids_from_custom_info(custom)
                order_ids = list(working)
                seen = set(order_ids)
                for row in await repo.get_executor_orders(record.executor_id):
                    oid = row.client_order_id
                    if not oid or oid in seen or oid in _SKIP_ORDER_IDS:
                        continue
                    if (row.status or "").upper() in _TERMINAL_ORDER_STATUSES:
                        continue
                    seen.add(oid)
                    order_ids.append(oid)
                stop_ids = _exchange_stop_ids(custom)
                resumed = record.executor_id in resumed_ids
                cancel_stop = bool(stop_ids) and not resumed and _open_base_is_zero(custom)
                if stop_ids:
                    stop_set = set(stop_ids)
                    order_ids = [oid for oid in order_ids if oid not in stop_set]
                if protected:
                    order_ids = [oid for oid in order_ids if oid not in protected]
                stops_to_cancel = [
                    stop_id for stop_id in stop_ids
                    if cancel_stop and stop_id not in protected
                ]
                if not order_ids and not stops_to_cancel:
                    continue
                db_orders = await order_repo.get_orders_by_client_ids(order_ids)
                db_status = {row.client_order_id: row.status for row in db_orders}
                exchange_ids = {
                    row.client_order_id: row.exchange_order_id
                    for row in db_orders
                    if row.exchange_order_id
                }
                pending.append({
                    "executor_id": record.executor_id,
                    "account_name": record.account_name,
                    "connector_name": record.connector_name,
                    "trading_pair": record.trading_pair,
                    "order_ids": order_ids,
                    "filled_ids": set(filled),
                    "db_status": db_status,
                    "exchange_ids": exchange_ids,
                    "stop_ids": stops_to_cancel,
                })

        for item in pending:
            connector = self._account_connector(item["account_name"], item["connector_name"])
            book = getattr(connector, "in_flight_orders", None) if connector is not None else None
            if not isinstance(book, dict):
                book = {}
            if item.get("stop_ids"):
                await self._cancel_exchange_stops(
                    item["account_name"],
                    item["connector_name"],
                    item["trading_pair"],
                    item["executor_id"],
                    item["stop_ids"],
                    "open_base is 0",
                )
            for order_id in item["order_ids"]:
                if order_id in item["filled_ids"]:
                    continue
                if not self._child_order_still_open(order_id, book, item["db_status"].get(order_id)):
                    continue
                try:
                    outcome = await self._cancel_software_order(
                        connector,
                        item["trading_pair"],
                        order_id,
                        (item.get("exchange_ids") or {}).get(order_id),
                    )
                except Exception as e:
                    logger.error(
                        "Failed to cancel open child order %s for executor %s on %s %s: %s",
                        order_id,
                        item["executor_id"],
                        item["connector_name"],
                        item["trading_pair"],
                        e,
                    )
                    continue
                if outcome == "already_gone":
                    logger.info(
                        "Software order %s for executor %s on %s %s was already gone",
                        order_id,
                        item["executor_id"],
                        item["connector_name"],
                        item["trading_pair"],
                    )
                    continue
                logger.info(
                    "Cancelled open child order %s for executor %s on %s %s before SYSTEM_CLEANUP",
                    order_id,
                    item["executor_id"],
                    item["connector_name"],
                    item["trading_pair"],
                )

    async def _cancel_finished_exchange_stops(self, protected_order_ids: Optional[set] = None) -> None:
        """Cancel exchange stops left by executors that already ended.

        A row that is not RUNNING and is not marked resume_on_startup no longer
        owns a protective stop, unless it closed POSITION_HOLD and kept the
        position on purpose. The hummingbot executor also cancels the stop when
        the position is flat or the take profit finishes. This pass is the
        safety net for a user stop or a FAILED close.
        """
        if not self.db_manager:
            return
        protected = set(protected_order_ids or ())
        async with self.db_manager.get_session_context() as session:
            records = await ExecutorRepository(session).get_executors(limit=None)
            pending = []
            for record in records:
                if (record.status or "") == "RUNNING":
                    continue
                if (record.close_type or "") == "POSITION_HOLD":
                    continue
                custom = _parse_stored_custom_info(record.final_state)
                if custom.get(_RESUME_ON_STARTUP) is True:
                    continue
                if custom.get(_EXCHANGE_STOP_CANCELLED) is True:
                    continue
                stop_ids = [
                    stop_id for stop_id in _exchange_stop_ids(custom)
                    if stop_id not in protected
                ]
                if not stop_ids:
                    continue
                pending.append({
                    "executor_id": record.executor_id,
                    "account_name": record.account_name,
                    "connector_name": record.connector_name,
                    "trading_pair": record.trading_pair,
                    "stop_ids": stop_ids,
                    "custom": custom,
                })
        for item in pending:
            error = await self._cancel_exchange_stops(
                item["account_name"],
                item["connector_name"],
                item["trading_pair"],
                item["executor_id"],
                item["stop_ids"],
                "the executor is not RUNNING and was not parked for resume",
            )
            custom = dict(item["custom"])
            custom[_RESUME_ON_STARTUP] = False
            if error:
                custom["exchange_stop_error"] = error
            else:
                custom[_EXCHANGE_STOP_CANCELLED] = True
                custom.pop("exchange_stop_error", None)
            try:
                encoded = json.dumps(custom, default=_json_default)
            except Exception:
                logger.error(
                    "Could not record exchange stop cancel for %s", item["executor_id"], exc_info=True
                )
                continue
            try:
                async with self.db_manager.get_session_context() as session:
                    await ExecutorRepository(session).update_executor(
                        item["executor_id"], final_state=encoded
                    )
            except Exception:
                logger.error(
                    "Could not record exchange stop cancel for %s", item["executor_id"], exc_info=True
                )

    @staticmethod
    def _child_order_still_open(order_id: str, book: Dict[str, Any], db_status: Optional[str]) -> bool:
        inflight = book.get(order_id)
        if inflight is not None:
            return _inflight_is_open(inflight)
        return _order_status_is_open(db_status)

    @staticmethod
    def _software_order_is_tracked(connector: Any, order_id: str) -> bool:
        book = getattr(connector, "in_flight_orders", None)
        if isinstance(book, dict) and order_id in book:
            return True
        finder = getattr(connector, "_find_order", None)
        if callable(finder):
            try:
                return finder(order_id) is not None
            except Exception:
                return False
        return False

    @staticmethod
    def _open_row_matches(row: Any, client_order_id: str, exchange_order_id: Optional[str]) -> bool:
        if isinstance(row, str):
            return row == client_order_id or (exchange_order_id is not None and row == exchange_order_id)
        if isinstance(row, dict):
            client = row.get("externalOid") or row.get("client_order_id") or row.get("clientOrderId")
            exchange = row.get("orderId") or row.get("exchange_order_id") or row.get("id")
        else:
            client = getattr(row, "client_order_id", None) or getattr(row, "externalOid", None)
            exchange = getattr(row, "exchange_order_id", None) or getattr(row, "orderId", None)
        if client is not None and str(client) == client_order_id:
            return True
        return exchange_order_id is not None and exchange is not None and str(exchange) == str(exchange_order_id)

    async def _open_orders(self, connector: Any, trading_pair: str) -> Optional[list]:
        """Open orders from the connector, or None when it has no query."""
        lister = getattr(connector, "list_open_orders", None)
        if callable(lister):
            try:
                result = lister(trading_pair)
            except TypeError:
                result = lister()
        else:
            lister = getattr(connector, "_open_order_rows", None)
            if not callable(lister):
                return None
            result = lister()
        if asyncio.iscoroutine(result):
            result = await result
        if result is None:
            return []
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            data = result.get("data", result)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for key in ("resultList", "list", "orders"):
                    rows = data.get(key)
                    if isinstance(rows, list):
                        return rows
        return []

    async def _cancel_untracked_software_order(
        self,
        connector: Any,
        trading_pair: str,
        client_order_id: str,
        exchange_order_id: Optional[str],
    ) -> bool:
        """Cancel an order the fresh tracker does not know.

        ``connector.cancel`` returns without calling the exchange when the
        order was never tracked by this process. Cancel by exchange order id
        through ``_place_cancel`` (or ``cancel_order_by_exchange_id``).

        False means the direct call reported that it did not cancel. The
        caller checks the open-order list before treating that as a rejection:
        an unknown id that is already gone is not a failed cancel.
        """
        placer = getattr(connector, "_place_cancel", None)
        if callable(placer):
            target = SimpleNamespace(
                client_order_id=client_order_id,
                exchange_order_id=exchange_order_id,
                trading_pair=trading_pair,
            )
            result = placer(client_order_id, target)
            if asyncio.iscoroutine(result):
                result = await result
            return result is not False
        named = getattr(connector, "cancel_order_by_exchange_id", None)
        if callable(named) and exchange_order_id:
            result = named(exchange_order_id)
            if asyncio.iscoroutine(result):
                result = await result
            return result is not False
        raise RuntimeError(f"no direct cancel for untracked order {client_order_id}")

    async def _cancel_software_order(
        self,
        connector: Any,
        trading_pair: str,
        client_order_id: str,
        exchange_order_id: Optional[str],
    ) -> Optional[str]:
        """Cancel one software order and confirm it is no longer open.

        A tracked order uses ``connector.cancel``. An untracked order uses the
        direct cancel path. When the connector can list open orders, success
        requires that the id is gone from that list.

        Returns ``already_gone`` when the direct cancel reports that it did
        not cancel, and the open-order list shows the id is not working.
        """
        direct_declined = False
        tracked = connector is not None and self._software_order_is_tracked(connector, client_order_id)
        if tracked:
            await self._cancel_on_connector(connector, trading_pair, client_order_id)
        else:
            try:
                accepted = await self._cancel_untracked_software_order(
                    connector, trading_pair, client_order_id, exchange_order_id
                )
            except RuntimeError as exc:
                if not str(exc).startswith("no direct cancel"):
                    raise
                # No direct path on this connector. The plain cancel is the
                # only call it exposes. An open-order query below rejects it
                # when that call left the order working.
                await self._cancel_on_connector(connector, trading_pair, client_order_id)
            else:
                direct_declined = not accepted
        open_rows = await self._open_orders(connector, trading_pair)
        if open_rows is None:
            if direct_declined:
                raise RuntimeError(f"direct cancel of {client_order_id} was rejected")
            return None
        still_open = any(
            self._open_row_matches(row, client_order_id, exchange_order_id) for row in open_rows
        )
        if still_open:
            if direct_declined:
                raise RuntimeError(f"direct cancel of {client_order_id} was rejected")
            raise RuntimeError(f"order {client_order_id} is still open after cancel")
        if direct_declined:
            return "already_gone"
        return None

    @staticmethod
    def _stop_cancel_error(result: Any, stop_id: str) -> Optional[str]:
        """None when the stop-cancel response accepted the id."""
        if result is None:
            return f"cancel_stop_orders returned nothing for {stop_id}"
        data: Any = result
        if isinstance(result, dict):
            if result.get("success") is False:
                detail = result.get("message") or result.get("msg") or result
                return f"cancel_stop_orders rejected {stop_id}: {detail}"
            code = result.get("code")
            if code not in (None, 0, "0"):
                return f"cancel_stop_orders code {code} for {stop_id}"
            data = result.get("data", [])
        rows = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw = row.get("errorCode", row.get("error_code", 0))
            try:
                code = int(raw or 0)
            except (TypeError, ValueError):
                code = 1
            if code != 0:
                detail = row.get("errorMsg") or row
                return f"cancel_stop_orders error {code} for {stop_id}: {detail}"
        return None

    async def _cancel_stop_on_connector(
        self,
        connector: Any,
        trading_pair: str,
        stop_id: str,
    ) -> Optional[str]:
        """None when the exchange accepted the stop cancel and the id is gone."""
        if connector is None:
            return f"connector is not loaded; cannot cancel stop {stop_id}"
        canceller = getattr(connector, "cancel_stop_orders", None)
        if not callable(canceller):
            return f"connector has no cancel_stop_orders; cannot cancel stop {stop_id}"
        try:
            result = canceller([stop_id])
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as exc:
            return str(exc)
        error = self._stop_cancel_error(result, stop_id)
        if error:
            return error
        lister = getattr(connector, "list_open_stop_orders", None)
        if not callable(lister):
            return None
        try:
            listed = lister(trading_pair)
            if asyncio.iscoroutine(listed):
                listed = await listed
        except Exception as exc:
            return f"could not confirm stop {stop_id} was cancelled: {exc}"
        rows = listed if isinstance(listed, list) else []
        if any(self._open_row_matches(row, stop_id, stop_id) for row in rows):
            return f"stop {stop_id} is still open after cancel_stop_orders"
        return None

    @staticmethod
    async def _cancel_on_connector(connector: Any, trading_pair: str, order_id: str) -> None:
        if connector is None:
            raise RuntimeError(f"connector is not loaded; cannot cancel {order_id}")
        cancel = getattr(connector, "cancel", None)
        if cancel is None or not callable(cancel):
            raise RuntimeError(f"connector cannot cancel {order_id}")
        try:
            result = cancel(trading_pair=trading_pair, client_order_id=order_id)
        except TypeError:
            result = cancel(trading_pair, order_id)
        if asyncio.iscoroutine(result):
            await result

    async def _control_loop(self):
        """Main control loop that updates all active executors."""
        while self._is_running:
            try:
                # Update timestamps for all trading interfaces via TradingService
                self._trading_service.update_all_timestamps()

                # Check for completed executors. Iterate a snapshot: the rent recording
                # below awaits, and create_executor runs in a request task that can add to
                # or remove from _active_executors while this loop is suspended.
                completed_ids = []
                for executor_id, executor in list(self._active_executors.items()):
                    if executor.is_closed:
                        completed_ids.append(executor_id)
                        continue
                    # Child order ids have to be on disk before a crash, or the next
                    # startup cannot tell which open orders belonged to this executor.
                    await self._persist_running_child_orders(executor_id, executor)
                    if (
                        self._executor_metadata.get(executor_id, {}).get("executor_type") == "lp_executor"
                        and executor_id not in self._lp_rent_recorded
                        and time.monotonic() >= self._lp_rent_retry_after.get(executor_id, 0.0)
                    ):
                        # Locked rent, stored while the position is still held rather than
                        # at completion — a position open for days should be answerable
                        # for the whole time it is open. Drops out of this branch as soon
                        # as it lands, so it is one read per executor, not one per tick.
                        await self._record_lp_position_rent(executor_id, executor)

                # Handle completed executors
                for executor_id in completed_ids:
                    await self._handle_executor_completion(executor_id)

                # Performance snapshots, after completion handling so a just-closed
                # executor is already out of _active_executors and gets only its terminal
                # row, not a duplicate periodic one. This is not a separate task on
                # purpose: the loop already ticks at 1 Hz and already awaits the database
                # inside the tick (see _record_lp_position_rent), so the cadence is one
                # guard rather than another thing to schedule and shut down.
                now = time.monotonic()
                if now - self._last_snapshot_at >= self.performance_snapshot_interval:
                    self._last_snapshot_at = now
                    await self._dump_executor_performance()
                    if (
                        self.performance_retention_days > 0
                        and now - self._last_prune_at >= self.PERFORMANCE_PRUNE_INTERVAL_SECONDS
                    ):
                        self._last_prune_at = now
                        await self._prune_performance_snapshots()

            except Exception as e:
                logger.error(f"Error in executor control loop: {e}", exc_info=True)

            await asyncio.sleep(self.update_interval)

    @staticmethod
    def _measured_rent(custom_info: Dict[str, Any], key: str) -> Optional[Decimal]:
        """A rent figure the executor actually observed, or None.

        The LP executor reports these as plain floats and defaults them to 0.0, so zero
        means "never measured" far more often than it means "measured and empty": a
        position still open has no refund yet, and an EVM CLMM has no rent at all. Rent
        that was genuinely locked is never zero. Storing the 0.0 would put a figure in
        the table that reads as an observation and is not one — the mistake GW-18 made,
        where a hardcoded 0 refund was indistinguishable from a position that earned
        nothing.
        """
        value = custom_info.get(key)
        if value is None:
            return None
        try:
            amount = Decimal(str(value))
        except (ArithmeticError, ValueError):
            return None
        return amount if amount > 0 else None

    async def _record_lp_position_rent(
        self, executor_id: str, executor: ExecutorBase, *, final: bool = False
    ) -> None:
        """Carry an LP executor's rent figures into the CLMM position table.

        ``final`` marks the call made from executor completion, which is the last one
        this executor will ever get: its retry state is torn down immediately after,
        so there is no later tick to hand the work to.

        `gateway_clmm_positions.position_rent` is written by hummingbot-api's OPEN route
        and `position_rent_refunded` by its CLOSE route. An executor holds its position
        through the wheel, talking to Gateway directly, so neither route runs: the poller
        discovers the position and files it with both columns NULL, and the close leaves
        no refund behind either. That is the recommended way to hold a CLMM position and
        the one path whose rent went unrecorded — ~0.0100572 SOL on Orca, more than the
        liquidity in a small position.

        The executor knows both figures; nothing was asking it. Locked rent is written as
        soon as the position row exists, rather than at completion, so a position held for
        days is answerable while it is held. The row may not exist yet on an early tick —
        discovery runs on its own schedule — in which case this is a no-op and the next
        tick tries again.
        """
        try:
            custom_info = executor.get_custom_info()
        except Exception as e:
            logger.debug(f"Could not read custom_info for {executor_id} while recording rent: {e}")
            return

        position_address = custom_info.get("position_address")
        if position_address:
            self._lp_position_addresses[executor_id] = position_address
        else:
            # Cleared by a successful close. The refund is only known now, so fall back to
            # the address this executor was holding.
            position_address = self._lp_position_addresses.get(executor_id)
        if not position_address:
            return

        position_rent = self._measured_rent(custom_info, "position_rent")
        position_rent_refunded = self._measured_rent(custom_info, "position_rent_refunded")
        if position_rent is None and position_rent_refunded is None:
            return

        try:
            async with self.db_manager.get_session_context() as session:
                position = await GatewayCLMMRepository(session).record_position_rent(
                    position_address,
                    position_rent=position_rent,
                    position_rent_refunded=position_rent_refunded,
                )
        except Exception as e:
            logger.error(f"Error recording rent for LP position {position_address}: {e}", exc_info=True)
            return

        if position is None:
            if not final:
                # The poller has simply not discovered the position yet; back off and let a
                # later tick try again. Not worth a warning while the executor still runs.
                self._lp_rent_retry_after[executor_id] = time.monotonic() + self.LP_RENT_RETRY_SECONDS
                return
            # Last call: scheduling a retry here would be theatre, since the caller clears
            # this executor's retry state on the next line. Say plainly what was lost and
            # what it was worth, so the figures can be recovered from the log.
            logger.error(
                f"LP executor {executor_id} finished holding position {position_address} "
                f"(rent={position_rent}, refund={position_rent_refunded}), but no row exists "
                "for it: the position was opened and closed inside a single discovery "
                "sweep, so it was never filed and these figures are not recorded anywhere "
                "but this log line."
            )
            return

        if position_rent is not None:
            self._lp_rent_recorded.add(executor_id)
        logger.debug(
            f"Recorded rent for LP position {position_address}: locked={position_rent}, "
            f"refunded={position_rent_refunded}"
        )

    async def _record_executor_swap(self, executor_id: str, executor: ExecutorBase) -> None:
        """Record a swap an executor made, so the swap history covers the recommended path.

        `gateway_swaps` is written by hummingbot-api's own /gateway/swap/* routes. An
        executor holds its connector through the wheel and talks to Gateway directly, so
        hummingbot-api never sees the call and had nothing to record. The table was not
        wrong, it was silently partial: POST /gateway/swaps/search and the swap summary
        described only swaps made by hand, and a caller reading "9 SOL-USDC swaps" had no
        way to learn that the most recent ones were missing.

        Same shape as _record_lp_position_rent: the executor knows what it did, so ask it
        at completion rather than waiting for a route that will never be called. Keyed on
        the transaction hash, which is what GW-43 added to custom_info — `order_id` is
        internal (buy-SOL-USDC-1787271213996599) and appears nowhere on chain, so before
        that there was nothing to key a row on.

        Skipped silently when there is no hash: a Gateway swap that never reached the
        chain has nothing to record, and an executor on a CEX is not a Gateway swap at all.
        """
        try:
            custom_info = executor.get_custom_info()
        except Exception as e:
            logger.debug(f"Could not read custom_info for {executor_id} while recording its swap: {e}")
            return

        transaction_hash = custom_info.get("transaction_hash")
        swap_provider = custom_info.get("swap_provider")
        if not transaction_hash or not swap_provider:
            return

        metadata = self._executor_metadata.get(executor_id, {})
        network = metadata.get("connector_name") or ""
        trading_pair = metadata.get("trading_pair") or ""
        if not network or "-" not in trading_pair:
            logger.warning(
                f"Executor {executor_id} swapped in {transaction_hash} but reports "
                f"network={network!r} pair={trading_pair!r}; not recorded."
            )
            return

        base_token, quote_token = split_trading_pair(trading_pair)
        side = str(custom_info.get("side") or "").upper()
        side = "BUY" if "BUY" in side else "SELL"

        amount_base = Decimal(str(custom_info.get("executed_amount_base") or 0))
        price = Decimal(str(custom_info.get("average_executed_price") or 0))
        if amount_base <= 0 or price <= 0:
            logger.warning(
                f"Executor {executor_id} swapped in {transaction_hash} but reports no "
                f"realized amounts (base={amount_base}, price={price}); not recorded."
            )
            return
        amount_quote = amount_base * price

        # A BUY spends quote to receive base; a SELL is the mirror image.
        input_amount, output_amount = (
            (amount_quote, amount_base) if side == "BUY" else (amount_base, amount_quote)
        )

        chain = network.split("-", 1)[0]
        # The provider travels as "jupiter/router"; the table stores the bare DEX name,
        # which is what the /gateway/swap routes write.
        connector = swap_provider.split("/", 1)[0]

        try:
            async with self.db_manager.get_session_context() as session:
                repo = GatewaySwapRepository(session)
                if await repo.get_swap_by_tx_hash(transaction_hash):
                    return
                await repo.create_swap({
                    "transaction_hash": transaction_hash,
                    "network": network,
                    "connector": connector,
                    "wallet_address": custom_info.get("wallet_address") or "",
                    "trading_pair": trading_pair,
                    "base_token": base_token,
                    "quote_token": quote_token,
                    "side": side,
                    "input_amount": input_amount,
                    "output_amount": output_amount,
                    "price": price,
                    # The LIVE tolerance the swap went out with, which is not
                    # config.slippage_pct when earlier attempts failed and widened it.
                    "slippage_pct": (Decimal(str(custom_info["slippage_pct"]))
                                     if custom_info.get("slippage_pct") is not None else None),
                    "gas_token": get_native_gas_token(chain),
                    "status": "CONFIRMED",
                })
            logger.info(
                f"Recorded executor swap {transaction_hash}: {side} {amount_base} "
                f"{base_token} @ {price} on {connector}/{network}"
            )
        except Exception as e:
            logger.error(f"Error recording executor swap {transaction_hash}: {e}", exc_info=True)

    # ========================================
    # Performance Snapshots
    # ========================================

    def _build_snapshot_row(
        self,
        executor_id: str,
        executor: ExecutorBase,
        *,
        is_terminal: bool,
        metrics: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        close_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """One executor_performance_snapshots row, from the two sources _format_executor_info reads.

        `metrics`, `status` and `close_type` let the terminal row reuse the figures
        _persist_executor_completed already computed rather than reading executor_info a
        second time -- the read that can raise and get silently substituted with zeros.

        Returns None when the metrics cannot be read at all: a row of fabricated zeros in
        the middle of a series is worse than a gap, because a reader cannot tell it from
        an executor that genuinely made nothing.
        """
        metadata = self._executor_metadata.get(executor_id, {})

        if metrics is None:
            try:
                info = executor.executor_info
                metrics = {
                    "net_pnl_quote": info.net_pnl_quote,
                    "net_pnl_pct": info.net_pnl_pct,
                    "cum_fees_quote": info.cum_fees_quote,
                    "filled_amount_quote": info.filled_amount_quote,
                }
            except Exception as e:
                logger.debug(f"Could not read executor_info for {executor_id} while snapshotting: {e}")
                return None

        if status is None:
            try:
                status = executor.status.name
            except Exception as e:
                logger.debug(f"Could not read status for {executor_id} while snapshotting: {e}")
                return None

        return {
            "executor_id": executor_id,
            "executor_type": metadata.get("executor_type") or "unknown",
            "account_name": metadata.get("account_name") or self.default_account,
            "connector_name": metadata.get("connector_name") or "",
            "trading_pair": metadata.get("trading_pair") or "",
            "controller_id": metadata.get("controller_id", "main"),
            "status": status,
            "close_type": close_type,
            "is_terminal": is_terminal,
            **metrics,
        }

    async def _dump_executor_performance(self):
        """Write one snapshot row per live executor.

        Only live executors are sampled because _active_executors is, by construction,
        the only place a running executor's performance exists -- its database row still
        carries the creation-time zeros until it completes. A closed executor is not
        missing from here: it got its terminal row at completion.

        Never lets a database failure out: this shares the control loop's tick, and a
        dropped snapshot must not stop executors from being updated.
        """
        if not self.db_manager or not self._active_executors:
            return

        snapshot_timestamp = datetime.now(timezone.utc)
        rows = []
        for executor_id, executor in list(self._active_executors.items()):
            row = self._build_snapshot_row(executor_id, executor, is_terminal=False)
            if row is not None:
                row["snapshot_timestamp"] = snapshot_timestamp
                rows.append(row)

        if not rows:
            return

        try:
            async with self.db_manager.get_session_context() as session:
                await ExecutorPerformanceRepository(session).save_snapshots(rows)
            logger.debug(f"Dumped {len(rows)} executor performance snapshots")
        except Exception as e:
            logger.error(f"Error saving executor performance snapshots: {e}", exc_info=True)

    async def _prune_performance_snapshots(self):
        """Delete snapshots older than the configured retention, from both snapshot tables.

        Gated on performance_retention_days > 0 by the caller: the default keeps
        everything, so an upgrade never starts deleting an operator's history.
        """
        if not self.db_manager or self.performance_retention_days <= 0:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.performance_retention_days)
        try:
            async with self.db_manager.get_session_context() as session:
                executor_rows, controller_rows = await ExecutorPerformanceRepository(
                    session
                ).prune_older_than(cutoff)
            if executor_rows or controller_rows:
                logger.info(
                    f"Pruned performance snapshots older than {cutoff.isoformat()}: "
                    f"{executor_rows} executor, {controller_rows} controller"
                )
        except Exception as e:
            logger.error(f"Error pruning performance snapshots: {e}", exc_info=True)

    async def get_executor_performance_history(
        self,
        executor_id: Optional[str] = None,
        executor_type: Optional[str] = None,
        controller_id: Optional[str] = None,
        account_name: Optional[str] = None,
        connector_name: Optional[str] = None,
        trading_pair: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = "5m"
    ):
        """Read an executor snapshot series. Mirrors BotsOrchestrator.get_controller_performance_history."""
        async with self.db_manager.get_session_context() as session:
            repo = ExecutorPerformanceRepository(
                session, grain_minutes=self.performance_snapshot_interval / 60.0
            )
            return await repo.get_performance_history(
                executor_id=executor_id,
                executor_type=executor_type,
                controller_id=controller_id,
                account_name=account_name,
                connector_name=connector_name,
                trading_pair=trading_pair,
                limit=limit,
                cursor=cursor,
                start_time=start_time,
                end_time=end_time,
                interval=interval,
            )

    async def get_latest_executor_performance(
        self,
        executor_id: Optional[str] = None,
        executor_type: Optional[str] = None,
        controller_id: Optional[str] = None,
        account_name: Optional[str] = None,
        connector_name: Optional[str] = None,
        trading_pair: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """The last snapshot of each matching executor. Mirrors
        BotsOrchestrator.get_latest_controller_performance."""
        async with self.db_manager.get_session_context() as session:
            repo = ExecutorPerformanceRepository(
                session, grain_minutes=self.performance_snapshot_interval / 60.0
            )
            return await repo.get_latest(
                executor_id=executor_id,
                executor_type=executor_type,
                controller_id=controller_id,
                account_name=account_name,
                connector_name=connector_name,
                trading_pair=trading_pair,
                limit=limit,
            )

    def _get_trading_interface(self, account_name: str) -> AccountTradingInterface:
        """Get or create an AccountTradingInterface for the account."""
        if account_name not in self._trading_interfaces:
            self._trading_interfaces[account_name] = self._trading_service.get_trading_interface(account_name)
        return self._trading_interfaces[account_name]

    def _validate_executor_config(
        self,
        executor_config: Dict[str, Any],
        default_timestamp: Optional[float] = None
    ) -> tuple[Type[ExecutorBase], Type[ExecutorConfigBase], ExecutorConfigBase]:
        """
        Validate the executor type and build the typed executor config.

        Pure validation step: no IO, no executor started, no DB access.

        Args:
            executor_config: Executor configuration dictionary (must include 'type')
            default_timestamp: Timestamp to set on the config if not provided
                (required for time-based features like time_limit)

        Returns:
            Tuple of (executor_class, config_class, typed_config)

        Raises:
            HTTPException: 400 if the type is missing/invalid or the config is invalid
        """
        executor_type = executor_config.get("type")
        if not executor_type:
            raise HTTPException(
                status_code=400,
                detail="executor_config must include 'type' field"
            )

        if executor_type not in self.EXECUTOR_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid executor type '{executor_type}'. Valid types: {list(self.EXECUTOR_REGISTRY.keys())}"
            )

        if "timestamp" not in executor_config or executor_config["timestamp"] is None:
            executor_config["timestamp"] = default_timestamp

        # The whole dict is passed to the config model. Range iceberg fields
        # (protective_exchange_stop, stop_loss_fallback_seconds,
        # allow_adverse_take_profit, protective_stop_buffer_pct,
        # exchange_stop_trigger_price) reach the executor once that model
        # declares them. This method does not keep a field whitelist.
        executor_class, config_class = self.EXECUTOR_REGISTRY[executor_type]
        try:
            typed_config = config_class(**executor_config)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid executor config: {str(e)}"
            )

        return executor_class, config_class, typed_config

    async def _prepare_market(self, account: str, connector_name: Optional[str], trading_pair: Optional[str]):
        """Ensure the connector and market for the executor are ready on the account's trading interface."""
        trading_interface = self._get_trading_interface(account)
        if connector_name:
            if trading_pair:
                await trading_interface.add_market(connector_name, trading_pair)
            else:
                await trading_interface.ensure_connector(connector_name)

    def _instantiate_and_register(
        self,
        executor_class: Type[ExecutorBase],
        typed_config: ExecutorConfigBase,
        trading_interface: AccountTradingInterface,
        metadata: Dict[str, Any]
    ) -> tuple[str, ExecutorBase]:
        """
        Instantiate the executor, register it in memory and start it.

        Args:
            executor_class: Executor class to instantiate
            typed_config: Validated typed executor config
            trading_interface: Trading interface acting as the executor's strategy
            metadata: Metadata dict to register for the executor

        Returns:
            Tuple of (executor_id, executor)

        Raises:
            HTTPException: 400 if the executor fails to instantiate
        """
        try:
            executor = executor_class(
                strategy=trading_interface,
                config=typed_config,
                update_interval=self.update_interval,
                max_retries=self.max_retries
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to create executor: {str(e)}"
            )

        executor_id = typed_config.id
        self._active_executors[executor_id] = executor
        self._executor_metadata[executor_id] = metadata

        # Set ContextVar so the asyncio Task created by start() inherits it
        token = current_executor_id.set(executor_id)
        executor.start()
        current_executor_id.reset(token)

        return executor_id, executor

    def _range_iceberg_slot_lock(self, slot: tuple) -> asyncio.Lock:
        """The lock for one account, connector, pair, and side.

        Per process only. A single uvicorn worker is assumed.
        """
        locks = getattr(self, "_range_iceberg_slot_locks", None)
        if locks is None:
            locks = {}
            self._range_iceberg_slot_locks = locks
        lock = locks.get(slot)
        if lock is None:
            lock = asyncio.Lock()
            locks[slot] = lock
        return lock

    @staticmethod
    def _range_iceberg_side_of_stored(config_raw: Any, final_state: Any) -> str:
        """Side of a stored executor. Config wins, then the saved custom_info."""
        if isinstance(config_raw, str) and config_raw:
            try:
                parsed = json.loads(config_raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("side") not in (None, ""):
                return _range_iceberg_side_name(parsed.get("side"))
        custom = _parse_stored_custom_info(final_state)
        if custom.get("side") not in (None, ""):
            return _range_iceberg_side_name(custom.get("side"))
        return "BUY"

    def _range_iceberg_memory_occupant(self, slot: tuple) -> Optional[str]:
        """Id of a live in-memory range iceberg on this slot, if any."""
        account, connector, pair, side = slot
        for executor_id, executor in self._active_executors.items():
            if self._executor_is_closed(executor):
                continue
            meta = self._executor_metadata.get(executor_id, {})
            if meta.get("executor_type") != "range_iceberg_executor":
                continue
            if (meta.get("account_name") or "") != account:
                continue
            if (meta.get("connector_name") or "") != connector:
                continue
            if _normalize_trading_pair(meta.get("trading_pair") or "") != pair:
                continue
            config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
            stored_side = config.get("side")
            if stored_side in (None, ""):
                stored_side = getattr(getattr(executor, "config", None), "side", None)
            if stored_side in (None, ""):
                info = self._safe_custom_info(executor)
                stored_side = info.get("side")
            if _range_iceberg_side_name(stored_side) == side:
                return executor_id
        return None

    def _executor_id_is_live(self, executor_id: str) -> bool:
        executor = self._active_executors.get(executor_id)
        return executor is not None and not self._executor_is_closed(executor)

    async def _range_iceberg_slot_rows(self, slot: tuple) -> List[tuple]:
        """Stored candidates on this slot.

        Columns are copied before the session closes. A later read of a
        detached row cannot see status, close_type, config, or final_state.
        The pair is matched after normalization.
        """
        if not getattr(self, "db_manager", None):
            return []
        account, connector, pair, side = slot
        stored: List[tuple] = []
        async with self.db_manager.get_session_context() as session:
            records = await ExecutorRepository(session).get_open_executors_for_pair(
                "range_iceberg_executor", account, connector
            )
            for record in records:
                stored.append((
                    record.executor_id,
                    record.status,
                    record.close_type,
                    record.trading_pair,
                    record.config,
                    record.final_state,
                ))
        matched: List[tuple] = []
        for row in stored:
            if _normalize_trading_pair(row[3] or "") != pair:
                continue
            if self._range_iceberg_side_of_stored(row[4], row[5]) != side:
                continue
            matched.append(row)
        return matched

    async def _exchange_side_is_flat(self, slot: tuple) -> bool:
        """True only when the position book was read and that side is flat.

        An unreadable book is not flat. The slot stays occupied.
        """
        account, connector, pair, side = slot
        snapshot = await self._exchange_open_base(account, connector, pair, side)
        return snapshot is not None and snapshot[0] == 0

    async def _exchange_stops_are_live(self, slot: tuple, custom_info: Any) -> bool:
        """True when a stored stop id is still open, or cannot be checked.

        No stored id means there is no live stop. A connector that cannot
        list stops leaves the stored ids treated as live.
        """
        stop_ids = _exchange_stop_ids(custom_info)
        if not stop_ids:
            return False
        account, connector_name, pair, _side = slot
        connector = self._account_connector(account, connector_name)
        lister = getattr(connector, "list_open_stop_orders", None) if connector is not None else None
        if not callable(lister):
            return True
        try:
            try:
                listed = lister(pair)
            except TypeError:
                listed = lister()
            if asyncio.iscoroutine(listed):
                listed = await listed
        except Exception:
            logger.error(
                "Could not list open stops for %s %s",
                connector_name,
                pair,
                exc_info=True,
            )
            return True
        rows = listed if isinstance(listed, list) else []
        for stop_id in stop_ids:
            if any(self._open_row_matches(row, stop_id, stop_id) for row in rows):
                return True
        return False

    async def _stale_range_iceberg_blocks(self, slot: tuple, custom_info: Any) -> bool:
        """A RUNNING row that is not in this process.

        Flat, with no live stop, does not block. An open position, a live
        stop, or an unreadable book does.
        """
        if not await self._exchange_side_is_flat(slot):
            return True
        return await self._exchange_stops_are_live(slot, custom_info)

    def _range_iceberg_conflict(self, slot: tuple, occupant: str) -> HTTPException:
        account, _connector, pair, side = slot
        detail = (
            f"A range iceberg is already open on {pair} {side} for {account} "
            f"(executor {occupant}). Close it first."
        )
        logger.info("%s", detail)
        return HTTPException(status_code=409, detail=detail)

    async def _assert_range_iceberg_slot_free(self, slot: tuple) -> None:
        """Reject a second open range iceberg on this account, connector, pair, and side.

        A live executor holds the slot. So does a TERMINATED POSITION_HOLD row
        that still has base or an exchange stop, until the exchange position
        on that side is flat or the row is marked slot_released. A RUNNING
        row that is not in memory holds the slot unless that side is flat and
        has no live stop. A long and a short are different slots, and so is
        the same side on another account.
        """
        occupant = self._range_iceberg_memory_occupant(slot)
        if occupant is None:
            for row in await self._range_iceberg_slot_rows(slot):
                executor_id, status, close_type, _pair, _config, final_state = row
                if self._executor_id_is_live(executor_id):
                    occupant = executor_id
                    break
                custom = _parse_stored_custom_info(final_state)
                if custom.get("slot_released") is True:
                    continue
                if (status or "") != "TERMINATED":
                    if await self._stale_range_iceberg_blocks(slot, custom):
                        occupant = executor_id
                        break
                    continue
                if close_type != "POSITION_HOLD":
                    continue
                if not _stored_position_hold_keeps_slot(custom):
                    continue
                if await self._exchange_side_is_flat(slot):
                    continue
                occupant = executor_id
                break
        if occupant is None:
            return
        raise self._range_iceberg_conflict(slot, occupant)

    async def release_range_iceberg_slot(
        self,
        account_name: str,
        connector_name: str,
        trading_pair: str,
        side: str,
        executor_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mark a parked range iceberg released so a new one can start.

        Does not close the position or cancel its exchange stop. A live
        in-memory executor is not released.
        """
        pair = _normalize_trading_pair(trading_pair)
        side_name = _range_iceberg_side_name(side)
        slot = (account_name, connector_name or "", pair, side_name)
        async with self._range_iceberg_slot_lock(slot):
            rows = await self._range_iceberg_slot_rows(slot)
            if executor_id:
                rows = [row for row in rows if row[0] == executor_id]
            target = None
            for row in rows:
                row_id, status, close_type, _pair, _config, final_state = row
                if self._executor_id_is_live(row_id):
                    if executor_id:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                f"Executor {row_id} is still running. "
                                "Stop it before releasing the slot."
                            ),
                        )
                    continue
                custom = _parse_stored_custom_info(final_state)
                if custom.get("slot_released") is True:
                    continue
                parked = close_type == "POSITION_HOLD" or (status or "") != "TERMINATED"
                if not parked:
                    continue
                target = row
                break
            if target is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No parked range iceberg on {pair} {side_name} "
                        f"for {account_name}."
                    ),
                )
            row_id = target[0]
            custom = dict(_parse_stored_custom_info(target[5]))
            custom["slot_released"] = True
            if not getattr(self, "db_manager", None):
                raise HTTPException(
                    status_code=404,
                    detail=f"Executor {row_id} not found",
                )
            async with self.db_manager.get_session_context() as session:
                updated = await ExecutorRepository(session).update_executor(
                    row_id,
                    final_state=json.dumps(custom),
                )
            if updated is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Executor {row_id} not found",
                )
        return {
            "executor_id": row_id,
            "released": True,
            "trading_pair": pair,
            "side": side_name,
            "account_name": account_name,
        }

    async def create_executor(
        self,
        executor_config: Dict[str, Any],
        account_name: Optional[str] = None,
        controller_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create and start a new executor.

        Args:
            executor_config: Executor configuration dictionary (must include 'type')
            account_name: Account to use (defaults to master_account)

        Returns:
            Dictionary with executor_id and initial status

        Raises:
            HTTPException: 409 when a range iceberg is already open on the same
            account, connector, trading pair, and side. The detail names that
            executor. A parked POSITION_HOLD still holds the slot until the
            exchange position is flat or the slot is released.
        """
        account = account_name or self.default_account
        trading_interface = self._get_trading_interface(account)

        # Validate executor type and build the typed config
        executor_class, _config_class, typed_config = self._validate_executor_config(
            executor_config, default_timestamp=trading_interface.current_timestamp
        )
        executor_type = executor_config["type"]

        # Ensure connector and market are ready
        connector_name = executor_config.get("connector_name")
        trading_pair = executor_config.get("trading_pair")
        await self._prepare_market(account, connector_name, trading_pair)

        # Instantiate the executor, register it in memory and start it
        controller_id = controller_id or getattr(typed_config, "controller_id", "main") or "main"
        metadata = {
            "account_name": account,
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "executor_type": executor_type,
            "controller_id": controller_id,
            "created_at": datetime.now(timezone.utc),
            "config": executor_config
        }
        slot = None
        if executor_type == "range_iceberg_executor":
            side_value = getattr(typed_config, "side", None)
            if side_value in (None, ""):
                side_value = executor_config.get("side")
            slot = (
                account,
                connector_name or "",
                _normalize_trading_pair(trading_pair),
                _range_iceberg_side_name(side_value),
            )

        async def _register_and_persist():
            created_id, created = self._instantiate_and_register(
                executor_class, typed_config, trading_interface, metadata
            )
            await self._persist_executor_created(created_id, created)
            return created_id, created

        if slot is None:
            executor_id, executor = await _register_and_persist()
        else:
            # The lock covers the open check and the register, so a second
            # create on this slot waits until the first is in memory and stored.
            # The lock is per process (single uvicorn worker).
            async with self._range_iceberg_slot_lock(slot):
                await self._assert_range_iceberg_slot_free(slot)
                executor_id, executor = await _register_and_persist()

        # Capture created_at before potential cleanup
        created_at = metadata["created_at"].isoformat()

        # Check if executor terminated immediately (e.g., insufficient balance)
        # If so, handle completion now rather than waiting for control loop
        if executor.is_closed:
            await self._handle_executor_completion(executor_id)

        logger.info(f"Created {executor_type} executor {executor_id} for {connector_name}/{trading_pair}")

        return {
            "executor_id": executor_id,
            "executor_type": executor_type,
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "controller_id": controller_id,
            "status": executor.status.name,
            "created_at": created_at
        }

    async def get_executors(
        self,
        account_name: Optional[str] = None,
        connector_name: Optional[str] = None,
        trading_pair: Optional[str] = None,
        executor_type: Optional[str] = None,
        status: Optional[str] = None,
        controller_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of executors with optional filtering.

        Combines active executors from memory with completed executors from database.

        Args:
            account_name: Filter by account name
            connector_name: Filter by connector name
            trading_pair: Filter by trading pair
            executor_type: Filter by executor type
            status: Filter by status
            controller_id: Filter by controller ID

        Returns:
            List of executor information dictionaries
        """
        result = []

        # Process active executors from memory
        for executor_id, executor in self._active_executors.items():
            metadata = self._executor_metadata.get(executor_id, {})

            # Apply filters
            if account_name and metadata.get("account_name") != account_name:
                continue
            if connector_name and metadata.get("connector_name") != connector_name:
                continue
            if trading_pair and metadata.get("trading_pair") != trading_pair:
                continue
            if executor_type and metadata.get("executor_type") != executor_type:
                continue
            if status and executor.status.name != status:
                continue
            if controller_id and metadata.get("controller_id", "main") != controller_id:
                continue

            result.append(self._format_executor_info(executor_id, executor))

        # Get completed executors from database
        if self.db_manager:
            try:
                async with self.db_manager.get_session_context() as session:
                    repo = ExecutorRepository(session)

                    db_executors = await repo.get_executors(
                        account_name=account_name,
                        connector_name=connector_name,
                        trading_pair=trading_pair,
                        executor_type=executor_type,
                        status=status,
                        controller_id=controller_id,
                        limit=limit
                    )

                    for record in db_executors:
                        # Skip if already in active executors (safety check)
                        if record.executor_id not in self._active_executors:
                            result.append(self._format_db_record(record))
            except Exception as e:
                logger.error(f"Error fetching executors from database: {e}")

        return result

    async def get_executor(self, executor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific executor.

        Checks active executors in memory first, then falls back to database.

        Args:
            executor_id: The executor ID

        Returns:
            Detailed executor information or None if not found
        """
        # Check active executors first (memory)
        executor = self._active_executors.get(executor_id)
        if executor:
            return self._format_executor_info(executor_id, executor)

        # Fallback to database for completed executors
        if self.db_manager:
            try:
                async with self.db_manager.get_session_context() as session:
                    repo = ExecutorRepository(session)

                    record = await repo.get_executor_by_id(executor_id)
                    if record:
                        return self._format_db_record(record)
            except Exception as e:
                logger.error(f"Error fetching executor from database: {e}")

        return None

    def get_executor_logs(
        self,
        executor_id: str,
        level: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Get captured log entries for an executor.

        Only available for active executors (logs are cleared on completion).

        Args:
            executor_id: The executor ID
            level: Optional filter by level (ERROR, WARNING, INFO, DEBUG)
            limit: Maximum number of entries to return

        Returns:
            List of log entry dicts
        """
        return self._log_capture.get_logs(executor_id, level=level, limit=limit)

    async def stop_executor(
        self,
        executor_id: str,
        keep_position: bool = False
    ) -> Dict[str, Any]:
        """
        Stop an active executor.

        Args:
            executor_id: The executor ID to stop
            keep_position: Whether to keep the position open

        Returns:
            Dictionary with stop confirmation
        """
        executor = self._active_executors.get(executor_id)
        if not executor:
            # Terminal executors are popped from memory within one control-loop tick,
            # so "not in memory" usually means "already terminated", not "unknown".
            # Fall back to the DB and answer with the final state as a no-op success.
            # This deliberately includes rows still marked RUNNING in the DB: an
            # executor known to the DB but absent from memory is dead regardless of
            # its stored status (completion race, failed persist, restart window),
            # and answering 404 there is the gateway#678 dead end. 404 is reserved
            # for executor ids the DB has never seen (or a DB outage, which
            # get_executor logs and swallows to None).
            db_record = await self.get_executor(executor_id)
            if db_record:
                custom_info = db_record.get("custom_info") or {}
                stop_ids = _exchange_stop_ids(custom_info)
                if (
                    stop_ids
                    and db_record.get("status") != "RUNNING"
                    and custom_info.get(_RESUME_ON_STARTUP) is not True
                    and db_record.get("close_type") != "POSITION_HOLD"
                ):
                    await self._cancel_exchange_stops(
                        db_record.get("account_name") or getattr(self, "default_account", "") or "",
                        db_record.get("connector_name") or "",
                        db_record.get("trading_pair") or "",
                        executor_id,
                        stop_ids,
                        "the executor is not RUNNING and was not parked for resume",
                    )
                logger.info(
                    f"Stop requested for already-terminated executor {executor_id} "
                    f"(db status: {db_record.get('status')}, close_type: {db_record.get('close_type')}) - no-op"
                )
                return {
                    "executor_id": executor_id,
                    "status": "already_terminated",
                    "keep_position": keep_position,
                    "close_type": db_record.get("close_type"),
                    "position_address": custom_info.get("position_address"),
                    "orphaned_position": bool(custom_info.get("orphaned_position", False)),
                    "hold_reason": custom_info.get("hold_reason"),
                }
            raise HTTPException(status_code=404, detail=f"Executor {executor_id} not found")

        if executor.is_closed:
            raise HTTPException(status_code=400, detail=f"Executor {executor_id} is already closed")

        # Trigger early stop. early_stop cancels software orders and leaves the
        # exchange stop in place for both keep_position values. A stop that does
        # not keep the position has to drop that exchange stop too. The executor
        # cancels it when the position goes flat. This is the safety net.
        try:
            executor.early_stop(keep_position=keep_position)
        except Exception as e:
            logger.error(f"Error stopping executor {executor_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error stopping executor: {str(e)}")

        if not keep_position:
            metadata = self._executor_metadata.get(executor_id, {})
            stop_ids = _exchange_stop_ids(self._safe_custom_info(executor))
            if stop_ids:
                await self._cancel_exchange_stops(
                    metadata.get("account_name") or getattr(self, "default_account", "") or "",
                    metadata.get("connector_name") or "",
                    metadata.get("trading_pair") or "",
                    executor_id,
                    stop_ids,
                    "stop was requested without keep_position",
                )

        logger.info(f"Initiated stop for executor {executor_id} (keep_position={keep_position})")

        return {
            "executor_id": executor_id,
            "status": "stopping",
            "keep_position": keep_position
        }

    async def get_orphaned_positions(self) -> List[Dict[str, Any]]:
        """
        List executors that terminated while potentially still owning an on-chain position.

        Covers both orphan classes:
        - close_type FAILED with a position_address in the persisted final state
          (e.g. an LP close that exhausted retries - gateway#678)
        - close_type SYSTEM_CLEANUP (RUNNING rows rewritten after an API restart);
          these have no final state, so the position address is unknown and the
          on-chain state must be reconciled externally

        This listing is DB-side only: cross-check candidates against on-chain reality
        (gateway CLMM/AMM positions-owned endpoints) before recovering. Recovered
        orphans are silenced with resolve_orphaned_position().

        Raises on DB errors rather than returning [] - "no orphans" from a broken DB
        would read as all-clear on a safety endpoint.
        """
        if not self.db_manager:
            raise RuntimeError("Orphan listing unavailable: no database configured")

        # lp_executor is the only executor type that owns an on-chain position
        # account; filtering in SQL keeps the limit meaningful (a Python-side filter
        # over the newest N mixed candidates can silently drop older real orphans).
        async with self.db_manager.get_session_context() as session:
            repo = ExecutorRepository(session)
            records = await repo.get_executors_by_close_types(
                ["FAILED", "SYSTEM_CLEANUP", "POSITION_HOLD"], executor_type="lp_executor"
            )

        orphans: List[Dict[str, Any]] = []
        for record in records:
            final_state: Dict[str, Any] = {}
            if record.final_state:
                try:
                    final_state = json.loads(record.final_state)
                except (json.JSONDecodeError, TypeError):
                    final_state = {}

            if final_state.get("orphan_resolved"):
                continue

            position_address = final_state.get("position_address")
            if record.close_type == "FAILED" and not position_address:
                # Failed without on-chain exposure - not an orphan
                continue
            if record.close_type == "POSITION_HOLD" and not (
                final_state.get("orphaned_position") or final_state.get("hold_reason")
            ):
                # Voluntary hold (keep_position=True stop) - position was closed on-chain
                continue

            # The DEX and pool live in the executor config, not in any column: for lp_executor the
            # connector_name column carries the network id. Both are needed to close the position,
            # and LP-executor positions are opened by the bot straight against Gateway so they are
            # never in the API's own CLMM position table to be looked up there.
            config: Dict[str, Any] = {}
            if record.config:
                try:
                    config = json.loads(record.config)
                except (json.JSONDecodeError, TypeError):
                    config = {}

            orphans.append({
                "executor_id": record.executor_id,
                "executor_type": record.executor_type,
                "account_name": record.account_name,
                "connector_name": record.connector_name,
                "trading_pair": record.trading_pair,
                "lp_provider": config.get("lp_provider"),
                "pool_address": config.get("pool_address"),
                "controller_id": record.controller_id or "main",
                "close_type": record.close_type,
                "closed_at": record.closed_at.isoformat() if record.closed_at else None,
                "position_address": position_address,
                "state": final_state.get("state"),
                "hold_reason": final_state.get("hold_reason"),
                "needs_onchain_reconciliation": position_address is None,
            })

        return orphans

    async def resolve_orphaned_position(self, executor_id: str) -> Dict[str, Any]:
        """
        Mark an orphaned position as recovered so it stops surfacing.

        Call this after the stranded on-chain position has been closed (or adopted)
        externally. Sets orphan_resolved in the persisted final state, which removes
        the record from get_orphaned_positions() and from agent-facing warnings.
        """
        if not self.db_manager:
            raise HTTPException(status_code=503, detail="No database configured")

        async with self.db_manager.get_session_context() as session:
            repo = ExecutorRepository(session)
            record = await repo.get_executor_by_id(executor_id)
            if not record:
                raise HTTPException(status_code=404, detail=f"Executor {executor_id} not found")
            if record.status == "RUNNING":
                raise HTTPException(status_code=400, detail=f"Executor {executor_id} is still running")

            final_state: Dict[str, Any] = {}
            if record.final_state:
                try:
                    final_state = json.loads(record.final_state)
                except (json.JSONDecodeError, TypeError):
                    final_state = {}

            is_involuntary_hold = record.close_type == "POSITION_HOLD" and (
                final_state.get("orphaned_position") or final_state.get("hold_reason")
            )
            if record.close_type not in ("FAILED", "SYSTEM_CLEANUP") and not is_involuntary_hold:
                raise HTTPException(
                    status_code=400,
                    detail=f"Executor {executor_id} (close_type: {record.close_type}) is not an orphan candidate",
                )
            final_state["orphaned_position"] = False
            final_state["orphan_resolved"] = True
            await repo.update_executor(executor_id=executor_id, final_state=json.dumps(final_state))

        logger.info(f"Orphaned position for executor {executor_id} marked resolved")
        return {"executor_id": executor_id, "orphan_resolved": True}

    async def _handle_executor_completion(self, executor_id: str):
        """Handle cleanup when an executor completes."""
        # Atomically claim the executor so a concurrent completion (e.g. the
        # control loop racing with the synchronous call in create_executor)
        # returns early instead of double-persisting / double-aggregating.
        executor = self._active_executors.pop(executor_id, None)
        if executor is None:
            return

        metadata = self._executor_metadata.get(executor_id, {})

        # Check if this is a POSITION_HOLD close type (keep_position=True)
        close_name = getattr(getattr(executor, "close_type", None), "name", None)
        if executor.close_type == CloseType.POSITION_HOLD or close_name == "POSITION_HOLD":
            await self._aggregate_position_hold(executor_id, executor, metadata)
        else:
            # FAILED and a user stop without keep_position leave the exchange
            # stop up. Cancel it here, and again on the next startup if this misses.
            stop_ids = _exchange_stop_ids(self._safe_custom_info(executor))
            if stop_ids:
                await self._cancel_exchange_stops(
                    metadata.get("account_name") or getattr(self, "default_account", "") or "",
                    metadata.get("connector_name") or "",
                    metadata.get("trading_pair") or "",
                    executor_id,
                    stop_ids,
                    "the executor ended without keep_position",
                )

        # Persist final state to database
        await self._persist_executor_completed(executor_id, executor)

        # The rent refund is only known once the close confirms, which is here. A
        # successful close has already cleared position_address from custom_info, so this
        # relies on the address remembered while the executor was live.
        if metadata.get("executor_type") == "lp_executor":
            await self._record_lp_position_rent(executor_id, executor, final=True)
        # A Gateway swap an executor made, which no /gateway/swap route ever saw. See
        # _record_executor_swap; non-Gateway executors carry no transaction hash and fall
        # straight back out.
        if self.db_manager:
            await self._record_executor_swap(executor_id, executor)
        self._lp_position_addresses.pop(executor_id, None)
        self._lp_rent_recorded.discard(executor_id)
        self._lp_rent_retry_after.pop(executor_id, None)
        getattr(self, "_persisted_child_order_ids", {}).pop(executor_id, None)

        # Active executor already claimed via pop above; drop its metadata last
        # (metadata is read above and re-fetched inside the persist/aggregate
        # helpers, so it must stay until after those awaits complete).
        if executor_id in self._executor_metadata:
            del self._executor_metadata[executor_id]

        # Clean up captured logs
        self._log_capture.clear(executor_id)

        close_type = executor.close_type.name if executor.close_type else "UNKNOWN"
        logger.info(f"Executor {executor_id} completed with close_type: {close_type}")

        # Surface stranded on-chain exposure loudly: a live position address on an
        # involuntary hold (hold_reason set) or a legacy FAILED means the position
        # has no automated owner from this point on.
        if executor.close_type in (CloseType.FAILED, CloseType.POSITION_HOLD):
            try:
                completion_info = executor.get_custom_info()
                position_address = completion_info.get("position_address")
                hold_reason = completion_info.get("hold_reason")
            except Exception:
                position_address = None
                hold_reason = None
            if position_address and (hold_reason or executor.close_type == CloseType.FAILED):
                logger.error(
                    f"Executor {executor_id} ended {close_type} with position {position_address} "
                    f"still open on-chain (hold_reason: {hold_reason}) - orphaned position requires "
                    "recovery (flagged in DB record; see /executors/positions/orphaned)"
                )

    def _format_executor_info(
        self,
        executor_id: str,
        executor: ExecutorBase
    ) -> Dict[str, Any]:
        """Format executor information for API response."""
        metadata = self._executor_metadata.get(executor_id, {})
        executor_type = metadata.get("executor_type")

        # Get executor_info as a dict and strip heavy custom_info fields BEFORE
        # serialization so they never get coerced (fill_events, grid
        # levels_by_state, etc.); then coerce in-place to JSON-compatible
        # primitives instead of doing a json.dumps/json.loads string round-trip.
        executor_info = executor.executor_info
        dumped = executor_info.model_dump()
        dumped["custom_info"] = self._strip_heavy_fields(dumped.get("custom_info"), executor_type)
        result = _coerce_json_compatible(dumped)

        # Add metadata
        result["executor_id"] = executor_id
        result["executor_type"] = executor_type
        result["account_name"] = metadata.get("account_name")
        result["created_at"] = metadata.get("created_at").isoformat() if metadata.get("created_at") else None

        if metadata.get("connector_name"):
            result["connector_name"] = metadata.get("connector_name")
        if metadata.get("trading_pair"):
            result["trading_pair"] = metadata.get("trading_pair")
        result["controller_id"] = metadata.get("controller_id", "main")

        # Read status/close_type directly from executor
        result["status"] = executor.status.name
        result["close_type"] = executor.close_type.name if executor.close_type else None
        result["is_active"] = not executor.is_closed

        # Add side from executor_info (it's a property, not serialized by model_dump)
        side = executor_info.side
        if side is not None:
            # Convert TradeType enum or int to string
            result["side"] = side.name if hasattr(side, 'name') else str(side)

        # Add log capture info
        result["error_count"] = self._log_capture.get_error_count(executor_id)
        result["last_error"] = self._log_capture.get_last_error(executor_id)
        result.update(_resume_search_fields(result.get("custom_info")))

        return result

    @staticmethod
    def _strip_heavy_fields(custom_info: Optional[Dict], executor_type: Optional[str] = None) -> Optional[Dict]:
        """Remove heavy fields from custom_info to reduce payload size."""
        if not custom_info:
            return custom_info
        heavy_fields = {"fill_events"}
        if executor_type == "grid_executor":
            heavy_fields |= {"levels_by_state", "filled_orders", "failed_orders", "canceled_orders"}
        return {k: v for k, v in custom_info.items() if k not in heavy_fields}

    def _format_db_record(self, record) -> Dict[str, Any]:
        """Format a database ExecutorRecord for API response."""
        # Parse error_log from DB for completed executors
        error_count = 0
        last_error = None
        if record.error_log:
            try:
                errors = json.loads(record.error_log)
                error_count = len(errors)
                if errors:
                    last_error = errors[-1].get("message")
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "executor_id": record.executor_id,
            "executor_type": record.executor_type,
            "account_name": record.account_name,
            "connector_name": record.connector_name,
            "trading_pair": record.trading_pair,
            "side": None,
            "status": record.status,
            "close_type": record.close_type,
            "is_active": record.status == "RUNNING",
            "is_trading": False,
            "timestamp": None,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "close_timestamp": record.closed_at.timestamp() if record.closed_at else None,
            "closed_at": record.closed_at.isoformat() if record.closed_at else None,
            "controller_id": record.controller_id or "main",
            "net_pnl_quote": float(record.net_pnl_quote) if record.net_pnl_quote else 0.0,
            "net_pnl_pct": float(record.net_pnl_pct) if record.net_pnl_pct else 0.0,
            "cum_fees_quote": float(record.cum_fees_quote) if record.cum_fees_quote else 0.0,
            "filled_amount_quote": float(record.filled_amount_quote) if record.filled_amount_quote else 0.0,
            "config": json.loads(record.config) if record.config else None,
            "custom_info": self._strip_heavy_fields(
                json.loads(record.final_state), record.executor_type
            ) if record.final_state else None,
            "error_count": error_count,
            "last_error": last_error,
            **_resume_search_fields(
                json.loads(record.final_state) if record.final_state else None
            ),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for active executors.

        Returns:
            Dictionary with aggregate statistics for active executors only.
        """
        executors = []

        # Get active executors from memory
        for executor_id, executor in self._active_executors.items():
            executors.append(self._format_executor_info(executor_id, executor))

        active_count = len(executors)
        total_pnl = sum(e.get("net_pnl_quote", 0) for e in executors)
        # filled_amount_quote is the volume traded on every executor type — an LP
        # executor derives it from the fees it earned rather than the capital it put up,
        # so this sums like with like and no separate field is needed.
        total_volume = sum(e.get("filled_amount_quote", 0) for e in executors)

        by_type: Dict[str, int] = {}
        by_connector: Dict[str, int] = {}
        by_status: Dict[str, int] = {}

        for e in executors:
            ex_type = e.get("executor_type", "unknown")
            connector = e.get("connector_name", "unknown")
            status = e.get("status", "unknown")

            by_type[ex_type] = by_type.get(ex_type, 0) + 1
            by_connector[connector] = by_connector.get(connector, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_active": active_count,
            "total_pnl_quote": total_pnl,
            "total_volume_quote": total_volume,
            "by_type": by_type,
            "by_connector": by_connector,
            "by_status": by_status
        }

    async def get_performance_report(
        self,
        controller_id: Optional[str] = None,
        market_data_service=None
    ) -> Dict[str, Any]:
        """
        Generate a performance report aggregating executor metrics.

        Combines database aggregations (completed executors) with in-memory
        active executor and position hold unrealized PnL.
        Excludes POSITION_HOLD close_type from realized PnL to avoid double-counting.

        Args:
            controller_id: Filter by controller ID (None = all)
            market_data_service: MarketDataService for position hold unrealized PnL

        Returns:
            Dictionary with performance metrics ready for PerformanceReportResponse.
        """
        report: Dict[str, Any] = {
            "controller_id": controller_id,
            "total_executors": 0,
            "by_status": {},
            "pnl_total_quote": 0.0,
            "unrealized_pnl_quote": 0.0,
            "global_pnl_quote": 0.0,
            "pnl_pct_avg": 0.0,
            "fees_total_quote": 0.0,
            "volume_total_quote": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": None,
            "by_type": [],
            "active_positions": 0,
        }

        if self.db_manager:
            # No try/except here on purpose: a database outage used to be swallowed and
            # reported as the zeroed report below, making it indistinguishable from
            # "no executors yet" to every consumer -- the route answered 200 with zeroes
            # and the /ws/executors performance channel pushed a confident row of them.
            # The failure now propagates: the route turns it into a 500 and the push loop
            # sends an error frame. The zeroed report is left to mean an empty dataset.
            async with self.db_manager.get_session_context() as session:
                repo = ExecutorRepository(session)
                db_data = await repo.get_performance_report(controller_id=controller_id)

            report["total_executors"] = db_data["total_executors"]
            report["by_status"] = db_data["status_counts"]
            report["pnl_total_quote"] = db_data["pnl_total_quote"]
            report["pnl_pct_avg"] = db_data["pnl_pct_avg"]
            report["fees_total_quote"] = db_data["fees_total_quote"]
            report["volume_total_quote"] = db_data["volume_total_quote"]
            report["win_rate"] = db_data["win_rate"]
            report["by_type"] = db_data["by_type"]

            # Sharpe ratio: mean(pnl) / std(pnl). The repository returns the sample
            # standard deviation as an aggregate -- None below 2 completed executors --
            # so this never depends on the number of rows in the table.
            pnl_std = db_data["pnl_std"]
            completed_count = db_data["completed_count"]
            if pnl_std:  # None below 2 executors, 0.0 when every PnL is identical
                mean_pnl = db_data["pnl_total_quote"] / completed_count
                report["sharpe_ratio"] = round(mean_pnl / pnl_std, 4)

        # --- Unrealized PnL from active executors ---
        unrealized_pnl = 0.0
        for executor_id, executor in self._active_executors.items():
            metadata = self._executor_metadata.get(executor_id, {})
            if controller_id and metadata.get("controller_id", "main") != controller_id:
                continue
            try:
                unrealized_pnl += float(executor.executor_info.net_pnl_quote)
            except Exception:
                pass

        # --- Unrealized PnL from position holds ---
        positions = self.get_positions_held(controller_id=controller_id)
        report["active_positions"] = len(positions)

        # Accumulate fees from position holds (already paid, reduce PnL)
        position_hold_fees = sum(float(p.cum_fees_quote) for p in positions)

        if market_data_service:
            # First pass: try oracle for each position, collect misses grouped by connector
            missing_by_connector: Dict[str, List[tuple]] = {}  # connector_key -> [(position, trading_pair)]
            for p in positions:
                # A hyphenated base symbol produced three parts and was skipped here,
                # so the position simply contributed nothing to unrealized PnL.
                try:
                    base, quote = split_trading_pair(p.trading_pair)
                except InvalidTradingPair:
                    continue
                rate = market_data_service.get_rate(base, quote)
                if rate is not None:
                    unrealized_pnl += float(p.get_unrealized_pnl(rate))
                else:
                    # Group by connector+account for batch fallback
                    connector_key = f"{p.connector_name}|{p.account_name}"
                    missing_by_connector.setdefault(connector_key, []).append((p, p.trading_pair))

            # Second pass: batch-fetch missing prices from the actual connectors
            for connector_key, items in missing_by_connector.items():
                connector_name, account_name = connector_key.split("|", 1)
                trading_pairs = [tp for _, tp in items]
                try:
                    prices = await market_data_service.get_prices(
                        connector_name=connector_name,
                        trading_pairs=trading_pairs,
                        account_name=account_name,
                    )
                    if isinstance(prices, dict) and "error" not in prices:
                        for pos, tp in items:
                            price = prices.get(tp)
                            if price is not None and price > 0:
                                unrealized_pnl += float(pos.get_unrealized_pnl(Decimal(str(price))))
                except Exception as e:
                    logger.warning(f"Fallback price fetch failed for {connector_name}: {e}")

        # Subtract position hold fees from unrealized PnL
        unrealized_pnl -= position_hold_fees

        report["unrealized_pnl_quote"] = round(unrealized_pnl, 8)
        report["position_hold_fees_quote"] = round(position_hold_fees, 8)
        report["global_pnl_quote"] = round(report["pnl_total_quote"] + unrealized_pnl, 8)

        return report

    async def _persist_executor_created(self, executor_id: str, executor: ExecutorBase):
        """Persist executor creation to database."""
        if not self.db_manager:
            return

        try:
            metadata = self._executor_metadata.get(executor_id, {})

            async with self.db_manager.get_session_context() as session:
                repo = ExecutorRepository(session)

                await repo.create_executor(
                    executor_id=executor_id,
                    executor_type=metadata.get("executor_type"),
                    account_name=metadata.get("account_name"),
                    connector_name=metadata.get("connector_name"),
                    trading_pair=metadata.get("trading_pair"),
                    config=json.dumps(metadata.get("config", {}), default=_json_default),
                    status=executor.status.name,
                    controller_id=metadata.get("controller_id", "main")
                )

            logger.debug(f"Persisted executor {executor_id} creation to database")

        except IntegrityError:
            # The executor closed before this INSERT landed and its completion already
            # wrote the row (see upsert_executor_completion). That row carries the final
            # state; re-inserting a RUNNING one is exactly what must not happen.
            logger.debug(
                f"Executor {executor_id} row already written by its completion; "
                f"skipping the creation insert"
            )
        except Exception as e:
            logger.error(f"Error persisting executor creation: {e}")

    async def _persist_executor_completed(self, executor_id: str, executor: ExecutorBase):
        """Persist executor completion to database.

        A failed write is logged and tried once more. Without that retry a
        finished executor stays RUNNING and holds its slot.
        """
        if not self.db_manager:
            return
        for attempt in (1, 2):
            try:
                await self._persist_executor_completed_once(executor_id, executor)
                return
            except Exception as error:
                logger.error(
                    "Error persisting executor completion for %s (attempt %s of 2): %s",
                    executor_id,
                    attempt,
                    error,
                )
        logger.error(
            "Completion write for executor %s failed after one retry. "
            "The row may still be RUNNING.",
            executor_id,
        )

    async def _persist_executor_completed_once(self, executor_id: str, executor: ExecutorBase):
        """One attempt to write the completion row."""
        try:
            # Read status/close_type directly from executor (most reliable)
            status_name = executor.status.name
            close_type = executor.close_type.name if executor.close_type else None

            # Get PnL values from executor_info
            try:
                executor_info = executor.executor_info
                net_pnl_quote = executor_info.net_pnl_quote
                net_pnl_pct = executor_info.net_pnl_pct
                cum_fees_quote = executor_info.cum_fees_quote
                filled_amount_quote = executor_info.filled_amount_quote
                metrics_are_measured = True
            except Exception as e:
                logger.debug(f"Error accessing executor_info for persistence: {e}")
                net_pnl_quote = Decimal("0")
                net_pnl_pct = Decimal("0")
                cum_fees_quote = Decimal("0")
                filled_amount_quote = Decimal("0")
                # The record still takes these zeros, as it always has. The snapshot
                # series does not: a terminal row is what a reader takes for the
                # executor's final value, and a fabricated zero there is indistinguishable
                # from an executor that genuinely made nothing.
                metrics_are_measured = False

            # Get custom_info directly from executor to avoid Pydantic serialization issues
            # with TrackedOrder and other complex types
            custom_info = executor.get_custom_info()

            # A stranded live on-chain position: an involuntary hold (close retries
            # exhausted -> POSITION_HOLD with hold_reason set, gateway#678) or a legacy
            # FAILED-with-position (force-stop straggler, older wheel). Flag it in the
            # persisted final_state so /executors/positions/orphaned, dashboards, and
            # agents can find and recover it. Voluntary holds never match: a successful
            # close clears position_address before the executor terminates.
            if custom_info.get("position_address") and (
                custom_info.get("hold_reason") or close_type == "FAILED"
            ):
                custom_info["orphaned_position"] = True
            # Serialize custom_info, fallback to None if serialization fails
            final_state_json = None
            metadata = self._executor_metadata.get(executor_id, {})
            executor_type = metadata.get("executor_type")
            if executor_type == "grid_executor":
                heavy_fields = {
                    "levels_by_state",
                    "filled_orders",
                    "failed_orders",
                    "canceled_orders",
                }
                custom_info = {k: v for k, v in custom_info.items() if k not in heavy_fields}

            try:
                final_state_json = json.dumps(custom_info, default=_json_default)
            except Exception as e:
                logger.warning(f"Failed to serialize custom_info for {executor_id}: {e}")
                # Try a simpler serialization without complex objects
                try:
                    simple_info = {k: v for k, v in custom_info.items()
                                   if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                    final_state_json = json.dumps(simple_info)
                except Exception:
                    final_state_json = None

            # Capture error logs before persisting
            error_log_json = None
            error_count = self._log_capture.get_error_count(executor_id)
            if error_count > 0:
                try:
                    error_entries = self._log_capture.get_logs(executor_id, level="ERROR")
                    error_log_json = json.dumps([
                        {
                            "timestamp": entry.get("timestamp"),
                            "message": entry.get("message"),
                            "exc_info": entry.get("exc_info"),
                        }
                        for entry in error_entries
                    ])
                except Exception as e:
                    logger.debug(f"Failed to serialize error logs for {executor_id}: {e}")

            async with self.db_manager.get_session_context() as session:
                repo = ExecutorRepository(session)

                # Upsert, not update: an executor that closes in milliseconds can reach
                # here before _persist_executor_created's INSERT has landed (or after it
                # failed outright), and a plain select-then-update would silently drop
                # this final state, leaving a phantom RUNNING executor forever.
                record, repaired = await repo.upsert_executor_completion(
                    executor_id=executor_id,
                    executor_type=executor_type,
                    account_name=metadata.get("account_name"),
                    connector_name=metadata.get("connector_name"),
                    trading_pair=metadata.get("trading_pair"),
                    controller_id=metadata.get("controller_id", "main"),
                    config=json.dumps(metadata.get("config", {}), default=_json_default),
                    created_at=metadata.get("created_at"),
                    status=status_name,
                    close_type=close_type,
                    net_pnl_quote=net_pnl_quote,
                    net_pnl_pct=net_pnl_pct,
                    cum_fees_quote=cum_fees_quote,
                    filled_amount_quote=filled_amount_quote,
                    final_state=final_state_json,
                    error_log=error_log_json
                )

                # The terminal row, in the same transaction as the record update so the
                # two never disagree. It is what makes a closed executor's series
                # answerable from executor_performance_snapshots alone -- no join, and no
                # "and then append the final value" rule for every future reader.
                #
                # Behind a SAVEPOINT, because the direction of the coupling matters: the
                # record update is the accounting and the snapshot is a point on a chart,
                # so a failing snapshot must roll back only itself. Without it a bad
                # INSERT here would abort the transaction and lose the completion, which
                # is the failure 60041a8 went to some trouble to make impossible.
                if record is not None and metrics_are_measured:
                    terminal_row = self._build_snapshot_row(
                        executor_id,
                        executor,
                        is_terminal=True,
                        metrics={
                            "net_pnl_quote": net_pnl_quote,
                            "net_pnl_pct": net_pnl_pct,
                            "cum_fees_quote": cum_fees_quote,
                            "filled_amount_quote": filled_amount_quote,
                        },
                        status=status_name,
                        close_type=close_type,
                    )
                    if terminal_row is not None:
                        try:
                            async with session.begin_nested():
                                await ExecutorPerformanceRepository(session).save_snapshots(
                                    [terminal_row]
                                )
                        except Exception as e:
                            logger.error(
                                f"Could not write the terminal performance snapshot for "
                                f"{executor_id}; its series ends at the last periodic row: {e}"
                            )

            if record is None:
                logger.error(
                    f"Could not persist completion for executor {executor_id}: no row to "
                    f"update and the repair insert did not take"
                )
                raise RuntimeError(
                    f"completion row for executor {executor_id} was not written"
                )
            elif repaired:
                logger.warning(
                    f"Executor {executor_id} completed before its creation row existed; "
                    f"inserted the record from its final state"
                )
            else:
                logger.debug(f"Persisted executor {executor_id} completion to database")

        except Exception:
            raise

    # ========================================
    # Position Hold Tracking Methods
    # ========================================

    def _get_position_key(
        self,
        account_name: str,
        connector_name: str,
        trading_pair: str,
        controller_id: str = "main"
    ) -> str:
        """Generate a unique key for position tracking."""
        return f"{account_name}|{connector_name}|{trading_pair}|{controller_id}"

    async def _aggregate_position_hold(
        self,
        executor_id: str,
        executor: ExecutorBase,
        metadata: Dict[str, Any]
    ):
        """
        Aggregate position data from an executor stopped with keep_position=True.

        This extracts the filled amounts from the executor and adds them to
        the aggregated position tracking.
        """
        account_name = metadata.get("account_name", self.default_account)
        connector_name = metadata.get("connector_name", "")
        trading_pair = metadata.get("trading_pair", "")
        controller_id = metadata.get("controller_id", "main")

        if not connector_name or not trading_pair:
            logger.warning(f"Cannot aggregate position for executor {executor_id}: missing connector/pair info")
            return

        position_key = self._get_position_key(account_name, connector_name, trading_pair, controller_id)

        # Get or create position hold
        if position_key not in self._positions_held:
            self._positions_held[position_key] = PositionHold(
                trading_pair=trading_pair,
                connector_name=connector_name,
                account_name=account_name,
                controller_id=controller_id
            )

        position = self._positions_held[position_key]

        # Extract filled amounts from executor
        try:
            # Try to get executor info
            try:
                executor_info = executor.executor_info
                custom_info = executor_info.custom_info or {}
            except Exception:
                custom_info = executor.get_custom_info() if hasattr(executor, 'get_custom_info') else {}

            # Get side from config or custom_info
            config = metadata.get("config", {})
            side = config.get("side", custom_info.get("side", "BUY"))

            # Extract filled amounts - try different sources
            filled_amount_base = Decimal("0")
            filled_amount_quote = Decimal("0")
            executor_type = metadata.get("executor_type")
            # Range iceberg reports open_base / filled_base / closed_base. A zero
            # open_base is a real empty position and must not fall through to
            # filled_amount_base, which these executors leave at 0.
            used_iceberg_base = False
            if executor_type == "range_iceberg_executor":
                held_base = range_iceberg_held_base(executor, custom_info)
                if held_base is not None:
                    filled_amount_base = held_base
                    used_iceberg_base = True

            # Try from executor attributes directly
            if used_iceberg_base:
                # Quote follows the open base, not the full entry quote.
                held_quote = range_iceberg_held_quote(executor, custom_info, filled_amount_base)
                filled_amount_quote = held_quote if held_quote is not None else Decimal("0")

            if not used_iceberg_base and hasattr(executor, 'filled_amount_base'):
                filled_amount_base = Decimal(str(executor.filled_amount_base or 0))
            if not used_iceberg_base and hasattr(executor, 'filled_amount_quote'):
                filled_amount_quote = Decimal(str(executor.filled_amount_quote or 0))

            # Fallback to custom_info
            if not used_iceberg_base and filled_amount_base == 0 and custom_info:
                filled_amount_base = Decimal(str(custom_info.get("filled_amount_base", 0)))
            if not used_iceberg_base and filled_amount_quote == 0 and custom_info:
                filled_amount_quote = Decimal(str(custom_info.get("filled_amount_quote", 0)))

            # Check for held_position_orders (used by grid_executor, position_executor, etc.)
            held_orders = custom_info.get("held_position_orders", []) if custom_info else []

            # Extract cumulative fees from the executor
            executor_fees = Decimal("0")
            try:
                executor_fees = Decimal(str(executor.cum_fees_quote or 0))
            except Exception:
                pass

            if held_orders:
                buy_filled_base = Decimal("0")
                buy_filled_quote = Decimal("0")
                sell_filled_base = Decimal("0")
                sell_filled_quote = Decimal("0")
                orders_fees = Decimal("0")

                for order in held_orders:
                    if isinstance(order, dict):
                        trade_type = order.get("trade_type", "BUY")
                        exec_base = Decimal(str(order.get("executed_amount_base", 0)))
                        exec_quote = Decimal(str(order.get("executed_amount_quote", 0)))
                        orders_fees += Decimal(str(order.get("cumulative_fee_paid_quote", 0)))

                        if trade_type == "BUY":
                            buy_filled_base += exec_base
                            buy_filled_quote += exec_quote
                        else:
                            sell_filled_base += exec_base
                            sell_filled_quote += exec_quote

                # Use order-level fees if available, otherwise fall back to executor-level
                fees = orders_fees if orders_fees > 0 else executor_fees

                # Add buy and sell fills separately
                if buy_filled_base > 0:
                    # Split fees proportionally between buy and sell by quote volume
                    total_quote = buy_filled_quote + sell_filled_quote
                    buy_fee_share = fees * (buy_filled_quote / total_quote) if total_quote > 0 else fees
                    position.add_fill("BUY", buy_filled_base, buy_filled_quote, executor_id, fees_quote=buy_fee_share)
                if sell_filled_base > 0:
                    total_quote = buy_filled_quote + sell_filled_quote
                    sell_fee_share = fees * (sell_filled_quote / total_quote) if total_quote > 0 else fees
                    position.add_fill("SELL", sell_filled_base, sell_filled_quote, executor_id, fees_quote=sell_fee_share)

                logger.info(
                    f"Aggregated executor {executor_id} to position {position_key}: "
                    f"buy={buy_filled_base} base, sell={sell_filled_base} base, fees={fees} quote"
                )

            elif filled_amount_base > 0:
                # For non-grid executors with a single side
                position.add_fill(side, filled_amount_base, filled_amount_quote, executor_id, fees_quote=executor_fees)
                logger.info(
                    f"Aggregated executor {executor_id} to position {position_key}: "
                    f"{side} {filled_amount_base} base @ {filled_amount_quote} quote"
                )
            else:
                logger.debug(f"Executor {executor_id} has no filled amounts to aggregate")

            # Persist position hold to the dedicated table
            await self._persist_position_hold(position)

        except Exception as e:
            logger.error(f"Error aggregating position for executor {executor_id}: {e}", exc_info=True)

    async def _persist_position_hold(self, position: PositionHold):
        """Persist a position hold to the dedicated position_holds table."""
        if not self.db_manager:
            return
        try:
            async with self.db_manager.get_session_context() as session:
                repo = ExecutorRepository(session)
                await repo.upsert_position_hold(
                    account_name=position.account_name,
                    connector_name=position.connector_name,
                    trading_pair=position.trading_pair,
                    controller_id=position.controller_id,
                    buy_amount_base=position.buy_amount_base,
                    buy_amount_quote=position.buy_amount_quote,
                    sell_amount_base=position.sell_amount_base,
                    sell_amount_quote=position.sell_amount_quote,
                    realized_pnl_quote=position.realized_pnl_quote,
                    cum_fees_quote=position.cum_fees_quote,
                    executor_ids=position.executor_ids,
                )
        except Exception as e:
            logger.error(f"Error persisting position hold: {e}", exc_info=True)

    def get_positions_held(
        self,
        account_name: Optional[str] = None,
        connector_name: Optional[str] = None,
        trading_pair: Optional[str] = None,
        controller_id: Optional[str] = None
    ) -> List[PositionHold]:
        """
        Get held positions with optional filtering.

        Args:
            account_name: Filter by account name
            connector_name: Filter by connector name
            trading_pair: Filter by trading pair
            controller_id: Filter by controller ID

        Returns:
            List of PositionHold objects matching the filters
        """
        positions = []

        for position in self._positions_held.values():
            # Apply filters
            if account_name and position.account_name != account_name:
                continue
            if connector_name and position.connector_name != connector_name:
                continue
            if trading_pair and position.trading_pair != trading_pair:
                continue
            if controller_id and position.controller_id != controller_id:
                continue

            # Only include positions with actual volume
            if position.buy_amount_base > 0 or position.sell_amount_base > 0:
                positions.append(position)

        return positions

    def get_position_held(
        self,
        account_name: str,
        connector_name: str,
        trading_pair: str,
        controller_id: str = "main"
    ) -> Optional[PositionHold]:
        """
        Get a specific held position.

        Args:
            account_name: Account name
            connector_name: Connector name
            trading_pair: Trading pair
            controller_id: Controller ID

        Returns:
            PositionHold or None if not found
        """
        position_key = self._get_position_key(account_name, connector_name, trading_pair, controller_id)
        return self._positions_held.get(position_key)

    async def clear_position_held(
        self,
        account_name: str,
        connector_name: str,
        trading_pair: str,
        controller_id: str = "main"
    ) -> bool:
        """
        Clear a specific held position (after manual close or full exit).

        Args:
            account_name: Account name
            connector_name: Connector name
            trading_pair: Trading pair
            controller_id: Controller ID

        Returns:
            True if cleared, False if not found
        """
        position_key = self._get_position_key(account_name, connector_name, trading_pair, controller_id)
        if position_key in self._positions_held:
            del self._positions_held[position_key]
            # Mark position hold as CLEARED in the dedicated table
            if self.db_manager:
                try:
                    async with self.db_manager.get_session_context() as session:
                        repo = ExecutorRepository(session)
                        cleared = await repo.clear_position_hold(
                            account_name=account_name,
                            connector_name=connector_name,
                            trading_pair=trading_pair,
                            controller_id=controller_id
                        )
                        logger.info(f"Cleared position hold record from database for {position_key}: {cleared}")
                except Exception as e:
                    logger.error(f"Failed to clear position hold from database: {e}", exc_info=True)
            logger.info(f"Cleared position hold for {position_key}")
            return True
        return False

    def get_positions_summary(self) -> Dict[str, Any]:
        """
        Get summary of all held positions.

        Returns:
            Dictionary with total positions, PnL, and position list
        """
        positions = self.get_positions_held()
        total_realized_pnl = sum(float(p.realized_pnl_quote) for p in positions)

        return {
            "total_positions": len(positions),
            "total_realized_pnl": total_realized_pnl,
            "positions": [
                {
                    "trading_pair": p.trading_pair,
                    "connector_name": p.connector_name,
                    "account_name": p.account_name,
                    "buy_amount_base": float(p.buy_amount_base),
                    "buy_amount_quote": float(p.buy_amount_quote),
                    "sell_amount_base": float(p.sell_amount_base),
                    "sell_amount_quote": float(p.sell_amount_quote),
                    "net_amount_base": float(p.net_amount_base),
                    "buy_breakeven_price": float(p.buy_breakeven_price) if p.buy_breakeven_price else None,
                    "sell_breakeven_price": float(p.sell_breakeven_price) if p.sell_breakeven_price else None,
                    "matched_amount_base": float(p.matched_amount_base),
                    "unmatched_amount_base": float(p.unmatched_amount_base),
                    "position_side": p.position_side,
                    "realized_pnl_quote": float(p.realized_pnl_quote),
                    "cum_fees_quote": float(p.cum_fees_quote),
                    "executor_count": len(p.executor_ids),
                    "executor_ids": p.executor_ids,
                    "last_updated": p.last_updated.isoformat() if p.last_updated else None
                }
                for p in positions
            ]
        }
