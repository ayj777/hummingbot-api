"""Refuse to start against a hummingbot core that predates what this API reads.

``environment.yml`` pins ``hummingbot`` to ayj777/hummingbot commit
``1e813e774ae0abbd8a84c95d78555db0162f7880`` (PR #10 head, not merged).
That head adds ``set_entry_paused``. Move the pin to the squash commit on
master after that PR merges. Those
changes are not in a PyPI release yet. An image built from an older core
comes up without the fields this API reads.

Nothing complains when that happens. ``_persist_executor_completed`` reads the executor's
figures inside a broad ``except Exception`` that logs at DEBUG and substitutes zeros, so a
missing attribute does not fail — it books every executor's pnl, fees, filled amount and
volume as 0 and says nothing. A whole deployment of silently zeroed accounting is a worse
outcome than not booting, so this check turns that into a startup error naming the field.

Drop a name from here once the release carrying it is the oldest one this API supports.
"""
from typing import List, Tuple

# (import path, attribute, what it is for) — each one a field this API reads off the
# core and cannot substitute.
REQUIRED_CORE_SURFACE: List[Tuple[str, str, str]] = [
    (
        "hummingbot.strategy_v2.models.executors:CloseType",
        "POSITION_CLOSED_EXTERNALLY",
        "an external full close, stored as close type 11",
    ),
    (
        "hummingbot.strategy_v2.executors.range_iceberg_executor.data_types:RangeIcebergExecutorConfig",
        "group_id",
        "optional label on a range iceberg that shares a side",
    ),
    (
        "hummingbot.strategy_v2.executors.range_iceberg_executor.data_types:RangeIcebergExecutorConfig",
        "sibling_base",
        "live size a tier must leave for the other tiers on its side",
    ),
]


def _resolve(path: str):
    module_path, _, name = path.partition(":")
    module = __import__(module_path, fromlist=[name])
    return getattr(module, name)


def require_core_surface() -> None:
    """Raise if the installed hummingbot is missing anything this API reads."""
    missing = []
    for path, attribute, purpose in REQUIRED_CORE_SURFACE:
        try:
            owner = _resolve(path)
        except (ImportError, AttributeError) as e:
            missing.append(f"  - {path} could not be imported ({e})")
            continue
        fields = getattr(owner, "model_fields", None)
        present = attribute in fields if fields is not None else hasattr(owner, attribute)
        if not present:
            missing.append(f"  - {path}.{attribute} — {purpose}")

    if missing:
        import hummingbot

        raise RuntimeError(
            "The installed hummingbot core is older than this API requires. Missing:\n"
            + "\n".join(missing)
            + f"\n\nInstalled hummingbot: {getattr(hummingbot, '__version__', 'unknown')}. "
            "Install a core that carries these fields (see environment.yml) and rebuild. "
            "Starting anyway would record every executor's pnl, fees and volume as zero."
        )
