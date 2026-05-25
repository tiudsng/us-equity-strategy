"""
Financial Scheduler — Built from scratch (no AGPL contamination)

Replaces Fincept's scheduler with proper US market calendar support:
- exchange_calendars for NYSE/NASDAQ trading days + US federal holidays
- America/New_York timezone for US market hours
- DSL: "market_open", "market_close", "09:30 EST", "16:00 EST"

Usage:
  from financial_scheduler import schedule_financial_task, get_next_market_open, is_market_open

  schedule_financial_task("US_Scan", "market_open", callback)    # Every market open
  schedule_financial_task("US_Close", "market_close", callback)  # Every market close
  schedule_financial_task("US_Week", "weekday 09:30 EST", callback)  # Every trading day 9:30 AM ET
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Market Calendar ──────────────────────────────────────────────────────────
try:
    import exchange_calendars as ecals
    import pytz
    EXCHANGE_CALENDARS_AVAILABLE = True
except ImportError:
    EXCHANGE_CALENDARS_AVAILABLE = False
    pytz = None


# ── Timezone helpers ──────────────────────────────────────────────────────────
NY_TZ = None
if EXCHANGE_CALENDARS_AVAILABLE:
    try:
        NY_TZ = ecals.ZoneInfo("America/New_York")
    except Exception:
        import zoneinfo
        NY_TZ = zoneinfo.ZoneInfo("America/New_York")


def now_ny() -> datetime:
    """Current time in New York (ET/EDT)"""
    if NY_TZ:
        return datetime.now(NY_TZ)
    return datetime.utcnow()


def to_utc(dt: datetime) -> datetime:
    """Convert naive or NY-aware datetime to UTC"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=NY_TZ)
    return dt.astimezone(timezone.utc)


# ── NYSE Calendar ──────────────────────────────────────────────────────────────
_NYSE_CAL: Optional[Any] = None


def get_nyse_calendar():
    """Lazily create NYSE calendar singleton"""
    global _NYSE_CAL
    if _NYSE_CAL is None and EXCHANGE_CALENDARS_AVAILABLE:
        _NYSE_CAL = ecals.get_calendar("NYSE")
    return _NYSE_CAL


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Check if NYSE is currently open"""
    cal = get_nyse_calendar()
    if cal is None:
        return True  # Fallback if exchange_calendars unavailable
    dt = dt or now_ny()
    import pandas as pd
    ts = pd.Timestamp(dt)
    return cal.is_open_at_time(ts)


def is_trading_day(dt: Optional[datetime] = None) -> bool:
    """Check if it's a US trading day (not weekend, not holiday)"""
    cal = get_nyse_calendar()
    if cal is None:
        return dt.weekday() < 5 if dt else True
    dt = dt or now_ny()
    import pandas as pd
    # is_session requires timezone-naive date (just date part)
    ts = pd.Timestamp(dt.date())
    return cal.is_session(ts)


def get_next_open(dt: Optional[datetime] = None) -> datetime:
    """Next market open (9:30 AM ET) as UTC naive datetime"""
    cal = get_nyse_calendar()
    if cal is None:
        return now_ny().replace(hour=14, minute=30, second=0, microsecond=0)
    dt = dt or now_ny()
    import pandas as pd
    ts = pd.Timestamp(dt)
    next_open = cal.next_open(ts)
    # Return naive UTC for storage consistency
    return next_open.replace(tzinfo=None)


def get_next_close(dt: Optional[datetime] = None) -> datetime:
    """Next market close (4:00 PM ET) as UTC naive datetime"""
    cal = get_nyse_calendar()
    if cal is None:
        return now_ny().replace(hour=21, minute=0, second=0, microsecond=0)
    dt = dt or now_ny()
    import pandas as pd
    ts = pd.Timestamp(dt)
    next_close = cal.next_close(ts)
    return next_close.replace(tzinfo=None)


# ── DSL Parser ─────────────────────────────────────────────────────────────────
# Patterns:
#   market_open        → next NYSE open
#   market_close       → next NYSE close
#   daily HH:MM EST    → daily at HH:MM NY time
#   weekday HH:MM EST  → Mon-Fri at HH:MM NY time
#   every N m/h/d      → every N minutes/hours/days
#   hourly             → every 1 hour

_EXPR_MARKET_OPEN = re.compile(r"^market_open$", re.IGNORECASE)
_EXPR_MARKET_CLOSE = re.compile(r"^market_close$", re.IGNORECASE)
_EXPR_DAILY_EST = re.compile(r"^daily\s+(\d{1,2}):(\d{2})\s*(EST|ET|EDT)?$", re.IGNORECASE)
_EXPR_WEEKDAY_EST = re.compile(r"^weekday\s+(\d{1,2}):(\d{2})\s*(EST|ET|EDT)?$", re.IGNORECASE)
_EXPR_EVERY = re.compile(r"^every\s+(\d+)\s*([mhd])$", re.IGNORECASE)
_EXPR_HOURLY = re.compile(r"^hourly$", re.IGNORECASE)


def compute_next_run(expr: str, after: datetime) -> datetime:
    """Compute next firing time strictly after `after` (UTC naive datetime)"""
    expr = expr.strip()

    # Market events
    m = _EXPR_MARKET_OPEN.match(expr)
    if m:
        # Convert `after` to NY time for calendar lookup
        after_ny = after
        if after.tzinfo is None and NY_TZ:
            after_ny = after.replace(tzinfo=NY_TZ)
        next_open_utc = get_next_open(after_ny)
        if next_open_utc <= after:
            # Already past today's open, get tomorrow's
            next_open_utc = get_next_open(after_ny + timedelta(days=1))
        return next_open_utc

    m = _EXPR_MARKET_CLOSE.match(expr)
    if m:
        after_ny = after
        if after.tzinfo is None and NY_TZ:
            after_ny = after.replace(tzinfo=NY_TZ)
        next_close_utc = get_next_close(after_ny)
        if next_close_utc <= after:
            next_close_utc = get_next_close(after_ny + timedelta(days=1))
        return next_close_utc

    # Daily at HH:MM NY time
    m = _EXPR_DAILY_EST.match(expr)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return _next_at_ny_time(after, h, mn, weekdays_only=False)

    # Weekday at HH:MM NY time
    m = _EXPR_WEEKDAY_EST.match(expr)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return _next_at_ny_time(after, h, mn, weekdays_only=True)

    # Every N minutes/hours/days
    m = _EXPR_EVERY.match(expr)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if n <= 0:
            raise ValueError(f"invalid period: {expr}")
        delta = {"m": timedelta(minutes=n), "h": timedelta(hours=n), "d": timedelta(days=n)}[unit]
        return after + delta

    # Hourly
    if _EXPR_HOURLY.match(expr):
        return after + timedelta(hours=1)

    raise ValueError(f"unrecognised schedule expression: {expr!r}")


def _next_at_ny_time(after_utc: datetime, hour: int, minute: int, weekdays_only: bool) -> datetime:
    """Compute next occurrence of hour:minute NY time after after_utc"""
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"invalid time {hour}:{minute}")

    # Convert after to NY
    if after_utc.tzinfo is None and NY_TZ:
        after_ny = after_utc.replace(tzinfo=NY_TZ)
    else:
        after_ny = after_utc

    # Find next candidate in NY time
    candidate_ny = after_ny.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate_ny <= after_ny:
        candidate_ny += timedelta(days=1)

    if weekdays_only:
        while candidate_ny.weekday() >= 5:  # Saturday=5, Sunday=6
            candidate_ny += timedelta(days=1)

    # Convert back to UTC naive for storage
    return candidate_ny.astimezone(timezone.utc).replace(tzinfo=None)


# ── Persistent Store ──────────────────────────────────────────────────────────

_DEFAULT_DB = str(Path("/home/ubuntu/.hermes/memory/schedules.db"))

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS financial_schedules (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    query         TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    schedule_expr TEXT NOT NULL,
    enabled       INTEGER NOT NULL DEFAULT 1,
    last_run_at   TEXT,
    next_run_at   TEXT NOT NULL,
    created_at    TEXT NOT NULL
)
"""
_CREATE_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_sched_due "
    "ON financial_schedules(enabled, next_run_at)"
)


class FinancialScheduleStore:
    """SQLite-backed financial schedule store with US market calendar support"""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    def create(
        self,
        name: str,
        query: str,
        schedule_expr: str,
        config: Optional[Dict[str, Any]] = None,
        start_now: bool = False,
    ) -> str:
        """Register a new financial schedule. Returns id."""
        sid = str(uuid.uuid4())
        now = datetime.utcnow()
        next_run = now if start_now else compute_next_run(schedule_expr, now)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO financial_schedules "
                "(id, name, query, config_json, schedule_expr, enabled, next_run_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (sid, name.strip(), query, json.dumps(config or {}),
                 schedule_expr.strip(), next_run.isoformat(), now.isoformat()),
            )
        return sid

    def list(self, enabled_only: bool = False, limit: int = 200) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if enabled_only:
                rows = conn.execute(
                    "SELECT * FROM financial_schedules WHERE enabled = 1 "
                    "ORDER BY next_run_at ASC LIMIT ?", (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM financial_schedules ORDER BY created_at DESC LIMIT ?", (limit,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, sid: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM financial_schedules WHERE id = ?", (sid,))

    def set_enabled(self, sid: str, enabled: bool) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE financial_schedules SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, sid),
            )

    def take_due(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Atomically claim all due schedules and advance their next_run_at"""
        now = now or datetime.utcnow()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM financial_schedules "
                "WHERE enabled = 1 AND next_run_at <= ?",
                (now.isoformat(),),
            ).fetchall()
            due: List[Dict[str, Any]] = [self._row_to_dict(r) for r in rows]
            for d in due:
                nxt = compute_next_run(d["schedule_expr"], now)
                conn.execute(
                    "UPDATE financial_schedules SET last_run_at = ?, next_run_at = ? WHERE id = ?",
                    (now.isoformat(), nxt.isoformat(), d["id"]),
                )
                d["last_run_at"] = now.isoformat()
                d["next_run_at"] = nxt.isoformat()
        return due

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        cfg_raw = d.pop("config_json", None)
        d["config"] = json.loads(cfg_raw) if cfg_raw else {}
        d["enabled"] = bool(d.get("enabled", 0))
        return d


# ── Convenience functions ─────────────────────────────────────────────────────

_store = FinancialScheduleStore()


def schedule_financial_task(
    name: str,
    schedule_expr: str,
    query: str = "",
    config: Optional[Dict[str, Any]] = None,
    start_now: bool = False,
) -> str:
    """Create a financial schedule (alias for _store.create)"""
    return _store.create(name, query, schedule_expr, config, start_now)


def list_financial_schedules(enabled_only: bool = False) -> List[Dict[str, Any]]:
    return _store.list(enabled_only)


def get_due_schedules(now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    return _store.take_due(now)


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Financial Scheduler Test ===")
    print(f"Current NY time: {now_ny().strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"NYSE Open right now: {is_market_open()}")
    print(f"Is trading day: {is_trading_day()}")
    print(f"Next market open (UTC): {get_next_open()}")
    print(f"Next market close (UTC): {get_next_close()}")

    now = datetime.utcnow()
    test_exprs = [
        "market_open",
        "market_close",
        "daily 09:30 EST",
        "weekday 09:30 EST",
        "weekday 16:00 EST",
        "every 5m",
        "hourly",
    ]
    print("\n=== Schedule Expression Tests ===")
    for expr in test_exprs:
        try:
            next_run = compute_next_run(expr, now)
            print(f"  {expr:25s} → {next_run.strftime('%Y-%m-%d %H:%M UTC')}")
        except ValueError as e:
            print(f"  {expr:25s} → ERROR: {e}")