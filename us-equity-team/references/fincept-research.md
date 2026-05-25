# FinceptTerminal Research — 2026-05-23

> Source: https://github.com/Fincept-Corporation/FinceptTerminal ⭐23,002  
> License: AGPL-3.0 (NOT directly usable in commercial projects)

## 🔑 Key Finding: Python Core, Not C++

Fincept's AI core and task scheduler are implemented in **pure Python** at:
```
fincept-qt/scripts/agents/finagent_core/
├── agentic/scheduler.py    ← 221 lines, clean DSL parser
├── super_agent.py          ← 815 lines, query routing
├── modules/team_module.py  ← 399 lines, multi-agent (uses agno)
└── agentic/runner.py       ← AgenticRunner with pause/resume
```

The C++ Qt layer (`fincept-qt/src/`) is primarily a UI shell — it delegates to Python via `PythonRunner` or `QProcess + stdin`.

**Implication**: For cloud-based agents (like Herman deployed on CVM), you do NOT need the C++ MCP Service. You need only the Python agent core, which can be called via HTTP/API.

## Scheduler DSL (Critical Flaw)

### Fincept's Implementation (AGPL — DO NOT COPY)

```python
_EXPR_MARKET_OPEN = re.compile(r"^market_open$", re.IGNORECASE)
_EXPR_MARKET_CLOSE = re.compile(r"^market_close$", re.IGNORECASE)
_EXPR_DAILY_EST = re.compile(r"^daily\s+(\d{1,2}):(\d{2})\s*(EST|ET|EDT)?$", re.IGNORECASE)
_EXPR_WEEKDAY_EST = re.compile(r"^weekday\s+(\d{1,2}):(\d{2})\s*(EST|ET|EDT)?$", re.IGNORECASE)
_EXPR_EVERY = re.compile(r"^every\s+(\d+)\s*([mhd])$", re.IGNORECASE)
_EXPR_HOURLY = re.compile(r"^hourly$", re.IGNORECASE)
```

### The Fatal Bug: No Timezone Handling

```python
def _next_at_time(after: datetime, hour: int, minute: int, weekdays_only: bool):
    candidate = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= after:
        candidate += timedelta(days=1)
    # weekdays_only uses Python's .weekday() — naive, no timezone
```

**Consequence**: `"weekday 16:00"` fires at 11:00 EST in winter, 12:00 EDT in summer — **1-2 hours BEFORE market close**.

### Our Fix: financial_scheduler.py (Apache-2.0 — clean)

```python
import exchange_calendars as ecals
cal = ecals.get_calendar("NYSE")
next_open_utc = cal.next_open(ts).replace(tzinfo=None)
```

**Tested results** (2026-05-23):
```
market_open               → 2026-05-26 13:30 UTC (Monday) ✅
market_close              → 2026-05-26 20:00 UTC ✅
daily 09:30 EST           → 2026-05-24 13:30 UTC ✅
weekday 09:30 EST         → 2026-05-25 13:30 UTC ✅
weekday 16:00 EST         → 2026-05-25 20:00 UTC ✅ (CORRECT!)
```

## Multi-Agent Team (agno — Apache-2.0, Safe)

```python
from agno.team import Team

team = Team(
    name="US Equity Analyst Team",
    mode="coordinate",
    leader_agent=analyst_agent,
    show_members_responses=True
)
team.add_agent(analyst, role="fundamental_analyst")
team.add_agent(momentum_agent, role="technical_analyst")
team.add_agent(risk_agent, role="risk_manager")
```

**License**: agno is Apache-2.0 — no AGPL contamination. Safe to use directly.

## AgenticRunner (Task Persistence)

```python
class AgenticRunner(ResumableTaskRunner):
    def start_task(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        task_id = self.state_mgr.create_task(query, config)
        self.state_mgr.update_status(task_id, "running")
        return self._run_loop(task_id, query, config, [], None)

    def resume_task(self, task_id: str) -> Dict[str, Any]:
        # Reads status from DB between steps
        # pause/cancel via DB flag polling, not process kill
```

**Key insight**: Task state in SQLite (`agent_tasks.db`), pause/cancel via `status` column.

## AGPL Isolation Strategy

```
┌─────────────────────────────────────┐
│  Herman Agent (自己的 Python)         │
│  ├── optimize_us_equity_v1          │
│  ├── hermes-memory                  │
│  └── financial_scheduler（自研）      │
└────────────┬────────────────────────┘
             │ HTTP API（干淨隔離）
             ▼
┌─────────────────────────────────────┐
│  FinceptTerminal（AGPL，外部）        │
│  └── 研究用途，不可直接整合            │
└─────────────────────────────────────┘
```