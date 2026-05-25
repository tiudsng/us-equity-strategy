---
name: us-equity-team
description: |
  美股多智能體團隊 + 金融級調度器。
  基於 agno Team (Apache-2.0) 建構的 Fundamental + Technical + Risk Agent 團隊，
  配合自研 financial_scheduler（exchange_calendars + NYSE calendar + EST時區）。
  用途：美股策略分析、多智能體協作、每週自動化報告。
trigger: "美股 Agent 團隊、agno team、金融調度、Fincept 研究"
version: "1.0.0"
category: investment
---

## 概述

```
us-equity-team/
├── agent_team.py              # agno Team — Leader + 3 agents
├── financial_scheduler.py     # 金融級調度器（自研，exchange_calendars）
└── references/
    └── fincept-research.md    # FinceptTerminal 研究（AGPL隔離策略）
```

**用途**：美股策略的 AI 團隊分析（Fundamental + Technical + Risk 協作）＋金融級排程（market_open/market_close/EST時區）。

---

## Agent Team 架構

```
Team Leader (Coordinator)
├── Fundamental Agent — 財報、DCF、P/E、估值
├── Technical Agent — 回測、RSI、Sharpe、Momentum
└── Risk Agent — 倉位限制、Drawdown、Observer Mode Gate
```

### 創建團隊

```python
from agent_team import create_us_equity_team, format_observer_report

team = create_us_equity_team(model="gpt-4o", api_key="sk-...")
response = team.run(f"Analyze AAPL for investment potential")
report = format_observer_report()
```

### CLI 用法

```bash
python3 agent_team.py --weekly-report   # 產生週報
python3 agent_team.py --stock AAPL       # 分析個股
```

---

## Financial Scheduler（金融級調度器）

### DSL 表達式

| 表達式 | 說明 | 範例 |
|--------|------|------|
| `market_open` | 下一個 NYSE 開盤（9:30 AM ET） | 每個交易日開盤時 |
| `market_close` | 下一個 NYSE 收盤（4:00 PM ET） | 每個交易日收盤時 |
| `daily HH:MM EST` | 每日（UTC） | `daily 09:30 EST` |
| `weekday HH:MM EST` | 每週一至五（UTC） | `weekday 16:00 EST` |
| `every Nm/h/d` | 每 N 分/小時/天 | `every 5m`, `every 2h` |
| `hourly` | 每小時（= every 1h） | 每小時 |

### 時區處理（修復 Fincept 的 bug）

```
Fincept bug: "weekday 16:00 EST" → UTC 16:00 = EST 11:00（美股未收盤！）
我們的修復: "weekday 16:00 EST" → UTC 20:00 (EST) / 19:00 (EDT) ✅
```

### 使用方式

```python
from financial_scheduler import (
    compute_next_run, is_market_open, is_trading_day,
    get_next_open, get_next_close, schedule_financial_task,
)

# 測試
now = datetime.utcnow()
for expr in ["market_open", "weekday 16:00 EST", "every 5m"]:
    print(f"{expr} → {compute_next_run(expr, now)}")
```

---

## 與 optimize_us_equity_v1 整合

```
每週六 Cron → optimize_us_equity_v1 → Phase 4: agent_team 分析
→ Phase 5: 回測驗證 → Phase 6: create_lesson → Phase 7: format_observer_report
```

---

## AGPL 隔離策略

FinceptTerminal 使用 AGPL-3.0，**不能直接複製原始碼**。

| 可用 | 不可用 |
|------|--------|
| agno（Apache-2.0） | Fork Fincept scheduler.py |
| 自建 financial_scheduler | 直接複製 Fincept 代碼 |
| HTTP API 呼叫 Fincept | 修改後閉源發布 |

---

## 安裝依賴

```bash
pip3 install agno exchange-calendars pandas pandas-market-calendars pyyaml
```

## 測試

```bash
python3 financial_scheduler.py  # 測試所有 DSL 表達式
```

## 相關文檔

- `references/fincept-research.md` — 詳細 Fincept 研究（Scheduler bug、agno、AGPL 分析）