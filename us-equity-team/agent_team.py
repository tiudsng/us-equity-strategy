"""
US Equity Agent Team — Built on agno Team (Apache-2.0, no AGPL contamination)

Multi-agent team for US equity strategy:
  - Leader:      Coordinates debate, synthesizes decision
  - Fundamental: Analyzes earnings, valuation, DCF, P/E
  - Technical:    Runs backtests, RSI, Sharpe, momentum
  - Risk:        Enforces position limits, observer-mode gate

Usage:
  python3 agent_team.py                    # Interactive mode
  python3 agent_team.py --weekly-report    # Generate weekly report
"""

from __future__ import annotations

import json
import os
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ── agno imports ─────────────────────────────────────────────────────────────
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.telegram_tools import TelegramTools

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from skills.investment.optimize_us_equity_v1.optimize_us_equity_with_memory import (
    optimize_us_equity_observer_mode,
    load_memory_stats,
)

# ── Config paths ──────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).parent
CONFIG_PATH = Path("/home/ubuntu/.hermes/us_equity_strategy/us_stock_strategy.yaml")
GOALS_PATH = Path("/home/ubuntu/.hermes/us_equity_strategy/strategy_goals.json")
LEDGER_PATH = Path("/home/ubuntu/.hermes/us_equity_strategy/trades_ledger.csv")
MEMORY_DIR = Path("/home/ubuntu/.hermes/memory")

TEAM_MODEL = os.environ.get("AGNO_MODEL", "gpt-4o")
TEAM_API_KEY = os.environ.get("OPENAI_API_KEY", "")

TELEGRAM_BOT_TOKEN = "8642765029:AAE3kn8_28mPOlWLC_4xfNs-RtQje9XCOm8"
TELEGRAM_CHAT_ID = "8217991576"


def load_strategy_config() -> dict:
    """Load us_stock_strategy.yaml"""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_goals() -> dict:
    """Load strategy_goals.json"""
    if not GOALS_PATH.exists():
        return {}
    with open(GOALS_PATH) as f:
        return json.load(f)


def format_observer_report() -> str:
    """Build observer-mode status report from memory stats + config"""
    cfg = load_strategy_config()
    goals = load_goals()
    mem_stats = load_memory_stats(MEMORY_DIR)
    params = cfg.get("learned_parameters", {})

    target_return = goals.get("strategy_goals", {}).get("success_metrics", {}).get("target_30day_return_pct", "N/A")
    min_sharpe = goals.get("strategy_goals", {}).get("success_metrics", {}).get("minimum_sharpe_ratio", "N/A")
    win_rate = goals.get("strategy_goals", {}).get("success_metrics", {}).get("win_rate_threshold_pct", "N/A")
    max_dd = goals.get("strategy_goals", {}).get("failure_conditions", {}).get("max_drawdown_pct", "N/A")
    consec_loss = goals.get("strategy_goals", {}).get("failure_conditions", {}).get("consecutive_losses_limit", "N/A")

    return f"""
## 📊 US Equity Strategy — Weekly Observer Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

### 🎯 Current Learned Parameters
| Parameter | Value |
|-----------|-------|
| RSI Period | {params.get('rsi_period', 'N/A')} |
| RSI Oversold | {params.get('rsi_oversold_threshold', 'N/A')} |
| RSI Overbought | {params.get('rsi_overbought_threshold', 'N/A')} |
| Trailing Stop | {params.get('trailing_stop_loss_pct', 'N/A')}% |
| Take Profit | {params.get('take_profit_pct', 'N/A')}% |
| Universe | {params.get('universe_filter', 'N/A')} |

### 📈 Success Thresholds
- Target 30-day Return: **{target_return}%**
- Minimum Sharpe Ratio: **{min_sharpe}**
- Win Rate Threshold: **{win_rate}%**

### 🚨 Failure Triggers
- Max Drawdown: **{max_dd}%**
- Consecutive Losses Limit: **{consec_loss}**

### 🧠 Memory System Stats
- Observations: {mem_stats.get('observations', 'N/A')}
- Memories: {mem_stats.get('memories', 'N/A')}
- Lessons: {mem_stats.get('lessons', 'N/A')}
- Sessions: {mem_stats.get('sessions', 'N/A')}

### 📁 Observer Mode
Team is in **observer mode** — no parameter changes until 30+ trades in ledger.
Current trades: {cfg.get('_trade_count', 'N/A')}
"""


# ── Agent Definitions ─────────────────────────────────────────────────────────

def create_fundamental_agent(model: str, api_key: str) -> Agent:
    """Fundamental Analysis Agent — earnings, DCF, P/E, valuation"""
    cfg = load_strategy_config()
    params = cfg.get("learned_parameters", {})

    return Agent(
        name="Fundamental Analyst",
        role="Fundamental Analysis Agent — specializes in earnings, DCF, P/E, book value, and intrinsic value calculations",
        model=OpenAIChat(id=model, api_key=api_key) if api_key else None,
        instructions=f"""You are a fundamental analysis agent for US equities.

Your responsibilities:
1. Analyze company financials (earnings, revenue growth, margins)
2. Run DCF (Discounted Cash Flow) valuation models
3. Calculate P/E, P/B, EV/EBITDA ratios
4. Assess competitive moat and industry position
5. Identify value traps vs. genuine opportunities

Current strategy parameters (DO NOT change these — they are managed by the Optimizer):
- RSI Period: {params.get('rsi_period', 14)}
- Universe: {params.get('universe_filter', 'SPY_COMPONENTS')}

When asked to analyze a stock:
1. Fetch latest earnings data
2. Calculate key fundamental metrics
3. Compare to sector averages
4. Provide a BUY/HOLD/SELL recommendation with thesis

Respond with structured analysis. Be concise but thorough.""",
        add_telegram_tool=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
    )


def create_technical_agent(model: str, api_key: str) -> Agent:
    """Technical Analysis Agent — backtests, RSI, Sharpe, momentum"""
    cfg = load_strategy_config()
    params = cfg.get("learned_parameters", {})
    goals = load_goals()

    return Agent(
        name="Technical Analyst",
        role="Technical Analysis Agent — runs backtests, computes RSI, Sharpe, momentum, volatility",
        model=OpenAIChat(id=model, api_key=api_key) if api_key else None,
        instructions=f"""You are a technical analysis agent for US equities.

Your responsibilities:
1. Run backtests on historical price data
2. Calculate RSI, MACD, Bollinger Bands
3. Compute Sharpe Ratio, Sortino Ratio, max drawdown
4. Identify momentum signals and trend reversals
5. Validate strategy parameters against historical performance

Current strategy parameters (DO NOT change these):
- RSI Period: {params.get('rsi_period', 14)}
- RSI Oversold: {params.get('rsi_oversold_threshold', 30)}
- RSI Overbought: {params.get('rsi_overbought_threshold', 70)}
- Trailing Stop: {params.get('trailing_stop_loss_pct', 5.0)}%
- Take Profit: {params.get('take_profit_pct', 15.0)}%

Success metrics from strategy_goals.json:
- Minimum Sharpe Ratio: {goals.get('strategy_goals', {}).get('success_metrics', {}).get('minimum_sharpe_ratio', 1.2)}
- Win Rate Threshold: {goals.get('strategy_goals', {}).get('success_metrics', {}).get('win_rate_threshold_pct', 55)}%
- Target 30-day Return: {goals.get('strategy_goals', {}).get('success_metrics', {}).get('target_30day_return_pct', 3.5)}%

When asked to analyze:
1. Fetch historical price data
2. Run the strategy simulation
3. Compute all metrics vs. thresholds
4. Report PASS/FAIL for each metric

Respond with structured backtest results. Be precise about numbers.""",
        add_telegram_tool=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
    )


def create_risk_agent(model: str, api_key: str) -> Agent:
    """Risk Management Agent — position limits, drawdown, compliance"""
    cfg = load_strategy_config()
    constraints = cfg.get("portfolio_constraints", {})
    goals = load_goals()

    return Agent(
        name="Risk Manager",
        role="Risk Management Agent — enforces position limits, drawdown bounds, and regulatory compliance",
        model=OpenAIChat(id=model, api_key=api_key) if api_key else None,
        instructions=f"""You are a risk management agent for US equities.

Your responsibilities:
1. Enforce position limits (max {constraints.get('max_open_positions', 10)} open positions)
2. Enforce single-stock capital cap ({constraints.get('max_capital_per_stock_pct', 10)}% max per stock)
3. Monitor cash reserve requirements ({constraints.get('cash_reserve_pct', 5)}% minimum)
4. Validate slippage tolerance ({constraints.get('slippage_tolerance_pct', 0.05)}%)
5. Check drawdown limits — FAIL if drawdown > {goals.get('strategy_goals', {}).get('failure_conditions', {}).get('max_drawdown_pct', 8)}%
6. Count consecutive losses — FAIL if > {goals.get('strategy_goals', {}).get('failure_conditions', {}).get('consecutive_losses_limit', 5)} losses

Failure conditions from strategy_goals.json:
- Max Drawdown: {goals.get('strategy_goals', {}).get('failure_conditions', {}).get('max_drawdown_pct', 8)}%
- Consecutive Losses: {goals.get('strategy_goals', {}).get('failure_conditions', {}).get('consecutive_losses_limit', 5)}
- Min Sharpe: {goals.get('strategy_goals', {}).get('failure_conditions', {}).get('unacceptable_sharpe_ratio', 0.8)}

IMPORTANT — Observer Mode Gate:
- System is in OBSERVER MODE until 30+ trades in ledger
- In observer mode: provide analysis but do NOT recommend trades
- After 30+ trades: full trading allowed

When asked to review:
1. Load current portfolio positions
2. Check all constraints against current state
3. Report COMPLIANT/NON-COMPLIANT per constraint
4. If any failure condition met, report TRIGGERED and reason

Respond with structured risk report.""",
        add_telegram_tool=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
    )


def create_leader_agent(model: str, api_key: str) -> Agent:
    """Team Leader — coordinates debate, synthesizes final decision"""
    return Agent(
        name="Team Leader",
        role="Team Leader — coordinates Fundamental, Technical, and Risk agents, synthesizes debate into final decision",
        model=OpenAIChat(id=model, api_key=api_key) if api_key else None,
        instructions="""You are the team leader for US Equity Strategy.

Your team consists of:
- Fundamental Analyst: evaluates earnings, DCF, P/E, intrinsic value
- Technical Analyst: runs backtests, RSI, momentum, Sharpe ratio
- Risk Manager: enforces position limits, drawdown, compliance

Coordination mode: coordinate (all agents work together)

Your job:
1. When given a stock query, distribute work to all 3 agents
2. Collect their analysis
3. Synthesize into a final recommendation: BUY / HOLD / SELL / SKIP
4. Always consider observer mode status (no trades until 30+ in ledger)

Final output format:
## Decision: [BUY/HOLD/SELL/SKIP]
## Confidence: [HIGH/MEDIUM/LOW]
## Reasoning: [2-3 sentence summary]
## Agent Votes:
  - Fundamental: [BUY/HOLD/SELL]
  - Technical: [BUY/HOLD/SELL]  
  - Risk: [BUY/HOLD/SELL/COMPLIANT_VIOLATION]

Be decisive. Do not hedge excessively. Trust the numbers.""",
        add_telegram_tool=True,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        team_members=[
            create_fundamental_agent(model, api_key),
            create_technical_agent(model, api_key),
            create_risk_agent(model, api_key),
        ],
    )


def create_us_equity_team(model: str = TEAM_MODEL, api_key: str = TEAM_API_KEY) -> Team:
    """Create the full US Equity Agent Team"""
    leader = create_leader_agent(model, api_key)

    return Team(
        name="US Equity Strategy Team",
        mode="coordinate",  # Agents collaborate, leader synthesizes
        model=OpenAIChat(id=model, api_key=api_key) if api_key else None,
        members=[leader],
        show_tool_calls=True,
        markdown=True,
        success_criteria="Team reaches consensus on BUY/HOLD/SELL decision",
    )


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="US Equity Agent Team")
    parser.add_argument("--weekly-report", action="store_true", help="Generate weekly observer report")
    parser.add_argument("--stock", type=str, help="Analyze a specific stock symbol")
    parser.add_argument("--model", type=str, default=TEAM_MODEL, help="Model to use")
    args = parser.parse_args()

    if args.weekly_report:
        report = format_observer_report()
        print(report)
        # Send to Telegram
        try:
            from telegram import Bot
            import asyncio
            bot = Bot(token=TELEGRAM_BOT_TOKEN)
            asyncio.run(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"```\n{report}\n```",
                parse_mode="MarkdownV2"
            ))
            print("\n✅ Report sent to Telegram")
        except Exception as e:
            print(f"\n⚠️ Telegram send failed: {e}")
        return

    if args.stock:
        team = create_us_equity_team(model=args.model)
        print(f"\n🔍 Analyzing {args.stock}...")
        response = team.run(
            f"Analyze {args.stock} for potential investment. "
            f"Consider fundamental (earnings, DCF), technical (RSI, momentum), "
            f"and risk (position limits, drawdown) perspectives. "
            f"Provide a final BUY/HOLD/SELL recommendation."
        )
        print(response.content)
        return

    # Default: print observer report
    print(format_observer_report())
    print("\nUsage:")
    print("  python3 agent_team.py --weekly-report   # Send weekly report to Telegram")
    print("  python3 agent_team.py --stock AAPL       # Analyze a specific stock")


if __name__ == "__main__":
    main()