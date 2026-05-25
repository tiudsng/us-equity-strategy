#!/usr/bin/env python3
"""
Data & Alert Module — 數據獲取 + Webhook 通知
串接 yfinance + markov_hmm_strategy + Telegram/Discord Webhook

agno Team 整合：
  Technical Agent → yfinance → markov_hmm_strategy.py → JSON → Webhook 通知
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional

import pandas as pd
import yfinance as yf

# ── Paths ──────────────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).parent
MARKOV_SCRIPT = SKILL_DIR / "markov_hmm_strategy.py"
LEDGER_PATH = Path("/home/ubuntu/.hermes/us_equity_strategy/trades_ledger.csv")
DATA_CACHE = Path("/home/ubuntu/.hermes/memory/yf_cache.db")

# ── Webhook Config ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8642765029:AAE3kn8_28mPOlWLC_4xfNs-RtQje9XCOm8"
TELEGRAM_CHAT_ID = "8217991576"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ── Default universe ────────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq 100 ETF
    "IWM",   # Russell 2000 ETF
    "DIA",   # Dow Jones ETF
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Google
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "TSLA",  # Tesla
]


# ── Data Fetching ─────────────────────────────────────────────────────────────

class YFDataResult(NamedTuple):
    ticker: str
    df: pd.DataFrame
    fetched_at: str
    rows: int


def fetch_ticker_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    force_refresh: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> YFDataResult:
    """
    Fetch daily OHLCV from yfinance.

    ⚠️ Critical design decision (per user feedback):
    - Default to "1y" daily bars (closed candles only)
    - Drop the last row if market is OPEN right now (avoid intraday noise)
    - Retry with backoff on rate limiting
    """
    import time

    cache_db = str(DATA_CACHE)
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    fetched_at = now_utc.strftime("%Y-%m-%d %H:%M UTC")

    # Check cache
    if not force_refresh:
        cached = _load_from_cache(ticker, period, interval)
        if cached is not None:
            return cached

    # Fetch fresh with retry
    last_error = None
    for attempt in range(max_retries):
        try:
            tick = yf.Ticker(ticker)
            df = tick.history(period=period, interval=interval, auto_adjust=True)

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Normalize index → date column
            df = df.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

            # ── Drop last row if market is currently open (avoid incomplete candle) ──
            last_date = pd.to_datetime(df["date"].iloc[-1])
            market_day = last_date.date()

            if market_day == now_utc.date() and now_utc.hour < 21:
                df = df.iloc[:-1]

            rows = len(df)
            result = YFDataResult(ticker=ticker, df=df, fetched_at=fetched_at, rows=rows)

            # Save to cache
            _save_to_cache(result)

            return result

        except Exception as e:
            last_error = e
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait = retry_delay * (2 ** attempt)
                print(f"⚠️ Rate limited, retrying in {wait:.0f}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    raise last_error


def _load_from_cache(ticker: str, period: str, interval: str) -> Optional[YFDataResult]:
    """Load from SQLite cache if fresh enough (max 1 hour)."""
    if not DATA_CACHE.exists():
        return None

    conn = sqlite3.connect(str(DATA_CACHE))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM yf_cache WHERE ticker=? AND period=? AND interval=? "
        "AND fetched_at > datetime('now', '-1 hour')",
        (ticker, period, interval),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    df = pd.read_json(row["df_json"])
    return YFDataResult(
        ticker=row["ticker"],
        df=df,
        fetched_at=row["fetched_at"],
        rows=row["rows"],
    )


def _save_to_cache(result: YFDataResult):
    """Persist fetched data to SQLite cache."""
    conn = sqlite3.connect(str(DATA_CACHE))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS yf_cache (
            ticker TEXT, period TEXT, interval TEXT,
            df_json TEXT, fetched_at TEXT, rows INTEGER,
            UNIQUE(ticker, period, interval)
        )
    """)
    conn.execute(
        "INSERT OR REPLACE INTO yf_cache VALUES (?, ?, ?, ?, ?, ?)",
        (
            result.ticker,
            "1y",
            "1d",
            result.df.to_json(),
            result.fetched_at,
            result.rows,
        ),
    )
    conn.commit()
    conn.close()


# ── Batch Fetch ───────────────────────────────────────────────────────────────

def batch_fetch(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
    max_workers: int = 3,
) -> List[YFDataResult]:
    """Fetch multiple tickers (sequential, respects rate limits)."""
    results = []
    for ticker in tickers:
        try:
            result = fetch_ticker_data(ticker, period, interval)
            results.append(result)
        except Exception as e:
            print(f"⚠️ Failed to fetch {ticker}: {e}")
    return results


# ── Markov Analysis ────────────────────────────────────────────────────────────

def run_markov_analysis(
    ticker: str,
    df: pd.DataFrame,
    lookback: int = 20,
    use_hmm: bool = True,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Run markov_hmm_strategy.py on a DataFrame and return parsed JSON.
    """
    import tempfile

    # Write temp CSV (date,close only)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name
        df[["date", "close"]].to_csv(f, index=False)

    cmd = [
        sys.executable, str(MARKOV_SCRIPT),
        "--ticker", ticker,
        "--csv", temp_path,
        "--lookback", str(lookback),
        "--json",
    ]
    if not use_hmm:
        cmd.append("--no-hmm")
    if as_of_date:
        cmd.extend(["--as-of-date", as_of_date])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {"error": result.stderr}

        report = json.loads(result.stdout)
        Path(temp_path).unlink()
        return report

    except Exception as e:
        return {"error": str(e)}


# ── Webhook Notifications ──────────────────────────────────────────────────────

def send_telegram(message: str, bot_token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID) -> dict:
    """Send Telegram message via bot API."""
    import urllib.request

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def send_discord(message: str, webhook_url: str = DISCORD_WEBHOOK_URL) -> dict:
    """Send Discord embed/webhook message."""
    import urllib.request

    if not webhook_url:
        return {"error": "DISCORD_WEBHOOK_URL not set"}

    payload = json.dumps({"content": message}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": resp.status}
    except Exception as e:
        return {"error": str(e)}


def format_signal_alert(report: dict) -> str:
    """Format a single ticker signal as Telegram/Discord markdown."""
    ticker = report["ticker"]
    signal = report["signal"]
    hmm = report.get("hmm_validation")
    agreement = hmm["agreement"] if hmm else True

    # Emoji per direction
    emoji = {"LONG": "🟢", "SHORT": "🔴", "NEUTRAL": "⚪"}.get(signal["direction"], "⚪")
    hmm_emoji = "✅" if agreement else "⚠️"

    # Confidence color
    conf_color = {"HIGH": "🔵", "MEDIUM": "🟡"}.get(signal["confidence"], "⚪")

    matrix = report["transition_matrix"]
    bear_pct = matrix["Bear"]["Bear"] * 100

    return f"""
{emoji} *{ticker}* — {signal["direction"]}
{conf_color} Confidence: {signal["confidence"]}
📊 Signal: `{signal["signal_raw"]:+.4f}` | Pos Size: `{signal["position_size_pct"]:.0f}%`
🐻 P(Bear): `{signal["p_bear_pct"]:.1f}%` | 🐂 P(Bull): `{signal["p_bull_pct"]:.1f}%`
🔗 HMM Agreement: {hmm_emoji} `{agreement}`
📈 Regime: **{report['current_state']}** | {hmm_emoji} HMM: `{hmm['hmm_predicted_state'] if hmm else 'N/A'}`
🔒 Observer Mode: {'Yes' if report['is_observer_mode'] else 'No'}
""".strip()


# ── Full Scan Pipeline ────────────────────────────────────────────────────────

class ScanResult(NamedTuple):
    scan_time: str
    total_tickers: int
    high_confidence: List[dict]
    all_results: List[dict]
    telegram_sent: Optional[dict]
    discord_sent: Optional[dict]


def run_daily_scan(
    tickers: List[str] = DEFAULT_TICKERS,
    lookback: int = 20,
    use_hmm: bool = True,
    as_of_date: Optional[str] = None,
    notify_telegram: bool = True,
    notify_discord: bool = False,
) -> ScanResult:
    """
    Full pipeline:
      1. Batch fetch yfinance data
      2. Run markov_hmm_strategy on each ticker
      3. Filter HIGH confidence + Agreement=True signals
      4. Send Telegram/Discord notifications
    """
    print(f"📡 Fetching {len(tickers)} tickers...")
    data_results = batch_fetch(tickers)

    print(f"🧮 Running Markov HMM analysis...")
    all_results = []
    high_confidence = []

    for data in data_results:
        report = run_markov_analysis(
            ticker=data.ticker,
            df=data.df,
            lookback=lookback,
            use_hmm=use_hmm,
            as_of_date=as_of_date,
        )
        if "error" not in report:
            all_results.append(report)

            # Filter: HIGH confidence + not observer mode
            if report["signal"]["confidence"] == "HIGH" and not report["is_observer_mode"]:
                high_confidence.append({
                    "ticker": data.ticker,
                    "direction": report["signal"]["direction"],
                    "signal_raw": report["signal"]["signal_raw"],
                    "position_size_pct": report["signal"]["position_size_pct"],
                    "hmm_agreement": report["hmm_validation"]["agreement"] if report.get("hmm_validation") else True,
                })

    # ── Send notifications ───────────────────────────────────────────────────
    tg_result = None
    dc_result = None

    if high_confidence:
        print(f"🚨 {len(high_confidence)} HIGH CONFIDENCE signals — sending alerts...")

        if notify_telegram:
            # Compose batch alert
            header = f"📊 *Markov HMM Daily Scan*\n🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            signals_block = "\n".join(
                f"- `{s['ticker']:5s}` {s['direction']} `{s['signal_raw']:+.4f}` ({s['position_size_pct']:.0f}%)"
                for s in high_confidence
            )
            message = header + signals_block
            tg_result = send_telegram(message)

        if notify_discord:
            dc_msg = f"📊 **Markov HMM Daily Scan** `{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}`\n"
            dc_msg += "\n".join(
                f"- `{s['ticker']}` {s['direction']} `{s['signal_raw']:+.4f}` ({s['position_size_pct']:.0f}%)"
                for s in high_confidence
            )
            dc_result = send_discord(dc_msg)

    return ScanResult(
        scan_time=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        total_tickers=len(data_results),
        high_confidence=high_confidence,
        all_results=all_results,
        telegram_sent=tg_result,
        discord_sent=dc_result,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Markov HMM Daily Scanner")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--as-of-date", type=str, help="Walk-forward cutoff (YYYY-MM-DD)")
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--json", action="store_true", help="Full JSON output")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else DEFAULT_TICKERS

    result = run_daily_scan(
        tickers=tickers,
        lookback=args.lookback,
        use_hmm=(not args.no_hmm),
        as_of_date=args.as_of_date,
        notify_telegram=(not args.no_telegram),
        notify_discord=(not args.no_discord),
    )

    if args.json:
        print(json.dumps({
            "scan_time": result.scan_time,
            "total_tickers": result.total_tickers,
            "high_confidence_signals": result.high_confidence,
            "all_results_count": len(result.all_results),
        }, indent=2))
    else:
        print(f"\n📊 Scan complete — {result.total_tickers} tickers, {len(result.high_confidence)} HIGH CONF signals")
        for s in result.high_confidence:
            print(f"  {s['ticker']:6s} {s['direction']:6s} {s['signal_raw']:+.4f}  ({s['position_size_pct']:.0f}%)")


if __name__ == "__main__":
    main()