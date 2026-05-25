#!/usr/bin/env python3
"""
Wrapper: run Markov HMM daily scan
Triggered by Cron: 30 21 * * 1-5 (Mon-Fri 21:30 HKT = 13:30 UTC ≈ post-market settlement)
"""
import subprocess, sys, json
from datetime import datetime, timezone

SCRIPT = "/home/ubuntu/.hermes/skills/investment/us-equity-team/data_alert.py"
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]

result = subprocess.run(
    [sys.executable, SCRIPT,
     "--tickers", ",".join(TICKERS),
     "--json"],
    capture_output=True, text=True, timeout=120
)

output = result.stdout.strip()
print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] Markov Daily Scan completed")
print(output)

# Parse high_confidence_signals for summary
if output:
    try:
        data = json.loads(output)
        hc = data.get("high_confidence_signals", [])
        total = data.get("all_results_count", 0)
        print(f"\n📊 Summary: {total} tickers scanned, {len(hc)} HIGH CONFIDENCE signals")
        for s in hc:
            print(f"  {s['ticker']:6s} {s['direction']:6s} {s['signal_raw']:+.4f} ({s['position_size_pct']:.0f}%) {'✅ AGREE' if s.get('hmm_agreement') else '⚠️ DISAGREE'}")
    except Exception:
        pass

sys.exit(result.returncode)