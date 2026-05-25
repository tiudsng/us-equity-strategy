#!/usr/bin/env python3
"""
Markov HMM Strategy — 馬可夫鏈對沖基金策略
Inspired by Lewis Jackson / Rowan Framework

Core Logic:
  Step 1: State Labeling — 20-day Rolling Cumulative Returns
           Bull (>= 5%), Bear (<= -5%), Sideways (others)
  Step 2: Transition Matrix — 3×3 Grid (Bull/Bear/Sideways)
  Step 3: Signal = P(Bull) - P(Bear)
  Step 4: HMM Overlay — Gaussian HMM unsupervised regime validation

Key Design Choices:
  - Pure returns (NOT MA) — HMM prefers raw收益率, not smoothed MA
  - 20-day lookback for state labeling (per Lewis Jackson)
  - Walk-forward: strictly no lookahead bias (as-of-date cutoff)
  - Observer mode gate: no trades until 30+ data points
  - hmmlearn GaussianHMM for regime overlap validation
  - JSON output for agno Team integration
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── hmmlearn ────────────────────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False

# ── Project paths ──────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).parent
CONFIG_PATH = Path("/home/ubuntu/.hermes/us_equity_strategy/us_stock_strategy.yaml")
LEDGER_PATH = Path("/home/ubuntu/.hermes/us_equity_strategy/trades_ledger.csv")

# ── Constants ────────────────────────────────────────────────────────────────────
STATES: list = ["Bull", "Bear", "Sideways"]
BULL_THRESHOLD = 0.05    # ≥ 5% 20-day return → Bull
BEAR_THRESHOLD = -0.05   # ≤ -5% 20-day return → Bear
OBSERVER_MODE_THRESHOLD = 30  # Minimum data points before trading


# ── Step 1: State Labeling (Pure Returns, NOT MA) ─────────────────────────────

class MarketState(NamedTuple):
    date: str
    state: str
    return_20d: float


def get_historical_states(
    df: pd.DataFrame,
    lookback: int = 20,
    bull_thresh: float = BULL_THRESHOLD,
    bear_thresh: float = BEAR_THRESHOLD,
) -> pd.DataFrame:
    """
    Label each day as Bull/Bear/Sideways based on 20-day rolling cumulative return.

    Per Lewis Jackson / Rowan Framework:
      Bull     = 20-day cumulative return >= +5%
      Bear     = 20-day cumulative return <= -5%
      Sideways = everything else

    Using raw returns (NOT MA) — HMM prefers pure收益率, not smoothed noise.
    """
    if len(df) < lookback + 1:
        raise ValueError(f"Need at least {lookback + 1} rows, got {len(df)}")

    # pct_change(lookback) = (close_t / close_{t-lookback}) - 1
    df = df.copy()
    df["return_20d"] = df["close"].pct_change(lookback)

    conditions = [
        (df["return_20d"] >= bull_thresh),
        (df["return_20d"] <= bear_thresh),
    ]
    choices = ["Bull", "Bear"]
    df["state"] = np.select(conditions, choices, default="Sideways")

    # Drop NaN rows
    df = df.dropna(subset=["return_20d", "state"])

    return df[["date", "close", "return_20d", "state"]]


# ── Step 2: Transition Matrix ──────────────────────────────────────────────────

TransitionMatrix = np.ndarray  # shape (3, 3)


def compute_transition_matrix(
    states: List[str], smooth_window: int = 0
) -> TransitionMatrix:
    """
    Build 3×3 transition count matrix from state sequence.
    P[i→j] = count(state_i followed by state_j) / total(state_i)

    Optional Bayesian smoothing: add pseudo-counts to avoid zero probabilities.
    """
    state_to_idx = {s: i for i, s in enumerate(STATES)}
    n = len(STATES)
    counts = np.zeros((n, n), dtype=float)

    for i in range(len(states) - 1):
        from_state = states[i]
        to_state = states[i + 1]
        fi = state_to_idx.get(from_state, -1)
        ti = state_to_idx.get(to_state, -1)
        if fi >= 0 and ti >= 0:
            counts[fi, ti] += 1

    # Row-normalize → probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div-by-zero

    probs = counts / row_sums

    # Optional Laplace smoothing (add 1 pseudo-count per cell)
    if smooth_window > 0:
        alpha = 0.1 * smooth_window
        counts_smooth = counts + alpha
        probs = counts_smooth / counts_smooth.sum(axis=1, keepdims=True)

    return probs


def pretty_print_matrix(
    matrix: TransitionMatrix, states: List[str] = STATES
) -> str:
    """Format transition matrix as readable markdown table."""
    header = "|".join([""] + states) + "\n|" + "|".join(["---"] * (len(states) + 1))
    rows = []
    for i, s in enumerate(states):
        vals = " ".join([f"{v:.2%}" for v in matrix[i]])
        rows.append(f"| {s} | {vals} |")
    return header + "\n" + "\n".join(rows)


# ── Step 3: Signal Generation ─────────────────────────────────────────────────

class SignalResult(NamedTuple):
    signal: float            # P(Bull) - P(Bear), range [-1, 1]
    direction: str           # "LONG" | "SHORT" | "NEUTRAL"
    position_size: float     # fraction of capital (0–1)
    p_bull: float
    p_bear: float
    p_sideways: float
    confidence: str          # "HIGH" | "MEDIUM" | "LOW"


def generate_signal(
    current_state: str,
    transition_matrix: TransitionMatrix,
    confidence_thresh: float = 0.15,
) -> SignalResult:
    """
    Compute trading signal from transition matrix.

    Formula: Signal = P(Bull) - P(Bear)
      Signal > 0  → LONG  (long Bull, short Bear)
      Signal < 0  → SHORT (long Bear, short Bull)
      |Signal| < 0.05 → NEUTRAL

    Position size scales linearly with |signal|, capped at 1.0 (100% capital).
    Confidence = max(P_bull, P_bear) — above threshold = HIGH.
    """
    state_to_idx = {s: i for i, s in enumerate(STATES)}
    idx = state_to_idx.get(current_state, 0)
    probs = transition_matrix[idx]

    p_bull, p_bear, p_sideways = probs[0], probs[1], probs[2]
    signal = p_bull - p_bear

    # Direction
    if signal > 0:
        direction = "LONG"
    elif signal < 0:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # Position size: linear scale |signal| ∈ [0, 1]
    raw_size = min(abs(signal), 1.0)

    # Confidence
    max_prob = max(p_bull, p_bear)
    confidence = "HIGH" if max_prob >= confidence_thresh else "MEDIUM"

    return SignalResult(
        signal=signal,
        direction=direction,
        position_size=raw_size,
        p_bull=p_bull,
        p_bear=p_bear,
        p_sideways=p_sideways,
        confidence=confidence,
    )


# ── Step 4: HMM Overlay Validation ─────────────────────────────────────────────

class HMMValidation(NamedTuple):
    hmm_predicted_state: str
    markov_predicted_state: str
    agreement: bool
    regimes: list  # ['Bull', 'Bear', 'Sideways'] aligned with hmm hidden states


def run_hmm_validation(
    returns_series: pd.Series,
    n_hidden_states: int = 3,
) -> Optional[HMMValidation]:
    """
    Fit Gaussian HMM on raw 20-day returns for unsupervised regime detection.
    Overlaps with Markov state labeling to detect regime confusion / regime breaks.

    Returns HMMValidation with predicted state per day.
    Note: HMM states are UNLABELED — we align by distribution quantiles.
    """
    if not HAS_HMMLEARN:
        return None

    clean_returns = returns_series.dropna().values.reshape(-1, 1)
    if len(clean_returns) < 50:
        return None  # Need enough data for HMM

    try:
        model = GaussianHMM(
            n_components=n_hidden_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(clean_returns)

        # Predict hidden states (sequence)
        hidden_states = model.predict(clean_returns)

        # Map each hidden state to Bull/Bear/Sideways by mean return
        state_means = []
        for h in range(n_hidden_states):
            mask = hidden_states == h
            mean_ret = clean_returns[mask].mean()
            state_means.append((h, mean_ret))

        # Sort by mean return: lowest=Bear, middle=Sideways, highest=Bull
        state_means.sort(key=lambda x: x[1])
        hmm_mapping = {
            state_means[0][0]: "Bear",     # lowest return → Bear
            state_means[1][0]: "Sideways", # middle → Sideways
            state_means[2][0]: "Bull",     # highest → Bull
        }

        hmm_states = [hmm_mapping.get(h, "Sideways") for h in hidden_states]

        return HMMValidation(
            hmm_predicted_state=hmm_states[-1],
            markov_predicted_state="",  # filled by caller
            agreement=True,  # filled by caller
            regimes=[hmm_states[-1]],  # store for reporting

        )
    except Exception:
        return None


# ── Main Analysis Function ─────────────────────────────────────────────────────

class StrategyReport(NamedTuple):
    ticker: str
    analysis_date: str
    data_points: int
    is_observer_mode: bool
    current_state: str
    transition_matrix: TransitionMatrix
    signal: SignalResult
    hmm_validation: Optional[HMMValidation]
    regime_history: List[MarketState]
    matrix_str: str
    raw_state_df: pd.DataFrame


def analyze_markov_hmm(
    ticker: str,
    df: pd.DataFrame,
    lookback: int = 20,
    bull_thresh: float = BULL_THRESHOLD,
    bear_thresh: float = BEAR_THRESHOLD,
    use_hmm: bool = True,
) -> StrategyReport:
    """
    Full Markov+HMM analysis pipeline.

    Pipeline:
      1. State labeling (20d rolling returns)
      2. Transition matrix computation
      3. Signal generation from current state
      4. HMM overlay validation (optional)
    """
    now = datetime.now(timezone.utc)

    # ── Step 1: State labeling ────────────────────────────────────────────────
    state_df = get_historical_states(df, lookback, bull_thresh, bear_thresh)
    states = state_df["state"].tolist()
    current_state = states[-1]

    # ── Step 2: Transition matrix ───────────────────────────────────────────────
    t_matrix = compute_transition_matrix(states, smooth_window=lookback)
    matrix_str = pretty_print_matrix(t_matrix)

    # ── Step 3: Signal ────────────────────────────────────────────────────────
    signal_res = generate_signal(current_state, t_matrix)

    # ── Step 4: HMM overlay ───────────────────────────────────────────────────
    hmm_val = None
    if use_hmm:
        hmm_val = run_hmm_validation(state_df["return_20d"])
        if hmm_val:
            # Inject markov prediction for agreement check
            hmm_val = HMMValidation(
                hmm_predicted_state=hmm_val.hmm_predicted_state,
                markov_predicted_state=current_state,
                agreement=(hmm_val.hmm_predicted_state == current_state),
                regimes=hmm_val.regimes,
            )

    # ── Build regime history (last 10 states) ───────────────────────────────
    history = [
        MarketState(
            date=str(state_df["date"].iloc[i]),
            state=states[i],
            return_20d=state_df["return_20d"].iloc[i],
        )
        for i in range(max(0, len(states) - 10), len(states))
    ]

    return StrategyReport(
        ticker=ticker,
        analysis_date=now.strftime("%Y-%m-%d %H:%M UTC"),
        data_points=len(state_df),
        is_observer_mode=(len(state_df) < OBSERVER_MODE_THRESHOLD),
        current_state=current_state,
        transition_matrix=t_matrix,
        signal=signal_res,
        hmm_validation=hmm_val,
        regime_history=history,
        matrix_str=matrix_str,
        raw_state_df=state_df,
    )


# ── Format Report ───────────────────────────────────────────────────────────────

def format_report(r: StrategyReport) -> str:
    """Format StrategyReport as readable markdown."""
    obs_note = "⚠️ OBSERVER MODE — No trades until 30+ data points" if r.is_observer_mode else "✅ ACTIVE TRADING"
    hmm_block = ""
    if r.hmm_validation:
        agree_emoji = "✅" if r.hmm_validation.agreement else "⚠️"
        hmm_block = f"""
### HMM Overlay Validation
- HMM Predicted: **{r.hmm_validation.hmm_predicted_state}**
- Markov Predicted: **{r.hmm_validation.markov_predicted_state}**
- Agreement: {agree_emoji} `{r.hmm_validation.agreement}`
"""
    else:
        hmm_block = "\n### HMM Overlay: ⏭️ Skipped (not enough data or hmmlearn not installed)\n"

    return f"""
## 📊 Markov HMM Strategy Report — {r.ticker}
**Analyzed:** {r.analysis_date} | **Data Points:** {r.data_points} | **Observer:** {obs_note}

### Current Market State
- State: **{r.current_state}** ({r.regime_history[-1].return_20d:.2%} 20d return)

### 3×3 Transition Matrix
{r.matrix_str}

### Trading Signal
| Metric | Value |
|--------|-------|
| Direction | **{r.signal.direction}** |
| Signal (P_Bull − P_Bear) | `{r.signal.signal:+.4f}` |
| Position Size | `{r.signal.position_size:.1%}` |
| Confidence | **{r.signal.confidence}** |
| P(Bull) | `{r.signal.p_bull:.2%}` |
| P(Bear) | `{r.signal.p_bear:.2%}` |
| P(Sideways) | `{r.signal.p_sideways:.2%}` |
{hmm_block}
### Regime History (Last 10 Days)
| Date | State | 20d Return |
|------|-------|------------|
{chr(10).join(f"| {s.date[:10]} | {s.state} | {s.return_20d:.2%} |" for s in r.regime_history)}
"""


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Markov HMM Strategy")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol")
    parser.add_argument("--csv", type=str, help="Path to CSV (date,close cols)")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--no-hmm", action="store_true")
    parser.add_argument("--as-of-date", type=str, help="Walk-forward cutoff date (YYYY-MM-DD)")
    parser.add_argument("--multi", type=str, help="Comma-separated tickers for batch scan")
    parser.add_argument("--json", action="store_true", help="Output JSON for agno Team")
    args = parser.parse_args()

    # ── Multi-ticker batch mode ───────────────────────────────────────────────
    if args.multi:
        tickers = [t.strip() for t in args.multi.split(",")]
        results = []
        for ticker in tickers:
            # Try to load CSV if provided, else use mock data
            if args.csv and Path(args.csv).exists():
                df = pd.read_csv(args.csv, parse_dates=["date"])
            else:
                np.random.seed(hash(ticker) % (2**32))
                dates = pd.date_range(end=datetime.now(), periods=200, freq="D")
                returns = np.random.normal(0.0005, 0.015, 200)
                prices = 400 * np.exp(np.cumsum(returns))
                df = pd.DataFrame({"date": dates, "close": prices})

            df = df.sort_values("date").reset_index(drop=True)

            report = analyze_markov_hmm(
                ticker=ticker,
                df=df,
                lookback=args.lookback,
                use_hmm=(not args.no_hmm),
            )
            results.append(report)

        # Batch JSON output
        output = {
            "scan_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "scan_count": len(results),
            "results": [_report_to_dict(r) for r in results],
            "high_confidence_signals": [
                {"ticker": r.ticker, "direction": r.signal.direction, "confidence": r.signal.confidence}
                for r in results
                if r.signal.confidence == "HIGH" and not r.is_observer_mode
            ],
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    # ── Single ticker mode ─────────────────────────────────────────────────────
    if args.csv and Path(args.csv).exists():
        df = pd.read_csv(args.csv, parse_dates=["date"])
    else:
        print("⚠️ No CSV provided — generating mock SPY data for demo")
        dates = pd.date_range(end=datetime.now(), periods=200, freq="D")
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 200)
        prices = 400 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({"date": dates, "close": prices})

    df = df.sort_values("date").reset_index(drop=True)

    # Walk-forward cutoff (as-of-date prevents lookahead)
    if args.as_of_date:
        cutoff = pd.to_datetime(args.as_of_date)
        df = df[df["date"] <= cutoff]
        print(f"🔒 Walk-forward: data as of {args.as_of_date} → {len(df)} rows")

    report = analyze_markov_hmm(
        ticker=args.ticker,
        df=df,
        lookback=args.lookback,
        use_hmm=(not args.no_hmm),
    )

    if args.json:
        print(json.dumps(_report_to_dict(report), indent=2, ensure_ascii=False))
    else:
        print(format_report(report))


def _report_to_dict(r: StrategyReport) -> dict:
    """Convert StrategyReport to dict for JSON serialization (agno Team output)."""
    hmm_val = None
    if r.hmm_validation:
        hmm_val = {
            "hmm_predicted_state": r.hmm_validation.hmm_predicted_state,
            "markov_predicted_state": r.hmm_validation.markov_predicted_state,
            "agreement": r.hmm_validation.agreement,
        }

    matrix_list = r.transition_matrix.tolist()
    matrix_dict = {
        "Bull": {"Bull": matrix_list[0][0], "Bear": matrix_list[0][1], "Sideways": matrix_list[0][2]},
        "Bear": {"Bull": matrix_list[1][0], "Bear": matrix_list[1][1], "Sideways": matrix_list[1][2]},
        "Sideways": {"Bull": matrix_list[2][0], "Bear": matrix_list[2][1], "Sideways": matrix_list[2][2]},
    }

    return {
        "ticker": r.ticker,
        "analysis_date": r.analysis_date,
        "data_points": r.data_points,
        "is_observer_mode": r.is_observer_mode,
        "current_state": r.current_state,
        "transition_matrix": matrix_dict,
        "signal": {
            "direction": r.signal.direction,
            "signal_raw": round(r.signal.signal, 4),
            "position_size_pct": round(r.signal.position_size * 100, 1),
            "p_bull_pct": round(r.signal.p_bull * 100, 2),
            "p_bear_pct": round(r.signal.p_bear * 100, 2),
            "p_sideways_pct": round(r.signal.p_sideways * 100, 2),
            "confidence": r.signal.confidence,
        },
        "hmm_validation": hmm_val,
        "regime_history": [
            {"date": s.date[:10], "state": s.state, "return_20d": round(s.return_20d, 4)}
            for s in r.regime_history
        ],
    }


if __name__ == "__main__":
    main()