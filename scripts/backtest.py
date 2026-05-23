#!/usr/bin/env python3
"""
backtest.py — 單一變數回測驗證腳本
當 Hermès 修改 learned_parameters 後，自動 call 呢個 script 做回測
"""
import pandas as pd
import numpy as np
import yaml
import json
import sys
from datetime import datetime, timedelta

YAML_FILE = "us_stock_strategy.yaml"
GOALS_FILE = "strategy_goals.json"
CSV_FILE = "trades_ledger.csv"

def load_config():
    with open(YAML_FILE) as f:
        return yaml.safe_load(f)

def load_goals():
    with open(GOALS_FILE) as f:
        return json.load(f)

def load_trades():
    try:
        df = pd.read_csv(CSV_FILE)
        return df.dropna(subset=['trade_id'])
    except:
        return pd.DataFrame()

def calculate_metrics(df):
    """計算核心指標"""
    if df.empty or 'realized_pnl_pct' not in df.columns:
        return None
    
    closed = df[df['position_status'] == 'CLOSED'].copy()
    if closed.empty:
        return None
    
    total = len(closed)
    wins = len(closed[closed['realized_pnl_pct'] > 0])
    win_rate = wins / total * 100 if total > 0 else 0
    
    mean_ret = closed['realized_pnl_pct'].mean()
    std_ret = closed['realized_pnl_pct'].std()
    sharpe = (mean_ret / std_ret * (252**0.5)) if std_ret > 0 else 0
    
    closed['cumsum'] = closed['realized_pnl_pct'].cumsum()
    running_max = closed['cumsum'].cummax()
    drawdown = (closed['cumsum'] - running_max).min()
    
    return {
        'total_trades': total,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'max_drawdown': drawdown,
        'mean_return': mean_ret,
        'std_return': std_ret,
    }

def validate_against_goals(metrics, goals):
    """對比 strategy_goals.json"""
    success = goals['strategy_goals']['success_metrics']
    failure = goals['strategy_goals']['failure_conditions']
    backtest_req = goals['strategy_goals']['backtest_requirements']
    
    if metrics is None:
        return {'pass': False, 'reason': 'No metrics available'}
    
    # Check if we have enough trades
    if metrics['total_trades'] < backtest_req['min_trades_for_significance']:
        return {
            'pass': None,  # Inconclusive
            'reason': f"Only {metrics['total_trades']} trades, need {backtest_req['min_trades_for_significance']} for significance"
        }
    
    # Check failure conditions
    if metrics['max_drawdown'] < -failure['max_drawdown_pct']:
        return {'pass': False, 'reason': f"Max drawdown {metrics['max_drawdown']:.2f}% exceeds {failure['max_drawdown_pct']}%"}
    
    if metrics['sharpe_ratio'] < failure['unacceptable_sharpe_ratio']:
        return {'pass': False, 'reason': f"Sharpe {metrics['sharpe_ratio']:.2f} below unacceptable {failure['unacceptable_sharpe_ratio']}"}
    
    # Check success conditions
    if metrics['win_rate'] >= success['win_rate_threshold_pct'] and metrics['sharpe_ratio'] >= success['minimum_sharpe_ratio']:
        return {'pass': True, 'reason': 'All success metrics met'}
    
    return {'pass': False, 'reason': 'Some metrics below threshold'}

def main():
    print(f"=== Backtest Report — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    
    cfg = load_config()
    goals = load_goals()
    trades = load_trades()
    
    print(f"Strategy: {cfg['metadata']['strategy_name']} v{cfg['metadata']['version']}")
    print(f"Parameters: RSI={cfg['learned_parameters']['rsi_oversold_threshold']}/{cfg['learned_parameters']['rsi_overbought_threshold']} "
          f"Stop={cfg['learned_parameters']['trailing_stop_loss_pct']}% "
          f"TP={cfg['learned_parameters']['take_profit_pct']}%")
    
    metrics = calculate_metrics(trades)
    
    if metrics:
        print(f"\n📊 Metrics ({metrics['total_trades']} closed trades):")
        print(f"   Win Rate:     {metrics['win_rate']:.1f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Mean Return:  {metrics['mean_return']:.3f}%")
    else:
        print("\n⚠️ No closed trades to analyze")
    
    validation = validate_against_goals(metrics, goals)
    
    print(f"\n{'✅' if validation['pass'] == True else '❌' if validation['pass'] == False else '⚠️'} Validation: {validation['reason']}")
    
    return 0 if validation['pass'] in (True, None) else 1

if __name__ == '__main__':
    sys.exit(main())