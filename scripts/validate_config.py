#!/usr/bin/env python3
"""validate_config.py — 驗證策略配置格式"""
import yaml, json, sys

def main():
    try:
        with open('us_stock_strategy.yaml') as f:
            cfg = yaml.safe_load(f)
        with open('strategy_goals.json') as f:
            goals = json.load(f)
        
        print(f"✅ Strategy: {cfg['metadata']['strategy_name']} v{cfg['metadata']['version']}")
        print(f"✅ RSI: {cfg['learned_parameters']['rsi_oversold_threshold']}/{cfg['learned_parameters']['rsi_overbought_threshold']}")
        print(f"✅ Stop Loss: {cfg['learned_parameters']['trailing_stop_loss_pct']}%")
        print(f"✅ Take Profit: {cfg['learned_parameters']['take_profit_pct']}%")
        print(f"✅ Sharpe threshold: {goals['strategy_goals']['success_metrics']['minimum_sharpe_ratio']}")
        print(f"✅ Win rate threshold: {goals['strategy_goals']['success_metrics']['win_rate_threshold_pct']}%")
        return 0
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())