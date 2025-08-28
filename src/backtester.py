# src/backtester.py

"""
Backtesting module for Kriterion Quant Trading System
Implements a realistic event-driven backtester for fixed-size trades.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import json
import os
from config import Config

class Backtester:
    """Class to perform realistic, event-driven backtesting."""
    
    def __init__(self, data_path: str, initial_capital: float = None, fees: float = None):
        """
        Initialize the backtester.
        """
        self.data_path = data_path
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.fees = fees or Config.TRADING_FEES

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Runs a realistic event-driven backtest simulating a trading account.
        """
        print("📊 Running realistic event-driven backtest...")
        
        if 'signal' not in df.columns or 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'signal' and 'close' columns")

        results = df.copy()
        
        cash = self.initial_capital
        shares = 0.0
        trade_size_dollars = self.initial_capital

        equity_over_time = []
        positions_over_time = []
        trades_log = []
        
        entry_details = {}

        for i in range(len(results)):
            current_price = results.iloc[i]['close']
            signal = results.iloc[i]['signal']
            current_date = results.index[i]

            if signal == 'BUY' and shares == 0:
                shares_to_buy = trade_size_dollars / current_price
                shares = shares_to_buy
                entry_details = {'entry_date': current_date, 'entry_price': current_price}

            elif signal == 'SELL' and shares > 0:
                if entry_details:
                    trade_return = (current_price - entry_details['entry_price']) / entry_details['entry_price']
                    
                    trades_log.append({
                        'entry_date': entry_details['entry_date'],
                        'entry_price': entry_details['entry_price'],
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'return': trade_return
                    })
                    
                    profit_loss = (current_price - entry_details['entry_price']) * shares - (trade_size_dollars * self.fees * 2)
                    cash += profit_loss

                shares = 0
                entry_details = {}

            current_equity = cash
            equity_over_time.append(current_equity)
            positions_over_time.append(1 if shares > 0 else 0)

        results['equity'] = equity_over_time
        results['position'] = positions_over_time
        results['returns'] = results['close'].pct_change().fillna(0)
        results['benchmark_equity'] = self.initial_capital * (1 + results['returns']).cumprod()
        
        metrics = self._calculate_metrics(results, trades_log)
        
        return {
            'results': results,
            'metrics': metrics,
            'final_equity': float(results['equity'].iloc[-1]),
            'total_return': float((results['equity'].iloc[-1] / self.initial_capital - 1) * 100),
            'trades_log': trades_log
        }

    def run_walk_forward_analysis(self, df: pd.DataFrame, in_sample_ratio: float = None) -> Dict:
        """
        Perform walk-forward analysis for robust validation.
        """
        print("🔄 Running walk-forward analysis...")
        
        in_sample_ratio = in_sample_ratio or Config.IN_SAMPLE_RATIO
        
        split_idx = int(len(df) * in_sample_ratio)
        
        is_data = df.iloc[:split_idx]
        oos_data = df.iloc[split_idx:]
        
        print(f"  In-Sample: {is_data.index[0].date()} to {is_data.index[-1].date()} ({len(is_data)} days)")
        print(f"  Out-of-Sample: {oos_data.index[0].date()} to {oos_data.index[-1].date()} ({len(oos_data)} days)")
        
        is_results = self.run_backtest(is_data) if len(is_data) > 10 else None
        oos_results = self.run_backtest(oos_data) if len(oos_data) > 10 else None
        
        result = {}
        if is_results:
            result['in_sample'] = is_results
            result['in_sample_metrics'] = is_results['metrics']
        
        if oos_results:
            result['out_of_sample'] = oos_results
            result['out_of_sample_metrics'] = oos_results['metrics']
            
        return result

    def _calculate_metrics(self, results: pd.DataFrame, trades_log: List[Dict]) -> Dict:
        """
        Calculate comprehensive backtest metrics from realistic simulation results.
        """
        total_return = (results['equity'].iloc[-1] / self.initial_capital - 1) * 100
        
        running_max = results['equity'].expanding().max()
        drawdown = (results['equity'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        daily_returns = results['equity'].pct_change().fillna(0)
        
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std()
            sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else float('inf')
            
            annual_return = daily_returns.mean() * 252 * 100
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0.0

        trade_returns = [t['return'] for t in trades_log]
        
        if trade_returns:
            winning_trades = sum(1 for r in trade_returns if r > (self.fees * 2))
            win_rate = (winning_trades / len(trade_returns)) * 100 if trade_returns else 0.0
            
            gross_profits = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
            
        return {
            'total_return_%': float(total_return),
            'max_drawdown_%': abs(float(max_drawdown)),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'total_trades': len(trade_returns),
            'win_rate_%': float(win_rate),
            'profit_factor': float(profit_factor)
        }
        
    def save_backtest_results(self, results: Dict) -> str:
        """
        Save backtest results to JSON file inside the data_path directory.
        """
        filepath = os.path.join(self.data_path, 'backtest_results.json')
        
        serializable_results = {}
        if 'in_sample_metrics' in results:
            serializable_results['in_sample'] = results['in_sample_metrics']
        if 'out_of_sample_metrics' in results:
            serializable_results['out_of_sample'] = results['out_of_sample_metrics']
            
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"💾 Backtest results saved to {filepath}")
        return filepath
