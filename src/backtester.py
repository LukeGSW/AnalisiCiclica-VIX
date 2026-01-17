"""
Backtesting module for Kriterion Quant Trading System
Implements a realistic event-driven backtester with Correct Timing (Next Open Execution)
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
        
        Parameters
        ----------
        data_path : str
            The path to the directory where backtest results will be saved.
        initial_capital : float, optional
            Starting capital. Defaults to Config.INITIAL_CAPITAL.
        fees : float, optional
            Trading fees as a percentage. Defaults to Config.TRADING_FEES.
        """
        self.data_path = data_path
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.fees = fees or Config.TRADING_FEES

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Runs a realistic event-driven backtest simulating a trading account.
        FIX: Trades are executed at the OPEN of the NEXT bar after the signal.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with signals and prices.
            
        Returns
        -------
        Dict
            A dictionary containing the results DataFrame, performance metrics, and trade log.
        """
        print("📊 Running realistic event-driven backtest (Next Open Execution)...")
        
        if 'signal' not in df.columns or 'close' not in df.columns or 'open' not in df.columns:
            raise ValueError("DataFrame must contain 'signal', 'close', and 'open' columns")

        results = df.copy()
        
        # --- Simulazione del Conto di Trading ---
        cash = self.initial_capital
        shares = 0.0
        trade_size_dollars = self.initial_capital # Usa il capitale iniziale come dimensione fissa
        
        equity_over_time = []
        positions_over_time = []
        trades_log = []
        entry_details = {}

        # Pre-fill lists with initial state
        # We start iterating from index 1 because we need to look at yesterday's signal
        
        # Initial state for day 0 (cannot trade yet)
        equity_over_time.append(self.initial_capital)
        positions_over_time.append(0)

        for i in range(1, len(results)):
            # Market Data for TODAY (Day i)
            current_date = results.index[i]
            today_open = results.iloc[i]['open']
            today_close = results.iloc[i]['close']
            
            # Signal from YESTERDAY (Day i-1)
            # This is the signal generated after yesterday's close
            yesterday_signal = results.iloc[i-1]['signal']

            # --- EXECUTION LOGIC (At Open of Day i) ---
            
            # Entry Logic: Yesterday said BUY, we are FLAT -> Buy at Today Open
            if yesterday_signal == 'BUY' and shares == 0:
                shares_to_buy = trade_size_dollars / today_open
                cost = shares_to_buy * today_open * (1 + self.fees)
                
                # Update shares (simplified fixed position sizing)
                shares = shares_to_buy
                entry_details = {'entry_date': current_date, 'entry_price': today_open}

            # Exit Logic: Yesterday said SELL, we are LONG -> Sell at Today Open
            elif yesterday_signal == 'SELL' and shares > 0:
                revenue = shares * today_open * (1 - self.fees)
                
                if entry_details:
                    # Calculate Trade Return
                    trade_return = (today_open - entry_details['entry_price']) / entry_details['entry_price']
                    
                    trades_log.append({
                        'entry_date': entry_details['entry_date'],
                        'entry_price': entry_details['entry_price'],
                        'exit_date': current_date,
                        'exit_price': today_open,
                        'return': trade_return
                    })
                    
                    # Update Cash (Profit/Loss realized)
                    # PnL = (Exit Price - Entry Price) * Shares - Fees
                    # Note: We subtract fees from both entry and exit implicit in the cost/revenue calc above?
                    # Let's align with previous logic:
                    # Profit = (Price_diff * shares) - (EntryCost_fees + ExitCost_fees)
                    # Ideally: cash += revenue - (original_cost_basis)
                    
                    # Simple PnL update to cash:
                    # We assume 'trade_size_dollars' was effectively 'frozen' or 'used'.
                    # Here we just add the PnL to the running cash pile.
                    # PnL = (today_open - entry_details['entry_price']) * shares - (trade_size_dollars * self.fees * 2)
                    
                    # A more precise cash tracking:
                    # On Buy: cash -= cost
                    # On Sell: cash += revenue
                    # But the original code didn't track "cash balance" strictly for buying power, 
                    # it used 'trade_size_dollars' as a fixed bet size. 
                    # Let's stick to the PnL accumulation logic to track equity.
                    
                    profit_loss = (today_open - entry_details['entry_price']) * shares - (trade_size_dollars * self.fees * 2)
                    cash += profit_loss

                shares = 0
                entry_details = {}

            # --- MARK TO MARKET (At Close of Day i) ---
            # Equity = Cash + Current Value of Positions
            # If flat, Equity = Cash
            # If long, Equity = Cash + (Shares * Today Close) 
            # Note: The 'Cash' variable here tracks "Realized Equity". 
            # To get "Floating Equity", we add the unrealized PnL.
            
            floating_pnl = 0
            if shares > 0:
                # Value of position at close
                position_value = shares * today_close
                # Cost basis was: shares * entry_price
                # But 'cash' already reflects the realized PnL of closed trades. 
                # To calculate Total Equity:
                # Total Equity = Realized Cash + (Position Value - Cost Basis)? 
                # No, simpler: 
                # Assume 'cash' is the account balance.
                # If we are in a trade, we "spent" money. 
                # Let's use the simple logic: Equity = Initial Capital + Sum(Realized PnL) + Unrealized PnL
                
                unrealized_pnl = (today_close - entry_details['entry_price']) * shares
                current_equity = cash + unrealized_pnl
            else:
                current_equity = cash
                
            equity_over_time.append(current_equity)
            positions_over_time.append(1 if shares > 0 else 0)

        # Aggiungi i risultati al DataFrame
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
        
        # Run backtests on each segment
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
        """Calculate comprehensive backtest metrics."""
        # Equity Curve Metrics
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

        # Trade Log Metrics
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
        """Save backtest results to JSON file inside the data_path directory."""
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
