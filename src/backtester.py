# src/backtester.py

import pandas as pd
import numpy as np
import json
import os

from config import Config

class Backtester:
    """
    Handles backtesting of trading signals using vectorized methods
    for speed and accuracy.
    """

    def __init__(self, data_path: str):
        """
        Initialize the backtester.

        Parameters
        ----------
        data_path : str
            The path to the directory where backtest results should be saved.
        """
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def run_backtest(self, df_signals: pd.DataFrame) -> dict:
        """
        Runs a vectorized backtest on the provided signals DataFrame.
        This method is suitable for a simple, single run on a given dataset.
        """
        if df_signals.empty:
            return {'metrics': {}, 'results': pd.DataFrame()}

        # 1. Calculate daily returns of the asset
        df = df_signals.copy()
        df['returns'] = df['close'].pct_change()

        # 2. Define positions based on signals (long-only strategy)
        # We enter the position on the day AFTER the signal (shift(1))
        df['position'] = np.nan
        df.loc[df['signal'] == 'BUY', 'position'] = 1
        df.loc[df['signal'] == 'SELL', 'position'] = 0
        df['position'].fillna(method='ffill', inplace=True)
        df['position'].fillna(0, inplace=True) # Start with no position
        
        # The actual position for calculation is the previous day's signal
        df['position'] = df['position'].shift(1).fillna(0)

        # 3. Calculate strategy returns
        df['strategy_returns'] = df['returns'] * df['position']
        
        # Apply trading fees on trades
        trades = df['position'].diff().abs()
        transaction_costs = trades * Config.TRADING_FEES
        df['strategy_returns'] -= transaction_costs

        # 4. Calculate equity curves
        initial_capital = Config.INITIAL_CAPITAL
        df['equity'] = initial_capital * (1 + df['strategy_returns']).cumprod()
        df['benchmark_equity'] = initial_capital * (1 + df['returns']).cumprod()

        # 5. Calculate performance metrics
        metrics = self._calculate_metrics(df)

        return {
            'metrics': metrics,
            'results': df[['equity', 'benchmark_equity', 'signal', 'position']]
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculates performance metrics from a backtest results DataFrame."""
        
        equity_series = df['equity']
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100
        
        # Calculate Drawdown
        cumulative_max = equity_series.cummax()
        drawdown = (equity_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        
        # Sharpe Ratio (assuming risk-free rate is 0)
        daily_returns = df['strategy_returns']
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
        
        # Trade analysis
        trades = df['position'].diff().abs()
        total_trades = trades.sum() / 2 # Each trade has an entry and exit
        
        trade_returns = daily_returns[trades == 1].copy()
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        
        total_profit = (1 + wins).prod() - 1
        total_loss = (1 + losses).prod() - 1
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else np.inf

        return {
            'total_return_%': round(total_return, 2),
            'max_drawdown_%': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': int(total_trades),
            'win_rate_%': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2)
        }

    def run_walk_forward_analysis(self, df_signals: pd.DataFrame) -> dict:
        """
        Performs a walk-forward analysis by splitting data into in-sample and
        out-of-sample periods.
        """
        if df_signals.empty:
            return {'in_sample': {}, 'out_of_sample': {}}
            
        split_point = int(len(df_signals) * Config.IN_SAMPLE_RATIO)
        
        df_in_sample = df_signals.iloc[:split_point]
        df_out_of_sample = df_signals.iloc[split_point:]

        print(f"🔬 Running Walk-Forward Analysis...")
        print(f"  - In-Sample: {df_in_sample.index[0].date()} to {df_in_sample.index[-1].date()} ({len(df_in_sample)} days)")
        print(f"  - Out-of-Sample: {df_out_of_sample.index[0].date()} to {df_out_of_sample.index[-1].date()} ({len(df_out_of_sample)} days)")

        # Run backtest on both periods
        in_sample_results = self.run_backtest(df_in_sample)
        out_of_sample_results = self.run_backtest(df_out_of_sample)

        return {
            'in_sample': in_sample_results['metrics'],
            'out_of_sample': out_of_sample_results['metrics']
        }

    def save_backtest_results(self, results: dict):
        """Saves the backtest results dictionary to a JSON file."""
        filepath = os.path.join(self.data_path, 'backtest_results.json')
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Backtest results saved to {filepath}")
