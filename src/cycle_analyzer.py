"""
Cycle analysis module for Kriterion Quant Trading System
Implements CAUSAL cycle analysis using Hilbert Transform (Expanding Window)
"""

import pandas as pd
import numpy as np
from scipy.signal import hilbert, welch, find_peaks
from scipy.stats import percentileofscore
import pywt
from typing import Tuple, Dict, Optional

from config import Config

class CycleAnalyzer:
    """Class to perform cycle analysis on price data"""
    
    def __init__(self, fast_window: int = None, slow_window: int = None):
        """
        Initialize the cycle analyzer
        
        Parameters
        ----------
        fast_window : int, optional
            Fast MA window. Defaults to Config.FAST_MA_WINDOW
        slow_window : int, optional
            Slow MA window. Defaults to Config.SLOW_MA_WINDOW
        """
        self.fast_window = fast_window or Config.FAST_MA_WINDOW
        self.slow_window = slow_window or Config.SLOW_MA_WINDOW
    
    def create_causal_oscillator(self, price_series: pd.Series) -> pd.Series:
        """
        Create a causal oscillator using dual moving average difference
        This avoids look-ahead bias
        
        Parameters
        ----------
        price_series : pd.Series
            Price series (typically close prices)
        
        Returns
        -------
        pd.Series
            Causal oscillator values
        """
        # Calculate moving averages
        fast_ma = price_series.rolling(window=self.fast_window).mean()
        slow_ma = price_series.rolling(window=self.slow_window).mean()
        
        # Oscillator is the difference
        oscillator = fast_ma - slow_ma
        
        return oscillator
    
    def apply_hilbert_transform(self, oscillator: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply Hilbert Transform to extract phase and amplitude in a CAUSAL manner.
        
        CRITICAL UPDATE: This method uses an EXPANDING WINDOW approach to calculate 
        the analytic signal. This eliminates Look-Ahead Bias (Repainting) which 
        occurs when applying FFT-based Hilbert transform on the entire dataset at once.
        
        Parameters
        ----------
        oscillator : pd.Series
            Oscillator series
        
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Phase and amplitude series (Causal)
        """
        # Remove NaN values for Hilbert transform calculation
        clean_oscillator = oscillator.dropna()
        
        if len(clean_oscillator) < 50:
            raise ValueError("Not enough data points for Hilbert transform")
        
        values = clean_oscillator.values
        index = clean_oscillator.index
        
        # Prepare arrays for causal results (filled with NaN initially)
        causal_phase = np.full(len(values), np.nan)
        causal_amplitude = np.full(len(values), np.nan)
        
        # Minimum window to start calculating transforms
        min_window = 50
        
        print(f"⏳ Computing Causal Hilbert Transform (Expanding Window) on {len(values)} points...")
        print("   Note: This prevents repainting but may take longer than standard analysis.")
        
        # --- EXPANDING WINDOW LOOP ---
        # Simulates real-time data arrival day by day
        for i in range(min_window, len(values)):
            # Window: from start of data up to current point i
            # This represents "all data available up to today"
            current_slice = values[:i+1]
            
            # Apply Hilbert Transform on available history
            analytic = hilbert(current_slice)
            
            # Extract metrics ONLY for the last point (the current simulated "today")
            # We discard the rest of the analytic signal for this step because 
            # past values in 'analytic' would have changed based on new data (repainting).
            last_analytic = analytic[-1]
            
            causal_phase[i] = np.angle(last_analytic)
            causal_amplitude[i] = np.abs(last_analytic)
            
            # Progress indicator for large datasets
            if i % 500 == 0 and i > 0:
                print(f"   Processed {i}/{len(values)} points")
        
        # Convert back to series with original index of the clean data
        phase_series = pd.Series(causal_phase, index=index, name='phase')
        amplitude_series = pd.Series(causal_amplitude, index=index, name='amplitude')
        
        # Realign with the original input oscillator index (handling initial NaNs from MAs)
        phase_series = phase_series.reindex(oscillator.index)
        amplitude_series = amplitude_series.reindex(oscillator.index)
        
        return phase_series, amplitude_series
    
    def classify_phase_quadrant(self, phase: pd.Series) -> pd.Series:
        """
        Classify phase into quadrants for trading signals
        
        Parameters
        ----------
        phase : pd.Series
            Phase series in radians (-π to π)
        
        Returns
        -------
        pd.Series
            Quadrant classification
        """
        bins = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        labels = Config.get_phase_labels()
        
        quadrants = pd.cut(phase, bins=bins, labels=labels, include_lowest=True)
        
        return quadrants
    
    def analyze_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete cycle analysis on price data
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC data
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added cycle analysis columns
        """
        print("⚙️ Performing cycle analysis...")
        
        # Create copy to avoid modifying original
        result_df = df.copy()
        
        # Step 1: Create causal oscillator
        oscillator = self.create_causal_oscillator(result_df['close'])
        result_df['oscillator'] = oscillator
        
        # Step 2: Apply Hilbert Transform (Now Causal)
        phase, amplitude = self.apply_hilbert_transform(oscillator)
        result_df['phase'] = phase
        result_df['amplitude'] = amplitude
        
        # Step 3: Classify phase quadrants
        result_df['phase_quadrant'] = self.classify_phase_quadrant(result_df['phase'])
        
        # Step 4: Determine if in bullish regime
        result_df['bullish_regime'] = result_df['phase_quadrant'].isin(Config.BULLISH_QUADRANTS)
        
        # Remove rows with NaN values (resulting from MAs and Hilbert warmup)
        result_df.dropna(inplace=True)
        
        print(f"✅ Cycle analysis complete. Analyzed {len(result_df)} data points")
        
        return result_df
    
    def run_spectral_analysis(self, oscillator: pd.Series) -> Dict:
        """
        Run spectral analysis using Welch periodogram
        
        Parameters
        ----------
        oscillator : pd.Series
            Oscillator series to analyze
        
        Returns
        -------
        Dict
            Spectral analysis results
        """
        clean_oscillator = oscillator.dropna().values
        
        # Welch periodogram
        frequencies, power = welch(clean_oscillator, fs=1, nperseg=min(Config.NPERSEG, len(clean_oscillator)//2))
        
        # Convert to periods
        periods = 1 / frequencies[1:] if len(frequencies) > 1 else np.array([])
        power = power[1:] if len(power) > 1 else np.array([])
        
        # Find peaks
        peaks, properties = find_peaks(power, prominence=np.max(power) * 0.1)
        
        dominant_period = periods[np.argmax(power)] if len(power) > 0 else None
        dominant_power = np.max(power) if len(power) > 0 else None
        
        return {
            'periods': periods,
            'power': power,
            'peaks': peaks,
            'dominant_period': dominant_period,
            'dominant_power': dominant_power
        }
    
    def run_monte_carlo_significance_test(
        self, 
        oscillator: pd.Series, 
        n_simulations: int = None
    ) -> Dict:
        """
        Test statistical significance of cycles using Monte Carlo
        
        Parameters
        ----------
        oscillator : pd.Series
            Oscillator series to test
        n_simulations : int, optional
            Number of simulations. Defaults to Config.MONTE_CARLO_SIMULATIONS
        
        Returns
        -------
        Dict
            Monte Carlo test results including p-value
        """
        n_simulations = n_simulations or Config.MONTE_CARLO_SIMULATIONS
        clean_oscillator = oscillator.dropna().values
        
        # Get observed max power
        spectral_results = self.run_spectral_analysis(oscillator)
        observed_max_power = spectral_results['dominant_power']
        
        if observed_max_power is None:
            return {'p_value': 1.0, 'significant': False}
        
        # Run simulations
        simulated_max_powers = []
        
        for _ in range(n_simulations):
            # Shuffle data to destroy temporal patterns
            shuffled = np.random.permutation(clean_oscillator)
            
            # Calculate periodogram for shuffled data
            _, sim_power = welch(shuffled, fs=1, nperseg=min(Config.NPERSEG, len(shuffled)//2))
            
            if len(sim_power) > 0:
                simulated_max_powers.append(np.max(sim_power))
        
        # Calculate p-value
        if simulated_max_powers:
            p_value = np.sum(np.array(simulated_max_powers) >= observed_max_power) / len(simulated_max_powers)
        else:
            p_value = 1.0
        
        return {
            'p_value': p_value,
            'significant': p_value < 0.05,
            'observed_power': observed_max_power,
            'simulated_powers': simulated_max_powers
        }
    
    def calculate_forward_returns(
        self, 
        df: pd.DataFrame, 
        horizons: list = [1, 5, 10, 21]
    ) -> pd.DataFrame:
        """
        Calculate forward returns for diagnostic analysis
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price and phase data
        horizons : list, optional
            Forward return horizons in days
        
        Returns
        -------
        pd.DataFrame
            DataFrame with forward returns added
        """
        result_df = df.copy()
        
        for horizon in horizons:
            result_df[f'fwd_return_{horizon}d'] = (
                result_df['close'].pct_change(periods=horizon).shift(-horizon) * 100
            )
        
        return result_df
    
    def get_phase_performance_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance map by phase quadrant
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with phase and forward returns
        
        Returns
        -------
        pd.DataFrame
            Performance statistics by phase quadrant
        """
        # Ensure forward returns are calculated
        if 'fwd_return_1d' not in df.columns:
            df = self.calculate_forward_returns(df)
        
        # Group by phase quadrant and calculate mean returns
        performance_map = df.groupby('phase_quadrant')[
            [f'fwd_return_{h}d' for h in [1, 5, 10, 21]]
        ].mean()
        
        return performance_map
