"""
Signal generation module for Kriterion Quant Trading System
Generates BUY/SELL/HOLD signals based on cycle analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional
import json
import os

from config import Config

class SignalGenerator:
    """Class to generate trading signals from cycle analysis"""
    
    def __init__(self, data_path: str):
        """
        Initialize the signal generator
        
        Parameters
        ----------
        data_path : str
            The path to the directory where signal files will be saved.
        """
        self.data_path = data_path
        self.last_signal = None
        self.signal_history = []

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on cycle phase
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cycle analysis (must contain 'phase_quadrant' column)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with trading signals added
        """
        print("🎯 Generating trading signals...")
        
        if 'phase_quadrant' not in df.columns:
            raise ValueError("DataFrame must contain 'phase_quadrant' column from cycle analysis")
        
        result_df = df.copy()
        
        # Initialize signal column
        result_df['signal'] = 'HOLD'
        
        # Detect regime changes
        result_df['in_bullish'] = result_df['phase_quadrant'].isin(Config.BULLISH_QUADRANTS)
        
        # Generate entry signals (transition into bullish regime)
        entries = result_df['in_bullish'] & ~result_df['in_bullish'].shift(1, fill_value=False)
        result_df.loc[entries, 'signal'] = 'BUY'
        
        # Generate exit signals (transition out of bullish regime)
        exits = ~result_df['in_bullish'] & result_df['in_bullish'].shift(1, fill_value=False)

        result_df.loc[exits, 'signal'] = 'SELL'
        
        # Add position tracking
        result_df['position'] = 0
        result_df.loc[result_df['in_bullish'], 'position'] = 1
        
        # Calculate signal strength (based on amplitude and phase clarity)
        result_df['signal_strength'] = self._calculate_signal_strength(result_df)
        
        # Add confidence level based on statistical significance
        result_df['confidence'] = self._calculate_confidence(result_df)
        
        print(f"✅ Generated {(result_df['signal'] != 'HOLD').sum()} trading signals")
        
        return result_df
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate signal strength based on amplitude and phase stability
        CORRECTED: Uses expanding window for normalization to avoid look-ahead bias.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with cycle analysis
        
        Returns
        -------
        pd.Series
            Signal strength values (0-100)
        """
        strength = pd.Series(50.0, index=df.index)
        
        if 'amplitude' in df.columns:
            # FIX: Use expanding min/max instead of global min/max
            # This ensures we only compare to past history, not future data
            expanding_min = df['amplitude'].expanding(min_periods=50).min()
            expanding_max = df['amplitude'].expanding(min_periods=50).max()
            
            # Avoid division by zero
            denom = expanding_max - expanding_min
            denom = denom.replace(0, 1)  # Replace 0 with 1 to avoid NaN
            
            amp_norm = (df['amplitude'] - expanding_min) / denom
            strength = amp_norm * 100
            
            # Adjust for phase stability
            if 'phase' in df.columns:
                phase_volatility = df['phase'].rolling(window=5).std()
                # Use expanding max for volatility normalization too
                max_vol = phase_volatility.expanding(min_periods=50).max().replace(0, 1)
                
                stability_factor = 1 - (phase_volatility / max_vol).fillna(0.5)
                strength = strength * stability_factor
        
        return strength.fillna(50.0)
    
    def _calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence level for signals
        """
        confidence = pd.Series('MEDIUM', index=df.index)
        
        if 'signal_strength' in df.columns:
            confidence[df['signal_strength'] >= 70] = 'HIGH'
            confidence[df['signal_strength'] <= 30] = 'LOW'
        
        return confidence
    
    def get_latest_signal(self, df: pd.DataFrame) -> Dict:
        """Get the latest trading signal with details"""
        latest_row = df.iloc[-1]
        
        signal_info = {
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'signal': latest_row['signal'],
            'position': 'LONG' if latest_row.get('position', 0) == 1 else 'FLAT',
            'phase_quadrant': latest_row.get('phase_quadrant', 'Unknown'),
            'phase_value': float(latest_row.get('phase', 0)),
            'oscillator_value': float(latest_row.get('oscillator', 0)),
            'signal_strength': float(latest_row.get('signal_strength', 50)),
            'confidence': latest_row.get('confidence', 'MEDIUM'),
            'price': float(latest_row['close']),
            'timestamp': datetime.now().isoformat()
        }
        
        self.signal_history.append(signal_info)
        self.last_signal = signal_info
        
        return signal_info
    
    def save_signals(self, df: pd.DataFrame) -> str:
        """Save signals to CSV and latest signal to JSON inside the data_path directory."""
        csv_filepath = os.path.join(self.data_path, 'signals.csv')
        
        columns_to_save = [
            'open', 'high', 'low', 'close', 'volume',
            'oscillator', 'phase', 'amplitude', 'phase_quadrant',
            'signal', 'position', 'signal_strength', 'confidence'
        ]
        columns_to_save = [col for col in columns_to_save if col in df.columns]
        
        df[columns_to_save].to_csv(csv_filepath)
        print(f"💾 Signals saved to {csv_filepath}")
        
        latest_signal = self.get_latest_signal(df)
        json_filepath = os.path.join(self.data_path, 'signals_latest.json')
        with open(json_filepath, 'w') as f:
            json.dump(latest_signal, f, indent=2)
        print(f"📄 Latest signal saved to {json_filepath}")
        
        return csv_filepath

    def load_signals(self) -> pd.DataFrame:
        """Load signals from CSV file inside the data_path directory."""
        filepath = os.path.join(self.data_path, 'signals.csv')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Signals file not found: {filepath}")
        
        df = pd.read_csv(filepath, index_col='date', parse_dates=True)
        print(f"📂 Loaded signals from {filepath}")
        return df

    def generate_alert_message(self, signal_info: Dict) -> str:
        """Generate a formatted alert message for notifications"""
        emoji_map = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⏸️'}
        confidence_emoji = {'HIGH': '⭐⭐⭐', 'MEDIUM': '⭐⭐', 'LOW': '⭐'}
        
        message = f"""
{emoji_map.get(signal_info['signal'], '❓')} **SIGNAL ALERT - {Config.TICKER}**
------------------------
📅 Date: {signal_info['date']}
📊 Signal: **{signal_info['signal']}**
💰 Price: ${signal_info['price']:.2f}
📈 Position: {signal_info['position']}

**Cycle Analysis:**
🔄 Phase: {signal_info['phase_quadrant']}
📐 Phase Value: {signal_info['phase_value']:.2f} rad
📊 Oscillator: {signal_info['oscillator_value']:.4f}

**Signal Quality:**
💪 Strength: {signal_info['signal_strength']:.1f}/100
{confidence_emoji.get(signal_info['confidence'], '')} Confidence: {signal_info['confidence']}

⏰ Generated: {signal_info['timestamp']}
"""
        return message
