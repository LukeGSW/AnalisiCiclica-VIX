"""
Notification module for Kriterion Quant Trading System
Enhanced Telegram notifications with professional formatting
"""

import requests
import json
from typing import Dict, Optional
from datetime import datetime

from config import Config


class TelegramNotifier:
    """Class to handle Telegram notifications with professional formatting"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        """
        Initialize the Telegram notifier
        
        Parameters
        ----------
        bot_token : str, optional
            Telegram bot token. Defaults to Config.TELEGRAM_BOT_TOKEN
        chat_id : str, optional
            Telegram chat ID. Defaults to Config.TELEGRAM_CHAT_ID
        """
        self.bot_token = bot_token or Config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        
        if not self.bot_token or not self.chat_id:
            print("⚠️ Telegram credentials not configured. Notifications disabled.")
            self.enabled = False
        else:
            self.enabled = True
            self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message via Telegram
        
        Parameters
        ----------
        message : str
            Message to send (HTML formatted)
        parse_mode : str, optional
            Parse mode for formatting. Default is "HTML"
        
        Returns
        -------
        bool
            True if message sent successfully
        """
        if not self.enabled:
            print("❌ Telegram notifications are disabled")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print("✅ Telegram notification sent successfully")
                return True
            else:
                print(f"❌ Failed to send Telegram notification: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error sending Telegram notification: {e}")
            return False
    
    def _get_signal_emoji(self, signal: str) -> str:
        """Get appropriate emoji for signal type"""
        emoji_map = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '⏸️'
        }
        return emoji_map.get(signal, '❓')
    
    def _get_confidence_stars(self, confidence: str) -> str:
        """Get star rating for confidence level"""
        stars_map = {
            'HIGH': '⭐⭐⭐',
            'MEDIUM': '⭐⭐☆',
            'LOW': '⭐☆☆'
        }
        return stars_map.get(confidence, '☆☆☆')
    
    def _get_strength_bar(self, strength: float) -> str:
        """Create visual progress bar for signal strength"""
        filled = int(strength / 10)
        empty = 10 - filled
        return '▓' * filled + '░' * empty
    
    def _format_price(self, price: float) -> str:
        """Format price with proper styling"""
        return f"${price:,.2f}"
    
    def _format_percentage(self, value: float, include_sign: bool = True) -> str:
        """Format percentage with proper styling"""
        if include_sign and value > 0:
            return f"+{value:.2f}%"
        return f"{value:.2f}%"
    
    def send_signal_alert(self, signal_info: Dict) -> bool:
        """
        Send a professionally formatted signal alert
        
        Parameters
        ----------
        signal_info : Dict
            Signal information dictionary
        
        Returns
        -------
        bool
            True if alert sent successfully
        """
        if not self.enabled:
            return False
        
        signal = signal_info['signal']
        signal_emoji = self._get_signal_emoji(signal)
        confidence_stars = self._get_confidence_stars(signal_info['confidence'])
        strength_bar = self._get_strength_bar(float(signal_info['signal_strength']))
        
        # Determine action and styling based on signal
        if signal == 'BUY':
            header_emoji = "🚀"
            action_text = "ENTER LONG POSITION"
            action_desc = "Hedging signal detected - Consider VIX exposure"
        elif signal == 'SELL':
            header_emoji = "📉"
            action_text = "EXIT LONG POSITION"
            action_desc = "Cycle peak detected - Consider removing hedge"
        else:
            header_emoji = "⏸️"
            action_text = "MAINTAIN POSITION"
            action_desc = "No regime change detected"
        
        # Extract phase info
        phase_quadrant = signal_info.get('phase_quadrant', 'Unknown')
        if '(' in str(phase_quadrant):
            phase_short = str(phase_quadrant).split('(')[1].replace(')', '')
        else:
            phase_short = str(phase_quadrant)
        
        message = f"""
{header_emoji} <b>KRITERION QUANT SIGNAL</b>
━━━━━━━━━━━━━━━━━━━━━━

{signal_emoji} <b>Signal:</b> <code>{signal}</code>
📊 <b>Asset:</b> {Config.TICKER}
💵 <b>Price:</b> <code>{self._format_price(float(signal_info['price']))}</code>
📅 <b>Date:</b> {signal_info['date']}

<b>┌─ CYCLE ANALYSIS ─────────┐</b>
│ 🔄 Phase: <code>{phase_short}</code>
│ 📐 Value: <code>{float(signal_info['phase_value']):.3f} rad</code>
│ 📈 Oscillator: <code>{float(signal_info['oscillator_value']):.4f}</code>
<b>└─────────────────────────┘</b>

<b>┌─ SIGNAL QUALITY ─────────┐</b>
│ 💪 Strength: <code>{float(signal_info['signal_strength']):.0f}/100</code>
│ {strength_bar}
│ {confidence_stars} <b>{signal_info['confidence']}</b> confidence
<b>└─────────────────────────┘</b>

<b>🎯 ACTION: {action_text}</b>
<i>{action_desc}</i>

⏰ <i>Generated: {datetime.now().strftime('%H:%M:%S UTC')}</i>
"""
        
        return self.send_message(message.strip())
    
    def send_backtest_summary(self, metrics: Dict) -> bool:
        """
        Send a backtest performance summary
        
        Parameters
        ----------
        metrics : Dict
            Backtest metrics dictionary
        
        Returns
        -------
        bool
            True if summary sent successfully
        """
        if not self.enabled:
            return False
        
        total_return = float(metrics.get('total_return_%', 0))
        return_emoji = "📈" if total_return >= 0 else "📉"
        
        sharpe = float(metrics.get('sharpe_ratio', 0))
        sharpe_rating = "🌟 Excellent" if sharpe >= 2 else "✅ Good" if sharpe >= 1 else "⚠️ Below average" if sharpe >= 0 else "❌ Negative"
        
        message = f"""
📊 <b>BACKTEST PERFORMANCE REPORT</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Asset:</b> {Config.TICKER}
<b>Strategy:</b> Hilbert Cycle Analysis

<b>┌─ RETURNS ────────────────┐</b>
│ {return_emoji} Total: <code>{self._format_percentage(total_return)}</code>
│ 📉 Max DD: <code>{self._format_percentage(float(metrics.get('max_drawdown_%', 0)), False)}</code>
<b>└─────────────────────────┘</b>

<b>┌─ RISK METRICS ───────────┐</b>
│ 📐 Sharpe: <code>{sharpe:.2f}</code> {sharpe_rating}
│ 📏 Sortino: <code>{float(metrics.get('sortino_ratio', 0)):.2f}</code>
│ 📊 Calmar: <code>{float(metrics.get('calmar_ratio', 0)):.2f}</code>
<b>└─────────────────────────┘</b>

<b>┌─ TRADING STATS ──────────┐</b>
│ 🔄 Trades: <code>{int(metrics.get('total_trades', 0))}</code>
│ ✅ Win Rate: <code>{float(metrics.get('win_rate_%', 0)):.1f}%</code>
│ 💰 Profit Factor: <code>{float(metrics.get('profit_factor', 0)):.2f}</code>
<b>└─────────────────────────┘</b>

⏰ <i>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</i>
"""
        
        return self.send_message(message.strip())
    
    def send_error_alert(self, error_message: str) -> bool:
        """
        Send an error alert with professional formatting
        
        Parameters
        ----------
        error_message : str
            Error message to send
        
        Returns
        -------
        bool
            True if alert sent successfully
        """
        if not self.enabled:
            return False
        
        message = f"""
🚨 <b>SYSTEM ERROR ALERT</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Asset:</b> {Config.TICKER}
<b>Component:</b> Kriterion Quant Engine

<b>┌─ ERROR DETAILS ──────────┐</b>
<code>{error_message[:500]}</code>
<b>└─────────────────────────┘</b>

<b>⚠️ Action Required</b>
Please check system logs for detailed diagnostics.

⏰ <i>Alert time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
        
        return self.send_message(message.strip())
    
    def send_daily_summary(self, signal_info: Dict, metrics: Dict) -> bool:
        """
        Send a comprehensive daily summary with professional formatting
        
        Parameters
        ----------
        signal_info : Dict
            Latest signal information
        metrics : Dict
            Latest backtest metrics
        
        Returns
        -------
        bool
            True if summary sent successfully
        """
        if not self.enabled:
            return False
        
        # Position status
        position = signal_info['position']
        if position == 'LONG':
            position_emoji = "🟢"
            position_status = "ACTIVE HEDGE"
            next_action = "📍 Monitoring for exit signal"
        else:
            position_emoji = "⚪"
            position_status = "NO POSITION"
            next_action = "👀 Watching for entry opportunity"
        
        # Phase info
        phase_quadrant = signal_info.get('phase_quadrant', 'Unknown')
        if '(' in str(phase_quadrant):
            phase_short = str(phase_quadrant).split('(')[1].replace(')', '')
        else:
            phase_short = str(phase_quadrant)
        
        # Determine cycle regime
        is_bullish = 'Salita' in str(phase_quadrant) or 'Minimo' in str(phase_quadrant)
        regime_emoji = "🌅" if is_bullish else "🌆"
        regime_text = "Bullish Cycle Phase" if is_bullish else "Bearish Cycle Phase"
        
        # Performance indicators
        total_return = float(metrics.get('total_return_%', 0))
        sharpe = float(metrics.get('sharpe_ratio', 0))
        
        # Visual strength indicator
        strength_bar = self._get_strength_bar(float(signal_info['signal_strength']))
        confidence_stars = self._get_confidence_stars(signal_info['confidence'])
        
        message = f"""
📰 <b>DAILY MARKET REPORT</b>
━━━━━━━━━━━━━━━━━━━━━━
<i>{datetime.now().strftime('%A, %B %d, %Y')}</i>

<b>┌─ POSITION STATUS ────────┐</b>
│ {position_emoji} <b>{position_status}</b>
│ 📊 Asset: {Config.TICKER}
│ 💵 Price: <code>{self._format_price(float(signal_info['price']))}</code>
│ 📅 Last Signal: <code>{signal_info['signal']}</code> ({signal_info['date']})
<b>└─────────────────────────┘</b>

<b>┌─ CYCLE STATUS ───────────┐</b>
│ {regime_emoji} <b>{regime_text}</b>
│ 🔄 Phase: <code>{phase_short}</code>
│ 📐 Value: <code>{float(signal_info['phase_value']):.3f} rad</code>
│ 
│ 💪 Strength: {float(signal_info['signal_strength']):.0f}/100
│ {strength_bar}
│ {confidence_stars} {signal_info['confidence']}
<b>└─────────────────────────┘</b>

<b>┌─ STRATEGY PERFORMANCE ───┐</b>
│ 📈 Return: <code>{self._format_percentage(total_return)}</code>
│ 📐 Sharpe: <code>{sharpe:.2f}</code>
│ 📉 Max DD: <code>{self._format_percentage(float(metrics.get('max_drawdown_%', 0)), False)}</code>
│ ✅ Win Rate: <code>{float(metrics.get('win_rate_%', 0)):.1f}%</code>
<b>└─────────────────────────┘</b>

<b>🎯 NEXT STEPS</b>
{next_action}

━━━━━━━━━━━━━━━━━━━━━━
<i>Kriterion Quant • VIX Hedging System</i>
<i>⏰ {datetime.now().strftime('%H:%M UTC')}</i>
"""
        
        return self.send_message(message.strip())
    
    def send_startup_notification(self) -> bool:
        """
        Send a system startup notification
        
        Returns
        -------
        bool
            True if notification sent successfully
        """
        if not self.enabled:
            return False
        
        message = f"""
🚀 <b>SYSTEM STARTUP</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Kriterion Quant Engine</b> initialized

<b>Configuration:</b>
• Asset: <code>{Config.TICKER}</code>
• Fast MA: <code>{Config.FAST_MA_WINDOW}</code>
• Slow MA: <code>{Config.SLOW_MA_WINDOW}</code>
• Capital: <code>${float(Config.INITIAL_CAPITAL):,.0f}</code>

✅ All systems operational

⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
        
        return self.send_message(message.strip())
    
    def test_connection(self) -> bool:
        """
        Test the Telegram connection with a styled message
        
        Returns
        -------
        bool
            True if connection successful
        """
        if not self.enabled:
            print("❌ Telegram notifications are disabled")
            return False
        
        message = f"""
✅ <b>CONNECTION TEST SUCCESSFUL</b>
━━━━━━━━━━━━━━━━━━━━━━

<b>Kriterion Quant</b> bot is connected and operational.

<b>Monitoring:</b> {Config.TICKER}
<b>Status:</b> 🟢 Online

<i>You will receive:</i>
• 🔔 Trading signals (BUY/SELL)
• 📊 Daily summaries
• 🚨 Error alerts

⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>
"""
        
        return self.send_message(message.strip())
