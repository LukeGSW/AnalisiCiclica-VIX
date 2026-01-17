# app.py

"""
Streamlit Dashboard for Kriterion Quant Trading System
Enhanced UI/UX Version - Professional Grade
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import sys

# Add src directory to path
sys.path.insert(0, 'src')

from config import Config
from data_fetcher import DataFetcher
from cycle_analyzer import CycleAnalyzer
from signal_generator import SignalGenerator
from backtester import Backtester

# Page configuration
st.set_page_config(
    page_title=f"Kriterion Quant - {Config.TICKER}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================ #
#                           PROFESSIONAL CSS STYLING                            #
# ============================================================================ #
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.7);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Status Cards */
    .status-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .status-card-bullish {
        background: linear-gradient(145deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
    }
    
    .status-card-bearish {
        background: linear-gradient(145deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
    }
    
    .status-card-neutral {
        background: linear-gradient(145deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 4px solid #ffc107;
    }
    
    /* Metric Cards Enhancement */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }
    
    [data-testid="metric-container"] > div:first-child {
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="metric-container"] > div:nth-child(2) {
        color: #1e293b !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    
    [data-testid="metric-container"] > div:nth-child(3) {
        color: #64748b !important;
        font-size: 0.8rem !important;
    }
    
    /* Signal Badge Styles */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(107, 114, 128, 0.4);
    }
    
    /* Position Badge */
    .position-long {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    
    .position-flat {
        background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Info Box Styling */
    .info-box {
        background: linear-gradient(145deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .info-box-warning {
        background: linear-gradient(145deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
    }
    
    .info-box-success {
        background: linear-gradient(145deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #6ee7b7;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #64748b;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #1e293b !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #cbd5e1 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Data Table Styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Progress Indicator */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Custom Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Confidence Indicator */
    .confidence-high {
        color: #10b981;
        font-weight: 700;
    }
    
    .confidence-medium {
        color: #f59e0b;
        font-weight: 700;
    }
    
    .confidence-low {
        color: #ef4444;
        font-weight: 700;
    }
    
    /* Phase Quadrant Colors */
    .quadrant-bullish {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .quadrant-bearish {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================ #
#                              COLOR PALETTE                                    #
# ============================================================================ #
COLORS = {
    'primary': '#3b82f6',
    'primary_dark': '#2563eb',
    'success': '#10b981',
    'success_dark': '#059669',
    'danger': '#ef4444',
    'danger_dark': '#dc2626',
    'warning': '#f59e0b',
    'neutral': '#64748b',
    'background': '#f8fafc',
    'text_primary': '#1e293b',
    'text_secondary': '#64748b',
    'chart_bullish': '#10b981',
    'chart_bearish': '#ef4444',
    'chart_price': '#3b82f6',
    'chart_ma_fast': '#8b5cf6',
    'chart_ma_slow': '#f59e0b',
    'chart_grid': '#e2e8f0',
}

# ============================================================================ #
#                           HELPER FUNCTIONS                                    #
# ============================================================================ #

def get_available_tickers(path='data'):
    """Scansiona la directory dati e restituisce un elenco di ticker disponibili."""
    if not os.path.exists(path):
        return []
    tickers = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return sorted(tickers)

@st.cache_data(ttl=600)
def load_data_from_path(ticker_path):
    """Carica tutti i dati necessari da un percorso specifico."""
    if not os.path.exists(ticker_path):
        return None

    data = {}
    try:
        signals_file = os.path.join(ticker_path, 'signals.csv')
        df = pd.read_csv(signals_file, index_col='date')
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_localize(None)
        data['signals'] = df

        latest_signal_file = os.path.join(ticker_path, 'signals_latest.json')
        with open(latest_signal_file, 'r') as f:
            data['latest_signal'] = json.load(f)

        backtest_file = os.path.join(ticker_path, 'backtest_results.json')
        with open(backtest_file, 'r') as f:
            data['backtest'] = json.load(f)

        summary_file = os.path.join(ticker_path, 'analysis_summary.json')
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)

    except Exception as e:
        st.error(f"Errore nel caricamento dei dati da {ticker_path}: {e}")
        return None

    return data


def run_analysis_for_ticker(ticker, lookback_years=20):
    """Run the complete analysis pipeline for a specific ticker."""
    try:
        with st.spinner(f'🔄 Running analysis for {ticker} with {lookback_years} years lookback...'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            ticker_data_path = os.path.join('data', ticker)
            os.makedirs(ticker_data_path, exist_ok=True)
            
            fetcher = DataFetcher(data_path=ticker_data_path)
            analyzer = CycleAnalyzer()
            generator = SignalGenerator(data_path=ticker_data_path)
            backtester = Backtester(data_path=ticker_data_path)
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_years*365.25)).strftime('%Y-%m-%d')
            
            status_text.text(f'📡 Fetching data for {ticker}...')
            progress_bar.progress(20)
            df = fetcher.fetch_historical_data(ticker=ticker, start=start_date, end=end_date)
            fetcher.save_data(df)
            
            status_text.text('🔄 Performing cycle analysis...')
            progress_bar.progress(40)
            df_analyzed = analyzer.analyze_cycle(df)
            
            status_text.text('🎯 Generating trading signals...')
            progress_bar.progress(60)
            df_signals = generator.generate_signals(df_analyzed)
            generator.save_signals(df_signals)
            
            status_text.text('📊 Running backtest...')
            progress_bar.progress(80)
            wf_results = backtester.run_walk_forward_analysis(df_signals)
            backtester.save_backtest_results(wf_results)
            
            status_text.text('📄 Creating summary...')
            latest_signal = generator.get_latest_signal(df_signals)
            spectral_results = analyzer.run_spectral_analysis(df_analyzed['oscillator'])
            monte_carlo_results = analyzer.run_monte_carlo_significance_test(df_analyzed['oscillator'])
            
            summary = {
                'timestamp': datetime.now().isoformat(), 'ticker': ticker,
                'lookback_years': lookback_years, 'data_points': len(df_signals),
                'date_range': {'start': df_signals.index[0].strftime('%Y-%m-%d'), 'end': df_signals.index[-1].strftime('%Y-%m-%d')},
                'latest_signal': latest_signal,
                'cycle_analysis': {
                    'dominant_period': float(spectral_results['dominant_period']) if spectral_results['dominant_period'] else None,
                    'p_value': float(monte_carlo_results['p_value']),
                    'significant': bool(monte_carlo_results['significant'])
                }
            }
            summary_file = os.path.join(ticker_data_path, 'analysis_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            progress_bar.progress(100)
            status_text.text(f'✅ Analysis for {ticker} complete!')
            return True, f"Analysis for {ticker} completed successfully!"
            
    except Exception as e:
        return False, f"Error during analysis for {ticker}: {e}"


def load_data():
    """Load all necessary data files"""
    data = {}
    
    signals_file = Config.SIGNALS_FILE
    if os.path.exists(signals_file):
        df = pd.read_csv(signals_file, index_col='date')
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_localize(None)
        data['signals'] = df
    else:
        return None
    
    latest_signal_file = signals_file.replace('.csv', '_latest.json')
    if os.path.exists(latest_signal_file):
        with open(latest_signal_file, 'r') as f:
            data['latest_signal'] = json.load(f)
    
    if os.path.exists(Config.BACKTEST_RESULTS_FILE):
        with open(Config.BACKTEST_RESULTS_FILE, 'r') as f:
            data['backtest'] = json.load(f)
    
    summary_file = os.path.join(Config.DATA_DIR, 'analysis_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)
    
    return data


# ============================================================================ #
#                           CHART FUNCTIONS                                     #
# ============================================================================ #

def create_price_chart(df):
    """Create professional interactive price chart with signals"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            '<b>Price Action & Trading Signals</b>',
            '<b>Cycle Oscillator</b>',
            '<b>Phase Indicator</b>'
        )
    )
    
    # Price line with gradient fill
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['close'], 
            name='Close Price',
            line=dict(color=COLORS['chart_price'], width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ), 
        row=1, col=1
    )
    
    # Buy signals with better visibility
    buy_signals = df[df['signal'] == 'BUY']
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index, 
                y=buy_signals['close'], 
                mode='markers',
                name='🟢 Buy Signal',
                marker=dict(
                    symbol='triangle-up',
                    size=14,
                    color=COLORS['success'],
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>BUY SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ), 
            row=1, col=1
        )
    
    # Sell signals with better visibility
    sell_signals = df[df['signal'] == 'SELL']
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index, 
                y=sell_signals['close'], 
                mode='markers',
                name='🔴 Sell Signal',
                marker=dict(
                    symbol='triangle-down',
                    size=14,
                    color=COLORS['danger'],
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>SELL SIGNAL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ), 
            row=1, col=1
        )
    
    # Oscillator with positive/negative coloring
    if 'oscillator' in df.columns:
        colors = [COLORS['success'] if val >= 0 else COLORS['danger'] for val in df['oscillator']]
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['oscillator'], 
                name='Oscillator',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='<b>Oscillator:</b> %{y:.4f}<extra></extra>'
            ), 
            row=2, col=1
        )
        fig.add_hline(y=0, row=2, col=1, line_dash="solid", line_color=COLORS['neutral'], line_width=1)
    
    # Phase indicator with quadrant shading
    if 'phase' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=df['phase'], 
                name='Phase',
                line=dict(color=COLORS['chart_ma_fast'], width=2),
                hovertemplate='<b>Phase:</b> %{y:.2f} rad<extra></extra>'
            ), 
            row=3, col=1
        )
        
        # Add quadrant reference lines
        for y_val in [np.pi/2, 0, -np.pi/2]:
            fig.add_hline(
                y=y_val, row=3, col=1, 
                line_dash="dot", 
                line_color=COLORS['chart_grid'], 
                line_width=1,
                opacity=0.7
            )
        
        # Add bullish/bearish zones
        fig.add_hrect(
            y0=-np.pi, y1=0, row=3, col=1,
            fillcolor=COLORS['success'], opacity=0.1,
            line_width=0
        )
        fig.add_hrect(
            y0=0, y1=np.pi, row=3, col=1,
            fillcolor=COLORS['danger'], opacity=0.1,
            line_width=0
        )
    
    # Update layout with professional styling
    fig.update_layout(
    height=750,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(30,41,59,0.95)",
        bordercolor="#334155",
        borderwidth=1,
        font=dict(size=11, color="#e2e8f0")
    ),
    hovermode='x unified',
    template='plotly_dark',
    margin=dict(l=60, r=40, t=80, b=40),
    paper_bgcolor='#0f172a',
    plot_bgcolor='#1e293b',
    font=dict(family="Inter, sans-serif", color="#e2e8f0")
)
    
    # Update axes styling
    fig.update_xaxes(
    title_text="Date", 
    row=3, col=1,
    gridcolor='#334155',
    showgrid=True,
    zeroline=False,
    title_font=dict(color="#94a3b8"),
    tickfont=dict(color="#94a3b8")
)
)
    )
    fig.update_yaxes(
        title_text="Price ($)", 
        row=1, col=1,
        gridcolor=COLORS['chart_grid'],
        showgrid=True,
        zeroline=False,
        tickformat='$,.2f'
    )
    fig.update_yaxes(
        title_text="Oscillator", 
        row=2, col=1,
        gridcolor=COLORS['chart_grid'],
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(
        title_text="Phase (rad)", 
        row=3, col=1,
        gridcolor=COLORS['chart_grid'],
        showgrid=True,
        zeroline=False,
        tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        ticktext=['-π', '-π/2', '0', 'π/2', 'π']
    )
    
    # Update subplot titles styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color=COLORS['text_primary'], family="Inter, sans-serif")
    
    return fig


def create_equity_chart(df_results: pd.DataFrame):
    """Create professional equity curve chart from backtest results."""
    fig = go.Figure()
    
    if 'equity' in df_results.columns:
        fig.add_trace(
            go.Scatter(
                x=df_results.index, 
                y=df_results['equity'], 
                name='Strategy',
                line=dict(color=COLORS['primary'], width=2.5),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.15)',
                hovertemplate='<b>Strategy</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            )
        )
    
    if 'benchmark_equity' in df_results.columns:
        fig.add_trace(
            go.Scatter(
                x=df_results.index, 
                y=df_results['benchmark_equity'], 
                name='Buy & Hold',
                line=dict(color=COLORS['neutral'], width=2, dash='dash'),
                hovertemplate='<b>Buy & Hold</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            )
        )
    
    fig.update_layout(
        title=dict(
            text='<b>Equity Curve Comparison</b>',
            font=dict(size=16, color=COLORS['text_primary'], family="Inter, sans-serif"),
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=400,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS['chart_grid'],
            borderwidth=1
        ),
        margin=dict(l=60, r=40, t=80, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,0.5)',
        font=dict(family="Inter, sans-serif", color=COLORS['text_primary'])
    )
    
    fig.update_xaxes(gridcolor=COLORS['chart_grid'], showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor=COLORS['chart_grid'], showgrid=True, zeroline=False, tickformat='$,.0f')
    
    return fig


def create_phase_distribution_chart(df):
    """Create professional phase distribution pie chart."""
    phase_counts = df['phase_quadrant'].value_counts()
    
    # Define colors for each quadrant
    colors = []
    for label in phase_counts.index:
        if 'Salita' in str(label) or 'Minimo' in str(label):
            colors.append(COLORS['success'])
        else:
            colors.append(COLORS['danger'])
    
    fig = go.Figure(data=[
        go.Pie(
            labels=[str(l).split('(')[1].replace(')', '') if '(' in str(l) else str(l) for l in phase_counts.index],
            values=phase_counts.values.tolist(),
            hole=0.5,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='percent+label',
            textfont=dict(size=11),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='<b>Cycle Phase Distribution</b>',
            font=dict(size=14, color=COLORS['text_primary'], family="Inter, sans-serif"),
            x=0.5
        ),
        height=320,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=COLORS['text_primary']),
        annotations=[
            dict(
                text='<b>Phase</b>',
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False,
                font=dict(color=COLORS['text_secondary'])
            )
        ]
    )
    
    return fig


def create_signal_strength_gauge(strength_value):
    """Create a gauge chart for signal strength."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=strength_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%', 'font': {'size': 28, 'color': COLORS['text_primary'], 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['text_secondary']},
            'bar': {'color': COLORS['primary']},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': COLORS['chart_grid'],
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {
                'line': {'color': COLORS['text_primary'], 'width': 2},
                'thickness': 0.75,
                'value': strength_value
            }
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color=COLORS['text_primary'])
    )
    
    return fig


def render_signal_badge(signal):
    """Render a styled signal badge."""
    if signal == 'BUY':
        return '<span class="signal-badge signal-buy">🟢 BUY</span>'
    elif signal == 'SELL':
        return '<span class="signal-badge signal-sell">🔴 SELL</span>'
    else:
        return '<span class="signal-badge signal-hold">⏸️ HOLD</span>'


def render_position_badge(position):
    """Render a styled position badge."""
    if position == 'LONG':
        return '<span class="position-long">💰 LONG</span>'
    else:
        return '<span class="position-flat">💤 FLAT</span>'


def render_confidence_badge(confidence):
    """Render a styled confidence indicator."""
    if confidence == 'HIGH':
        return '<span class="confidence-high">⭐⭐⭐ HIGH</span>'
    elif confidence == 'MEDIUM':
        return '<span class="confidence-medium">⭐⭐ MEDIUM</span>'
    else:
        return '<span class="confidence-low">⭐ LOW</span>'


# ============================================================================ #
#                              MAIN DASHBOARD                                   #
# ============================================================================ #

def main():
    """Main dashboard function"""
    
    # ==================== SIDEBAR ==================== #
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #f1f5f9; margin-bottom: 0.5rem;">⚙️ Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        available_tickers = get_available_tickers()
        
        if not available_tickers:
            st.warning("⚠️ No data found. Run an analysis via GitHub Actions first.")
            st.stop()
        
        st.markdown("### 📈 Select Asset")
        selected_ticker = st.selectbox(
            "Ticker Symbol",
            available_tickers,
            label_visibility="collapsed"
        )
    
    # Load data for selected ticker
    ticker_data_path = os.path.join('data', selected_ticker)
    data = load_data_from_path(ticker_data_path)
    
    # ==================== MAIN HEADER ==================== #
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Kriterion Quant Trading System</h1>
        <p>Cycle-Based VIX Hedging Strategy • <b>{selected_ticker}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if data is None:
        st.warning(f"⚠️ Data for {selected_ticker} is missing or corrupted. Please re-run the analysis.")
        return
    
    df_signals = data['signals']
    
    # ==================== SIDEBAR CONTINUED ==================== #
    with st.sidebar:
        summary = data.get('summary', {})
        current_lookback = summary.get('lookback_years', 20)
        date_range_info = summary.get('date_range', {})
        
        st.markdown("### 📊 Current Settings")
        st.markdown(f"""
        <div class="info-box" style="background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.2);">
            <p style="color: #e2e8f0; margin: 0; font-size: 0.9rem;">
            <b>Lookback:</b> {current_lookback} years<br>
            <b>Range:</b> {date_range_info.get('start', 'N/A')} → {date_range_info.get('end', 'N/A')}<br>
            <b>Fast MA:</b> {Config.FAST_MA_WINDOW} days<br>
            <b>Slow MA:</b> {Config.SLOW_MA_WINDOW} days<br>
            <b>Capital:</b> ${float(Config.INITIAL_CAPITAL):,.0f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔄 Update Analysis")
        
        new_lookback = st.slider(
            "Lookback (years)", 
            1, 20, 
            value=current_lookback, 
            step=1,
            help="Number of years of historical data to analyze"
        )
        
        if new_lookback != current_lookback:
            new_start = (datetime.now() - timedelta(days=new_lookback*365.25)).strftime('%Y-%m-%d')
            st.caption(f"📅 New range: {new_start} → Today")
        
        if st.button("🚀 Run New Analysis", use_container_width=True):
            success, message = run_analysis_for_ticker(selected_ticker, new_lookback)
            if success:
                st.success(message)
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(message)
        
        # Cycle Analysis Summary
        if 'cycle_analysis' in summary:
            st.markdown("---")
            st.markdown("### 🔬 Cycle Analysis")
            cycle_info = summary['cycle_analysis']
            
            if cycle_info.get('dominant_period'):
                st.metric(
                    "Dominant Cycle", 
                    f"{float(cycle_info['dominant_period']):.1f} days"
                )
            
            p_val = float(cycle_info.get('p_value', 1))
            is_sig = cycle_info.get('significant', False)
            st.metric(
                "Significance", 
                f"p = {p_val:.4f}",
                "✅ Valid" if is_sig else "⚠️ Weak"
            )
        
        st.markdown("---")
        st.markdown("### 📅 Date Filter")
        
        date_range = st.date_input(
            "Select range",
            value=(df_signals.index[0].date(), df_signals.index[-1].date()),
            min_value=df_signals.index[0].date(),
            max_value=df_signals.index[-1].date(),
            key="date_filter_widget"
        )
        
        if len(date_range) == 2:
            start_ts = pd.Timestamp(date_range[0])
            end_ts = pd.Timestamp(date_range[1])
            mask = (df_signals.index >= start_ts) & (df_signals.index <= end_ts)
            df_filtered = df_signals.loc[mask]
        else:
            df_filtered = df_signals
    
    # ==================== MAIN CONTENT TABS ==================== #
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Current Status", 
        "📈 Cycle Analysis", 
        "🎯 Backtest Results", 
        "📋 Signal History"
    ])
    
    # ==================== TAB 1: CURRENT STATUS ==================== #
    with tab1:
        if 'latest_signal' in data:
            latest = data['latest_signal']
            
            # Signal Overview Cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                signal_color = "#10b981" if latest['signal'] == 'BUY' else "#ef4444" if latest['signal'] == 'SELL' else "#64748b"
                st.markdown(f"""
                <div class="status-card" style="border-left: 4px solid {signal_color};">
                    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 600;">LAST SIGNAL</p>
                    {render_signal_badge(latest['signal'])}
                    <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.75rem;">📅 {latest['date']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="status-card">
                    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 600;">POSITION</p>
                    {render_position_badge(latest['position'])}
                    <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.75rem;">💵 ${float(latest['price']):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="status-card">
                    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 600;">SIGNAL STRENGTH</p>
                    <p style="font-size: 1.75rem; font-weight: 700; color: #1e293b; font-family: 'JetBrains Mono', monospace; margin: 0;">
                        {float(latest['signal_strength']):.0f}<span style="font-size: 1rem; color: #64748b;">/100</span>
                    </p>
                    <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem;">{render_confidence_badge(latest['confidence'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                quadrant_text = str(latest['phase_quadrant']).split('(')[1].replace(')', '') if '(' in str(latest['phase_quadrant']) else str(latest['phase_quadrant'])
                is_bullish = 'Salita' in str(latest['phase_quadrant']) or 'Minimo' in str(latest['phase_quadrant'])
                quadrant_class = 'quadrant-bullish' if is_bullish else 'quadrant-bearish'
                st.markdown(f"""
                <div class="status-card">
                    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 600;">CYCLE PHASE</p>
                    <span class="{quadrant_class}">{quadrant_text}</span>
                    <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.75rem;">📐 {float(latest['phase_value']):.2f} rad</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            # Signal Details
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.markdown("### 📍 Signal Details")
                
                detail_col1, detail_col2 = st.columns(2)
                with detail_col1:
                    st.markdown(f"""
                    <div class="info-box">
                        <p style="margin: 0; color: #334155;">
                        <b>Oscillator Value:</b> {float(latest['oscillator_value']):.4f}<br>
                        <b>Phase Quadrant:</b> {latest['phase_quadrant']}<br>
                        <b>Generated:</b> {latest['timestamp'][:19]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with detail_col2:
                    if latest['signal'] == 'BUY':
                        st.markdown("""
                        <div class="info-box info-box-success">
                            <p style="margin: 0; color: #065f46; font-weight: 600;">
                            🎯 <b>Recommended Action</b><br>
                            Enter Long Position (Hedging Exposure)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif latest['signal'] == 'SELL':
                        st.markdown("""
                        <div class="info-box info-box-warning">
                            <p style="margin: 0; color: #92400e; font-weight: 600;">
                            🎯 <b>Recommended Action</b><br>
                            Exit Long Position (Remove Hedge)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                            <p style="margin: 0; color: #334155; font-weight: 600;">
                            🎯 <b>Recommended Action</b><br>
                            Maintain Current Position
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_right:
                st.markdown("### 💪 Signal Strength")
                fig_gauge = create_signal_strength_gauge(float(latest['signal_strength']))
                st.plotly_chart(fig_gauge, use_container_width=True)
    
    # ==================== TAB 2: CYCLE ANALYSIS ==================== #
    with tab2:
        st.markdown("### 📈 Price Chart with Signals")
        fig_price = create_price_chart(df_filtered)
        st.plotly_chart(fig_price, use_container_width=True)
        
        if 'phase_quadrant' in df_filtered.columns:
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig_pie = create_phase_distribution_chart(df_filtered)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("### 📊 Signal Statistics")
                
                total_signals = len(df_filtered[df_filtered['signal'] != 'HOLD'])
                buy_signals_count = len(df_filtered[df_filtered['signal'] == 'BUY'])
                sell_signals_count = len(df_filtered[df_filtered['signal'] == 'SELL'])
                
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Total Signals", total_signals)
                with stat_col2:
                    st.metric("Buy Signals", buy_signals_count, delta=None)
                with stat_col3:
                    st.metric("Sell Signals", sell_signals_count, delta=None)
                
                # Time in position
                if 'position' in df_filtered.columns:
                    time_in_position = (df_filtered['position'] == 1).mean() * 100
                    st.metric(
                        "Time in Position", 
                        f"{time_in_position:.1f}%",
                        help="Percentage of time spent in a LONG position"
                    )
    
    # ==================== TAB 3: BACKTEST RESULTS ==================== #
    with tab3:
        st.markdown("### 💰 Equity Curve")
        
        backtester = Backtester(data_path=ticker_data_path)
        backtest_visual_output = backtester.run_backtest(df_filtered)
        results_visual_df = backtest_visual_output.get('results')
        
        if results_visual_df is not None:
            fig_equity = create_equity_chart(results_visual_df)
            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.warning("Could not generate equity curve for the selected range.")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### 📊 Performance Metrics")
        
        backtest_data = data.get('backtest', {})
        
        if 'in_sample' in backtest_data and 'out_of_sample' in backtest_data:
            is_metrics = backtest_data['in_sample']
            oos_metrics = backtest_data['out_of_sample']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-box" style="background: linear-gradient(145deg, #eff6ff 0%, #dbeafe 100%); border-color: #93c5fd;">
                    <h4 style="color: #1e40af; margin-top: 0;">📘 In-Sample Performance</h4>
                </div>
                """, unsafe_allow_html=True)
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Total Return", f"{float(is_metrics.get('total_return_%', 0)):.2f}%")
                    st.metric("Sharpe Ratio", f"{float(is_metrics.get('sharpe_ratio', 0)):.2f}")
                    st.metric("Win Rate", f"{float(is_metrics.get('win_rate_%', 0)):.1f}%")
                with m2:
                    st.metric("Max Drawdown", f"{float(is_metrics.get('max_drawdown_%', 0)):.2f}%")
                    st.metric("Profit Factor", f"{float(is_metrics.get('profit_factor', 0)):.2f}")
                    st.metric("Total Trades", f"{int(is_metrics.get('total_trades', 0))}")
            
            with col2:
                st.markdown("""
                <div class="info-box info-box-success">
                    <h4 style="color: #065f46; margin-top: 0;">📗 Out-of-Sample Performance</h4>
                </div>
                """, unsafe_allow_html=True)
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Total Return", f"{float(oos_metrics.get('total_return_%', 0)):.2f}%")
                    st.metric("Sharpe Ratio", f"{float(oos_metrics.get('sharpe_ratio', 0)):.2f}")
                    st.metric("Win Rate", f"{float(oos_metrics.get('win_rate_%', 0)):.1f}%")
                with m2:
                    st.metric("Max Drawdown", f"{float(oos_metrics.get('max_drawdown_%', 0)):.2f}%")
                    st.metric("Profit Factor", f"{float(oos_metrics.get('profit_factor', 0)):.2f}")
                    st.metric("Total Trades", f"{int(oos_metrics.get('total_trades', 0))}")
        else:
            st.markdown("*Walk-Forward results not available. Showing simple backtest metrics:*")
            metrics = backtest_visual_output.get('metrics', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return_%', 0):.2f}%")
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown_%', 0):.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Win Rate", f"{metrics.get('win_rate_%', 0):.1f}%")
            with col3:
                st.metric("Total Trades", f"{int(metrics.get('total_trades', 0))}")
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    
    # ==================== TAB 4: SIGNAL HISTORY ==================== #
    with tab4:
        st.markdown("### 📝 Recent Trading Signals")
        
        recent_signals = df_filtered[df_filtered['signal'] != 'HOLD'].tail(20).copy()
        
        if not recent_signals.empty:
            # Format for display
            display_df = recent_signals[['close', 'signal', 'phase_quadrant', 'signal_strength', 'confidence']].copy()
            display_df.columns = ['Price', 'Signal', 'Phase', 'Strength', 'Confidence']
            display_df.index = display_df.index.strftime('%Y-%m-%d')
            display_df.index.name = 'Date'
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
            display_df['Strength'] = display_df['Strength'].apply(lambda x: f"{x:.0f}/100")
            display_df['Phase'] = display_df['Phase'].apply(lambda x: str(x).split('(')[1].replace(')', '') if '(' in str(x) else str(x))
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No signals in the selected date range.")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### 💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_filtered.to_csv()
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv,
                file_name=f"kriterion_{selected_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if not recent_signals.empty:
                signals_csv = recent_signals.to_csv()
                st.download_button(
                    label="📥 Download Signals Only (CSV)",
                    data=signals_csv,
                    file_name=f"kriterion_signals_{selected_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # ==================== FOOTER ==================== #
    st.markdown(f"""
    <div class="footer">
        <p><b>Kriterion Quant Trading System</b> • Cycle-Based VIX Hedging Strategy</p>
        <p>Last updated: {data.get('summary', {}).get('timestamp', 'Unknown')[:19]}</p>
        <p style="font-size: 0.75rem; color: #94a3b8;">
            ⚠️ This is a research tool. Past performance does not guarantee future results.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    main()
