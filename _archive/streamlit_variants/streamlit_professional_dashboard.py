"""
Professional PSX Quantitative Trading System Dashboard
Comprehensive interface for all trading system capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import asyncio
import warnings
import glob
import os
import joblib
import shutil

# Import our trading system components
from psx_dps_fetcher import PSXDPSFetcher
from enhanced_data_fetcher import EnhancedDataFetcher
from intraday_signal_analyzer import IntradaySignalAnalyzer
from intraday_risk_manager import IntradayRiskManager
from backtesting_engine import WalkForwardBacktester
from intraday_backtesting_engine import IntradayWalkForwardBacktester
from portfolio_optimizer import PortfolioOptimizer
from ml_model_system import MLModelSystem
from feature_engineering import FeatureEngineer
from quant_system_config import SystemConfig

# Page configuration
st.set_page_config(
    page_title="PSX Quantitative Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-metric {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_system():
    """Initialize all system components"""
    if 'system_initialized' not in st.session_state:
        with st.spinner("Initializing PSX Quantitative Trading System..."):
            try:
                st.session_state.config = SystemConfig()
                st.session_state.psx_fetcher = PSXDPSFetcher()
                st.session_state.enhanced_fetcher = EnhancedDataFetcher()
                st.session_state.signal_analyzer = IntradaySignalAnalyzer()
                st.session_state.risk_manager = IntradayRiskManager(st.session_state.config)
                st.session_state.portfolio_optimizer = PortfolioOptimizer(st.session_state.config)
                st.session_state.ml_system = MLModelSystem(st.session_state.config)
                st.session_state.feature_engineer = FeatureEngineer(st.session_state.config)
                st.session_state.backtester = WalkForwardBacktester(st.session_state.config)
                st.session_state.intraday_backtester = IntradayWalkForwardBacktester(st.session_state.config)
                
                st.session_state.system_initialized = True
                st.success("✅ All systems initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ System initialization failed: {str(e)}")
                st.session_state.system_initialized = False

def render_header():
    """Render professional header"""
    st.markdown('<h1 class="main-header">PSX Quantitative Trading System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-metric">
            <h4>🎯 Institutional Grade</h4>
            <p>Professional quantitative trading platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-metric">
            <h4>📊 Real-Time Data</h4>
            <p>PSX DPS Official API Integration</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-metric">
            <h4>🤖 AI-Powered</h4>
            <p>Machine Learning & Advanced Analytics</p>
        </div>
        """, unsafe_allow_html=True)

def render_system_overview():
    """Render system capabilities overview"""
    st.markdown("## 🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Data & Signals</h4>
            <ul>
                <li>✅ PSX DPS Real-time API Integration</li>
                <li>✅ EODHD Premium Data Backup</li>
                <li>✅ Intraday Signal Generation</li>
                <li>✅ Advanced Feature Engineering</li>
                <li>✅ Cross-sectional Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Machine Learning</h4>
            <ul>
                <li>✅ LightGBM Ensemble Models</li>
                <li>✅ Purged Time Series Cross-Validation</li>
                <li>✅ Walk-Forward Analysis</li>
                <li>✅ Feature Selection & Engineering</li>
                <li>✅ Model Performance Monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>💼 Portfolio Management</h4>
            <ul>
                <li>✅ Kelly Criterion Position Sizing</li>
                <li>✅ Risk Parity Optimization</li>
                <li>✅ Sector Exposure Limits</li>
                <li>✅ Dynamic Rebalancing</li>
                <li>✅ Turnover Management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🛡️ Risk Management</h4>
            <ul>
                <li>✅ Real-time Stop Loss/Take Profit</li>
                <li>✅ Position Size Controls</li>
                <li>✅ Drawdown Circuit Breakers</li>
                <li>✅ Correlation Monitoring</li>
                <li>✅ VaR & Stress Testing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_performance_targets():
    """Display performance targets and benchmarks"""
    st.markdown("## 🎯 Performance Targets")
    
    config = st.session_state.config
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Annual Alpha Target",
            f"{config.performance.annual_alpha_target:.1%}",
            help="Target alpha vs KSE100 benchmark"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio Target", 
            f"{config.performance.sharpe_ratio_target:.1f}",
            help="Risk-adjusted return target"
        )
    
    with col3:
        st.metric(
            "Max Drawdown Limit",
            f"{config.performance.max_drawdown_limit:.0%}",
            help="Maximum allowed portfolio drawdown"
        )
    
    with col4:
        st.metric(
            "Target Volatility",
            f"{config.performance.target_volatility:.0%}",
            help="Annual volatility target"
        )

def render_live_data_section():
    """Render live market data section"""
    st.markdown("## 📊 Live Market Data")
    
    # Symbol selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbols = st.multiselect(
            "Select PSX Symbols",
            options=['HBL', 'UBL', 'MCB', 'ABL', 'BAFL', 'ENGRO', 'FFC', 'LUCK', 'DG.KHAN', 'NESTLE'],
            default=['HBL', 'UBL', 'MCB'],
            help="Select symbols for live data display"
        )
    
    with col2:
        auto_refresh = st.checkbox("Auto Refresh", value=False)
    
    with col3:
        if st.button("🔄 Refresh Data") or auto_refresh:
            fetch_and_display_live_data(symbols)

def fetch_and_display_live_data(symbols):
    """Fetch and display live data from PSX DPS"""
    if not symbols:
        st.warning("Please select at least one symbol")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    data_container = st.empty()
    
    try:
        fetcher = st.session_state.psx_fetcher
        live_data = {}
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Fetching {symbol} data...")
            progress_bar.progress((i + 1) / len(symbols))
            
            try:
                # Fetch intraday ticks
                ticks = fetcher.fetch_intraday_ticks(symbol, limit=100)
                
                if not ticks.empty:
                    latest_price = ticks['price'].iloc[-1]
                    latest_volume = ticks['volume'].iloc[-1]
                    day_change = ((latest_price - ticks['price'].iloc[0]) / ticks['price'].iloc[0]) * 100
                    
                    live_data[symbol] = {
                        'Price': f"{latest_price:.2f} PKR",
                        'Volume': f"{latest_volume:,}",
                        'Day Change': f"{day_change:+.2f}%",
                        'Ticks': len(ticks),
                        'Status': "✅ Live"
                    }
                else:
                    live_data[symbol] = {
                        'Price': "N/A",
                        'Volume': "N/A", 
                        'Day Change': "N/A",
                        'Ticks': 0,
                        'Status': "❌ No Data"
                    }
                    
            except Exception as e:
                live_data[symbol] = {
                    'Price': "Error",
                    'Volume': "Error",
                    'Day Change': "Error", 
                    'Ticks': 0,
                    'Status': f"❌ {str(e)[:20]}..."
                }
        
        # Display live data table
        if live_data:
            df = pd.DataFrame(live_data).T
            
            with data_container.container():
                st.markdown("### Live Market Data")
                st.dataframe(
                    df,
                    use_container_width=True,
                    height=300
                )
                
                # Show last update time
                st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        status_text.text("✅ Data fetch completed!")
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"❌ Error fetching live data: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def render_backtesting_section():
    """Render backtesting interface"""
    st.markdown("## 🔬 Strategy Backtesting")
    
    tab1, tab2 = st.tabs(["📅 Daily/Weekly Backtesting", "⏰ Intraday Backtesting"])
    
    with tab1:
        render_daily_backtesting()
    
    with tab2:
        render_intraday_backtesting()

def render_daily_backtesting():
    """Render daily backtesting interface"""
    st.markdown("### Walk-Forward Daily Strategy Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=730),  # 2 years ago
            help="Backtesting start date"
        )
        
        universe_size = st.slider(
            "Universe Size",
            min_value=10,
            max_value=100,
            value=50,
            help="Number of symbols in trading universe"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.now(),
            help="Backtesting end date"
        )
        
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            options=["DAILY", "WEEKLY", "MONTHLY"],
            index=2,
            help="Portfolio rebalancing frequency"
        )
    
    if st.button("🚀 Run Daily Backtest", type="primary"):
        run_daily_backtest(start_date, end_date, universe_size, rebalance_freq)

def run_daily_backtest(start_date, end_date, universe_size, rebalance_freq):
    """Execute daily backtesting"""
    with st.spinner("Running daily backtest... This may take a while."):
        try:
            st.info("Initializing daily backtest...")
            
            # 1. Get a list of symbols for the universe
            all_symbols = ['HBL', 'UBL', 'MCB', 'ABL', 'BAFL', 'ENGRO', 'FFC', 'LUCK', 'DG.KHAN', 'NESTLE', 'OGDC', 'PPL', 'SYS', 'TRG']
            universe = all_symbols[:universe_size]
            
            # 2. Fetch daily price data
            st.write(f"Fetching daily price data for {len(universe)} symbols...")
            price_data = st.session_state.enhanced_fetcher.fetch_daily_data(universe, start_date, end_date)
            
            if price_data.empty:
                st.warning("Could not fetch sufficient price data for the selected period and universe.")
                return

            price_data_pivot = price_data.pivot(index='date', columns='symbol', values='adjClose')

            # 3. Initialize and run the backtester
            st.write("Running walk-forward backtest... This is computationally intensive.")
            backtester = WalkForwardBacktester(st.session_state.config)
            
            backtester.rebalance_freq = rebalance_freq.upper()

            results = backtester.run_backtest(
                price_data=price_data_pivot,
                start_date=start_date,
                end_date=end_date
            )
            
            if not results:
                st.warning("Backtest completed with no results. The underlying engine may have issues or there may have been no valid trading periods.")
                return

            # 4. Display aggregate results
            st.markdown("### 📈 Walk-Forward Backtest Results")
            agg_metrics = backtester.get_aggregate_metrics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{agg_metrics.get('total_return', 0):.2%}")
            with col2:
                st.metric("Annualized Return", f"{agg_metrics.get('annualized_return', 0):.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{agg_metrics.get('sharpe_ratio', 0):.2f}")
            with col4:
                st.metric("Max Drawdown", f"{agg_metrics.get('max_drawdown', 0):.2%}")

            # 5. Plot equity curve
            if not backtester.equity_curve.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=backtester.equity_curve.index,
                    y=backtester.equity_curve.values,
                    name="Strategy Equity",
                    line=dict(color='#1f4e79', width=2)
                ))
                if not backtester.benchmark_curve.empty:
                    fig.add_trace(go.Scatter(
                        x=backtester.benchmark_curve.index,
                        y=backtester.benchmark_curve.values,
                        name="Benchmark Equity",
                        line=dict(color='#888888', width=1, dash='dash')
                    ))
                fig.update_layout(title="Walk-Forward Portfolio Equity Curve", yaxis_title="Portfolio Value (PKR)")
                st.plotly_chart(fig, use_container_width=True)
            
            # 6. Show period breakdown
            with st.expander("Show Period-by-Period Breakdown"):
                st.text(backtester.generate_report())

        except Exception as e:
            st.error(f"❌ Daily backtest failed with an error.")
            st.error(f"Error: {e}")
            st.info("This may be due to an inconsistency in the backtesting engine, such as a call to a non-existent `train_and_validate` method. Further fixes to the engine might be required.")
            import traceback
            st.text(traceback.format_exc())

def render_intraday_backtesting():
    """Render intraday backtesting interface"""
    st.markdown("### High-Frequency Intraday Strategy Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_date = st.date_input(
            "Test Date",
            value=datetime.now(),
            help="Date for intraday backtesting"
        )
        
        symbols = st.multiselect(
            "Trading Symbols",
            options=['HBL', 'UBL', 'MCB', 'ABL', 'ENGRO', 'LUCK', 'FFC'],
            default=['HBL', 'UBL', 'MCB'],
            help="Symbols for intraday trading"
        )
    
    with col2:
        data_frequency = st.selectbox(
            "Data Frequency",
            options=["1min", "5min", "tick"],
            index=1,
            help="Tick data, minute bars, or 5-minute bars"
        )
        
        max_positions = st.slider(
            "Max Positions",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum simultaneous positions"
        )
    
    if st.button("🚀 Run Intraday Backtest", type="primary"):
        run_intraday_backtest(test_date, symbols, data_frequency, max_positions)

def run_intraday_backtest(test_date, symbols, data_frequency, max_positions):
    """Execute intraday backtesting"""
    if not symbols:
        st.warning("Please select at least one symbol for the backtest.")
        return

    with st.spinner(f"Running intraday backtest for {test_date.strftime('%Y-%m-%d')} on {len(symbols)} symbols..."):
        try:
            # 1. Initialize the backtester
            backtester = IntradayWalkForwardBacktester(st.session_state.config)
            
            # Note: To backtest the hybrid model, you would generate and pass ml_biases here
            # backtester = IntradayWalkForwardBacktester(st.session_state.config, ml_biases=get_ml_biases(symbols))
            
            # 2. Run the backtest for the selected day
            results = backtester.run_intraday_backtest(
                symbols=symbols,
                start_date=datetime.combine(test_date, time.min),
                end_date=datetime.combine(test_date, time.max),
                data_frequency=data_frequency
            )

            if not results:
                st.warning("No trading activity or data found for the selected day. This can happen on weekends, holidays, or if there was no data.")
                return

            # 3. Get results (for a single day, there will be one result object)
            day_result = results[0]

            # 4. Display results
            st.markdown("### ⚡ Intraday Backtest Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total P&L", f"{day_result.total_pnl:,.2f} PKR")
            with col2:
                st.metric("Number of Trades", f"{day_result.num_trades}")
            with col3:
                st.metric("Win Rate", f"{day_result.win_rate:.1%}")
            with col4:
                st.metric("Sharpe Ratio (Intraday)", f"{day_result.sharpe_ratio:.2f}")

            # Display more detailed metrics
            with st.expander("Show Detailed Performance Metrics"):
                perf_data = {
                    "Profit Factor": f"{day_result.profit_factor:.2f}",
                    "Max Intraday Drawdown": f"{day_result.max_intraday_drawdown:.2%}",
                    "Average Hold Time (min)": f"{day_result.average_hold_time_minutes:.1f}",
                    "Commissions Paid": f"{day_result.commission_paid:,.2f} PKR",
                    "Slippage Cost": f"{day_result.slippage_cost:,.2f} PKR",
                    "Signals Generated": day_result.signals_generated,
                    "Signals Executed": day_result.signals_executed,
                }
                st.json(perf_data)


            # 5. Plot equity curve
            if not day_result.equity_curve.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=day_result.equity_curve.index,
                    y=day_result.equity_curve.values,
                    name="Intraday P&L",
                    line=dict(color='#2e8b57', width=2)
                ))
                
                fig.update_layout(
                    title=f"Intraday Equity Curve for {test_date.strftime('%Y-%m-%d')}",
                    xaxis_title="Time",
                    yaxis_title="Portfolio Value (PKR)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 6. Display trades
            if not day_result.trades.empty:
                st.markdown("#### Trades Executed")
                st.dataframe(day_result.trades)

        except Exception as e:
            st.error(f"❌ Intraday backtest failed: {e}")
            import traceback
            st.text(traceback.format_exc())

def render_system_status():
    """Render system status and health checks"""
    st.markdown("## 🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌐 Data Sources")
        
        # Test PSX DPS connection
        try:
            fetcher = st.session_state.psx_fetcher
            test_data = fetcher.fetch_intraday_ticks('HBL', limit=1)
            if not test_data.empty:
                st.success("✅ PSX DPS API - Connected")
            else:
                st.warning("⚠️ PSX DPS API - No data")
        except Exception as e:
            st.error(f"❌ PSX DPS API - Error: {str(e)[:30]}...")
        
        # Test enhanced data fetcher
        try:
            enhanced = st.session_state.enhanced_fetcher
            st.success("✅ Enhanced Data Fetcher - Initialized")
        except:
            st.error("❌ Enhanced Data Fetcher - Error")
    
    with col2:
        st.markdown("### 🤖 System Components")
        
        components = [
            ("Signal Analyzer", "signal_analyzer"),
            ("Risk Manager", "risk_manager"), 
            ("Portfolio Optimizer", "portfolio_optimizer"),
            ("ML System", "ml_system"),
            ("Backtesting Engine", "backtester")
        ]
        
        for name, attr in components:
            if hasattr(st.session_state, attr):
                st.success(f"✅ {name} - Ready")
            else:
                st.error(f"❌ {name} - Not initialized")

def render_documentation():
    """Render system documentation"""
    st.markdown("## 📚 Documentation")
    
    tab1, tab2, tab3 = st.tabs(["🏗️ Architecture", "📊 API Reference", "🚀 Getting Started"])
    
    with tab1:
        st.markdown("""
        ### System Architecture
        
        The PSX Quantitative Trading System is built with a modular architecture:
        
        **Data Layer:**
        - PSX DPS Official API (Primary)
        - EODHD Premium API (Backup)
        - Real-time tick data processing
        
        **Signal Generation:**
        - Momentum and mean reversion indicators
        - Machine learning predictions
        - Cross-sectional equity ranking
        
        **Portfolio Management:**
        - Kelly criterion position sizing
        - Risk parity optimization
        - Dynamic rebalancing
        
        **Risk Management:**
        - Real-time stop loss/take profit
        - Position limits and drawdown controls
        - Correlation monitoring
        
        **Backtesting:**
        - Walk-forward validation
        - Out-of-sample testing
        - Performance attribution
        """)
    
    with tab2:
        st.markdown("""
        ### API Reference
        
        **PSX DPS Integration:**
        ```python
        fetcher = PSXDPSFetcher()
        data = fetcher.fetch_intraday_ticks('HBL')
        # Returns: DataFrame with [timestamp, price, volume]
        ```
        
        **Signal Generation:**
        ```python
        analyzer = IntradaySignalAnalyzer()
        signal = analyzer.analyze_symbol('HBL')
        # Returns: IntradaySignal with action, confidence, targets
        ```
        
        **Backtesting:**
        ```python
        backtester = IntradayWalkForwardBacktester(config)
        results = backtester.run_intraday_backtest(symbols, start, end)
        # Returns: List of IntradayBacktestResult objects
        ```
        """)
    
    with tab3:
        st.markdown("""
        ### Getting Started
        
        **1. Configuration:**
        - Set your performance targets in `SystemConfig`
        - Configure risk parameters and position sizes
        - Select your trading universe
        
        **2. Data Setup:**
        - PSX DPS API provides real-time data
        - EODHD API requires subscription for backup data
        - Historical data for backtesting
        
        **3. Strategy Development:**
        - Customize signal generation logic
        - Define entry/exit rules
        - Set risk management parameters
        
        **4. Backtesting:**
        - Test strategies with walk-forward validation
        - Analyze performance metrics
        - Refine parameters based on results
        
        **5. Live Trading:**
        - Deploy with paper trading first
        - Monitor performance vs backtests
        - Scale gradually with proven results
        """)

import glob
import os

# ... (other imports remain the same) ...

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_ml_biases(symbols: list) -> dict:
    """
    Generates and returns the latest ML model predictions (biases) for a list of symbols.
    """
    st.info("🤖 Generating ML daily biases... (This is cached and runs periodically)")
    try:
        ml_system = st.session_state.ml_system
        
        # 1. Find the model to use
        model_path = st.session_state.get('active_model_path', None)
        if not model_path:
            model_dirs = glob.glob("models/models_*")
            if not model_dirs:
                st.warning("No trained ML models found. Cannot generate ML biases.")
                return {}
            model_path = max(model_dirs, key=os.path.getmtime)

        st.write(f"Loading ML model from: `{os.path.basename(model_path)}`")
        ml_system.load_models(model_path)

        # 3. Get data for feature engineering (e.g., last 365 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        with st.spinner(f"Fetching daily data for {len(symbols)} symbols..."):
            data = st.session_state.enhanced_fetcher.fetch_daily_data(symbols, start_date, end_date)
        
        if data.empty:
            st.warning("Could not fetch daily data for feature engineering.")
            return {}

        # 4. Engineer features
        with st.spinner("Engineering features for ML model..."):
            feature_engineer = st.session_state.feature_engineer
            features_df = feature_engineer.create_all_features(data)

        # 5. Get the most recent features for each symbol for prediction
        latest_features = features_df.groupby('symbol').last()
        if latest_features.empty:
            st.warning("Could not generate latest features for prediction.")
            return {}

        # 6. Predict
        with st.spinner("Generating ML predictions..."):
            predictions = ml_system.predict(latest_features)
        
        # 7. Create bias dictionary
        biases = dict(zip(latest_features.index, predictions))
        
        st.success("✅ ML daily biases generated successfully.")
        return biases

    except Exception as e:
        st.error(f"Failed to generate ML biases: {e}")
        return {}

def render_signal_chart(signal, symbol, data_fetcher):
    """Renders a detailed price chart for a given signal."""
    
    with st.spinner(f"Loading chart for {symbol}..."):
        try:
            # Fetch last 60 minutes of tick data for a better chart
            ticks = data_fetcher.fetch_intraday_ticks(symbol, limit=500) # Fetch more ticks
            if ticks.empty:
                st.warning("Could not fetch sufficient data for chart.")
                return
            
            # Keep only the last 60 minutes of data
            last_hour = ticks.index.max() - timedelta(minutes=60)
            chart_data = ticks[ticks.index >= last_hour]

            if chart_data.empty:
                st.warning("No data in the last hour to plot.")
                return

            # Resample to 1-minute OHLC bars
            ohlc = chart_data['price'].resample('1min').ohlc()
            
            fig = go.Figure(data=[go.Candlestick(x=ohlc.index,
                                               open=ohlc['open'],
                                               high=ohlc['high'],
                                               low=ohlc['low'],
                                               close=ohlc['close'], name=symbol)])
            
            # Add lines for signal levels
            fig.add_hline(y=signal.entry_price, line_dash="dot", line_color="blue", annotation_text="Entry", annotation_position="bottom right")
            fig.add_hline(y=signal.target_price, line_dash="solid", line_color="green", annotation_text="Target", annotation_position="bottom right")
            fig.add_hline(y=signal.stop_loss, line_dash="solid", line_color="red", annotation_text="Stop Loss", annotation_position="bottom right")

            fig.update_layout(
                title=f"Price Chart for {symbol} Signal",
                yaxis_title="Price (PKR)",
                xaxis_title="Time",
                xaxis_rangeslider_visible=False,
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not generate chart for {symbol}: {e}")

def render_live_signals_section():
    """Render live trading signals"""
    st.markdown("## 📡 Live Trading Signals")

    # Symbol selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbols_to_scan = st.multiselect(
            "Select Symbols to Scan for Signals",
            options=['HBL', 'UBL', 'MCB', 'ABL', 'BAFL', 'ENGRO', 'FFC', 'LUCK', 'DG.KHAN', 'NESTLE'],
            default=['HBL', 'UBL', 'MCB', 'FFC', 'LUCK'],
            help="Select symbols to actively scan for trading signals."
        )
    with col2:
        min_confidence = st.slider(
            "Minimum Confidence", min_value=50, max_value=100, value=70,
            help="Minimum confidence level for a signal to be shown."
        )
    with col3:
        use_ml_hybrid = st.checkbox("Enable ML Hybrid Mode", value=False, help="Integrate daily ML forecasts into intraday signals.")

    if st.button("🔍 Scan for Signals", type="primary"):
        if not symbols_to_scan:
            st.warning("Please select at least one symbol to scan.")
            return

        ml_biases = {}
        if use_ml_hybrid:
            ml_biases = get_ml_biases(symbols_to_scan)
            if ml_biases:
                st.info(f"🧠 ML Hybrid Mode is ON. Using forecasts for {len(ml_biases)} symbols.")

        with st.spinner(f"Scanning {len(symbols_to_scan)} symbols for trading signals..."):
            try:
                analyzer = IntradaySignalAnalyzer(ml_biases=ml_biases)
                alerts = analyzer.get_live_alerts(symbols_to_scan, min_confidence=min_confidence)

                if not alerts:
                    st.success("✅ No strong trading signals found at the moment.")
                    return

                st.markdown(f"### Found {len(alerts)} Trading Alert(s):")

                for alert in alerts:
                    signal_color = "green" if "BUY" in alert.signal_type.name else "red"
                    
                    with st.expander(f"{alert.signal_type.name.replace('_', ' ')}: {alert.symbol} (Confidence: {alert.confidence:.1f}%)", expanded=True):
                        st.markdown(f"""
                        <div style="border-left: 5px solid {signal_color}; padding-left: 10px; margin-bottom: 10px;">
                            <p>
                                <strong>Entry:</strong> {alert.entry_price:.2f} |
                                <strong>Target:</strong> {alert.target_price:.2f} |
                                <strong>Stop Loss:</strong> {alert.stop_loss:.2f} |
                                <strong>R/R Ratio:</strong> {alert.risk_reward_ratio:.2f}
                            </p>
                            <p><strong>Reasoning:</strong></p>
                            <ul>
                                {''.join([f"<li>{reason}</li>" for reason in alert.reasoning])}
                            </ul>
                            <p style="font-size: 0.8em; color: #888;">Analyzed at: {alert.analysis_time.strftime('%H:%M:%S')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        render_signal_chart(alert, alert.symbol, st.session_state.psx_fetcher)

            except Exception as e:
                st.error(f"❌ An error occurred while scanning for signals: {e}")

def render_model_management_section():
    """Render the model management dashboard"""
    st.markdown("## 🗂️ ML Model Management")

    st.info("Manage, inspect, and select the active machine learning model for predictions.")

    # Find all saved model directories
    model_dirs = sorted(glob.glob("models/models_*"), reverse=True)
    
    if not model_dirs:
        st.warning("No trained models found in the `models/` directory.")
        st.info("You can train a new model using a training script, e.g., `train_models.py`.")
        return

    # Display active model
    active_model = st.session_state.get('active_model_path', 'None (defaults to latest)')
    st.success(f"**Active Model for Live Signals:** `{os.path.basename(active_model)}`")

    # Model selection
    selected_model_dir = st.selectbox("Select a Model Version to Inspect", options=model_dirs, format_func=lambda x: os.path.basename(x))

    if selected_model_dir:
        try:
            metadata_path = os.path.join(selected_model_dir, "metadata.joblib")
            metadata = joblib.load(metadata_path)

            st.markdown(f"### Inspection for `{os.path.basename(selected_model_dir)}`")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Set as Active Model", key=f"activate_{selected_model_dir}"):
                    st.session_state.active_model_path = selected_model_dir
                    st.success(f"Model `{os.path.basename(selected_model_dir)}` is now active!")
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete Model", key=f"delete_{selected_model_dir}", type="secondary"):
                    st.session_state.confirm_delete_model = selected_model_dir
                
            if st.session_state.get('confirm_delete_model') == selected_model_dir:
                st.warning(f"Are you sure you want to delete `{os.path.basename(selected_model_dir)}`? This cannot be undone.")
                if st.button("Yes, permanently delete", key=f"confirm_delete_btn_{selected_model_dir}"):
                    shutil.rmtree(selected_model_dir)
                    del st.session_state.confirm_delete_model
                    st.success("Model deleted.")
                    st.rerun()

            # Display Model Performance
            st.markdown("#### CV Performance Metrics")
            perf = metadata.get('model_performance', {})
            if perf:
                perf_metrics = {
                    "Mean Information Coefficient (IC)": f"{perf.get('mean_ic', 0):.4f}",
                    "Std Dev of IC": f"{perf.get('std_ic', 0):.4f}",
                    "Mean R^2 Score": f"{perf.get('mean_r2', 0):.4f}",
                    "Mean RMSE": f"{perf.get('mean_rmse', 0):.4f}",
                }
                st.json(perf_metrics)
            else:
                st.info("No cross-validation performance data found in metadata.")

            # Display Top Features
            st.markdown("#### Top 20 Features")
            features = metadata.get('feature_importance', {})
            if features:
                sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
                top_features_df = pd.DataFrame(sorted_features[:20], columns=['Feature', 'Importance'])
                st.dataframe(top_features_df)
            else:
                st.info("No feature importance data found.")

            with st.expander("View Ensemble Weights and Config"):
                st.markdown("##### Ensemble Weights")
                st.json(metadata.get('ensemble_weights', {}))

                st.markdown("##### Model Config")
                st.json(metadata.get('config', {}))

        except FileNotFoundError:
            st.error(f"Could not find `metadata.joblib` for model `{os.path.basename(selected_model_dir)}`.")
        except Exception as e:
            st.error(f"An error occurred while inspecting the model: {e}")


def main():
    """Main dashboard application"""
    
    # Initialize system
    initialize_system()
    
    if not st.session_state.get('system_initialized', False):
        st.error("❌ System not initialized. Please check configuration.")
        return
    
    # Render header
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Module",
        options=[
            "🏠 Overview",
            "📡 Live Signals",
            "🗂️ Model Management",
            "📊 Live Data", 
            "🔬 Backtesting",
            "🎯 Performance",
            "🔧 System Status",
            "📚 Documentation"
        ]
    )
    
    # Page routing
    if page == "🏠 Overview":
        render_system_overview()
        render_performance_targets()
        
    elif page == "📡 Live Signals":
        render_live_signals_section()
        
    elif page == "🗂️ Model Management":
        render_model_management_section()
        
    elif page == "📊 Live Data":
        render_live_data_section()
        
    elif page == "🔬 Backtesting":
        render_backtesting_section()
        
    elif page == "🎯 Performance":
        st.markdown("## 🎯 Performance Analytics")
        st.info("📈 Portfolio performance analytics coming soon...")
        
    elif page == "🔧 System Status":
        render_system_status()
        
    elif page == "📚 Documentation":
        render_documentation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>PSX Quantitative Trading System | Built with ❤️ for Professional Trading</p>
        <p>🔒 Real-time data • 🤖 AI-powered • 📊 Institutional-grade • 🚀 Production-ready</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()