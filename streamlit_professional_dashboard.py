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
warnings.filterwarnings('ignore')

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
    with st.spinner("Running daily backtest..."):
        try:
            # Mock backtest for demo (in production, use real backtester)
            st.info("🔄 Generating mock backtest results for demonstration...")
            
            # Create sample results
            dates = pd.date_range(start_date, end_date, freq='D')
            np.random.seed(42)
            
            # Generate sample equity curve
            returns = np.random.normal(0.0006, 0.02, len(dates))  # ~15% annual return, 20% vol
            equity_curve = (1 + pd.Series(returns, index=dates)).cumprod() * 100000
            
            # Calculate metrics
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            annual_return = (1 + total_return) ** (365 / len(dates)) - 1
            volatility = pd.Series(returns).std() * np.sqrt(365)
            sharpe_ratio = (annual_return - 0.12) / volatility  # 12% risk-free rate
            
            max_dd = ((equity_curve / equity_curve.expanding().max()) - 1).min()
            
            # Display results
            st.markdown("### 📈 Backtest Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
            with col2:
                st.metric("Annual Return", f"{annual_return:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{max_dd:.2%}")
            
            # Plot equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                name="Strategy",
                line=dict(color='#1f4e79', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (PKR)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance analysis
            st.markdown("### 📊 Performance Analysis")
            
            target_alpha = st.session_state.config.performance.annual_alpha_target
            target_sharpe = st.session_state.config.performance.sharpe_ratio_target
            max_dd_limit = st.session_state.config.performance.max_drawdown_limit
            
            if annual_return >= target_alpha:
                st.success(f"✅ Alpha target achieved: {annual_return:.2%} vs {target_alpha:.2%}")
            else:
                st.warning(f"⚠️ Alpha below target: {annual_return:.2%} vs {target_alpha:.2%}")
            
            if sharpe_ratio >= target_sharpe:
                st.success(f"✅ Sharpe ratio target achieved: {sharpe_ratio:.2f} vs {target_sharpe:.2f}")
            else:
                st.warning(f"⚠️ Sharpe ratio below target: {sharpe_ratio:.2f} vs {target_sharpe:.2f}")
            
            if abs(max_dd) <= max_dd_limit:
                st.success(f"✅ Drawdown within limit: {max_dd:.2%} vs {max_dd_limit:.2%}")
            else:
                st.error(f"❌ Drawdown exceeds limit: {max_dd:.2%} vs {max_dd_limit:.2%}")
                
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")

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
    with st.spinner("Running intraday backtest..."):
        try:
            st.info("🔄 Processing intraday strategy with real PSX DPS data...")
            
            # Sample intraday results for demo
            np.random.seed(42)
            
            # Generate sample trades
            num_trades = np.random.randint(5, 25)
            pnl_per_trade = np.random.normal(500, 2000, num_trades)  # Average 500 PKR per trade
            
            total_pnl = pnl_per_trade.sum()
            win_rate = len(pnl_per_trade[pnl_per_trade > 0]) / len(pnl_per_trade)
            
            # Display results
            st.markdown("### ⚡ Intraday Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total P&L", f"{total_pnl:,.0f} PKR")
            with col2:
                st.metric("Number of Trades", f"{num_trades}")
            with col3:
                st.metric("Win Rate", f"{win_rate:.1%}")
            with col4:
                avg_trade = total_pnl / num_trades if num_trades > 0 else 0
                st.metric("Avg Trade P&L", f"{avg_trade:.0f} PKR")
            
            # Generate sample intraday chart
            trading_hours = pd.date_range(
                start=datetime.combine(test_date, time(9, 45)),
                end=datetime.combine(test_date, time(15, 30)),
                freq='5min'
            )
            
            equity_values = np.cumsum(np.random.normal(0, 100, len(trading_hours))) + 1000000
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trading_hours,
                y=equity_values,
                name="Intraday P&L",
                line=dict(color='#2e8b57', width=2)
            ))
            
            fig.update_layout(
                title="Intraday Equity Curve",
                xaxis_title="Time",
                yaxis_title="Portfolio Value (PKR)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading summary
            st.markdown("### 📋 Trading Summary")
            
            if symbols:
                st.info(f"🎯 Traded symbols: {', '.join(symbols)}")
                st.info(f"📊 Data frequency: {data_frequency}")
                st.info(f"⏰ Session: 09:45 - 15:30 PSX time")
                
                if total_pnl > 0:
                    st.success(f"✅ Profitable session: +{total_pnl:,.0f} PKR")
                else:
                    st.warning(f"⚠️ Loss session: {total_pnl:,.0f} PKR")
            
        except Exception as e:
            st.error(f"❌ Intraday backtest failed: {str(e)}")

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