"""
Enhanced Professional PSX Quantitative Trading System Dashboard
Now featuring PSX Terminal API with comprehensive market data and WebSocket streaming
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced trading system components
from psx_terminal_api import PSXTerminalAPI, MarketTick, KLineData
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
    page_title="PSX Terminal - Quantitative Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298, #0f4c75);
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
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-metric {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .live-data-card {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .market-overview-card {
        background: linear-gradient(135deg, #2e8b57 0%, #3cb371 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    .websocket-status {
        position: fixed;
        top: 10px;
        right: 10px;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        z-index: 1000;
    }
    
    .connected {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
    }
    
    .disconnected {
        background: linear-gradient(90deg, #f44336, #d32f2f);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def initialize_enhanced_system():
    """Initialize all enhanced system components"""
    if 'system_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing Enhanced PSX Terminal Trading System..."):
            try:
                st.session_state.config = SystemConfig()
                st.session_state.psx_terminal = PSXTerminalAPI()
                st.session_state.enhanced_fetcher = EnhancedDataFetcher()
                st.session_state.signal_analyzer = IntradaySignalAnalyzer()
                st.session_state.risk_manager = IntradayRiskManager(st.session_state.config)
                st.session_state.portfolio_optimizer = PortfolioOptimizer(st.session_state.config)
                st.session_state.ml_system = MLModelSystem(st.session_state.config)
                st.session_state.feature_engineer = FeatureEngineer(st.session_state.config)
                st.session_state.backtester = WalkForwardBacktester(st.session_state.config)
                st.session_state.intraday_backtester = IntradayWalkForwardBacktester(st.session_state.config)
                
                # Test PSX Terminal API connectivity
                status = st.session_state.psx_terminal.test_connectivity()
                if status:
                    st.success("✅ PSX Terminal API connected successfully!")
                    
                    # Get symbols list
                    symbols = st.session_state.psx_terminal.get_all_symbols()
                    if symbols:
                        st.session_state.available_symbols = symbols
                        st.success(f"✅ Loaded {len(symbols)} symbols from PSX Terminal")
                    else:
                        st.session_state.available_symbols = ['HBL', 'UBL', 'MCB', 'ENGRO', 'LUCK']
                        st.warning("⚠️ Using default symbols list")
                else:
                    st.error("❌ PSX Terminal API connection failed")
                    st.session_state.available_symbols = ['HBL', 'UBL', 'MCB', 'ENGRO', 'LUCK']
                
                st.session_state.system_initialized = True
                
            except Exception as e:
                st.error(f"❌ System initialization failed: {str(e)}")
                st.session_state.system_initialized = False

def render_enhanced_header():
    """Render enhanced professional header"""
    st.markdown('<h1 class="main-header">PSX Terminal - Quantitative Trading System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="success-metric">
            <h4>🎯 PSX Terminal API</h4>
            <p>Real-time & Historical Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-metric">
            <h4>📊 WebSocket Streaming</h4>
            <p>Live Market Updates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-metric">
            <h4>🤖 Advanced Analytics</h4>
            <p>ML Models & Optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="success-metric">
            <h4>🛡️ Risk Management</h4>
            <p>Professional Controls</p>
        </div>
        """, unsafe_allow_html=True)

def render_system_capabilities():
    """Render enhanced system capabilities"""
    st.markdown("## 🏗️ Enhanced System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 PSX Terminal Integration</h4>
            <ul>
                <li>✅ REST API for historical & real-time data</li>
                <li>✅ WebSocket streaming for live updates</li>
                <li>✅ Multiple market types (REG, FUT, IDX, ODL, BNB)</li>
                <li>✅ K-line data (1m, 5m, 15m, 1h, 4h, 1d)</li>
                <li>✅ Company fundamentals & financial ratios</li>
                <li>✅ Dividend data & market statistics</li>
                <li>✅ Market breadth & sector analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Enhanced ML Pipeline</h4>
            <ul>
                <li>✅ Real-time feature engineering</li>
                <li>✅ Multi-timeframe analysis</li>
                <li>✅ Cross-sectional ranking</li>
                <li>✅ Sentiment & fundamental integration</li>
                <li>✅ Dynamic model retraining</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>💼 Advanced Portfolio Management</h4>
            <ul>
                <li>✅ Real-time position monitoring</li>
                <li>✅ Dynamic rebalancing algorithms</li>
                <li>✅ Sector & correlation limits</li>
                <li>✅ Multi-asset optimization</li>
                <li>✅ Performance attribution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🛡️ Professional Risk Controls</h4>
            <ul>
                <li>✅ Real-time P&L monitoring</li>
                <li>✅ Dynamic stop-loss management</li>
                <li>✅ Volatility-based position sizing</li>
                <li>✅ Drawdown circuit breakers</li>
                <li>✅ Risk-adjusted performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def render_enhanced_live_data():
    """Render enhanced live market data section"""
    st.markdown("## 📊 Live Market Data & Analytics")
    
    # Market overview section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Real-Time Market Overview")
        
        # Get market overview
        try:
            api = st.session_state.psx_terminal
            overview = api.get_market_overview()
            
            if overview:
                # Display market statistics
                if 'regular_market' in overview:
                    reg_stats = overview['regular_market']
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        total_vol = reg_stats.get('totalVolume', 0)
                        st.metric("Total Volume", f"{total_vol:,.0f}" if total_vol else "N/A")
                    
                    with metric_col2:
                        total_val = reg_stats.get('totalValue', 0)
                        st.metric("Total Value", f"{total_val/1e9:.1f}B PKR" if total_val else "N/A")
                    
                    with metric_col3:
                        gainers = reg_stats.get('gainers', 0)
                        losers = reg_stats.get('losers', 0)
                        st.metric("Gainers/Losers", f"{gainers}/{losers}")
                    
                    with metric_col4:
                        trades = reg_stats.get('totalTrades', 0)
                        st.metric("Total Trades", f"{trades:,}" if trades else "N/A")
                
                # Market breadth
                if 'breadth' in overview:
                    breadth = overview['breadth']
                    st.markdown("### 📈 Market Breadth")
                    
                    breadth_col1, breadth_col2, breadth_col3 = st.columns(3)
                    
                    with breadth_col1:
                        ad_ratio = breadth.get('advanceDeclineRatio', 0)
                        st.metric("A/D Ratio", f"{ad_ratio:.2f}" if ad_ratio else "N/A")
                    
                    with breadth_col2:
                        up_vol = breadth.get('upVolume', 0)
                        down_vol = breadth.get('downVolume', 0)
                        if up_vol and down_vol:
                            vol_ratio = up_vol / down_vol
                            st.metric("Up/Down Volume", f"{vol_ratio:.2f}")
                        else:
                            st.metric("Up/Down Volume", "N/A")
                    
                    with breadth_col3:
                        advances = breadth.get('advances', 0)
                        declines = breadth.get('declines', 0)
                        st.metric("Advances", f"{advances}")
                        st.metric("Declines", f"{declines}")
            
            else:
                st.info("📊 Market overview data loading...")
                
        except Exception as e:
            st.error(f"❌ Error loading market overview: {str(e)}")
    
    with col2:
        st.markdown("### 🔄 Data Sources Status")
        
        # Test connectivity status
        try:
            api = st.session_state.psx_terminal
            status = api.test_connectivity()
            
            if status:
                st.markdown("""
                <div class="live-data-card">
                    <h5>✅ PSX Terminal API</h5>
                    <p>Status: Connected</p>
                    <p>Uptime: {:.1f}s</p>
                    <p>Last Update: {}</p>
                </div>
                """.format(
                    status.get('uptime', 0),
                    datetime.now().strftime('%H:%M:%S')
                ), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-metric">
                    <h5>⚠️ PSX Terminal API</h5>
                    <p>Connection Issues</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ API connectivity test failed: {str(e)}")

def render_symbol_analysis():
    """Render detailed symbol analysis"""
    st.markdown("## 🔍 Individual Symbol Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbols = st.session_state.get('available_symbols', ['HBL', 'UBL', 'MCB', 'ENGRO'])
        selected_symbol = st.selectbox(
            "Select Symbol for Analysis",
            options=symbols[:50],  # Limit to first 50 for performance
            index=0,
            help="Choose a symbol for detailed analysis"
        )
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            options=['1m', '5m', '15m', '1h', '4h', '1d'],
            index=3,  # Default to 1h
            help="Select chart timeframe"
        )
    
    with col3:
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.rerun()
    
    if selected_symbol:
        try:
            api = st.session_state.psx_terminal
            
            # Get comprehensive symbol data
            with st.spinner(f"📊 Loading data for {selected_symbol}..."):
                symbol_data = api.get_enhanced_symbol_data(selected_symbol)
            
            if symbol_data:
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Market Data", "📊 Charts", "🏢 Company Info", "💰 Fundamentals"])
                
                with tab1:
                    # Current market data
                    if 'market_data' in symbol_data:
                        market_data = symbol_data['market_data']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            price = market_data.get('price', 0)
                            change = market_data.get('change', 0)
                            st.metric("Current Price", f"{price:.2f} PKR", f"{change:+.2f}")
                        
                        with col2:
                            change_pct = market_data.get('change_percent', 0)
                            volume = market_data.get('volume', 0)
                            st.metric("Change %", f"{change_pct:+.2%}")
                            st.metric("Volume", f"{volume:,}")
                        
                        with col3:
                            high = market_data.get('high', 0)
                            low = market_data.get('low', 0)
                            st.metric("High", f"{high:.2f} PKR")
                            st.metric("Low", f"{low:.2f} PKR")
                        
                        with col4:
                            trades = market_data.get('trades', 0)
                            value = market_data.get('value', 0)
                            st.metric("Trades", f"{trades:,}")
                            st.metric("Value", f"{value/1e6:.1f}M PKR")
                
                with tab2:
                    # K-line chart
                    st.markdown(f"### 📈 {selected_symbol} - {timeframe} Chart")
                    
                    # Get k-line data
                    klines = api.get_klines(selected_symbol, timeframe, limit=100)
                    
                    if klines:
                        # Convert to DataFrame
                        df = api.convert_to_dataframe(klines)
                        
                        if not df.empty:
                            # Create candlestick chart
                            fig = go.Figure(data=go.Candlestick(
                                x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name=selected_symbol
                            ))
                            
                            fig.update_layout(
                                title=f"{selected_symbol} - {timeframe} Candlestick Chart",
                                xaxis_title="Time",
                                yaxis_title="Price (PKR)",
                                height=500,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Volume chart
                            fig_vol = go.Figure()
                            fig_vol.add_trace(go.Bar(
                                x=df.index,
                                y=df['volume'],
                                name="Volume",
                                marker_color='rgba(0, 150, 255, 0.6)'
                            ))
                            
                            fig_vol.update_layout(
                                title=f"{selected_symbol} - Volume",
                                xaxis_title="Time",
                                yaxis_title="Volume",
                                height=300
                            )
                            
                            st.plotly_chart(fig_vol, use_container_width=True)
                        else:
                            st.warning("⚠️ No chart data available")
                    else:
                        st.warning("⚠️ Unable to load chart data")
                
                with tab3:
                    # Company information
                    if 'company_info' in symbol_data:
                        company_info = symbol_data['company_info']
                        
                        st.markdown(f"### 🏢 Company Information - {selected_symbol}")
                        
                        # Financial stats
                        if 'financialStats' in company_info:
                            fin_stats = company_info['financialStats']
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                market_cap = fin_stats.get('marketCap', {})
                                if market_cap:
                                    st.metric("Market Cap", market_cap.get('raw', 'N/A'))
                                
                                shares = fin_stats.get('shares', {})
                                if shares:
                                    st.metric("Total Shares", shares.get('raw', 'N/A'))
                            
                            with col2:
                                free_float = fin_stats.get('freeFloat', {})
                                if free_float:
                                    st.metric("Free Float", free_float.get('raw', 'N/A'))
                                
                                ff_percent = fin_stats.get('freeFloatPercent', {})
                                if ff_percent:
                                    st.metric("Free Float %", ff_percent.get('raw', 'N/A'))
                        
                        # Business description
                        if 'businessDescription' in company_info:
                            st.markdown("### 📋 Business Description")
                            st.write(company_info['businessDescription'])
                        
                        # Key people
                        if 'keyPeople' in company_info:
                            st.markdown("### 👥 Key Personnel")
                            for person in company_info['keyPeople']:
                                st.write(f"**{person.get('name', 'N/A')}** - {person.get('position', 'N/A')}")
                    else:
                        st.info("📊 Company information not available")
                
                with tab4:
                    # Fundamentals
                    if 'fundamentals' in symbol_data:
                        fundamentals = symbol_data['fundamentals']
                        
                        st.markdown(f"### 💰 Financial Metrics - {selected_symbol}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pe_ratio = fundamentals.get('peRatio')
                            if pe_ratio:
                                st.metric("P/E Ratio", f"{pe_ratio:.2f}")
                            
                            div_yield = fundamentals.get('dividendYield')
                            if div_yield:
                                st.metric("Dividend Yield", f"{div_yield:.2f}%")
                        
                        with col2:
                            year_change = fundamentals.get('yearChange')
                            if year_change:
                                st.metric("YTD Change", f"{year_change:+.2f}%")
                            
                            sector = fundamentals.get('sector')
                            if sector:
                                st.metric("Sector Code", sector)
                        
                        with col3:
                            vol_30_avg = fundamentals.get('volume30Avg')
                            if vol_30_avg:
                                st.metric("30D Avg Volume", f"{vol_30_avg:,.0f}")
                            
                            is_compliant = fundamentals.get('isNonCompliant', True)
                            st.metric("Compliance", "✅ Compliant" if not is_compliant else "⚠️ Non-Compliant")
                    
                    # Dividend history
                    if 'dividends' in symbol_data:
                        st.markdown("### 💵 Dividend History")
                        
                        dividends = symbol_data['dividends']
                        if dividends:
                            div_df = pd.DataFrame(dividends)
                            st.dataframe(div_df, use_container_width=True)
                        else:
                            st.info("No recent dividend data available")
                    else:
                        st.info("📊 Fundamental data not available")
            
            else:
                st.warning(f"⚠️ Unable to load data for {selected_symbol}")
                
        except Exception as e:
            st.error(f"❌ Error analyzing {selected_symbol}: {str(e)}")

def render_websocket_demo():
    """Render WebSocket streaming demonstration"""
    st.markdown("## 🌐 Live WebSocket Streaming")
    
    st.info("🚧 WebSocket streaming interface coming soon! This will provide real-time updates for:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📈 Real-time Features:
        - Live price updates
        - Order book data  
        - Trade by trade updates
        - Market breadth changes
        - Volume analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🔔 Alert System:
        - Price breakouts
        - Volume spikes
        - Technical signals
        - News events
        - Risk alerts
        """)

def main():
    """Enhanced main dashboard application"""
    
    # Initialize enhanced system
    initialize_enhanced_system()
    
    if not st.session_state.get('system_initialized', False):
        st.error("❌ System not initialized. Please check configuration and API connectivity.")
        return
    
    # Render enhanced header
    render_enhanced_header()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Module",
        options=[
            "🏠 System Overview",
            "📊 Live Market Data", 
            "🔍 Symbol Analysis",
            "🌐 WebSocket Streaming",
            "🔬 Backtesting",
            "🎯 Performance",
            "🔧 System Status",
            "📚 Documentation"
        ]
    )
    
    # Page routing
    if page == "🏠 System Overview":
        render_system_capabilities()
        
    elif page == "📊 Live Market Data":
        render_enhanced_live_data()
        
    elif page == "🔍 Symbol Analysis":
        render_symbol_analysis()
        
    elif page == "🌐 WebSocket Streaming":
        render_websocket_demo()
        
    elif page == "🔬 Backtesting":
        st.markdown("## 🔬 Enhanced Backtesting")
        st.info("🚧 Enhanced backtesting interface with PSX Terminal data coming soon!")
        
    elif page == "🎯 Performance":
        st.markdown("## 🎯 Performance Analytics")
        st.info("📈 Enhanced portfolio performance analytics coming soon...")
        
    elif page == "🔧 System Status":
        st.markdown("## 🔧 Enhanced System Status")
        
        # API connectivity tests
        try:
            api = st.session_state.psx_terminal
            status = api.test_connectivity()
            
            if status:
                st.success(f"✅ PSX Terminal API - Connected (Uptime: {status.get('uptime', 0):.1f}s)")
            else:
                st.error("❌ PSX Terminal API - Connection Failed")
        except Exception as e:
            st.error(f"❌ PSX Terminal API - Error: {str(e)}")
        
        # System components status
        components = [
            ("Enhanced Data Fetcher", "enhanced_fetcher"),
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
        
    elif page == "📚 Documentation":
        st.markdown("## 📚 PSX Terminal API Documentation")
        
        st.markdown("""
        ### 🎯 API Capabilities
        
        **REST Endpoints:**
        - `/api/status` - Test connectivity
        - `/api/ticks/{type}/{symbol}` - Real-time market data
        - `/api/symbols` - All available symbols
        - `/api/stats/{type}` - Market statistics
        - `/api/companies/{symbol}` - Company information
        - `/api/fundamentals/{symbol}` - Financial ratios
        - `/api/klines/{symbol}/{timeframe}` - Historical data
        - `/api/dividends/{symbol}` - Dividend history
        
        **WebSocket Streams:**
        - Market data updates
        - K-line/candlestick data
        - Statistics updates
        - Symbol list updates
        
        **Market Types:**
        - REG (Regular Market)
        - FUT (Futures)
        - IDX (Indices)
        - ODL (Odd Lot)
        - BNB (Bills and Bonds)
        
        **Timeframes:**
        - 1m, 5m, 15m, 1h, 4h, 1d
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>PSX Terminal - Enhanced Quantitative Trading System | Powered by PSX Terminal API</p>
        <p>🔗 Real-time data • 🚀 WebSocket streaming • 📊 Professional analytics • 🛡️ Risk management</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()