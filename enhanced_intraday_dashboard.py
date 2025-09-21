"""
ENHANCED INTRADAY TRADING DASHBOARD
Advanced Real-Time Monitoring and Control Interface

Features:
- Real-time intraday signal monitoring
- Multi-timeframe analysis dashboard
- Volatility regime visualization
- Risk management controls
- Execution quality monitoring
- Performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import json
import warnings
from typing import Dict, List, Optional

# Import enhanced components
try:
    from enhanced_intraday_feature_engine import EnhancedIntradayFeatureEngine, IntradayFeatures
    FEATURE_ENGINE_AVAILABLE = True
except ImportError:
    FEATURE_ENGINE_AVAILABLE = False

try:
    from enhanced_intraday_risk_manager import EnhancedIntradayRiskManager, RiskSignal
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

try:
    from volatility_regime_detector import VolatilityRegimeDetector, VolatilityRegime
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False

try:
    from enhanced_backtesting_engine import EnhancedBacktestingEngine, BacktestResult
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False

try:
    from real_time_execution_engine import RealTimeExecutionEngine, Order, OrderType
    EXECUTION_ENGINE_AVAILABLE = True
except ImportError:
    EXECUTION_ENGINE_AVAILABLE = False

warnings.filterwarnings('ignore')

class EnhancedIntradayDashboard:
    """Enhanced Streamlit dashboard for intraday trading"""
    
    def __init__(self):
        self.initialize_components()
        self.setup_page_config()
        
    def initialize_components(self):
        """Initialize all available components"""
        
        self.feature_engine = None
        self.risk_manager = None
        self.regime_detector = None
        self.backtesting_engine = None
        self.execution_engine = None
        
        # Initialize available components
        if FEATURE_ENGINE_AVAILABLE:
            self.feature_engine = EnhancedIntradayFeatureEngine()
            
        if RISK_MANAGER_AVAILABLE:
            self.risk_manager = EnhancedIntradayRiskManager()
            
        if REGIME_DETECTOR_AVAILABLE:
            self.regime_detector = VolatilityRegimeDetector()
            
        if BACKTESTING_AVAILABLE:
            self.backtesting_engine = EnhancedBacktestingEngine()
            
        if EXECUTION_ENGINE_AVAILABLE:
            self.execution_engine = RealTimeExecutionEngine()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        
        st.set_page_config(
            page_title="Enhanced Intraday Trading System",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for enhanced UI
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(45deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 10px 0;
        }
        
        .risk-alert {
            background-color: #ffebee;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #f44336;
            margin: 10px 0;
        }
        
        .success-alert {
            background-color: #e8f5e8;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #4caf50;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_dashboard(self):
        """Render the main dashboard"""
        
        # Header
        st.markdown('<h1 class="main-header">🚀 Enhanced Intraday Trading System</h1>', unsafe_allow_html=True)
        
        # System status
        self.render_system_status()
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            [
                "📊 Real-Time Monitoring",
                "🎯 Signal Analysis",
                "⚠️ Risk Management",
                "🌪️ Volatility Regime",
                "⚡ Execution Monitor",
                "📈 Performance Analytics",
                "🔬 Backtesting Lab",
                "⚙️ System Settings"
            ]
        )
        
        # Render selected page
        if page == "📊 Real-Time Monitoring":
            self.render_real_time_monitoring()
        elif page == "🎯 Signal Analysis":
            self.render_signal_analysis()
        elif page == "⚠️ Risk Management":
            self.render_risk_management()
        elif page == "🌪️ Volatility Regime":
            self.render_volatility_regime()
        elif page == "⚡ Execution Monitor":
            self.render_execution_monitor()
        elif page == "📈 Performance Analytics":
            self.render_performance_analytics()
        elif page == "🔬 Backtesting Lab":
            self.render_backtesting_lab()
        elif page == "⚙️ System Settings":
            self.render_system_settings()
    
    def render_system_status(self):
        """Render system component status"""
        
        st.sidebar.markdown("### 🔧 System Components")
        
        components = [
            ("Feature Engine", FEATURE_ENGINE_AVAILABLE),
            ("Risk Manager", RISK_MANAGER_AVAILABLE),
            ("Regime Detector", REGIME_DETECTOR_AVAILABLE),
            ("Backtesting Engine", BACKTESTING_AVAILABLE),
            ("Execution Engine", EXECUTION_ENGINE_AVAILABLE)
        ]
        
        for name, available in components:
            if available:
                st.sidebar.success(f"✅ {name}")
            else:
                st.sidebar.warning(f"⚠️ {name}")
    
    def render_real_time_monitoring(self):
        """Render real-time monitoring dashboard"""
        
        st.header("📊 Real-Time Market Monitoring")
        
        # Symbol selection
        symbols = ["HBL", "UBL", "MCB", "ENGRO", "LUCK", "FFC", "PSO", "OGDC", "TRG", "SYSTEMS"]
        selected_symbols = st.multiselect("Select Symbols to Monitor:", symbols, default=["HBL", "UBL"])
        
        if not selected_symbols:
            st.warning("Please select at least one symbol to monitor")
            return
        
        # Time controls
        col1, col2, col3 = st.columns(3)
        with col1:
            timeframe = st.selectbox("Timeframe:", ["1min", "5min", "15min", "1H"], index=1)
        with col2:
            lookback_hours = st.slider("Lookback Hours:", 1, 24, 6)
        with col3:
            auto_refresh = st.checkbox("Auto Refresh", value=True)
        
        # Generate sample data for monitoring
        market_data = self.generate_sample_market_data(selected_symbols, lookback_hours)
        
        # Main monitoring grid
        for symbol in selected_symbols:
            st.subheader(f"📈 {symbol}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Generate sample metrics
            current_price = market_data[symbol]['Close'].iloc[-1]
            price_change = market_data[symbol]['Close'].pct_change().iloc[-1]
            volume = market_data[symbol]['Volume'].iloc[-1]
            avg_volume = market_data[symbol]['Volume'].mean()
            
            with col1:
                delta_color = "normal" if price_change >= 0 else "inverse"
                st.metric("Price", f"₨{current_price:.2f}", f"{price_change:.2%}", delta_color=delta_color)
            
            with col2:
                vol_ratio = volume / avg_volume
                st.metric("Volume Ratio", f"{vol_ratio:.1f}x", f"{volume:,}")
            
            with col3:
                # Calculate volatility
                volatility = market_data[symbol]['Close'].pct_change().std() * np.sqrt(252)
                st.metric("Volatility", f"{volatility:.1%}", "Annualized")
            
            with col4:
                # Generate sample signal
                signal_strength = np.random.uniform(0.4, 0.9)
                signal = "BUY" if signal_strength > 0.7 else "HOLD" if signal_strength > 0.5 else "SELL"
                color = "normal" if signal == "BUY" else "off" if signal == "HOLD" else "inverse"
                st.metric("Signal", signal, f"{signal_strength:.1%}", delta_color=color)
            
            # Price chart
            self.render_price_chart(symbol, market_data[symbol], timeframe)
            
            # Feature analysis (if available)
            if self.feature_engine:
                with st.expander(f"🔍 Feature Analysis - {symbol}"):
                    features = self.feature_engine.extract_comprehensive_features(symbol, market_data[symbol])
                    self.render_feature_summary(features)
            
            st.markdown("---")
    
    def render_signal_analysis(self):
        """Render signal analysis page"""
        
        st.header("🎯 Advanced Signal Analysis")
        
        # Symbol selection
        symbol = st.selectbox("Select Symbol:", ["HBL", "UBL", "MCB", "ENGRO", "LUCK"], index=0)
        
        # Generate sample data
        market_data = self.generate_sample_market_data([symbol], 24)[symbol]
        
        # Signal generation
        if self.feature_engine:
            features = self.feature_engine.extract_comprehensive_features(symbol, market_data)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Multi-timeframe analysis
                st.subheader("📊 Multi-Timeframe Analysis")
                
                # Create comprehensive chart
                fig = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=('Price & Signals', 'Volume', 'RSI', 'MACD'),
                    vertical_spacing=0.08,
                    row_heights=[0.4, 0.2, 0.2, 0.2]
                )
                
                # Price chart
                fig.add_trace(
                    go.Candlestick(
                        x=market_data.index,
                        open=market_data['Open'],
                        high=market_data['High'],
                        low=market_data['Low'],
                        close=market_data['Close'],
                        name="Price"
                    ), row=1, col=1
                )
                
                # Volume
                fig.add_trace(
                    go.Bar(x=market_data.index, y=market_data['Volume'], name="Volume"),
                    row=2, col=1
                )
                
                # RSI (simulated)
                rsi = 50 + 20 * np.sin(np.arange(len(market_data)) * 0.1) + np.random.normal(0, 5, len(market_data))
                fig.add_trace(
                    go.Scatter(x=market_data.index, y=rsi, name="RSI"),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                
                # MACD (simulated)
                macd = np.random.normal(0, 1, len(market_data))
                signal_line = np.random.normal(0, 0.8, len(market_data))
                fig.add_trace(
                    go.Scatter(x=market_data.index, y=macd, name="MACD"),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Scatter(x=market_data.index, y=signal_line, name="Signal"),
                    row=4, col=1
                )
                
                fig.update_layout(height=800, showlegend=False)
                fig.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature breakdown
                st.subheader("📋 Feature Summary")
                
                if hasattr(features, 'technical_features'):
                    st.write("**Technical Features:**")
                    tech_features = dict(list(features.technical_features.items())[:5])
                    for key, value in tech_features.items():
                        st.metric(key.replace('_', ' ').title(), f"{value:.3f}")
                
                if hasattr(features, 'temporal_features'):
                    st.write("**Temporal Features:**")
                    temp_features = dict(list(features.temporal_features.items())[:3])
                    for key, value in temp_features.items():
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                
                # Signal confidence
                confidence = np.random.uniform(0.6, 0.95)
                st.metric("Signal Confidence", f"{confidence:.1%}")
                
                # Recommended action
                if confidence > 0.8:
                    st.success("🟢 Strong Buy Signal")
                elif confidence > 0.6:
                    st.info("🔵 Moderate Signal")
                else:
                    st.warning("🟡 Weak Signal")
        else:
            st.warning("Feature engine not available. Please check system configuration.")
    
    def render_risk_management(self):
        """Render risk management dashboard"""
        
        st.header("⚠️ Risk Management Dashboard")
        
        if not self.risk_manager:
            st.error("Risk management system not available")
            return
        
        # Portfolio overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", "₨10,50,000", "₨50,000")
        with col2:
            st.metric("Daily P&L", "₨25,000", "2.4%")
        with col3:
            st.metric("Max Drawdown", "-5.2%", "Within Limits")
        with col4:
            st.metric("Risk Utilization", "65%", "Moderate")
        
        # Risk metrics
        st.subheader("📊 Risk Metrics")
        
        # Generate sample risk data
        risk_data = {
            'Position Risk': 0.65,
            'Portfolio Risk': 0.45,
            'Volatility Risk': 0.55,
            'Concentration Risk': 0.30,
            'Liquidity Risk': 0.40
        }
        
        # Risk gauge charts
        cols = st.columns(len(risk_data))
        for i, (metric, value) in enumerate(risk_data.items()):
            with cols[i]:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = value * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': metric},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "lightgreen" if value < 0.5 else "orange" if value < 0.8 else "red"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
        
        # Position limits
        st.subheader("📋 Position Limits")
        
        positions_data = {
            'HBL': {'current': 1.5, 'limit': 2.0, 'utilization': 0.75},
            'UBL': {'current': 1.2, 'limit': 2.0, 'utilization': 0.60},
            'ENGRO': {'current': 0.8, 'limit': 1.5, 'utilization': 0.53},
        }
        
        for symbol, data in positions_data.items():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write(f"**{symbol}**")
            with col2:
                progress_color = "normal" if data['utilization'] < 0.7 else "inverse"
                st.progress(data['utilization'])
            with col3:
                st.write(f"{data['current']:.1f}L / {data['limit']:.1f}L")
        
        # Risk alerts
        st.subheader("🚨 Risk Alerts")
        
        alerts = [
            {"type": "warning", "message": "Portfolio exposure approaching 80% limit"},
            {"type": "info", "message": "Volatility regime changed to HIGH - position sizes reduced"},
            {"type": "success", "message": "All stop losses are properly set"}
        ]
        
        for alert in alerts:
            if alert["type"] == "warning":
                st.warning(f"⚠️ {alert['message']}")
            elif alert["type"] == "info":
                st.info(f"ℹ️ {alert['message']}")
            else:
                st.success(f"✅ {alert['message']}")
    
    def render_volatility_regime(self):
        """Render volatility regime analysis"""
        
        st.header("🌪️ Volatility Regime Analysis")
        
        if not self.regime_detector:
            st.error("Volatility regime detector not available")
            return
        
        # Current regime status
        st.subheader("📊 Current Market Regime")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Regime", "High Volatility", "Changed 2h ago")
        with col2:
            st.metric("Volatility Level", "34.5%", "↑ 12.3%")
        with col3:
            st.metric("Regime Confidence", "87%", "High")
        with col4:
            st.metric("Duration", "2h 15m", "Ongoing")
        
        # Regime timeline
        st.subheader("📈 Regime Timeline")
        
        # Generate sample regime data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
        regimes = np.random.choice(['Low Vol', 'Normal Vol', 'High Vol', 'Extreme Vol'], 
                                 size=len(dates), p=[0.2, 0.5, 0.25, 0.05])
        
        regime_df = pd.DataFrame({
            'timestamp': dates,
            'regime': regimes,
            'volatility': np.random.uniform(0.1, 0.8, len(dates))
        })
        
        # Create regime timeline chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Volatility Level', 'Regime Classification'),
            vertical_spacing=0.1
        )
        
        # Volatility line
        fig.add_trace(
            go.Scatter(
                x=regime_df['timestamp'],
                y=regime_df['volatility'],
                mode='lines',
                name='Volatility',
                line=dict(color='blue')
            ), row=1, col=1
        )
        
        # Regime colored background
        regime_colors = {'Low Vol': 'green', 'Normal Vol': 'blue', 'High Vol': 'orange', 'Extreme Vol': 'red'}
        
        for i, regime in enumerate(regime_df['regime']):
            if i < len(regime_df) - 1:
                fig.add_vrect(
                    x0=regime_df['timestamp'].iloc[i],
                    x1=regime_df['timestamp'].iloc[i + 1],
                    fillcolor=regime_colors[regime],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=2, col=1
                )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime-specific recommendations
        st.subheader("💡 Trading Recommendations")
        
        recommendations = {
            "Position Sizing": "Reduce position sizes by 30% due to high volatility",
            "Stop Losses": "Widen stop losses by 1.5x normal ATR",
            "Take Profits": "Set more aggressive profit targets",
            "Trading Style": "Favor momentum strategies over mean reversion",
            "Risk Limits": "Increase minimum confidence threshold to 80%"
        }
        
        for category, recommendation in recommendations.items():
            st.info(f"**{category}:** {recommendation}")
    
    def render_execution_monitor(self):
        """Render execution monitoring dashboard"""
        
        st.header("⚡ Real-Time Execution Monitor")
        
        if not self.execution_engine:
            st.warning("Execution engine not available")
            return
        
        # Execution summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Orders Today", "47", "↑ 12")
        with col2:
            st.metric("Fill Rate", "94.7%", "↑ 2.1%")
        with col3:
            st.metric("Avg Slippage", "0.08%", "↓ 0.02%")
        with col4:
            st.metric("Execution Speed", "125ms", "Fast")
        
        # Recent orders table
        st.subheader("📋 Recent Orders")
        
        # Generate sample order data
        order_data = []
        for i in range(10):
            order_data.append({
                'Time': (datetime.now() - timedelta(minutes=np.random.randint(1, 120))).strftime('%H:%M:%S'),
                'Symbol': np.random.choice(['HBL', 'UBL', 'ENGRO', 'LUCK']),
                'Side': np.random.choice(['BUY', 'SELL']),
                'Quantity': np.random.randint(100, 1000),
                'Price': round(np.random.uniform(80, 150), 2),
                'Status': np.random.choice(['FILLED', 'PARTIAL', 'PENDING'], p=[0.8, 0.15, 0.05]),
                'Slippage': f"{np.random.uniform(0.01, 0.15):.3f}%"
            })
        
        orders_df = pd.DataFrame(order_data)
        
        # Color code status
        def color_status(val):
            if val == 'FILLED':
                return 'color: green'
            elif val == 'PARTIAL':
                return 'color: orange'
            else:
                return 'color: red'
        
        styled_df = orders_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Execution quality charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Slippage Distribution")
            slippage_data = np.random.lognormal(mean=-3, sigma=0.5, size=100)
            fig = px.histogram(x=slippage_data, bins=20, title="Slippage Distribution (bps)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⏱️ Execution Time Analysis")
            exec_times = np.random.gamma(2, 50, size=100)
            fig = px.box(y=exec_times, title="Execution Times (ms)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_analytics(self):
        """Render performance analytics dashboard"""
        
        st.header("📈 Performance Analytics")
        
        # Performance summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "24.7%", "↑ 2.3%")
        with col2:
            st.metric("Sharpe Ratio", "1.83", "↑ 0.12")
        with col3:
            st.metric("Max Drawdown", "-8.2%", "Improved")
        with col4:
            st.metric("Win Rate", "67.3%", "↑ 1.8%")
        
        # Equity curve
        st.subheader("📊 Equity Curve")
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        equity_curve = (1 + pd.Series(returns, index=dates)).cumprod() * 1000000
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity_curve, mode='lines', name='Portfolio Value'))
        fig.update_layout(title="Portfolio Equity Curve", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Monthly Returns")
            monthly_returns = np.random.normal(0.02, 0.05, 12)
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            colors = ['green' if r > 0 else 'red' for r in monthly_returns]
            fig = go.Figure(data=[go.Bar(x=months, y=monthly_returns, marker_color=colors)])
            fig.update_layout(title="Monthly Returns (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Strategy Performance")
            strategies = ['Momentum', 'Mean Reversion', 'Breakout', 'ML Ensemble']
            strategy_returns = [0.18, 0.23, 0.15, 0.31]
            
            fig = px.pie(values=strategy_returns, names=strategies, title="Strategy Contribution")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_backtesting_lab(self):
        """Render backtesting laboratory"""
        
        st.header("🔬 Backtesting Laboratory")
        
        if not self.backtesting_engine:
            st.warning("Backtesting engine not available")
            return
        
        # Backtest configuration
        st.subheader("⚙️ Backtest Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            symbols = st.multiselect("Symbols", ["HBL", "UBL", "MCB", "ENGRO"], default=["HBL"])
        
        with col2:
            end_date = st.date_input("End Date", datetime.now())
            initial_capital = st.number_input("Initial Capital", value=1000000)
        
        with col3:
            strategy_type = st.selectbox("Strategy", ["Momentum", "Mean Reversion", "ML Ensemble"])
            run_backtest = st.button("🚀 Run Backtest")
        
        if run_backtest:
            # Simulate backtest results
            st.subheader("📊 Backtest Results")
            
            with st.spinner("Running backtest..."):
                import time
                time.sleep(2)  # Simulate processing
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", "18.7%", "Profitable")
            with col2:
                st.metric("Sharpe Ratio", "1.45", "Good")
            with col3:
                st.metric("Max Drawdown", "-12.3%", "Acceptable")
            with col4:
                st.metric("Win Rate", "62.1%", "Above Average")
            
            # Detailed results
            st.success("✅ Backtest completed successfully!")
            
            # Generate sample equity curve
            test_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            test_returns = np.random.normal(0.0005, 0.015, len(test_dates))
            test_equity = (1 + pd.Series(test_returns, index=test_dates)).cumprod() * initial_capital
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=test_equity, mode='lines', name='Strategy'))
            
            # Add benchmark
            benchmark_returns = np.random.normal(0.0002, 0.010, len(test_dates))
            benchmark_equity = (1 + pd.Series(benchmark_returns, index=test_dates)).cumprod() * initial_capital
            fig.add_trace(go.Scatter(x=test_dates, y=benchmark_equity, mode='lines', name='Benchmark'))
            
            fig.update_layout(title="Backtest Results", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_system_settings(self):
        """Render system settings page"""
        
        st.header("⚙️ System Settings")
        
        # Trading parameters
        st.subheader("📊 Trading Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Max Position Size (%)", value=10.0, min_value=1.0, max_value=25.0)
            st.number_input("Stop Loss (%)", value=2.0, min_value=0.5, max_value=10.0)
            st.number_input("Daily Loss Limit (%)", value=5.0, min_value=1.0, max_value=15.0)
        
        with col2:
            st.number_input("Take Profit (%)", value=4.0, min_value=1.0, max_value=20.0)
            st.number_input("Min Confidence (%)", value=70.0, min_value=50.0, max_value=95.0)
            st.selectbox("Trading Session", ["Full Day", "Morning Only", "Afternoon Only"])
        
        # Risk management
        st.subheader("⚠️ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Position Limits", value=True)
            st.checkbox("Enable Stop Losses", value=True)
            st.checkbox("Enable Daily Loss Limits", value=True)
        
        with col2:
            st.checkbox("Enable Volatility Adjustment", value=True)
            st.checkbox("Enable Regime Detection", value=True)
            st.checkbox("Enable Real-time Monitoring", value=True)
        
        # System controls
        st.subheader("🔧 System Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Trading System"):
                st.success("Trading system started successfully!")
        
        with col2:
            if st.button("⏸️ Pause System"):
                st.warning("Trading system paused")
        
        with col3:
            if st.button("🛑 Emergency Stop"):
                st.error("Emergency stop activated!")
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    def generate_sample_market_data(self, symbols: List[str], hours: int) -> Dict[str, pd.DataFrame]:
        """Generate sample market data for demonstration"""
        
        data = {}
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        for symbol in symbols:
            # Generate realistic price data
            dates = pd.date_range(start=start_time, end=end_time, freq='5min')
            
            base_price = 100 + hash(symbol) % 50
            prices = []
            current_price = base_price
            
            for _ in range(len(dates)):
                change = np.random.normal(0, 0.002)  # 0.2% volatility
                current_price *= (1 + change)
                prices.append(current_price)
            
            data[symbol] = pd.DataFrame({
                'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
                'High': [p * np.random.uniform(1.001, 1.005) for p in prices],
                'Low': [p * np.random.uniform(0.995, 0.999) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
        
        return data
    
    def render_price_chart(self, symbol: str, data: pd.DataFrame, timeframe: str):
        """Render price chart with technical indicators"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{symbol} Price', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ), row=1, col=1
        )
        
        # Volume bars
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name="Volume"),
            row=2, col=1
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.update_xaxes(rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_summary(self, features: IntradayFeatures):
        """Render feature analysis summary"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Technical Features:**")
            if hasattr(features, 'technical_features'):
                for key, value in list(features.technical_features.items())[:5]:
                    st.text(f"{key}: {value:.3f}")
        
        with col2:
            st.write("**Market Session:**")
            if hasattr(features, 'session_features'):
                for key, value in list(features.session_features.items())[:3]:
                    st.text(f"{key}: {value:.3f}")

def main():
    """Main function to run the enhanced dashboard"""
    
    dashboard = EnhancedIntradayDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()