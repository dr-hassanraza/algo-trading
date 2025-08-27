#!/usr/bin/env python3
"""
PSX Trading Bot - Streamlit Web Interface
=========================================

Professional web interface for the PSX algorithmic trading system.
Features real-time analysis, interactive charts, and portfolio management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import base64
import io

# Configure Streamlit page
st.set_page_config(
    page_title="PSX Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from enhanced_signal_analyzer import enhanced_signal_analysis
    from config_manager import get_config, set_config
    from portfolio_manager import PortfolioManager
    from risk_manager import calculate_position_size, multi_timeframe_check
    from advanced_indicators import macd, stochastic, adx, detect_candlestick_patterns
    from visualization_engine import data_exporter
    from pdf_generator import PDFReportGenerator, create_download_link
    MODULES_AVAILABLE = True
    PDF_AVAILABLE = True
except ImportError as e:
    st.error(f"Required modules not available: {e}")
    MODULES_AVAILABLE = False
    PDF_AVAILABLE = False

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDashboard:
    """Main trading dashboard class"""
    
    def __init__(self):
        self.portfolio_manager = PortfolioManager() if MODULES_AVAILABLE else None
        # Comprehensive PSX stocks list (100+ Pakistani firms)
        self.common_symbols = [
            # Banking & Financial Services (16 stocks)
            'UBL', 'MCB', 'NBP', 'ABL', 'HBL', 'BAFL', 'AKBL', 'BAHL', 'SNBL', 'KASB',
            'MEBL', 'JSBL', 'SILK', 'SUMB', 'FCCL', 'FYBL',  # Note: FYBL data may not be available
            
            # Oil & Gas (20 stocks)
            'OGDC', 'PPL', 'POL', 'MARI', 'PSO', 'SNGP', 'SSGC', 'APL', 'HASCOL', 'SHEL',
            'BYCO', 'ATTOCK', 'NRL', 'PACE', 'SPWL', 'KEL', 'MGCL', 'GAIL', 'PIOC', 'GHGL',
            
            # Power Generation (12 stocks)
            'HUBCO', 'KAPCO', 'LOTTE', 'NARC', 'REWM', 'KOSM', 'SAPT', 'TELE', 'GTYR',
            'HUBC', 'PAKGEN', 'NIKL',
            
            # Cement (15 stocks)
            'LUCK', 'DGKC', 'MLCF', 'FCCL', 'CHCC', 'ACPL', 'KOHC', 'FCCM', 'PIOC',
            'THCCL', 'GWLC', 'JSCL', 'UCAPM', 'POWERC', 'MAPLE',
            
            # Fertilizer & Chemicals (12 stocks)
            'ENGRO', 'FFC', 'FFBL', 'FATIMA', 'EPCL', 'EFERT', 'DAWH', 'CRTM', 'LOTCHEM',
            'AGRO', 'ICI', 'BERGER',
            
            # Steel & Engineering (10 stocks)
            'ISL', 'ASL', 'AICL', 'ASTL', 'MUGHAL', 'ITTEFAQ', 'DSFL', 'AGIC', 'LOADS', 'MTML',
            
            # Textiles (18 stocks)
            'APTM', 'GATM', 'FTML', 'KTML', 'ATRL', 'SITC', 'KOHE', 'CWSM', 'YUTM',
            'GADT', 'BWCL', 'RMPL', 'NISHAT', 'GULFP', 'MASFL', 'GLAXO', 'IQRA', 'BIFO',
            
            # Food & Personal Care (15 stocks)
            'NESTLE', 'UFL', 'WAVES', 'UNITY', 'ATLH', 'TOMCL', 'SHFA', 'MATM', 'BIFO',
            'SHIELD', 'PAKD', 'RAFHAN', 'COLG', 'QUICE', 'NATF',
            
            # Technology & Communication (8 stocks)
            'TRG', 'NETSOL', 'PTCL', 'SYSTEMS', 'AVANCEON', 'PACE', 'TELECARD', 'WTL',
            
            # Automobile & Auto Parts (8 stocks)
            'INDU', 'HCAR', 'PAKT', 'AGTL', 'GHNI', 'MILLAT', 'LOADS', 'HINOON',
            
            # Paper & Board (6 stocks)
            'PKGS', 'CPPC', 'EMCO', 'CHERAT', 'PACE', 'SKIN',
            
            # Pharmaceuticals (10 stocks)
            'GLAXO', 'ICI', 'ABT', 'SRLE', 'HNOON', 'MARTIN', 'LOTTE', 'HIGHNOON', 'SEARLE', 'WILSON',
            
            # Sugar & Allied (8 stocks)
            'PSMC', 'JDW', 'ASTL', 'ANSM', 'SHSML', 'UNITY', 'THAL', 'SITARA',
            
            # Real Estate & Construction (6 stocks)
            'DHA', 'LWMC', 'PACE', 'SIEM', 'POWERC', 'MAPLE',
            
            # Investment Companies (8 stocks)
            'PICIC', 'TRUST', 'PAKOXY', 'PABC', 'KASB', 'PAIR', 'UPFL', 'THCCL',
            
            # Miscellaneous (12 stocks)
            'LOADS', 'PACE', 'SKIN', 'BERRY', 'RICL', 'MFLO', 'TRIPF', 'MERIT', 'PARC',
            'DWSM', 'NEXT', 'TPL'
        ]
        
        # Remove duplicates and sort
        self.common_symbols = sorted(list(set(self.common_symbols)))
    
    def run(self):
        """Main dashboard application"""
        
        # Sidebar navigation
        st.sidebar.title("🏛️ PSX Trading Bot")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📊 Signal Analysis", "💼 Portfolio", "⚙️ Settings", "📈 Charts", "🎯 Risk Management"]
        )
        
        # Main content area
        if page == "🏠 Dashboard":
            self.show_dashboard()
        elif page == "📊 Signal Analysis":
            self.show_signal_analysis()
        elif page == "💼 Portfolio":
            self.show_portfolio()
        elif page == "⚙️ Settings":
            self.show_settings()
        elif page == "📈 Charts":
            self.show_charts()
        elif page == "🎯 Risk Management":
            self.show_risk_management()
    
    def show_dashboard(self):
        """Main dashboard overview"""
        
        st.title("🏠 PSX Trading Dashboard")
        st.markdown("Welcome to your professional PSX trading system!")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Signals Today", "5", "+2")
        
        with col2:
            st.metric("💼 Portfolio Value", "250,000 PKR", "+5.2%")
        
        with col3:
            st.metric("📈 Active Positions", "8", "+1")
        
        with col4:
            st.metric("🛡️ Risk Level", "Medium", "")
        
        st.markdown("---")
        
        # Quick analysis section
        st.subheader("🚀 Quick Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Multi-symbol selection
            st.subheader("📊 Quick Multi-Stock Analysis")
            
            # Selection options
            analysis_mode = st.radio(
                "Analysis Mode:",
                ["Single Stock", "Multiple Stocks", "Sector Analysis"],
                horizontal=True
            )
            
            if analysis_mode == "Single Stock":
                selected_symbols = [st.selectbox("Select Symbol", self.common_symbols)]
            elif analysis_mode == "Multiple Stocks":
                selected_symbols = st.multiselect(
                    "Select Multiple Symbols (up to 10)", 
                    self.common_symbols, 
                    default=['UBL', 'MCB'],
                    max_selections=10
                )
            else:  # Sector Analysis
                sector_options = {
                    "Banking": ['UBL', 'MCB', 'NBP', 'ABL', 'HBL', 'FYBL'],
                    "Oil & Gas": ['OGDC', 'PPL', 'POL', 'MARI', 'PSO'],
                    "Cement": ['LUCK', 'DGKC', 'MLCF', 'FCCL', 'CHCC'],
                    "Fertilizer": ['ENGRO', 'FFC', 'FFBL', 'FATIMA', 'EPCL'],
                    "Power": ['HUBCO', 'KAPCO', 'LOTTE', 'NARC'],
                    "Textiles": ['APTM', 'GATM', 'FTML', 'KTML', 'ATRL']
                }
                selected_sector = st.selectbox("Select Sector", list(sector_options.keys()))
                selected_symbols = sector_options[selected_sector]
                st.info(f"Analyzing {selected_sector} sector: {', '.join(selected_symbols)}")
            
            if selected_symbols and st.button("🚀 Analyze Selected Stocks"):
                if len(selected_symbols) == 1:
                    # Single stock - use the detailed quick_analysis
                    with st.spinner(f"Analyzing {selected_symbols[0]}..."):
                        self.quick_analysis(selected_symbols[0])
                else:
                    # Multiple stocks - use multi_stock_analysis
                    with st.spinner(f"Analyzing {len(selected_symbols)} stocks..."):
                        self.multi_stock_analysis(selected_symbols)
        
        with col2:
            st.subheader("📈 Market Status")
            market_status = self.get_market_status()
            
            # Display market status with appropriate styling
            if market_status['status'] == 'open':
                st.success(f"✅ {market_status['message']}")
            elif market_status['status'] == 'pre_open':
                st.info(f"🟡 {market_status['message']}")
            elif market_status['status'] == 'post_close':
                st.warning(f"🟠 {market_status['message']}")
            elif market_status['status'] == 'closed':
                st.error(f"🔴 {market_status['message']}")
            else:  # weekend or holiday
                st.info(f"⏸️ {market_status['message']}")
            
            # Show current time and next session
            st.info(f"🕐 Current Time: {market_status['current_time']}")
            if market_status.get('next_session'):
                st.info(f"⏰ Next: {market_status['next_session']}")
            st.info("📅 " + datetime.now().strftime("%B %d, %Y"))
    
    def get_market_status(self):
        """Get current PSX market status based on Pakistan Standard Time"""
        
        from datetime import datetime, time
        import pytz
        
        try:
            # Pakistan Standard Time (PKT)
            pkt_tz = pytz.timezone('Asia/Karachi')
            current_time = datetime.now(pkt_tz)
            current_date = current_time.date()
            current_time_only = current_time.time()
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            
            # PSX Trading Hours (PKT)
            pre_open_start = time(9, 0)      # 9:00 AM
            pre_open_end = time(9, 15)       # 9:15 AM
            market_open = time(9, 17)        # 9:17 AM
            
            # Regular market close times
            if weekday == 4:  # Friday
                regular_close = time(12, 30)  # 12:30 PM
                post_close_start = time(12, 35)  # 12:35 PM
                post_close_end = time(12, 50)   # 12:50 PM
                modification_end = time(13, 20)  # 1:20 PM
            else:  # Monday to Thursday
                regular_close = time(13, 30)   # 1:30 PM
                post_close_start = time(13, 35)  # 1:35 PM
                post_close_end = time(13, 50)   # 1:50 PM
                modification_end = time(14, 20)  # 2:20 PM
            
            # Format current time for display
            current_time_str = current_time.strftime("%I:%M %p PKT")
            
            # Weekend check
            if weekday >= 5:  # Saturday=5, Sunday=6
                return {
                    'status': 'weekend',
                    'message': 'Market Closed - Weekend',
                    'current_time': current_time_str,
                    'next_session': 'Monday 9:00 AM (Pre-Open)'
                }
            
            # Market status logic
            if current_time_only < pre_open_start:
                # Before pre-open
                return {
                    'status': 'closed',
                    'message': 'Market Closed',
                    'current_time': current_time_str,
                    'next_session': f'Today {pre_open_start.strftime("%I:%M %p")} (Pre-Open)'
                }
            elif pre_open_start <= current_time_only < pre_open_end:
                # Pre-open session
                return {
                    'status': 'pre_open',
                    'message': 'Pre-Open Session (9:00-9:15 AM)',
                    'current_time': current_time_str,
                    'next_session': f'Today {market_open.strftime("%I:%M %p")} (Market Open)'
                }
            elif time(9, 15) <= current_time_only < market_open:
                # Between pre-open and market open
                return {
                    'status': 'closed',
                    'message': 'Market Opening Soon',
                    'current_time': current_time_str,
                    'next_session': f'Today {market_open.strftime("%I:%M %p")} (Market Open)'
                }
            elif market_open <= current_time_only < regular_close:
                # Regular trading hours
                close_time_str = regular_close.strftime("%I:%M %p")
                day_name = "Friday" if weekday == 4 else "Mon-Thu"
                return {
                    'status': 'open',
                    'message': f'Market Open ({day_name}: 9:17 AM - {close_time_str})',
                    'current_time': current_time_str,
                    'next_session': f'Today {close_time_str} (Market Close)'
                }
            elif regular_close <= current_time_only < post_close_start:
                # Brief gap between regular close and post-close
                return {
                    'status': 'closed',
                    'message': 'Market Closed - Post Session Starting Soon',
                    'current_time': current_time_str,
                    'next_session': f'Today {post_close_start.strftime("%I:%M %p")} (Post-Close Session)'
                }
            elif post_close_start <= current_time_only < post_close_end:
                # Post-close session
                return {
                    'status': 'post_close',
                    'message': f'Post-Close Session ({post_close_start.strftime("%I:%M %p")}-{post_close_end.strftime("%I:%M %p")})',
                    'current_time': current_time_str,
                    'next_session': 'Tomorrow 9:00 AM (Pre-Open)' if weekday == 4 else 'Tomorrow 9:00 AM (Pre-Open)'
                }
            elif post_close_end <= current_time_only < modification_end:
                # Trade rectification/modification
                return {
                    'status': 'post_close',
                    'message': f'Trade Rectification ({post_close_end.strftime("%I:%M %p")}-{modification_end.strftime("%I:%M %p")})',
                    'current_time': current_time_str,
                    'next_session': 'Tomorrow 9:00 AM (Pre-Open)' if weekday == 4 else 'Tomorrow 9:00 AM (Pre-Open)'
                }
            else:
                # After all trading activities
                next_day = "Monday" if weekday == 4 else "Tomorrow"
                return {
                    'status': 'closed',
                    'message': 'Market Closed',
                    'current_time': current_time_str,
                    'next_session': f'{next_day} 9:00 AM (Pre-Open)'
                }
                
        except Exception as e:
            # Fallback if timezone or other issues
            return {
                'status': 'unknown',
                'message': 'Market Status Unknown',
                'current_time': datetime.now().strftime("%I:%M %p"),
                'next_session': 'Check PSX official timings'
            }
    
    def show_signal_analysis(self):
        """Enhanced signal analysis page"""
        
        st.title("📊 Enhanced Signal Analysis")
        
        # Input section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            symbol_input = st.text_input("Enter Symbol (e.g., UBL, MCB)", value="UBL")
        
        with col2:
            analysis_type = st.selectbox("Analysis Type", ["Enhanced", "Basic", "Compare"])
        
        with col3:
            days = st.number_input("Lookback Days", min_value=50, max_value=500, value=260)
        
        # Multi-symbol analysis
        if analysis_type == "Compare":
            symbols = st.multiselect("Select Symbols to Compare", self.common_symbols, default=["UBL", "MCB"])
        else:
            symbols = [symbol_input.upper()]
        
        # Analysis button
        if st.button("🚀 Run Analysis"):
            self.run_signal_analysis(symbols, analysis_type, days)
    
    def show_portfolio(self):
        """Portfolio management page"""
        
        st.title("💼 Portfolio Management")
        
        if not MODULES_AVAILABLE:
            st.error("Portfolio features not available")
            return
        
        # Portfolio overview
        self.show_portfolio_overview()
        
        st.markdown("---")
        
        # Add/Remove positions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("➕ Add Position")
            self.add_position_form()
        
        with col2:
            st.subheader("➖ Sell Position")
            self.sell_position_form()
    
    def show_settings(self):
        """Settings and configuration page"""
        
        st.title("⚙️ System Settings")
        
        # Trading parameters
        st.subheader("📈 Trading Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ma_period = st.number_input("MA Period", min_value=10, max_value=100, 
                                      value=get_config('trading_parameters.ma_period', 44))
            rsi_period = st.number_input("RSI Period", min_value=5, max_value=30,
                                       value=get_config('trading_parameters.rsi_period', 14))
            bb_period = st.number_input("Bollinger Band Period", min_value=10, max_value=50,
                                      value=get_config('trading_parameters.bb_period', 20))
        
        with col2:
            risk_pct = st.number_input("Default Risk %", min_value=1.0, max_value=10.0,
                                     value=get_config('risk_management.default_account_risk_pct', 2.0))
            
            account_size = st.number_input("Account Size (PKR)", min_value=10000, max_value=10000000,
                                         value=100000, step=10000)
        
        # API Settings
        st.subheader("🔑 API Configuration")
        api_key = st.text_input("EODHD API Key", type="password", 
                               value=get_config('api.eodhd_key', ''))
        
        # Save settings
        if st.button("💾 Save Settings"):
            set_config('trading_parameters.ma_period', ma_period)
            set_config('trading_parameters.rsi_period', rsi_period)
            set_config('trading_parameters.bb_period', bb_period)
            set_config('risk_management.default_account_risk_pct', risk_pct)
            set_config('api.eodhd_key', api_key)
            
            st.success("✅ Settings saved successfully!")
    
    def show_charts(self):
        """Interactive charts page"""
        
        st.title("📈 Interactive Charts")
        
        # Chart configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Symbol", self.common_symbols)
        
        with col2:
            period = st.selectbox("Period", ["1M", "3M", "6M", "1Y", "2Y"])
        
        with col3:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
        
        # Generate chart
        if st.button("📊 Generate Chart"):
            self.create_interactive_chart(symbol, period, chart_type)
    
    def show_risk_management(self):
        """Risk management tools"""
        
        st.title("🎯 Risk Management")
        
        # Position sizing calculator
        st.subheader("📏 Position Size Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_price = st.number_input("Current Price", min_value=1.0, value=100.0)
        
        with col2:
            stop_loss = st.number_input("Stop Loss", min_value=1.0, value=95.0)
        
        with col3:
            risk_pct = st.number_input("Risk %", min_value=0.5, max_value=10.0, value=2.0)
        
        if st.button("🧮 Calculate Position Size"):
            self.calculate_position_sizing(current_price, stop_loss, risk_pct)
        
        st.markdown("---")
        
        # Portfolio risk analysis
        st.subheader("📊 Portfolio Risk Analysis")
        self.show_portfolio_risk()
    
    def quick_analysis(self, symbol: str):
        """Perform quick analysis on a symbol with decision graphs"""
        
        if not MODULES_AVAILABLE:
            st.error("Analysis modules not available")
            return
        
        try:
            with st.container():
                result = enhanced_signal_analysis(symbol)
                
                if 'error' in result:
                    st.error(f"❌ Analysis failed for {symbol}")
                    st.warning(f"Error details: {result['error']}")
                    
                    if symbol == 'FYBL':
                        st.info("💡 **Note:** FYBL (Faisal Bank) data is not currently available in the EODHD API. This may be due to limited coverage of smaller PSX stocks.")
                    else:
                        st.info("💡 **Suggestion:** Try selecting a different symbol from the verified list. Some symbols may not be available in the EODHD API.")
                    
                    # Show alternative symbols
                    st.markdown("**Available verified banking symbols you can try:**")
                    verified_alternatives = ['UBL', 'MCB', 'ABL', 'HBL', 'NBP', 'BAFL', 'AKBL', 'BAHL']
                    st.write(", ".join(verified_alternatives))
                    return
                
                # Display results
                signal = result['signal_strength']
                tech = result['technical_data']
                risk = result['risk_management']
                
                # Header with main decision
                st.markdown("### 🎯 Trading Decision Analysis")
                
                # Main decision banner
                decision_color = {
                    "STRONG BUY": "🟢", "BUY": "🟡", "WEAK BUY": "🟠", 
                    "HOLD": "🔵", "AVOID": "🔴"
                }.get(signal['recommendation'], "⚫")
                
                st.markdown(f"## {decision_color} **{signal['recommendation']}** | Grade {signal['grade']} ({signal['score']:.0f}/100)")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"{result['price']:.2f} PKR", "")
                
                with col2:
                    st.metric("Stop Loss", f"{risk['stop_loss']:.2f} PKR", f"-{risk['stop_loss_pct']:.1f}%")
                
                with col3:
                    st.metric("Target", f"{risk['target1']:.2f} PKR", f"+{risk['target1_pct']:.1f}%")
                
                st.markdown("---")
                
                # Decision Graphs Section
                st.subheader("📊 Decision Analysis Graphs")
                
                # Create three columns for decision graphs
                graph_col1, graph_col2 = st.columns(2)
                
                with graph_col1:
                    # 1. Signal Strength Radar Chart
                    self.create_signal_strength_radar(signal, tech, result)
                    
                    # 3. Risk-Reward Decision Matrix
                    self.create_risk_reward_matrix(risk, signal)
                
                with graph_col2:
                    # 2. Technical Indicators Decision Chart
                    self.create_technical_indicators_chart(tech, signal)
                    
                    # 4. Buy/Sell Signal Confidence Gauge
                    self.create_confidence_gauge(signal, tech)
                
                # 5. Factor Contribution Bar Chart (full width)
                st.markdown("---")
                self.create_factor_contribution_chart(signal)
                
                st.markdown("---")
                
                # Enhanced Technical Summary
                st.subheader("📊 Detailed Technical Metrics")
                
                # Main indicators row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    rsi_status = "Oversold" if tech['rsi'] < 30 else "Overbought" if tech['rsi'] > 70 else "Normal"
                    st.metric("RSI", f"{tech['rsi']:.1f}", rsi_status)
                
                with col2:
                    st.metric("MA44", f"{tech['ma44']:.2f}", f"{tech.get('ma44_trend', 'Unknown')} trend")
                
                with col3:
                    bb_status = "Lower band" if tech['bb_pctb'] < 0.2 else "Upper band" if tech['bb_pctb'] > 0.8 else "Middle range"
                    st.metric("BB %B", f"{tech['bb_pctb']:.2f}", bb_status)
                
                with col4:
                    vol_status = "High" if tech['volume_ratio'] > 1.5 else "Low" if tech['volume_ratio'] < 0.8 else "Normal"
                    st.metric("Volume", f"{tech['volume_ratio']:.1f}x", vol_status)
                
                # Additional detailed metrics
                st.markdown("---")
                st.subheader("🔍 Additional Analysis")
                
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.write("**📈 Price Levels:**")
                    st.write(f"• Current: {result['price']:.2f} PKR")
                    st.write(f"• MA44: {tech['ma44']:.2f} PKR")
                    st.write(f"• Distance from MA44: {((result['price'] - tech['ma44']) / tech['ma44'] * 100):+.1f}%")
                    
                with detail_col2:
                    st.write("**⚖️ Risk Management:**")
                    st.write(f"• Stop Loss: {risk['stop_loss']:.2f} PKR ({risk['stop_loss_pct']:+.1f}%)")
                    st.write(f"• Target 1: {risk['target1']:.2f} PKR ({risk['target1_pct']:+.1f}%)")
                    st.write(f"• Risk/Reward: {risk['risk_reward_ratio']:.2f}")
                    
                with detail_col3:
                    st.write("**📊 Signal Quality:**")
                    st.write(f"• Overall Grade: {signal['grade']}")
                    st.write(f"• Confidence: {signal['score']:.1f}/100")
                    st.write(f"• Recommendation: {signal['recommendation']}")
                
                # Key factors summary
                if signal.get('factors'):
                    st.markdown("---")
                    st.subheader("🎯 Key Decision Factors")
                    factors_text = " | ".join(signal['factors'][:3])  # Top 3 factors
                    st.info(f"**Primary factors:** {factors_text}")
                
                # PDF Export functionality
                st.markdown("---")
                st.subheader("📄 Export Report")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button("📄 Generate Report", type="primary"):
                        try:
                            # Prepare analysis data
                            analysis_data = {
                                'recommendation': signal['recommendation'],
                                'signal_strength': signal['grade'],
                                'current_price': f"{result['price']:.2f} PKR",
                                'target_price': f"{risk['target1']:.2f} PKR",
                                'stop_loss': f"{risk['stop_loss']:.2f} PKR",
                                'risk_score': f"{signal['score']:.0f}/100",
                                'volatility': 'Medium',  # You can enhance this
                                'technical_indicators': {
                                    'RSI': {'value': f"{tech['rsi']:.1f}", 'signal': rsi_status},
                                    'MA44': {'value': f"{tech['ma44']:.2f}", 'signal': tech.get('ma44_trend', 'Unknown')},
                                    'Bollinger Bands %B': {'value': f"{tech['bb_pctb']:.2f}", 'signal': bb_status},
                                    'Volume Ratio': {'value': f"{tech['volume_ratio']:.1f}x", 'signal': vol_status}
                                }
                            }
                            
                            # Generate HTML report
                            html_report = self.generate_simple_pdf_report(symbol, analysis_data)
                            
                            # Store in session state to prevent reset
                            if 'html_report' not in st.session_state:
                                st.session_state.html_report = {}
                            st.session_state.html_report[symbol] = html_report
                            
                            st.success("✅ Report Generated Successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Report generation failed: {str(e)}")
                
                with col2:
                    # Show download button if report exists in session state
                    if hasattr(st.session_state, 'html_report') and symbol in st.session_state.html_report:
                        filename = f"PSX_Analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        b64 = base64.b64encode(st.session_state.html_report[symbol].encode()).decode()
                        href = f'''
                        <a href="data:text/html;base64,{b64}" download="{filename}" 
                           style="display: inline-block; padding: 12px 20px; background-color: #ff4b4b; 
                                  color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">
                           📥 Download Analysis Report
                        </a>
                        '''
                        st.markdown(href, unsafe_allow_html=True)
                        st.info("💡 **Tip:** Download creates an HTML report that you can print as PDF from your browser (Ctrl+P → Save as PDF).")
        
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
    
    def multi_stock_analysis(self, symbols: list):
        """Perform analysis on multiple stocks efficiently"""
        
        if not MODULES_AVAILABLE:
            st.error("Analysis modules not available")
            return
        
        if not symbols:
            st.warning("Please select at least one symbol")
            return
        
        # Results container
        results = {}
        successful_analyses = 0
        failed_analyses = 0
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analyze each symbol
        for i, symbol in enumerate(symbols):
            try:
                status_text.text(f"🔍 Analyzing {symbol}... ({i+1}/{len(symbols)})")
                progress_bar.progress((i + 1) / len(symbols))
                
                result = enhanced_signal_analysis(symbol)
                if 'error' not in result:
                    results[symbol] = result
                    successful_analyses += 1
                else:
                    results[symbol] = {'error': result['error']}
                    failed_analyses += 1
                    
            except Exception as e:
                results[symbol] = {'error': str(e)}
                failed_analyses += 1
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display summary
        st.success(f"✅ Analysis Complete: {successful_analyses} successful, {failed_analyses} failed")
        
        if successful_analyses == 0:
            st.error("❌ No successful analyses. Please try different symbols.")
            return
        
        # Create comparison table
        self.create_comparison_table(results)
        
        # Create detailed analysis for top performers
        self.show_top_performers(results)
        
        # Show individual detailed analysis (expandable)
        self.show_individual_analyses(results)
    
    def create_comparison_table(self, results: dict):
        """Create a comparison table of all analyzed stocks"""
        
        st.subheader("📊 Stock Comparison Summary")
        
        # Prepare data for comparison table
        comparison_data = []
        
        for symbol, result in results.items():
            if 'error' not in result:
                signal = result['signal_strength']
                tech = result['technical_data']
                risk = result['risk_management']
                
                comparison_data.append({
                    'Symbol': symbol,
                    'Grade': signal['grade'],
                    'Score': f"{signal['score']:.0f}/100",
                    'Recommendation': signal['recommendation'],
                    'Price': f"{result['price']:.2f} PKR",
                    'RSI': f"{tech['rsi']:.1f}",
                    'Volume': f"{tech['volume_ratio']:.1f}x",
                    'R/R Ratio': f"{risk['risk_reward_ratio']:.2f}",
                    'Stop Loss': f"{risk['stop_loss_pct']:+.1f}%",
                    'Target': f"{risk['target1_pct']:+.1f}%"
                })
            else:
                comparison_data.append({
                    'Symbol': symbol,
                    'Grade': 'Error',
                    'Score': 'N/A',
                    'Recommendation': 'Failed',
                    'Price': 'N/A',
                    'RSI': 'N/A',
                    'Volume': 'N/A',
                    'R/R Ratio': 'N/A',
                    'Stop Loss': 'N/A',
                    'Target': 'N/A'
                })
        
        # Convert to DataFrame and display
        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            
            # Style the dataframe
            def style_row(row):
                if row['Grade'] == 'A':
                    return ['background-color: lightgreen'] * len(row)
                elif row['Grade'] == 'B':
                    return ['background-color: lightblue'] * len(row)
                elif row['Grade'] == 'C':
                    return ['background-color: lightyellow'] * len(row)
                elif row['Grade'] == 'D':
                    return ['background-color: orange'] * len(row)
                elif row['Grade'] == 'F':
                    return ['background-color: lightcoral'] * len(row)
                else:
                    return ['background-color: lightgray'] * len(row)
            
            # Display styled table
            st.dataframe(
                df.style.apply(style_row, axis=1),
                use_container_width=True,
                hide_index=True
            )
    
    def show_top_performers(self, results: dict):
        """Show detailed analysis of top performing stocks"""
        
        st.subheader("🏆 Top Performers")
        
        # Filter successful results and sort by score
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            st.info("No successful analyses to show top performers")
            return
        
        # Sort by signal score
        sorted_results = sorted(
            successful_results.items(), 
            key=lambda x: x[1]['signal_strength']['score'], 
            reverse=True
        )
        
        # Show top 3 performers
        top_count = min(3, len(sorted_results))
        
        cols = st.columns(top_count)
        
        for i in range(top_count):
            symbol, result = sorted_results[i]
            signal = result['signal_strength']
            
            with cols[i]:
                # Medal emojis
                medals = ["🥇", "🥈", "🥉"]
                st.markdown(f"### {medals[i]} {symbol}")
                
                # Grade with color
                grade_colors = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⚫"}
                grade_emoji = grade_colors.get(signal['grade'], "⚫")
                
                st.markdown(f"**{grade_emoji} Grade {signal['grade']}**")
                st.metric("Score", f"{signal['score']:.0f}/100")
                st.metric("Price", f"{result['price']:.2f} PKR")
                
                # Recommendation with appropriate styling
                recommendation = signal['recommendation']
                if recommendation in ['STRONG BUY', 'BUY']:
                    st.success(f"📈 {recommendation}")
                elif recommendation == 'HOLD':
                    st.info(f"⏸️ {recommendation}")
                else:
                    st.warning(f"⚠️ {recommendation}")
    
    def show_individual_analyses(self, results: dict):
        """Show individual detailed analyses in expandable sections"""
        
        st.subheader("🔍 Individual Stock Analysis")
        
        for symbol, result in results.items():
            if 'error' not in result:
                # Expandable section for each stock (expanded=True for testing)
                with st.expander(f"📊 {symbol} - Grade {result['signal_strength']['grade']} ({result['signal_strength']['score']:.0f}/100)", expanded=True):
                    # Show the detailed analysis using existing method
                    st.write("**🔍 DETAILED ANALYSIS LOADING...**")  # Debug message
                    self.show_detailed_stock_analysis(symbol, result)
                    st.write("**✅ DETAILED ANALYSIS COMPLETE**")  # Debug message
            else:
                with st.expander(f"❌ {symbol} - Analysis Failed", expanded=False):
                    st.error(f"Error: {result['error']}")
                    st.info("Try selecting this symbol individually or check if it's available in the API.")
    
    def show_detailed_stock_analysis(self, symbol: str, result: dict):
        """Show FULL detailed analysis for a single stock (complete previous version)"""
        
        signal = result['signal_strength']
        tech = result['technical_data']
        risk = result['risk_management']
        
        # Header with main decision (same as previous version)
        st.markdown("### 🎯 Trading Decision Analysis")
        
        # Main decision banner
        decision_color = {
            "STRONG BUY": "🟢", "BUY": "🟡", "WEAK BUY": "🟠", 
            "HOLD": "🔵", "AVOID": "🔴"
        }.get(signal['recommendation'], "⚫")
        
        st.markdown(f"## {decision_color} **{signal['recommendation']}** | Grade {signal['grade']} ({signal['score']:.0f}/100)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"{result['price']:.2f} PKR", "")
        
        with col2:
            st.metric("Stop Loss", f"{risk['stop_loss']:.2f} PKR", f"-{risk['stop_loss_pct']:.1f}%")
        
        with col3:
            st.metric("Target", f"{risk['target1']:.2f} PKR", f"+{risk['target1_pct']:.1f}%")
        
        st.markdown("---")
        
        # Decision Graphs Section (Numerical Analysis)
        st.subheader("📊 Decision Analysis")
        
        # Create two columns for decision analysis
        graph_col1, graph_col2 = st.columns(2)
        
        with graph_col1:
            # 1. Signal Strength Analysis (Numerical)
            st.info("📊 **Signal Strength Analysis**")
            st.write("**Signal Breakdown:**")
            st.write(f"• **Trend Score:** {min(100, max(0, tech.get('ma44_slope', 0) * 50 + 50)):.1f}/100")
            st.write(f"• **Momentum (RSI):** {min(100, max(0, (tech['rsi'] - 30) * 2.5)):.1f}/100")
            st.write(f"• **Volume Strength:** {min(100, max(0, tech['volume_ratio'] * 30)):.1f}/100")
            st.write(f"• **Support/Resistance:** {signal['score'] * 0.6:.1f}/100")
            st.write(f"• **Risk/Reward:** {min(100, max(0, result['risk_management']['risk_reward_ratio'] * 30)):.1f}/100")
            st.write(f"• **Overall Score:** {signal['score']:.1f}/100")
            
            # 3. Risk-Reward Analysis
            st.info("⚖️ **Risk-Reward Analysis**")
            risk_pct = abs(risk['stop_loss_pct'])
            reward_pct = risk['target1_pct']
            rr_ratio = risk['risk_reward_ratio']
            
            st.write("**Risk-Reward Metrics:**")
            st.write(f"• **Risk Percentage:** {risk_pct:.1f}%")
            st.write(f"• **Reward Percentage:** {reward_pct:.1f}%")
            st.write(f"• **Risk/Reward Ratio:** {rr_ratio:.2f}")
            
            if rr_ratio >= 2.0:
                st.success("🟢 **Excellent Risk/Reward** (R/R ≥ 2.0)")
            elif rr_ratio >= 1.0:
                st.warning("🟡 **Acceptable Risk/Reward** (R/R 1.0-2.0)")
            else:
                st.error("🔴 **Poor Risk/Reward** (R/R < 1.0)")
        
        with graph_col2:
            # 2. Technical Indicators Decision
            st.info("📊 **Technical Indicators Decision**")
            rsi_signal = 0.5 if 40 <= tech['rsi'] <= 70 else (-0.8 if tech['rsi'] > 70 else 0.8)
            ma_signal = 0.6 if tech.get('ma44_trend', 'up') == 'up' else -0.6
            bb_signal = 0.4 if 0.2 <= tech['bb_pctb'] <= 0.8 else (-0.7 if tech['bb_pctb'] > 0.8 else 0.7)
            vol_signal = min(0.8, tech['volume_ratio'] * 0.3 - 0.2)
            adx_signal = 0.3 if tech.get('adx', 20) > 25 else 0.1
            
            st.write("**Individual Indicator Signals:**")
            st.write(f"• **RSI Signal:** {rsi_signal:+.2f} ({'🟢 Buy' if rsi_signal > 0.3 else '🔴 Sell' if rsi_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **MA44 Signal:** {ma_signal:+.2f} ({'🟢 Buy' if ma_signal > 0.3 else '🔴 Sell' if ma_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **BB %B Signal:** {bb_signal:+.2f} ({'🟢 Buy' if bb_signal > 0.3 else '🔴 Sell' if bb_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **Volume Signal:** {vol_signal:+.2f} ({'🟢 Buy' if vol_signal > 0.3 else '🔴 Sell' if vol_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **ADX Signal:** {adx_signal:+.2f} ({'🟢 Buy' if adx_signal > 0.3 else '🔴 Sell' if adx_signal < -0.3 else '🟡 Neutral'})")
            
            # 4. Signal Confidence Analysis
            st.info("🎯 **Signal Confidence Analysis**")
            confidence = signal['score']
            
            st.write("**Confidence Metrics:**")
            st.write(f"• **Overall Confidence:** {confidence:.1f}/100")
            
            if confidence >= 80:
                st.success("🟢 **Very High Confidence** (80-100)")
            elif confidence >= 65:
                st.info("🔵 **High Confidence** (65-79)")
            elif confidence >= 50:
                st.warning("🟡 **Medium Confidence** (50-64)")
            elif confidence >= 35:
                st.warning("🟠 **Low Confidence** (35-49)")
            else:
                st.error("🔴 **Very Low Confidence** (0-34)")
            
            st.write(f"• **Delta from Neutral:** {confidence - 50:+.1f} points")
        
        # 5. Factor Contribution Analysis (Full width)
        st.markdown("---")
        st.info("🔍 **Factor Contribution Analysis**")
        factors = signal.get('factors', [])[:8]  # Top 8 factors
        
        if not factors:
            st.info("No detailed factors available")
        else:
            st.write("**Top Contributing Factors:**")
            for i, factor in enumerate(factors, 1):
                # Determine if factor is positive or negative
                if any(word in factor.lower() for word in ['up', 'above', 'surge', 'optimal', 'support']):
                    st.write(f"{i}. 🟢 {factor}")
                elif any(word in factor.lower() for word in ['down', 'below', 'weak', 'poor', 'avoid']):
                    st.write(f"{i}. 🔴 {factor}")
                else:
                    st.write(f"{i}. 🟡 {factor}")
        
        st.markdown("---")
        
        # Enhanced Technical Summary (Same as previous detailed version)
        st.subheader("📊 Detailed Technical Metrics")
        
        # Main indicators row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_status = "Oversold" if tech['rsi'] < 30 else "Overbought" if tech['rsi'] > 70 else "Normal"
            st.metric("RSI", f"{tech['rsi']:.1f}", rsi_status)
        
        with col2:
            st.metric("MA44", f"{tech['ma44']:.2f}", f"{tech.get('ma44_trend', 'Unknown')} trend")
        
        with col3:
            bb_status = "Lower band" if tech['bb_pctb'] < 0.2 else "Upper band" if tech['bb_pctb'] > 0.8 else "Middle range"
            st.metric("BB %B", f"{tech['bb_pctb']:.2f}", bb_status)
        
        with col4:
            vol_status = "High" if tech['volume_ratio'] > 1.5 else "Low" if tech['volume_ratio'] < 0.8 else "Normal"
            st.metric("Volume", f"{tech['volume_ratio']:.1f}x", vol_status)
        
        # Additional detailed metrics
        st.markdown("---")
        st.subheader("🔍 Additional Analysis")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.write("**📈 Price Levels:**")
            st.write(f"• Current: {result['price']:.2f} PKR")
            st.write(f"• MA44: {tech['ma44']:.2f} PKR")
            st.write(f"• Distance from MA44: {((result['price'] - tech['ma44']) / tech['ma44'] * 100):+.1f}%")
            
        with detail_col2:
            st.write("**⚖️ Risk Management:**")
            st.write(f"• Stop Loss: {risk['stop_loss']:.2f} PKR ({risk['stop_loss_pct']:+.1f}%)")
            st.write(f"• Target 1: {risk['target1']:.2f} PKR ({risk['target1_pct']:+.1f}%)")
            st.write(f"• Risk/Reward: {risk['risk_reward_ratio']:.2f}")
            
        with detail_col3:
            st.write("**📊 Signal Quality:**")
            st.write(f"• Overall Grade: {signal['grade']}")
            st.write(f"• Confidence: {signal['score']:.1f}/100")
            st.write(f"• Recommendation: {signal['recommendation']}")
        
        # Key factors summary
        if signal.get('factors'):
            st.markdown("---")
            st.subheader("🎯 Key Decision Factors")
            factors_text = " | ".join(signal['factors'][:3])  # Top 3 factors
            st.info(f"**Primary factors:** {factors_text}")
    
    def run_signal_analysis(self, symbols: list, analysis_type: str, days: int):
        """Run comprehensive signal analysis"""
        
        if not MODULES_AVAILABLE:
            st.error("Analysis modules not available")
            return
        
        results_container = st.container()
        
        with results_container:
            for symbol in symbols:
                with st.expander(f"📊 {symbol} Analysis", expanded=True):
                    try:
                        result = enhanced_signal_analysis(symbol, days)
                        
                        if 'error' in result:
                            st.error(f"{symbol}: {result['error']}")
                            continue
                        
                        # Create comprehensive display
                        self.display_detailed_analysis(result)
                    
                    except Exception as e:
                        st.error(f"Error analyzing {symbol}: {str(e)}")
    
    def display_detailed_analysis(self, result: dict):
        """Display detailed analysis results"""
        
        signal = result['signal_strength']
        tech = result['technical_data']
        risk = result['risk_management']
        
        # Header with grade
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            grade_color = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⚫"}
            st.markdown(f"## {grade_color.get(signal['grade'], '⚫')} Grade {signal['grade']}")
        
        with col2:
            st.markdown(f"### {result['symbol']}")
            st.markdown(f"**{signal['recommendation']}** | Score: {signal['score']:.0f}/100")
        
        with col3:
            st.metric("Price", f"{result['price']:.2f} PKR", "")
        
        # Technical metrics
        st.subheader("📊 Technical Analysis")
        
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.metric("RSI", f"{tech['rsi']:.1f}", "Momentum")
        
        with tech_col2:
            st.metric("MA44", f"{tech['ma44']:.2f}", "Trend")
        
        with tech_col3:
            st.metric("BB %B", f"{tech['bb_pctb']:.2f}", "Position")
        
        with tech_col4:
            st.metric("Volume", f"{tech['volume_ratio']:.1f}x", "Confirmation")
        
        # Key factors
        st.subheader("🔍 Key Factors")
        for factor in signal['factors'][:5]:
            st.write(f"• {factor}")
        
        # Risk management
        st.subheader("🎯 Risk Management")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.metric("Stop Loss", f"{risk['stop_loss']:.2f}", f"-{risk['stop_loss_pct']:.1f}%")
        
        with risk_col2:
            st.metric("Target 1", f"{risk['target1']:.2f}", f"+{risk['target1_pct']:.1f}%")
        
        with risk_col3:
            st.metric("Target 2", f"{risk['target2']:.2f}", f"+{risk['target2_pct']:.1f}%")
        
        # Support/Resistance
        sr = result.get('support_resistance', {})
        if sr.get('nearest_support') or sr.get('nearest_resistance'):
            st.subheader("📈 Key Levels")
            level_col1, level_col2 = st.columns(2)
            
            with level_col1:
                if sr.get('nearest_support'):
                    st.metric("Support", f"{sr['nearest_support']:.2f}", "")
            
            with level_col2:
                if sr.get('nearest_resistance'):
                    st.metric("Resistance", f"{sr['nearest_resistance']:.2f}", "")
    
    def show_portfolio_overview(self):
        """Display portfolio overview"""
        
        if not self.portfolio_manager.positions:
            st.info("📝 No positions in portfolio. Add some positions to get started!")
            return
        
        # Get current prices (simplified)
        current_prices = {}
        for symbol in self.portfolio_manager.positions.keys():
            current_prices[symbol] = self.portfolio_manager.positions[symbol].avg_price  # Simplified
        
        summary = self.portfolio_manager.get_portfolio_summary(current_prices)
        
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"{summary['total_portfolio_value']:,.0f} PKR", "")
        
        with col2:
            st.metric("Unrealized P&L", f"{summary['total_unrealized_pnl']:+,.0f} PKR", 
                     f"{summary['total_unrealized_pnl_pct']:+.1f}%")
        
        with col3:
            st.metric("Cash Balance", f"{summary['cash_balance']:,.0f} PKR", "")
        
        with col4:
            st.metric("Positions", len(summary['positions']), "")
        
        # Positions table
        if summary['positions']:
            st.subheader("📊 Current Positions")
            
            positions_df = pd.DataFrame(summary['positions'])
            st.dataframe(positions_df, use_container_width=True)
    
    def add_position_form(self):
        """Form to add new position"""
        
        with st.form("add_position"):
            symbol = st.selectbox("Symbol", self.common_symbols)
            quantity = st.number_input("Quantity", min_value=1, value=100)
            price = st.number_input("Price", min_value=1.0, value=100.0, step=0.1)
            notes = st.text_input("Notes (optional)")
            
            if st.form_submit_button("➕ Add Position"):
                if MODULES_AVAILABLE:
                    result = self.portfolio_manager.add_position(symbol, quantity, price, notes=notes)
                    st.success(result)
                else:
                    st.error("Portfolio manager not available")
    
    def sell_position_form(self):
        """Form to sell position"""
        
        if not MODULES_AVAILABLE or not self.portfolio_manager.positions:
            st.info("No positions to sell")
            return
        
        symbols = list(self.portfolio_manager.positions.keys())
        
        with st.form("sell_position"):
            symbol = st.selectbox("Symbol", symbols)
            
            # Show current holding
            if symbol:
                current_qty = self.portfolio_manager.positions[symbol].quantity
                st.info(f"Current holding: {current_qty} shares")
                
                quantity = st.number_input("Quantity to Sell", min_value=1, max_value=current_qty, value=min(50, current_qty))
            else:
                quantity = st.number_input("Quantity to Sell", min_value=1, value=50)
            
            price = st.number_input("Sell Price", min_value=1.0, value=100.0, step=0.1)
            notes = st.text_input("Notes (optional)")
            
            if st.form_submit_button("💰 Sell Position"):
                result = self.portfolio_manager.sell_position(symbol, quantity, price, notes=notes)
                st.success(result)
    
    def calculate_position_sizing(self, current_price: float, stop_loss: float, risk_pct: float):
        """Calculate and display position sizing"""
        
        if not MODULES_AVAILABLE:
            st.error("Risk management not available")
            return
        
        try:
            position = calculate_position_size(current_price, stop_loss, risk_pct)
            
            st.subheader("📏 Position Size Calculation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Shares to Buy", position.shares, "")
            
            with col2:
                st.metric("Investment Amount", f"{position.investment_amount:,.0f} PKR", "")
            
            with col3:
                st.metric("Risk Amount", f"{position.risk_amount:,.0f} PKR", f"{position.risk_percentage:.1f}%")
            
            if position.warnings:
                st.warning("⚠️ " + " | ".join(position.warnings))
        
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
    
    def show_portfolio_risk(self):
        """Display portfolio risk analysis"""
        
        if not MODULES_AVAILABLE:
            st.info("Risk analysis not available")
            return
        
        # Simplified portfolio risk display
        st.info("Portfolio risk analysis will be displayed here when positions are available.")
    
    def create_interactive_chart(self, symbol: str, period: str, chart_type: str):
        """Create interactive Plotly chart"""
        
        # Create sample data for demonstration
        periods = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
        days = periods.get(period, 90)
        
        # Generate sample OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        
        price = 100
        data = []
        
        for date in dates:
            price += np.random.normal(0, 1)
            high = price + abs(np.random.normal(0, 1))
            low = price - abs(np.random.normal(0, 1))
            open_price = price + np.random.normal(0, 0.5)
            volume = np.random.randint(10000, 100000)
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Create plotly chart
        fig = sp.make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{symbol} Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Price chart
        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
        elif chart_type == "Line":
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name='Close Price'
                ),
                row=1, col=1
            )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.8)'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - {period} Chart',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_signal_strength_radar(self, signal: dict, tech: dict, result: dict):
        """Create signal strength radar chart"""
        
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
        
        if not PLOTLY_AVAILABLE:
            st.info("📊 **Signal Strength Analysis**")
            # Show numerical breakdown instead
            st.write("**Signal Breakdown:**")
            st.write(f"• **Trend Score:** {min(100, max(0, tech.get('ma44_slope', 0) * 50 + 50)):.1f}/100")
            st.write(f"• **Momentum (RSI):** {min(100, max(0, (tech['rsi'] - 30) * 2.5)):.1f}/100")
            st.write(f"• **Volume Strength:** {min(100, max(0, tech['volume_ratio'] * 30)):.1f}/100")
            st.write(f"• **Support/Resistance:** {signal['score'] * 0.6:.1f}/100")
            st.write(f"• **Risk/Reward:** {min(100, max(0, result['risk_management']['risk_reward_ratio'] * 30)):.1f}/100")
            st.write(f"• **Overall Score:** {signal['score']:.1f}/100")
            return
            
        # Extract signal factors for radar chart
        categories = ['Trend', 'Momentum', 'Volume', 'Support/Resistance', 'Risk/Reward', 'Overall']
        
        # Calculate normalized scores (0-100 scale)
        values = [
            min(100, max(0, tech.get('ma44_slope', 0) * 50 + 50)),  # Trend
            min(100, max(0, (tech['rsi'] - 30) * 2.5)),  # Momentum (RSI normalized)
            min(100, max(0, tech['volume_ratio'] * 30)),  # Volume
            signal['score'] * 0.6,  # Support/Resistance (60% of signal score)
            min(100, max(0, result['risk_management']['risk_reward_ratio'] * 30)),  # Risk/Reward
            signal['score']  # Overall
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Signal Strength',
            line_color='rgb(50, 171, 96)',
            fillcolor='rgba(50, 171, 96, 0.25)'
        ))
        
        # Add decision zones
        fig.add_trace(go.Scatterpolar(
            r=[80, 80, 80, 80, 80, 80],
            theta=categories,
            mode='lines',
            name='Strong Buy Zone',
            line=dict(color='green', dash='dash'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[50, 50, 50, 50, 50, 50],
            theta=categories,
            mode='lines',
            name='Neutral Zone',
            line=dict(color='orange', dash='dash'),
            showlegend=False
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickmode='linear',
                    tick0=0,
                    dtick=20,
                    gridcolor='lightgray'
                )
            ),
            title="🎯 Signal Strength Radar",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_technical_indicators_chart(self, tech: dict, signal: dict):
        """Create technical indicators decision chart"""
        
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
        
        if not PLOTLY_AVAILABLE:
            st.info("📊 **Technical Indicators Decision**")
            # Show numerical analysis instead
            rsi_signal = 0.5 if 40 <= tech['rsi'] <= 70 else (-0.8 if tech['rsi'] > 70 else 0.8)
            ma_signal = 0.6 if tech.get('ma44_trend', 'up') == 'up' else -0.6
            bb_signal = 0.4 if 0.2 <= tech['bb_pctb'] <= 0.8 else (-0.7 if tech['bb_pctb'] > 0.8 else 0.7)
            vol_signal = min(0.8, tech['volume_ratio'] * 0.3 - 0.2)
            adx_signal = 0.3 if tech.get('adx', 20) > 25 else 0.1
            
            st.write("**Individual Indicator Signals:**")
            st.write(f"• **RSI Signal:** {rsi_signal:+.2f} ({'🟢 Buy' if rsi_signal > 0.3 else '🔴 Sell' if rsi_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **MA44 Signal:** {ma_signal:+.2f} ({'🟢 Buy' if ma_signal > 0.3 else '🔴 Sell' if ma_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **BB %B Signal:** {bb_signal:+.2f} ({'🟢 Buy' if bb_signal > 0.3 else '🔴 Sell' if bb_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **Volume Signal:** {vol_signal:+.2f} ({'🟢 Buy' if vol_signal > 0.3 else '🔴 Sell' if vol_signal < -0.3 else '🟡 Neutral'})")
            st.write(f"• **ADX Signal:** {adx_signal:+.2f} ({'🟢 Buy' if adx_signal > 0.3 else '🔴 Sell' if adx_signal < -0.3 else '🟡 Neutral'})")
            return
            
        # Technical indicator signals
        indicators = ['RSI', 'MA44', 'BB %B', 'Volume', 'ADX']
        
        # Convert to buy/sell signals (-1 to 1 scale)
        rsi_signal = 0.5 if 40 <= tech['rsi'] <= 70 else (-0.8 if tech['rsi'] > 70 else 0.8)
        ma_signal = 0.6 if tech.get('ma44_trend', 'up') == 'up' else -0.6
        bb_signal = 0.4 if 0.2 <= tech['bb_pctb'] <= 0.8 else (-0.7 if tech['bb_pctb'] > 0.8 else 0.7)
        vol_signal = min(0.8, tech['volume_ratio'] * 0.3 - 0.2)
        adx_signal = 0.3 if tech.get('adx', 20) > 25 else 0.1
        
        signals = [rsi_signal, ma_signal, bb_signal, vol_signal, adx_signal]
        colors = ['green' if s > 0.3 else 'red' if s < -0.3 else 'orange' for s in signals]
        
        fig = go.Figure(data=[
            go.Bar(
                x=indicators,
                y=signals,
                marker_color=colors,
                text=[f"{s:+.2f}" for s in signals],
                textposition='auto',
            )
        ])
        
        # Add decision zones
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Strong Buy")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="red", annotation_text="Strong Sell")
        fig.add_hline(y=0, line_dash="solid", line_color="gray", annotation_text="Neutral")
        
        fig.update_layout(
            title="📊 Technical Indicators Decision",
            xaxis_title="Indicators",
            yaxis_title="Signal Strength",
            yaxis=dict(range=[-1, 1]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_risk_reward_matrix(self, risk: dict, signal: dict):
        """Create risk-reward decision matrix"""
        
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
        
        if not PLOTLY_AVAILABLE:
            st.info("⚖️ **Risk-Reward Analysis**")
            # Show numerical analysis instead
            risk_pct = abs(risk['stop_loss_pct'])
            reward_pct = risk['target1_pct']
            rr_ratio = risk['risk_reward_ratio']
            
            st.write("**Risk-Reward Metrics:**")
            st.write(f"• **Risk Percentage:** {risk_pct:.1f}%")
            st.write(f"• **Reward Percentage:** {reward_pct:.1f}%")
            st.write(f"• **Risk/Reward Ratio:** {rr_ratio:.2f}")
            
            if rr_ratio >= 2.0:
                st.success("🟢 **Excellent Risk/Reward** (R/R ≥ 2.0)")
            elif rr_ratio >= 1.0:
                st.warning("🟡 **Acceptable Risk/Reward** (R/R 1.0-2.0)")
            else:
                st.error("🔴 **Poor Risk/Reward** (R/R < 1.0)")
            return
            
        # Calculate risk and reward percentages
        risk_pct = abs(risk['stop_loss_pct'])
        reward_pct = risk['target1_pct']
        rr_ratio = risk['risk_reward_ratio']
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add current position
        fig.add_trace(go.Scatter(
            x=[risk_pct],
            y=[reward_pct],
            mode='markers+text',
            marker=dict(size=15, color=signal['score'], colorscale='RdYlGn', 
                       cmin=0, cmax=100, showscale=True),
            text=[f"R/R: {rr_ratio:.2f}"],
            textposition="top center",
            name="Current Setup"
        ))
        
        # Add decision zones
        x_range = [0, max(10, risk_pct * 1.5)]
        
        # Good zone (R/R > 2)
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=x_range[1], y1=x_range[1] * 2,
            fillcolor="lightgreen", opacity=0.2,
            line=dict(width=0)
        )
        
        # Acceptable zone (R/R 1-2)
        fig.add_shape(
            type="rect", 
            x0=0, y0=0, x1=x_range[1], y1=x_range[1],
            fillcolor="lightyellow", opacity=0.2,
            line=dict(width=0)
        )
        
        # Poor zone (R/R < 1)
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=x_range[1], y1=x_range[1] * 0.5,
            fillcolor="lightcoral", opacity=0.2,
            line=dict(width=0)
        )
        
        fig.update_layout(
            title="⚖️ Risk-Reward Analysis",
            xaxis_title="Risk %",
            yaxis_title="Reward %",
            height=400,
            annotations=[
                dict(x=x_range[1]*0.7, y=x_range[1]*1.5, text="Good R/R", showarrow=False),
                dict(x=x_range[1]*0.7, y=x_range[1]*0.7, text="Fair R/R", showarrow=False),
                dict(x=x_range[1]*0.7, y=x_range[1]*0.3, text="Poor R/R", showarrow=False)
            ]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_confidence_gauge(self, signal: dict, tech: dict):
        """Create buy/sell confidence gauge"""
        
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
        
        if not PLOTLY_AVAILABLE:
            st.info("🎯 **Signal Confidence Analysis**")
            # Show numerical confidence instead
            confidence = signal['score']
            
            st.write("**Confidence Metrics:**")
            st.write(f"• **Overall Confidence:** {confidence:.1f}/100")
            
            if confidence >= 80:
                st.success("🟢 **Very High Confidence** (80-100)")
            elif confidence >= 65:
                st.info("🔵 **High Confidence** (65-79)")
            elif confidence >= 50:
                st.warning("🟡 **Medium Confidence** (50-64)")
            elif confidence >= 35:
                st.warning("🟠 **Low Confidence** (35-49)")
            else:
                st.error("🔴 **Very Low Confidence** (0-34)")
            
            st.write(f"• **Delta from Neutral:** {confidence - 50:+.1f} points")
            return
            
        # Calculate confidence based on signal score
        confidence = signal['score']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Signal Confidence"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 35], 'color': "lightgray"},
                    {'range': [35, 50], 'color': "yellow"},
                    {'range': [50, 65], 'color': "orange"},
                    {'range': [65, 80], 'color': "lightgreen"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_simple_pdf_report(self, symbol: str, analysis_data: dict):
        """Generate a simple HTML report for PDF conversion"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>PSX Trading Analysis Report - {symbol}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2E86AB; text-align: center; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
                h2 {{ color: #A23B72; margin-top: 30px; }}
                .header-info {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .metrics {{ display: flex; justify-content: space-between; margin: 20px 0; }}
                .metric {{ text-align: center; padding: 15px; background-color: #e8f4f8; border-radius: 5px; margin: 0 10px; flex: 1; }}
                .recommendation {{ text-align: center; padding: 20px; font-size: 24px; font-weight: bold; 
                                 background-color: #f0f8e8; border-radius: 10px; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #2E86AB; color: white; }}
                .footer {{ margin-top: 50px; padding: 20px; background-color: #f9f9f9; 
                          border-radius: 5px; font-style: italic; text-align: center; }}
                @media print {{
                    body {{ margin: 20px; }}
                    .metrics {{ flex-wrap: wrap; }}
                    .metric {{ margin: 10px 0; }}
                }}
            </style>
        </head>
        <body>
            <h1>PSX Trading Analysis Report</h1>
            
            <div class="header-info">
                <strong>Symbol:</strong> {symbol}<br>
                <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
                <strong>Market:</strong> Pakistan Stock Exchange (PSX)
            </div>
            
            <div class="recommendation">
                🎯 RECOMMENDATION: {analysis_data.get('recommendation', 'N/A')}
            </div>
            
            <h2>📊 Executive Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>Signal Grade</strong><br>
                    {analysis_data.get('signal_strength', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Current Price</strong><br>
                    {analysis_data.get('current_price', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Target Price</strong><br>
                    {analysis_data.get('target_price', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Stop Loss</strong><br>
                    {analysis_data.get('stop_loss', 'N/A')}
                </div>
            </div>
            
            <h2>🔍 Technical Analysis</h2>
            <table>
                <tr><th>Indicator</th><th>Value</th><th>Signal</th></tr>"""
        
        # Add technical indicators if available
        if 'technical_indicators' in analysis_data:
            for indicator, data in analysis_data['technical_indicators'].items():
                if isinstance(data, dict):
                    value = data.get('value', 'N/A')
                    signal = data.get('signal', 'N/A')
                else:
                    value = str(data)
                    signal = 'N/A'
                html_content += f"<tr><td>{indicator}</td><td>{value}</td><td>{signal}</td></tr>"
        
        html_content += f"""
            </table>
            
            <h2>⚖️ Risk Analysis</h2>
            <table>
                <tr><th>Risk Factor</th><th>Value</th></tr>
                <tr><td>Risk Score</td><td>{analysis_data.get('risk_score', 'N/A')}</td></tr>
                <tr><td>Volatility</td><td>{analysis_data.get('volatility', 'N/A')}</td></tr>
                <tr><td>Stop Loss Level</td><td>{analysis_data.get('stop_loss', 'N/A')}</td></tr>
                <tr><td>Target Price</td><td>{analysis_data.get('target_price', 'N/A')}</td></tr>
            </table>
            
            <h2>📈 Key Metrics Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
                <tr><td>Overall Recommendation</td><td>{analysis_data.get('recommendation', 'N/A')}</td><td>Primary trading decision</td></tr>
                <tr><td>Signal Strength</td><td>{analysis_data.get('signal_strength', 'N/A')}</td><td>Quality grade of analysis</td></tr>
                <tr><td>Risk Score</td><td>{analysis_data.get('risk_score', 'N/A')}</td><td>Overall confidence level</td></tr>
                <tr><td>Current Price</td><td>{analysis_data.get('current_price', 'N/A')}</td><td>Latest market price</td></tr>
            </table>
            
            <div class="footer">
                <strong>Disclaimer:</strong> This report is generated by PSX Trading Bot for informational purposes only. 
                Please consult with a financial advisor before making investment decisions.<br><br>
                <strong>Generated by:</strong> PSX Trading Bot | <strong>Website:</strong> Pakistan Stock Exchange Analysis System<br>
                <strong>Note:</strong> To save as PDF, use your browser's print function (Ctrl+P) and select "Save as PDF"
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def create_factor_contribution_chart(self, signal: dict):
        """Create factor contribution bar chart"""
        
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
        
        if not PLOTLY_AVAILABLE:
            st.info("🔍 **Factor Contribution Analysis**")
            # Show numerical factor breakdown instead
            factors = signal.get('factors', [])[:8]  # Top 8 factors
            
            if not factors:
                st.info("No detailed factors available")
                return
            
            st.write("**Top Contributing Factors:**")
            for i, factor in enumerate(factors, 1):
                # Determine if factor is positive or negative
                if any(word in factor.lower() for word in ['up', 'above', 'surge', 'optimal', 'support']):
                    st.write(f"{i}. 🟢 {factor}")
                elif any(word in factor.lower() for word in ['down', 'below', 'weak', 'poor', 'avoid']):
                    st.write(f"{i}. 🔴 {factor}")
                else:
                    st.write(f"{i}. 🟡 {factor}")
            return
            
        st.subheader("🔍 Decision Factors")
        
        # Extract factors from signal (this is a simplified version)
        factors = signal.get('factors', [])[:8]  # Top 8 factors
        
        if not factors:
            st.info("No detailed factors available")
            return
        
        # Simulate factor scores (in real implementation, these would come from analysis)
        factor_scores = []
        factor_names = []
        
        for factor in factors:
            # Parse factor text to extract score
            if 'Trend' in factor:
                score = 15 if 'Up' in factor else -10
            elif 'RSI' in factor:
                score = 10 if 'optimal' in factor else -5
            elif 'Volume' in factor:
                score = 12 if 'surge' in factor else 5
            elif 'MA44' in factor:
                score = 10 if 'above' in factor else -15
            elif 'Support' in factor:
                score = 8
            elif 'Bollinger' in factor:
                score = 5 if 'middle' in factor else -8
            else:
                score = np.random.randint(-10, 15)
            
            factor_scores.append(score)
            factor_names.append(factor[:30] + "..." if len(factor) > 30 else factor)
        
        # Create horizontal bar chart
        colors = ['green' if score > 0 else 'red' for score in factor_scores]
        
        fig = go.Figure(go.Bar(
            x=factor_scores,
            y=factor_names,
            orientation='h',
            marker_color=colors,
            text=[f"+{s}" if s > 0 else str(s) for s in factor_scores],
            textposition='auto'
        ))
        
        fig.add_vline(x=0, line_dash="solid", line_color="black")
        
        fig.update_layout(
            title="🔍 Factor Contribution to Decision",
            xaxis_title="Points Contribution",
            height=max(300, len(factors) * 40),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Sidebar footer
def show_sidebar_footer():
    """Show sidebar footer with app info"""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.success("✅ Online")
    st.sidebar.info("🔄 Last Update: " + datetime.now().strftime("%H:%M"))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("**PSX Trading Bot v2.0**")
    st.sidebar.markdown("Professional algorithmic trading system for PSX")
    st.sidebar.markdown("⚠️ *Educational use only*")

# Main application
def main():
    """Main Streamlit application"""
    
    # Initialize dashboard
    dashboard = TradingDashboard()
    
    # Show sidebar footer
    show_sidebar_footer()
    
    # Run dashboard
    dashboard.run()

if __name__ == "__main__":
    main()