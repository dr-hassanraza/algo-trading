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

# Add minimal stubs for remaining methods to prevent errors
    def show_signal_analysis(self):
        st.title("📊 Signal Analysis")
        st.info("Signal analysis functionality - under development")
    
    def show_portfolio(self):
        st.title("💼 Portfolio")
        st.info("Portfolio functionality - under development")
    
    def show_settings(self):
        st.title("⚙️ Settings")
        st.info("Settings functionality - under development")
    
    def show_charts(self):
        st.title("📈 Charts")
        st.info("Charts functionality - under development")
    
    def show_risk_management(self):
        st.title("🎯 Risk Management")
        st.info("Risk management functionality - under development")

    def create_signal_strength_radar(self, signal, tech, result):
        st.info("📊 Signal Strength Analysis - Chart would appear here")
    
    def create_technical_indicators_chart(self, tech, signal):
        st.info("📊 Technical Indicators Chart - Chart would appear here")
    
    def create_risk_reward_matrix(self, risk, signal):
        st.info("⚖️ Risk-Reward Matrix - Chart would appear here")
    
    def create_confidence_gauge(self, signal, tech):
        st.info("🎯 Confidence Gauge - Chart would appear here")
    
    def create_factor_contribution_chart(self, signal):
        st.info("🔍 Factor Contribution Chart - Chart would appear here")

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