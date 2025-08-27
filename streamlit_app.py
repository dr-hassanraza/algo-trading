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
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Required modules not available: {e}")
    MODULES_AVAILABLE = False

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

        except Exception as e:
            # Handle any errors during analysis
            st.error(f"Error analyzing {symbol}: {str(e)}")

            # Create comprehensive report text
            report_content = f"""
PSX TRADING ANALYSIS REPORT
===========================
Symbol: {symbol}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Market: Pakistan Stock Exchange (PSX)

RECOMMENDATION: {signal['recommendation']}
Grade: {signal['grade']} ({signal['score']:.0f}/100)

CURRENT METRICS:
• Current Price: {result['price']:.2f} PKR
• Target Price: {risk['target1']:.2f} PKR
• Stop Loss: {risk['stop_loss']:.2f} PKR
• Risk/Reward Ratio: {risk['risk_reward_ratio']:.2f}

TECHNICAL ANALYSIS:
• RSI: {tech['rsi']:.1f} ({rsi_status})
• MA44: {tech['ma44']:.2f} PKR ({tech.get('ma44_trend', 'Unknown')} trend)
• Bollinger Bands %B: {tech['bb_pctb']:.2f} ({bb_status})
• Volume Ratio: {tech['volume_ratio']:.1f}x ({vol_status})

RISK MANAGEMENT:
• Stop Loss: {risk['stop_loss']:.2f} PKR ({risk['stop_loss_pct']:+.1f}%)
• Target 1: {risk['target1']:.2f} PKR ({risk['target1_pct']:+.1f}%)
• Distance from MA44: {((result['price'] - tech['ma44']) / tech['ma44'] * 100):+.1f}%

Generated by PSX Trading Bot
"""
            # Create download button
            b64 = base64.b64encode(report_content.encode()).decode()
            filename = f"PSX_Analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            href = f'<a href="data:text/plain;base64,{b64}" download="{filename}" style="display: inline-block; padding: 12px 20px; background-color: #ff4b4b; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">📄 Download Analysis Report</a>'

            st.success("✅ Report Generated Successfully!")
            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Report generation failed: {str(e)}")

with col2:
    st.info("💡 **Tip:** Downloads a detailed text report with all analysis data.")

    
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
                    # Handle any errors during analysis
                    st.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
                    
                    # Create comprehensive display
                    self.display_detailed_analysis(result)
                
                except Exception as e:
                    # Handle the error if any exception occurs
                    st.error(f"Error analyzing {symbol}: {str(e)}")
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
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ]
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_factor_contribution_chart(self, signal: dict):
        """Create factor contribution chart"""
        
        try:
            import plotly.graph_objects as go
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False
        
        if not PLOTLY_AVAILABLE:
            st.info("📊 **Factor Contribution Analysis**")
            # Show textual contribution breakdown
            st.write("**Factor Contributions:**")
            for factor, contribution in signal.get('factors_contribution', {}).items():
                st.write(f"• **{factor}:** {contribution:+.1f} points")
            return
        
        # Factor contributions
        factors = list(signal.get('factors_contribution', {}).keys())
        contributions = list(signal.get('factors_contribution', {}).values())
        
        fig = go.Figure(go.Bar(
            x=factors,
            y=contributions,
            text=[f"{c:+.1f}" for c in contributions],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="📊 Factor Contribution to Signal",
            xaxis_title="Factors",
            yaxis_title="Contribution Points",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
