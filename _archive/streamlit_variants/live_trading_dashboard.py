#!/usr/bin/env python3
"""
Live Intraday Trading Dashboard for PSX
=======================================

Real-time trading dashboard integrating all intraday components:
- PSX DPS tick-by-tick data
- Live signal analysis
- Risk management
- Trading alerts
- Position monitoring
- Performance tracking

Perfect for active intraday trading with PSX stocks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import datetime as dt
import time
import json
from typing import Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our intraday modules
from psx_dps_fetcher import PSXDPSFetcher
from intraday_signal_analyzer import IntradaySignalAnalyzer, SignalType
from intraday_risk_manager import IntradayRiskManager, RiskLevel

# Configure page
st.set_page_config(
    page_title="PSX Live Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LiveTradingDashboard:
    """Main live trading dashboard class"""
    
    def __init__(self):
        self.fetcher = PSXDPSFetcher()
        self.analyzer = IntradaySignalAnalyzer()
        
        # Initialize session state
        if 'risk_manager' not in st.session_state:
            st.session_state.risk_manager = IntradayRiskManager(
                initial_capital=1000000,  # 10 lakh PKR default
                risk_level=RiskLevel.MODERATE
            )
        
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['UBL', 'MCB', 'HBL', 'LUCK', 'FFC', 'PPL']
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
        
        self.risk_manager = st.session_state.risk_manager
    
    def render_header(self):
        """Render dashboard header with key metrics"""
        
        st.title("⚡ PSX Live Intraday Trading Dashboard")
        
        # Market status
        market_status = self.fetcher.get_market_status()
        status_color = "🟢" if market_status['is_trading'] else "🔴"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Market Status", f"{status_color} {market_status['status'].title()}")
        
        with col2:
            metrics = self.risk_manager.get_risk_metrics()
            st.metric("Available Capital", f"{metrics.available_capital:,.0f} PKR")
        
        with col3:
            st.metric("Active Positions", metrics.positions_count)
        
        with col4:
            daily_pnl_color = "normal" if metrics.daily_pnl >= 0 else "inverse"
            st.metric("Daily P&L", f"{metrics.daily_pnl:,.0f} PKR", 
                     delta_color=daily_pnl_color)
        
        with col5:
            st.metric("Win Rate", f"{metrics.win_rate:.1%}")
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls and settings"""
        
        st.sidebar.header("⚙️ Dashboard Controls")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "Auto Refresh (30s)", 
            value=st.session_state.auto_refresh
        )
        
        if st.sidebar.button("🔄 Manual Refresh"):
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Risk Management Settings
        st.sidebar.header("⚖️ Risk Settings")
        
        new_capital = st.sidebar.number_input(
            "Capital (PKR)", 
            min_value=100000, 
            max_value=10000000, 
            value=int(self.risk_manager.initial_capital),
            step=100000
        )
        
        new_risk_level = st.sidebar.selectbox(
            "Risk Level",
            options=[level.value for level in RiskLevel],
            index=list(RiskLevel).index(self.risk_manager.risk_level)
        )
        
        if st.sidebar.button("Update Risk Settings"):
            st.session_state.risk_manager = IntradayRiskManager(
                initial_capital=new_capital,
                risk_level=RiskLevel(new_risk_level)
            )
            st.success("Risk settings updated!")
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Watchlist Management
        st.sidebar.header("👀 Watchlist")
        
        # Add new symbol
        new_symbol = st.sidebar.text_input("Add Symbol", placeholder="e.g., ENGRO")
        if st.sidebar.button("Add to Watchlist") and new_symbol:
            if new_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol.upper())
                st.success(f"Added {new_symbol.upper()} to watchlist")
                st.rerun()
        
        # Current watchlist
        for symbol in st.session_state.watchlist:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(symbol)
            if col2.button("❌", key=f"remove_{symbol}"):
                st.session_state.watchlist.remove(symbol)
                st.rerun()
    
    def render_live_signals(self):
        """Render live trading signals"""
        
        st.subheader("📈 Live Trading Signals")
        
        # Get signals for watchlist
        with st.spinner("Analyzing symbols..."):
            signals = self.analyzer.analyze_multiple_symbols(
                st.session_state.watchlist, 
                analysis_period_minutes=20
            )
        
        if not signals:
            st.warning("No signals available")
            return
        
        # Create signals table
        signal_data = []
        for signal in signals:
            signal_data.append({
                'Symbol': signal.symbol,
                'Signal': signal.signal_type.value,
                'Confidence': f"{signal.confidence:.1f}%",
                'Entry Price': f"{signal.entry_price:.2f}",
                'Target': f"{signal.target_price:.2f}",
                'Stop Loss': f"{signal.stop_loss:.2f}",
                'R/R': f"{signal.risk_reward_ratio:.2f}",
                'Momentum': signal.momentum_direction,
                'Volume Support': "✅" if signal.volume_support else "❌",
                'Hold Period': f"{signal.holding_period_minutes}m"
            })
        
        df = pd.DataFrame(signal_data)
        
        # Color code signals
        def color_signal(val):
            if val == 'STRONG_BUY':
                return 'background-color: #00ff00; color: black'
            elif val == 'BUY':
                return 'background-color: #90EE90; color: black'
            elif val == 'STRONG_SELL':
                return 'background-color: #ff0000; color: white'
            elif val == 'SELL':
                return 'background-color: #FFA07A; color: black'
            else:
                return 'background-color: #f0f0f0; color: black'
        
        # Apply styling
        styled_df = df.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Show actionable signals
        actionable_signals = [s for s in signals if s.signal_type != SignalType.HOLD and s.confidence >= 65]
        
        if actionable_signals:
            st.subheader("🚨 High-Confidence Alerts")
            
            for signal in actionable_signals[:3]:  # Top 3
                with st.expander(f"{signal.symbol} - {signal.signal_type.value} ({signal.confidence:.1f}%)"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Entry Details:**")
                        st.write(f"Price: {signal.entry_price:.2f} PKR")
                        st.write(f"Target: {signal.target_price:.2f} PKR")
                        st.write(f"Stop: {signal.stop_loss:.2f} PKR")
                    
                    with col2:
                        st.write("**Risk Management:**")
                        sizing = self.risk_manager.calculate_position_size(signal)
                        st.write(f"Quantity: {sizing['recommended_quantity']:,}")
                        st.write(f"Risk: {sizing['risk_amount']:,.0f} PKR")
                        st.write(f"Position: {sizing.get('position_value', 0):,.0f} PKR")
                    
                    with col3:
                        st.write("**Signal Quality:**")
                        st.write(f"Confidence: {signal.confidence:.1f}%")
                        st.write(f"R/R Ratio: {signal.risk_reward_ratio:.2f}")
                        st.write(f"Reasoning: {'; '.join(signal.reasoning[:2])}")
                    
                    # Trade action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"Execute Trade - {signal.symbol}", key=f"trade_{signal.symbol}"):
                            self.execute_trade(signal, sizing['recommended_quantity'])
                    
                    with col2:
                        if st.button(f"Add Alert - {signal.symbol}", key=f"alert_{signal.symbol}"):
                            self.add_alert(signal)
                    
                    with col3:
                        if st.button(f"View Chart - {signal.symbol}", key=f"chart_{signal.symbol}"):
                            self.show_intraday_chart(signal.symbol)
    
    def render_positions(self):
        """Render active positions"""
        
        st.subheader("💼 Active Positions")
        
        # Update positions
        update_result = self.risk_manager.update_positions()
        
        # Show update notifications
        if update_result['stops_triggered']:
            for stop in update_result['stops_triggered']:
                st.error(f"🛑 Stop Loss Triggered: {stop['symbol']} - P&L: {stop['realized_pnl']:,.0f} PKR")
        
        if update_result['targets_hit']:
            for target in update_result['targets_hit']:
                st.success(f"🎯 Target Hit: {target['symbol']} - P&L: {target['realized_pnl']:,.0f} PKR")
        
        if update_result['trailing_stops_adjusted']:
            for trail in update_result['trailing_stops_adjusted']:
                st.info(f"📈 Trailing Stop Updated: {trail['symbol']} - New Stop: {trail['new_trailing_stop']:.2f} PKR")
        
        # Position summary
        summary = self.risk_manager.get_position_summary()
        
        if summary.get('active_positions', 0) == 0:
            st.info("No active positions")
            return
        
        # Positions table
        position_data = []
        for pos in summary['positions']:
            position_data.append({
                'Symbol': pos['symbol'],
                'Type': pos['position_type'],
                'Quantity': f"{pos['quantity']:,}",
                'Entry': f"{pos['entry_price']:.2f}",
                'Current': f"{pos['current_price']:.2f}",
                'P&L': f"{pos['unrealized_pnl']:,.0f}",
                'P&L %': f"{pos['pnl_pct']:+.2f}%",
                'Stop Loss': f"{pos['stop_loss']:.2f}",
                'Target': f"{pos['take_profit']:.2f}",
                'Holding': f"{pos['holding_minutes']:.0f}m"
            })
        
        positions_df = pd.DataFrame(position_data)
        
        # Color code P&L
        def color_pnl(val):
            if '+' in str(val):
                return 'color: green'
            elif '-' in str(val):
                return 'color: red'
            else:
                return 'color: black'
        
        styled_positions = positions_df.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
        st.dataframe(styled_positions, use_container_width=True)
        
        # Position actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Close All Positions"):
                if st.confirm("Close all positions?"):
                    self.close_all_positions()
        
        with col2:
            if st.button("Update Stops"):
                st.info("Trailing stops updated automatically")
        
        with col3:
            st.metric("Total Unrealized P&L", f"{summary['total_unrealized_pnl']:,.0f} PKR")
    
    def render_intraday_chart(self, symbol: str):
        """Render intraday tick chart"""
        
        st.subheader(f"📊 Intraday Chart - {symbol}")
        
        # Get tick data
        ticks_df = self.fetcher.fetch_intraday_ticks(symbol, limit=100)
        
        if ticks_df.empty:
            st.warning(f"No tick data available for {symbol}")
            return
        
        # Create plotly chart
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=ticks_df['datetime_pkt'],
            y=ticks_df['price'],
            mode='lines+markers',
            name='Price',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=ticks_df['datetime_pkt'],
            y=ticks_df['volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3,
            marker_color='orange'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - Live Intraday Data",
            xaxis_title="Time (PKT)",
            yaxis_title="Price (PKR)",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"{ticks_df.iloc[0]['price']:.2f} PKR")
        
        with col2:
            price_change = ticks_df.iloc[0]['price'] - ticks_df.iloc[-1]['price']
            st.metric("Price Change", f"{price_change:+.2f} PKR")
        
        with col3:
            st.metric("Total Volume", f"{ticks_df['volume'].sum():,}")
        
        with col4:
            st.metric("Trades Count", len(ticks_df))
    
    def execute_trade(self, signal, quantity):
        """Execute a trade"""
        
        try:
            # Validate trade
            validation = self.risk_manager.validate_trade_entry(signal, quantity)
            
            if not validation['is_valid']:
                st.error(f"Trade blocked: {'; '.join(validation['blocking_issues'])}")
                return
            
            # Show warnings if any
            if validation['warnings']:
                st.warning(f"Warnings: {'; '.join(validation['warnings'])}")
            
            # Execute trade
            result = self.risk_manager.enter_position(signal, quantity)
            
            if result['success']:
                st.success(f"✅ Trade executed: {signal.symbol} - {quantity:,} shares")
                
                # Add to alert history
                alert = {
                    'timestamp': dt.datetime.now(),
                    'type': 'TRADE_EXECUTED',
                    'symbol': signal.symbol,
                    'action': signal.signal_type.value,
                    'quantity': quantity,
                    'price': signal.entry_price,
                    'message': f"Executed {signal.signal_type.value} for {signal.symbol}"
                }
                st.session_state.alert_history.append(alert)
                
                # Update display
                st.rerun()
            else:
                st.error(f"Trade failed: {result['reason']}")
                
        except Exception as e:
            st.error(f"Trade execution error: {e}")
    
    def add_alert(self, signal):
        """Add trading alert"""
        
        alert = {
            'timestamp': dt.datetime.now(),
            'type': 'SIGNAL_ALERT',
            'symbol': signal.symbol,
            'action': signal.signal_type.value,
            'confidence': signal.confidence,
            'price': signal.entry_price,
            'message': f"{signal.signal_type.value} alert for {signal.symbol} - {signal.confidence:.1f}% confidence"
        }
        
        st.session_state.alert_history.append(alert)
        st.success(f"Alert added for {signal.symbol}")
    
    def close_all_positions(self):
        """Close all active positions"""
        
        positions_to_close = list(self.risk_manager.positions.keys())
        
        for symbol in positions_to_close:
            try:
                position = self.risk_manager.positions[symbol]
                current_data = self.fetcher.fetch_real_time_data(symbol)
                
                if current_data:
                    exit_result = self.risk_manager._exit_position(
                        position, 
                        current_data['price'], 
                        'MANUAL_CLOSE'
                    )
                    st.info(f"Closed {symbol}: P&L {exit_result['realized_pnl']:,.0f} PKR")
                
            except Exception as e:
                st.error(f"Error closing {symbol}: {e}")
        
        st.success("All positions closed")
        st.rerun()
    
    def show_intraday_chart(self, symbol):
        """Show detailed intraday chart in expander"""
        
        with st.expander(f"📊 Detailed Chart - {symbol}", expanded=True):
            self.render_intraday_chart(symbol)
    
    def render_alerts_history(self):
        """Render alerts and notifications history"""
        
        st.subheader("🔔 Alerts & Notifications")
        
        if not st.session_state.alert_history:
            st.info("No alerts yet")
            return
        
        # Show recent alerts (last 10)
        recent_alerts = st.session_state.alert_history[-10:]
        recent_alerts.reverse()  # Most recent first
        
        for alert in recent_alerts:
            timestamp = alert['timestamp'].strftime("%H:%M:%S")
            
            if alert['type'] == 'TRADE_EXECUTED':
                st.success(f"**{timestamp}** - {alert['message']}")
            elif alert['type'] == 'SIGNAL_ALERT':
                st.info(f"**{timestamp}** - {alert['message']}")
            else:
                st.write(f"**{timestamp}** - {alert['message']}")
        
        # Clear alerts button
        if st.button("Clear Alert History"):
            st.session_state.alert_history = []
            st.rerun()
    
    def render_performance_summary(self):
        """Render performance metrics"""
        
        st.subheader("📊 Performance Summary")
        
        metrics = self.risk_manager.get_risk_metrics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Capital")
            st.metric("Initial Capital", f"{metrics.total_capital:,.0f} PKR")
            st.metric("Available Capital", f"{metrics.available_capital:,.0f} PKR")
            st.metric("Used Capital", f"{metrics.used_capital:,.0f} PKR")
        
        with col2:
            st.subheader("Risk Metrics")
            st.metric("Total Exposure", f"{metrics.total_exposure:,.0f} PKR")
            st.metric("Current Drawdown", f"{metrics.current_drawdown:.1%}")
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.1%}")
        
        with col3:
            st.subheader("Trading Stats")
            st.metric("Win Rate", f"{metrics.win_rate:.1%}")
            st.metric("Avg Win", f"{metrics.avg_win:,.0f} PKR")
            st.metric("Avg Loss", f"{metrics.avg_loss:,.0f} PKR")
    
    def run(self):
        """Main dashboard runner"""
        
        # Render components
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔥 Live Signals", 
            "💼 Positions", 
            "📊 Charts", 
            "🔔 Alerts",
            "📈 Performance"
        ])
        
        with tab1:
            self.render_live_signals()
        
        with tab2:
            self.render_positions()
        
        with tab3:
            # Chart selector
            chart_symbol = st.selectbox("Select Symbol for Chart", st.session_state.watchlist)
            if chart_symbol:
                self.render_intraday_chart(chart_symbol)
        
        with tab4:
            self.render_alerts_history()
        
        with tab5:
            self.render_performance_summary()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(30)
            st.rerun()

# Main entry point
def main():
    """Main function"""
    
    dashboard = LiveTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()