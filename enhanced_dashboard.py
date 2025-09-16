"""
Enhanced Dashboard Components for Integrated Trading System
Professional-grade UI components for displaying comprehensive signal analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any

class EnhancedDashboard:
    """Enhanced dashboard components for professional signal display"""
    
    @staticmethod
    def render_enhanced_signal_card(signal_data: Dict[str, Any], symbol: str, market_data: Dict[str, Any]):
        """Render enhanced signal card with comprehensive analysis"""
        
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0)
        entry_price = signal_data.get('entry_price', 0)
        
        # Signal color mapping
        signal_colors = {
            'BUY': '#00C851',    # Green
            'SELL': '#FF4444',   # Red  
            'HOLD': '#FFA726'    # Orange
        }
        
        signal_color = signal_colors.get(signal, '#6c757d')
        
        # Create enhanced signal card
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, {signal_color}15 0%, {signal_color}05 100%);
            border-left: 5px solid {signal_color};
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                <h3 style='color: {signal_color}; margin: 0; font-size: 1.8rem; font-weight: bold;'>
                    {symbol} - {signal}
                </h3>
                <div style='text-align: right;'>
                    <div style='font-size: 1.4rem; font-weight: bold; color: {signal_color};'>
                        {confidence:.1f}% Confidence
                    </div>
                    <div style='font-size: 1.1rem; color: #666; margin-top: 0.2rem;'>
                        Price: {entry_price:.2f} PKR
                    </div>
                </div>
            </div>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;'>
                <div style='background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 8px;'>
                    <div style='font-size: 0.9rem; color: #666; margin-bottom: 0.3rem;'>Entry Point</div>
                    <div style='font-size: 1.2rem; font-weight: bold; color: #333;'>{entry_price:.2f}</div>
                </div>
                <div style='background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 8px;'>
                    <div style='font-size: 0.9rem; color: #666; margin-bottom: 0.3rem;'>Stop Loss</div>
                    <div style='font-size: 1.2rem; font-weight: bold; color: #ff4444;'>{signal_data.get("stop_loss", 0):.2f}</div>
                </div>
                <div style='background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 8px;'>
                    <div style='font-size: 0.9rem; color: #666; margin-bottom: 0.3rem;'>Take Profit</div>
                    <div style='font-size: 1.2rem; font-weight: bold; color: #00c851;'>{signal_data.get("take_profit", 0):.2f}</div>
                </div>
                <div style='background: rgba(255,255,255,0.7); padding: 0.8rem; border-radius: 8px;'>
                    <div style='font-size: 0.9rem; color: #666; margin-bottom: 0.3rem;'>Position Size</div>
                    <div style='font-size: 1.2rem; font-weight: bold; color: #333;'>{signal_data.get("position_size", 0):.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis breakdown
        with st.expander(f"📊 {symbol} Detailed Analysis", expanded=False):
            EnhancedDashboard.render_analysis_breakdown(signal_data, symbol)
    
    @staticmethod
    def render_analysis_breakdown(signal_data: Dict[str, Any], symbol: str):
        """Render detailed analysis breakdown"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Signal Components**")
            reasons = signal_data.get('reasons', [])
            for i, reason in enumerate(reasons[:5], 1):
                st.markdown(f"{i}. {reason}")
            
            # Technical scores if available
            if 'technical_score' in signal_data:
                st.markdown("**📈 Analysis Scores**")
                technical = signal_data.get('technical_score', 0)
                ml_score = signal_data.get('ml_score', 0)
                fundamental = signal_data.get('fundamental_score', 0)
                
                st.markdown(f"• Technical: {technical:+.2f}")
                st.markdown(f"• ML Model: {ml_score:+.2f}")
                st.markdown(f"• Fundamental: {fundamental:+.2f}")
        
        with col2:
            st.markdown("**🔍 Risk Assessment**")
            
            volume_support = signal_data.get('volume_support', False)
            liquidity_ok = signal_data.get('liquidity_ok', True)
            risk_score = signal_data.get('risk_score', 50)
            
            st.markdown(f"• Volume Support: {'✅' if volume_support else '❌'}")
            st.markdown(f"• Liquidity: {'✅' if liquidity_ok else '❌'}")
            st.markdown(f"• Risk Score: {risk_score:.1f}/100")
            
            # Risk/Reward calculation
            entry = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit = signal_data.get('take_profit', 0)
            
            if entry > 0 and stop_loss > 0 and take_profit > 0:
                risk = abs(entry - stop_loss) / entry * 100
                reward = abs(take_profit - entry) / entry * 100
                rr_ratio = reward / risk if risk > 0 else 0
                
                st.markdown(f"• Risk: {risk:.1f}%")
                st.markdown(f"• Reward: {reward:.1f}%")
                st.markdown(f"• R/R Ratio: {rr_ratio:.2f}")
    
    @staticmethod
    def render_portfolio_summary(signals: List[Dict[str, Any]]):
        """Render portfolio-level summary"""
        
        if not signals:
            return
        
        # Calculate portfolio metrics
        total_signals = len(signals)
        buy_signals = len([s for s in signals if s.get('signal') == 'BUY'])
        sell_signals = len([s for s in signals if s.get('signal') == 'SELL'])
        hold_signals = len([s for s in signals if s.get('signal') == 'HOLD'])
        
        avg_confidence = np.mean([s.get('confidence', 0) for s in signals])
        high_confidence = len([s for s in signals if s.get('confidence', 0) > 75])
        
        # Portfolio summary card
        st.markdown(f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        '>
            <h2 style='margin-top: 0; text-align: center; font-size: 1.8rem;'>
                📊 Portfolio Summary
            </h2>
            
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1.5rem; margin-top: 2rem;'>
                <div style='text-align: center; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
                    <div style='font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;'>{buy_signals}</div>
                    <div style='font-size: 1rem; opacity: 0.9;'>🟢 Buy Signals</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>{buy_signals/total_signals*100:.0f}% of stocks</div>
                </div>
                
                <div style='text-align: center; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
                    <div style='font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;'>{sell_signals}</div>
                    <div style='font-size: 1rem; opacity: 0.9;'>🔴 Sell Signals</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>{sell_signals/total_signals*100:.0f}% of stocks</div>
                </div>
                
                <div style='text-align: center; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
                    <div style='font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;'>{hold_signals}</div>
                    <div style='font-size: 1rem; opacity: 0.9;'>🟡 Hold Signals</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>{hold_signals/total_signals*100:.0f}% of stocks</div>
                </div>
                
                <div style='text-align: center; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
                    <div style='font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;'>{high_confidence}</div>
                    <div style='font-size: 1rem; opacity: 0.9;'>⭐ High Confidence</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>Above 75% confidence</div>
                </div>
                
                <div style='text-align: center; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
                    <div style='font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;'>{avg_confidence:.0f}%</div>
                    <div style='font-size: 1rem; opacity: 0.9;'>📊 Avg Confidence</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>Overall accuracy</div>
                </div>
                
                <div style='text-align: center; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
                    <div style='font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;'>{"Bullish" if buy_signals > sell_signals else "Bearish" if sell_signals > buy_signals else "Neutral"}</div>
                    <div style='font-size: 1rem; opacity: 0.9;'>📈 Market Sentiment</div>
                    <div style='font-size: 0.9rem; opacity: 0.7;'>{abs(buy_signals - sell_signals)} signal difference</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_signal_distribution_chart(signals: List[Dict[str, Any]]):
        """Render signal distribution visualization"""
        
        if not signals:
            return
        
        # Prepare data
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidence_data = {'BUY': [], 'SELL': [], 'HOLD': []}
        
        for signal in signals:
            sig_type = signal.get('signal', 'HOLD')
            confidence = signal.get('confidence', 0)
            signal_counts[sig_type] += 1
            confidence_data[sig_type].append(confidence)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal distribution pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(signal_counts.keys()),
                values=list(signal_counts.values()),
                hole=0.4,
                marker_colors=['#00C851', '#FF4444', '#FFA726']
            )])
            
            fig_pie.update_layout(
                title="Signal Distribution",
                showlegend=True,
                height=300,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution
            all_confidences = [s.get('confidence', 0) for s in signals]
            
            fig_hist = go.Figure(data=[go.Histogram(
                x=all_confidences,
                nbinsx=10,
                marker_color='rgba(102, 126, 234, 0.6)',
                marker_line=dict(color='rgba(102, 126, 234, 1)', width=1)
            )])
            
            fig_hist.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence Level (%)",
                yaxis_title="Number of Signals",
                height=300,
                margin=dict(t=50, b=50, l=50, r=20)
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
    
    @staticmethod
    def render_performance_metrics(signals: List[Dict[str, Any]]):
        """Render enhanced performance metrics"""
        
        if not signals:
            return
        
        st.markdown("### 📈 Enhanced Performance Metrics")
        
        # Calculate advanced metrics
        total_position_size = sum(s.get('position_size', 0) for s in signals)
        avg_risk_score = np.mean([s.get('risk_score', 50) for s in signals])
        
        # Volume support analysis
        volume_supported = len([s for s in signals if s.get('volume_support', False)])
        volume_support_pct = volume_supported / len(signals) * 100
        
        # Technical vs ML agreement
        technical_scores = [s.get('technical_score', 0) for s in signals if s.get('technical_score') is not None]
        ml_scores = [s.get('ml_score', 0) for s in signals if s.get('ml_score') is not None]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Allocation",
                value=f"{total_position_size:.1f}%",
                help="Total portfolio allocation across all signals"
            )
        
        with col2:
            st.metric(
                label="Avg Risk Score", 
                value=f"{avg_risk_score:.1f}/100",
                help="Lower is better - average risk across all positions"
            )
        
        with col3:
            st.metric(
                label="Volume Support",
                value=f"{volume_support_pct:.0f}%",
                help="Percentage of signals with volume confirmation"
            )
        
        with col4:
            diversification = len(set(s.get('signal', 'HOLD') for s in signals))
            st.metric(
                label="Signal Diversity",
                value=f"{diversification}/3",
                help="Number of different signal types (BUY/SELL/HOLD)"
            )

def main():
    """Test enhanced dashboard components"""
    st.set_page_config(page_title="Enhanced Dashboard Test", layout="wide")
    
    # Sample signal data for testing
    sample_signals = [
        {
            'symbol': 'HBL',
            'signal': 'BUY',
            'confidence': 78.5,
            'entry_price': 255.18,
            'stop_loss': 251.35,
            'take_profit': 262.84,
            'reasons': ['ML: BUY (78%)', 'Tech: Bullish indicators', 'Volume: Strong support'],
            'volume_support': True,
            'liquidity_ok': True,
            'position_size': 5.2,
            'technical_score': 1.8,
            'ml_score': 2.1,
            'fundamental_score': 0.8,
            'risk_score': 35
        },
        {
            'symbol': 'MCB',
            'signal': 'SELL', 
            'confidence': 68.2,
            'entry_price': 349.68,
            'stop_loss': 354.93,
            'take_profit': 339.19,
            'reasons': ['ML: SELL (68%)', 'Tech: Bearish trend', 'Volume: Weak'],
            'volume_support': False,
            'liquidity_ok': True,
            'position_size': 3.8,
            'technical_score': -1.5,
            'ml_score': -1.8,
            'fundamental_score': -0.2,
            'risk_score': 45
        }
    ]
    
    st.title("🚀 Enhanced Trading Dashboard")
    
    dashboard = EnhancedDashboard()
    
    # Render portfolio summary
    dashboard.render_portfolio_summary(sample_signals)
    
    # Render individual signals
    for signal in sample_signals:
        market_data = {'price': signal['entry_price']}
        dashboard.render_enhanced_signal_card(signal, signal['symbol'], market_data)
    
    # Render charts
    dashboard.render_signal_distribution_chart(sample_signals)
    dashboard.render_performance_metrics(sample_signals)

if __name__ == "__main__":
    main()