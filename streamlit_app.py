"""
PSX Terminal Trading System - Streamlit Cloud Compatible Version
Focused on PSX Terminal API integration with minimal dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PSX Terminal - Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
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
    
    .live-data-card {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
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
    }
</style>
""", unsafe_allow_html=True)

class SimplePSXAPI:
    """Simplified PSX Terminal API client for Streamlit Cloud"""
    
    def __init__(self):
        self.base_url = "https://psxterminal.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PSX-Trading-System/1.0',
            'Accept': 'application/json'
        })

    def test_connectivity(self):
        """Test API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/api/status", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None

    def get_symbols(self):
        """Get all symbols"""
        try:
            response = self.session.get(f"{self.base_url}/api/symbols", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('data', []) if data.get('success') else []
        except Exception as e:
            st.error(f"Symbols Error: {str(e)}")
            return []

    def get_market_data(self, symbol):
        """Get market data for symbol"""
        try:
            url = f"{self.base_url}/api/ticks/REG/{symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('data') if data.get('success') else None
        except Exception as e:
            st.error(f"Market Data Error for {symbol}: {str(e)}")
            return None

    def get_klines(self, symbol, timeframe='1h', limit=50):
        """Get k-line data"""
        try:
            url = f"{self.base_url}/api/klines/{symbol}/{timeframe}"
            response = self.session.get(url, params={'limit': limit}, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get('data', []) if data.get('success') else []
        except Exception as e:
            st.error(f"K-line Error for {symbol}: {str(e)}")
            return []

    def get_market_stats(self):
        """Get market statistics"""
        try:
            response = self.session.get(f"{self.base_url}/api/stats/REG", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('data') if data.get('success') else None
        except Exception as e:
            st.error(f"Market Stats Error: {str(e)}")
            return None

@st.cache_data(ttl=60)
def get_cached_symbols():
    """Cache symbols for 1 minute"""
    api = SimplePSXAPI()
    return api.get_symbols()

@st.cache_data(ttl=30)
def get_cached_market_data(symbol):
    """Cache market data for 30 seconds"""
    api = SimplePSXAPI()
    return api.get_market_data(symbol)

def render_header():
    """Render header"""
    st.markdown('<h1 class="main-header">PSX Terminal Trading System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="success-metric">
            <h4>📊 PSX Terminal API</h4>
            <p>Real-time Market Data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-metric">
            <h4>🎯 Live Updates</h4>
            <p>500+ PSX Securities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-metric">
            <h4>📈 Interactive Charts</h4>
            <p>Multiple Timeframes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="success-metric">
            <h4>🏢 Company Data</h4>
            <p>Financial Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

def render_system_status():
    """Render system status"""
    st.markdown("## 🔧 System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            api = SimplePSXAPI()
            status = api.test_connectivity()
            
            if status:
                st.success(f"✅ PSX Terminal API Connected")
                st.info(f"Uptime: {status.get('uptime', 0)} seconds")
                st.info(f"Status: {status.get('status', 'Unknown')}")
            else:
                st.error("❌ PSX Terminal API Connection Failed")
        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)}")
    
    with col2:
        symbols = get_cached_symbols()
        if symbols:
            st.success(f"✅ Symbols Loaded: {len(symbols)}")
            st.info(f"Sample symbols: {', '.join(symbols[:5])}")
        else:
            st.error("❌ Unable to load symbols")

def render_market_overview():
    """Render market overview"""
    st.markdown("## 📊 Market Overview")
    
    try:
        api = SimplePSXAPI()
        stats = api.get_market_stats()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_vol = stats.get('totalVolume', 0)
                st.metric("Total Volume", f"{total_vol:,.0f}" if total_vol else "N/A")
            
            with col2:
                total_val = stats.get('totalValue', 0)
                if total_val:
                    st.metric("Total Value", f"{total_val/1e9:.1f}B PKR")
                else:
                    st.metric("Total Value", "N/A")
            
            with col3:
                gainers = stats.get('gainers', 0)
                losers = stats.get('losers', 0)
                st.metric("Gainers", f"{gainers}")
                st.metric("Losers", f"{losers}")
            
            with col4:
                trades = stats.get('totalTrades', 0)
                st.metric("Total Trades", f"{trades:,}" if trades else "N/A")
        else:
            st.info("📊 Market statistics loading...")
    except Exception as e:
        st.error(f"Market overview error: {str(e)}")

def render_symbol_analysis():
    """Render symbol analysis"""
    st.markdown("## 🔍 Symbol Analysis")
    
    symbols = get_cached_symbols()
    if not symbols:
        st.error("Unable to load symbols")
        return
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Limit symbols for better performance
        display_symbols = symbols[:100] if len(symbols) > 100 else symbols
        selected_symbol = st.selectbox(
            "Select Symbol",
            options=display_symbols,
            index=0,
            help=f"Choose from {len(display_symbols)} symbols"
        )
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            options=['1h', '4h', '1d'],
            index=0
        )
    
    with col3:
        if st.button("🔄 Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    if selected_symbol:
        tab1, tab2 = st.tabs(["📊 Market Data", "📈 Chart"])
        
        with tab1:
            # Market data
            market_data = get_cached_market_data(selected_symbol)
            
            if market_data:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    price = market_data.get('price', 0)
                    change = market_data.get('change', 0)
                    st.metric("Price", f"{price:.2f} PKR", f"{change:+.2f}")
                
                with col2:
                    change_pct = market_data.get('changePercent', 0)
                    st.metric("Change %", f"{change_pct:+.2%}")
                
                with col3:
                    volume = market_data.get('volume', 0)
                    st.metric("Volume", f"{volume:,}")
                
                with col4:
                    trades = market_data.get('trades', 0)
                    st.metric("Trades", f"{trades:,}")
                
                # Additional metrics
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    high = market_data.get('high', 0)
                    st.metric("High", f"{high:.2f}")
                
                with col6:
                    low = market_data.get('low', 0)
                    st.metric("Low", f"{low:.2f}")
                
                with col7:
                    value = market_data.get('value', 0)
                    st.metric("Value", f"{value/1e6:.1f}M PKR")
                
                with col8:
                    status = market_data.get('st', 'Unknown')
                    st.metric("Status", status)
            else:
                st.error(f"Unable to load data for {selected_symbol}")
        
        with tab2:
            # Chart
            api = SimplePSXAPI()
            klines = api.get_klines(selected_symbol, timeframe, 50)
            
            if klines:
                # Create DataFrame
                df = pd.DataFrame(klines)
                if not df.empty and 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('datetime', inplace=True)
                    
                    # Candlestick chart
                    fig = go.Figure(data=go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=selected_symbol
                    ))
                    
                    fig.update_layout(
                        title=f"{selected_symbol} - {timeframe} Chart",
                        xaxis_title="Time",
                        yaxis_title="Price (PKR)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    if 'volume' in df.columns:
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
                    st.warning("Chart data format issue")
            else:
                st.warning(f"Unable to load chart data for {selected_symbol}")

def render_documentation():
    """Render documentation"""
    st.markdown("## 📚 Documentation")
    
    st.markdown("""
    ### 🎯 PSX Terminal API Integration
    
    This system provides real-time access to Pakistan Stock Exchange data through the PSX Terminal API.
    
    **Features:**
    - Real-time market data for 500+ PSX securities
    - Interactive candlestick charts
    - Market overview and statistics
    - Professional data visualization
    - Streamlit Cloud optimized performance
    
    **Data Sources:**
    - **Primary**: PSX Terminal API (https://psxterminal.com)
    - **Coverage**: All PSX listed securities
    - **Update Frequency**: Real-time during market hours
    - **Timeframes**: 1h, 4h, 1d charts available
    
    **How to Use:**
    1. Check system status for API connectivity
    2. View market overview for broad market insights
    3. Analyze individual symbols with charts and data
    4. Use refresh button for latest data
    """)

def main():
    """Main application"""
    
    # Render header
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Section",
        options=[
            "🏠 System Status",
            "📊 Market Overview", 
            "🔍 Symbol Analysis",
            "📚 Documentation"
        ]
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()
    
    # Page routing
    if page == "🏠 System Status":
        render_system_status()
        
    elif page == "📊 Market Overview":
        render_market_overview()
        
    elif page == "🔍 Symbol Analysis":
        render_symbol_analysis()
        
    elif page == "📚 Documentation":
        render_documentation()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>PSX Terminal Trading System | Real-time Pakistan Stock Exchange Data</p>
        <p>📊 Live Data • 🎯 Professional Analysis • 🚀 Streamlit Cloud Optimized</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()