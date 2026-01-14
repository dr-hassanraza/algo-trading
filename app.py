"""
PSX Professional Trading System - Streamlit Dashboard
======================================================
Main entry point for Streamlit Cloud deployment.
Uses PSX data sources (no yfinance needed).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PSX Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PSX Data Fetcher (No yfinance needed)
class PSXDataFetcher:
    """Fetch data from Pakistan Stock Exchange"""

    def __init__(self):
        self.base_url = "https://dps.psx.com.pk"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PSX-Trading-System/1.0',
            'Accept': 'application/json'
        })

    def get_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch stock data from PSX DPS API"""
        try:
            url = f"{self.base_url}/timeseries/int/{symbol}"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data and 'data' in data and data['data']:
                df = pd.DataFrame(data['data'], columns=['timestamp', 'price', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['price'] = pd.to_numeric(df['price'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df = df.dropna()
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {str(e)[:50]}")
            return pd.DataFrame()

    def get_market_summary(self) -> dict:
        """Get PSX market summary"""
        try:
            url = f"{self.base_url}/market-summary"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    if df.empty or len(df) < 20:
        return df

    df = df.copy()

    # Moving Averages
    df['SMA_5'] = df['price'].rolling(5).mean()
    df['SMA_20'] = df['price'].rolling(20).mean()
    df['EMA_12'] = df['price'].ewm(span=12).mean()
    df['EMA_26'] = df['price'].ewm(span=26).mean()

    # RSI
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['price'].rolling(20).mean()
    bb_std = df['price'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Volume MA
    df['Volume_MA'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA']

    return df


def generate_signal(df: pd.DataFrame) -> dict:
    """Generate trading signal from indicators"""
    if df.empty or len(df) < 20:
        return {'signal': 'HOLD', 'confidence': 0, 'reasons': ['Insufficient data']}

    latest = df.iloc[-1]
    signal_points = 0
    reasons = []

    # RSI Analysis
    rsi = latest.get('RSI', 50)
    if pd.notna(rsi):
        if rsi < 30:
            signal_points += 2
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 40:
            signal_points += 1
        elif rsi > 70:
            signal_points -= 2
            reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi > 60:
            signal_points -= 1

    # Moving Average Analysis
    if pd.notna(latest.get('SMA_5')) and pd.notna(latest.get('SMA_20')):
        if latest['SMA_5'] > latest['SMA_20']:
            signal_points += 1
            reasons.append("Short MA above Long MA")
        else:
            signal_points -= 1

    # MACD Analysis
    if pd.notna(latest.get('MACD_Hist')):
        if latest['MACD_Hist'] > 0:
            signal_points += 1
        else:
            signal_points -= 1

    # Bollinger Band Analysis
    if pd.notna(latest.get('BB_Lower')) and pd.notna(latest.get('BB_Upper')):
        price = latest['price']
        if price < latest['BB_Lower']:
            signal_points += 2
            reasons.append("Price below lower Bollinger Band")
        elif price > latest['BB_Upper']:
            signal_points -= 2
            reasons.append("Price above upper Bollinger Band")

    # Volume Analysis
    if pd.notna(latest.get('Volume_Ratio')):
        if latest['Volume_Ratio'] > 1.5:
            reasons.append(f"High volume ({latest['Volume_Ratio']:.1f}x avg)")

    # Determine Signal
    if signal_points >= 3:
        signal = 'STRONG_BUY'
        confidence = min(90, 60 + signal_points * 5)
    elif signal_points >= 1:
        signal = 'BUY'
        confidence = min(75, 50 + signal_points * 5)
    elif signal_points <= -3:
        signal = 'STRONG_SELL'
        confidence = min(90, 60 + abs(signal_points) * 5)
    elif signal_points <= -1:
        signal = 'SELL'
        confidence = min(75, 50 + abs(signal_points) * 5)
    else:
        signal = 'HOLD'
        confidence = 50

    return {
        'signal': signal,
        'confidence': confidence,
        'reasons': reasons if reasons else ['Mixed signals'],
        'rsi': rsi,
        'price': latest['price']
    }


def create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create interactive price chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f'{symbol} Price', 'Volume', 'RSI']
    )

    # Price and MAs
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['price'], name='Price',
                  line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )

    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['SMA_20'], name='SMA 20',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )

    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_Upper'], name='BB Upper',
                      line=dict(color='gray', width=1, dash='dot')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['BB_Lower'], name='BB Lower',
                      line=dict(color='gray', width=1, dash='dot')),
            row=1, col=1
        )

    # Volume
    colors = ['green' if df['price'].iloc[i] >= df['price'].iloc[i-1] else 'red'
              for i in range(1, len(df))]
    colors.insert(0, 'gray')

    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['volume'], name='Volume',
              marker_color=colors, opacity=0.7),
        row=2, col=1
    )

    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI',
                      line=dict(color='purple', width=1)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


# Main App
def main():
    st.title("📈 PSX Professional Trading System")
    st.markdown("*Real-time trading signals for Pakistan Stock Exchange*")

    # Sidebar
    st.sidebar.header("Settings")

    # Stock symbols
    default_symbols = ['HBL', 'UBL', 'MCB', 'LUCK', 'FFC', 'PSO', 'OGDC', 'ENGRO', 'TRG', 'HUBC']

    selected_symbol = st.sidebar.selectbox(
        "Select Stock",
        options=default_symbols,
        index=0
    )

    custom_symbol = st.sidebar.text_input("Or enter custom symbol:")
    if custom_symbol:
        selected_symbol = custom_symbol.upper()

    # Initialize data fetcher
    fetcher = PSXDataFetcher()

    # Main content
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader(f"📊 {selected_symbol} Analysis")

    # Fetch and analyze data
    with st.spinner(f"Fetching data for {selected_symbol}..."):
        df = fetcher.get_stock_data(selected_symbol)

    if df.empty:
        st.error(f"No data available for {selected_symbol}. Please try another symbol.")
        return

    # Calculate indicators
    df = calculate_indicators(df)

    # Generate signal
    signal_data = generate_signal(df)

    # Display signal
    with col2:
        signal = signal_data['signal']
        confidence = signal_data['confidence']

        if 'BUY' in signal:
            st.success(f"**Signal: {signal}**")
        elif 'SELL' in signal:
            st.error(f"**Signal: {signal}**")
        else:
            st.info(f"**Signal: {signal}**")

        st.metric("Confidence", f"{confidence:.0f}%")

    with col3:
        latest_price = df['price'].iloc[-1]
        price_change = df['price'].pct_change().iloc[-1] * 100

        st.metric(
            "Current Price",
            f"PKR {latest_price:,.2f}",
            f"{price_change:+.2f}%"
        )

        if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]):
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

    # Price Chart
    st.plotly_chart(create_price_chart(df, selected_symbol), use_container_width=True)

    # Signal Details
    st.subheader("📋 Signal Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Signal Reasons:**")
        for reason in signal_data['reasons']:
            st.markdown(f"- {reason}")

    with col2:
        st.markdown("**Key Metrics:**")
        latest = df.iloc[-1]

        metrics = {
            'Price': f"PKR {latest['price']:,.2f}",
            'Volume': f"{latest['volume']:,.0f}",
        }

        if 'RSI' in df.columns and pd.notna(latest['RSI']):
            metrics['RSI'] = f"{latest['RSI']:.1f}"
        if 'MACD_Hist' in df.columns and pd.notna(latest['MACD_Hist']):
            metrics['MACD Histogram'] = f"{latest['MACD_Hist']:.2f}"

        for key, value in metrics.items():
            st.markdown(f"- **{key}:** {value}")

    # Quick Hitlist
    st.subheader("🎯 Quick Market Scan")

    if st.button("Scan Top Stocks"):
        with st.spinner("Scanning market..."):
            results = []

            for sym in default_symbols[:10]:
                try:
                    sym_df = fetcher.get_stock_data(sym)
                    if not sym_df.empty and len(sym_df) >= 20:
                        sym_df = calculate_indicators(sym_df)
                        sig = generate_signal(sym_df)
                        results.append({
                            'Symbol': sym,
                            'Price': f"{sym_df['price'].iloc[-1]:,.2f}",
                            'Signal': sig['signal'],
                            'Confidence': f"{sig['confidence']:.0f}%",
                            'RSI': f"{sig.get('rsi', 0):.1f}" if sig.get('rsi') else 'N/A'
                        })
                except:
                    continue

            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
            else:
                st.warning("Could not fetch market data. Please try again.")

    # Footer
    st.markdown("---")
    st.markdown(
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data source: PSX DPS API*"
    )
    st.caption("This is for educational purposes only. Always do your own research before trading.")


if __name__ == "__main__":
    main()
