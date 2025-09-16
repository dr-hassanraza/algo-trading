#!/usr/bin/env python3
"""
Minimal Streamlit test to verify system functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="Signal Test", layout="wide")

st.title("🔧 Signal Generation Test")

def generate_test_signals(num_stocks=200):
    """Generate test signals to verify system works"""
    signals = []
    
    # PSX symbols
    symbols = [f"STOCK{i:03d}" for i in range(1, num_stocks + 1)]
    
    for symbol in symbols:
        # Deterministic randomness based on symbol
        random.seed(hash(symbol) % 1000)
        force_signal = random.random()
        
        if force_signal < 0.4:  # 40% BUY
            signals.append({
                'Symbol': symbol,
                'Signal': 'BUY',
                'Confidence': f"{random.randint(45, 85)}%",
                'Price': f"{random.uniform(50, 500):.2f} PKR"
            })
        elif force_signal < 0.7:  # 30% SELL
            signals.append({
                'Symbol': symbol,
                'Signal': 'SELL', 
                'Confidence': f"{random.randint(45, 85)}%",
                'Price': f"{random.uniform(50, 500):.2f} PKR"
            })
    
    return signals

if st.button("🚀 Test Signal Generation"):
    with st.spinner("Generating test signals..."):
        signals = generate_test_signals(200)
        
        buy_signals = [s for s in signals if s['Signal'] == 'BUY']
        sell_signals = [s for s in signals if s['Signal'] == 'SELL'] 
        
        total_signals = len(buy_signals) + len(sell_signals)
        hit_rate = (total_signals / 200) * 100
        
        st.success("✅ Test Complete!")
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 BUY Opportunities", len(buy_signals))
        with col2:
            st.metric("🔴 SELL Opportunities", len(sell_signals))
        with col3:
            st.metric("📊 Hit Rate", f"{hit_rate:.1f}%")
        with col4:
            st.metric("🔍 Scanned", "200 stocks")
        
        if buy_signals:
            st.subheader("🟢 BUY Signals")
            st.dataframe(pd.DataFrame(buy_signals[:10]))  # Show first 10
        
        if sell_signals:
            st.subheader("🔴 SELL Signals")
            st.dataframe(pd.DataFrame(sell_signals[:10]))  # Show first 10

st.write("---")
st.info("This is a minimal test to verify Streamlit functionality works correctly.")