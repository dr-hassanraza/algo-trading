"""
Quick test of the comprehensive analysis function
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime

def test_confluence_analysis():
    """Test the confluence analysis function"""
    
    # Create sample data similar to what streamlit app will use
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    base_price = 255.0
    
    # Generate realistic price movement
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.randint(50000, 200000, 100)
    })
    
    print("🧪 Testing Comprehensive Analysis Function...")
    print(f"📊 Data: {len(df)} rows, Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    
    # Test the function (copy from streamlit app)
    def generate_confluence_analysis(df, current_price, symbol):
        """Generate simplified confluence analysis"""
        
        # Calculate indicators
        indicators = {}
        indicators['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        indicators['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        indicators['sma_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        indicators['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        indicators['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        indicators['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        indicators['macd'] = ta.trend.macd(df['Close'])
        indicators['macd_signal'] = ta.trend.macd_signal(df['Close'])
        indicators['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        indicators['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        indicators['bb_middle'] = ta.volatility.bollinger_mavg(df['Close'])
        indicators['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        
        # Analyze signals
        signals = {'bullish': [], 'bearish': [], 'neutral': []}
        
        # Golden Cross Analysis
        sma_50 = indicators['sma_50'].iloc[-1] if not indicators['sma_50'].empty else 0
        sma_200 = indicators['sma_200'].iloc[-1] if not indicators['sma_200'].empty else 0
        if sma_50 > sma_200 and sma_50 > 0 and sma_200 > 0:
            signals['bullish'].append({
                'name': 'Golden Cross (SMA50/200)',
                'description': f'Bullish: SMA50({sma_50:.2f}) > SMA200({sma_200:.2f})'
            })
        elif sma_50 < sma_200 and sma_50 > 0 and sma_200 > 0:
            signals['bearish'].append({
                'name': 'Death Cross (SMA50/200)',
                'description': f'Bearish: SMA50({sma_50:.2f}) < SMA200({sma_200:.2f})'
            })
        
        # RSI Analysis
        rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 50
        if rsi > 50:
            signals['bullish'].append({
                'name': 'RSI Analysis',
                'description': f'RSI Bullish: {rsi:.2f}'
            })
        elif rsi < 50:
            signals['bearish'].append({
                'name': 'RSI Analysis',
                'description': f'RSI Bearish: {rsi:.2f}'
            })
        
        # Calculate percentages
        total_signals = len(signals['bullish']) + len(signals['bearish']) + len(signals['neutral'])
        bullish_pct = len(signals['bullish']) / total_signals * 100 if total_signals > 0 else 0
        bearish_pct = len(signals['bearish']) / total_signals * 100 if total_signals > 0 else 0
        
        # Determine overall signal
        if bullish_pct > bearish_pct and bullish_pct > 40:
            overall_signal = 'BUY'
        elif bearish_pct > bullish_pct and bearish_pct > 40:
            overall_signal = 'SELL'
        else:
            overall_signal = 'HOLD'
        
        return {
            'confluence': {
                'overall_signal': overall_signal,
                'bullish_signals': signals['bullish'],
                'bearish_signals': signals['bearish'],
                'neutral_signals': signals['neutral'],
                'bullish_pct': bullish_pct,
                'bearish_pct': bearish_pct,
                'total_signals': total_signals
            },
            'indicators': indicators
        }
    
    # Test the analysis
    current_price = df['Close'].iloc[-1]
    result = generate_confluence_analysis(df, current_price, 'TEST')
    
    confluence = result['confluence']
    
    print(f"✅ Analysis completed successfully!")
    print(f"📊 Overall Signal: {confluence['overall_signal']}")
    print(f"🟢 Bullish: {len(confluence['bullish_signals'])}/{confluence['total_signals']} ({confluence['bullish_pct']:.0f}%)")
    print(f"🔴 Bearish: {len(confluence['bearish_signals'])}/{confluence['total_signals']} ({confluence['bearish_pct']:.0f}%)")
    print(f"⚪ Neutral: {len(confluence['neutral_signals'])}/{confluence['total_signals']}")
    
    # Test indicators
    indicators = result['indicators']
    rsi = indicators['rsi'].iloc[-1] if not indicators['rsi'].empty else 0
    sma_20 = indicators['sma_20'].iloc[-1] if not indicators['sma_20'].empty else 0
    
    print(f"📈 RSI: {rsi:.2f}")
    print(f"📈 SMA20: {sma_20:.2f}")
    print(f"📈 Current Price: {current_price:.2f}")
    
    print("\n🎯 Sample Signals:")
    for signal in confluence['bullish_signals']:
        print(f"  ✅ {signal['name']}: {signal['description']}")
    
    for signal in confluence['bearish_signals']:
        print(f"  ❌ {signal['name']}: {signal['description']}")
    
    print("\n✅ Comprehensive analysis test completed successfully!")
    return True

if __name__ == "__main__":
    test_confluence_analysis()