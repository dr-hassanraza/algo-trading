#!/usr/bin/env python3
"""
Quick debug script to test signal generation outside Streamlit
"""
import sys
sys.path.append('/Users/macair2020/Desktop/Algo_Trading')

from streamlit_app import PSXAlgoTradingSystemFallback, safe_generate_signal, get_cached_real_time_data

def test_signal_generation():
    """Test signal generation for a few symbols"""
    system = PSXAlgoTradingSystemFallback()
    test_symbols = ['HBL', 'ENGRO', 'LUCK', 'PSO', 'UBL']
    
    print("🔍 Testing Signal Generation...")
    print("=" * 50)
    
    results = {
        'BUY': 0,
        'SELL': 0, 
        'HOLD': 0
    }
    
    for symbol in test_symbols:
        try:
            # Get market data
            market_data = get_cached_real_time_data(symbol)
            if market_data:
                # Generate signal
                signal_data = safe_generate_signal(symbol, market_data, system, data_points=100)
                
                signal = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence', 0)
                
                print(f"{symbol:6s} | {signal:10s} | {confidence:5.1f}% | Price: {market_data['price']:.2f}")
                
                results[signal] = results.get(signal, 0) + 1
            else:
                print(f"{symbol:6s} | NO_DATA     | ----- | Price: ----")
                
        except Exception as e:
            print(f"{symbol:6s} | ERROR       | ----- | {str(e)[:30]}")
    
    print("=" * 50)
    print(f"📊 Summary:")
    for signal, count in results.items():
        print(f"   {signal}: {count}")
    
    total = sum(results.values())
    if total > 0:
        hit_rate = ((results['BUY'] + results['SELL']) / total) * 100
        print(f"   Hit Rate: {hit_rate:.1f}%")
    else:
        print("   Hit Rate: 0.0% (No valid signals)")

if __name__ == "__main__":
    test_signal_generation()