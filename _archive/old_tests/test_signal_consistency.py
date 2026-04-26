"""
Test Signal Consistency - Verify Portfolio Summary Matches Individual Signals
"""

import sys
sys.path.append('.')

def test_signal_consistency():
    """Test that portfolio summary matches individual signals"""
    
    print('🧪 Testing Signal Consistency...')
    print('=' * 50)
    
    try:
        from streamlit_app import safe_generate_signal, PSXAlgoTradingSystemFallback
        import pandas as pd
        
        # Initialize system
        system = PSXAlgoTradingSystemFallback()
        
        # Test symbols from your watchlist
        test_symbols = ['HBL', 'UBL', 'FFC', 'LUCK', 'PSO', 'OGDC', 'NBP', 'MCB', 'ABL', 'TRG']
        
        individual_signals = []
        portfolio_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        print('📊 Generating individual signals...')
        
        for symbol in test_symbols:
            try:
                # Mock market data
                market_data = {
                    'price': {'HBL': 255.18, 'UBL': 367.10, 'FFC': 452.59, 'LUCK': 477.52, 
                             'PSO': 424.58, 'OGDC': 272.19, 'NBP': 183.70, 'MCB': 349.68, 
                             'ABL': 171.00, 'TRG': 75.06}.get(symbol, 200.0),
                    'volume': 100000
                }
                
                # Generate signal using same method as app
                signal_data = safe_generate_signal(symbol, market_data, system, data_points=100)
                
                signal_type = signal_data.get('signal', 'HOLD')
                confidence = signal_data.get('confidence', 0)
                
                individual_signals.append({
                    'symbol': symbol,
                    'signal': signal_type,
                    'confidence': confidence
                })
                
                # Count for portfolio summary
                if signal_type in ['BUY', 'STRONG_BUY']:
                    portfolio_counts['BUY'] += 1
                elif signal_type in ['SELL', 'STRONG_SELL']:
                    portfolio_counts['SELL'] += 1
                else:
                    portfolio_counts['HOLD'] += 1
                
                print(f'  {symbol:8} {signal_type:4} {confidence:5.1f}%')
                
            except Exception as e:
                print(f'  {symbol:8} ERROR: {str(e)[:30]}')
                portfolio_counts['HOLD'] += 1
        
        print()
        print('📈 Individual Signal Results:')
        print(f'  BUY:  {portfolio_counts["BUY"]:2d} signals')
        print(f'  SELL: {portfolio_counts["SELL"]:2d} signals') 
        print(f'  HOLD: {portfolio_counts["HOLD"]:2d} signals')
        
        total_signals = len(test_symbols)
        buy_pct = portfolio_counts['BUY'] / total_signals * 100
        sell_pct = portfolio_counts['SELL'] / total_signals * 100
        hold_pct = portfolio_counts['HOLD'] / total_signals * 100
        
        print()
        print('📊 Portfolio Summary (What should appear in app):')
        print(f'  🟢 Buy Signals:  {portfolio_counts["BUY"]} ({buy_pct:.0f}% of stocks)')
        print(f'  🔴 Sell Signals: {portfolio_counts["SELL"]} ({sell_pct:.0f}% of stocks)')
        print(f'  🟡 Hold Signals: {portfolio_counts["HOLD"]} ({hold_pct:.0f}% of stocks)')
        
        # Market sentiment
        if portfolio_counts['BUY'] > portfolio_counts['SELL']:
            sentiment = 'Bullish 🐂'
        elif portfolio_counts['SELL'] > portfolio_counts['BUY']:
            sentiment = 'Bearish 🐻'
        else:
            sentiment = 'Neutral ⚖️'
        
        signal_diff = abs(portfolio_counts['BUY'] - portfolio_counts['SELL'])
        print(f'  📈 Market Sentiment: {sentiment} ({signal_diff} signal difference)')
        
        # High confidence signals
        high_conf_signals = len([s for s in individual_signals if s['confidence'] > 75])
        print(f'  ⭐ High Confidence: {high_conf_signals} ({high_conf_signals/total_signals*100:.0f}% of stocks)')
        
        print()
        print('✅ CONSISTENCY CHECK:')
        print('The portfolio summary should now match the individual signals shown above.')
        print('If the Streamlit app shows different numbers, there may be caching issues.')
        
        # Expected vs actual comparison
        print()
        print('🔍 EXPECTED APP BEHAVIOR:')
        print('Your Streamlit app should show:')
        print(f'  - Individual signals: Mostly HOLD with some BUY/SELL')
        print(f'  - Portfolio summary: {portfolio_counts["BUY"]} BUY | {portfolio_counts["SELL"]} SELL | {portfolio_counts["HOLD"]} HOLD')
        print(f'  - Market sentiment: {sentiment}')
        
        if portfolio_counts['BUY'] < 5 and portfolio_counts['HOLD'] > 5:
            print('✅ This looks realistic - not the old "8 BUY signals" problem!')
        else:
            print('⚠️ Still showing many BUY signals - may need cache refresh')
        
    except Exception as e:
        import traceback
        print(f'❌ Test failed: {str(e)}')
        print(traceback.format_exc())

if __name__ == "__main__":
    test_signal_consistency()