"""
PSX Trading Hitlist Generator - Enhanced ML System
Fast generation of top trading opportunities
"""

import sys
import os
sys.path.append('.')

from integrated_signal_system import IntegratedTradingSystem
import pandas as pd
import numpy as np
from datetime import datetime
import concurrent.futures
from typing import List, Dict

def generate_signal_for_symbol(symbol: str) -> Dict:
    """Generate signal for a single symbol"""
    try:
        system = IntegratedTradingSystem()
        signal = system.generate_integrated_signal(symbol)
        
        return {
            'Symbol': symbol,
            'Signal': signal.signal,
            'Confidence': signal.confidence,
            'Entry': signal.entry_price,
            'Stop': signal.stop_loss,
            'Target': signal.take_profit,
            'Position': signal.position_size,
            'Volume_Support': signal.volume_support,
            'Tech_Score': signal.technical_score,
            'ML_Score': signal.ml_score,
            'Fund_Score': signal.fundamental_score,
            'Risk': signal.risk_score,
            'Reasons': signal.reasons[:2]
        }
    except Exception as e:
        return None

from all_psx_tickers import STOCK_SYMBOLS_ONLY

def generate_psx_hitlist() -> str:
    """Generate comprehensive PSX trading hitlist and return as a string"""
    output_lines = []

    def print_to_output(*args, **kwargs):
        output_lines.append(" ".join(map(str, args)))

    print_to_output('🎯 PSX TRADING HITLIST - Enhanced ML + Technical Analysis')
    print_to_output('=' * 65)
    
    # Use the first 50 stocks from the comprehensive list for this evaluation
    psx_symbols = STOCK_SYMBOLS_ONLY[:50]
    print_to_output(f'📊 Analyzing the first {len(psx_symbols)} stocks from the full PSX list...')
    print_to_output()
    
    # Generate signals for all symbols
    signals = []
    
    for symbol in psx_symbols:
        try:
            system = IntegratedTradingSystem()
            signal = system.generate_integrated_signal(symbol)
            
            signals.append({
                'Symbol': symbol,
                'Signal': signal.signal,
                'Confidence': signal.confidence,
                'Entry': signal.entry_price,
                'Stop': signal.stop_loss,
                'Target': signal.take_profit,
                'Position': signal.position_size,
                'Volume_Support': signal.volume_support,
                'Tech_Score': signal.technical_score,
                'ML_Score': signal.ml_score,
                'Fund_Score': signal.fundamental_score,
                'Risk': signal.risk_score,
                'Reasons': signal.reasons[:2] if signal.reasons else ['Technical analysis']
            })
            
            print_to_output(f'✓ {symbol} analyzed')
            
        except Exception as e:
            print_to_output(f'✗ {symbol} failed: {str(e)[:30]}')
            continue
    
    # Sort by confidence
    signals.sort(key=lambda x: x['Confidence'], reverse=True)
    
    # Categorize signals
    buy_signals = [s for s in signals if s['Signal'] == 'BUY' and s['Confidence'] > 60]
    sell_signals = [s for s in signals if s['Signal'] == 'SELL' and s['Confidence'] > 60] 
    hold_signals = [s for s in signals if s['Signal'] == 'HOLD']
    
    print_to_output()
    print_to_output(f'📈 TOP BUY OPPORTUNITIES ({len(buy_signals)} signals):')
    print_to_output('-' * 65)
    
    for i, signal in enumerate(buy_signals[:6], 1):  # Top 6 buy signals
        reasons_str = ' | '.join(signal['Reasons']) if signal['Reasons'] else 'Analysis'
        vol_icon = '✅' if signal['Volume_Support'] else '❌'
        
        # Calculate R/R ratio
        risk = abs(signal['Entry'] - signal['Stop']) / signal['Entry'] * 100
        reward = abs(signal['Target'] - signal['Entry']) / signal['Entry'] * 100
        rr_ratio = reward / risk if risk > 0 else 0
        
        print_to_output(f'{i:2d}. {signal["Symbol"]:8} BUY   {signal["Confidence"]:5.1f}%  '
              f'Entry: {signal["Entry"]:6.1f}  SL: {signal["Stop"]:6.1f}  '
              f'TP: {signal["Target"]:6.1f}')
        print_to_output(f'    Pos: {signal["Position"]:4.1f}%  Vol:{vol_icon}  '
              f'R/R: {rr_ratio:.1f}  Tech: {signal["Tech_Score"]:+.1f}  '
              f'ML: {signal["ML_Score"]:+.1f}')
        print_to_output(f'    Reason: {reasons_str[:55]}')
        print_to_output()
    
    if sell_signals:
        print_to_output(f'📉 TOP SELL OPPORTUNITIES ({len(sell_signals)} signals):')
        print_to_output('-' * 65)
        
        for i, signal in enumerate(sell_signals[:4], 1):  # Top 4 sell signals
            reasons_str = ' | '.join(signal['Reasons']) if signal['Reasons'] else 'Analysis'
            vol_icon = '✅' if signal['Volume_Support'] else '❌'
            
            risk = abs(signal['Entry'] - signal['Stop']) / signal['Entry'] * 100
            reward = abs(signal['Entry'] - signal['Target']) / signal['Entry'] * 100
            rr_ratio = reward / risk if risk > 0 else 0
            
            print_to_output(f'{i:2d}. {signal["Symbol"]:8} SELL  {signal["Confidence"]:5.1f}%  '
                  f'Entry: {signal["Entry"]:6.1f}  SL: {signal["Stop"]:6.1f}  '
                  f'TP: {signal["Target"]:6.1f}')
            print_to_output(f'    Pos: {signal["Position"]:4.1f}%  Vol:{vol_icon}  '
              f'R/R: {rr_ratio:.1f}  Tech: {signal["Tech_Score"]:+.1f}  '
              f'ML: {signal["ML_Score"]:+.1f}')
            print_to_output(f'    Reason: {reasons_str[:55]}')
            print_to_output()
    
    print_to_output(f'⚡ MARKET OVERVIEW:')
    print_to_output('-' * 65)
    
    total_signals = len(signals)
    buy_count = len([s for s in signals if s['Signal'] == 'BUY'])
    sell_count = len([s for s in signals if s['Signal'] == 'SELL'])
    hold_count = len([s for s in signals if s['Signal'] == 'HOLD'])
    
    if signals:
        avg_confidence = np.mean([s['Confidence'] for s in signals])
        high_confidence = len([s for s in signals if s['Confidence'] > 75])
        volume_supported = len([s for s in signals if s['Volume_Support']])
        total_position = sum(s['Position'] for s in buy_signals + sell_signals)
        
        print_to_output(f'🎯 Signal Distribution: {buy_count} BUY | {sell_count} SELL | {hold_count} HOLD')
        print_to_output(f'📊 Average Confidence: {avg_confidence:.1f}% | High Confidence (>75%): {high_confidence}')
        print_to_output(f'💰 Total Suggested Allocation: {total_position:.1f}%')
        print_to_output(f'📈 Volume Support: {volume_supported}/{total_signals} ({volume_supported/total_signals*100:.1f}%)')
        
        # Market sentiment
        if buy_count > sell_count:
            sentiment = 'Bullish 🐂'
        elif sell_count > buy_count:
            sentiment = 'Bearish 🐻'
        else:
            sentiment = 'Neutral ⚖️'
        
        signal_diff = abs(buy_count - sell_count)
        print_to_output(f'📊 Market Sentiment: {sentiment} ({signal_diff} signal difference)')
        
        # Top performers by category
        if buy_signals:
            top_buy = max(buy_signals, key=lambda x: x['Confidence'])
            print_to_output(f'⭐ Top Buy: {top_buy["Symbol"]} ({top_buy["Confidence"]:.1f}% confidence)')
        
        if sell_signals:
            top_sell = max(sell_signals, key=lambda x: x['Confidence'])
            print_to_output(f'⭐ Top Sell: {top_sell["Symbol"]} ({top_sell["Confidence"]:.1f}% confidence)')
    
    print_to_output()
    print_to_output('🔥 PRIORITY WATCHLIST:')
    print_to_output('-' * 65)
    
    # Create priority list from top signals
    priority_list = []
    if buy_signals:
        priority_list.extend(buy_signals[:3])
    if sell_signals:
        priority_list.extend(sell_signals[:2])
    
    priority_list.sort(key=lambda x: x['Confidence'], reverse=True)
    
    for i, signal in enumerate(priority_list[:5], 1):
        action_color = '🟢' if signal['Signal'] == 'BUY' else '🔴' if signal['Signal'] == 'SELL' else '🟡'
        print_to_output(f'{i}. {action_color} {signal["Symbol"]:8} {signal["Signal"]:4} '
              f'{signal["Confidence"]:5.1f}% | Entry: {signal["Entry"]:6.1f} | '
              f'Pos: {signal["Position"]:4.1f}%')
    
    print_to_output()
    print_to_output('⚠️  RISK DISCLAIMER:')
    print_to_output('• This is algorithmic analysis for educational purposes only')
    print_to_output('• Always conduct your own research and risk assessment')  
    print_to_output('• Past performance does not guarantee future results')
    print_to_output('• Consider market conditions and your risk tolerance')
    print_to_output(f'• Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    return "\n".join(output_lines)

if __name__ == "__main__":
    hitlist_content = generate_psx_hitlist()
    print(hitlist_content) # Also print to console for immediate feedback
    with open('PSX_HITLIST_CURRENT.md', 'w') as f:
        f.write(hitlist_content)
    print("\n✅ Hitlist generated and saved to PSX_HITLIST_CURRENT.md")