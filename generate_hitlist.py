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

def generate_psx_hitlist():
    """Generate comprehensive PSX trading hitlist"""
    
    print('🎯 PSX TRADING HITLIST - Enhanced ML + Technical Analysis')
    print('=' * 65)
    
    # Top liquid PSX stocks
    psx_symbols = [
        # Banking (High Priority)
        'HBL', 'UBL', 'MCB', 'ABL', 'NBP', 
        # Industrial Leaders
        'LUCK', 'FFC', 'ENGRO', 'PSO', 'OGDC',
        # Technology & Growth
        'TRG', 'SYSTEMS', 
        # FMCG Staples
        'NESTLE', 'COLG',
        # Other Major Stocks
        'PPL', 'BAFL', 'DGKC', 'MLCF', 'CHCC'
    ]
    
    print(f'📊 Analyzing {len(psx_symbols)} major PSX stocks...')
    print()
    
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
            
            print(f'✓ {symbol} analyzed')
            
        except Exception as e:
            print(f'✗ {symbol} failed: {str(e)[:30]}')
            continue
    
    # Sort by confidence
    signals.sort(key=lambda x: x['Confidence'], reverse=True)
    
    # Categorize signals
    buy_signals = [s for s in signals if s['Signal'] == 'BUY' and s['Confidence'] > 60]
    sell_signals = [s for s in signals if s['Signal'] == 'SELL' and s['Confidence'] > 60] 
    hold_signals = [s for s in signals if s['Signal'] == 'HOLD']
    
    print()
    print(f'📈 TOP BUY OPPORTUNITIES ({len(buy_signals)} signals):')
    print('-' * 65)
    
    for i, signal in enumerate(buy_signals[:6], 1):  # Top 6 buy signals
        reasons_str = ' | '.join(signal['Reasons']) if signal['Reasons'] else 'Analysis'
        vol_icon = '✅' if signal['Volume_Support'] else '❌'
        
        # Calculate R/R ratio
        risk = abs(signal['Entry'] - signal['Stop']) / signal['Entry'] * 100
        reward = abs(signal['Target'] - signal['Entry']) / signal['Entry'] * 100
        rr_ratio = reward / risk if risk > 0 else 0
        
        print(f'{i:2d}. {signal["Symbol"]:8} BUY   {signal["Confidence"]:5.1f}%  '
              f'Entry: {signal["Entry"]:6.1f}  SL: {signal["Stop"]:6.1f}  '
              f'TP: {signal["Target"]:6.1f}')
        print(f'    Pos: {signal["Position"]:4.1f}%  Vol:{vol_icon}  '
              f'R/R: {rr_ratio:.1f}  Tech: {signal["Tech_Score"]:+.1f}  '
              f'ML: {signal["ML_Score"]:+.1f}')
        print(f'    Reason: {reasons_str[:55]}')
        print()
    
    if sell_signals:
        print(f'📉 TOP SELL OPPORTUNITIES ({len(sell_signals)} signals):')
        print('-' * 65)
        
        for i, signal in enumerate(sell_signals[:4], 1):  # Top 4 sell signals
            reasons_str = ' | '.join(signal['Reasons']) if signal['Reasons'] else 'Analysis'
            vol_icon = '✅' if signal['Volume_Support'] else '❌'
            
            risk = abs(signal['Entry'] - signal['Stop']) / signal['Entry'] * 100
            reward = abs(signal['Entry'] - signal['Target']) / signal['Entry'] * 100
            rr_ratio = reward / risk if risk > 0 else 0
            
            print(f'{i:2d}. {signal["Symbol"]:8} SELL  {signal["Confidence"]:5.1f}%  '
                  f'Entry: {signal["Entry"]:6.1f}  SL: {signal["Stop"]:6.1f}  '
                  f'TP: {signal["Target"]:6.1f}')
            print(f'    Pos: {signal["Position"]:4.1f}%  Vol:{vol_icon}  '
                  f'R/R: {rr_ratio:.1f}  Tech: {signal["Tech_Score"]:+.1f}  '
                  f'ML: {signal["ML_Score"]:+.1f}')
            print(f'    Reason: {reasons_str[:55]}')
            print()
    
    print(f'⚡ MARKET OVERVIEW:')
    print('-' * 65)
    
    total_signals = len(signals)
    buy_count = len([s for s in signals if s['Signal'] == 'BUY'])
    sell_count = len([s for s in signals if s['Signal'] == 'SELL'])
    hold_count = len([s for s in signals if s['Signal'] == 'HOLD'])
    
    if signals:
        avg_confidence = np.mean([s['Confidence'] for s in signals])
        high_confidence = len([s for s in signals if s['Confidence'] > 75])
        volume_supported = len([s for s in signals if s['Volume_Support']])
        total_position = sum(s['Position'] for s in buy_signals + sell_signals)
        
        print(f'🎯 Signal Distribution: {buy_count} BUY | {sell_count} SELL | {hold_count} HOLD')
        print(f'📊 Average Confidence: {avg_confidence:.1f}% | High Confidence (>75%): {high_confidence}')
        print(f'💰 Total Suggested Allocation: {total_position:.1f}%')
        print(f'📈 Volume Support: {volume_supported}/{total_signals} ({volume_supported/total_signals*100:.1f}%)')
        
        # Market sentiment
        if buy_count > sell_count:
            sentiment = 'Bullish 🐂'
        elif sell_count > buy_count:
            sentiment = 'Bearish 🐻'
        else:
            sentiment = 'Neutral ⚖️'
        
        signal_diff = abs(buy_count - sell_count)
        print(f'📊 Market Sentiment: {sentiment} ({signal_diff} signal difference)')
        
        # Top performers by category
        if buy_signals:
            top_buy = max(buy_signals, key=lambda x: x['Confidence'])
            print(f'⭐ Top Buy: {top_buy["Symbol"]} ({top_buy["Confidence"]:.1f}% confidence)')
        
        if sell_signals:
            top_sell = max(sell_signals, key=lambda x: x['Confidence'])
            print(f'⭐ Top Sell: {top_sell["Symbol"]} ({top_sell["Confidence"]:.1f}% confidence)')
    
    print()
    print('🔥 PRIORITY WATCHLIST:')
    print('-' * 65)
    
    # Create priority list from top signals
    priority_list = []
    if buy_signals:
        priority_list.extend(buy_signals[:3])
    if sell_signals:
        priority_list.extend(sell_signals[:2])
    
    priority_list.sort(key=lambda x: x['Confidence'], reverse=True)
    
    for i, signal in enumerate(priority_list[:5], 1):
        action_color = '🟢' if signal['Signal'] == 'BUY' else '🔴' if signal['Signal'] == 'SELL' else '🟡'
        print(f'{i}. {action_color} {signal["Symbol"]:8} {signal["Signal"]:4} '
              f'{signal["Confidence"]:5.1f}% | Entry: {signal["Entry"]:6.1f} | '
              f'Pos: {signal["Position"]:4.1f}%')
    
    print()
    print('⚠️  RISK DISCLAIMER:')
    print('• This is algorithmic analysis for educational purposes only')
    print('• Always conduct your own research and risk assessment')  
    print('• Past performance does not guarantee future results')
    print('• Consider market conditions and your risk tolerance')
    print(f'• Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == "__main__":
    generate_psx_hitlist()