#!/usr/bin/env python3
"""
Enhanced Signal Analyzer
========================

Advanced technical analysis with multiple indicators, signal strength scoring,
and comprehensive risk management for PSX trading.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import datetime as dt

# Load environment variables
def load_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    import os
                    os.environ[key] = value
    except FileNotFoundError:
        pass

load_env()

from psx_bbands_candle_scanner import compute_indicators, EODHDFetcher, TODAY


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range for volatility"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr


def find_support_resistance(df: pd.DataFrame, window: int = 10) -> Dict:
    """Find key support and resistance levels"""
    
    # Find local highs and lows
    highs = df['High'].rolling(window=window, center=True).max() == df['High']
    lows = df['Low'].rolling(window=window, center=True).min() == df['Low']
    
    resistance_levels = df[highs]['High'].tail(5).tolist()
    support_levels = df[lows]['Low'].tail(5).tolist()
    
    current_price = df['Close'].iloc[-1]
    
    # Find nearest support and resistance
    resistance_levels = [r for r in resistance_levels if r > current_price]
    support_levels = [s for s in support_levels if s < current_price]
    
    nearest_resistance = min(resistance_levels) if resistance_levels else None
    nearest_support = max(support_levels) if support_levels else None
    
    return {
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
        'all_support': support_levels,
        'all_resistance': resistance_levels
    }


def volume_analysis(df: pd.DataFrame) -> Dict:
    """Analyze volume patterns"""
    
    # Volume moving averages
    vol_ma_10 = df['Volume'].rolling(10).mean()
    vol_ma_50 = df['Volume'].rolling(50).mean()
    
    current_volume = df['Volume'].iloc[-1]
    avg_volume_10 = vol_ma_10.iloc[-1]
    avg_volume_50 = vol_ma_50.iloc[-1]
    
    # Volume surge detection
    volume_surge = current_volume > (avg_volume_50 * 1.5)
    volume_above_average = current_volume > avg_volume_10
    
    # Volume trend
    volume_trending_up = vol_ma_10.iloc[-1] > vol_ma_10.iloc[-5]
    
    return {
        'current_volume': current_volume,
        'avg_volume_10': avg_volume_10,
        'avg_volume_50': avg_volume_50,
        'volume_surge': volume_surge,
        'volume_above_average': volume_above_average,
        'volume_trending_up': volume_trending_up,
        'volume_ratio': current_volume / avg_volume_50 if avg_volume_50 > 0 else 1
    }


def enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced technical indicators"""
    
    # Start with basic indicators
    enhanced_df = compute_indicators(df)
    
    # Add RSI
    enhanced_df['RSI'] = rsi(enhanced_df['Close'])
    
    # Add ATR for volatility
    enhanced_df['ATR'] = atr(enhanced_df['High'], enhanced_df['Low'], enhanced_df['Close'])
    
    # Add volume indicators
    enhanced_df['Volume_MA_10'] = enhanced_df['Volume'].rolling(10).mean()
    enhanced_df['Volume_MA_50'] = enhanced_df['Volume'].rolling(50).mean()
    
    # Price momentum
    enhanced_df['Price_Change_5d'] = enhanced_df['Close'].pct_change(5) * 100
    enhanced_df['Price_Change_10d'] = enhanced_df['Close'].pct_change(10) * 100
    
    # Volatility indicators
    enhanced_df['Price_Volatility'] = enhanced_df['Close'].rolling(20).std()
    
    # Add trend detection (FIX: These were missing!)
    enhanced_df['MA44_slope10'] = enhanced_df['MA44'].pct_change(10)  # 10-day slope
    enhanced_df['trend_ma44_up'] = enhanced_df['MA44_slope10'] > 0.005  # 0.5% growth over 10 days
    enhanced_df['close_gt_ma44'] = enhanced_df['Close'] > enhanced_df['MA44']
    
    # Add more trend indicators
    enhanced_df['ma44_trend'] = enhanced_df['trend_ma44_up'].apply(lambda x: 'up' if x else 'down')
    
    return enhanced_df


def calculate_signal_strength(row: pd.Series, volume_data: Dict, support_resistance: Dict) -> Dict:
    """Calculate comprehensive signal strength score"""
    
    score = 0
    max_score = 100
    factors = []
    
    # 1. Trend Strength (20 points)
    if row.get('trend_ma44_up', False):
        trend_strength = min(row.get('MA44_slope10', 0) * 1000, 10)  # Scale slope
        score += 10 + trend_strength
        factors.append(f"Strong uptrend (+{10 + trend_strength:.1f})")
    else:
        factors.append("Downtrend (-10)")
        score -= 10
    
    # 2. Position vs MA44 (15 points)
    if row.get('close_gt_ma44', False):
        distance = ((row['Close'] - row['MA44']) / row['MA44']) * 100
        position_score = min(distance * 2, 15)
        score += position_score
        factors.append(f"Above MA44 (+{position_score:.1f})")
    else:
        factors.append("Below MA44 (-15)")
        score -= 15
    
    # 3. RSI Analysis (15 points) - FIXED: More momentum-friendly
    rsi_val = row.get('RSI', 50)
    if 45 <= rsi_val <= 65:  # Optimal neutral range
        rsi_score = 15
        factors.append(f"RSI optimal ({rsi_val:.1f}) (+15)")
    elif 35 <= rsi_val <= 75:  # Good range with slight momentum
        rsi_score = 12
        factors.append(f"RSI good ({rsi_val:.1f}) (+12)")
    elif 25 <= rsi_val <= 85:  # Acceptable range - includes momentum
        rsi_score = 8
        factors.append(f"RSI acceptable ({rsi_val:.1f}) (+8)")
    elif rsi_val < 25:  # Oversold - potential opportunity
        rsi_score = 5
        factors.append(f"RSI oversold ({rsi_val:.1f}) (+5)")
    else:  # Overbought but not zero - momentum can continue
        rsi_score = 3
        factors.append(f"RSI overbought ({rsi_val:.1f}) (+3)")
    score += rsi_score
    
    # 4. Volume Confirmation (15 points)
    if volume_data['volume_surge']:
        score += 15
        factors.append(f"Volume surge (+15)")
    elif volume_data['volume_above_average']:
        score += 10
        factors.append(f"Above avg volume (+10)")
    elif volume_data['volume_trending_up']:
        score += 5
        factors.append(f"Volume trending up (+5)")
    else:
        factors.append("Low volume (-5)")
        score -= 5
    
    # 5. Bollinger Band Position (10 points) - FIXED: More balanced
    bb_pctb = row.get('BB_pctB', 0.5)
    if 0.4 <= bb_pctb <= 0.7:  # Sweet spot
        bb_score = 10
        factors.append(f"BB optimal zone (+10)")
    elif 0.2 <= bb_pctb <= 0.85:  # Good range
        bb_score = 8
        factors.append(f"BB good zone (+8)")
    elif 0.1 <= bb_pctb <= 0.95:  # Acceptable - includes momentum
        bb_score = 5
        factors.append(f"BB acceptable (+5)")
    elif bb_pctb < 0.1:  # Oversold - opportunity
        bb_score = 3
        factors.append(f"BB oversold (+3)")
    else:  # Overbought but not zero
        bb_score = 2
        factors.append(f"BB overbought (+2)")
    score += bb_score
    
    # 6. Support/Resistance (10 points)
    current_price = row['Close']
    if support_resistance['nearest_support']:
        support_distance = ((current_price - support_resistance['nearest_support']) / current_price) * 100
        if support_distance < 5:  # Near support
            score += 10
            factors.append(f"Near support (+10)")
        elif support_distance < 10:
            score += 5
            factors.append(f"Above support (+5)")
    
    # 7. Momentum (10 points)
    price_change_5d = row.get('Price_Change_5d', 0)
    if price_change_5d > 2:
        momentum_score = min(price_change_5d * 2, 10)
        score += momentum_score
        factors.append(f"Strong momentum (+{momentum_score:.1f})")
    elif price_change_5d > 0:
        score += 5
        factors.append(f"Positive momentum (+5)")
    else:
        factors.append(f"Negative momentum (-5)")
        score -= 5
    
    # 8. Volatility Check (5 points)
    atr_val = row.get('ATR', 0)
    avg_price = row['Close']
    volatility_pct = (atr_val / avg_price) * 100 if avg_price > 0 else 0
    
    if volatility_pct < 3:  # Low volatility is good for entries
        score += 5
        factors.append(f"Low volatility (+5)")
    elif volatility_pct > 6:  # High volatility is risky
        score -= 5
        factors.append(f"High volatility (-5)")
    
    # CRITICAL FIX: Apply overbought penalty
    rsi_val = row.get('RSI', 50)
    bb_pctb = row.get('BB_pctB', 0.5)
    
    # Severe penalty for extreme overbought conditions
    if rsi_val > 80 and bb_pctb > 1.0:
        overbought_penalty = 25  # Major penalty
        score -= overbought_penalty
        factors.append(f"EXTREME OVERBOUGHT PENALTY (-{overbought_penalty})")
    elif rsi_val > 75 and bb_pctb > 0.9:
        overbought_penalty = 15  # Moderate penalty
        score -= overbought_penalty
        factors.append(f"Overbought penalty (-{overbought_penalty})")
    elif rsi_val > 70 and bb_pctb > 0.85:
        overbought_penalty = 8  # Light penalty
        score -= overbought_penalty
        factors.append(f"Mild overbought penalty (-{overbought_penalty})")
    
    # Normalize score to 0-100
    final_score = max(0, min(100, score))
    
    # Determine signal grade - FIXED: More reasonable thresholds
    if final_score >= 75:
        grade = "A"
        recommendation = "STRONG BUY"
    elif final_score >= 60:
        grade = "B"
        recommendation = "BUY"
    elif final_score >= 45:
        grade = "C"
        recommendation = "WEAK BUY"
    elif final_score >= 30:
        grade = "D"
        recommendation = "HOLD"
    else:
        grade = "F"
        recommendation = "AVOID"
    
    return {
        'score': final_score,
        'grade': grade,
        'recommendation': recommendation,
        'factors': factors,  # Show all factors including penalties
        'rsi': rsi_val,
        'volume_ratio': volume_data['volume_ratio']
    }


def calculate_risk_management(row: pd.Series, support_resistance: Dict) -> Dict:
    """Calculate risk management suggestions"""
    
    current_price = row['Close']
    atr_val = row.get('ATR', current_price * 0.02)  # Default 2% if no ATR
    
    # Stop loss suggestions
    atr_stop = current_price - (atr_val * 2)  # 2x ATR stop
    support_stop = support_resistance['nearest_support'] if support_resistance['nearest_support'] else current_price * 0.95
    ma44_stop = row['MA44'] * 0.98  # 2% below MA44
    
    stop_loss = max(atr_stop, support_stop, ma44_stop)
    stop_loss_pct = ((current_price - stop_loss) / current_price) * 100
    
    # Take profit targets
    resistance = support_resistance['nearest_resistance']
    if resistance:
        target1 = resistance * 0.95  # 5% before resistance
        target2 = resistance * 1.02  # Break above resistance
    else:
        target1 = current_price * 1.08  # 8% gain
        target2 = current_price * 1.15  # 15% gain
    
    target1_pct = ((target1 - current_price) / current_price) * 100
    target2_pct = ((target2 - current_price) / current_price) * 100
    
    # Risk-reward ratio
    avg_target = (target1 + target2) / 2
    avg_gain_pct = ((avg_target - current_price) / current_price) * 100
    risk_reward = avg_gain_pct / stop_loss_pct if stop_loss_pct > 0 else 0
    
    # Position sizing (based on 2% account risk)
    account_risk_pct = 2.0  # Risk 2% of account per trade
    # This would need actual account size, for now show formula
    
    return {
        'stop_loss': stop_loss,
        'stop_loss_pct': stop_loss_pct,
        'target1': target1,
        'target1_pct': target1_pct,
        'target2': target2,
        'target2_pct': target2_pct,
        'risk_reward_ratio': risk_reward,
        'account_risk_pct': account_risk_pct,
        'atr_volatility': (atr_val / current_price) * 100
    }


def get_correct_symbol_format(symbol: str) -> str:
    """Get the correct EODHD format for Pakistani stocks"""
    
    # Remove any existing suffix
    base_symbol = symbol.upper().split('.')[0]
    
    # Special symbol mappings for EODHD (common issues)
    symbol_mappings = {
        'ABBOTT': 'ABT.KAR',
        'DG.KHAN': 'DGKC.KAR', 
        'SEARLE': 'SRLE.KAR',
        'NESTLE': 'NESTLE.KAR',
        'COLG': 'COLGATE.KAR',
        'GLAXO': 'GLAXO.KAR',
        'SHIELD': 'SHIELD.KAR',
        'MARTIN': 'MARTIN.KAR',
        'HIGHNOON': 'HNOON.KAR',
        'TELECARD': 'TCARD.KAR',
        'GHANDHARA': 'GHNI.KAR',
        'CHERAT': 'CPPC.KAR'
    }
    
    # Check if symbol has a special mapping
    if base_symbol in symbol_mappings:
        return symbol_mappings[base_symbol]
    
    # Try different exchange suffixes for Pakistani stocks
    possible_suffixes = ['.KAR', '.PSX', '.XKAR']
    
    # Default to .KAR if no special mapping
    return f"{base_symbol}.KAR"


def enhanced_signal_analysis(symbol: str, days: int = 260) -> Dict:
    """Complete enhanced signal analysis for a symbol"""
    
    try:
        # Fetch data with correct symbol format
        import os
        fetcher = EODHDFetcher(os.getenv('EODHD_API_KEY'))
        end = TODAY
        start = end - dt.timedelta(days=days+40)
        
        # Get correct symbol format
        formatted_symbol = get_correct_symbol_format(symbol)
        
        try:
            raw_data = fetcher.fetch(formatted_symbol, start, end)
        except Exception as e:
            # If primary format fails, try alternatives
            base_symbol = symbol.upper().split('.')[0]
            alternative_formats = [f"{base_symbol}.PSX", f"{base_symbol}.XKAR", f"{base_symbol}.KAR"]
            
            raw_data = None
            last_error = str(e)
            
            for alt_symbol in alternative_formats:
                try:
                    raw_data = fetcher.fetch(alt_symbol, start, end)
                    formatted_symbol = alt_symbol
                    break
                except Exception as alt_e:
                    last_error = str(alt_e)
                    continue
            
            if raw_data is None:
                return {
                    'error': f'Could not fetch data for {symbol}. Tried formats: {alternative_formats}. Last error: {last_error}'
                }
        
        # Calculate enhanced indicators
        enhanced_data = enhanced_indicators(raw_data)
        
        if enhanced_data.empty:
            return {'error': 'Insufficient data for analysis'}
        
        latest = enhanced_data.iloc[-1]
        
        # Analyze volume
        volume_data = volume_analysis(enhanced_data)
        
        # Find support/resistance
        support_resistance = find_support_resistance(enhanced_data)
        
        # Calculate signal strength
        signal_strength = calculate_signal_strength(latest, volume_data, support_resistance)
        
        # Calculate risk management
        risk_mgmt = calculate_risk_management(latest, support_resistance)
        
        # Enhanced signal evaluation
        enhanced_signal = {
            'symbol': formatted_symbol,
            'date': latest['Date'].strftime('%Y-%m-%d'),
            'price': float(latest['Close']),
            'signal_strength': signal_strength,
            'risk_management': risk_mgmt,
            'technical_data': {
                'ma44': float(latest['MA44']),
                'bb_pctb': float(latest['BB_pctB']),
                'rsi': float(latest['RSI']),
                'atr': float(latest['ATR']),
                'volume_ratio': volume_data['volume_ratio'],
                'ma44_trend': latest.get('ma44_trend', 'unknown'),
                'ma44_slope': float(latest.get('MA44_slope10', 0)),
                'trend_ma44_up': bool(latest.get('trend_ma44_up', False)),
                'close_gt_ma44': bool(latest.get('close_gt_ma44', False))
            },
            'support_resistance': support_resistance,
            'volume_analysis': volume_data
        }
        
        return enhanced_signal
        
    except Exception as e:
        return {'error': str(e)}


# Test function
if __name__ == '__main__':
    import os
    
    test_symbols = ['UBL', 'MCB', 'FFC']
    
    print("🚀 Enhanced Signal Analysis Test")
    print("=" * 50)
    
    for symbol in test_symbols:
        print(f"\n📊 Analyzing {symbol}...")
        result = enhanced_signal_analysis(symbol)
        
        if 'error' not in result:
            strength = result['signal_strength']
            print(f"   Signal: {strength['grade']} ({strength['score']:.1f}/100) - {strength['recommendation']}")
            print(f"   Price: {result['price']:.2f} | RSI: {strength['rsi']:.1f}")
            print(f"   Risk/Reward: {result['risk_management']['risk_reward_ratio']:.2f}")
        else:
            print(f"   Error: {result['error']}")