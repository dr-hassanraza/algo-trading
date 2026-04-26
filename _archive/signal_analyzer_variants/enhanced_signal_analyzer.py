#!/usr/bin/env python3
"""
Enhanced Signal Analyzer
========================

Advanced technical analysis with multiple indicators, signal strength scoring,
and comprehensive risk management for PSX trading. Enhanced with additional
technical indicators (MACD, Stochastic, ADX), fundamental data integration
via EODHD API, and adjusted scoring for better accuracy in PSX market.
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import datetime as dt
import requests
from bs4 import BeautifulSoup  # For potential scraping if needed
import json
import time
from textblob import TextBlob  # For basic sentiment analysis
import yfinance as yf  # Alternative data source
import ml_model # Import the new machine learning model

# Load environment variables
def load_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass

load_env()

from psx_bbands_candle_scanner import compute_indicators, EODHDFetcher, TODAY
from enhanced_data_fetcher import EnhancedDataFetcher
from real_time_price_fetcher import get_enhanced_price_info


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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD Histogram"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # Histogram


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Stochastic Oscillator (%K)"""
    l14 = low.rolling(window=period).min()
    h14 = high.rolling(window=period).max()
    return 100 * (close - l14) / (h14 - l14)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (ADX) - Simplified"""
    tr = atr(high, low, close, period)
    # Full ADX requires +DI, -DI; this is a placeholder approximation
    # For better accuracy, implement full +DI/-DI
    up = high - high.shift(1)
    down = low.shift(1) - low
    pdm = np.where((up > down) & (up > 0), up, 0)
    mdm = np.where((down > up) & (down > 0), down, 0)
    pdm = pd.Series(pdm).rolling(period).mean()
    mdm = pd.Series(mdm).rolling(period).mean()
    pdi = 100 * pdm / tr
    mdi = 100 * mdm / tr
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = dx.rolling(period).mean()
    return adx


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


def get_sector_info(symbol: str) -> Dict:
    """Get sector and industry information for PSX stocks"""
    # PSX sector mapping based on common sectors
    sector_mapping = {
        # Banking & Financial
        'UBL': {'sector': 'Banking', 'industry': 'Commercial Banks', 'market_cap_tier': 'Large'},
        'MCB': {'sector': 'Banking', 'industry': 'Commercial Banks', 'market_cap_tier': 'Large'},
        'NBP': {'sector': 'Banking', 'industry': 'Commercial Banks', 'market_cap_tier': 'Large'},
        'ABL': {'sector': 'Banking', 'industry': 'Commercial Banks', 'market_cap_tier': 'Large'},
        'HBL': {'sector': 'Banking', 'industry': 'Commercial Banks', 'market_cap_tier': 'Large'},
        'BAFL': {'sector': 'Banking', 'industry': 'Commercial Banks', 'market_cap_tier': 'Medium'},
        'AKBL': {'sector': 'Banking', 'industry': 'Commercial Banks', 'market_cap_tier': 'Medium'},
        
        # Oil & Gas
        'OGDC': {'sector': 'Oil & Gas', 'industry': 'Oil & Gas Exploration', 'market_cap_tier': 'Large'},
        'PPL': {'sector': 'Oil & Gas', 'industry': 'Oil & Gas Exploration', 'market_cap_tier': 'Large'},
        'POL': {'sector': 'Oil & Gas', 'industry': 'Oil & Gas Marketing', 'market_cap_tier': 'Large'},
        'PSO': {'sector': 'Oil & Gas', 'industry': 'Oil & Gas Marketing', 'market_cap_tier': 'Large'},
        'MARI': {'sector': 'Oil & Gas', 'industry': 'Oil & Gas Exploration', 'market_cap_tier': 'Large'},
        
        # Cement
        'LUCK': {'sector': 'Cement', 'industry': 'Cement Manufacturing', 'market_cap_tier': 'Large'},
        'DGKC': {'sector': 'Cement', 'industry': 'Cement Manufacturing', 'market_cap_tier': 'Large'},
        'MLCF': {'sector': 'Cement', 'industry': 'Cement Manufacturing', 'market_cap_tier': 'Medium'},
        
        # Fertilizer
        'ENGRO': {'sector': 'Fertilizer', 'industry': 'Fertilizer Manufacturing', 'market_cap_tier': 'Large'},
        'FFC': {'sector': 'Fertilizer', 'industry': 'Fertilizer Manufacturing', 'market_cap_tier': 'Large'},
        'FFBL': {'sector': 'Fertilizer', 'industry': 'Fertilizer Manufacturing', 'market_cap_tier': 'Medium'},
        
        # Power
        'HUBCO': {'sector': 'Power', 'industry': 'Power Generation', 'market_cap_tier': 'Large'},
        'KAPCO': {'sector': 'Power', 'industry': 'Power Generation', 'market_cap_tier': 'Medium'},
        
        # Food & Personal Care
        'NESTLE': {'sector': 'Food', 'industry': 'Food Products', 'market_cap_tier': 'Large'},
        'UFL': {'sector': 'Food', 'industry': 'Food Products', 'market_cap_tier': 'Medium'},
        
        # Technology
        'TRG': {'sector': 'Technology', 'industry': 'IT Services', 'market_cap_tier': 'Medium'},
        'NETSOL': {'sector': 'Technology', 'industry': 'Software', 'market_cap_tier': 'Small'},
        
        # Pharmaceuticals
        'GLAXO': {'sector': 'Pharmaceuticals', 'industry': 'Pharmaceuticals', 'market_cap_tier': 'Medium'},
        'ICI': {'sector': 'Pharmaceuticals', 'industry': 'Pharmaceuticals', 'market_cap_tier': 'Medium'},
    }
    
    base_symbol = symbol.upper().split('.')[0]
    return sector_mapping.get(base_symbol, {
        'sector': 'Unknown', 
        'industry': 'Unknown', 
        'market_cap_tier': 'Unknown'
    })


def get_market_sentiment() -> Dict:
    """Get overall PSX/KSE100 market sentiment indicators"""
    try:
        # Simulate market sentiment based on recent performance
        # In production, this could fetch real sentiment data
        
        current_date = dt.datetime.now()
        
        # Simulate market metrics (in production, fetch from APIs)
        kse100_change = 92.46  # YoY change based on research
        monthly_change = 8.97  # Monthly change
        
        # Market sentiment score (0-100)
        if kse100_change > 50:
            sentiment_score = 85
            sentiment_label = "Very Bullish"
        elif kse100_change > 20:
            sentiment_score = 70
            sentiment_label = "Bullish"
        elif kse100_change > 0:
            sentiment_score = 55
            sentiment_label = "Mildly Positive"
        elif kse100_change > -10:
            sentiment_score = 40
            sentiment_label = "Neutral"
        else:
            sentiment_score = 25
            sentiment_label = "Bearish"
        
        # VIX-like volatility indicator (simulated)
        volatility_index = max(10, min(50, 20 + abs(monthly_change - 5)))
        
        return {
            'kse100_ytd_change': kse100_change,
            'kse100_monthly_change': monthly_change,
            'market_sentiment_score': sentiment_score,
            'market_sentiment_label': sentiment_label,
            'volatility_index': volatility_index,
            'market_phase': 'Bull Market' if kse100_change > 20 else 'Bear Market' if kse100_change < -20 else 'Sideways',
            'last_updated': current_date.strftime('%Y-%m-%d %H:%M')
        }
        
    except Exception as e:
        return {
            'kse100_ytd_change': 0,
            'kse100_monthly_change': 0,
            'market_sentiment_score': 50,
            'market_sentiment_label': 'Neutral',
            'volatility_index': 25,
            'market_phase': 'Unknown',
            'last_updated': 'N/A',
            'error': str(e)
        }


def analyze_news_sentiment(symbol: str) -> Dict:
    """Basic news sentiment analysis for the symbol"""
    try:
        # In production, this would fetch real news data from sources like:
        # - Google News API
        # - Financial news APIs  
        # - Social media sentiment
        
        # For now, simulate sentiment based on symbol characteristics
        base_symbol = symbol.upper().split('.')[0]
        sector_info = get_sector_info(base_symbol)
        
        # Simulate sector-based sentiment
        sector_sentiment = {
            'Banking': 75,  # Banking sector generally positive in Pakistan
            'Oil & Gas': 70,  # Energy sector strength
            'Cement': 65,   # Construction growth
            'Fertilizer': 68,  # Agricultural importance
            'Technology': 80,  # Tech sector optimism
            'Food': 72,     # Consumer staples
            'Pharmaceuticals': 78,  # Healthcare demand
            'Power': 60     # Regulatory challenges
        }
        
        sector = sector_info.get('sector', 'Unknown')
        base_sentiment = sector_sentiment.get(sector, 60)
        
        # Add some randomness to simulate news impact
        import random
        random.seed(hash(symbol) % 1000)  # Consistent randomness per symbol
        news_impact = random.randint(-10, 15)  # Slight positive bias
        
        final_sentiment = max(0, min(100, base_sentiment + news_impact))
        
        # Categorize sentiment
        if final_sentiment >= 75:
            sentiment_label = 'Very Positive'
        elif final_sentiment >= 60:
            sentiment_label = 'Positive'
        elif final_sentiment >= 40:
            sentiment_label = 'Neutral'
        elif final_sentiment >= 25:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Very Negative'
        
        return {
            'news_sentiment_score': final_sentiment,
            'news_sentiment_label': sentiment_label,
            'sector_sentiment': base_sentiment,
            'news_impact': news_impact,
            'confidence': 0.6  # Medium confidence for simulated data
        }
        
    except Exception as e:
        return {
            'news_sentiment_score': 50,
            'news_sentiment_label': 'Neutral',
            'sector_sentiment': 50,
            'news_impact': 0,
            'confidence': 0.3,
            'error': str(e)
        }


def fetch_fundamentals(symbol: str, api_key: str) -> Dict:
    """Enhanced fundamental data fetching with additional metrics"""
    # First try EODHD API (existing functionality)
    url = f"https://eodhd.com/api/fundamentals/{symbol}?api_token={api_key}&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        valuation = data.get('Valuation', {})
        pe = valuation.get('TrailingPE')
        eps = valuation.get('TrailingEPS')
        
        # Try to get additional metrics
        financials = data.get('Financials', {})
        balance_sheet = financials.get('Balance_Sheet', {})
        income_statement = financials.get('Income_Statement', {})
        
        return {
            'PE': float(pe) if pe else None,
            'EPS': float(eps) if eps else None,
            'book_value': valuation.get('BookValue'),
            'pb_ratio': valuation.get('PriceBookMRQ'),
            'debt_equity': balance_sheet.get('totalDebt', {}).get('yearly', {}).get('2023') if balance_sheet else None,
            'roe': valuation.get('ReturnOnEquityTTM'),
            'roa': valuation.get('ReturnOnAssetsTTM'),
            'dividend_yield': valuation.get('DividendYieldTTM'),
            'source': 'EODHD'
        }
    except Exception as e:
        print(f"Error fetching fundamentals for {symbol}: {e}")
        
        # Fallback: Estimate fundamentals based on sector and market cap
        return estimate_fundamentals(symbol)


def estimate_fundamentals(symbol: str) -> Dict:
    """Estimate fundamental metrics based on sector averages for PSX"""
    base_symbol = symbol.upper().split('.')[0]
    sector_info = get_sector_info(base_symbol)
    sector = sector_info.get('sector', 'Unknown')
    
    # Sector-based fundamental estimates for PSX (based on research)
    sector_fundamentals = {
        'Banking': {'pe': 8.5, 'pb': 1.2, 'roe': 15.5, 'dividend_yield': 4.2},
        'Oil & Gas': {'pe': 12.0, 'pb': 1.8, 'roe': 18.0, 'dividend_yield': 6.5},
        'Cement': {'pe': 6.8, 'pb': 1.4, 'roe': 22.0, 'dividend_yield': 8.0},
        'Fertilizer': {'pe': 7.2, 'pb': 2.1, 'roe': 28.0, 'dividend_yield': 7.5},
        'Power': {'pe': 5.5, 'pb': 0.9, 'roe': 12.0, 'dividend_yield': 12.0},
        'Technology': {'pe': 15.0, 'pb': 3.5, 'roe': 25.0, 'dividend_yield': 2.0},
        'Food': {'pe': 11.0, 'pb': 2.5, 'roe': 20.0, 'dividend_yield': 5.0},
        'Pharmaceuticals': {'pe': 13.5, 'pb': 2.8, 'roe': 19.0, 'dividend_yield': 3.5}
    }
    
    defaults = {'pe': 10.0, 'pb': 1.5, 'roe': 15.0, 'dividend_yield': 5.0}
    fundamentals = sector_fundamentals.get(sector, defaults)
    
    # Add some variation based on market cap tier
    market_cap_tier = sector_info.get('market_cap_tier', 'Medium')
    if market_cap_tier == 'Large':
        # Large caps typically have lower PE, higher stability
        fundamentals['pe'] *= 0.9
        fundamentals['dividend_yield'] *= 1.1
    elif market_cap_tier == 'Small':
        # Small caps typically have higher PE, higher growth potential
        fundamentals['pe'] *= 1.3
        fundamentals['roe'] *= 1.2
        fundamentals['dividend_yield'] *= 0.8
    
    return {
        'PE': fundamentals['pe'],
        'EPS': None,  # Cannot estimate without price
        'PB_ratio': fundamentals['pb'],
        'ROE': fundamentals['roe'],
        'dividend_yield': fundamentals['dividend_yield'],
        'source': 'Estimated',
        'sector_avg': True
    }


def enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced technical indicators including MACD, Stochastic, ADX"""

    # Start with basic indicators
    enhanced_df = compute_indicators(df)

    # Add RSI
    enhanced_df['RSI'] = rsi(enhanced_df['Close'])

    # Add ATR for volatility
    enhanced_df['ATR'] = atr(enhanced_df['High'], enhanced_df['Low'], enhanced_df['Close'])

    # Add MACD Histogram
    enhanced_df['MACD_Hist'] = macd(enhanced_df['Close'])

    # Add Stochastic
    enhanced_df['Stoch'] = stochastic(enhanced_df['High'], enhanced_df['Low'], enhanced_df['Close'])

    # Add ADX
    enhanced_df['ADX'] = adx(enhanced_df['High'], enhanced_df['Low'], enhanced_df['Close'])

    # Add volume indicators
    enhanced_df['Volume_MA_10'] = enhanced_df['Volume'].rolling(10).mean()
    enhanced_df['Volume_MA_50'] = enhanced_df['Volume'].rolling(50).mean()

    # Price momentum
    enhanced_df['Price_Change_5d'] = enhanced_df['Close'].pct_change(5) * 100
    enhanced_df['Price_Change_10d'] = enhanced_df['Close'].pct_change(10) * 100

    # Volatility indicators
    enhanced_df['Price_Volatility'] = enhanced_df['Close'].rolling(20).std()

    # Add trend detection
    enhanced_df['MA44_slope10'] = enhanced_df['MA44'].pct_change(10)  # 10-day slope
    enhanced_df['trend_ma44_up'] = enhanced_df['MA44_slope10'] > 0.005  # 0.5% growth over 10 days
    enhanced_df['close_gt_ma44'] = enhanced_df['Close'] > enhanced_df['MA44']

    # Add more trend indicators
    enhanced_df['ma44_trend'] = enhanced_df['trend_ma44_up'].apply(lambda x: 'up' if x else 'down')

    return enhanced_df


def calculate_signal_strength(row: pd.Series, volume_data: Dict, support_resistance: Dict, fundamentals: Dict, market_sentiment: Dict = None, news_sentiment: Dict = None, sector_info: Dict = None, ml_prediction: Dict = None) -> Dict:
    """Calculate comprehensive signal strength score with enhanced fundamental and sentiment analysis"""

    score = 0
    max_score = 100
    factors = []
    
    # Initialize sentiment data if not provided
    if market_sentiment is None:
        market_sentiment = get_market_sentiment()
    if news_sentiment is None:
        news_sentiment = {'news_sentiment_score': 50}
    if sector_info is None:
        sector_info = {'sector': 'Unknown'}
    if ml_prediction is None:
        ml_prediction = {'status': 'error', 'message': 'ML model not used'}

    # 1. Trend Strength (20 points) - FIXED: More realistic trend scoring
    ma44_slope = row.get('MA44_slope10', 0)
    if row.get('trend_ma44_up', False):
        # More conservative trend scoring - require stronger slopes
        if ma44_slope > 0.02:  # Strong uptrend (>2%)
            trend_strength = min(ma44_slope * 500, 10)  # Reduced multiplier
            score += 15 + trend_strength
            factors.append(f"Strong uptrend ({ma44_slope*100:.1f}%) (+{15 + trend_strength:.1f})")
        elif ma44_slope > 0.01:  # Moderate uptrend (>1%)
            score += 12
            factors.append(f"Moderate uptrend ({ma44_slope*100:.1f}%) (+12)")
        else:  # Weak uptrend
            score += 7
            factors.append(f"Weak uptrend ({ma44_slope*100:.1f}%) (+7)")
    else:
        # Downtrend penalty based on slope strength
        if ma44_slope < -0.02:
            score -= 15
            factors.append(f"Strong downtrend ({ma44_slope*100:.1f}%) (-15)")
        else:
            score -= 8
            factors.append(f"Downtrend ({ma44_slope*100:.1f}%) (-8)")

    # 2. Position vs MA44 (15 points)
    if row.get('close_gt_ma44', False):
        distance = ((row['Close'] - row['MA44']) / row['MA44']) * 100
        position_score = min(distance * 2, 15)
        score += position_score
        factors.append(f"Above MA44 (+{position_score:.1f})")
    else:
        factors.append("Below MA44 (-15)")
        score -= 15

    # 3. RSI Analysis (15 points)
    rsi_val = row.get('RSI', 50)
    if 45 <= rsi_val <= 65:
        rsi_score = 15
        factors.append(f"RSI optimal ({rsi_val:.1f}) (+15)")
    elif 35 <= rsi_val <= 75:
        rsi_score = 12
        factors.append(f"RSI good ({rsi_val:.1f}) (+12)")
    elif 25 <= rsi_val <= 85:
        rsi_score = 8
        factors.append(f"RSI acceptable ({rsi_val:.1f}) (+8)")
    elif rsi_val < 25:
        rsi_score = 5
        factors.append(f"RSI oversold ({rsi_val:.1f}) (+5)")
    else:
        rsi_score = 3
        factors.append(f"RSI overbought ({rsi_val:.1f}) (+3)")
    score += rsi_score

    # 4. Volume Confirmation (15 points) - FIXED: More conservative volume scoring
    volume_ratio = volume_data['volume_ratio']
    if volume_data['volume_surge']:  # >1.5x average
        score += 15
        factors.append(f"Volume surge ({volume_ratio:.1f}x) (+15)")
    elif volume_data['volume_above_average'] and volume_ratio > 1.2:
        score += 10
        factors.append(f"Strong volume ({volume_ratio:.1f}x) (+10)")
    elif volume_data['volume_above_average']:
        score += 5
        factors.append(f"Above avg volume ({volume_ratio:.1f}x) (+5)")
    elif volume_data['volume_trending_up']:
        score += 3
        factors.append(f"Volume trending up (+3)")
    elif volume_ratio < 0.7:  # Low volume penalty
        score -= 8
        factors.append(f"Low volume ({volume_ratio:.1f}x) (-8)")
    else:
        score -= 3
        factors.append(f"Below average volume ({volume_ratio:.1f}x) (-3)")

    # 5. Bollinger Band Position (10 points)
    bb_pctb = row.get('BB_pctB', 0.5)
    if 0.4 <= bb_pctb <= 0.7:
        bb_score = 10
        factors.append(f"BB optimal zone (+10)")
    elif 0.2 <= bb_pctb <= 0.85:
        bb_score = 8
        factors.append(f"BB good zone (+8)")
    elif 0.1 <= bb_pctb <= 0.95:
        bb_score = 5
        factors.append(f"BB acceptable (+5)")
    elif bb_pctb < 0.1:
        bb_score = 3
        factors.append(f"BB oversold (+3)")
    else:
        bb_score = 2
        factors.append(f"BB overbought (+2)")
    score += bb_score

    # 6. Support/Resistance (10 points)
    current_price = row['Close']
    if support_resistance['nearest_support']:
        support_distance = ((current_price - support_resistance['nearest_support']) / current_price) * 100
        if support_distance < 5:
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

    if volatility_pct < 3:
        score += 5
        factors.append(f"Low volatility (+5)")
    elif volatility_pct > 6:
        score -= 5
        factors.append(f"High volatility (-5)")

    # 9. MACD Confirmation (10 points)
    macd_hist = row.get('MACD_Hist', 0)
    if macd_hist > 0:
        macd_score = min(macd_hist * 10, 10) if macd_hist > 0 else 0  # Scale
        score += macd_score
        factors.append(f"Positive MACD (+{macd_score:.1f})")
    else:
        score -= 5
        factors.append(f"Negative MACD (-5)")

    # 10. Stochastic Analysis (10 points)
    stoch_val = row.get('Stoch', 50)
    if 20 < stoch_val < 80:
        stoch_score = 10
        factors.append(f"Stochastic in good range (+10)")
    elif stoch_val <= 20:
        stoch_score = 5  # Oversold opportunity
        factors.append(f"Stochastic oversold (+5)")
    else:
        stoch_score = 3  # Overbought but momentum
        factors.append(f"Stochastic overbought (+3)")
    score += stoch_score

    # 11. ADX Trend Strength (10 points)
    adx_val = row.get('ADX', 0)
    if adx_val > 25:
        score += 10
        factors.append(f"Strong trend (ADX >25) (+10)")
    elif adx_val > 20:
        score += 5
        factors.append(f"Moderate trend (ADX >20) (+5)")
    else:
        score -= 5
        factors.append(f"Weak trend (ADX low) (-5)")

    # 12. Enhanced Fundamentals Analysis (15 points)
    pe = fundamentals.get('PE')
    pb_ratio = fundamentals.get('PB_ratio')
    roe = fundamentals.get('ROE')
    dividend_yield = fundamentals.get('dividend_yield')
    
    # PE Ratio Analysis (8 points)
    if pe is not None:
        if pe < 6:  # Very undervalued for PSX
            score += 8
            factors.append(f"Extremely undervalued (PE {pe:.1f}) (+8)")
        elif pe < 10:
            score += 6
            factors.append(f"Undervalued (PE {pe:.1f}) (+6)")
        elif pe < 15:
            score += 3
            factors.append(f"Fair valuation (PE {pe:.1f}) (+3)")
        elif pe > 25:
            score -= 5
            factors.append(f"Overvalued (PE {pe:.1f}) (-5)")
    
    # ROE Analysis (4 points)
    if roe is not None:
        if roe > 20:
            score += 4
            factors.append(f"Excellent ROE ({roe:.1f}%) (+4)")
        elif roe > 15:
            score += 2
            factors.append(f"Good ROE ({roe:.1f}%) (+2)")
        elif roe < 10:
            score -= 2
            factors.append(f"Poor ROE ({roe:.1f}%) (-2)")
    
    # Dividend Yield (3 points)
    if dividend_yield is not None:
        if dividend_yield > 8:  # High dividend yield for PSX
            score += 3
            factors.append(f"High dividend yield ({dividend_yield:.1f}%) (+3)")
        elif dividend_yield > 5:
            score += 1
            factors.append(f"Good dividend yield ({dividend_yield:.1f}%) (+1)")

    # 13. Market Sentiment Impact (5 points) - REDUCED: More conservative market sentiment
    market_score = market_sentiment.get('market_sentiment_score', 50)
    # Reduce impact of market sentiment to avoid inflated scores
    if market_score >= 80:
        sentiment_boost = 5
        factors.append(f"Very bullish market (+5)")
    elif market_score >= 65:
        sentiment_boost = 3
        factors.append(f"Bullish market (+3)")
    elif market_score >= 55:
        sentiment_boost = 1
        factors.append(f"Positive market (+1)")
    elif market_score <= 35:
        sentiment_boost = -4
        factors.append(f"Bearish market (-4)")
    else:
        sentiment_boost = 0
    score += sentiment_boost

    # 14. News Sentiment (3 points) - REDUCED: Conservative news sentiment impact
    news_score = news_sentiment.get('news_sentiment_score', 50)
    if news_score >= 80:
        score += 3
        factors.append(f"Very positive news sentiment (+3)")
    elif news_score >= 65:
        score += 2
        factors.append(f"Positive news sentiment (+2)")
    elif news_score <= 25:
        score -= 3
        factors.append(f"Very negative news sentiment (-3)")
    elif news_score <= 40:
        score -= 1
        factors.append(f"Negative news sentiment (-1)")

    # 15. Sector Performance Bonus (4 points)
    sector = sector_info.get('sector', 'Unknown')
    sector_multipliers = {
        'Technology': 1.2,  # Tech sector premium
        'Banking': 1.1,     # Stable banking sector
        'Oil & Gas': 1.1,   # Energy importance
        'Pharmaceuticals': 1.15,  # Healthcare demand
        'Fertilizer': 1.1,  # Agricultural importance
        'Cement': 1.0,      # Cyclical sector
        'Power': 0.9        # Regulatory challenges
    }
    
    sector_multiplier = sector_multipliers.get(sector, 1.0)
    if sector_multiplier > 1.05:
        sector_bonus = 4
        factors.append(f"Strong sector ({sector}) (+4)")
    elif sector_multiplier < 0.95:
        sector_bonus = -2
        factors.append(f"Challenging sector ({sector}) (-2)")
    else:
        sector_bonus = 0
    score += sector_bonus

    # 16. Machine Learning Prediction (15 points)
    if ml_prediction and ml_prediction['status'] == 'success':
        probability = ml_prediction['probability']
        if ml_prediction['prediction'] == 1:
            ml_score = 0
            if probability > 0.7:
                ml_score = 15
            elif probability > 0.6:
                ml_score = 10
            elif probability > 0.5:
                ml_score = 5
            score += ml_score
            factors.append(f"ML Model BUY ({probability:.2f}) (+{ml_score})")
        else:
            score -= 10
            factors.append(f"ML Model SELL ({probability:.2f}) (-10)")


    # Enhanced penalty system - FIXED: More comprehensive risk factors
    
    # 1. Overbought penalty (enhanced)
    if rsi_val > 80 and bb_pctb > 1.0:
        overbought_penalty = 25
        score -= overbought_penalty
        factors.append(f"EXTREME OVERBOUGHT PENALTY (-{overbought_penalty})")
    elif rsi_val > 75 and bb_pctb > 0.9:
        overbought_penalty = 15
        score -= overbought_penalty
        factors.append(f"Overbought penalty (-{overbought_penalty})")
    elif rsi_val > 70 and bb_pctb > 0.85:
        overbought_penalty = 8
        score -= overbought_penalty
        factors.append(f"Mild overbought penalty (-{overbought_penalty})")
    
    # 2. Low volume penalty for trend stocks
    if row.get('trend_ma44_up', False) and volume_ratio < 0.8:
        volume_penalty = 10
        score -= volume_penalty
        factors.append(f"Uptrend with low volume (-{volume_penalty})")
    
    # 3. Conflicting signals penalty
    conflicting_signals = 0
    if row.get('trend_ma44_up', False) and macd_hist < 0:
        conflicting_signals += 1
    if row.get('close_gt_ma44', False) and price_change_5d < -2:
        conflicting_signals += 1
    if volume_ratio < 0.7 and row.get('trend_ma44_up', False):
        conflicting_signals += 1
        
    if conflicting_signals >= 2:
        mixed_signals_penalty = 8
        score -= mixed_signals_penalty
        factors.append(f"Mixed signals penalty (-{mixed_signals_penalty})")

    # Normalize score to 0-100 range
    final_score = max(0, min(100, score))

    # FIXED: More conservative grading system
    sector_adjustment = 0
    if sector in ['Technology', 'Pharmaceuticals']:
        sector_adjustment = 1  # Reduced premium
    elif sector in ['Power']:
        sector_adjustment = -2  # Reduced penalty
    
    # Apply volume-based final adjustment
    volume_final_adjustment = 0
    if volume_ratio < 0.5:  # Very low volume is concerning
        volume_final_adjustment = -5
    elif volume_ratio > 2.0:  # Very high volume is positive
        volume_final_adjustment = 2
    
    adjusted_score = max(0, min(100, final_score + sector_adjustment + volume_final_adjustment))

    # More conservative grading thresholds - FIXED
    if adjusted_score >= 85:
        grade = "A+"
        recommendation = "STRONG BUY"
    elif adjusted_score >= 75:
        grade = "A"
        recommendation = "STRONG BUY"
    elif adjusted_score >= 65:
        grade = "B+"
        recommendation = "BUY"
    elif adjusted_score >= 55:
        grade = "B"
        recommendation = "BUY"
    elif adjusted_score >= 45:
        grade = "C+"
        recommendation = "WEAK BUY"
    elif adjusted_score >= 35:
        grade = "C"
        recommendation = "WEAK BUY"
    elif adjusted_score >= 25:
        grade = "D"
        recommendation = "HOLD"
    else:
        grade = "F"
        recommendation = "AVOID"

    return {
        'score': adjusted_score,
        'raw_score': final_score,
        'grade': grade,
        'recommendation': recommendation,
        'factors': factors,
        'rsi': rsi_val,
        'volume_ratio': volume_data['volume_ratio'],
        'market_sentiment': market_sentiment.get('market_sentiment_label', 'Unknown'),
        'news_sentiment': news_sentiment.get('news_sentiment_label', 'Unknown'),
        'sector': sector,
        'sector_adjustment': sector_adjustment,
        'volume_adjustment': volume_final_adjustment,
        'volume_ratio_actual': volume_ratio
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
    """Complete enhanced signal analysis with fundamental and sentiment analysis"""

    try:
        # Fetch data with correct symbol format
        api_key = os.getenv('EODHD_API_KEY')
        fetcher = EnhancedDataFetcher(api_key)
        end = TODAY
        start = end - dt.timedelta(days=days+40)

        # Get correct symbol format
        formatted_symbol = get_correct_symbol_format(symbol)
        base_symbol = symbol.upper().split('.')[0]

        try:
            raw_data = fetcher.fetch(formatted_symbol, start, end)
        except Exception as e:
            # If primary format fails, try alternatives
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

        # Get ML prediction
        ml_prediction = ml_model.predict_signal(enhanced_data)

        # Get enhanced analysis data
        sector_info = get_sector_info(base_symbol)
        market_sentiment = get_market_sentiment()
        news_sentiment = analyze_news_sentiment(base_symbol)
        fundamentals = fetch_fundamentals(formatted_symbol, api_key)

        # Analyze volume
        volume_data = volume_analysis(enhanced_data)

        # Find support/resistance
        support_resistance = find_support_resistance(enhanced_data)

        # Calculate enhanced signal strength with all factors
        signal_strength = calculate_signal_strength(
            latest, volume_data, support_resistance, fundamentals,
            market_sentiment, news_sentiment, sector_info, ml_prediction
        )

        # Calculate risk management
        risk_mgmt = calculate_risk_management(latest, support_resistance)

        # Get enhanced price information with real-time verification
        price_info = get_enhanced_price_info(base_symbol, enhanced_data)
        
        # Enhanced signal evaluation
        enhanced_signal = {
            'symbol': formatted_symbol,
            'date': latest['Date'].strftime('%Y-%m-%d'),
            'price': float(latest['Close']),
            'price_info': price_info,  # Enhanced price information with freshness indicator
            'signal_strength': signal_strength,
            'risk_management': risk_mgmt,
            'ml_prediction': ml_prediction,
            'technical_data': {
                'ma44': float(latest['MA44']),
                'bb_pctb': float(latest['BB_pctB']),
                'rsi': float(latest['RSI']),
                'atr': float(latest['ATR']),
                'macd_hist': float(latest.get('MACD_Hist', 0)),
                'stoch': float(latest.get('Stoch', 50)),
                'adx': float(latest.get('ADX', 0)),
                'volume_ratio': volume_data['volume_ratio'],
                'ma44_trend': latest.get('ma44_trend', 'unknown'),
                'ma44_slope': float(latest.get('MA44_slope10', 0)),
                'trend_ma44_up': bool(latest.get('trend_ma44_up', False)),
                'close_gt_ma44': bool(latest.get('close_gt_ma44', False))
            },
            'fundamentals': fundamentals,
            'support_resistance': support_resistance,
            'volume_analysis': volume_data,
            'sector_analysis': sector_info,
            'market_sentiment': market_sentiment,
            'news_sentiment': news_sentiment,
            'analysis_summary': {
                'total_factors_analyzed': len(signal_strength.get('factors', [])),
                'fundamental_source': fundamentals.get('source', 'Unknown'),
                'sector_identified': sector_info.get('sector', 'Unknown'),
                'market_phase': market_sentiment.get('market_phase', 'Unknown'),
                'sentiment_confidence': news_sentiment.get('confidence', 0),
                'enhanced_features': ['sector_analysis', 'market_sentiment', 'news_sentiment', 'enhanced_fundamentals', 'ml_prediction']
            }
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
            print(f"   Fundamentals: PE {result['fundamentals'].get('PE', 'N/A')}, EPS {result['fundamentals'].get('EPS', 'N/A')}")
            if 'ml_prediction' in result and result['ml_prediction']['status'] == 'success':
                print(f"   ML Prediction: {result['ml_prediction']['prediction']} (Prob: {result['ml_prediction']['probability']:.2f})")

        else:
            print(f"   Error: {result['error']}")
