#!/usr/bin/env python3
"""
Advanced Technical Indicators
=============================

Professional-grade technical indicators including MACD, Stochastic, ADX,
candlestick patterns, and VWAP for enhanced trading signals.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List
from config_manager import get_config

logger = logging.getLogger(__name__)

def macd(close: pd.Series, fast: int = None, slow: int = None, signal: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    Returns: (macd_line, signal_line, histogram)
    """
    fast = fast or get_config('trading_parameters.macd_fast', 12)
    slow = slow or get_config('trading_parameters.macd_slow', 26)
    signal = signal or get_config('trading_parameters.macd_signal', 9)
    
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    logger.debug(f"MACD calculated with periods {fast}/{slow}/{signal}")
    return macd_line, signal_line, histogram


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
               k_period: int = None, d_period: int = None) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator
    Returns: (%K, %D)
    """
    k_period = k_period or get_config('trading_parameters.stoch_k', 14)
    d_period = d_period or get_config('trading_parameters.stoch_d', 3)
    
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    logger.debug(f"Stochastic calculated with periods K={k_period}, D={d_period}")
    return k_percent, d_percent


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX (Average Directional Index) for trend strength
    Returns: (adx, plus_di, minus_di)
    """
    period = period or get_config('trading_parameters.adx_period', 14)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    plus_dm = np.where((high - high.shift()) > (low.shift() - low), 
                       np.maximum(high - high.shift(), 0), 0)
    minus_dm = np.where((low.shift() - low) > (high - high.shift()), 
                        np.maximum(low.shift() - low, 0), 0)
    
    # Smooth the values
    tr_smooth = pd.Series(true_range).rolling(window=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
    
    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    logger.debug(f"ADX calculated with period {period}")
    return adx, plus_di, minus_di


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate VWAP (Volume Weighted Average Price)
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    logger.debug("VWAP calculated")
    return vwap


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    logger.debug(f"Williams %R calculated with period {period}")
    return wr


def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI)
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    logger.debug(f"CCI calculated with period {period}")
    return cci


def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect common candlestick patterns
    Returns dictionary of pattern names and whether they're present in the latest candle
    """
    if len(df) < 2:
        return {}
    
    current = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else current
    
    open_price = current['Open']
    close_price = current['Close']
    high_price = current['High']
    low_price = current['Low']
    
    prev_open = previous['Open']
    prev_close = previous['Close']
    prev_high = previous['High']
    prev_low = previous['Low']
    
    # Calculate body and shadow sizes
    body_size = abs(close_price - open_price)
    total_range = high_price - low_price
    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price
    
    prev_body_size = abs(prev_close - prev_open)
    
    # Pattern detection
    patterns = {}
    
    # Hammer
    patterns['hammer'] = (
        lower_shadow > 2 * body_size and
        upper_shadow < 0.1 * total_range and
        body_size > 0.1 * total_range
    )
    
    # Hanging Man (bearish hammer at top)
    patterns['hanging_man'] = patterns['hammer']  # Context dependent
    
    # Doji
    patterns['doji'] = body_size < 0.1 * total_range
    
    # Dragonfly Doji
    patterns['dragonfly_doji'] = (
        patterns['doji'] and
        lower_shadow > 2 * body_size and
        upper_shadow < 0.1 * total_range
    )
    
    # Gravestone Doji
    patterns['gravestone_doji'] = (
        patterns['doji'] and
        upper_shadow > 2 * body_size and
        lower_shadow < 0.1 * total_range
    )
    
    # Bullish Engulfing
    patterns['bullish_engulfing'] = (
        prev_close < prev_open and  # Previous red candle
        close_price > open_price and  # Current green candle
        open_price < prev_close and  # Opens below previous close
        close_price > prev_open and  # Closes above previous open
        body_size > prev_body_size  # Larger body
    )
    
    # Bearish Engulfing
    patterns['bearish_engulfing'] = (
        prev_close > prev_open and  # Previous green candle
        close_price < open_price and  # Current red candle
        open_price > prev_close and  # Opens above previous close
        close_price < prev_open and  # Closes below previous open
        body_size > prev_body_size  # Larger body
    )
    
    # Morning Star (simplified - needs 3 candles)
    if len(df) >= 3:
        prev2 = df.iloc[-3]
        prev2_close = prev2['Close']
        prev2_open = prev2['Open']
        
        patterns['morning_star'] = (
            prev2_close < prev2_open and  # First candle bearish
            abs(prev_close - prev_open) < 0.3 * abs(prev2_close - prev2_open) and  # Second small
            close_price > open_price and  # Third bullish
            close_price > (prev2_close + prev2_open) / 2  # Closes above midpoint of first
        )
        
        patterns['evening_star'] = (
            prev2_close > prev2_open and  # First candle bullish
            abs(prev_close - prev_open) < 0.3 * abs(prev2_close - prev2_open) and  # Second small
            close_price < open_price and  # Third bearish
            close_price < (prev2_close + prev2_open) / 2  # Closes below midpoint of first
        )
    
    # Shooting Star
    patterns['shooting_star'] = (
        upper_shadow > 2 * body_size and
        lower_shadow < 0.1 * total_range and
        body_size > 0.1 * total_range
    )
    
    # Inverted Hammer
    patterns['inverted_hammer'] = patterns['shooting_star']  # Context dependent
    
    # Count active patterns
    active_patterns = [name for name, active in patterns.items() if active]
    if active_patterns:
        logger.debug(f"Detected candlestick patterns: {active_patterns}")
    
    return patterns


def calculate_fibonacci_levels(df: pd.DataFrame, period: int = 50) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels based on recent high/low
    """
    if len(df) < period:
        period = len(df)
    
    recent_data = df.tail(period)
    high = recent_data['High'].max()
    low = recent_data['Low'].min()
    
    diff = high - low
    
    levels = {
        'high': high,
        'low': low,
        'fib_786': high - 0.786 * diff,
        'fib_618': high - 0.618 * diff,
        'fib_500': high - 0.500 * diff,
        'fib_382': high - 0.382 * diff,
        'fib_236': high - 0.236 * diff
    }
    
    logger.debug(f"Fibonacci levels calculated for {period} period range")
    return levels


def momentum_divergence(price: pd.Series, indicator: pd.Series, period: int = 10) -> Dict[str, bool]:
    """
    Detect bullish/bearish divergence between price and momentum indicators
    """
    if len(price) < period * 2:
        return {'bullish_divergence': False, 'bearish_divergence': False}
    
    # Find recent highs and lows
    price_highs = price.rolling(window=period, center=True).max() == price
    price_lows = price.rolling(window=period, center=True).min() == price
    
    indicator_highs = indicator.rolling(window=period, center=True).max() == indicator
    indicator_lows = indicator.rolling(window=period, center=True).min() == indicator
    
    # Get recent peaks
    recent_price_highs = price[price_highs].tail(2)
    recent_price_lows = price[price_lows].tail(2)
    recent_indicator_highs = indicator[indicator_highs].tail(2)
    recent_indicator_lows = indicator[indicator_lows].tail(2)
    
    bullish_divergence = False
    bearish_divergence = False
    
    # Bullish divergence: price makes lower low, indicator makes higher low
    if len(recent_price_lows) >= 2 and len(recent_indicator_lows) >= 2:
        if (recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2] and
            recent_indicator_lows.iloc[-1] > recent_indicator_lows.iloc[-2]):
            bullish_divergence = True
    
    # Bearish divergence: price makes higher high, indicator makes lower high
    if len(recent_price_highs) >= 2 and len(recent_indicator_highs) >= 2:
        if (recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2] and
            recent_indicator_highs.iloc[-1] < recent_indicator_highs.iloc[-2]):
            bearish_divergence = True
    
    if bullish_divergence or bearish_divergence:
        logger.debug(f"Divergence detected - Bullish: {bullish_divergence}, Bearish: {bearish_divergence}")
    
    return {
        'bullish_divergence': bullish_divergence,
        'bearish_divergence': bearish_divergence
    }


# Test function
if __name__ == '__main__':
    # Create sample data for testing
    import datetime as dt
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    price = 100
    data = []
    
    for date in dates:
        price += np.random.normal(0, 2)
        high = price + abs(np.random.normal(0, 1))
        low = price - abs(np.random.normal(0, 1))
        open_price = price + np.random.normal(0, 0.5)
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    
    print("🧪 Testing Advanced Indicators...")
    
    # Test MACD
    macd_line, signal_line, histogram = macd(df['Close'])
    print(f"MACD: {macd_line.iloc[-1]:.2f}, Signal: {signal_line.iloc[-1]:.2f}")
    
    # Test Stochastic
    k, d = stochastic(df['High'], df['Low'], df['Close'])
    print(f"Stochastic: %K={k.iloc[-1]:.2f}, %D={d.iloc[-1]:.2f}")
    
    # Test ADX
    adx_val, plus_di, minus_di = adx(df['High'], df['Low'], df['Close'])
    print(f"ADX: {adx_val.iloc[-1]:.2f}, +DI: {plus_di.iloc[-1]:.2f}, -DI: {minus_di.iloc[-1]:.2f}")
    
    # Test VWAP
    vwap_val = vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    print(f"VWAP: {vwap_val.iloc[-1]:.2f}")
    
    # Test candlestick patterns
    patterns = detect_candlestick_patterns(df)
    active_patterns = [name for name, active in patterns.items() if active]
    print(f"Active patterns: {active_patterns}")
    
    print("✅ Advanced indicators test complete!")