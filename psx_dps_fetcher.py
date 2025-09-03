#!/usr/bin/env python3
"""
PSX DPS Official JSON API Data Fetcher
======================================

Direct integration with https://dps.psx.com.pk/timeseries/int/{SYMBOL}
This provides the most accurate and authoritative PSX stock data using
the official JSON API endpoint.

Features:
- Real-time stock prices via official JSON API
- Historical time series data  
- Trading volumes with timestamps
- Market status detection
- High-frequency tick data

API Format: [timestamp, price, volume]
Example: [1756878579, 385.0, 5000]
"""

import requests
import pandas as pd
import datetime as dt
from typing import Dict, List, Optional, Any
import logging
import time
import json

logger = logging.getLogger(__name__)

class PSXDPSFetcher:
    """Official PSX DPS JSON API fetcher for real-time and historical data"""
    
    def __init__(self):
        self.base_url = "https://dps.psx.com.pk/timeseries/int"
        self.session = requests.Session()
        
        # Set headers for JSON API requests
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://dps.psx.com.pk/',
            'Origin': 'https://dps.psx.com.pk'
        })
        
        # Cache for API responses
        self.cache = {}
        self.cache_expiry = 30  # 30 seconds for real-time data
        
        # Rate limiting
        self.last_request = 0
        self.min_delay = 0.2  # 200ms between requests
        
        # Common PSX symbols for validation
        self.known_symbols = [
            'UBL', 'MCB', 'HBL', 'ABL', 'BAFL', 'NBP', 'BOP', 'MEBL',
            'LUCK', 'DGKC', 'FCCL', 'CHCC', 'MLCF', 'PIOC', 'KOHC',
            'PPL', 'OGDC', 'POL', 'MARI', 'MPCL', 'PSO', 'SNGP', 'SSGC',
            'ENGRO', 'FFC', 'FATIMA', 'ICI', 'LOTTE', 'NESTLE', 'PTC',
            'KTML', 'KAPCO', 'HUBCO', 'KEL', 'PTCL', 'TRG', 'SYSTEMS',
            'ARPL', 'AIRLINK', 'WTL', 'TPL', 'UNITY', 'AVN', 'BIPL'
        ]
        
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_request = time.time()
    
    def fetch_real_time_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch real-time data for a stock symbol using PSX DPS JSON API
        
        Args:
            symbol: Stock symbol (e.g., 'UBL', 'MCB')
            
        Returns:
            Dictionary with current price, volume, and timestamp
        """
        clean_symbol = self._clean_symbol(symbol)
        
        # Check cache first
        cache_key = f"realtime_{clean_symbol}"
        if self._is_cached(cache_key):
            logger.debug(f"Using cached real-time data for {clean_symbol}")
            return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{clean_symbol}"
            logger.debug(f"Fetching real-time data from: {url}")
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 1 and data.get('data'):
                time_series = data['data']
                
                if time_series:
                    # Get the most recent data point
                    latest = time_series[0]  # First entry is most recent
                    
                    result = {
                        'symbol': clean_symbol,
                        'price': float(latest[1]),
                        'volume': int(latest[2]),
                        'timestamp': int(latest[0]),
                        'datetime': dt.datetime.fromtimestamp(latest[0]),
                        'currency': 'PKR',
                        'source': 'PSX DPS Official API',
                        'confidence': 10,  # Highest confidence - official API
                        'is_current': True,
                        'data_freshness': 'Real-time'
                    }
                    
                    # Cache the result
                    self._cache_data(cache_key, result)
                    
                    logger.info(f"✅ Real-time data for {clean_symbol}: {result['price']:.2f} PKR")
                    return result
                
            logger.warning(f"No data returned for {clean_symbol}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Network error fetching {clean_symbol}: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Data parsing error for {clean_symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {clean_symbol}: {e}")
            return None
    
    def fetch_intraday_ticks(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """
        Fetch complete intraday tick-by-tick data for high-frequency analysis
        
        Args:
            symbol: Stock symbol (e.g., 'FFC', 'UBL')
            limit: Maximum number of ticks to return (None = all available)
            
        Returns:
            DataFrame with columns: ['timestamp', 'price', 'volume', 'datetime_pkt']
            Sorted newest to oldest (as returned by PSX DPS)
        """
        clean_symbol = self._clean_symbol(symbol)
        
        # Use shorter cache for intraday data (10 seconds)
        cache_key = f"intraday_{clean_symbol}_{limit or 'all'}"
        if cache_key in self.cache:
            cache_time = self.cache[cache_key]['timestamp']
            if (time.time() - cache_time) < 10:  # 10 second cache
                logger.debug(f"Using cached intraday data for {clean_symbol}")
                return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{clean_symbol}"
            logger.debug(f"Fetching intraday ticks from: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 1 and data.get('data'):
                time_series = data['data']
                
                if not time_series:
                    return pd.DataFrame()
                
                # Limit data if requested
                if limit and len(time_series) > limit:
                    time_series = time_series[:limit]
                
                # Parse tick data
                tick_data = []
                for tick in time_series:
                    unix_timestamp, price, volume = tick
                    
                    # Convert to Pakistan time (UTC+5)
                    utc_time = dt.datetime.fromtimestamp(unix_timestamp, tz=dt.timezone.utc)
                    pkt_time = utc_time + dt.timedelta(hours=5)
                    
                    tick_data.append({
                        'timestamp': unix_timestamp,
                        'price': float(price),
                        'volume': int(volume),
                        'datetime_utc': utc_time,
                        'datetime_pkt': pkt_time.replace(tzinfo=None),  # Remove timezone for easier handling
                        'time_pkt': pkt_time.strftime('%H:%M:%S')
                    })
                
                df = pd.DataFrame(tick_data)
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': df,
                    'timestamp': time.time()
                }
                
                logger.info(f"✅ Intraday ticks for {clean_symbol}: {len(df)} trades")
                return df
                
            logger.warning(f"No intraday data returned for {clean_symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {clean_symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, symbol: str, start_date: dt.date = None, end_date: dt.date = None) -> pd.DataFrame:
        """
        Fetch historical data for a stock symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with historical OHLCV data
        """
        clean_symbol = self._clean_symbol(symbol)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = dt.date.today()
        if start_date is None:
            start_date = end_date - dt.timedelta(days=30)
        
        cache_key = f"historical_{clean_symbol}_{start_date}_{end_date}"
        if self._is_cached(cache_key):
            logger.debug(f"Using cached historical data for {clean_symbol}")
            return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{clean_symbol}"
            logger.debug(f"Fetching historical data from: {url}")
            
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 1 and data.get('data'):
                time_series = data['data']
                
                if not time_series:
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df_data = []
                for entry in time_series:
                    timestamp, price, volume = entry
                    dt_obj = dt.datetime.fromtimestamp(timestamp)
                    
                    # Filter by date range if specified
                    if start_date and dt_obj.date() < start_date:
                        continue
                    if end_date and dt_obj.date() > end_date:
                        continue
                    
                    df_data.append({
                        'Date': dt_obj,
                        'Price': float(price),
                        'Volume': int(volume),
                        'Timestamp': timestamp
                    })
                
                if not df_data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('Date').reset_index(drop=True)
                
                # Convert to OHLCV format by grouping by date
                ohlcv_df = self._convert_to_ohlcv(df)
                
                if not ohlcv_df.empty:
                    # Cache the result
                    self._cache_data(cache_key, ohlcv_df)
                    logger.info(f"✅ Historical data for {clean_symbol}: {len(ohlcv_df)} days")
                
                return ohlcv_df
                
            logger.warning(f"No historical data returned for {clean_symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {clean_symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_real_time(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch real-time data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with symbol as key and real-time data as value
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_real_time_data(symbol)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                results[symbol] = None
        
        return results
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get PSX market status"""
        now = dt.datetime.now()
        
        # PSX trading hours: 9:30 AM - 3:30 PM PKT (Monday-Friday)
        if now.weekday() >= 5:  # Weekend
            return {
                'status': 'closed',
                'message': 'PSX Closed (Weekend)',
                'next_open': 'Monday 9:30 AM PKT',
                'is_trading': False
            }
        
        if now.time() < dt.time(9, 30):
            return {
                'status': 'pre_market',
                'message': 'Pre-Market (Opens 9:30 AM PKT)',
                'next_event': 'Market opens at 9:30 AM',
                'is_trading': False
            }
        elif now.time() > dt.time(15, 30):
            return {
                'status': 'closed',
                'message': 'PSX Closed (Opens 9:30 AM tomorrow)',
                'next_event': 'Market opens tomorrow at 9:30 AM',
                'is_trading': False
            }
        else:
            return {
                'status': 'open',
                'message': 'PSX Open (Closes 3:30 PM PKT)',
                'next_event': 'Market closes at 3:30 PM',
                'is_trading': True
            }
    
    def search_symbols(self, query: str) -> List[str]:
        """Search for symbols matching the query"""
        query_upper = query.upper()
        matches = [symbol for symbol in self.known_symbols if query_upper in symbol]
        return matches[:10]  # Return top 10 matches
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is available"""
        clean_symbol = self._clean_symbol(symbol)
        
        # Quick validation against known symbols
        if clean_symbol in self.known_symbols:
            return True
        
        # Test with API call
        try:
            result = self.fetch_real_time_data(clean_symbol)
            return result is not None
        except:
            return False
    
    def _clean_symbol(self, symbol: str) -> str:
        """Clean and standardize symbol format"""
        return symbol.replace('.KAR', '').replace('.PSX', '').upper().strip()
    
    def _convert_to_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert tick data to OHLCV format by grouping by date"""
        if df.empty:
            return pd.DataFrame()
        
        # Group by date
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        ohlcv_data = []
        for date, group in df.groupby('Date'):
            group_sorted = group.sort_values('Timestamp')
            
            ohlcv_data.append({
                'Date': pd.Timestamp(date),
                'Open': group_sorted['Price'].iloc[0],
                'High': group_sorted['Price'].max(),
                'Low': group_sorted['Price'].min(),
                'Close': group_sorted['Price'].iloc[-1],
                'Volume': group_sorted['Volume'].sum()
            })
        
        ohlcv_df = pd.DataFrame(ohlcv_data)
        return ohlcv_df.sort_values('Date').reset_index(drop=True)
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_expiry
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def get_volume_profile(self, symbol: str, time_window_minutes: int = 30) -> Dict[str, Any]:
        """
        Analyze volume profile for the last N minutes
        
        Args:
            symbol: Stock symbol
            time_window_minutes: Time window for analysis
            
        Returns:
            Dictionary with volume profile analysis
        """
        try:
            df = self.fetch_intraday_ticks(symbol)
            
            if df.empty:
                return {}
            
            # Filter to time window
            cutoff_time = dt.datetime.now() - dt.timedelta(minutes=time_window_minutes)
            recent_df = df[df['datetime_pkt'] >= cutoff_time].copy()
            
            if recent_df.empty:
                return {}
            
            # Calculate volume profile
            total_volume = recent_df['volume'].sum()
            avg_price = (recent_df['price'] * recent_df['volume']).sum() / total_volume
            
            # Price levels with high volume (VWAP areas)
            price_min = recent_df['price'].min()
            price_max = recent_df['price'].max()
            price_range = price_max - price_min
            
            # Divide into price levels
            num_levels = min(10, len(recent_df))
            if num_levels < 2:
                num_levels = 2
                
            level_size = price_range / num_levels
            
            volume_by_level = {}
            for i in range(num_levels):
                level_min = price_min + (i * level_size)
                level_max = level_min + level_size
                
                level_trades = recent_df[
                    (recent_df['price'] >= level_min) & 
                    (recent_df['price'] < level_max)
                ]
                
                level_volume = level_trades['volume'].sum()
                level_price = (level_min + level_max) / 2
                
                if level_volume > 0:
                    volume_by_level[f"{level_price:.2f}"] = level_volume
            
            # Find high volume nodes (POC - Point of Control)
            poc_price = max(volume_by_level.keys(), key=lambda k: volume_by_level[k]) if volume_by_level else avg_price
            
            return {
                'total_trades': len(recent_df),
                'total_volume': total_volume,
                'vwap': avg_price,
                'price_range': price_range,
                'poc_price': float(poc_price) if isinstance(poc_price, str) else poc_price,
                'volume_by_level': volume_by_level,
                'high_volume_price': float(poc_price) if isinstance(poc_price, str) else poc_price,
                'time_window_minutes': time_window_minutes
            }
            
        except Exception as e:
            logger.error(f"Volume profile analysis error for {symbol}: {e}")
            return {}
    
    def get_price_momentum(self, symbol: str, lookback_minutes: int = 15) -> Dict[str, Any]:
        """
        Calculate price momentum indicators for intraday trading
        
        Args:
            symbol: Stock symbol
            lookback_minutes: Minutes to look back for momentum calculation
            
        Returns:
            Dictionary with momentum indicators
        """
        try:
            df = self.fetch_intraday_ticks(symbol, limit=100)  # Get recent trades
            
            if df.empty or len(df) < 2:
                return {}
            
            # Current vs historical comparison
            current_price = df.iloc[0]['price']
            
            # Filter to time window
            cutoff_time = dt.datetime.now() - dt.timedelta(minutes=lookback_minutes)
            recent_df = df[df['datetime_pkt'] >= cutoff_time].copy()
            
            if len(recent_df) < 2:
                return {}
            
            # Price momentum
            oldest_price = recent_df.iloc[-1]['price']  # Oldest in window
            price_change = current_price - oldest_price
            price_change_pct = (price_change / oldest_price) * 100
            
            # Volume momentum
            recent_volume = recent_df['volume'].sum()
            avg_trade_volume = recent_df['volume'].mean()
            
            # Trade frequency
            trade_frequency = len(recent_df) / lookback_minutes  # trades per minute
            
            # Price velocity (change per minute)
            time_diff = (recent_df.iloc[0]['datetime_pkt'] - recent_df.iloc[-1]['datetime_pkt']).total_seconds() / 60
            price_velocity = price_change / max(time_diff, 1)  # PKR per minute
            
            # Support/Resistance levels from recent trades
            support_level = recent_df['price'].min()
            resistance_level = recent_df['price'].max()
            
            # Momentum classification
            momentum_strength = "Strong" if abs(price_change_pct) > 2.0 else \
                               "Moderate" if abs(price_change_pct) > 0.5 else "Weak"
            
            momentum_direction = "Bullish" if price_change > 0 else "Bearish" if price_change < 0 else "Neutral"
            
            return {
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'momentum_strength': momentum_strength,
                'momentum_direction': momentum_direction,
                'recent_volume': recent_volume,
                'avg_trade_volume': avg_trade_volume,
                'trade_frequency': trade_frequency,
                'price_velocity': price_velocity,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'lookback_minutes': lookback_minutes,
                'trades_analyzed': len(recent_df)
            }
            
        except Exception as e:
            logger.error(f"Price momentum analysis error for {symbol}: {e}")
            return {}
    
    def get_liquidity_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze market liquidity based on recent tick data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with liquidity metrics
        """
        try:
            df = self.fetch_intraday_ticks(symbol, limit=50)  # Last 50 trades
            
            if df.empty:
                return {}
            
            # Time between trades (liquidity indicator)
            df_sorted = df.sort_values('timestamp')  # Oldest first for proper diff
            df_sorted['time_diff'] = df_sorted['timestamp'].diff()
            
            avg_time_between_trades = df_sorted['time_diff'].mean()
            
            # Volume distribution
            total_volume = df['volume'].sum()
            large_trades = df[df['volume'] >= df['volume'].quantile(0.8)]  # Top 20% by volume
            large_trade_volume = large_trades['volume'].sum()
            large_trade_pct = (large_trade_volume / total_volume) * 100
            
            # Price impact analysis
            price_std = df['price'].std()
            price_range = df['price'].max() - df['price'].min()
            
            # Liquidity score (inverse of time between trades, higher = more liquid)
            liquidity_score = 1 / max(avg_time_between_trades, 1) * 1000  # Normalize
            
            liquidity_level = "High" if liquidity_score > 10 else \
                             "Medium" if liquidity_score > 5 else "Low"
            
            return {
                'avg_time_between_trades': avg_time_between_trades,
                'total_trades': len(df),
                'total_volume': total_volume,
                'large_trade_percentage': large_trade_pct,
                'price_volatility': price_std,
                'price_range': price_range,
                'liquidity_score': liquidity_score,
                'liquidity_level': liquidity_level
            }
            
        except Exception as e:
            logger.error(f"Liquidity analysis error for {symbol}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache = {}

# Convenience functions for external use
def get_official_psx_price(symbol: str) -> Optional[Dict[str, Any]]:
    """Get official PSX price for a symbol"""
    fetcher = PSXDPSFetcher()
    return fetcher.fetch_real_time_data(symbol)

def get_official_psx_historical(symbol: str, days: int = 30) -> pd.DataFrame:
    """Get official PSX historical data"""
    fetcher = PSXDPSFetcher()
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days)
    return fetcher.fetch_historical_data(symbol, start_date, end_date)

# Test function
def test_psx_dps_fetcher():
    """Test the PSX DPS fetcher"""
    print("🚀 Testing PSX DPS Official JSON API Data Fetcher")
    print("=" * 50)
    
    fetcher = PSXDPSFetcher()
    
    # Test market status
    market_status = fetcher.get_market_status()
    print(f"📊 Market Status: {market_status['message']}")
    
    # Test real-time data
    test_symbols = ['UBL', 'MCB', 'LUCK', 'FFC']
    print(f"\n📈 Testing Real-time Data:")
    
    for symbol in test_symbols:
        try:
            data = fetcher.fetch_real_time_data(symbol)
            if data:
                print(f"   ✅ {symbol}: {data['price']:.2f} PKR (Vol: {data['volume']:,})")
                print(f"      📅 Updated: {data['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"   ❌ {symbol}: No data available")
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    # Test historical data
    print(f"\n📊 Testing Historical Data:")
    for symbol in ['UBL', 'MCB']:
        try:
            historical = fetcher.fetch_historical_data(symbol)
            if not historical.empty:
                print(f"   ✅ {symbol}: {len(historical)} days of historical data")
                print(f"      📅 Range: {historical['Date'].min().date()} to {historical['Date'].max().date()}")
                print(f"      💰 Latest close: {historical['Close'].iloc[-1]:.2f} PKR")
            else:
                print(f"   ❌ {symbol}: No historical data")
        except Exception as e:
            print(f"   ❌ {symbol}: Historical error - {e}")
    
    print(f"\n🏁 PSX DPS fetcher test completed!")

if __name__ == "__main__":
    test_psx_dps_fetcher()