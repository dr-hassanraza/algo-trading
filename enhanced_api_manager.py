#!/usr/bin/env python3
"""
Enhanced API Manager for Real-Time Pricing
==========================================

Comprehensive API management system with multiple data sources,
fallback mechanisms, rate limiting, and data validation.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import json
from pathlib import Path
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """Standardized price data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str
    quality_score: float = 1.0

@dataclass
class APIConfig:
    """API configuration structure"""
    name: str
    enabled: bool
    base_url: str
    timeout: int = 10
    retry_count: int = 3
    rate_limit_per_minute: int = 60
    requires_auth: bool = False
    api_key_env_var: Optional[str] = None

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls = max_calls_per_minute
        self.calls = []
        
    def can_make_call(self) -> bool:
        """Check if we can make an API call within rate limits"""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
    
    def wait_time(self) -> float:
        """Get time to wait before next call"""
        if not self.calls:
            return 0
        oldest_call = min(self.calls)
        return max(0, 60 - (time.time() - oldest_call))

class DataCache:
    """Simple in-memory cache for API data"""
    
    def __init__(self, default_ttl: int = 30):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def _get_key(self, symbol: str, source: str) -> str:
        """Generate cache key"""
        return f"{source}_{symbol}"
    
    def get(self, symbol: str, source: str) -> Optional[PriceData]:
        """Get cached data if not expired"""
        key = self._get_key(symbol, source)
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.default_ttl:
                logger.info(f"Cache hit for {symbol} from {source}")
                return data
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, symbol: str, source: str, data: PriceData):
        """Cache data with timestamp"""
        key = self._get_key(symbol, source)
        self.cache[key] = (data, time.time())
        logger.info(f"Cached data for {symbol} from {source}")
    
    def clear(self):
        """Clear all cached data"""
        self.cache.clear()

class EnhancedAPIManager:
    """Enhanced API Manager with multiple sources and fallback"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.apis = self._load_api_configs(config_file)
        self.rate_limiters = {name: RateLimiter(api.rate_limit_per_minute) 
                             for name, api in self.apis.items()}
        self.cache = DataCache()
        self.stats = {
            'api_calls': {},
            'cache_hits': 0,
            'failures': {},
            'last_success': {}
        }
        
        # Initialize stats
        for api_name in self.apis:
            self.stats['api_calls'][api_name] = 0
            self.stats['failures'][api_name] = 0
    
    def _load_api_configs(self, config_file: Optional[str]) -> Dict[str, APIConfig]:
        """Load API configurations"""
        
        # Default configurations
        default_configs = {
            'psx_dps': APIConfig(
                name='PSX DPS',
                enabled=True,
                base_url='https://dps.psx.com.pk',
                timeout=10,
                rate_limit_per_minute=60
            ),
            'psx_data_reader': APIConfig(
                name='PSX Data Reader',
                enabled=True,
                base_url='internal',
                timeout=5,
                rate_limit_per_minute=30
            ),
            'alpha_vantage': APIConfig(
                name='Alpha Vantage',
                enabled=bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
                base_url='https://www.alphavantage.co',
                timeout=15,
                rate_limit_per_minute=5,  # Free tier limit
                requires_auth=True,
                api_key_env_var='ALPHA_VANTAGE_API_KEY'
            ),
            'polygon': APIConfig(
                name='Polygon.io',
                enabled=bool(os.getenv('POLYGON_API_KEY')),
                base_url='https://api.polygon.io',
                timeout=10,
                rate_limit_per_minute=100,
                requires_auth=True,
                api_key_env_var='POLYGON_API_KEY'
            )
        }
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                # Update default configs with file configs
                # Implementation depends on your preferred format
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return default_configs
    
    def get_real_time_price(self, symbol: str) -> Optional[PriceData]:
        """Get real-time price with fallback mechanism"""
        
        # Try cache first
        for api_name in self.apis:
            cached_data = self.cache.get(symbol, api_name)
            if cached_data:
                self.stats['cache_hits'] += 1
                return cached_data
        
        # Try APIs in priority order
        api_priority = ['psx_dps', 'psx_data_reader', 'alpha_vantage', 'polygon']
        
        for api_name in api_priority:
            if api_name not in self.apis or not self.apis[api_name].enabled:
                continue
                
            try:
                data = self._fetch_from_api(symbol, api_name)
                if data and self._validate_data(data):
                    self.cache.set(symbol, api_name, data)
                    self.stats['last_success'][api_name] = datetime.now()
                    return data
            except Exception as e:
                logger.error(f"API {api_name} failed for {symbol}: {e}")
                self.stats['failures'][api_name] += 1
                continue
        
        logger.warning(f"All APIs failed for {symbol}, no data available")
        return None
    
    def _fetch_from_api(self, symbol: str, api_name: str) -> Optional[PriceData]:
        """Fetch data from specific API"""
        
        api_config = self.apis[api_name]
        rate_limiter = self.rate_limiters[api_name]
        
        # Check rate limiting
        if not rate_limiter.can_make_call():
            wait_time = rate_limiter.wait_time()
            logger.warning(f"Rate limit hit for {api_name}, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            if not rate_limiter.can_make_call():
                raise Exception(f"Rate limit exceeded for {api_name}")
        
        self.stats['api_calls'][api_name] += 1
        
        if api_name == 'psx_dps':
            return self._fetch_psx_dps(symbol, api_config)
        elif api_name == 'psx_data_reader':
            return self._fetch_psx_data_reader(symbol, api_config)
        elif api_name == 'alpha_vantage':
            return self._fetch_alpha_vantage(symbol, api_config)
        elif api_name == 'polygon':
            return self._fetch_polygon(symbol, api_config)
        else:
            raise Exception(f"Unknown API: {api_name}")
    
    def _fetch_psx_dps(self, symbol: str, config: APIConfig) -> Optional[PriceData]:
        """Fetch from PSX DPS API"""
        
        url = f"{config.base_url}/timeseries/int/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://dps.psx.com.pk/'
        }
        
        response = requests.get(url, headers=headers, timeout=config.timeout)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and data['data']:
                # PSX DPS returns tick data: [timestamp, price, volume]
                ticks = data['data']
                
                if not ticks:
                    return None
                
                # Calculate actual OHLC from tick data
                prices = [float(tick[1]) for tick in ticks]
                volumes = [int(tick[2]) for tick in ticks]
                
                # Most recent tick (first in array - they come newest first)
                latest_tick = ticks[0]
                current_price = float(latest_tick[1])
                current_timestamp = latest_tick[0]
                
                # Calculate today's OHLC from all ticks
                # Note: PSX DPS gives intraday ticks, so we use all available data
                high_price = max(prices)
                low_price = min(prices)
                total_volume = sum(volumes)
                
                # For open price, use the oldest tick (last in array) or current if only one tick
                if len(ticks) > 1:
                    open_price = float(ticks[-1][1])  # Oldest tick
                else:
                    open_price = current_price
                
                return PriceData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(current_timestamp),  # Most recent timestamp
                    open=open_price,
                    high=high_price,  # Actual high from tick data
                    low=low_price,    # Actual low from tick data
                    close=current_price,  # Most recent price
                    volume=total_volume,  # Sum of all tick volumes
                    source='PSX DPS (Real OHLC)',
                    quality_score=0.95  # Higher quality - real OHLC data
                )
        
        return None
    
    def _fetch_psx_data_reader(self, symbol: str, config: APIConfig) -> Optional[PriceData]:
        """Fetch from PSX Data Reader"""
        
        try:
            from psx_data_reader_fetcher import PSXDataFetcher
            
            fetcher = PSXDataFetcher()
            current_data = fetcher.fetch_current_data(symbol)
            
            if current_data:
                return PriceData(
                    symbol=symbol,
                    timestamp=current_data.timestamp,
                    open=current_data.open,
                    high=current_data.high,
                    low=current_data.low,
                    close=current_data.price,
                    volume=current_data.volume,
                    source='PSX Data Reader',
                    quality_score=0.8
                )
        except ImportError:
            logger.warning("PSX Data Reader not available")
        except Exception as e:
            logger.error(f"PSX Data Reader error: {e}")
        
        return None
    
    def _fetch_alpha_vantage(self, symbol: str, config: APIConfig) -> Optional[PriceData]:
        """Fetch from Alpha Vantage API"""
        
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            logger.warning("Alpha Vantage API key not found")
            return None
        
        # Alpha Vantage requires different symbol format for international stocks
        # This would need customization for PSX symbols
        url = f"{config.base_url}/query"
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': f'{symbol}.KAR',  # PSX extension
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=config.timeout)
        
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return PriceData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=float(quote['02. open']),
                    high=float(quote['03. high']),
                    low=float(quote['04. low']),
                    close=float(quote['05. price']),
                    volume=int(quote['06. volume']),
                    source='Alpha Vantage',
                    quality_score=0.95
                )
        
        return None
    
    def _fetch_polygon(self, symbol: str, config: APIConfig) -> Optional[PriceData]:
        """Fetch from Polygon.io API"""
        
        api_key = os.getenv(config.api_key_env_var)
        if not api_key:
            logger.warning("Polygon API key not found")
            return None
        
        # Polygon.io mainly covers US markets
        # This would need customization for PSX symbols
        url = f"{config.base_url}/v2/aggs/ticker/{symbol}/prev"
        params = {'apikey': api_key}
        
        response = requests.get(url, params=params, timeout=config.timeout)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                return PriceData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(result['t'] / 1000),
                    open=result['o'],
                    high=result['h'],
                    low=result['l'],
                    close=result['c'],
                    volume=result['v'],
                    source='Polygon.io',
                    quality_score=0.95
                )
        
        return None
    
    def _validate_data(self, data: PriceData) -> bool:
        """Validate price data quality"""
        
        checks = {
            'price_positive': data.close > 0,
            'high_ge_low': data.high >= data.low,
            'close_in_range': data.low <= data.close <= data.high,
            'volume_non_negative': data.volume >= 0,
            'recent_timestamp': data.timestamp > datetime.now() - timedelta(hours=1),
            'reasonable_price': 1 <= data.close <= 100000  # PKR range check
        }
        
        passed_checks = sum(checks.values())
        data.quality_score = passed_checks / len(checks)
        
        if data.quality_score < 0.7:
            logger.warning(f"Low quality data for {data.symbol}: {checks}")
        
        return data.quality_score >= 0.7
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical data with fallback"""
        
        for api_name in ['psx_data_reader', 'alpha_vantage']:
            if api_name not in self.apis or not self.apis[api_name].enabled:
                continue
            
            try:
                if api_name == 'psx_data_reader':
                    from psx_data_reader_fetcher import PSXDataFetcher
                    fetcher = PSXDataFetcher()
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=days)
                    
                    data = fetcher.fetch_historical_data(symbol, start_date, end_date)
                    if not data.empty:
                        return data
                        
            except Exception as e:
                logger.error(f"Historical data fetch failed for {api_name}: {e}")
                continue
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            'api_calls': self.stats['api_calls'],
            'cache_hits': self.stats['cache_hits'],
            'failures': self.stats['failures'],
            'last_success': {k: v.isoformat() if v else None 
                           for k, v in self.stats['last_success'].items()},
            'total_calls': sum(self.stats['api_calls'].values()),
            'success_rate': {api: 1 - (failures / max(1, self.stats['api_calls'][api]))
                           for api, failures in self.stats['failures'].items()}
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all APIs"""
        results = {}
        
        for api_name, api_config in self.apis.items():
            if not api_config.enabled:
                results[api_name] = False
                continue
                
            try:
                # Quick health check with a known symbol
                data = self._fetch_from_api('HBL', api_name)
                results[api_name] = data is not None
            except Exception:
                results[api_name] = False
        
        return results

# Singleton instance
api_manager = EnhancedAPIManager()

def get_real_time_price(symbol: str) -> Optional[PriceData]:
    """Convenience function to get real-time price"""
    return api_manager.get_real_time_price(symbol)

def get_api_stats() -> Dict[str, Any]:
    """Convenience function to get API statistics"""
    return api_manager.get_stats()

# Test function
def test_api_manager():
    """Test the API manager with sample symbols"""
    
    print("🧪 Testing Enhanced API Manager")
    print("=" * 50)
    
    test_symbols = ['HBL', 'UBL', 'FFC']
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}...")
        
        start_time = time.time()
        price_data = get_real_time_price(symbol)
        elapsed = time.time() - start_time
        
        if price_data:
            print(f"✅ Success: {price_data.source}")
            print(f"   Price: ₨{price_data.close:.2f}")
            print(f"   Quality: {price_data.quality_score:.1%}")
            print(f"   Latency: {elapsed:.2f}s")
        else:
            print(f"❌ Failed to get data")
    
    # Show statistics
    stats = get_api_stats()
    print(f"\n📈 API Statistics:")
    print(f"   Total calls: {stats['total_calls']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    
    for api, calls in stats['api_calls'].items():
        success_rate = stats['success_rate'][api]
        print(f"   {api}: {calls} calls, {success_rate:.1%} success")

if __name__ == "__main__":
    test_api_manager()