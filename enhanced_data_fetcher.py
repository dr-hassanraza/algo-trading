#!/usr/bin/env python3
"""
Enhanced Data Fetcher for PSX Trading System
===========================================

Integrates multiple data sources into the existing trading system:
1. EODHD API (primary)
2. yfinance (backup)
3. Multi-source fallback system
4. Enhanced error handling and caching

This replaces the basic EODHDFetcher with a more robust solution.
"""

import os
import pandas as pd
import datetime as dt
import time
import logging
from typing import Dict, List, Optional, Union
import requests
from dataclasses import dataclass

# Import our existing modules
from psx_bbands_candle_scanner import EODHDFetcher, TODAY
from multi_data_source_api import MultiSourceDataAPI, StockData

# Import PSX DPS official data fetcher (HIGHEST PRIORITY)
try:
    from psx_dps_fetcher import PSXDPSFetcher
    PSX_DPS_AVAILABLE = True
except ImportError:
    PSX_DPS_AVAILABLE = False
    logger.warning("PSX DPS Official fetcher not available")

# Import PSX data reader (backup)
try:
    from psx_data_reader_fetcher import PSXDataFetcher
    PSX_READER_AVAILABLE = True
except ImportError:
    PSX_READER_AVAILABLE = False
    logger.warning("PSX Data Reader not available")

# Import accurate PSX price fetcher
try:
    from psx_web_scraper import get_most_accurate_price
    ACCURATE_PRICE_AVAILABLE = True
except ImportError:
    ACCURATE_PRICE_AVAILABLE = False
    logger.warning("Accurate price fetcher not available")

logger = logging.getLogger(__name__)

class EnhancedDataFetcher:
    """Enhanced data fetcher with multiple sources and fallback mechanisms"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('EODHD_API_KEY')
        self.eodhd_fetcher = EODHDFetcher(self.api_key)
        self.multi_source_api = MultiSourceDataAPI()
        
        # Initialize PSX DPS official fetcher (HIGHEST PRIORITY)
        self.psx_dps_fetcher = None
        if PSX_DPS_AVAILABLE:
            try:
                self.psx_dps_fetcher = PSXDPSFetcher()
                logger.info("✅ PSX DPS Official API initialized (PRIMARY SOURCE)")
            except Exception as e:
                logger.error(f"Failed to initialize PSX DPS fetcher: {e}")
                self.psx_dps_fetcher = None # Ensure it's None on failure
        else:
            logger.warning("PSX DPS Official fetcher not available, will use fallback sources.")

        # Initialize PSX data reader if available (backup)
        self.psx_fetcher = None
        if PSX_READER_AVAILABLE:
            try:
                self.psx_fetcher = PSXDataFetcher()
                logger.info("✅ PSX Data Reader initialized (BACKUP SOURCE)")
            except Exception as e:
                logger.error(f"Failed to initialize PSX Data Reader: {e}")
        
        # Cache for recent data to avoid repeated API calls
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Success tracking for data source reliability
        self.source_success_rates = {
            'psx_dps_official': {'success': 0, 'total': 0},
            'eodhd_premium': {'success': 0, 'total': 0},
            'psx_reader': {'success': 0, 'total': 0},
            'eodhd': {'success': 0, 'total': 0},
            'yfinance': {'success': 0, 'total': 0},
            'multi_source': {'success': 0, 'total': 0}
        }
    
    def fetch(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """
        Enhanced fetch method with multiple data sources
        
        Args:
            symbol: Stock symbol (e.g., 'UBL.KAR')
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if self._is_cached(cache_key):
            logger.info(f"Using cached data for {symbol}")
            return self.cache[cache_key]['data']
        
        # Define the data source priority list.
        data_sources = []

        # 1. Primary Source: PSX DPS Official API
        if self.psx_dps_fetcher:
            data_sources.append(('psx_dps_official', self._fetch_psx_dps_official))

        # 2. Secondary Source: EODHD Premium (Purchased API)
        data_sources.append(('eodhd_premium', self._fetch_eodhd_premium))

        # 3. Backup Sources
        if self.psx_fetcher:
            data_sources.append(('psx_reader', self._fetch_psx_reader))
        
        data_sources.extend([
            ('eodhd', self._fetch_eodhd),
            ('yfinance', self._fetch_yfinance),
            ('multi_source', self._fetch_multi_source)
        ])
        
        for source_name, fetch_func in data_sources:
            try:
                logger.info(f"Trying to fetch {symbol} from {source_name}")
                data = fetch_func(symbol, start_date, end_date)
                
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched {symbol} from {source_name} ({len(data)} rows)")
                    
                    # Update success tracking
                    self._update_success_rate(source_name, True)
                    
                    # Cache the result
                    self._cache_data(cache_key, data)
                    
                    return data
                else:
                    self._update_success_rate(source_name, False)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from {source_name}: {e}")
                self._update_success_rate(source_name, False)
                continue
        
        # If all sources fail, try fallback sample data
        logger.warning(f"All data sources failed for {symbol}, using fallback sample data")
        
        try:
            from temp_data_fix import get_fallback_data
            fallback_data = get_fallback_data(symbol.split('.')[0], start_date, end_date)
            if not fallback_data.empty:
                logger.info(f"Using fallback sample data for {symbol} ({len(fallback_data)} rows)")
                return fallback_data
        except Exception as e:
            logger.error(f"Fallback data generation failed: {e}")
        
        # Final fallback - raise exception
        raise Exception(f"Failed to fetch data for {symbol} from all available sources including fallback")
    
    def _fetch_psx_dps_official(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Fetch data using PSX DPS Official API (HIGHEST PRIORITY)"""
        if not self.psx_dps_fetcher:
            return pd.DataFrame()
        
        try:
            return self.psx_dps_fetcher.fetch_historical_data(symbol, start_date, end_date)
        except Exception as e:
            logger.error(f"PSX DPS Official fetch error: {e}")
            return pd.DataFrame()
    
    def _fetch_eodhd_premium(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Fetch data using EODHD Premium API (purchased API - highest priority)"""
        try:
            from eodhd_premium_fetcher import EODHDPremiumFetcher
            premium_fetcher = EODHDPremiumFetcher()
            return premium_fetcher.get_historical_data(symbol, start_date, end_date)
        except ImportError:
            logger.warning("EODHD Premium fetcher not available")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"EODHD Premium fetch error: {e}")
            return pd.DataFrame()
    
    def _fetch_psx_reader(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Fetch data using PSX Data Reader (direct PSX access)"""
        if not self.psx_fetcher:
            return pd.DataFrame()
        
        return self.psx_fetcher.fetch_historical_data(symbol, start_date, end_date)
    
    def _fetch_eodhd(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Fetch data using EODHD API (existing method)"""
        return self.eodhd_fetcher.fetch(symbol, start_date, end_date)
    
    def _fetch_yfinance(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Fetch data using yfinance"""
        try:
            import yfinance as yf
            
            # Convert symbol format for yfinance
            base_symbol = symbol.split('.')[0]
            yf_symbols = [f"{base_symbol}.KAR", f"{base_symbol}.PSX", base_symbol]
            
            for yf_symbol in yf_symbols:
                try:
                    ticker = yf.Ticker(yf_symbol)
                    data = ticker.history(start=start_date, end=end_date + dt.timedelta(days=1))
                    
                    if not data.empty:
                        # Convert to our standard format
                        data = data.reset_index()
                        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        return data
                        
                except Exception as e:
                    continue
            
            return pd.DataFrame()
            
        except ImportError:
            logger.warning("yfinance not installed")
            return pd.DataFrame()
    
    def _fetch_multi_source(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Fetch data using multi-source API (for current/recent data only)"""
        
        # Multi-source API is best for current data, not historical
        # Only use if we need recent data (within last 30 days)
        if (dt.date.today() - end_date).days > 30:
            return pd.DataFrame()
        
        try:
            base_symbol = symbol.split('.')[0]
            stock_data = self.multi_source_api.get_stock_data(base_symbol)
            
            if stock_data:
                # Create a single-row DataFrame for current data
                current_data = pd.DataFrame([{
                    'Date': pd.Timestamp.now().date(),
                    'Open': stock_data.price,
                    'High': stock_data.price,
                    'Low': stock_data.price,
                    'Close': stock_data.price,
                    'Volume': stock_data.volume
                }])
                return current_data
            
        except Exception as e:
            logger.warning(f"Multi-source fetch error: {e}")
        
        return pd.DataFrame()
    
    def get_real_time_data(self, symbol: str) -> Optional[StockData]:
        """Get real-time stock data with PSX DPS as primary source"""
        
        # Try PSX DPS Official first (HIGHEST PRIORITY)
        if self.psx_dps_fetcher:
            try:
                psx_data = self.psx_dps_fetcher.fetch_real_time_data(symbol)
                if psx_data:
                    # Convert to StockData format
                    return StockData(
                        symbol=psx_data['symbol'],
                        price=psx_data['price'],
                        volume=psx_data['volume'],
                        change=0,  # PSX DPS doesn't provide change directly
                        change_percent=0.0,  # PSX DPS doesn't provide change percent directly
                        source=psx_data['source']
                    )
            except Exception as e:
                logger.warning(f"PSX DPS real-time fetch failed: {e}")
        
        # Fallback to multi-source API
        base_symbol = symbol.split('.')[0]
        return self.multi_source_api.get_stock_data(base_symbol)
    
    def get_multiple_real_time(self, symbols: List[str]) -> Dict[str, Optional[StockData]]:
        """Get real-time data for multiple symbols with PSX DPS priority"""
        results = {}
        
        # Try PSX DPS first for all symbols
        if self.psx_dps_fetcher:
            try:
                psx_results = self.psx_dps_fetcher.fetch_multiple_real_time(symbols)
                for symbol, data in psx_results.items():
                    if data:
                        results[symbol] = StockData(
                            symbol=data['symbol'],
                            price=data['price'],
                            volume=data['volume'],
                            change=0,
                            change_percent=0.0,
                            source=data['source']
                        )
                    else:
                        results[symbol] = None
            except Exception as e:
                logger.warning(f"PSX DPS multiple fetch failed: {e}")
        
        # For any symbols that failed, try multi-source API as fallback
        for symbol in symbols:
            if symbol not in results or results[symbol] is None:
                try:
                    base_symbol = symbol.split('.')[0]
                    fallback_data = self.multi_source_api.get_stock_data(base_symbol)
                    results[symbol] = fallback_data
                except Exception as e:
                    logger.warning(f"Multi-source fallback failed for {symbol}: {e}")
                    results[symbol] = None
        
        return results
    
    def get_psx_tickers(self) -> List[str]:
        """Get list of available PSX tickers using PSX Data Reader"""
        if self.psx_fetcher:
            return self.psx_fetcher.get_available_tickers()
        return []
    
    def search_psx_symbols(self, search_term: str) -> List[str]:
        """Search for PSX symbols matching a term"""
        if self.psx_fetcher:
            return self.psx_fetcher.search_symbol(search_term)
        return []
    
    def get_psx_current_data(self, symbol: str):
        """Get current PSX data using most accurate source available"""
        
        # Try PSX DPS Official first (HIGHEST PRIORITY)
        if self.psx_dps_fetcher:
            try:
                psx_dps_data = self.psx_dps_fetcher.fetch_real_time_data(symbol)
                if psx_dps_data and psx_dps_data.get('price'):
                    return psx_dps_data
            except Exception as e:
                logger.warning(f"PSX DPS current data fetch failed: {e}")
        
        # Try accurate price fetcher second
        if ACCURATE_PRICE_AVAILABLE:
            try:
                accurate_price = get_most_accurate_price(symbol)
                if accurate_price and accurate_price.get('price'):
                    return accurate_price
            except Exception as e:
                logger.warning(f"Accurate price fetch failed: {e}")
        
        # Fallback to PSX reader
        if self.psx_fetcher:
            return self.psx_fetcher.fetch_current_data(symbol)
        return None
    
    def get_verified_current_price(self, symbol: str) -> dict:
        """Get verified current price with accuracy indicators"""
        
        # Try PSX DPS Official first (HIGHEST ACCURACY)
        if self.psx_dps_fetcher:
            try:
                psx_data = self.psx_dps_fetcher.fetch_real_time_data(symbol)
                if psx_data and psx_data.get('price'):
                    return psx_data
            except Exception as e:
                logger.error(f"PSX DPS price verification failed: {e}")
        
        # Try accurate price fetcher as fallback
        if ACCURATE_PRICE_AVAILABLE:
            try:
                return get_most_accurate_price(symbol)
            except Exception as e:
                logger.error(f"Price verification failed: {e}")
        
        return {
            'price': None,
            'source': 'Not available',
            'confidence': 0,
            'is_current': False,
            'data_freshness': 'No accurate source available'
        }
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_expiry
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def _update_success_rate(self, source: str, success: bool):
        """Update success rate tracking for data sources"""
        if source in self.source_success_rates:
            self.source_success_rates[source]['total'] += 1
            if success:
                self.source_success_rates[source]['success'] += 1
    
    def get_source_reliability(self) -> Dict[str, float]:
        """Get reliability statistics for each data source"""
        reliability = {}
        for source, stats in self.source_success_rates.items():
            if stats['total'] > 0:
                reliability[source] = stats['success'] / stats['total']
            else:
                reliability[source] = 0.0
        return reliability
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache = {}
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        return {
            'cached_symbols': len(self.cache),
            'cache_size_mb': sum(len(str(data['data'])) for data in self.cache.values()) / 1024 / 1024,
            'oldest_cache': min([data['timestamp'] for data in self.cache.values()]) if self.cache else None
        }

# Integration function to replace EODHDFetcher in existing code
def create_enhanced_fetcher(api_key: str = None) -> EnhancedDataFetcher:
    """Create an enhanced data fetcher instance"""
    return EnhancedDataFetcher(api_key)

# Test the enhanced fetcher
def test_enhanced_fetcher():
    """Test the enhanced data fetcher"""
    
    print("🚀 Testing Enhanced Data Fetcher")
    print("=" * 50)
    
    fetcher = EnhancedDataFetcher()
    
    # Test historical data fetch
    test_symbols = ['UBL.KAR', 'MCB.KAR']
    end_date = TODAY
    start_date = end_date - dt.timedelta(days=30)
    
    for symbol in test_symbols:
        print(f"\n📊 Testing historical data for {symbol}...")
        try:
            data = fetcher.fetch(symbol, start_date, end_date)
            if not data.empty:
                print(f"   ✅ Got {len(data)} rows of historical data")
                print(f"   📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
                print(f"   💰 Latest close: {data['Close'].iloc[-1]:.2f}")
            else:
                print(f"   ❌ No historical data available")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test real-time data
    print(f"\n📈 Testing real-time data...")
    symbols = ['UBL', 'MCB', 'LUCK']
    real_time_data = fetcher.get_multiple_real_time([f"{s}.KAR" for s in symbols])
    
    for symbol, data in real_time_data.items():
        if data:
            print(f"   ✅ {symbol}: {data.price:.2f} PKR ({data.change:+.2f}) - {data.source}")
        else:
            print(f"   ❌ {symbol}: No real-time data")
    
    # Test PSX DPS specifically
    print(f"\n🏛️ Testing PSX DPS Official API:")
    psx_test_symbols = ['UBL', 'FFC', 'LUCK']
    for symbol in psx_test_symbols:
        current_data = fetcher.get_verified_current_price(symbol)
        if current_data and current_data.get('price'):
            print(f"   ✅ {symbol}: {current_data['price']:.2f} PKR - {current_data['source']}")
            print(f"      📊 Confidence: {current_data.get('confidence', 'N/A')}/10")
        else:
            print(f"   ❌ {symbol}: No PSX DPS data")
    
    # Show reliability stats
    print(f"\n📊 Data Source Reliability:")
    reliability = fetcher.get_source_reliability()
    for source, rate in reliability.items():
        print(f"   {source}: {rate:.1%} success rate")
    
    print(f"\n🏁 Enhanced fetcher test completed!")

if __name__ == "__main__":
    test_enhanced_fetcher()