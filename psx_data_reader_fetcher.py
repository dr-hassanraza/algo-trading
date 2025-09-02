#!/usr/bin/env python3
"""
PSX Data Reader Integration for Trading System
==============================================

Direct integration with psx-data-reader library for live PSX data extraction.
This replaces API-based data fetching with direct PSX data access.

Features:
- Live data extraction from PSX
- Historical data retrieval 
- Ticker list management
- Enhanced error handling
- Cache integration
"""

import pandas as pd
import datetime as dt
import time
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

# Import psx-data-reader
try:
    from psx import stocks, tickers
    PSX_AVAILABLE = True
except ImportError:
    PSX_AVAILABLE = False
    print("Warning: psx-data-reader not installed. Run: pip install psx-data-reader")

logger = logging.getLogger(__name__)

@dataclass
class PSXStockData:
    """Data structure for PSX stock information"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    timestamp: dt.datetime
    source: str = "PSX"

class PSXDataFetcher:
    """PSX Data Reader based fetcher for live PSX data"""
    
    def __init__(self):
        if not PSX_AVAILABLE:
            raise ImportError("psx-data-reader library not available. Install with: pip install psx-data-reader")
        
        # Cache for recent data to avoid repeated calls
        self.cache = {}
        self.cache_expiry = 60  # 1 minute for live data
        self.ticker_cache = None
        self.ticker_cache_expiry = 3600  # 1 hour for ticker list
        
        # Get available tickers
        self._refresh_tickers()
    
    def _refresh_tickers(self):
        """Refresh the list of available tickers"""
        try:
            ticker_data = tickers()
            if ticker_data is not None:
                # Convert DataFrame to list if needed
                if hasattr(ticker_data, 'tolist'):
                    self.available_tickers = ticker_data.tolist()
                elif hasattr(ticker_data, 'values'):
                    self.available_tickers = ticker_data.values.flatten().tolist()
                elif isinstance(ticker_data, list):
                    self.available_tickers = ticker_data
                else:
                    # Try to extract ticker symbols from DataFrame columns/index
                    if hasattr(ticker_data, 'index'):
                        self.available_tickers = ticker_data.index.tolist()
                    elif hasattr(ticker_data, 'columns'):
                        self.available_tickers = ticker_data.columns.tolist()
                    else:
                        self.available_tickers = []
            else:
                self.available_tickers = []
            
            self.ticker_cache = time.time()
            logger.info(f"Loaded {len(self.available_tickers)} PSX tickers")
        except Exception as e:
            logger.error(f"Failed to load PSX tickers: {e}")
            self.available_tickers = []
    
    def get_available_tickers(self) -> List[str]:
        """Get list of all available PSX tickers"""
        # Refresh ticker cache if expired
        if (self.ticker_cache is None or 
            time.time() - self.ticker_cache > self.ticker_cache_expiry):
            self._refresh_tickers()
        
        return self.available_tickers if self.available_tickers else []
    
    def fetch_historical_data(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """
        Fetch historical data using psx-data-reader
        
        Args:
            symbol: Stock symbol (e.g., 'SILK', 'UBL')
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        
        # Clean symbol (remove .KAR suffix if present)
        clean_symbol = symbol.replace('.KAR', '').replace('.PSX', '').upper()
        
        # Check cache first
        cache_key = f"{clean_symbol}_{start_date}_{end_date}"
        if self._is_cached(cache_key):
            logger.info(f"Using cached data for {clean_symbol}")
            return self.cache[cache_key]['data']
        
        try:
            logger.info(f"Fetching historical data for {clean_symbol} from {start_date} to {end_date}")
            
            # Use psx-data-reader to get historical data
            data = stocks(clean_symbol, start=start_date, end=end_date)
            
            if data is not None and not data.empty:
                logger.info(f"Raw data columns: {list(data.columns)}")
                logger.info(f"Raw data shape: {data.shape}")
                
                # Ensure proper column formatting
                if 'Date' not in data.columns:
                    if data.index.name in ['Date', 'date', 'TIME']:
                        data = data.reset_index()
                    elif 'TIME' in data.columns:
                        data = data.rename(columns={'TIME': 'Date'})
                
                # Standardize column names - be more flexible with mapping
                column_mapping = {}
                for col in data.columns:
                    col_lower = col.lower()
                    if col_lower in ['date', 'time']:
                        column_mapping[col] = 'Date'
                    elif col_lower in ['open', 'o']:
                        column_mapping[col] = 'Open'
                    elif col_lower in ['high', 'h']:
                        column_mapping[col] = 'High'
                    elif col_lower in ['low', 'l']:
                        column_mapping[col] = 'Low'
                    elif col_lower in ['close', 'c']:
                        column_mapping[col] = 'Close'
                    elif col_lower in ['volume', 'v', 'vol']:
                        column_mapping[col] = 'Volume'
                
                # Apply column mapping
                if column_mapping:
                    data = data.rename(columns=column_mapping)
                
                # Ensure we have the required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in data.columns:
                        if col == 'Volume':
                            data[col] = 0  # Default volume if not available
                        else:
                            logger.warning(f"Missing required column: {col}")
                
                # Select only required columns in correct order
                data = data[required_columns]
                
                # Convert Date to datetime if it's not already
                if data['Date'].dtype == 'object':
                    data['Date'] = pd.to_datetime(data['Date'])
                
                # Sort by date
                data = data.sort_values('Date')
                
                logger.info(f"Successfully fetched {len(data)} rows for {clean_symbol}")
                
                # Cache the result
                self._cache_data(cache_key, data)
                
                return data
            else:
                logger.warning(f"No data returned for {clean_symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {clean_symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_current_data(self, symbol: str) -> Optional[PSXStockData]:
        """
        Fetch current/latest data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'SILK', 'UBL')
            
        Returns:
            PSXStockData object with current information
        """
        
        clean_symbol = symbol.replace('.KAR', '').replace('.PSX', '').upper()
        
        # For current data, we can fetch today's data and use the latest record
        try:
            today = dt.date.today()
            yesterday = today - dt.timedelta(days=1)
            
            # Get recent data (last 2 days to ensure we get latest)
            recent_data = self.fetch_historical_data(clean_symbol, yesterday, today)
            
            if not recent_data.empty:
                latest = recent_data.iloc[-1]
                
                # Calculate change (requires previous day data)
                change = 0.0
                change_percent = 0.0
                
                if len(recent_data) > 1:
                    prev_close = recent_data.iloc[-2]['Close']
                    change = latest['Close'] - prev_close
                    change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
                
                return PSXStockData(
                    symbol=clean_symbol,
                    price=latest['Close'],
                    change=change,
                    change_percent=change_percent,
                    volume=int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
                    high=latest['High'],
                    low=latest['Low'],
                    open=latest['Open'],
                    timestamp=pd.Timestamp.now(),
                    source="PSX"
                )
            
        except Exception as e:
            logger.error(f"Error fetching current data for {clean_symbol}: {e}")
        
        return None
    
    def fetch_multiple_current(self, symbols: List[str]) -> Dict[str, Optional[PSXStockData]]:
        """
        Fetch current data for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their current data
        """
        result = {}
        
        for symbol in symbols:
            result[symbol] = self.fetch_current_data(symbol)
            # Small delay to avoid overwhelming the data source
            time.sleep(0.1)
        
        return result
    
    def search_symbol(self, search_term: str) -> List[str]:
        """
        Search for symbols matching a term
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of matching symbols
        """
        available = self.get_available_tickers()
        search_term = search_term.upper()
        
        matches = [ticker for ticker in available if search_term in ticker.upper()]
        return matches[:20]  # Limit to 20 matches
    
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
        
        # Clean old cache entries (keep only last 50)
        if len(self.cache) > 50:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache = {}
    
    def get_cache_info(self) -> Dict:
        """Get information about cached data"""
        return {
            'cached_items': len(self.cache),
            'cache_size_mb': sum(len(str(data['data'])) for data in self.cache.values()) / 1024 / 1024,
            'oldest_cache': min([data['timestamp'] for data in self.cache.values()]) if self.cache else None,
            'available_tickers': len(self.available_tickers) if self.available_tickers else 0
        }

# Compatibility functions for existing code
def fetch_psx_data(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Compatibility function for existing code"""
    fetcher = PSXDataFetcher()
    return fetcher.fetch_historical_data(symbol, start_date, end_date)

def get_psx_tickers() -> List[str]:
    """Get list of available PSX tickers"""
    fetcher = PSXDataFetcher()
    return fetcher.get_available_tickers()

# Test function
def test_psx_fetcher():
    """Test the PSX data fetcher"""
    
    print("🚀 Testing PSX Data Reader Fetcher")
    print("=" * 50)
    
    if not PSX_AVAILABLE:
        print("❌ PSX Data Reader not available")
        return
    
    fetcher = PSXDataFetcher()
    
    # Test ticker list
    print(f"\n📋 Available tickers: {len(fetcher.get_available_tickers())}")
    
    # Test search
    search_results = fetcher.search_symbol("BANK")
    print(f"🔍 Search for 'BANK': {search_results[:5]}")  # Show first 5
    
    # Test historical data
    test_symbols = ['SILK', 'UBL', 'MCB']
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=30)
    
    for symbol in test_symbols:
        print(f"\n📊 Testing historical data for {symbol}...")
        try:
            data = fetcher.fetch_historical_data(symbol, start_date, end_date)
            if not data.empty:
                print(f"   ✅ Got {len(data)} rows of data")
                print(f"   📅 Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
                print(f"   💰 Latest close: {data['Close'].iloc[-1]:.2f}")
            else:
                print(f"   ❌ No data available")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test current data
    print(f"\n📈 Testing current data...")
    for symbol in test_symbols[:2]:  # Test first 2 symbols
        current = fetcher.fetch_current_data(symbol)
        if current:
            print(f"   ✅ {current.symbol}: {current.price:.2f} PKR ({current.change:+.2f}, {current.change_percent:+.1f}%)")
        else:
            print(f"   ❌ {symbol}: No current data")
    
    # Show cache info
    cache_info = fetcher.get_cache_info()
    print(f"\n💾 Cache info: {cache_info['cached_items']} items, {cache_info['cache_size_mb']:.2f} MB")
    
    print(f"\n🏁 PSX fetcher test completed!")

if __name__ == "__main__":
    test_psx_fetcher()