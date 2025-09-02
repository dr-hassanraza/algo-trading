#!/usr/bin/env python3
"""
Financial Modeling Prep API Data Fetcher
========================================

Integration with Financial Modeling Prep for accurate PSX stock data
API Key: e4GjeSUUFPC0PAaHpo88f9GCLI91Cuil

Features:
- Real-time stock prices
- Historical data
- Company profiles
- Financial statements
- Market status
"""

import requests
import pandas as pd
import datetime as dt
from typing import Dict, List, Optional, Any
import logging
import time

logger = logging.getLogger(__name__)

class FMPDataFetcher:
    """Financial Modeling Prep API data fetcher"""
    
    def __init__(self, api_key: str = "e4GjeSUUFPC0PAaHpo88f9GCLI91Cuil"):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PSX-Trading-Bot/1.0'
        })
        
        # Cache for API responses
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Rate limiting
        self.last_request = 0
        self.rate_limit_delay = 0.2  # 200ms between requests
        
    def _make_request(self, endpoint: str, params: dict = None) -> Optional[Any]:
        """Make API request with rate limiting and error handling"""
        
        # Rate limiting
        time_since_last = time.time() - self.last_request
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            # Add API key to parameters
            if params is None:
                params = {}
            params['apikey'] = self.api_key
            
            url = f"{self.base_url}/{endpoint}"
            
            logger.info(f"FMP API request: {endpoint}")
            response = self.session.get(url, params=params, timeout=10)
            self.last_request = time.time()
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error messages
            if isinstance(data, dict) and 'Error Message' in data:
                logger.error(f"FMP API Error: {data['Error Message']}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FMP API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"FMP API error: {e}")
            return None
    
    def get_stock_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time stock quote"""
        
        # Try different symbol formats for PSX stocks
        psx_symbols = [
            f"{symbol.upper()}.KAR",  # Karachi format
            f"{symbol.upper()}.PSX",  # PSX format
            symbol.upper()
        ]
        
        for psx_symbol in psx_symbols:
            try:
                data = self._make_request(f"quote/{psx_symbol}")
                
                if data and isinstance(data, list) and len(data) > 0:
                    quote = data[0]
                    return {
                        'symbol': quote.get('symbol'),
                        'price': float(quote.get('price', 0)),
                        'change': float(quote.get('change', 0)),
                        'change_percent': float(quote.get('changesPercentage', 0)),
                        'volume': int(quote.get('volume', 0)),
                        'market_cap': quote.get('marketCap'),
                        'pe_ratio': quote.get('pe'),
                        'day_low': float(quote.get('dayLow', 0)),
                        'day_high': float(quote.get('dayHigh', 0)),
                        'year_low': float(quote.get('yearLow', 0)),
                        'year_high': float(quote.get('yearHigh', 0)),
                        'timestamp': dt.datetime.now(),
                        'source': 'Financial Modeling Prep',
                        'api_symbol': psx_symbol
                    }
                    
            except Exception as e:
                logger.debug(f"Quote request failed for {psx_symbol}: {e}")
                continue
        
        return None
    
    def get_historical_data(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Get historical stock data"""
        
        # Try different symbol formats
        psx_symbols = [
            f"{symbol.upper()}.KAR",
            f"{symbol.upper()}.PSX", 
            symbol.upper()
        ]
        
        for psx_symbol in psx_symbols:
            try:
                # Calculate date range
                days_diff = (end_date - start_date).days
                
                if days_diff <= 30:
                    # Use daily endpoint for short periods
                    endpoint = f"historical-price-full/{psx_symbol}"
                    params = {
                        'from': start_date.strftime('%Y-%m-%d'),
                        'to': end_date.strftime('%Y-%m-%d')
                    }
                else:
                    # Use historical daily endpoint for longer periods
                    endpoint = f"historical-price-full/{psx_symbol}"
                    params = {
                        'from': start_date.strftime('%Y-%m-%d'),
                        'to': end_date.strftime('%Y-%m-%d')
                    }
                
                data = self._make_request(endpoint, params)
                
                if data and isinstance(data, dict) and 'historical' in data:
                    historical = data['historical']
                    
                    if historical:
                        # Convert to DataFrame
                        df = pd.DataFrame(historical)
                        
                        # Standardize column names
                        column_mapping = {
                            'date': 'Date',
                            'open': 'Open',
                            'high': 'High', 
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        }
                        
                        df = df.rename(columns=column_mapping)
                        
                        # Ensure we have required columns
                        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        for col in required_columns:
                            if col not in df.columns:
                                if col == 'Volume':
                                    df[col] = 0
                                else:
                                    logger.warning(f"Missing column {col} for {psx_symbol}")
                        
                        # Convert Date to datetime and sort
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.sort_values('Date')
                        
                        # Filter date range
                        df = df[
                            (df['Date'].dt.date >= start_date) & 
                            (df['Date'].dt.date <= end_date)
                        ]
                        
                        if not df.empty:
                            logger.info(f"FMP: Got {len(df)} rows for {psx_symbol}")
                            return df[required_columns]
                        
            except Exception as e:
                logger.debug(f"Historical data request failed for {psx_symbol}: {e}")
                continue
        
        return pd.DataFrame()
    
    def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information"""
        
        psx_symbols = [f"{symbol.upper()}.KAR", f"{symbol.upper()}.PSX", symbol.upper()]
        
        for psx_symbol in psx_symbols:
            try:
                data = self._make_request(f"profile/{psx_symbol}")
                
                if data and isinstance(data, list) and len(data) > 0:
                    profile = data[0]
                    return {
                        'symbol': profile.get('symbol'),
                        'company_name': profile.get('companyName'),
                        'industry': profile.get('industry'),
                        'sector': profile.get('sector'),
                        'description': profile.get('description'),
                        'website': profile.get('website'),
                        'market_cap': profile.get('mktCap'),
                        'employees': profile.get('fullTimeEmployees'),
                        'country': profile.get('country'),
                        'currency': profile.get('currency')
                    }
                    
            except Exception as e:
                logger.debug(f"Profile request failed for {psx_symbol}: {e}")
                continue
        
        return None
    
    def search_symbols(self, query: str, limit: int = 20) -> List[str]:
        """Search for symbols matching query"""
        try:
            data = self._make_request("search", {'query': query, 'limit': limit})
            
            if data and isinstance(data, list):
                # Filter for PSX symbols
                psx_symbols = []
                for item in data:
                    symbol = item.get('symbol', '')
                    if '.KAR' in symbol or '.PSX' in symbol or item.get('exchangeShortName') == 'PSX':
                        psx_symbols.append(symbol)
                
                return psx_symbols[:limit]
                
        except Exception as e:
            logger.error(f"Symbol search failed: {e}")
        
        return []
    
    def get_market_status(self) -> Dict[str, str]:
        """Get current market status"""
        try:
            # FMP doesn't have direct PSX market status, so we'll use time-based logic
            now = dt.datetime.now()
            
            # PSX trading hours: 9:30 AM - 3:30 PM PKT (Monday-Friday)
            if now.weekday() >= 5:  # Weekend
                return {
                    'status': 'closed',
                    'message': 'Market Closed (Weekend)',
                    'next_open': 'Monday 9:30 AM'
                }
            
            # Assume Pakistan time (adjust as needed)
            if now.time() < dt.time(9, 30):
                return {
                    'status': 'pre_market',
                    'message': 'Pre-Market (Opens 9:30 AM)',
                    'next_event': 'Market opens at 9:30 AM'
                }
            elif now.time() > dt.time(15, 30):
                return {
                    'status': 'closed',
                    'message': 'Market Closed',
                    'next_event': 'Market opens tomorrow at 9:30 AM'
                }
            else:
                return {
                    'status': 'open',
                    'message': 'Market Open',
                    'next_event': 'Market closes at 3:30 PM'
                }
                
        except Exception as e:
            logger.error(f"Market status error: {e}")
            return {
                'status': 'unknown',
                'message': 'Market status unavailable',
                'next_event': 'Unknown'
            }
    
    def test_connection(self) -> bool:
        """Test API connection and key validity"""
        try:
            # Test with a simple quote request
            test_data = self._make_request("quote/AAPL")  # Test with Apple stock
            return test_data is not None
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

# Test function
def test_fmp_api():
    """Test the Financial Modeling Prep API"""
    print("🚀 Testing Financial Modeling Prep API")
    print("=" * 50)
    
    fetcher = FMPDataFetcher()
    
    # Test connection
    print("🔌 Testing API connection...")
    if fetcher.test_connection():
        print("✅ API connection successful")
    else:
        print("❌ API connection failed")
        return
    
    # Test stock quote for UBL
    print("\n💰 Testing UBL stock quote...")
    quote = fetcher.get_stock_quote('UBL')
    if quote:
        print(f"✅ UBL Quote:")
        print(f"   Symbol: {quote['symbol']}")
        print(f"   Price: {quote['price']:.2f} PKR")
        print(f"   Change: {quote['change']:+.2f} ({quote['change_percent']:+.2f}%)")
        print(f"   Volume: {quote['volume']:,}")
        print(f"   Source: {quote['source']}")
    else:
        print("❌ UBL quote failed")
    
    # Test historical data
    print(f"\n📊 Testing UBL historical data...")
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=30)
    
    historical = fetcher.get_historical_data('UBL', start_date, end_date)
    if not historical.empty:
        print(f"✅ Historical data: {len(historical)} rows")
        print(f"   Date range: {historical['Date'].min().date()} to {historical['Date'].max().date()}")
        print(f"   Latest close: {historical['Close'].iloc[-1]:.2f}")
    else:
        print("❌ Historical data failed")
    
    # Test company profile
    print(f"\n🏢 Testing UBL company profile...")
    profile = fetcher.get_company_profile('UBL')
    if profile:
        print(f"✅ Company: {profile.get('company_name', 'N/A')}")
        print(f"   Sector: {profile.get('sector', 'N/A')}")
        print(f"   Country: {profile.get('country', 'N/A')}")
    else:
        print("❌ Company profile failed")
    
    print("\n🏁 FMP API test completed!")

if __name__ == "__main__":
    test_fmp_api()