#!/usr/bin/env python3
"""
EODHD Premium API Data Fetcher
==============================

Integration with EODHD premium API service for accurate financial data.
API Key: 68a350864b1140.05317137

Features:
- Real-time stock prices
- Historical data with extended coverage
- Fundamental data
- Technical indicators
- Exchange rates
- Market news and sentiment
- International markets including PSX
"""

import requests
import pandas as pd
import datetime as dt
from typing import Dict, List, Optional, Any, Union
import logging
import time
import json

logger = logging.getLogger(__name__)

class EODHDPremiumFetcher:
    """EODHD Premium API data fetcher"""
    
    def __init__(self, api_key: str = "68a350864b1140.05317137"):
        self.api_key = api_key
        self.base_url = "https://eodhd.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Professional-Trading-Bot/2.0'
        })
        
        # Cache for API responses
        self.cache = {}
        self.cache_expiry = 60  # 1 minute for real-time data
        
        # Rate limiting for premium API
        self.last_request = 0
        self.rate_limit_delay = 0.1  # 100ms between requests (premium allows faster)
        
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request = time.time()
    
    def _make_request(self, endpoint: str, params: dict = None) -> Optional[Any]:
        """Make API request with enhanced error handling"""
        
        self._rate_limit()
        
        try:
            # Add API key to parameters
            if params is None:
                params = {}
            params['api_token'] = self.api_key
            params['fmt'] = 'json'
            
            url = f"{self.base_url}/{endpoint}"
            
            logger.info(f"EODHD Premium API request: {endpoint}")
            response = self.session.get(url, params=params, timeout=15)
            
            # Handle different HTTP status codes
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 402:
                logger.error("EODHD API: Payment required - check subscription status")
                return None
            elif response.status_code == 403:
                logger.error("EODHD API: Access forbidden - check API key permissions")
                return None
            elif response.status_code == 429:
                logger.warning("EODHD API: Rate limit exceeded - waiting longer")
                time.sleep(1)  # Wait before retry
                return self._make_request(endpoint, params)
            else:
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"EODHD API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"EODHD API JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"EODHD API unexpected error: {e}")
            return None
    
    def get_real_time_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time stock quote"""
        
        # Try different symbol formats for PSX stocks
        psx_formats = [
            f"{symbol.upper()}.KAR",  # Karachi format
            f"{symbol.upper()}.PSX",  # PSX format
            f"{symbol.upper()}.XKAR", # Extended Karachi
            symbol.upper()
        ]
        
        for psx_symbol in psx_formats:
            try:
                data = self._make_request(f"real-time/{psx_symbol}")
                
                if data and isinstance(data, dict):
                    # Check if we got valid price data
                    if 'close' in data and data['close'] not in [None, 0, '0']:
                        return {
                            'symbol': data.get('code', psx_symbol),
                            'price': float(data.get('close', 0)),
                            'open': float(data.get('open', 0)),
                            'high': float(data.get('high', 0)),
                            'low': float(data.get('low', 0)),
                            'volume': int(data.get('volume', 0)),
                            'change': float(data.get('change', 0)),
                            'change_percent': float(data.get('change_p', 0)),
                            'previous_close': float(data.get('previousClose', 0)),
                            'timestamp': data.get('timestamp'),
                            'source': 'EODHD Premium Real-time',
                            'api_symbol': psx_symbol,
                            'currency': 'PKR'
                        }
                    
            except Exception as e:
                logger.debug(f"Real-time quote failed for {psx_symbol}: {e}")
                continue
        
        return None
    
    def get_historical_data(self, symbol: str, start_date: dt.date, end_date: dt.date, 
                          interval: str = 'd') -> pd.DataFrame:
        """Get historical stock data with premium features"""
        
        psx_formats = [
            f"{symbol.upper()}.KAR",
            f"{symbol.upper()}.PSX", 
            f"{symbol.upper()}.XKAR",
            symbol.upper()
        ]
        
        for psx_symbol in psx_formats:
            try:
                params = {
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d'),
                    'period': interval  # d, w, m for daily, weekly, monthly
                }
                
                data = self._make_request(f"eod/{psx_symbol}", params)
                
                if data and isinstance(data, list) and len(data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Standardize column names
                    column_mapping = {
                        'date': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low', 
                        'close': 'Close',
                        'adjusted_close': 'Adj Close',
                        'volume': 'Volume'
                    }
                    
                    df = df.rename(columns=column_mapping)
                    
                    # Ensure required columns exist
                    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        if col not in df.columns:
                            if col == 'Volume':
                                df[col] = 0
                            elif col == 'Date':
                                continue
                            else:
                                logger.warning(f"Missing column {col} for {psx_symbol}")
                    
                    # Convert Date to datetime
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.sort_values('Date')
                        
                        # Filter date range
                        df = df[
                            (df['Date'].dt.date >= start_date) & 
                            (df['Date'].dt.date <= end_date)
                        ]
                        
                        if not df.empty:
                            logger.info(f"EODHD Premium: Got {len(df)} rows for {psx_symbol}")
                            return df[required_columns]
                    
            except Exception as e:
                logger.debug(f"Historical data failed for {psx_symbol}: {e}")
                continue
        
        return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get fundamental data (earnings, ratios, financials)"""
        
        psx_formats = [f"{symbol.upper()}.KAR", f"{symbol.upper()}.PSX", symbol.upper()]
        
        for psx_symbol in psx_formats:
            try:
                data = self._make_request(f"fundamentals/{psx_symbol}")
                
                if data and isinstance(data, dict):
                    # Extract key fundamental metrics
                    general = data.get('General', {})
                    highlights = data.get('Highlights', {})
                    valuation = data.get('Valuation', {})
                    
                    return {
                        'symbol': general.get('Code', psx_symbol),
                        'company_name': general.get('Name'),
                        'sector': general.get('Sector'),
                        'industry': general.get('Industry'),
                        'market_cap': highlights.get('MarketCapitalization'),
                        'pe_ratio': highlights.get('PERatio'),
                        'eps': highlights.get('EarningsShare'),
                        'dividend_yield': highlights.get('DividendYield'),
                        'book_value': highlights.get('BookValue'),
                        'price_to_book': valuation.get('PriceBookMRQ'),
                        'return_on_equity': valuation.get('ReturnOnEquityTTM'),
                        'debt_to_equity': valuation.get('DebtToEquityMRQ'),
                        'currency': general.get('CurrencyCode', 'PKR'),
                        'exchange': general.get('Exchange'),
                        'country': general.get('Country')
                    }
                    
            except Exception as e:
                logger.debug(f"Fundamental data failed for {psx_symbol}: {e}")
                continue
        
        return None
    
    def get_technical_indicators(self, symbol: str, indicator: str, 
                               period: int = 14, start_date: dt.date = None, 
                               end_date: dt.date = None) -> pd.DataFrame:
        """Get technical indicators (RSI, MACD, SMA, etc.)"""
        
        if start_date is None:
            end_date = dt.date.today()
            start_date = end_date - dt.timedelta(days=365)
        if end_date is None:
            end_date = dt.date.today()
        
        psx_formats = [f"{symbol.upper()}.KAR", f"{symbol.upper()}.PSX", symbol.upper()]
        
        for psx_symbol in psx_formats:
            try:
                params = {
                    'function': indicator.upper(),  # RSI, MACD, SMA, EMA, etc.
                    'period': period,
                    'from': start_date.strftime('%Y-%m-%d'),
                    'to': end_date.strftime('%Y-%m-%d')
                }
                
                data = self._make_request(f"technical/{psx_symbol}", params)
                
                if data and isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    return df
                    
            except Exception as e:
                logger.debug(f"Technical indicators failed for {psx_symbol}: {e}")
                continue
        
        return pd.DataFrame()
    
    def get_exchange_symbols(self, exchange: str = 'KAR') -> List[str]:
        """Get all available symbols for an exchange"""
        
        try:
            data = self._make_request(f"exchange-symbol-list/{exchange}")
            
            if data and isinstance(data, list):
                symbols = [item.get('Code', '') for item in data if item.get('Code')]
                return [s for s in symbols if s]  # Remove empty strings
            
        except Exception as e:
            logger.error(f"Exchange symbols fetch failed: {e}")
        
        return []
    
    def get_market_news(self, symbols: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get market news and sentiment"""
        
        try:
            params = {'limit': limit}
            
            if symbols:
                # Convert symbols to EODHD format
                formatted_symbols = []
                for symbol in symbols:
                    if not any(x in symbol for x in ['.KAR', '.PSX']):
                        symbol = f"{symbol}.KAR"
                    formatted_symbols.append(symbol)
                params['s'] = ','.join(formatted_symbols)
            
            data = self._make_request("news", params)
            
            if data and isinstance(data, list):
                return [
                    {
                        'title': item.get('title'),
                        'content': item.get('content'),
                        'date': item.get('date'),
                        'symbols': item.get('symbols', []),
                        'sentiment': item.get('sentiment'),
                        'link': item.get('link')
                    }
                    for item in data
                ]
            
        except Exception as e:
            logger.error(f"Market news fetch failed: {e}")
        
        return []
    
    def get_forex_rates(self, base: str = 'USD', target: str = 'PKR') -> Optional[Dict[str, Any]]:
        """Get forex exchange rates"""
        
        try:
            forex_pair = f"{base}{target}"
            data = self._make_request(f"real-time/{forex_pair}.FOREX")
            
            if data and isinstance(data, dict):
                return {
                    'pair': forex_pair,
                    'rate': float(data.get('close', 0)),
                    'change': float(data.get('change', 0)),
                    'change_percent': float(data.get('change_p', 0)),
                    'timestamp': data.get('timestamp'),
                    'source': 'EODHD Premium Forex'
                }
                
        except Exception as e:
            logger.error(f"Forex rates fetch failed: {e}")
        
        return None
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test API connection and subscription status"""
        
        try:
            # Test with a simple quote request
            test_data = self._make_request("real-time/AAPL.US")
            
            if test_data:
                return {
                    'status': 'success',
                    'message': 'EODHD Premium API connected successfully',
                    'subscription': 'Active',
                    'test_symbol': 'AAPL.US',
                    'test_price': test_data.get('close')
                }
            else:
                return {
                    'status': 'failed',
                    'message': 'EODHD API connection failed',
                    'subscription': 'Unknown'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'EODHD API test failed: {e}',
                'subscription': 'Unknown'
            }

# Integration functions
def get_eodhd_premium_price(symbol: str) -> Optional[Dict[str, Any]]:
    """Get premium EODHD price with enhanced accuracy"""
    
    fetcher = EODHDPremiumFetcher()
    
    print(f"💎 Fetching EODHD Premium price for {symbol}...")
    
    try:
        # Try real-time quote first
        result = fetcher.get_real_time_quote(symbol)
        
        if result and result.get('price', 0) > 0:
            print(f"✅ EODHD Premium: {result['price']:.2f} PKR")
            print(f"📊 Change: {result.get('change', 0):+.2f} ({result.get('change_percent', 0):+.2f}%)")
            
            return {
                'price': result['price'],
                'currency': result['currency'],
                'source': result['source'],
                'confidence': 8,  # High confidence for premium API
                'is_current': True,
                'data_freshness': 'EODHD Premium real-time',
                'timestamp': dt.datetime.now(),
                'change': result.get('change'),
                'change_percent': result.get('change_percent'),
                'volume': result.get('volume'),
                'open': result.get('open'),
                'high': result.get('high'),
                'low': result.get('low')
            }
        else:
            print(f"❌ EODHD Premium: No real-time data for {symbol}")
            return None
            
    except Exception as e:
        logger.error(f"EODHD Premium fetch failed: {e}")
        print(f"❌ EODHD Premium error: {e}")
        return None

# Test function
def test_eodhd_premium():
    """Test EODHD Premium API functionality"""
    print("🚀 Testing EODHD Premium API")
    print("=" * 50)
    
    fetcher = EODHDPremiumFetcher()
    
    # Test API connection
    print("🔌 Testing API connection...")
    connection_test = fetcher.test_api_connection()
    print(f"Status: {connection_test['status']}")
    print(f"Message: {connection_test['message']}")
    
    if connection_test['status'] != 'success':
        print("❌ API connection failed - check subscription")
        return
    
    # Test PSX stock quotes
    test_symbols = ['UBL', 'MCB', 'ABL']
    
    for symbol in test_symbols:
        print(f"\n💰 Testing {symbol} premium quote...")
        result = get_eodhd_premium_price(symbol)
        
        if result:
            print(f"   ✅ Price: {result['price']:.2f} PKR")
            print(f"   📊 Volume: {result.get('volume', 0):,}")
            print(f"   🎯 Confidence: {result['confidence']}/10")
        else:
            print(f"   ❌ No data available")
    
    # Test fundamental data
    print(f"\n🏢 Testing fundamental data for UBL...")
    fundamentals = fetcher.get_fundamental_data('UBL')
    if fundamentals:
        print(f"   Company: {fundamentals.get('company_name')}")
        print(f"   Sector: {fundamentals.get('sector')}")
        print(f"   PE Ratio: {fundamentals.get('pe_ratio')}")
        print(f"   Market Cap: {fundamentals.get('market_cap')}")
    
    # Test forex rates
    print(f"\n💱 Testing USD/PKR forex rate...")
    forex = fetcher.get_forex_rates('USD', 'PKR')
    if forex:
        print(f"   USD/PKR: {forex['rate']:.2f}")
        print(f"   Change: {forex['change']:+.4f}")
    
    print(f"\n🏁 EODHD Premium test completed!")

if __name__ == "__main__":
    test_eodhd_premium()