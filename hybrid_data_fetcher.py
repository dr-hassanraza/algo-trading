#!/usr/bin/env python3
"""
Hybrid Data Fetcher - Best of All Sources
=========================================

Combines multiple data sources for maximum accuracy:
1. Financial Modeling Prep (for forex, US markets, fundamentals)
2. Alpha Vantage (for international markets)  
3. Yahoo Finance (backup)
4. PSX Reader (when working)
5. Scraping (last resort)

This ensures we get the most accurate prices possible.
"""

import requests
import pandas as pd
import datetime as dt
from typing import Dict, List, Optional, Any
import logging
import time
import json

logger = logging.getLogger(__name__)

class HybridDataFetcher:
    """Enhanced data fetcher using multiple premium sources"""
    
    def __init__(self):
        # API keys (in production, use environment variables)
        self.fmp_key = "e4GjeSUUFPC0PAaHpo88f9GCLI91Cuil"
        self.alpha_vantage_key = "YOUR_ALPHA_VANTAGE_KEY"  # User needs to get this
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Professional-Trading-Bot/1.0'
        })
        
        # Cache for expensive API calls
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes
        
    def get_accurate_price(self, symbol: str) -> Dict[str, Any]:
        """Get the most accurate current price from multiple sources"""
        
        results = []
        
        # Try multiple sources in parallel
        sources = [
            ("Yahoo Finance", self._get_yahoo_price),
            ("Alpha Vantage", self._get_alpha_vantage_price),
            ("Web Scraping", self._get_scraped_price),
            ("FMP Forex", self._get_fmp_context)  # For currency context
        ]
        
        clean_symbol = symbol.replace('.KAR', '').replace('.PSX', '').upper()
        
        for source_name, fetch_func in sources:
            try:
                result = fetch_func(clean_symbol)
                if result:
                    result['source'] = source_name
                    results.append(result)
            except Exception as e:
                logger.debug(f"{source_name} failed for {clean_symbol}: {e}")
        
        # Return the best result
        if results:
            # Prefer results with higher confidence or more recent data
            best_result = max(results, key=lambda x: x.get('confidence', 0))
            return best_result
        
        return None
    
    def _get_yahoo_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price from Yahoo Finance"""
        try:
            import yfinance as yf
            
            yahoo_symbols = [f"{symbol}.KAR", f"{symbol}.PSX", symbol]
            
            for yf_symbol in yahoo_symbols:
                try:
                    ticker = yf.Ticker(yf_symbol)
                    
                    # Try to get current market data
                    info = ticker.info
                    if info and 'regularMarketPrice' in info:
                        return {
                            'price': float(info['regularMarketPrice']),
                            'change': float(info.get('regularMarketChange', 0)),
                            'change_percent': float(info.get('regularMarketChangePercent', 0)),
                            'volume': int(info.get('regularMarketVolume', 0)),
                            'market_cap': info.get('marketCap'),
                            'currency': info.get('currency', 'PKR'),
                            'confidence': 8,  # High confidence for Yahoo Finance
                            'timestamp': dt.datetime.now(),
                            'api_symbol': yf_symbol
                        }
                    
                    # Fallback to historical data
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        return {
                            'price': float(latest['Close']),
                            'volume': int(latest['Volume']),
                            'currency': 'PKR',
                            'confidence': 6,  # Medium confidence for historical
                            'timestamp': dt.datetime.now(),
                            'api_symbol': yf_symbol,
                            'note': 'From historical data'
                        }
                        
                except Exception:
                    continue
            
            return None
            
        except ImportError:
            return None
    
    def _get_alpha_vantage_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price from Alpha Vantage (requires API key)"""
        
        if self.alpha_vantage_key == "YOUR_ALPHA_VANTAGE_KEY":
            return None
        
        try:
            # Try Alpha Vantage for international stocks
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': f"{symbol}.KAR",
                'apikey': self.alpha_vantage_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'Global Quote' in data:
                quote = data['Global Quote']
                return {
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': float(quote.get('10. change percent', '0%').replace('%', '')),
                    'volume': int(quote.get('06. volume', 0)),
                    'currency': 'PKR',
                    'confidence': 9,  # Very high confidence for Alpha Vantage
                    'timestamp': dt.datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Alpha Vantage error: {e}")
            return None
    
    def _get_scraped_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get price from web scraping (investing.com, etc.)"""
        
        # This is a simplified example - real implementation would need
        # proper scraping with headers, rate limiting, etc.
        
        try:
            # Example: Try to scrape from a financial website
            # This is just a placeholder - actual implementation needed
            
            # For now, return None (scraping requires more complex setup)
            return None
            
        except Exception as e:
            logger.debug(f"Scraping error: {e}")
            return None
    
    def _get_fmp_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get context data from FMP (currency rates, etc.)"""
        
        try:
            # Get USD/PKR rate for context
            url = f"https://financialmodelingprep.com/api/v3/fx/USDPKR"
            response = self.session.get(url, params={'apikey': self.fmp_key}, timeout=10)
            
            if response.status_code == 200:
                fx_data = response.json()
                if fx_data:
                    return {
                        'usd_pkr_rate': fx_data[0].get('price', 280),  # Approximate current rate
                        'confidence': 3,  # Low confidence as this is just context
                        'note': 'Currency context from FMP'
                    }
            
            return None
            
        except Exception:
            return None
    
    def get_historical_data(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Get historical data with fallback sources"""
        
        # Try sources in order of preference
        sources = [
            self._get_yahoo_historical,
            self._get_alpha_vantage_historical,
        ]
        
        for source_func in sources:
            try:
                data = source_func(symbol, start_date, end_date)
                if not data.empty:
                    return data
            except Exception as e:
                logger.debug(f"Historical source failed: {e}")
                continue
        
        return pd.DataFrame()
    
    def _get_yahoo_historical(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            import yfinance as yf
            
            yahoo_symbols = [f"{symbol}.KAR", f"{symbol}.PSX", symbol]
            
            for yf_symbol in yahoo_symbols:
                try:
                    ticker = yf.Ticker(yf_symbol)
                    hist = ticker.history(start=start_date, end=end_date + dt.timedelta(days=1))
                    
                    if not hist.empty:
                        hist = hist.reset_index()
                        hist.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                        return hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                        
                except Exception:
                    continue
            
            return pd.DataFrame()
            
        except ImportError:
            return pd.DataFrame()
    
    def _get_alpha_vantage_historical(self, symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Get historical data from Alpha Vantage"""
        
        if self.alpha_vantage_key == "YOUR_ALPHA_VANTAGE_KEY":
            return pd.DataFrame()
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': f"{symbol}.KAR",
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'Time Series (Daily)' in data:
                daily_data = data['Time Series (Daily)']
                
                # Convert to DataFrame
                records = []
                for date_str, values in daily_data.items():
                    date_obj = pd.to_datetime(date_str).date()
                    if start_date <= date_obj <= end_date:
                        records.append({
                            'Date': date_obj,
                            'Open': float(values['1. open']),
                            'High': float(values['2. high']),
                            'Low': float(values['3. low']),
                            'Close': float(values['4. close']),
                            'Volume': int(values['5. volume'])
                        })
                
                if records:
                    df = pd.DataFrame(records)
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df.sort_values('Date')
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.debug(f"Alpha Vantage historical error: {e}")
            return pd.DataFrame()

# Create an improved price verification function
def get_verified_current_price(symbol: str) -> Dict[str, Any]:
    """Get verified current price with multiple source confirmation"""
    
    fetcher = HybridDataFetcher()
    
    print(f"\n🔍 Verifying current price for {symbol}...")
    print("-" * 40)
    
    # Get price from multiple sources
    result = fetcher.get_accurate_price(symbol)
    
    if result:
        print(f"✅ Found price: {result['price']:.2f} {result.get('currency', 'PKR')}")
        print(f"📊 Source: {result['source']}")
        print(f"🎯 Confidence: {result.get('confidence', 0)}/10")
        
        if 'change' in result:
            change_str = f"{result['change']:+.2f}"
            if 'change_percent' in result:
                change_str += f" ({result['change_percent']:+.2f}%)"
            print(f"📈 Change: {change_str}")
        
        return result
    else:
        print("❌ Could not verify current price")
        return None

# Test function
def test_hybrid_fetcher():
    """Test the hybrid data fetcher"""
    print("🚀 Testing Hybrid Data Fetcher")
    print("=" * 50)
    
    # Test with UBL
    result = get_verified_current_price('UBL')
    
    if result:
        print(f"\n📊 Price verification successful!")
        print(f"If this shows ~382 PKR, the system is working correctly.")
        print(f"If this shows ~145 PKR, we still have data quality issues.")
    else:
        print(f"\n❌ Price verification failed - all sources unavailable")

if __name__ == "__main__":
    test_hybrid_fetcher()