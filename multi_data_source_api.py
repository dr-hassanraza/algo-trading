#!/usr/bin/env python3
"""
Multi-Source Data API for PSX Trading
=====================================

A hybrid data fetching system that combines multiple APIs:
1. EODHD API (current) - for historical price data  
2. PSX Official Data Portal - for official data
3. Alpha Vantage - for additional fundamentals
4. yfinance - as backup for some stocks
5. Custom web scraping - for PSX-specific data

This provides more reliable and comprehensive data for Pakistani stocks.
"""

import os
import requests
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Union
import datetime as dt
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Standard data structure for stock information"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    book_value: Optional[float] = None
    source: str = "unknown"
    timestamp: str = ""

class MultiSourceDataAPI:
    """Multi-source data fetching API for PSX stocks"""
    
    def __init__(self):
        self.eodhd_api_key = os.getenv('EODHD_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')  # You can get free key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # PSX symbol mapping for better success rates
        self.psx_symbol_mapping = {
            'UBL': {'full_name': 'United Bank Limited', 'sector': 'Banking'},
            'MCB': {'full_name': 'MCB Bank Limited', 'sector': 'Banking'},
            'HBL': {'full_name': 'Habib Bank Limited', 'sector': 'Banking'},
            'ABL': {'full_name': 'Allied Bank Limited', 'sector': 'Banking'},
            'NBP': {'full_name': 'National Bank of Pakistan', 'sector': 'Banking'},
            'OGDC': {'full_name': 'Oil & Gas Development Company', 'sector': 'Oil & Gas'},
            'PPL': {'full_name': 'Pakistan Petroleum Limited', 'sector': 'Oil & Gas'},
            'PSO': {'full_name': 'Pakistan State Oil', 'sector': 'Oil & Gas'},
            'LUCK': {'full_name': 'Lucky Cement Limited', 'sector': 'Cement'},
            'DGKC': {'full_name': 'DG Khan Cement', 'sector': 'Cement'},
            'ENGRO': {'full_name': 'Engro Corporation Limited', 'sector': 'Fertilizer'},
            'FFC': {'full_name': 'Fauji Fertilizer Company', 'sector': 'Fertilizer'},
            'TRG': {'full_name': 'TRG Pakistan Limited', 'sector': 'Technology'},
            'NESTLE': {'full_name': 'Nestle Pakistan Limited', 'sector': 'Food'}
        }
    
    def get_stock_data(self, symbol: str, use_fallback: bool = True) -> Optional[StockData]:
        """Get comprehensive stock data using multiple sources"""
        
        logger.info(f"Fetching data for {symbol} using multi-source approach")
        
        # Try data sources in order of preference
        data_sources = [
            self._get_eodhd_data,
            self._get_yfinance_data,
            self._get_psx_web_data,
            self._get_alpha_vantage_data
        ]
        
        for source_func in data_sources:
            try:
                data = source_func(symbol)
                if data:
                    logger.info(f"Successfully got data for {symbol} from {data.source}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to get data from {source_func.__name__}: {e}")
                continue
        
        logger.error(f"Failed to get data for {symbol} from all sources")
        return None
    
    def _get_eodhd_data(self, symbol: str) -> Optional[StockData]:
        """Get data from EODHD API (current working source)"""
        
        if not self.eodhd_api_key:
            return None
            
        # Format symbol for EODHD
        formatted_symbol = f"{symbol.upper()}.KAR"
        url = f"https://eodhd.com/api/real-time/{formatted_symbol}?api_token={self.eodhd_api_key}&fmt=json"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'code' in data and data['code']:
                return StockData(
                    symbol=symbol,
                    price=float(data.get('close', 0)),
                    change=float(data.get('change', 0)),
                    change_percent=float(data.get('change_p', 0)),
                    volume=int(data.get('volume', 0)),
                    source="EODHD",
                    timestamp=data.get('timestamp', '')
                )
        except Exception as e:
            logger.warning(f"EODHD API error for {symbol}: {e}")
            return None
    
    def _get_yfinance_data(self, symbol: str) -> Optional[StockData]:
        """Get data from yfinance (backup source)"""
        
        try:
            import yfinance as yf
            
            # Try different symbol formats for PSX
            symbol_formats = [
                f"{symbol}.KAR",  # Karachi Stock Exchange format
                f"{symbol}.PSX",  # Pakistan Stock Exchange format
                symbol            # Plain symbol
            ]
            
            for sym_format in symbol_formats:
                try:
                    ticker = yf.Ticker(sym_format)
                    info = ticker.info
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty and 'regularMarketPrice' in info:
                        latest = hist.iloc[-1]
                        
                        return StockData(
                            symbol=symbol,
                            price=float(info.get('regularMarketPrice', latest['Close'])),
                            change=float(info.get('regularMarketChange', 0)),
                            change_percent=float(info.get('regularMarketChangePercent', 0)),
                            volume=int(latest['Volume']),
                            market_cap=info.get('marketCap'),
                            pe_ratio=info.get('forwardPE'),
                            eps=info.get('trailingEps'),
                            dividend_yield=info.get('dividendYield'),
                            book_value=info.get('bookValue'),
                            source="yfinance",
                            timestamp=str(dt.datetime.now())
                        )
                except:
                    continue
                    
        except ImportError:
            logger.warning("yfinance not available")
            return None
        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")
            return None
    
    def _get_psx_web_data(self, symbol: str) -> Optional[StockData]:
        """Scrape data from PSX official website"""
        
        try:
            # PSX has an API endpoint for real-time data
            url = f"https://dps.psx.com.pk/market-watch"
            
            # This would require more sophisticated scraping
            # For now, return None as it needs specific implementation
            logger.info(f"PSX web scraping not implemented yet for {symbol}")
            return None
            
        except Exception as e:
            logger.warning(f"PSX web error for {symbol}: {e}")
            return None
    
    def _get_alpha_vantage_data(self, symbol: str) -> Optional[StockData]:
        """Get data from Alpha Vantage API"""
        
        if not self.alpha_vantage_key:
            return None
            
        try:
            # Alpha Vantage global quote endpoint
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': f"{symbol}.KAR",
                'apikey': self.alpha_vantage_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                
                return StockData(
                    symbol=symbol,
                    price=float(quote.get('05. price', 0)),
                    change=float(quote.get('09. change', 0)),
                    change_percent=float(quote.get('10. change percent', '0').replace('%', '')),
                    volume=int(quote.get('06. volume', 0)),
                    source="Alpha Vantage",
                    timestamp=quote.get('07. latest trading day', '')
                )
                
        except Exception as e:
            logger.warning(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    def get_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[StockData]]:
        """Get data for multiple stocks efficiently"""
        
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.get_stock_data(symbol)
            time.sleep(0.5)  # Rate limiting
            
        return results
    
    def get_kse100_index(self) -> Optional[Dict]:
        """Get KSE100 index data"""
        
        try:
            # Try multiple sources for KSE100
            sources = [
                self._get_kse100_eodhd,
                self._get_kse100_trading_economics,
                self._get_kse100_psx
            ]
            
            for source_func in sources:
                try:
                    data = source_func()
                    if data:
                        return data
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to get KSE100 data: {e}")
            
        return None
    
    def _get_kse100_eodhd(self) -> Optional[Dict]:
        """Get KSE100 from EODHD"""
        
        if not self.eodhd_api_key:
            return None
            
        try:
            url = f"https://eodhd.com/api/real-time/KSE100.INDX?api_token={self.eodhd_api_key}&fmt=json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'value': float(data.get('close', 0)),
                'change': float(data.get('change', 0)),
                'change_percent': float(data.get('change_p', 0)),
                'source': 'EODHD',
                'timestamp': data.get('timestamp', '')
            }
            
        except Exception as e:
            logger.warning(f"EODHD KSE100 error: {e}")
            return None
    
    def _get_kse100_trading_economics(self) -> Optional[Dict]:
        """Get KSE100 from Trading Economics"""
        
        # This would require web scraping Trading Economics
        # Implementation can be added later
        return None
    
    def _get_kse100_psx(self) -> Optional[Dict]:
        """Get KSE100 from PSX official site"""
        
        # This would require scraping PSX official website
        # Implementation can be added later
        return None

# Test the multi-source API
def test_multi_source_api():
    """Test the multi-source data API"""
    
    print("🚀 Testing Multi-Source Data API for PSX")
    print("=" * 50)
    
    api = MultiSourceDataAPI()
    
    # Test symbols
    test_symbols = ['UBL', 'MCB', 'OGDC', 'LUCK']
    
    print("\\n📊 Testing individual stocks...")
    for symbol in test_symbols:
        print(f"\\n🔍 Testing {symbol}...")
        data = api.get_stock_data(symbol)
        
        if data:
            print(f"   ✅ {symbol}: {data.price:.2f} PKR ({data.change:+.2f} | {data.change_percent:+.2f}%)")
            print(f"   📊 Volume: {data.volume:,} | Source: {data.source}")
        else:
            print(f"   ❌ {symbol}: No data available")
    
    print("\\n📈 Testing KSE100 index...")
    kse100 = api.get_kse100_index()
    if kse100:
        print(f"   ✅ KSE100: {kse100['value']:.2f} ({kse100['change']:+.2f} | {kse100['change_percent']:+.2f}%)")
        print(f"   📊 Source: {kse100['source']}")
    else:
        print(f"   ❌ KSE100: No data available")
    
    print("\\n🏁 Multi-source API test completed!")

if __name__ == "__main__":
    test_multi_source_api()