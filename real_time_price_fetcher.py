#!/usr/bin/env python3
"""
Real-time Price Fetcher for PSX Stocks
======================================

Attempts to get current/today's price from multiple sources:
1. PSX official data (when working)
2. Financial websites scraping
3. API sources
4. Fallback to latest historical + market status indicator
"""

import requests
import pandas as pd
import datetime as dt
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RealTimePriceFetcher:
    """Fetch real-time prices for PSX stocks"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current price for a PSX stock symbol
        
        Returns:
            Dict with price info or None if not available
        """
        
        # Try multiple methods in order
        methods = [
            self._get_price_from_psx_website,
            self._get_price_from_investing_com,
            self._get_price_from_yahoo_finance,
        ]
        
        clean_symbol = symbol.replace('.KAR', '').replace('.PSX', '').upper()
        
        for method in methods:
            try:
                result = method(clean_symbol)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Price method {method.__name__} failed for {clean_symbol}: {e}")
                continue
        
        return None
    
    def _get_price_from_psx_website(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Try to get price from PSX official website"""
        # This would require scraping PSX website
        # For now, return None as we don't have direct access
        return None
    
    def _get_price_from_investing_com(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Try to get price from investing.com"""
        try:
            # This is a simplified example - actual implementation would need
            # to handle investing.com's specific URL structure for PSX stocks
            url = f"https://www.investing.com/search/?q={symbol}"
            # Implementation would go here
            return None
        except Exception:
            return None
    
    def _get_price_from_yahoo_finance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Try to get current price from Yahoo Finance"""
        try:
            # Try different Yahoo Finance symbol formats for PSX
            yahoo_symbols = [f"{symbol}.KAR", f"{symbol}.PSX", symbol]
            
            for yahoo_symbol in yahoo_symbols:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(yahoo_symbol)
                    info = ticker.info
                    
                    if info and 'regularMarketPrice' in info:
                        return {
                            'price': info['regularMarketPrice'],
                            'change': info.get('regularMarketChange', 0),
                            'change_percent': info.get('regularMarketChangePercent', 0),
                            'volume': info.get('regularMarketVolume', 0),
                            'source': f'Yahoo Finance ({yahoo_symbol})',
                            'timestamp': dt.datetime.now(),
                            'market_time': info.get('regularMarketTime')
                        }
                except Exception:
                    continue
            
            return None
        except Exception:
            return None
    
    def get_market_status(self) -> Dict[str, str]:
        """Get current PSX market status"""
        now = dt.datetime.now()
        
        # PSX trading hours: 9:30 AM - 3:30 PM PKT (Monday-Friday)
        # Note: This is simplified - actual implementation should check holidays
        
        if now.weekday() >= 5:  # Weekend
            return {
                'status': 'closed',
                'message': 'Market Closed (Weekend)',
                'next_open': 'Monday 9:30 AM'
            }
        
        # Convert to Pakistan time (approximate)
        pkt_time = now  # Assuming system is in PKT or adjust as needed
        
        if pkt_time.time() < dt.time(9, 30):
            return {
                'status': 'pre_market',
                'message': 'Pre-Market (Opens 9:30 AM)',
                'next_event': 'Market opens at 9:30 AM'
            }
        elif pkt_time.time() > dt.time(15, 30):
            return {
                'status': 'closed',
                'message': 'Market Closed (Opens 9:30 AM tomorrow)',
                'next_event': 'Market opens tomorrow at 9:30 AM'
            }
        else:
            return {
                'status': 'open',
                'message': 'Market Open (Closes 3:30 PM)',
                'next_event': 'Market closes at 3:30 PM'
            }

def get_enhanced_price_info(symbol: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get enhanced price information combining real-time and historical data
    
    Args:
        symbol: Stock symbol
        historical_data: DataFrame with historical OHLCV data
    
    Returns:
        Dictionary with comprehensive price information
    """
    
    fetcher = RealTimePriceFetcher()
    market_status = fetcher.get_market_status()
    
    # Try to get real-time price (prefer accurate PSX prices)
    try:
        from psx_web_scraper import get_most_accurate_price
        accurate_price = get_most_accurate_price(symbol)
        if accurate_price and accurate_price.get('price'):
            # Use accurate price as "real-time" data
            real_time_data = {
                'price': accurate_price['price'],
                'source': accurate_price['source'],
                'timestamp': dt.datetime.now(),
                'confidence': accurate_price.get('confidence', 10)
            }
        else:
            real_time_data = fetcher.get_current_price(symbol)
    except ImportError:
        real_time_data = fetcher.get_current_price(symbol)
    
    if not historical_data.empty:
        latest_historical = historical_data.iloc[-1]
        latest_date = pd.to_datetime(latest_historical['Date']).date()
        latest_price = float(latest_historical['Close'])
        
        # Check if historical data is recent (within 1 day for market days)
        days_old = (dt.date.today() - latest_date).days
        is_recent = days_old <= 1
        
        if real_time_data and real_time_data.get('confidence', 0) >= 8:
            # Use real-time data if high confidence (includes manual database entries)
            return {
                'price': real_time_data['price'],
                'source': real_time_data['source'],
                'is_current': True,
                'market_status': market_status,
                'last_update': real_time_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'historical_price': latest_price,
                'historical_date': latest_date.strftime('%Y-%m-%d'),
                'data_freshness': 'Verified current price',
                'confidence': real_time_data.get('confidence', 10)
            }
        elif real_time_data and market_status['status'] == 'open':
            # Use real-time data if market is open and available
            return {
                'price': real_time_data['price'],
                'source': real_time_data['source'],
                'is_current': True,
                'market_status': market_status,
                'last_update': real_time_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'historical_price': latest_price,
                'historical_date': latest_date.strftime('%Y-%m-%d'),
                'data_freshness': 'Real-time'
            }
        else:
            # Use historical data with status indicator
            freshness = 'Current' if is_recent else f'{days_old} days old'
            
            return {
                'price': latest_price,
                'source': 'Historical data',
                'is_current': is_recent and market_status['status'] != 'open',
                'market_status': market_status,
                'last_update': latest_date.strftime('%Y-%m-%d'),
                'data_freshness': freshness,
                'note': 'Latest available data' if not is_recent else 'Recent market data'
            }
    else:
        return {
            'price': None,
            'source': 'No data available',
            'is_current': False,
            'market_status': market_status,
            'data_freshness': 'No data',
            'error': 'No historical or real-time data available'
        }

# Test function
def test_price_fetcher():
    """Test the real-time price fetcher"""
    print("🕐 Testing Real-Time Price Fetcher")
    print("=" * 50)
    
    # Test market status
    fetcher = RealTimePriceFetcher()
    market_status = fetcher.get_market_status()
    print(f"📈 Market Status: {market_status}")
    
    # Test with some sample data
    test_symbols = ['UBL', 'MCB', 'ABL']
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}:")
        
        # Create sample historical data
        sample_data = pd.DataFrame({
            'Date': [dt.date.today() - dt.timedelta(days=1)],
            'Close': [250.0]  # Sample price
        })
        
        enhanced_info = get_enhanced_price_info(symbol, sample_data)
        
        print(f"   💰 Price: {enhanced_info.get('price', 'N/A')}")
        print(f"   📊 Source: {enhanced_info.get('source', 'N/A')}")
        print(f"   🕐 Freshness: {enhanced_info.get('data_freshness', 'N/A')}")
        print(f"   ✅ Current: {enhanced_info.get('is_current', False)}")

if __name__ == "__main__":
    test_price_fetcher()