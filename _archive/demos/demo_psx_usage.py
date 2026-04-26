#!/usr/bin/env python3
"""
Demo: How to use PSX Data Reader for live data extraction

This shows how to use the psx-data-reader library once it's properly working,
and provides fallback functionality when the library has issues.
"""

import pandas as pd
import datetime as dt
from typing import List, Optional

# Example of how you would use psx-data-reader when it works properly
def demo_psx_usage():
    """
    Demo showing how to use psx-data-reader for live PSX data extraction
    """
    print("🚀 PSX Data Reader Usage Demo")
    print("=" * 50)
    
    # Example 1: Get available tickers
    print("\n📋 Example 1: Getting Available Tickers")
    print("-" * 30)
    try:
        from psx import tickers
        ticker_df = tickers()
        if not ticker_df.empty:
            symbols = ticker_df['symbol'].tolist()
            print(f"✅ Found {len(symbols)} available symbols")
            print("First 10 symbols:", symbols[:10])
            
            # Filter for actual stocks (not bonds/TFCs)
            stocks_only = ticker_df[
                (ticker_df['isDebt'] != True) & 
                (ticker_df['isETF'] != True)
            ]['symbol'].tolist()
            print(f"📊 Stocks only (excluding bonds/ETFs): {len(stocks_only)}")
            print("First 10 stock symbols:", stocks_only[:10])
        else:
            print("❌ No tickers data available")
    except Exception as e:
        print(f"❌ Error getting tickers: {e}")
    
    # Example 2: Historical data for specific symbols
    print("\n📊 Example 2: Historical Data Extraction")
    print("-" * 30)
    
    # These are the symbols you mentioned in your question
    example_symbols = ['SILK']  # Start with one symbol
    end_date = dt.date.today()
    start_date = dt.date(2025, 1, 1)  # As you mentioned in your example
    
    print(f"Trying to get data for SILK from {start_date} to {end_date}")
    
    # This is how you would use it (based on your example):
    print("\n📝 Code example:")
    print("from psx import stocks")
    print('data = stocks("SILK", start=datetime.date(2025,1,1), end=datetime.date.today())')
    
    try:
        from psx import stocks
        data = stocks("SILK", start=start_date, end=end_date)
        
        if data is not None and not data.empty:
            print(f"✅ Success! Got {len(data)} rows of data")
            print(f"📅 Columns: {list(data.columns)}")
            print(f"📊 Data shape: {data.shape}")
            print(f"🗓️ Date range: {data.index.min()} to {data.index.max()}")
            print("\n📋 Sample data:")
            print(data.head())
        else:
            print("❌ No data returned")
            
    except Exception as e:
        print(f"❌ Error getting historical data: {e}")
        print("\n💡 Note: The psx-data-reader library appears to have some issues")
        print("   This might be due to:")
        print("   - Website structure changes")
        print("   - Network connectivity")
        print("   - Library compatibility issues")
    
    # Example 3: Alternative approach - Manual data structure
    print("\n🔧 Example 3: Creating Sample Data Structure")
    print("-" * 30)
    
    # Show what the data structure would look like when working
    sample_data = pd.DataFrame({
        'Date': pd.date_range(start='2025-01-01', periods=30, freq='D'),
        'Open': [100 + i*0.5 for i in range(30)],
        'High': [102 + i*0.5 for i in range(30)],  
        'Low': [98 + i*0.5 for i in range(30)],
        'Close': [101 + i*0.5 for i in range(30)],
        'Volume': [1000000 + i*10000 for i in range(30)]
    })
    
    print("✅ Sample data structure for PSX stock data:")
    print(sample_data.head())
    print(f"\n📊 Data types:")
    print(sample_data.dtypes)
    
    return sample_data

def create_psx_compatible_fetcher():
    """
    Create a data fetcher that would work with psx-data-reader
    when the library is functioning properly
    """
    
    class PSXDataFetcher:
        def __init__(self):
            self.available = self._check_psx_availability()
        
        def _check_psx_availability(self) -> bool:
            try:
                from psx import stocks, tickers
                # Test with a simple call
                ticker_df = tickers()
                return not ticker_df.empty
            except Exception:
                return False
        
        def get_tickers(self) -> List[str]:
            """Get list of available PSX tickers"""
            if not self.available:
                return []
            
            try:
                from psx import tickers
                ticker_df = tickers()
                return ticker_df['symbol'].tolist()
            except Exception as e:
                print(f"Error getting tickers: {e}")
                return []
        
        def fetch_data(self, symbol: str, start_date: dt.date, end_date: dt.date) -> Optional[pd.DataFrame]:
            """Fetch historical data for a symbol"""
            if not self.available:
                return None
                
            try:
                from psx import stocks
                data = stocks(symbol, start=start_date, end=end_date)
                
                if data is not None and not data.empty:
                    # Ensure proper formatting
                    if data.index.name != 'Date':
                        data = data.reset_index()
                    
                    return data
                return None
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return None
    
    return PSXDataFetcher()

if __name__ == "__main__":
    # Run the demo
    sample_data = demo_psx_usage()
    
    print("\n" + "="*60)
    print("📝 USAGE SUMMARY")
    print("="*60)
    print("""
✅ When psx-data-reader is working properly, use:
   
   # Get available tickers
   from psx import tickers
   ticker_list = tickers()
   
   # Get historical data
   from psx import stocks
   data = stocks("SILK", start=datetime.date(2025,1,1), end=datetime.date.today())

❌ Current Status: The library has some issues that need to be resolved
   
💡 Alternative: Use our existing enhanced_data_fetcher.py which provides:
   - Multiple data sources (EODHD, yfinance, multi-source API)
   - Fallback mechanisms
   - Better error handling
   - Caching functionality
   
🔧 To use our enhanced system:
   from enhanced_data_fetcher import EnhancedDataFetcher
   fetcher = EnhancedDataFetcher()
   data = fetcher.fetch("UBL.KAR", start_date, end_date)
    """)