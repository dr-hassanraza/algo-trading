#!/usr/bin/env python3
"""
Direct test of PSX data reader library to understand the data format
"""

import datetime as dt
from psx import stocks, tickers

# Test tickers function
print("Testing tickers()...")
try:
    ticker_data = tickers()
    print(f"Type: {type(ticker_data)}")
    print(f"Shape: {getattr(ticker_data, 'shape', 'N/A')}")
    print(f"Columns: {getattr(ticker_data, 'columns', 'N/A')}")
    print(f"Index: {getattr(ticker_data, 'index', 'N/A')}")
    if hasattr(ticker_data, 'head'):
        print("First few rows:")
        print(ticker_data.head())
except Exception as e:
    print(f"Error with tickers(): {e}")

print("\n" + "="*50 + "\n")

# Test stocks function
print("Testing stocks() for SILK...")
try:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=5)
    
    data = stocks("SILK", start=start_date, end=end_date)
    print(f"Type: {type(data)}")
    
    if data is not None:
        print(f"Is empty: {data.empty if hasattr(data, 'empty') else 'N/A'}")
        print(f"Shape: {getattr(data, 'shape', 'N/A')}")
        print(f"Columns: {list(data.columns) if hasattr(data, 'columns') else 'N/A'}")
        print(f"Index: {data.index.name if hasattr(data, 'index') else 'N/A'}")
        
        if hasattr(data, 'head') and not data.empty:
            print("First few rows:")
            print(data.head())
            print(f"\nData types:")
            print(data.dtypes)
    else:
        print("No data returned")
        
except Exception as e:
    print(f"Error with stocks(): {e}")