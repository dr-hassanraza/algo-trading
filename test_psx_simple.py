#!/usr/bin/env python3
"""
Test PSX library with different approaches
"""

import datetime as dt
from psx import stocks, tickers
import pandas as pd

# First let's get the ticker symbols properly
print("Getting ticker symbols...")
ticker_df = tickers()
ticker_symbols = ticker_df['symbol'].tolist()
print(f"Found {len(ticker_symbols)} symbols")
print("First 10 symbols:", ticker_symbols[:10])

# Try a few different symbols with different date ranges
test_symbols = ['SILK', 'UBL', 'MCB']

for symbol in test_symbols:
    print(f"\n{'='*30}")
    print(f"Testing {symbol}")
    print('='*30)
    
    # Try different date ranges
    date_ranges = [
        (dt.date(2024, 1, 1), dt.date(2024, 12, 31)),  # Full year 2024
        (dt.date(2024, 12, 1), dt.date(2024, 12, 31)),  # December 2024
        (dt.date.today() - dt.timedelta(days=30), dt.date.today()),  # Last 30 days
    ]
    
    for start, end in date_ranges:
        print(f"\nTrying date range: {start} to {end}")
        try:
            data = stocks(symbol, start=start, end=end)
            if data is not None and not data.empty:
                print(f"✅ Success! Got {len(data)} rows")
                print(f"Columns: {list(data.columns)}")
                print(f"Date range in data: {data.index.min()} to {data.index.max()}")
                break  # Success, no need to try other date ranges
            else:
                print("❌ Empty data")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ All date ranges failed for {symbol}")

# Test with symbols that are more likely to have data
print(f"\n{'='*50}")
print("Testing with symbols from ticker list...")

# Try with some symbols from the actual ticker list
for symbol in ticker_symbols[:5]:  # Test first 5 symbols
    print(f"\nTesting {symbol}...")
    try:
        data = stocks(symbol, start=dt.date(2024, 1, 1), end=dt.date.today())
        if data is not None and not data.empty:
            print(f"✅ Got data for {symbol}: {len(data)} rows")
            print(f"Columns: {list(data.columns)}")
            break
    except Exception as e:
        print(f"❌ Error for {symbol}: {e}")
else:
    print("❌ No symbols returned data")