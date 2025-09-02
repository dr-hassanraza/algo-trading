#!/usr/bin/env python3
"""
Temporary fix to provide sample data when both PSX and EODHD fail
"""

import pandas as pd
import datetime as dt
import numpy as np

def create_sample_data(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Create realistic sample data for PSX stocks when APIs fail"""
    
    # Calculate number of trading days (exclude weekends roughly)
    total_days = (end_date - start_date).days
    trading_days = int(total_days * 5/7)  # Rough estimate excluding weekends
    
    if trading_days <= 0:
        trading_days = 1
    
    # Create date range (business days only)
    date_range = pd.bdate_range(start=start_date, end=end_date)
    if len(date_range) > trading_days:
        date_range = date_range[:trading_days]
    
    # Base prices for common PSX stocks (realistic ranges)
    base_prices = {
        'ABL': 250,    # Allied Bank
        'UBL': 180,    # United Bank
        'MCB': 220,    # MCB Bank
        'HBL': 150,    # Habib Bank
        'NBP': 40,     # National Bank
        'SILK': 1.5,   # Silk Bank
        'LUCK': 800,   # Lucky Cement
        'ENGRO': 400,  # Engro Corporation
        'OGDC': 90,    # Oil & Gas Development
        'PSO': 200,    # Pakistan State Oil
    }
    
    # Get base price for this symbol
    base_price = base_prices.get(symbol.upper(), 100)  # Default to 100 if unknown
    
    # Generate realistic stock data with some volatility
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
    
    prices = []
    current_price = base_price
    
    for i in range(len(date_range)):
        # Add some realistic daily volatility (±2% typical)
        daily_change = np.random.normal(0, 0.02)  # 2% daily volatility
        current_price *= (1 + daily_change)
        
        # Ensure price doesn't go negative
        current_price = max(current_price, base_price * 0.5)
        prices.append(current_price)
    
    # Create OHLC data
    data = []
    for i, (date, close) in enumerate(zip(date_range, prices)):
        # Create realistic OHLC from close price
        volatility = close * 0.03  # 3% intraday range
        
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        open_price = close + np.random.uniform(-volatility/2, volatility/2)
        
        # Ensure OHLC relationships are correct
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (10K to 1M shares typical)
        volume = int(np.random.uniform(10000, 1000000))
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def get_fallback_data(symbol: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Get fallback data when all APIs fail"""
    
    print(f"⚠️  APIs unavailable, generating sample data for {symbol}")
    print(f"📊 This is realistic sample data based on typical PSX patterns")
    
    return create_sample_data(symbol, start_date, end_date)

# Test the function
if __name__ == "__main__":
    test_data = get_fallback_data('ABL', dt.date(2024, 11, 6), dt.date.today())
    print(f"\n✅ Generated {len(test_data)} days of sample data for ABL")
    print("\n📊 Sample data:")
    print(test_data.head())
    print(f"\n📈 Price range: {test_data['Low'].min():.2f} - {test_data['High'].max():.2f}")
    print(f"💰 Latest close: {test_data['Close'].iloc[-1]:.2f}")