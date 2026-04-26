"""
Test script for PSX Intraday Backtesting System
Demonstrates backtesting with real PSX DPS API data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import List, Dict
import asyncio
import warnings
warnings.filterwarnings('ignore')

from intraday_backtesting_engine import IntradayWalkForwardBacktester
from psx_dps_fetcher import PSXDPSFetcher
from quant_system_config import SystemConfig

class MockPSXDPSFetcher:
    """Mock PSX DPS fetcher for testing when API is unavailable"""
    
    def fetch_intraday_ticks(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """Generate realistic intraday tick data"""
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate data for current trading day
        today = datetime.now().replace(hour=9, minute=45, second=0, microsecond=0)
        end_time = today.replace(hour=15, minute=30)
        
        # Create minute-by-minute data
        timestamps = pd.date_range(today, end_time, freq='1min')
        
        # Starting price based on symbol
        if 'HBL' in symbol:
            base_price = 120.0
        elif 'UBL' in symbol:
            base_price = 85.0
        elif 'ENGRO' in symbol:
            base_price = 280.0
        elif 'LUCK' in symbol:
            base_price = 650.0
        else:
            base_price = np.random.uniform(50, 300)
        
        data = []
        price = base_price
        
        for i, ts in enumerate(timestamps):
            # Market hours simulation
            hour = ts.hour
            minute = ts.minute
            
            # Higher volatility at market open/close
            if hour == 9 or hour == 15:
                base_vol = 0.003  # 0.3%
            else:
                base_vol = 0.001  # 0.1%
            
            # Random walk with intraday patterns
            change = np.random.normal(0, base_vol)
            
            # Add some momentum/reversal patterns
            if i > 5:
                # Simple momentum
                recent_change = (price - base_price) / base_price
                momentum_factor = recent_change * 0.1
                change += momentum_factor
            
            price = max(price * (1 + change), 0.1)  # Prevent negative prices
            
            # Volume simulation (higher during volatility)
            base_volume = np.random.randint(1000, 10000)
            volume_mult = 1 + abs(change) * 10
            volume = int(base_volume * volume_mult)
            
            data.append({
                'price': round(price, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df

def test_intraday_backtest_system():
    """Test the intraday backtesting system"""
    print("=== PSX Intraday Backtesting System Test ===\n")
    
    # Initialize system
    config = SystemConfig()
    
    # Test with mock fetcher first
    backtester = IntradayWalkForwardBacktester(config)
    backtester.psx_fetcher = MockPSXDPSFetcher()  # Use mock for testing
    
    # Test symbols
    test_symbols = ['HBL', 'UBL', 'ENGRO', 'LUCK', 'MCB']
    
    # Test period (recent days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)  # Last 5 trading days
    
    print(f"Testing intraday backtesting for {len(test_symbols)} symbols")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Using mock data for testing...\n")
    
    try:
        # Run intraday backtest
        results = backtester.run_intraday_backtest(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            data_frequency='1min'
        )
        
        print(f"Backtest completed! Processed {len(results)} trading days.\n")
        
        if results:
            # Display detailed results
            print("=== DETAILED RESULTS ===")
            
            for i, result in enumerate(results):
                print(f"\nDay {i+1}: {result.trading_date.strftime('%Y-%m-%d')}")
                print(f"  P&L: {result.total_pnl:,.0f} PKR")
                print(f"  Return: {result.total_return:.2%}")
                print(f"  Trades: {result.num_trades}")
                print(f"  Win Rate: {result.win_rate:.1%}")
                print(f"  Symbols Traded: {', '.join(result.symbols_traded)}")
                print(f"  Avg Hold Time: {result.average_hold_time_minutes:.1f} minutes")
                print(f"  Signals: {result.signals_generated} generated, {result.signals_executed} executed")
                print(f"  Commission: {result.commission_paid:,.0f} PKR")
                print(f"  Max Drawdown: {result.max_intraday_drawdown:.2%}")
            
            # Generate comprehensive report
            print("\n" + "="*80)
            print(backtester.generate_intraday_report())
            
            # Aggregate performance
            perf = backtester.get_aggregate_performance()
            print("\n=== PERFORMANCE ANALYSIS ===")
            print(f"Overall Return: {perf.get('total_return', 0):.2%}")
            print(f"Annualized Return: {perf.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            print(f"Daily Win Rate: {perf.get('win_rate_days', 0):.1%}")
            print(f"Average Daily Trades: {perf.get('avg_daily_trades', 0):.1f}")
            
        else:
            print("No results generated. Check data availability and system configuration.")
    
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()

def test_with_real_psx_data():
    """Test with real PSX DPS API data"""
    print("=== Testing with Real PSX DPS Data ===\n")
    
    # Initialize with real PSX fetcher
    config = SystemConfig()
    backtester = IntradayWalkForwardBacktester(config)
    
    # Test with a few liquid symbols
    liquid_symbols = ['HBL', 'UBL', 'MCB']  # Major banks with good liquidity
    
    # Test for today only
    today = datetime.now()
    
    print(f"Testing real-time data for: {', '.join(liquid_symbols)}")
    print(f"Date: {today.date()}\n")
    
    try:
        # Test individual symbol fetch first
        print("Testing individual symbol data fetch...")
        for symbol in liquid_symbols:
            try:
                data = backtester.psx_fetcher.fetch_intraday_ticks(symbol)
                if not data.empty:
                    print(f"✅ {symbol}: {len(data)} ticks received")
                    print(f"   Latest: {data.index[-1]} | Price: {data['price'].iloc[-1]:.2f} | Volume: {data['volume'].iloc[-1]:,}")
                else:
                    print(f"⚠️  {symbol}: No data received")
            except Exception as e:
                print(f"❌ {symbol}: Error - {str(e)}")
        
        print("\nRunning single-day backtest with real data...")
        
        results = backtester.run_intraday_backtest(
            symbols=liquid_symbols,
            start_date=today,
            end_date=today,
            data_frequency='1min'
        )
        
        if results:
            result = results[0]
            print(f"\n=== REAL DATA RESULTS ===")
            print(f"P&L: {result.total_pnl:,.0f} PKR")
            print(f"Trades: {result.num_trades}")
            print(f"Signals: {result.signals_generated} generated, {result.signals_executed} executed")
            print(f"Symbols traded: {', '.join(result.symbols_traded)}")
            
            if not result.trades.empty:
                print(f"\nTrade Details:")
                for _, trade in result.trades.head(5).iterrows():
                    print(f"  {trade['timestamp'].strftime('%H:%M')} - {trade['action']} {trade['symbol']}")
                    print(f"    Price: {trade['price']:.2f} | Shares: {trade['shares']} | P&L: {trade.get('pnl', 0):.0f}")
        else:
            print("No results from real data test.")
    
    except Exception as e:
        print(f"Error testing real data: {str(e)}")
        print("This is expected if market is closed or API is unavailable.")

def demo_intraday_features():
    """Demonstrate key intraday features"""
    print("=== Intraday Trading Features Demo ===\n")
    
    features = [
        "✅ Real-time PSX DPS API integration",
        "✅ Tick-by-tick and minute-bar processing", 
        "✅ Intraday signal generation with momentum/reversal",
        "✅ Dynamic position sizing and risk management",
        "✅ Stop-loss and take-profit automation",
        "✅ Maximum hold time enforcement (4 hours default)",
        "✅ Trading hours enforcement (9:45 AM - 3:30 PM PSX time)",
        "✅ Commission and slippage modeling",
        "✅ Intraday drawdown monitoring",
        "✅ Real-time equity curve tracking",
        "✅ Signal execution rate monitoring",
        "✅ End-of-day position closure",
        "✅ Comprehensive intraday performance metrics"
    ]
    
    print("Key Features:")
    for feature in features:
        print(f"  {feature}")
    
    print(f"\nData Sources:")
    print(f"  • Primary: PSX DPS Official API (https://dps.psx.com.pk/timeseries/int/{{SYMBOL}})")
    print(f"  • Format: [timestamp, price, volume] tick data")
    print(f"  • Update: Real-time during market hours")
    print(f"  • Coverage: All PSX listed securities")
    
    print(f"\nTrading Rules:")
    print(f"  • Max position size: 10% of capital or 100K PKR")
    print(f"  • Max daily trades: 50")
    print(f"  • Commission: 0.15% per trade")
    print(f"  • Slippage: 5 basis points")
    print(f"  • Signal interval: 5 minutes minimum")
    print(f"  • Max hold time: 4 hours")

if __name__ == "__main__":
    # Demo key features
    demo_intraday_features()
    
    print("\n" + "="*80)
    
    # Run mock data test
    test_intraday_backtest_system()
    
    print("\n" + "="*80)
    
    # Optional: Test with real data (will work when market is open)
    try_real_data = input("\nTry real PSX DPS API data? (y/n): ").lower().strip() == 'y'
    
    if try_real_data:
        test_with_real_psx_data()
    else:
        print("Skipping real data test. Run during market hours for live data testing.")
    
    print("\n" + "="*80)
    print("INTRADAY BACKTESTING SYSTEM TEST COMPLETE!")
    print("="*80)
    
    print("\nNext Steps:")
    print("1. Run during PSX market hours (9:45 AM - 3:30 PM PKT) for live data")
    print("2. Customize symbols list for your trading universe") 
    print("3. Adjust risk parameters in SystemConfig")
    print("4. Integrate with your signal generation logic")
    print("5. Add real-time alerts and execution interfaces")