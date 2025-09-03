"""
Test script for the comprehensive backtesting system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from backtesting_engine import WalkForwardBacktester, BacktestResult
from quant_system_config import SystemConfig

def generate_mock_psx_data(symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Generate realistic mock PSX OHLCV data for backtesting"""
    np.random.seed(42)
    
    dates = pd.date_range(start_date, end_date, freq='D')
    # Remove weekends
    dates = dates[dates.weekday < 5]
    
    # Create multi-level columns for OHLCV data
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    columns = pd.MultiIndex.from_product([symbols, ohlcv_cols], names=['Symbol', 'Field'])
    
    price_data = pd.DataFrame(index=dates, columns=columns)
    
    for symbol in symbols:
        # Start with realistic PSX prices
        if 'BANK' in symbol:
            initial_price = np.random.uniform(50, 200)
        elif 'OIL' in symbol or 'ENERGY' in symbol:
            initial_price = np.random.uniform(100, 500)
        elif 'CEMENT' in symbol:
            initial_price = np.random.uniform(20, 80)
        else:
            initial_price = np.random.uniform(30, 300)
        
        # Generate realistic OHLCV data
        price = initial_price
        momentum = 0
        
        for i, date in enumerate(dates):
            # Base return with momentum and mean reversion
            base_return = np.random.normal(0, 0.02)  # 2% daily volatility
            momentum_factor = momentum * 0.1
            mean_reversion = -0.05 * (price - initial_price) / initial_price
            
            daily_return = base_return + momentum_factor + mean_reversion
            daily_return = np.clip(daily_return, -0.10, 0.10)  # Limit to ±10%
            
            # Open price (slightly gap from previous close)
            if i == 0:
                open_price = initial_price
            else:
                gap = np.random.normal(0, 0.005)  # Small overnight gap
                open_price = price * (1 + gap)
            
            # Close price
            close_price = open_price * (1 + daily_return)
            
            # High and Low
            intraday_range = abs(daily_return) + np.random.uniform(0.01, 0.03)
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, intraday_range))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, intraday_range))
            
            # Volume (realistic for PSX)
            base_volume = np.random.uniform(100000, 5000000)  # 100K to 5M shares
            volume_multiplier = 1 + abs(daily_return) * 2  # Higher volume on big moves
            volume = int(base_volume * volume_multiplier)
            
            # Store data
            price_data.loc[date, (symbol, 'Open')] = open_price
            price_data.loc[date, (symbol, 'High')] = high_price
            price_data.loc[date, (symbol, 'Low')] = low_price
            price_data.loc[date, (symbol, 'Close')] = close_price
            price_data.loc[date, (symbol, 'Volume')] = volume
            
            # Update price and momentum for next day
            price = close_price
            momentum = momentum * 0.8 + daily_return * 0.2
    
    # Convert to simple column structure (Close prices only for features)
    close_prices = pd.DataFrame(index=dates)
    for symbol in symbols:
        close_prices[symbol] = price_data[(symbol, 'Close')]
    
    return close_prices.astype(float)

def generate_mock_fundamental_data(symbols: List[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate mock fundamental data"""
    np.random.seed(42)
    
    # Generate quarterly data
    quarterly_dates = pd.date_range(dates[0], dates[-1], freq='QS')
    
    fundamental_data = pd.DataFrame(index=quarterly_dates)
    
    for symbol in symbols:
        # Market cap (in millions PKR)
        base_mcap = np.random.uniform(10000, 100000)
        fundamental_data[f'{symbol}_market_cap'] = base_mcap * (1 + np.random.normal(0, 0.1, len(quarterly_dates))).cumprod()
        
        # P/E ratio
        fundamental_data[f'{symbol}_pe_ratio'] = np.random.uniform(8, 25, len(quarterly_dates))
        
        # ROE
        fundamental_data[f'{symbol}_roe'] = np.random.uniform(0.08, 0.30, len(quarterly_dates))
        
        # Debt to equity
        fundamental_data[f'{symbol}_debt_equity'] = np.random.uniform(0.2, 2.0, len(quarterly_dates))
    
    # Forward fill to daily frequency
    daily_fundamental = fundamental_data.reindex(dates, method='ffill')
    return daily_fundamental

def run_backtest_demo():
    """Run a comprehensive backtest demonstration"""
    print("=== PSX Quantitative Trading System Backtest Demo ===\n")
    
    # Create test universe - realistic PSX symbols
    psx_symbols = [
        'HBL', 'UBL', 'MCB', 'BAFL', 'ABL',  # Banking
        'TRG', 'SYSTEMS', 'NETSOL', 'AVN', 'INBOX',  # Technology
        'LUCK', 'DG.KHAN', 'CHCC', 'MLCF', 'FCCL',  # Cement
        'PSO', 'HASCOL', 'SHEL', 'SSGC', 'SNGP',  # Energy
        'ENGRO', 'FFC', 'EFERT', 'FATIMA', 'EPCL',  # Fertilizer
        'NESTLE', 'UNILEVER', 'COLG', 'KTML', 'WAVES',  # Consumer
        'OGDC', 'PPL', 'MARI', 'POL', 'MPCL',  # Oil & Gas
        'KTM', 'INDU', 'ABOT', 'SEARL', 'GATM',  # Pharma
        'ATRL', 'NML', 'ASTL', 'UNITY', 'YOUW',  # Textiles
        'PKGP', 'CHERAT', 'LOADS', 'BIFO', 'MERIT'  # Others
    ]
    
    # Generate test data
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print("Generating mock PSX data...")
    price_data = generate_mock_psx_data(psx_symbols, start_date, end_date)
    fundamental_data = generate_mock_fundamental_data(psx_symbols, price_data.index)
    
    # Generate benchmark (KSE100 proxy)
    benchmark_returns = price_data.pct_change().mean(axis=1)
    
    print(f"Data generated: {len(price_data)} days, {len(psx_symbols)} symbols\n")
    
    # Initialize system
    config = SystemConfig()
    backtester = WalkForwardBacktester(config)
    
    # Run backtest
    print("Starting walk-forward backtest...")
    print("This may take a few minutes as we train models on each period...\n")
    
    backtest_start = datetime(2022, 1, 1)  # Start backtest after sufficient training data
    results = backtester.run_backtest(
        price_data=price_data,
        fundamental_data=fundamental_data,
        benchmark_data=benchmark_returns,
        start_date=backtest_start,
        end_date=end_date
    )
    
    # Generate and display report
    print("\n" + "="*80)
    print(backtester.generate_report())
    
    # Additional analysis
    if results:
        print("\n=== DETAILED ANALYSIS ===")
        
        # Performance by period
        period_returns = [r.total_return for r in results]
        period_alphas = [r.alpha for r in results]
        period_sharpes = [r.sharpe_ratio for r in results]
        
        print(f"\nPERIOD STATISTICS:")
        print(f"Average Period Return: {np.mean(period_returns):.2%}")
        print(f"Std Dev of Returns: {np.std(period_returns):.2%}")
        print(f"Average Alpha: {np.mean(period_alphas):.2%}")
        print(f"Average Sharpe: {np.mean(period_sharpes):.2f}")
        
        # Risk analysis
        all_returns = pd.concat([r.daily_returns for r in results if len(r.daily_returns) > 0])
        if len(all_returns) > 0:
            print(f"\nRISK ANALYSIS:")
            print(f"VaR (95%): {all_returns.quantile(0.05):.2%}")
            print(f"Expected Shortfall: {all_returns[all_returns <= all_returns.quantile(0.05)].mean():.2%}")
            print(f"Skewness: {all_returns.skew():.2f}")
            print(f"Kurtosis: {all_returns.kurtosis():.2f}")
        
        # Trading analysis
        all_trades = pd.concat([r.trades for r in results if not r.trades.empty])
        if not all_trades.empty:
            print(f"\nTRADING ANALYSIS:")
            print(f"Total Trades: {len(all_trades)}")
            print(f"Avg Trades per Period: {len(all_trades) / len(results):.1f}")
            print(f"Long Trades: {len(all_trades[all_trades['type'] == 'BUY'])}")
            print(f"Short Trades: {len(all_trades[all_trades['type'] == 'SELL'])}")
            
            total_commission = all_trades['commission'].sum()
            total_value = all_trades['value'].abs().sum()
            print(f"Total Commission: {total_commission:,.0f} PKR")
            print(f"Commission as % of Volume: {total_commission/total_value:.3%}")

def create_performance_visualizations(backtester: WalkForwardBacktester):
    """Create performance visualization charts"""
    if not backtester.results or backtester.equity_curve.empty:
        print("No results available for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PSX Quantitative Trading System Performance', fontsize=16)
    
    # Equity curve
    axes[0, 0].plot(backtester.equity_curve.index, backtester.equity_curve.values, label='Strategy', linewidth=2)
    axes[0, 0].plot(backtester.benchmark_curve.index, backtester.benchmark_curve.values, label='Benchmark', linewidth=2, alpha=0.7)
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drawdown
    cumulative = backtester.equity_curve / backtester.equity_curve.iloc[0]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    axes[0, 1].plot(drawdown.index, drawdown.values, color='red')
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rolling Sharpe
    all_returns = pd.concat([r.daily_returns for r in backtester.results if len(r.daily_returns) > 0])
    if len(all_returns) > 0:
        rolling_sharpe = all_returns.rolling(60).mean() / all_returns.rolling(60).std() * np.sqrt(252)
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 0].axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='Target (1.5)')
        axes[1, 0].set_title('Rolling 60-Day Sharpe Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Return distribution
    axes[1, 1].hist(all_returns.values, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(all_returns.mean(), color='red', linestyle='--', label=f'Mean: {all_returns.mean():.3f}')
    axes[1, 1].set_title('Daily Returns Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/macair2020/Desktop/Algo_Trading/backtest_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance charts saved as 'backtest_performance.png'")

if __name__ == "__main__":
    # Run the backtest demonstration
    run_backtest_demo()
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE!")
    print("="*80)
    
    # Additional testing options
    print("\nTo run with visualization (requires matplotlib):")
    print("python test_backtesting_system.py --visualize")
    
    print("\nTo run with real PSX data:")
    print("1. Ensure data is available in expected format")
    print("2. Modify the data loading section in this script")
    print("3. Run with --real-data flag")