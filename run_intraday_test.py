import warnings
from datetime import datetime, timedelta, time

from intraday_backtesting_engine import IntradayWalkForwardBacktester
from quant_system_config import SystemConfig

warnings.filterwarnings('ignore')

def run_test():
    """
    Runs a sample intraday backtest for the last trading day.
    """
    print("🚀 Starting Intraday Performance Test...")
    
    # --- Parameters ---
    symbols = ['HBL', 'UBL', 'MCB', 'FFC', 'LUCK']
    # NOTE: The PSX DPS API only provides intraday data for the CURRENT trading day.
    # This test will only find data if run on a trading day.
    test_date = datetime.now()
    data_frequency = '1min'
    
    print(f"Date: {test_date.strftime('%Y-%m-%d')}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Frequency: {data_frequency}\n")

    try:
        # --- Initialization ---
        config = SystemConfig()
        backtester = IntradayWalkForwardBacktester(config)
        
        # --- Run Backtest ---
        print("⚙️ Running backtest...")
        results = backtester.run_intraday_backtest(
            symbols=symbols,
            start_date=datetime.combine(test_date, time.min),
            end_date=datetime.combine(test_date, time.max),
            data_frequency=data_frequency
        )
        
        if not results:
            print("❌ Backtest completed with no results. This could be due to a non-trading day (weekend/holiday) or no data available.")
            return

        # --- Report Results ---
        print("\n--- Intraday Performance Report ---")
        report = backtester.generate_intraday_report()
        print(report)
        
        # You can also access detailed results from the first result object
        day_result = results[0]
        print("\n--- Key Metrics ---")
        print(f"Total P&L: {day_result.total_pnl:,.2f} PKR")
        print(f"Win Rate: {day_result.win_rate:.1%}")
        print(f"Sharpe Ratio (Intraday): {day_result.sharpe_ratio:.2f}")
        print(f"Number of Trades: {day_result.num_trades}")
        print("-" * 35)

    except Exception as e:
        print(f"\n❌ An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
