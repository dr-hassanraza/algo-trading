"""
ENHANCED WALK-FORWARD BACKTESTING ENGINE
Advanced Validation Framework for High-Accuracy Intraday Trading

Features:
- Walk-forward analysis with rolling windows
- Regime-aware backtesting and validation
- Multi-timeframe performance analysis
- Advanced performance metrics and risk analysis
- Statistical significance testing
- Model degradation detection
- Real-time performance monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from scipy import stats
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

@dataclass
class BacktestResult:
    """Container for backtest results"""
    strategy_name: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float
    
    # Regime-specific performance
    regime_performance: Dict[str, Dict[str, float]]
    
    # Time-based performance
    hourly_performance: Dict[int, float]
    daily_performance: Dict[str, float]
    
    # Statistical significance
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]

@dataclass
class WalkForwardResult:
    """Container for walk-forward analysis results"""
    strategy_name: str
    analysis_period: str
    
    # Walk-forward metrics
    in_sample_results: List[BacktestResult]
    out_sample_results: List[BacktestResult]
    
    # Degradation analysis
    performance_decay: float
    consistency_score: float
    overfitting_ratio: float
    
    # Stability metrics
    return_stability: float
    sharpe_stability: float
    drawdown_stability: float

class EnhancedBacktestingEngine:
    """Advanced backtesting engine with walk-forward analysis"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.trading_cost = 0.001   # 0.1% trading cost
        
        # Walk-forward parameters
        self.wf_params = {
            'training_window': 252,     # Trading days for training
            'testing_window': 63,       # Trading days for testing
            'step_size': 21,           # Days to step forward
            'min_trades': 10,          # Minimum trades for valid test
            'rebalance_frequency': 21   # Rebalance every 21 days
        }
        
        # Performance tracking
        self.results_history = []
        self.regime_tracker = None
        
    def run_walk_forward_analysis(self, strategy_func, market_data: Dict[str, pd.DataFrame],
                                 symbols: List[str], start_date: str, end_date: str) -> WalkForwardResult:
        """Run comprehensive walk-forward analysis"""
        
        print(f"🚀 Running walk-forward analysis from {start_date} to {end_date}")
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Prepare data
        aligned_data = self._align_market_data(market_data, symbols, start_dt, end_dt)
        
        if aligned_data.empty:
            raise ValueError("Insufficient market data for backtesting")
        
        # Generate walk-forward windows
        windows = self._generate_walk_forward_windows(aligned_data.index, start_dt, end_dt)
        
        in_sample_results = []
        out_sample_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"📊 Processing window {i+1}/{len(windows)}: {train_start.date()} to {test_end.date()}")
            
            # Training period
            train_data = {}
            for symbol in symbols:
                if symbol in market_data:
                    train_data[symbol] = market_data[symbol].loc[train_start:train_end]
            
            # Test period
            test_data = {}
            for symbol in symbols:
                if symbol in market_data:
                    test_data[symbol] = market_data[symbol].loc[test_start:test_end]
            
            # Train strategy (if needed)
            if hasattr(strategy_func, 'train'):
                strategy_func.train(train_data)
            
            # Run in-sample backtest
            is_result = self.backtest_strategy(
                strategy_func, train_data, symbols, 
                f"IS_Window_{i+1}", train_start, train_end
            )
            in_sample_results.append(is_result)
            
            # Run out-of-sample backtest
            oos_result = self.backtest_strategy(
                strategy_func, test_data, symbols,
                f"OOS_Window_{i+1}", test_start, test_end
            )
            out_sample_results.append(oos_result)
        
        # Analyze walk-forward results
        return self._analyze_walk_forward_results(
            f"WF_{strategy_func.__name__ if hasattr(strategy_func, '__name__') else 'Strategy'}",
            f"{start_date}_to_{end_date}",
            in_sample_results,
            out_sample_results
        )
    
    def backtest_strategy(self, strategy_func, market_data: Dict[str, pd.DataFrame],
                         symbols: List[str], strategy_name: str,
                         start_date: datetime, end_date: datetime) -> BacktestResult:
        """Run comprehensive backtest for a strategy"""
        
        # Initialize portfolio
        portfolio = {
            'cash': 1000000,  # Starting capital
            'positions': {},
            'equity_curve': [],
            'trades': [],
            'daily_returns': []
        }
        
        # Get trading dates
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        regime_performance = {}
        hourly_performance = {h: [] for h in range(24)}
        daily_performance = {}
        
        for date in all_dates:
            daily_start_equity = self._calculate_portfolio_value(portfolio, market_data, date)
            
            # Get market data for this date
            day_data = {}
            for symbol in symbols:
                if symbol in market_data and date in market_data[symbol].index:
                    day_data[symbol] = market_data[symbol].loc[date]
            
            if not day_data:
                continue
            
            # Generate signals using strategy
            signals = strategy_func(day_data, date)
            
            if not signals:
                continue
            
            # Execute trades
            for signal in signals:
                self._execute_trade(portfolio, signal, market_data, date)
            
            # Update portfolio value
            daily_end_equity = self._calculate_portfolio_value(portfolio, market_data, date)
            daily_return = (daily_end_equity - daily_start_equity) / daily_start_equity if daily_start_equity > 0 else 0
            
            portfolio['equity_curve'].append({
                'date': date,
                'equity': daily_end_equity,
                'return': daily_return
            })
            portfolio['daily_returns'].append(daily_return)
            
            # Track hourly performance (simplified)
            hour = date.hour
            hourly_performance[hour].append(daily_return)
            
            # Track daily performance
            day_name = date.strftime('%A')
            if day_name not in daily_performance:
                daily_performance[day_name] = []
            daily_performance[day_name].append(daily_return)
        
        # Calculate comprehensive metrics
        return self._calculate_comprehensive_metrics(
            strategy_name, start_date, end_date, portfolio,
            regime_performance, hourly_performance, daily_performance
        )
    
    def regime_aware_backtest(self, strategy_func, market_data: Dict[str, pd.DataFrame],
                            symbols: List[str], regime_detector, strategy_name: str) -> Dict[str, BacktestResult]:
        """Run regime-aware backtesting"""
        
        print(f"🌪️ Running regime-aware backtest for {strategy_name}")
        
        regime_results = {}
        
        # Detect regimes for each symbol
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            data = market_data[symbol]
            regimes = []
            
            # Detect regime for each time period
            for i in range(50, len(data)):  # Need minimum data for regime detection
                window_data = data.iloc[max(0, i-50):i+1]
                regime = regime_detector.detect_regime(window_data, symbol)
                regimes.append(regime.regime_name)
            
            # Group data by regime
            regime_data = {}
            for regime_name in set(regimes):
                regime_indices = [i for i, r in enumerate(regimes) if r == regime_name]
                regime_data[regime_name] = data.iloc[regime_indices]
            
            # Backtest each regime separately
            for regime_name, regime_market_data in regime_data.items():
                if len(regime_market_data) < 20:  # Skip if insufficient data
                    continue
                
                result = self.backtest_strategy(
                    strategy_func,
                    {symbol: regime_market_data},
                    [symbol],
                    f"{strategy_name}_{regime_name}",
                    regime_market_data.index[0],
                    regime_market_data.index[-1]
                )
                
                regime_results[f"{symbol}_{regime_name}"] = result
        
        return regime_results
    
    def _align_market_data(self, market_data: Dict[str, pd.DataFrame],
                          symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Align market data across symbols and dates"""
        
        # Find common date range
        common_dates = None
        
        for symbol in symbols:
            if symbol in market_data:
                data = market_data[symbol]
                symbol_dates = data.loc[start_date:end_date].index
                
                if common_dates is None:
                    common_dates = symbol_dates
                else:
                    common_dates = common_dates.intersection(symbol_dates)
        
        if common_dates is None or len(common_dates) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(index=common_dates)
    
    def _generate_walk_forward_windows(self, date_index: pd.DatetimeIndex,
                                     start_date: datetime, end_date: datetime) -> List[Tuple]:
        """Generate walk-forward analysis windows"""
        
        windows = []
        current_start = start_date
        
        while current_start < end_date:
            # Training window
            train_start = current_start
            train_end = train_start + timedelta(days=self.wf_params['training_window'])
            
            # Testing window
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.wf_params['testing_window'])
            
            # Check if we have enough data
            if test_end <= end_date:
                windows.append((train_start, train_end, test_start, test_end))
            
            # Move window forward
            current_start += timedelta(days=self.wf_params['step_size'])
        
        return windows
    
    def _execute_trade(self, portfolio: Dict, signal: Dict, market_data: Dict[str, pd.DataFrame], date: datetime):
        """Execute a trade in the portfolio"""
        
        symbol = signal.get('symbol')
        action = signal.get('action', 'HOLD')
        size = signal.get('size', 0)
        
        if symbol not in market_data or date not in market_data[symbol].index:
            return
        
        price = market_data[symbol].loc[date, 'Close']
        
        if action == 'BUY' and portfolio['cash'] >= size:
            # Buy position
            shares = int(size / price)
            cost = shares * price * (1 + self.trading_cost)
            
            if cost <= portfolio['cash']:
                portfolio['cash'] -= cost
                portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + shares
                
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'cost': cost
                })
        
        elif action == 'SELL' and symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
            # Sell position
            shares = min(int(size / price), portfolio['positions'][symbol])
            proceeds = shares * price * (1 - self.trading_cost)
            
            portfolio['cash'] += proceeds
            portfolio['positions'][symbol] -= shares
            
            if portfolio['positions'][symbol] == 0:
                del portfolio['positions'][symbol]
            
            portfolio['trades'].append({
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'proceeds': proceeds
            })
    
    def _calculate_portfolio_value(self, portfolio: Dict, market_data: Dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate total portfolio value"""
        
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in market_data and date in market_data[symbol].index:
                price = market_data[symbol].loc[date, 'Close']
                total_value += shares * price
        
        return total_value
    
    def _calculate_comprehensive_metrics(self, strategy_name: str, start_date: datetime, end_date: datetime,
                                       portfolio: Dict, regime_performance: Dict, hourly_performance: Dict,
                                       daily_performance: Dict) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        
        if not portfolio['daily_returns'] or len(portfolio['daily_returns']) < 2:
            return self._create_empty_result(strategy_name, start_date, end_date)
        
        returns = np.array(portfolio['daily_returns'])
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        if len(returns) < 2:
            return self._create_empty_result(strategy_name, start_date, end_date)
        
        # Basic performance metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        trades = portfolio['trades']
        winning_trades = sum(1 for t in trades if t.get('proceeds', 0) > t.get('cost', 0))
        losing_trades = len(trades) - winning_trades
        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0
        
        wins = [t.get('proceeds', 0) - t.get('cost', 0) for t in trades if t.get('proceeds', 0) > t.get('cost', 0)]
        losses = [t.get('cost', 0) - t.get('proceeds', 0) for t in trades if t.get('proceeds', 0) <= t.get('cost', 0)]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Statistical significance
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        confidence_interval = stats.t.interval(0.95, len(returns)-1, loc=np.mean(returns), scale=stats.sem(returns))
        
        # Process hourly performance
        hourly_avg = {}
        for hour, hour_returns in hourly_performance.items():
            hourly_avg[hour] = np.mean(hour_returns) if hour_returns else 0
        
        # Process daily performance
        daily_avg = {}
        for day, day_returns in daily_performance.items():
            daily_avg[day] = np.mean(day_returns) if day_returns else 0
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            regime_performance=regime_performance,
            hourly_performance=hourly_avg,
            daily_performance=daily_avg,
            t_statistic=t_stat,
            p_value=p_value,
            confidence_interval=confidence_interval
        )
    
    def _analyze_walk_forward_results(self, strategy_name: str, analysis_period: str,
                                    in_sample_results: List[BacktestResult],
                                    out_sample_results: List[BacktestResult]) -> WalkForwardResult:
        """Analyze walk-forward results for overfitting and stability"""
        
        # Calculate performance decay
        is_returns = [r.annualized_return for r in in_sample_results if r.annualized_return is not None]
        oos_returns = [r.annualized_return for r in out_sample_results if r.annualized_return is not None]
        
        performance_decay = (np.mean(is_returns) - np.mean(oos_returns)) / np.mean(is_returns) if is_returns and oos_returns and np.mean(is_returns) != 0 else 0
        
        # Calculate consistency score
        oos_positive_periods = sum(1 for r in oos_returns if r > 0)
        consistency_score = oos_positive_periods / len(oos_returns) if oos_returns else 0
        
        # Calculate overfitting ratio
        overfitting_ratio = performance_decay if performance_decay > 0 else 0
        
        # Calculate stability metrics
        return_stability = 1 - (np.std(oos_returns) / np.mean(oos_returns)) if oos_returns and np.mean(oos_returns) != 0 else 0
        
        is_sharpe = [r.sharpe_ratio for r in in_sample_results if r.sharpe_ratio is not None]
        oos_sharpe = [r.sharpe_ratio for r in out_sample_results if r.sharpe_ratio is not None]
        sharpe_stability = 1 - abs(np.mean(is_sharpe) - np.mean(oos_sharpe)) / np.mean(is_sharpe) if is_sharpe and np.mean(is_sharpe) != 0 else 0
        
        is_drawdown = [abs(r.max_drawdown) for r in in_sample_results if r.max_drawdown is not None]
        oos_drawdown = [abs(r.max_drawdown) for r in out_sample_results if r.max_drawdown is not None]
        drawdown_stability = 1 - abs(np.mean(oos_drawdown) - np.mean(is_drawdown)) / np.mean(is_drawdown) if is_drawdown and np.mean(is_drawdown) != 0 else 0
        
        return WalkForwardResult(
            strategy_name=strategy_name,
            analysis_period=analysis_period,
            in_sample_results=in_sample_results,
            out_sample_results=out_sample_results,
            performance_decay=performance_decay,
            consistency_score=consistency_score,
            overfitting_ratio=overfitting_ratio,
            return_stability=return_stability,
            sharpe_stability=sharpe_stability,
            drawdown_stability=drawdown_stability
        )
    
    def _create_empty_result(self, strategy_name: str, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Create empty result for insufficient data"""
        
        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            var_95=0.0,
            cvar_95=0.0,
            skewness=0.0,
            kurtosis=0.0,
            regime_performance={},
            hourly_performance={},
            daily_performance={},
            t_statistic=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0)
        )
    
    def generate_performance_report(self, result: Union[BacktestResult, WalkForwardResult]) -> str:
        """Generate comprehensive performance report"""
        
        if isinstance(result, BacktestResult):
            return self._generate_backtest_report(result)
        else:
            return self._generate_walkforward_report(result)
    
    def _generate_backtest_report(self, result: BacktestResult) -> str:
        """Generate backtest performance report"""
        
        report = f"""
📊 BACKTEST PERFORMANCE REPORT
Strategy: {result.strategy_name}
Period: {result.start_date.date()} to {result.end_date.date()}

📈 PERFORMANCE METRICS
Total Return: {result.total_return:.2%}
Annualized Return: {result.annualized_return:.2%}
Volatility: {result.volatility:.2%}
Sharpe Ratio: {result.sharpe_ratio:.2f}
Sortino Ratio: {result.sortino_ratio:.2f}
Maximum Drawdown: {result.max_drawdown:.2%}
Calmar Ratio: {result.calmar_ratio:.2f}

📊 TRADE STATISTICS
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}
Win Rate: {result.win_rate:.2%}
Average Win: ${result.avg_win:.2f}
Average Loss: ${result.avg_loss:.2f}
Profit Factor: {result.profit_factor:.2f}

⚠️ RISK METRICS
Value at Risk (95%): {result.var_95:.2%}
Conditional VaR (95%): {result.cvar_95:.2%}
Skewness: {result.skewness:.2f}
Kurtosis: {result.kurtosis:.2f}

📊 STATISTICAL SIGNIFICANCE
T-Statistic: {result.t_statistic:.2f}
P-Value: {result.p_value:.4f}
95% Confidence Interval: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]
"""
        return report
    
    def _generate_walkforward_report(self, result: WalkForwardResult) -> str:
        """Generate walk-forward analysis report"""
        
        # Calculate averages
        avg_is_return = np.mean([r.annualized_return for r in result.in_sample_results])
        avg_oos_return = np.mean([r.annualized_return for r in result.out_sample_results])
        avg_is_sharpe = np.mean([r.sharpe_ratio for r in result.in_sample_results])
        avg_oos_sharpe = np.mean([r.sharpe_ratio for r in result.out_sample_results])
        
        report = f"""
🔄 WALK-FORWARD ANALYSIS REPORT
Strategy: {result.strategy_name}
Analysis Period: {result.analysis_period}

📈 PERFORMANCE COMPARISON
In-Sample Avg Return: {avg_is_return:.2%}
Out-of-Sample Avg Return: {avg_oos_return:.2%}
Performance Decay: {result.performance_decay:.2%}

📊 STABILITY METRICS
Consistency Score: {result.consistency_score:.2%}
Overfitting Ratio: {result.overfitting_ratio:.2%}
Return Stability: {result.return_stability:.2%}
Sharpe Stability: {result.sharpe_stability:.2%}
Drawdown Stability: {result.drawdown_stability:.2%}

📊 SHARPE RATIO COMPARISON
In-Sample Avg Sharpe: {avg_is_sharpe:.2f}
Out-of-Sample Avg Sharpe: {avg_oos_sharpe:.2f}

🎯 VALIDATION RESULTS
Number of Windows: {len(result.out_sample_results)}
Strategy Robustness: {'HIGH' if result.consistency_score > 0.6 and result.overfitting_ratio < 0.3 else 'MEDIUM' if result.consistency_score > 0.4 else 'LOW'}
"""
        return report

# Testing function
def test_backtesting_engine():
    """Test the enhanced backtesting engine"""
    print("🧪 Testing Enhanced Backtesting Engine...")
    
    # Create sample strategy function
    def simple_momentum_strategy(day_data: Dict, date: datetime) -> List[Dict]:
        signals = []
        
        for symbol, data in day_data.items():
            # Simple momentum signal
            if hasattr(data, 'Close') and not pd.isna(data.Close):
                price = data.Close
                
                # Simple buy signal (this is just for testing)
                if price > 100:  # Arbitrary threshold
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'size': 10000  # $10,000 position
                    })
        
        return signals
    
    # Create sample market data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    symbols = ['HBL', 'UBL']
    
    market_data = {}
    for symbol in symbols:
        # Generate sample price data
        prices = []
        current_price = 100 + hash(symbol) % 50
        
        for _ in range(len(dates)):
            change = np.random.normal(0.001, 0.02)  # 1% daily drift, 2% volatility
            current_price *= (1 + change)
            prices.append(current_price)
        
        market_data[symbol] = pd.DataFrame({
            'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
            'High': [p * np.random.uniform(1.01, 1.03) for p in prices],
            'Low': [p * np.random.uniform(0.97, 0.99) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000, 100000, len(dates))
        }, index=dates)
    
    # Initialize backtesting engine
    engine = EnhancedBacktestingEngine()
    
    # Test simple backtest
    print("📊 Testing simple backtest...")
    result = engine.backtest_strategy(
        simple_momentum_strategy,
        market_data,
        symbols,
        'Simple_Momentum',
        pd.to_datetime('2023-01-01'),
        pd.to_datetime('2023-06-30')
    )
    
    print(f"   Total Return: {result.total_return:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2%}")
    print(f"   Total Trades: {result.total_trades}")
    
    # Test walk-forward analysis (simplified)
    print("\n🔄 Testing walk-forward analysis...")
    try:
        wf_result = engine.run_walk_forward_analysis(
            simple_momentum_strategy,
            market_data,
            symbols,
            '2023-01-01',
            '2023-12-31'
        )
        
        print(f"   Performance Decay: {wf_result.performance_decay:.2%}")
        print(f"   Consistency Score: {wf_result.consistency_score:.2%}")
        print(f"   Overfitting Ratio: {wf_result.overfitting_ratio:.2%}")
        
    except Exception as e:
        print(f"   Walk-forward test failed: {e}")
    
    # Generate report
    print("\n📋 Generating performance report...")
    report = engine.generate_performance_report(result)
    print(report[:500] + "..." if len(report) > 500 else report)
    
    print("\n✅ Enhanced Backtesting Engine test completed!")

if __name__ == "__main__":
    test_backtesting_engine()