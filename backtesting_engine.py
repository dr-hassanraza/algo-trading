"""
Professional Backtesting Engine for PSX Quantitative Trading System
Implements walk-forward validation with proper leakage prevention
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer
from ml_model_system import MLModelSystem
from portfolio_optimizer import PortfolioOptimizer
from quant_system_config import SystemConfig

@dataclass
class BacktestResult:
    """Container for backtest results"""
    period_start: datetime
    period_end: datetime
    total_return: float
    annualized_return: float
    benchmark_return: float
    alpha: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    var_95: float
    skewness: float
    kurtosis: float
    daily_returns: pd.Series
    portfolio_values: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    performance_attribution: Dict[str, float]

class WalkForwardBacktester:
    """
    Walk-forward backtesting engine that prevents look-ahead bias
    and implements proper out-of-sample testing
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.feature_eng = FeatureEngineer(config)
        self.ml_system = MLModelSystem(config)
        self.portfolio_opt = PortfolioOptimizer(config)
        
        # Backtest parameters
        self.train_window_months = config.model.train_window_months
        self.validation_months = config.model.validation_months
        self.test_months = 1  # Test on 1 month at a time
        self.rebalance_freq = config.portfolio.rebalance_frequency
        
        # Performance tracking
        self.results = []
        self.equity_curve = pd.Series(dtype=float)
        self.benchmark_curve = pd.Series(dtype=float)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run_backtest(self, 
                    price_data: pd.DataFrame,
                    fundamental_data: Optional[pd.DataFrame] = None,
                    benchmark_data: Optional[pd.Series] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> List[BacktestResult]:
        """
        Run walk-forward backtest with proper time series validation
        
        Args:
            price_data: DataFrame with OHLCV data, symbols as columns
            fundamental_data: Optional fundamental data
            benchmark_data: Benchmark returns (KSE100)
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            List of BacktestResult objects for each period
        """
        self.logger.info("Starting walk-forward backtest...")
        
        # Prepare data
        if start_date is None:
            start_date = price_data.index[0] + timedelta(days=self.train_window_months * 30)
        if end_date is None:
            end_date = price_data.index[-1]
            
        if benchmark_data is None:
            # Use equal-weighted portfolio as benchmark
            benchmark_data = price_data.pct_change().mean(axis=1)
        
        # Generate walk-forward periods
        test_periods = self._generate_walk_forward_periods(start_date, end_date)
        
        # Initialize portfolio tracking
        portfolio_value = 100000  # Start with 100K
        cash = portfolio_value
        positions = pd.DataFrame()
        all_trades = []
        equity_values = []
        benchmark_values = [portfolio_value]
        
        for i, (train_start, train_end, val_start, val_end, test_start, test_end) in enumerate(test_periods):
            self.logger.info(f"Processing period {i+1}/{len(test_periods)}: {test_start} to {test_end}")
            
            try:
                # Extract period data
                train_data = price_data[train_start:train_end]
                val_data = price_data[val_start:val_end]
                test_data = price_data[test_start:test_end]
                
                if len(train_data) < 252:  # Need at least 1 year of training data
                    self.logger.warning(f"Insufficient training data for period {i+1}")
                    continue
                
                # Feature engineering for training period
                # Create simple momentum features for each symbol
                train_features = pd.DataFrame(index=train_data.index)
                val_features = pd.DataFrame(index=val_data.index)
                
                for symbol in train_data.columns:
                    if symbol in train_data.columns and not train_data[symbol].isna().all():
                        # Basic momentum features
                        returns = train_data[symbol].pct_change()
                        train_features[f'{symbol}_return_5d'] = returns.rolling(5).mean()
                        train_features[f'{symbol}_return_21d'] = returns.rolling(21).mean()
                        train_features[f'{symbol}_vol_21d'] = returns.rolling(21).std()
                        
                        # Validation features
                        val_returns = val_data[symbol].pct_change() if symbol in val_data.columns else pd.Series(index=val_data.index)
                        val_features[f'{symbol}_return_5d'] = val_returns.rolling(5).mean()
                        val_features[f'{symbol}_return_21d'] = val_returns.rolling(21).mean()
                        val_features[f'{symbol}_vol_21d'] = val_returns.rolling(21).std()
                
                # Drop rows with all NaN values
                train_features = train_features.dropna(how='all')
                val_features = val_features.dropna(how='all')
                
                # Train model
                model_results = self.ml_system.train_and_validate(
                    train_features, val_features, 
                    forward_returns_days=self.config.model.label_forward_days
                )
                
                if model_results is None:
                    self.logger.warning(f"Model training failed for period {i+1}")
                    continue
                
                # Generate predictions for test period
                test_features = pd.DataFrame(index=test_data.index)
                
                for symbol in test_data.columns:
                    if symbol in test_data.columns and not test_data[symbol].isna().all():
                        returns = test_data[symbol].pct_change()
                        test_features[f'{symbol}_return_5d'] = returns.rolling(5).mean()
                        test_features[f'{symbol}_return_21d'] = returns.rolling(21).mean()
                        test_features[f'{symbol}_vol_21d'] = returns.rolling(21).std()
                
                test_features = test_features.dropna(how='all')
                predictions = self.ml_system.predict(test_features)
                
                # Portfolio construction for each rebalancing date
                period_trades, period_returns, period_positions = self._run_period_backtest(
                    test_data, predictions, cash, positions
                )
                
                # Update tracking
                all_trades.extend(period_trades)
                if len(period_returns) > 0:
                    portfolio_value *= (1 + period_returns.sum())
                    equity_values.extend([portfolio_value] * len(period_returns))
                    
                    # Update benchmark
                    bench_returns = benchmark_data[test_start:test_end]
                    if len(bench_returns) > 0:
                        benchmark_values.extend([benchmark_values[-1] * (1 + bench_returns).cumprod().iloc[-1]])
                
                # Store period results
                if len(period_returns) > 0:
                    result = self._calculate_period_metrics(
                        period_returns, benchmark_data[test_start:test_end],
                        test_start, test_end, period_positions, period_trades
                    )
                    self.results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error in period {i+1}: {str(e)}")
                continue
        
        # Store final equity curves
        self.equity_curve = pd.Series(equity_values, 
                                    index=price_data[start_date:end_date].index[:len(equity_values)])
        self.benchmark_curve = pd.Series(benchmark_values[1:], 
                                       index=price_data[start_date:end_date].index[:len(benchmark_values)-1])
        
        self.logger.info(f"Backtest completed. Processed {len(self.results)} periods.")
        return self.results

    def _generate_walk_forward_periods(self, start_date: datetime, end_date: datetime) -> List[Tuple]:
        """Generate walk-forward training/validation/test periods"""
        periods = []
        current_date = start_date
        
        while current_date < end_date:
            # Training period
            train_start = current_date - timedelta(days=self.train_window_months * 30)
            train_end = current_date - timedelta(days=self.validation_months * 30 + self.test_months * 30)
            
            # Validation period
            val_start = train_end + timedelta(days=1)
            val_end = current_date - timedelta(days=self.test_months * 30)
            
            # Test period
            test_start = val_end + timedelta(days=1)
            test_end = current_date
            
            # Add embargo period to prevent leakage
            embargo_days = getattr(self.config.model, 'embargo_days', 2)
            val_end -= timedelta(days=embargo_days)
            test_start += timedelta(days=embargo_days)
            
            if test_start < test_end and train_start > start_date - timedelta(days=self.train_window_months * 30):
                periods.append((train_start, train_end, val_start, val_end, test_start, test_end))
            
            # Move forward by test period
            current_date += timedelta(days=self.test_months * 30)
        
        return periods

    def _run_period_backtest(self, price_data: pd.DataFrame, predictions: pd.DataFrame,
                           initial_cash: float, initial_positions: pd.DataFrame) -> Tuple[List, pd.Series, pd.DataFrame]:
        """Run backtest for a single period with rebalancing"""
        trades = []
        returns = []
        positions = initial_positions.copy() if not initial_positions.empty else pd.DataFrame()
        cash = initial_cash
        
        # Get rebalancing dates
        rebalance_dates = self._get_rebalancing_dates(price_data.index)
        
        for date in rebalance_dates:
            if date not in price_data.index or date not in predictions.index:
                continue
                
            # Get current predictions
            current_predictions = predictions.loc[date]
            current_prices = price_data.loc[date]
            
            # Remove NaN predictions and prices
            valid_symbols = current_predictions.dropna().index.intersection(current_prices.dropna().index)
            if len(valid_symbols) < self.config.portfolio.min_positions:
                continue
            
            # Portfolio optimization
            try:
                target_weights = self.portfolio_opt.optimize_portfolio_with_constraints(
                    current_predictions[valid_symbols]
                )
                
                # Execute trades
                period_trades, new_positions, cash = self._execute_rebalance(
                    target_weights, current_prices[valid_symbols], positions, cash, date
                )
                
                trades.extend(period_trades)
                positions = new_positions
                
                # Calculate period return
                if not positions.empty and date < price_data.index[-1]:
                    next_date_idx = price_data.index.get_loc(date) + 1
                    if next_date_idx < len(price_data.index):
                        next_date = price_data.index[next_date_idx]
                        next_prices = price_data.loc[next_date]
                        
                        # Calculate return from price changes
                        period_return = 0
                        for symbol in positions.index:
                            if symbol in next_prices.index and not pd.isna(next_prices[symbol]):
                                price_change = (next_prices[symbol] - current_prices[symbol]) / current_prices[symbol]
                                position_return = positions.loc[symbol, 'weight'] * price_change
                                period_return += position_return
                        
                        returns.append(period_return)
                
            except Exception as e:
                self.logger.warning(f"Portfolio optimization failed on {date}: {str(e)}")
                continue
        
        return trades, pd.Series(returns), positions

    def _get_rebalancing_dates(self, date_index: pd.DatetimeIndex) -> List[datetime]:
        """Get rebalancing dates based on frequency"""
        if self.rebalance_freq == 'DAILY':
            return date_index.tolist()
        elif self.rebalance_freq == 'WEEKLY':
            return [date for date in date_index if date.weekday() == 0]  # Monday
        elif self.rebalance_freq == 'MONTHLY':
            return [date for date in date_index if date.day <= 7 and date.weekday() == 0]  # First Monday
        else:
            return date_index.tolist()

    def _execute_rebalance(self, target_weights: pd.Series, prices: pd.Series,
                          current_positions: pd.DataFrame, cash: float, 
                          date: datetime) -> Tuple[List, pd.DataFrame, float]:
        """Execute portfolio rebalancing"""
        trades = []
        
        # Calculate current portfolio value
        portfolio_value = cash
        if not current_positions.empty:
            for symbol in current_positions.index:
                if symbol in prices.index and not pd.isna(prices[symbol]):
                    position_value = current_positions.loc[symbol, 'shares'] * prices[symbol]
                    portfolio_value += position_value
        
        # Calculate target positions
        new_positions = pd.DataFrame(columns=['shares', 'weight', 'value'])
        remaining_cash = cash
        
        for symbol, weight in target_weights.items():
            if abs(weight) < 0.001:  # Skip very small weights
                continue
                
            target_value = portfolio_value * weight
            target_shares = int(target_value / prices[symbol])
            
            # Current shares (if any)
            current_shares = current_positions.loc[symbol, 'shares'] if symbol in current_positions.index else 0
            
            # Calculate trade
            shares_to_trade = target_shares - current_shares
            
            if abs(shares_to_trade) > 0:
                trade_value = shares_to_trade * prices[symbol]
                commission = abs(trade_value) * self.config.execution.commission_rate
                
                # Record trade
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'shares': shares_to_trade,
                    'price': prices[symbol],
                    'value': trade_value,
                    'commission': commission,
                    'type': 'BUY' if shares_to_trade > 0 else 'SELL'
                }
                trades.append(trade)
                
                # Update cash
                remaining_cash -= (trade_value + commission)
            
            # Update position
            if target_shares != 0:
                new_positions.loc[symbol] = {
                    'shares': target_shares,
                    'weight': weight,
                    'value': target_shares * prices[symbol]
                }
        
        return trades, new_positions, remaining_cash

    def _calculate_period_metrics(self, returns: pd.Series, benchmark_returns: pd.Series,
                                start_date: datetime, end_date: datetime,
                                positions: pd.DataFrame, trades: List) -> BacktestResult:
        """Calculate comprehensive performance metrics for a period"""
        
        if len(returns) == 0:
            # Return empty result for periods with no returns
            return BacktestResult(
                period_start=start_date, period_end=end_date,
                total_return=0, annualized_return=0, benchmark_return=0,
                alpha=0, beta=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, volatility=0, win_rate=0, profit_factor=0,
                calmar_ratio=0, var_95=0, skewness=0, kurtosis=0,
                daily_returns=pd.Series(), portfolio_values=pd.Series(),
                positions=pd.DataFrame(), trades=pd.DataFrame(),
                performance_attribution={}
            )
        
        # Align returns with benchmark
        aligned_bench = benchmark_returns.reindex(returns.index).fillna(0)
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        benchmark_return = (1 + aligned_bench).prod() - 1
        
        # Annualize returns
        days = (end_date - start_date).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Ratios
        risk_free_rate = 0.12  # PKR 12% risk-free rate
        excess_returns = annualized_return - risk_free_rate
        sharpe_ratio = excess_returns / volatility if volatility > 0 else 0
        sortino_ratio = excess_returns / downside_vol if downside_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Alpha and Beta
        if len(aligned_bench) > 1 and aligned_bench.std() > 0:
            covariance = np.cov(returns, aligned_bench)[0, 1]
            beta = covariance / aligned_bench.var()
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
        else:
            alpha = annualized_return - benchmark_return
            beta = 1.0
        
        # Additional metrics
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else 0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        var_95 = returns.quantile(0.05)
        
        skewness = returns.skew() if len(returns) > 2 else 0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0
        
        # Portfolio values
        portfolio_values = (1 + returns).cumprod() * 100000
        
        # Performance attribution
        attribution = {
            'security_selection': alpha,
            'market_timing': beta * (benchmark_return - risk_free_rate),
            'interaction': 0,  # Simplified
            'total': total_return
        }
        
        return BacktestResult(
            period_start=start_date,
            period_end=end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            skewness=skewness,
            kurtosis=kurtosis,
            daily_returns=returns,
            portfolio_values=portfolio_values,
            positions=positions,
            trades=pd.DataFrame(trades),
            performance_attribution=attribution
        )

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all periods"""
        if not self.results:
            return {}
        
        # Combine all returns
        all_returns = pd.concat([r.daily_returns for r in self.results if len(r.daily_returns) > 0])
        
        if len(all_returns) == 0:
            return {}
        
        total_return = (1 + all_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(all_returns)) - 1
        
        volatility = all_returns.std() * np.sqrt(252)
        sharpe = (annualized_return - 0.12) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + all_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = len(all_returns[all_returns > 0]) / len(all_returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_periods': len(self.results),
            'profitable_periods': len([r for r in self.results if r.total_return > 0])
        }

    def generate_report(self) -> str:
        """Generate comprehensive backtest report"""
        if not self.results:
            return "No backtest results available."
        
        aggregate = self.get_aggregate_metrics()
        
        report = f"""
=== PSX QUANTITATIVE TRADING SYSTEM BACKTEST REPORT ===

AGGREGATE PERFORMANCE:
Total Return: {aggregate.get('total_return', 0):.2%}
Annualized Return: {aggregate.get('annualized_return', 0):.2%}
Volatility: {aggregate.get('volatility', 0):.2%}
Sharpe Ratio: {aggregate.get('sharpe_ratio', 0):.2f}
Maximum Drawdown: {aggregate.get('max_drawdown', 0):.2%}
Win Rate: {aggregate.get('win_rate', 0):.2%}

PERIOD ANALYSIS:
Total Periods: {aggregate.get('total_periods', 0)}
Profitable Periods: {aggregate.get('profitable_periods', 0)}
Success Rate: {aggregate.get('profitable_periods', 0) / max(aggregate.get('total_periods', 1), 1):.2%}

TARGET ACHIEVEMENT:
Alpha Target (6%): {'✓' if aggregate.get('annualized_return', 0) >= 0.06 else '✗'}
Sharpe Target (1.5): {'✓' if aggregate.get('sharpe_ratio', 0) >= 1.5 else '✗'}
Drawdown Limit (20%): {'✓' if aggregate.get('max_drawdown', 0) >= -0.20 else '✗'}

PERIOD BREAKDOWN:
"""
        
        for i, result in enumerate(self.results):
            report += f"""
Period {i+1}: {result.period_start.strftime('%Y-%m-%d')} to {result.period_end.strftime('%Y-%m-%d')}
  Return: {result.total_return:.2%} | Alpha: {result.alpha:.2%} | Sharpe: {result.sharpe_ratio:.2f}
  Max DD: {result.max_drawdown:.2%} | Win Rate: {result.win_rate:.2%}
"""
        
        return report