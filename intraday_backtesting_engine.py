"""
Intraday Backtesting Engine for PSX DPS Real-Time Data
Handles tick-by-tick data from https://dps.psx.com.pk/timeseries/int/{SYMBOL}
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

from psx_dps_fetcher import PSXDPSFetcher
from intraday_signal_analyzer import IntradaySignalAnalyzer, IntradaySignal
from intraday_risk_manager import IntradayRiskManager
from quant_system_config import SystemConfig

@dataclass
class IntradayBacktestResult:
    """Container for intraday backtest results"""
    trading_date: datetime
    session_start: datetime
    session_end: datetime
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_return: float
    num_trades: int
    win_rate: float
    profit_factor: float
    max_intraday_drawdown: float
    sharpe_ratio: float
    symbols_traded: List[str]
    average_hold_time_minutes: float
    commission_paid: float
    slippage_cost: float
    trades: pd.DataFrame
    equity_curve: pd.Series
    positions: pd.DataFrame
    signals_generated: int
    signals_executed: int

class IntradayWalkForwardBacktester:
    """
    Intraday backtesting engine using real PSX DPS tick data
    Handles minute-by-minute and tick-by-tick trading decisions
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.psx_fetcher = PSXDPSFetcher()
        self.signal_analyzer = IntradaySignalAnalyzer()
        self.risk_manager = IntradayRiskManager(config)
        
        # Intraday specific parameters
        self.trading_start_time = time(9, 45)  # PSX opens 9:45 AM
        self.trading_end_time = time(15, 30)   # PSX closes 3:30 PM
        self.min_signal_interval = 5  # Minutes between signal generation
        self.max_position_hold_time = 4 * 60  # 4 hours max hold time
        
        # Backtesting parameters
        self.initial_capital = 1000000  # 1M PKR
        self.max_daily_trades = 50
        self.commission_rate = 0.0015  # 0.15% commission
        self.slippage_bps = 5  # 5 basis points slippage
        
        self.results = []
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run_intraday_backtest(self, 
                             symbols: List[str],
                             start_date: datetime,
                             end_date: datetime,
                             data_frequency: str = '1min') -> List[IntradayBacktestResult]:
        """
        Run intraday backtesting using PSX DPS real-time data
        
        Args:
            symbols: List of PSX symbols to trade
            start_date: Start date for backtesting
            end_date: End date for backtesting
            data_frequency: '1min', '5min', or 'tick'
        """
        self.logger.info(f"Starting intraday backtest for {len(symbols)} symbols")
        self.logger.info(f"Period: {start_date.date()} to {end_date.date()}")
        
        # Generate trading dates (weekdays only)
        trading_dates = pd.date_range(start_date, end_date, freq='B')  # Business days only
        
        for trading_date in trading_dates:
            try:
                self.logger.info(f"Processing trading session: {trading_date.date()}")
                
                # Get intraday data for all symbols
                intraday_data = self._fetch_intraday_data(symbols, trading_date, data_frequency)
                
                if intraday_data.empty:
                    self.logger.warning(f"No data available for {trading_date.date()}")
                    continue
                
                # Run single day backtest
                day_result = self._run_single_day_backtest(
                    intraday_data, trading_date, symbols
                )
                
                if day_result:
                    self.results.append(day_result)
                    
            except Exception as e:
                self.logger.error(f"Error processing {trading_date.date()}: {str(e)}")
                continue
        
        self.logger.info(f"Backtest completed. Processed {len(self.results)} trading days.")
        return self.results

    def _fetch_intraday_data(self, symbols: List[str], trading_date: datetime, 
                           frequency: str) -> pd.DataFrame:
        """Fetch intraday data from PSX DPS API"""
        all_data = {}
        
        for symbol in symbols:
            try:
                # Get tick data from PSX DPS
                ticks_df = self.psx_fetcher.fetch_intraday_ticks(symbol)
                
                if ticks_df.empty:
                    continue

                # --- BUG FIX: Convert to and set DatetimeIndex ---
                ticks_df['datetime_pkt'] = pd.to_datetime(ticks_df['datetime_pkt'])
                ticks_df.set_index('datetime_pkt', inplace=True)
                # --- END FIX ---

                # Filter for trading date
                trading_day_data = ticks_df[
                    ticks_df.index.date == trading_date.date()
                ]
                
                if trading_day_data.empty:
                    continue
                
                # Resample based on frequency
                if frequency == 'tick':
                    resampled_data = trading_day_data
                elif frequency == '1min':
                    resampled_data = self._resample_to_minutes(trading_day_data, 1)
                elif frequency == '5min':
                    resampled_data = self._resample_to_minutes(trading_day_data, 5)
                else:
                    resampled_data = trading_day_data
                
                # Store OHLCV data
                all_data[symbol] = resampled_data
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all symbols into single DataFrame
        return self._combine_symbol_data(all_data)

    def _resample_to_minutes(self, tick_data: pd.DataFrame, minutes: int) -> pd.DataFrame:
        """Resample tick data to minute bars"""
        if tick_data.empty:
            return pd.DataFrame()
        
        # Create OHLCV bars
        ohlcv = tick_data['price'].resample(f'{minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        volume = tick_data['volume'].resample(f'{minutes}min').sum()
        
        # Combine OHLCV
        bars = pd.concat([ohlcv, volume.rename('volume')], axis=1)
        bars = bars.dropna()
        
        return bars

    def _combine_symbol_data(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple symbols into unified DataFrame"""
        if not symbol_data:
            return pd.DataFrame()
        
        # Get all unique timestamps
        all_timestamps = set()
        for data in symbol_data.values():
            all_timestamps.update(data.index)
        
        timestamps = sorted(all_timestamps)
        
        # Create combined DataFrame
        columns = []
        for symbol in symbol_data.keys():
            columns.extend([f'{symbol}_price', f'{symbol}_volume'])
        
        combined_df = pd.DataFrame(index=timestamps, columns=columns)
        
        for symbol, data in symbol_data.items():
            # Use close price if available, otherwise use price
            price_col = 'close' if 'close' in data.columns else 'price'
            combined_df[f'{symbol}_price'] = data[price_col]
            combined_df[f'{symbol}_volume'] = data['volume']
        
        # Forward fill missing values (within reason)
        combined_df = combined_df.fillna(method='ffill', limit=5)
        
        return combined_df

    def _run_single_day_backtest(self, intraday_data: pd.DataFrame, 
                               trading_date: datetime, symbols: List[str]) -> IntradayBacktestResult:
        """Run backtest for a single trading day"""
        
        # Initialize day trading variables
        cash = self.initial_capital
        positions = {}
        trades = []
        equity_curve = []
        signals_generated = 0
        signals_executed = 0
        
        # Get trading session bounds
        session_start = datetime.combine(trading_date.date(), self.trading_start_time)
        session_end = datetime.combine(trading_date.date(), self.trading_end_time)
        
        # Filter data to trading hours
        trading_data = intraday_data[
            (intraday_data.index >= session_start) & 
            (intraday_data.index <= session_end)
        ]
        
        if trading_data.empty:
            return None
        
        # Process each time point
        for timestamp in trading_data.index[::self.min_signal_interval]:  # Every N minutes
            current_data = trading_data.loc[:timestamp]  # All data up to this point
            current_prices = trading_data.loc[timestamp]
            
            # Generate signals for each symbol
            for symbol in symbols:
                try:
                    price_col = f'{symbol}_price'
                    volume_col = f'{symbol}_volume'
                    
                    if price_col not in current_prices or pd.isna(current_prices[price_col]):
                        continue
                    
                    current_price = current_prices[price_col]
                    current_volume = current_prices.get(volume_col, 0)
                    
                    # Get symbol historical data for signal generation
                    symbol_history = current_data[price_col].dropna()
                    
                    if len(symbol_history) < 20:  # Need minimum history
                        continue
                    
                    # Generate intraday signal
                    signal = self._generate_intraday_signal(
                        symbol, symbol_history, current_price, current_volume, timestamp
                    )
                    
                    signals_generated += 1
                    
                    if signal and signal.action in ['BUY', 'SELL']:
                        # Risk management check
                        if self._should_execute_signal(signal, positions, cash):
                            # Execute trade
                            trade_result = self._execute_intraday_trade(
                                signal, current_price, timestamp, cash, positions
                            )
                            
                            if trade_result:
                                trades.append(trade_result)
                                cash = trade_result['remaining_cash']
                                
                                # Update positions
                                if signal.action == 'BUY':
                                    positions[symbol] = {
                                        'shares': trade_result['shares'],
                                        'entry_price': current_price,
                                        'entry_time': timestamp,
                                        'stop_loss': signal.stop_loss,
                                        'take_profit': signal.take_profit
                                    }
                                elif signal.action == 'SELL' and symbol in positions:
                                    del positions[symbol]
                                
                                signals_executed += 1
                
                except Exception as e:
                    self.logger.debug(f"Error processing signal for {symbol}: {str(e)}")
                    continue
            
            # Check existing positions for exits
            positions_to_close = []
            for symbol, position in positions.items():
                price_col = f'{symbol}_price'
                if price_col in current_prices and not pd.isna(current_prices[price_col]):
                    current_price = current_prices[price_col]
                    
                    # Check stop loss / take profit / time exit
                    should_exit = self._should_exit_position(
                        position, current_price, timestamp
                    )
                    
                    if should_exit:
                        # Execute exit trade
                        exit_trade = self._execute_exit_trade(
                            symbol, position, current_price, timestamp
                        )
                        
                        if exit_trade:
                            trades.append(exit_trade)
                            cash += exit_trade['proceeds']
                            positions_to_close.append(symbol)
            
            # Remove closed positions
            for symbol in positions_to_close:
                if symbol in positions:
                    del positions[symbol]
            
            # Calculate current equity
            portfolio_value = cash
            for symbol, position in positions.items():
                price_col = f'{symbol}_price'
                if price_col in current_prices and not pd.isna(current_prices[price_col]):
                    current_price = current_prices[price_col]
                    portfolio_value += position['shares'] * current_price
            
            equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_value': portfolio_value - cash
            })
        
        # Close all remaining positions at session end
        final_prices = trading_data.iloc[-1]
        for symbol, position in positions.items():
            price_col = f'{symbol}_price'
            if price_col in final_prices and not pd.isna(final_prices[price_col]):
                final_price = final_prices[price_col]
                
                exit_trade = self._execute_exit_trade(
                    symbol, position, final_price, session_end, reason='SESSION_END'
                )
                
                if exit_trade:
                    trades.append(exit_trade)
                    cash += exit_trade['proceeds']
        
        # Calculate day results
        return self._calculate_day_metrics(
            trades, equity_curve, session_start, session_end, trading_date,
            symbols, signals_generated, signals_executed
        )

    def _generate_intraday_signal(self, symbol: str, price_history: pd.Series, 
                                current_price: float, current_volume: float,
                                timestamp: datetime) -> Optional[IntradaySignal]:
        """Generate intraday trading signal"""
        try:
            # Use existing signal analyzer with limited data
            signal = self.signal_analyzer.analyze_symbol(
                symbol, analysis_period_minutes=30
            )
            
            # Override with current data if signal analyzer fails
            if not signal:
                # Simple momentum signal
                if len(price_history) >= 10:
                    short_ma = price_history.tail(5).mean()
                    long_ma = price_history.tail(10).mean()
                    
                    if short_ma > long_ma * 1.002:  # 0.2% threshold
                        return IntradaySignal(
                            symbol=symbol,
                            timestamp=timestamp,
                            action='BUY',
                            confidence=0.6,
                            target_price=current_price * 1.01,
                            stop_loss=current_price * 0.995,
                            take_profit=current_price * 1.015
                        )
                    elif short_ma < long_ma * 0.998:
                        return IntradaySignal(
                            symbol=symbol,
                            timestamp=timestamp,
                            action='SELL',
                            confidence=0.6,
                            target_price=current_price * 0.99,
                            stop_loss=current_price * 1.005,
                            take_profit=current_price * 0.985
                        )
            
            return signal
            
        except Exception as e:
            self.logger.debug(f"Signal generation error for {symbol}: {str(e)}")
            return None

    def _should_execute_signal(self, signal: IntradaySignal, positions: Dict, cash: float) -> bool:
        """Risk management checks before executing signal"""
        
        # Don't trade if already have position in this symbol
        if signal.symbol in positions:
            return False
        
        # Check if we have enough cash
        position_size = min(cash * 0.1, 100000)  # Max 10% of cash or 100K PKR
        required_cash = position_size + (position_size * self.commission_rate)
        
        if required_cash > cash:
            return False
        
        # Check daily trade limit
        return True

    def _execute_intraday_trade(self, signal: IntradaySignal, price: float, 
                              timestamp: datetime, cash: float, positions: Dict) -> Optional[Dict]:
        """Execute intraday trade"""
        try:
            # Calculate position size
            position_value = min(cash * 0.1, 100000)  # Max 10% or 100K PKR
            shares = int(position_value / price)
            
            if shares <= 0:
                return None
            
            # Calculate costs
            trade_value = shares * price
            commission = trade_value * self.commission_rate
            slippage = trade_value * (self.slippage_bps / 10000)
            total_cost = trade_value + commission + slippage
            
            if total_cost > cash:
                return None
            
            return {
                'symbol': signal.symbol,
                'timestamp': timestamp,
                'action': signal.action,
                'shares': shares,
                'price': price,
                'trade_value': trade_value,
                'commission': commission,
                'slippage': slippage,
                'total_cost': total_cost,
                'remaining_cash': cash - total_cost,
                'signal_confidence': signal.confidence
            }
            
        except Exception as e:
            self.logger.debug(f"Trade execution error: {str(e)}")
            return None

    def _should_exit_position(self, position: Dict, current_price: float, 
                            timestamp: datetime) -> bool:
        """Check if position should be exited"""
        
        # Stop loss check
        if position['stop_loss'] and current_price <= position['stop_loss']:
            return True
        
        # Take profit check
        if position['take_profit'] and current_price >= position['take_profit']:
            return True
        
        # Time-based exit (max hold time)
        time_held = (timestamp - position['entry_time']).total_seconds() / 60
        if time_held > self.max_position_hold_time:
            return True
        
        return False

    def _execute_exit_trade(self, symbol: str, position: Dict, price: float, 
                          timestamp: datetime, reason: str = 'SIGNAL') -> Optional[Dict]:
        """Execute position exit"""
        try:
            shares = position['shares']
            trade_value = shares * price
            commission = trade_value * self.commission_rate
            slippage = trade_value * (self.slippage_bps / 10000)
            proceeds = trade_value - commission - slippage
            
            # Calculate P&L
            entry_cost = shares * position['entry_price']
            pnl = proceeds - entry_cost
            
            return {
                'symbol': symbol,
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'trade_value': trade_value,
                'commission': commission,
                'slippage': slippage,
                'proceeds': proceeds,
                'pnl': pnl,
                'entry_price': position['entry_price'],
                'hold_time_minutes': (timestamp - position['entry_time']).total_seconds() / 60,
                'exit_reason': reason
            }
            
        except Exception as e:
            self.logger.debug(f"Exit trade error: {str(e)}")
            return None

    def _calculate_day_metrics(self, trades: List[Dict], equity_curve: List[Dict],
                             session_start: datetime, session_end: datetime,
                             trading_date: datetime, symbols: List[str],
                             signals_generated: int, signals_executed: int) -> IntradayBacktestResult:
        """Calculate comprehensive day trading metrics"""
        
        if not trades:
            return IntradayBacktestResult(
                trading_date=trading_date,
                session_start=session_start,
                session_end=session_end,
                total_pnl=0,
                realized_pnl=0,
                unrealized_pnl=0,
                total_return=0,
                num_trades=0,
                win_rate=0,
                profit_factor=0,
                max_intraday_drawdown=0,
                sharpe_ratio=0,
                symbols_traded=[],
                average_hold_time_minutes=0,
                commission_paid=0,
                slippage_cost=0,
                trades=pd.DataFrame(),
                equity_curve=pd.Series(),
                positions=pd.DataFrame(),
                signals_generated=signals_generated,
                signals_executed=signals_executed
            )
        
        trades_df = pd.DataFrame(trades)
        
        # P&L calculations
        realized_pnl = trades_df[trades_df['action'] == 'SELL']['pnl'].sum() if 'pnl' in trades_df.columns else 0
        total_commission = trades_df['commission'].sum()
        total_slippage = trades_df['slippage'].sum()
        
        # Portfolio metrics
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            total_return = (equity_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
            
            # Intraday drawdown
            running_max = equity_df['portfolio_value'].expanding().max()
            drawdown = (equity_df['portfolio_value'] - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            total_return = 0
            max_drawdown = 0
        
        # Trade statistics
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        if not sell_trades.empty and 'pnl' in sell_trades.columns:
            winning_trades = sell_trades[sell_trades['pnl'] > 0]
            losing_trades = sell_trades[sell_trades['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            
            avg_hold_time = sell_trades['hold_time_minutes'].mean() if 'hold_time_minutes' in sell_trades.columns else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_hold_time = 0
        
        # Sharpe ratio (annualized for intraday)
        if not equity_df.empty and len(equity_df) > 1:
            returns = equity_df['portfolio_value'].pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 78)  # 78 5-min periods per day
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return IntradayBacktestResult(
            trading_date=trading_date,
            session_start=session_start,
            session_end=session_end,
            total_pnl=realized_pnl,
            realized_pnl=realized_pnl,
            unrealized_pnl=0,  # All positions closed at end of day
            total_return=total_return,
            num_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_intraday_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            symbols_traded=list(trades_df['symbol'].unique()) if not trades_df.empty else [],
            average_hold_time_minutes=avg_hold_time,
            commission_paid=total_commission,
            slippage_cost=total_slippage,
            trades=trades_df,
            equity_curve=pd.Series(equity_df['portfolio_value'].values, 
                                 index=equity_df['timestamp']) if not equity_df.empty else pd.Series(),
            positions=pd.DataFrame(),  # All closed at end of day
            signals_generated=signals_generated,
            signals_executed=signals_executed
        )

    def get_aggregate_performance(self) -> Dict[str, Any]:
        """Calculate aggregate performance across all trading days"""
        if not self.results:
            return {}
        
        total_pnl = sum(r.total_pnl for r in self.results)
        total_commission = sum(r.commission_paid for r in self.results)
        total_trades = sum(r.num_trades for r in self.results)
        
        profitable_days = len([r for r in self.results if r.total_pnl > 0])
        
        # Average metrics
        avg_daily_return = np.mean([r.total_return for r in self.results])
        avg_daily_trades = np.mean([r.num_trades for r in self.results])
        avg_win_rate = np.mean([r.win_rate for r in self.results if r.win_rate > 0])
        
        # Risk metrics
        daily_returns = [r.total_return for r in self.results]
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
        
        max_daily_loss = min(r.total_pnl for r in self.results)
        
        return {
            'total_pnl': total_pnl,
            'total_return': total_pnl / self.initial_capital,
            'annualized_return': avg_daily_return * 252,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'profitable_days': profitable_days,
            'total_days': len(self.results),
            'win_rate_days': profitable_days / len(self.results),
            'avg_daily_trades': avg_daily_trades,
            'avg_win_rate': avg_win_rate,
            'max_daily_loss': max_daily_loss,
            'commission_as_pct_pnl': total_commission / abs(total_pnl) if total_pnl != 0 else 0
        }

    def generate_intraday_report(self) -> str:
        """Generate comprehensive intraday trading report"""
        if not self.results:
            return "No intraday backtest results available."
        
        perf = self.get_aggregate_performance()
        
        report = f"""
=== PSX INTRADAY TRADING SYSTEM BACKTEST REPORT ===

AGGREGATE PERFORMANCE:
Total P&L: {perf.get('total_pnl', 0):,.0f} PKR
Total Return: {perf.get('total_return', 0):.2%}
Annualized Return: {perf.get('annualized_return', 0):.2%}
Volatility: {perf.get('volatility', 0):.2%}
Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}

TRADING STATISTICS:
Total Trading Days: {perf.get('total_days', 0)}
Profitable Days: {perf.get('profitable_days', 0)} ({perf.get('win_rate_days', 0):.1%})
Total Trades: {perf.get('total_trades', 0)}
Average Daily Trades: {perf.get('avg_daily_trades', 0):.1f}
Average Win Rate: {perf.get('avg_win_rate', 0):.1%}

COSTS:
Total Commission: {perf.get('total_commission', 0):,.0f} PKR
Commission as % of P&L: {perf.get('commission_as_pct_pnl', 0):.2%}
Max Daily Loss: {perf.get('max_daily_loss', 0):,.0f} PKR

DAILY BREAKDOWN:
"""
        
        for i, result in enumerate(self.results[:10]):  # Show first 10 days
            report += f"""
Day {i+1} ({result.trading_date.strftime('%Y-%m-%d')}):
  P&L: {result.total_pnl:,.0f} PKR | Return: {result.total_return:.2%}
  Trades: {result.num_trades} | Win Rate: {result.win_rate:.1%}
  Signals: {result.signals_generated} generated, {result.signals_executed} executed
  Max Drawdown: {result.max_intraday_drawdown:.2%}
"""
        
        if len(self.results) > 10:
            report += f"\n... and {len(self.results) - 10} more days"
        
        return report