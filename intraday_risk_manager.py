#!/usr/bin/env python3
"""
Intraday Risk Management System for PSX Trading
===============================================

Advanced risk management for high-frequency intraday trading using PSX DPS data.
Features:
- Dynamic position sizing based on volatility
- Real-time stop loss and take profit management
- Risk exposure monitoring
- Portfolio heat maps
- Trade timing optimization
- Maximum drawdown protection
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, field
from enum import Enum
import json

from psx_dps_fetcher import PSXDPSFetcher
from intraday_signal_analyzer import IntradaySignal, SignalType

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"

@dataclass
class Position:
    """Active trading position"""
    symbol: str
    entry_price: float
    current_price: float
    quantity: int
    position_type: str  # 'LONG' or 'SHORT'
    entry_time: dt.datetime
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    risk_amount: float = 0.0
    trailing_stop: Optional[float] = None

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_capital: float
    available_capital: float
    used_capital: float
    total_exposure: float
    max_position_size: float
    current_drawdown: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    positions_count: int
    daily_pnl: float

class IntradayRiskManager:
    """Comprehensive intraday risk management system"""
    
    def __init__(self, initial_capital: float = 1000000, risk_level: RiskLevel = RiskLevel.MODERATE):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_level = risk_level
        self.fetcher = PSXDPSFetcher()
        
        # Risk parameters by level
        self.risk_params = {
            RiskLevel.CONSERVATIVE: {
                'max_position_risk': 0.01,  # 1% per position
                'max_portfolio_risk': 0.05,  # 5% total
                'max_correlation_exposure': 0.15,  # 15% in correlated positions
                'max_daily_loss': 0.03,  # 3% daily stop
                'position_size_multiplier': 0.5
            },
            RiskLevel.MODERATE: {
                'max_position_risk': 0.02,  # 2% per position
                'max_portfolio_risk': 0.08,  # 8% total
                'max_correlation_exposure': 0.25,  # 25% in correlated positions
                'max_daily_loss': 0.05,  # 5% daily stop
                'position_size_multiplier': 1.0
            },
            RiskLevel.AGGRESSIVE: {
                'max_position_risk': 0.03,  # 3% per position
                'max_portfolio_risk': 0.12,  # 12% total
                'max_correlation_exposure': 0.35,  # 35% in correlated positions
                'max_daily_loss': 0.08,  # 8% daily stop
                'position_size_multiplier': 1.5
            }
        }
        
        # Active positions and trade history
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.daily_max_loss = 0.0
        
        # Market hours for PSX (9:30 AM - 3:30 PM PKT)
        self.market_open = dt.time(9, 30)
        self.market_close = dt.time(15, 30)
        
    def calculate_position_size(self, signal: IntradaySignal, 
                               current_volatility: float = None) -> Dict[str, any]:
        """
        Calculate optimal position size based on signal and risk parameters
        
        Args:
            signal: IntradaySignal from analyzer
            current_volatility: Current price volatility (optional)
            
        Returns:
            Dictionary with position size recommendations
        """
        try:
            params = self.risk_params[self.risk_level]
            
            # Base risk amount per trade
            max_risk_amount = self.current_capital * params['max_position_risk']
            
            # Calculate risk per share
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                risk_per_share = abs(signal.entry_price - signal.stop_loss)
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                risk_per_share = abs(signal.stop_loss - signal.entry_price)
            else:
                return {'recommended_quantity': 0, 'risk_amount': 0, 'reason': 'Hold signal'}
            
            if risk_per_share <= 0:
                return {'recommended_quantity': 0, 'risk_amount': 0, 'reason': 'Invalid risk parameters'}
            
            # Basic position size
            base_quantity = int(max_risk_amount / risk_per_share)
            
            # Adjust for signal confidence
            confidence_multiplier = signal.confidence / 100.0
            adjusted_quantity = int(base_quantity * confidence_multiplier)
            
            # Adjust for volatility if provided
            if current_volatility:
                volatility_adjustment = min(1.0, 2.0 / max(current_volatility, 0.5))
                adjusted_quantity = int(adjusted_quantity * volatility_adjustment)
            
            # Apply risk level multiplier
            final_quantity = int(adjusted_quantity * params['position_size_multiplier'])
            
            # Position value limits (max 10% in one position for safety)
            max_position_value = self.current_capital * 0.10
            if final_quantity * signal.entry_price > max_position_value:
                final_quantity = int(max_position_value / signal.entry_price)
            
            # Ensure minimum viable quantity
            final_quantity = max(final_quantity, 1) if final_quantity > 0 else 0
            
            # Calculate final position value and risk
            position_value = final_quantity * signal.entry_price
            final_risk_amount = final_quantity * risk_per_share
            
            return {
                'recommended_quantity': final_quantity,
                'risk_amount': final_risk_amount,
                'position_value': position_value,
                'risk_percentage': (final_risk_amount / self.current_capital) * 100,
                'confidence_adjusted': True,
                'volatility_adjusted': current_volatility is not None,
                'max_quantity_possible': base_quantity
            }
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return {'recommended_quantity': 0, 'risk_amount': 0, 'reason': f'Calculation error: {e}'}
    
    def validate_trade_entry(self, signal: IntradaySignal, quantity: int) -> Dict[str, any]:
        """
        Validate if trade entry is safe under current risk parameters
        
        Args:
            signal: IntradaySignal to validate
            quantity: Proposed quantity
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'blocking_issues': [],
            'recommendations': []
        }
        
        try:
            params = self.risk_params[self.risk_level]
            
            # Check if market is open
            if not self._is_market_open():
                validation['blocking_issues'].append("Market is closed")
                validation['is_valid'] = False
            
            # Check daily loss limit
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital
            if self.daily_pnl < 0 and daily_loss_pct >= params['max_daily_loss']:
                validation['blocking_issues'].append(f"Daily loss limit exceeded: {daily_loss_pct:.1%}")
                validation['is_valid'] = False
            
            # Check if symbol already has position
            if signal.symbol in self.positions:
                validation['warnings'].append(f"Already have position in {signal.symbol}")
            
            # Check portfolio exposure
            position_value = quantity * signal.entry_price
            total_exposure = sum(pos.quantity * pos.current_price for pos in self.positions.values())
            total_exposure += position_value
            
            exposure_pct = total_exposure / self.current_capital
            if exposure_pct > params['max_portfolio_risk'] * 2:  # 2x buffer for exposure
                validation['blocking_issues'].append(f"Portfolio exposure too high: {exposure_pct:.1%}")
                validation['is_valid'] = False
            elif exposure_pct > params['max_portfolio_risk']:
                validation['warnings'].append(f"High portfolio exposure: {exposure_pct:.1%}")
            
            # Check position concentration
            position_pct = position_value / self.current_capital
            if position_pct > 0.25:  # More than 25% in single position
                validation['blocking_issues'].append(f"Position too large: {position_pct:.1%} of capital")
                validation['is_valid'] = False
            elif position_pct > 0.15:
                validation['warnings'].append(f"Large position size: {position_pct:.1%}")
            
            # Check available capital
            if position_value > self.current_capital * 0.5:
                validation['warnings'].append("Using large portion of available capital")
            
            # Check signal quality
            if signal.confidence < 60:
                validation['warnings'].append(f"Low signal confidence: {signal.confidence:.1f}%")
            
            if signal.risk_reward_ratio < 1.5:
                validation['warnings'].append(f"Low risk/reward ratio: {signal.risk_reward_ratio:.2f}")
            
            # Time-based checks
            current_time = dt.datetime.now().time()
            
            # Avoid trades in first 15 minutes (opening volatility)
            early_trading_end = dt.time(9, 45)
            if current_time < early_trading_end:
                validation['warnings'].append("Early trading session - higher volatility")
            
            # Avoid trades in last 30 minutes (closing effects)
            late_trading_start = dt.time(15, 0)
            if current_time > late_trading_start:
                validation['warnings'].append("Late trading session - prepare for close")
            
            # Recommendations
            if validation['warnings'] and validation['is_valid']:
                validation['recommendations'].append("Consider reducing position size due to warnings")
            
            if signal.confidence > 80 and signal.risk_reward_ratio > 2.0:
                validation['recommendations'].append("High-quality signal - consider full position")
            
        except Exception as e:
            logger.error(f"Trade validation error: {e}")
            validation['blocking_issues'].append(f"Validation error: {e}")
            validation['is_valid'] = False
        
        return validation
    
    def enter_position(self, signal: IntradaySignal, quantity: int) -> Dict[str, any]:
        """
        Enter a new position with full risk management
        
        Args:
            signal: IntradaySignal to trade
            quantity: Quantity to trade
            
        Returns:
            Dictionary with position entry results
        """
        try:
            # Validate trade first
            validation = self.validate_trade_entry(signal, quantity)
            if not validation['is_valid']:
                return {
                    'success': False,
                    'reason': '; '.join(validation['blocking_issues']),
                    'validation': validation
                }
            
            # Calculate position details
            position_value = quantity * signal.entry_price
            position_type = 'LONG' if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else 'SHORT'
            
            # Risk amount
            if position_type == 'LONG':
                risk_amount = quantity * abs(signal.entry_price - signal.stop_loss)
            else:
                risk_amount = quantity * abs(signal.stop_loss - signal.entry_price)
            
            # Create position
            position = Position(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                quantity=quantity,
                position_type=position_type,
                entry_time=dt.datetime.now(),
                stop_loss=signal.stop_loss,
                take_profit=signal.target_price,
                risk_amount=risk_amount
            )
            
            # Add to positions
            self.positions[signal.symbol] = position
            
            # Update capital
            self.current_capital -= position_value  # Assuming we're using margin/cash
            
            # Log trade
            trade_log = {
                'action': 'ENTER',
                'symbol': signal.symbol,
                'quantity': quantity,
                'price': signal.entry_price,
                'position_type': position_type,
                'timestamp': dt.datetime.now().isoformat(),
                'risk_amount': risk_amount,
                'signal_confidence': signal.confidence,
                'reasoning': signal.reasoning
            }
            self.trade_history.append(trade_log)
            
            return {
                'success': True,
                'position': position,
                'trade_log': trade_log,
                'validation': validation,
                'capital_remaining': self.current_capital
            }
            
        except Exception as e:
            logger.error(f"Position entry error: {e}")
            return {
                'success': False,
                'reason': f'Entry error: {e}',
                'validation': {}
            }
    
    def update_positions(self) -> Dict[str, any]:
        """Update all positions with current market prices and manage risk"""
        
        update_results = {
            'positions_updated': 0,
            'stops_triggered': [],
            'targets_hit': [],
            'trailing_stops_adjusted': [],
            'errors': []
        }
        
        try:
            for symbol, position in list(self.positions.items()):
                try:
                    # Get current price
                    current_data = self.fetcher.fetch_real_time_data(symbol)
                    if not current_data:
                        continue
                    
                    current_price = current_data['price']
                    position.current_price = current_price
                    
                    # Calculate P&L
                    if position.position_type == 'LONG':
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                        
                        # Check for stop loss
                        if current_price <= position.stop_loss:
                            exit_result = self._exit_position(position, current_price, 'STOP_LOSS')
                            update_results['stops_triggered'].append(exit_result)
                            continue
                        
                        # Check for take profit
                        if current_price >= position.take_profit:
                            exit_result = self._exit_position(position, current_price, 'TAKE_PROFIT')
                            update_results['targets_hit'].append(exit_result)
                            continue
                        
                        # Update trailing stop
                        trailing_result = self._update_trailing_stop(position, current_price)
                        if trailing_result:
                            update_results['trailing_stops_adjusted'].append(trailing_result)
                    
                    else:  # SHORT position
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                        
                        # Check for stop loss
                        if current_price >= position.stop_loss:
                            exit_result = self._exit_position(position, current_price, 'STOP_LOSS')
                            update_results['stops_triggered'].append(exit_result)
                            continue
                        
                        # Check for take profit
                        if current_price <= position.take_profit:
                            exit_result = self._exit_position(position, current_price, 'TAKE_PROFIT')
                            update_results['targets_hit'].append(exit_result)
                            continue
                    
                    # Track max favorable/adverse movement
                    if position.position_type == 'LONG':
                        favorable_move = current_price - position.entry_price
                        position.max_favorable = max(position.max_favorable, favorable_move)
                        position.max_adverse = min(position.max_adverse, favorable_move)
                    else:
                        favorable_move = position.entry_price - current_price
                        position.max_favorable = max(position.max_favorable, favorable_move)
                        position.max_adverse = min(position.max_adverse, favorable_move)
                    
                    update_results['positions_updated'] += 1
                    
                except Exception as e:
                    logger.error(f"Error updating position {symbol}: {e}")
                    update_results['errors'].append(f"{symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Position update error: {e}")
            update_results['errors'].append(f"General error: {e}")
        
        return update_results
    
    def _exit_position(self, position: Position, exit_price: float, exit_reason: str) -> Dict:
        """Exit a position and calculate final P&L"""
        
        try:
            # Calculate final P&L
            if position.position_type == 'LONG':
                realized_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                realized_pnl = (position.entry_price - exit_price) * position.quantity
            
            position.realized_pnl = realized_pnl
            
            # Update capital
            position_value = position.quantity * exit_price
            self.current_capital += position_value
            
            # Update daily P&L
            self.daily_pnl += realized_pnl
            
            # Log trade
            trade_log = {
                'action': 'EXIT',
                'symbol': position.symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'realized_pnl': realized_pnl,
                'exit_reason': exit_reason,
                'position_type': position.position_type,
                'timestamp': dt.datetime.now().isoformat(),
                'holding_period_minutes': (dt.datetime.now() - position.entry_time).total_seconds() / 60
            }
            self.trade_history.append(trade_log)
            
            # Remove from active positions
            del self.positions[position.symbol]
            
            return {
                'symbol': position.symbol,
                'realized_pnl': realized_pnl,
                'exit_reason': exit_reason,
                'trade_log': trade_log
            }
            
        except Exception as e:
            logger.error(f"Position exit error: {e}")
            return {'symbol': position.symbol, 'error': str(e)}
    
    def _update_trailing_stop(self, position: Position, current_price: float) -> Optional[Dict]:
        """Update trailing stop loss for profitable positions"""
        
        if position.position_type != 'LONG':
            return None  # Only implement for long positions for now
        
        # Only trail if position is profitable
        if current_price <= position.entry_price:
            return None
        
        # Calculate trailing stop (2% below current high)
        trailing_distance = current_price * 0.02
        new_trailing_stop = current_price - trailing_distance
        
        # Only update if new stop is higher than current stop
        if position.trailing_stop is None or new_trailing_stop > position.trailing_stop:
            old_stop = position.trailing_stop
            position.trailing_stop = new_trailing_stop
            position.stop_loss = new_trailing_stop  # Update actual stop loss
            
            return {
                'symbol': position.symbol,
                'old_trailing_stop': old_stop,
                'new_trailing_stop': new_trailing_stop,
                'current_price': current_price
            }
        
        return None
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        try:
            # Position metrics
            total_exposure = sum(pos.quantity * pos.current_price for pos in self.positions.values())
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # Capital metrics
            used_capital = self.initial_capital - self.current_capital
            available_capital = self.current_capital
            
            # Calculate trade statistics
            completed_trades = [t for t in self.trade_history if t['action'] == 'EXIT']
            
            if completed_trades:
                pnls = [t['realized_pnl'] for t in completed_trades]
                wins = [pnl for pnl in pnls if pnl > 0]
                losses = [pnl for pnl in pnls if pnl < 0]
                
                win_rate = len(wins) / len(pnls) if pnls else 0
                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                profit_factor = abs(sum(wins) / sum(losses)) if losses else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            # Drawdown calculation
            running_capital = self.initial_capital
            max_capital = self.initial_capital
            max_drawdown = 0
            
            for trade in completed_trades:
                running_capital += trade['realized_pnl']
                max_capital = max(max_capital, running_capital)
                current_drawdown = (max_capital - running_capital) / max_capital
                max_drawdown = max(max_drawdown, current_drawdown)
            
            current_drawdown = (max_capital - (self.current_capital + total_unrealized_pnl)) / max_capital
            
            return RiskMetrics(
                total_capital=self.initial_capital,
                available_capital=available_capital,
                used_capital=used_capital,
                total_exposure=total_exposure,
                max_position_size=self.risk_params[self.risk_level]['max_position_risk'] * self.current_capital,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                sharpe_ratio=0.0,  # Would need more data for proper calculation
                positions_count=len(self.positions),
                daily_pnl=self.daily_pnl + total_unrealized_pnl
            )
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return RiskMetrics(
                total_capital=self.initial_capital,
                available_capital=self.current_capital,
                used_capital=0,
                total_exposure=0,
                max_position_size=0,
                current_drawdown=0,
                max_drawdown=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                sharpe_ratio=0,
                positions_count=len(self.positions),
                daily_pnl=0
            )
    
    def _is_market_open(self) -> bool:
        """Check if PSX market is currently open"""
        now = dt.datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close
    
    def get_position_summary(self) -> Dict[str, any]:
        """Get summary of all active positions"""
        
        if not self.positions:
            return {'message': 'No active positions'}
        
        summary = {
            'active_positions': len(self.positions),
            'total_unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
            'positions': []
        }
        
        for symbol, pos in self.positions.items():
            pos_summary = {
                'symbol': symbol,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'quantity': pos.quantity,
                'position_type': pos.position_type,
                'unrealized_pnl': pos.unrealized_pnl,
                'pnl_pct': (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'risk_amount': pos.risk_amount,
                'holding_minutes': (dt.datetime.now() - pos.entry_time).total_seconds() / 60
            }
            summary['positions'].append(pos_summary)
        
        return summary

# Test function
def test_risk_manager():
    """Test the intraday risk management system"""
    print("🚀 Testing Intraday Risk Management System")
    print("=" * 55)
    
    # Create risk manager
    risk_manager = IntradayRiskManager(
        initial_capital=500000,  # 5 lakh PKR
        risk_level=RiskLevel.MODERATE
    )
    
    print(f"💰 Initial Capital: {risk_manager.initial_capital:,} PKR")
    print(f"⚖️ Risk Level: {risk_manager.risk_level.value}")
    
    # Test position sizing
    from intraday_signal_analyzer import IntradaySignalAnalyzer
    
    analyzer = IntradaySignalAnalyzer()
    
    print(f"\n📊 Testing Position Sizing...")
    
    # Analyze UBL for testing
    signal = analyzer.analyze_symbol('UBL', analysis_period_minutes=20)
    
    print(f"   Signal: {signal.signal_type.value} ({signal.confidence:.1f}%)")
    print(f"   Entry: {signal.entry_price:.2f} PKR")
    print(f"   Target: {signal.target_price:.2f} PKR") 
    print(f"   Stop: {signal.stop_loss:.2f} PKR")
    
    # Calculate position size
    sizing = risk_manager.calculate_position_size(signal)
    
    if sizing['recommended_quantity'] > 0:
        print(f"   Recommended Quantity: {sizing['recommended_quantity']:,} shares")
        print(f"   Position Value: {sizing['position_value']:,.0f} PKR")
        print(f"   Risk Amount: {sizing['risk_amount']:,.0f} PKR")
        print(f"   Risk %: {sizing['risk_percentage']:.2f}%")
        
        # Test trade validation
        print(f"\n🔍 Testing Trade Validation...")
        validation = risk_manager.validate_trade_entry(signal, sizing['recommended_quantity'])
        
        print(f"   Valid Trade: {validation['is_valid']}")
        if validation['warnings']:
            print(f"   Warnings: {'; '.join(validation['warnings'])}")
        if validation['blocking_issues']:
            print(f"   Issues: {'; '.join(validation['blocking_issues'])}")
        if validation['recommendations']:
            print(f"   Tips: {'; '.join(validation['recommendations'])}")
        
        # Test position entry if valid
        if validation['is_valid']:
            print(f"\n📈 Testing Position Entry...")
            entry_result = risk_manager.enter_position(signal, sizing['recommended_quantity'])
            
            if entry_result['success']:
                print(f"   ✅ Position entered successfully")
                print(f"   Remaining Capital: {entry_result['capital_remaining']:,.0f} PKR")
                
                # Test position updates
                print(f"\n🔄 Testing Position Updates...")
                update_result = risk_manager.update_positions()
                print(f"   Positions Updated: {update_result['positions_updated']}")
                
                # Get position summary
                summary = risk_manager.get_position_summary()
                print(f"   Active Positions: {summary['active_positions']}")
                print(f"   Unrealized P&L: {summary['total_unrealized_pnl']:,.0f} PKR")
                
            else:
                print(f"   ❌ Position entry failed: {entry_result['reason']}")
    else:
        print(f"   ❌ No position recommended: {sizing.get('reason', 'Unknown')}")
    
    # Test risk metrics
    print(f"\n📊 Testing Risk Metrics...")
    metrics = risk_manager.get_risk_metrics()
    
    print(f"   Available Capital: {metrics.available_capital:,.0f} PKR")
    print(f"   Total Exposure: {metrics.total_exposure:,.0f} PKR")
    print(f"   Active Positions: {metrics.positions_count}")
    print(f"   Daily P&L: {metrics.daily_pnl:,.0f} PKR")
    print(f"   Win Rate: {metrics.win_rate:.1%}")
    
    print(f"\n🏁 Risk management system test completed!")

if __name__ == "__main__":
    test_risk_manager()