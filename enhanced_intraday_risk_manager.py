"""
ENHANCED INTRADAY RISK MANAGEMENT SYSTEM
Advanced Risk Controls for High-Accuracy Intraday Trading

Features:
- Dynamic position sizing based on volatility and confidence
- Daily, weekly, and monthly loss limits
- Real-time risk monitoring and circuit breakers
- Volatility-adjusted stop losses
- PSX-specific risk controls
- Portfolio-level risk management
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    symbol: str
    timestamp: datetime
    
    # Position risk
    position_size: float
    max_position_size: float
    risk_per_trade: float
    
    # Portfolio risk
    total_exposure: float
    available_capital: float
    portfolio_risk: float
    
    # Daily limits
    daily_pnl: float
    daily_loss_limit: float
    daily_profit_target: float
    
    # Volatility risk
    current_volatility: float
    volatility_percentile: float
    vol_adjusted_size: float
    
    # Market risk
    market_session_risk: float
    liquidity_risk: float
    concentration_risk: float

@dataclass
class RiskSignal:
    """Risk management signal"""
    symbol: str
    action: str  # ALLOW, REDUCE, BLOCK, EMERGENCY_EXIT
    reason: str
    recommended_size: float
    risk_score: float
    warnings: List[str]

class EnhancedIntradayRiskManager:
    """Advanced risk management for intraday trading"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk parameters
        self.risk_params = {
            # Position sizing
            'max_risk_per_trade': 0.02,  # 2% max risk per trade
            'base_position_size': 0.05,  # 5% base position size
            'max_position_size': 0.15,   # 15% max position per symbol
            'max_portfolio_exposure': 0.80,  # 80% max total exposure
            
            # Daily limits
            'daily_loss_limit': 0.05,    # 5% daily loss limit
            'daily_profit_target': 0.10, # 10% daily profit target
            'max_trades_per_day': 100,   # Max trades per day
            'max_trades_per_symbol': 10, # Max trades per symbol per day
            
            # Volatility adjustments
            'vol_lookback_days': 20,     # Volatility calculation period
            'high_vol_threshold': 1.5,   # High volatility threshold
            'low_vol_threshold': 0.7,    # Low volatility threshold
            'vol_adjustment_factor': 0.5, # Volatility adjustment strength
            
            # Market session risk
            'opening_risk_multiplier': 1.5,  # Higher risk during opening
            'closing_risk_multiplier': 1.3,  # Higher risk during closing
            'lunch_risk_multiplier': 0.8,    # Lower risk during lunch
            
            # PSX specific
            'min_liquidity_threshold': 1000,  # Minimum daily volume
            'illiquid_size_reduction': 0.5,   # Size reduction for illiquid stocks
            'concentration_limit': 0.25,      # Max concentration per sector
        }
        
        # State tracking
        self.positions = {}          # Current positions
        self.daily_pnl = 0.0        # Daily P&L
        self.daily_trades = 0       # Number of trades today
        self.symbol_trades = {}     # Trades per symbol
        self.portfolio_exposure = 0.0 # Total portfolio exposure
        
        # Risk history
        self.risk_history = []
        self.volatility_cache = {}
        
        # Emergency stop
        self.emergency_stop = False
        self.emergency_reason = ""
        
        # PSX market hours
        self.psx_hours = {
            'open': time(9, 15),
            'close': time(15, 30),
            'lunch_start': time(12, 30),
            'lunch_end': time(13, 30)
        }
    
    def evaluate_trade_risk(self, symbol: str, signal_strength: float, 
                           current_price: float, proposed_size: float,
                           market_data: pd.DataFrame) -> RiskSignal:
        """Evaluate risk for a proposed trade"""
        
        # Check emergency stop
        if self.emergency_stop:
            return RiskSignal(
                symbol=symbol,
                action='BLOCK',
                reason=f'Emergency stop: {self.emergency_reason}',
                recommended_size=0.0,
                risk_score=100.0,
                warnings=['Emergency stop active']
            )
        
        warnings = []
        risk_score = 0.0
        
        # 1. Check daily limits
        daily_check = self._check_daily_limits()
        if not daily_check['allowed']:
            return RiskSignal(
                symbol=symbol,
                action='BLOCK',
                reason=daily_check['reason'],
                recommended_size=0.0,
                risk_score=100.0,
                warnings=[daily_check['reason']]
            )
        
        # 2. Check position limits
        position_check = self._check_position_limits(symbol, proposed_size)
        if not position_check['allowed']:
            if position_check['max_allowed'] > 0:
                warnings.append(f"Position size reduced: {position_check['reason']}")
                proposed_size = position_check['max_allowed']
            else:
                return RiskSignal(
                    symbol=symbol,
                    action='BLOCK',
                    reason=position_check['reason'],
                    recommended_size=0.0,
                    risk_score=100.0,
                    warnings=[position_check['reason']]
                )
        
        # 3. Volatility adjustment
        vol_metrics = self._calculate_volatility_metrics(symbol, market_data)
        vol_adjusted_size = self._adjust_size_for_volatility(proposed_size, vol_metrics)
        
        if vol_adjusted_size < proposed_size:
            warnings.append(f"Size reduced due to high volatility: {vol_metrics['volatility_percentile']:.1f}%")
            proposed_size = vol_adjusted_size
            risk_score += 10
        
        # 4. Market session risk
        session_risk = self._assess_market_session_risk()
        session_adjusted_size = proposed_size * session_risk['multiplier']
        
        if session_adjusted_size < proposed_size:
            warnings.append(f"Size reduced due to market session risk: {session_risk['reason']}")
            proposed_size = session_adjusted_size
            risk_score += 5
        
        # 5. Liquidity risk
        liquidity_risk = self._assess_liquidity_risk(symbol, market_data)
        if liquidity_risk['high_risk']:
            liquidity_adjusted_size = proposed_size * self.risk_params['illiquid_size_reduction']
            warnings.append(f"Size reduced due to low liquidity: {liquidity_risk['reason']}")
            proposed_size = liquidity_adjusted_size
            risk_score += 15
        
        # 6. Signal strength adjustment
        confidence_adjusted_size = self._adjust_size_for_confidence(proposed_size, signal_strength)
        if confidence_adjusted_size < proposed_size:
            warnings.append(f"Size reduced due to low signal confidence: {signal_strength:.1f}%")
            proposed_size = confidence_adjusted_size
            risk_score += 5
        
        # 7. Portfolio concentration risk
        concentration_check = self._check_concentration_risk(symbol, proposed_size)
        if not concentration_check['allowed']:
            if concentration_check['max_allowed'] > 0:
                warnings.append(f"Size reduced due to concentration risk: {concentration_check['reason']}")
                proposed_size = concentration_check['max_allowed']
                risk_score += 10
            else:
                return RiskSignal(
                    symbol=symbol,
                    action='BLOCK',
                    reason=concentration_check['reason'],
                    recommended_size=0.0,
                    risk_score=100.0,
                    warnings=[concentration_check['reason']]
                )
        
        # Determine final action
        if risk_score >= 50:
            action = 'REDUCE'
        elif risk_score >= 30:
            action = 'ALLOW'
        else:
            action = 'ALLOW'
        
        # Final size check
        min_size = self.initial_capital * 0.001  # 0.1% minimum
        if proposed_size < min_size:
            action = 'BLOCK'
            proposed_size = 0.0
            warnings.append('Position size below minimum threshold')
        
        return RiskSignal(
            symbol=symbol,
            action=action,
            reason='Risk assessment completed',
            recommended_size=proposed_size,
            risk_score=risk_score,
            warnings=warnings
        )
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                           direction: str, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate volatility-adjusted stop loss and take profit"""
        
        # Calculate ATR for volatility
        atr = self._calculate_atr(market_data)
        if atr == 0:
            atr = entry_price * 0.02  # 2% fallback
        
        # Volatility metrics
        vol_metrics = self._calculate_volatility_metrics(symbol, market_data)
        vol_multiplier = 1.0
        
        if vol_metrics['volatility_percentile'] > 80:
            vol_multiplier = 1.5  # Wider stops in high vol
        elif vol_metrics['volatility_percentile'] < 20:
            vol_multiplier = 0.7  # Tighter stops in low vol
        
        # Market session adjustment
        session_risk = self._assess_market_session_risk()
        session_multiplier = session_risk.get('stop_multiplier', 1.0)
        
        # Calculate stops
        stop_distance = atr * vol_multiplier * session_multiplier
        
        if direction.upper() == 'BUY':
            stop_loss = entry_price - (stop_distance * 2)
            take_profit = entry_price + (stop_distance * 3)
        else:  # SELL
            stop_loss = entry_price + (stop_distance * 2)
            take_profit = entry_price - (stop_distance * 3)
        
        return stop_loss, take_profit
    
    def update_portfolio_state(self, symbol: str, action: str, size: float, 
                              price: float, pnl: float = 0.0):
        """Update portfolio state after trade execution"""
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = {'size': 0, 'avg_price': 0, 'unrealized_pnl': 0}
        
        if action.upper() in ['BUY', 'SELL']:
            self.positions[symbol]['size'] += size if action.upper() == 'BUY' else -size
            self.positions[symbol]['avg_price'] = price  # Simplified
            
            # Update trade counters
            self.daily_trades += 1
            self.symbol_trades[symbol] = self.symbol_trades.get(symbol, 0) + 1
        
        # Update P&L
        if pnl != 0:
            self.daily_pnl += pnl
            self.current_capital += pnl
        
        # Recalculate portfolio exposure
        self._update_portfolio_exposure()
        
        # Check emergency conditions
        self._check_emergency_conditions()
    
    def get_risk_metrics(self, symbol: str) -> RiskMetrics:
        """Get comprehensive risk metrics"""
        
        current_time = datetime.now()
        
        # Position risk
        position = self.positions.get(symbol, {'size': 0, 'avg_price': 0})
        position_value = abs(position['size']) * position['avg_price']
        max_position_value = self.current_capital * self.risk_params['max_position_size']
        
        # Portfolio risk
        available_capital = self.current_capital - self.portfolio_exposure
        portfolio_risk_ratio = self.portfolio_exposure / self.current_capital if self.current_capital > 0 else 0
        
        # Daily risk
        daily_loss_limit = self.current_capital * self.risk_params['daily_loss_limit']
        daily_profit_target = self.current_capital * self.risk_params['daily_profit_target']
        
        return RiskMetrics(
            symbol=symbol,
            timestamp=current_time,
            position_size=position_value,
            max_position_size=max_position_value,
            risk_per_trade=self.risk_params['max_risk_per_trade'],
            total_exposure=self.portfolio_exposure,
            available_capital=available_capital,
            portfolio_risk=portfolio_risk_ratio,
            daily_pnl=self.daily_pnl,
            daily_loss_limit=daily_loss_limit,
            daily_profit_target=daily_profit_target,
            current_volatility=0.0,  # Would be calculated from market data
            volatility_percentile=50.0,  # Would be calculated
            vol_adjusted_size=position_value,
            market_session_risk=self._assess_market_session_risk()['risk_level'],
            liquidity_risk=0.0,  # Would be calculated
            concentration_risk=portfolio_risk_ratio
        )
    
    def _check_daily_limits(self) -> Dict[str, Union[bool, str]]:
        """Check daily trading limits"""
        
        # P&L limits
        loss_limit = self.initial_capital * self.risk_params['daily_loss_limit']
        profit_target = self.initial_capital * self.risk_params['daily_profit_target']
        
        if self.daily_pnl <= -loss_limit:
            return {'allowed': False, 'reason': f'Daily loss limit reached: {self.daily_pnl:.2f}'}
        
        if self.daily_pnl >= profit_target:
            return {'allowed': False, 'reason': f'Daily profit target reached: {self.daily_pnl:.2f}'}
        
        # Trade count limits
        if self.daily_trades >= self.risk_params['max_trades_per_day']:
            return {'allowed': False, 'reason': f'Daily trade limit reached: {self.daily_trades}'}
        
        return {'allowed': True, 'reason': 'Daily limits OK'}
    
    def _check_position_limits(self, symbol: str, proposed_size: float) -> Dict[str, Union[bool, str, float]]:
        """Check position size limits"""
        
        current_position = abs(self.positions.get(symbol, {'size': 0})['size'])
        max_position = self.current_capital * self.risk_params['max_position_size']
        
        if current_position + proposed_size > max_position:
            max_allowed = max(0, max_position - current_position)
            return {
                'allowed': max_allowed > 0,
                'reason': f'Position limit exceeded for {symbol}',
                'max_allowed': max_allowed
            }
        
        # Portfolio exposure check
        max_exposure = self.current_capital * self.risk_params['max_portfolio_exposure']
        if self.portfolio_exposure + proposed_size > max_exposure:
            max_allowed = max(0, max_exposure - self.portfolio_exposure)
            return {
                'allowed': max_allowed > 0,
                'reason': 'Portfolio exposure limit exceeded',
                'max_allowed': max_allowed
            }
        
        return {'allowed': True, 'reason': 'Position limits OK', 'max_allowed': proposed_size}
    
    def _calculate_volatility_metrics(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics for risk adjustment"""
        
        if market_data.empty or len(market_data) < 20:
            return {'current_vol': 0.02, 'volatility_percentile': 50.0, 'vol_regime': 'normal'}
        
        # Calculate returns
        returns = market_data['Close'].pct_change().dropna()
        
        if len(returns) < 2:
            return {'current_vol': 0.02, 'volatility_percentile': 50.0, 'vol_regime': 'normal'}
        
        # Current volatility (20-period)
        current_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized
        
        # Historical volatility percentile
        if len(returns) >= 60:
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            vol_percentile = (rolling_vol <= current_vol).mean() * 100
        else:
            vol_percentile = 50.0
        
        # Volatility regime
        if vol_percentile > 80:
            vol_regime = 'high'
        elif vol_percentile < 20:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        return {
            'current_vol': current_vol,
            'volatility_percentile': vol_percentile,
            'vol_regime': vol_regime
        }
    
    def _adjust_size_for_volatility(self, base_size: float, vol_metrics: Dict[str, float]) -> float:
        """Adjust position size based on volatility"""
        
        vol_percentile = vol_metrics['volatility_percentile']
        
        if vol_percentile > 80:  # High volatility
            adjustment = 1 - (self.risk_params['vol_adjustment_factor'] * 0.5)
        elif vol_percentile < 20:  # Low volatility
            adjustment = 1 + (self.risk_params['vol_adjustment_factor'] * 0.3)
        else:  # Normal volatility
            adjustment = 1.0
        
        return base_size * adjustment
    
    def _assess_market_session_risk(self) -> Dict[str, Union[float, str]]:
        """Assess risk based on market session"""
        
        current_time = datetime.now().time()
        
        # Opening session (high volatility)
        if self.psx_hours['open'] <= current_time <= time(9, 45):
            return {
                'risk_level': 3,
                'multiplier': 1 / self.risk_params['opening_risk_multiplier'],
                'stop_multiplier': self.risk_params['opening_risk_multiplier'],
                'reason': 'Opening session - high volatility'
            }
        
        # Closing session (high volatility)
        elif time(15, 0) <= current_time <= self.psx_hours['close']:
            return {
                'risk_level': 3,
                'multiplier': 1 / self.risk_params['closing_risk_multiplier'],
                'stop_multiplier': self.risk_params['closing_risk_multiplier'],
                'reason': 'Closing session - high volatility'
            }
        
        # Lunch time (low liquidity)
        elif self.psx_hours['lunch_start'] <= current_time <= self.psx_hours['lunch_end']:
            return {
                'risk_level': 2,
                'multiplier': self.risk_params['lunch_risk_multiplier'],
                'stop_multiplier': 1.2,
                'reason': 'Lunch session - low liquidity'
            }
        
        # Normal session
        else:
            return {
                'risk_level': 1,
                'multiplier': 1.0,
                'stop_multiplier': 1.0,
                'reason': 'Normal trading session'
            }
    
    def _assess_liquidity_risk(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Union[bool, str]]:
        """Assess liquidity risk for the symbol"""
        
        if market_data.empty:
            return {'high_risk': True, 'reason': 'No market data available'}
        
        # Average volume over last 20 periods
        avg_volume = market_data['Volume'].tail(20).mean()
        
        if avg_volume < self.risk_params['min_liquidity_threshold']:
            return {
                'high_risk': True,
                'reason': f'Low liquidity: avg volume {avg_volume:.0f} < threshold {self.risk_params["min_liquidity_threshold"]}'
            }
        
        # Check for volume spikes or drops
        recent_volume = market_data['Volume'].iloc[-1]
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio < 0.3:  # Very low current volume
            return {
                'high_risk': True,
                'reason': f'Current volume too low: {volume_ratio:.2f}x average'
            }
        
        return {'high_risk': False, 'reason': 'Liquidity adequate'}
    
    def _adjust_size_for_confidence(self, base_size: float, signal_strength: float) -> float:
        """Adjust position size based on signal confidence"""
        
        # Signal strength should be 0-100
        confidence_ratio = signal_strength / 100.0
        
        # Scale position size with confidence
        if confidence_ratio >= 0.8:
            return base_size  # Full size for high confidence
        elif confidence_ratio >= 0.6:
            return base_size * 0.8  # 80% size for medium confidence
        elif confidence_ratio >= 0.4:
            return base_size * 0.6  # 60% size for low confidence
        else:
            return base_size * 0.3  # 30% size for very low confidence
    
    def _check_concentration_risk(self, symbol: str, proposed_size: float) -> Dict[str, Union[bool, str, float]]:
        """Check portfolio concentration risk"""
        
        # Simplified sector mapping (in practice, this would come from a database)
        sector_map = {
            'HBL': 'banking', 'UBL': 'banking', 'MCB': 'banking',
            'ENGRO': 'chemical', 'FFC': 'fertilizer',
            'LUCK': 'cement', 'MLCF': 'cement'
        }
        
        symbol_sector = sector_map.get(symbol, 'other')
        
        # Calculate current sector exposure
        sector_exposure = 0
        for pos_symbol, position in self.positions.items():
            if sector_map.get(pos_symbol, 'other') == symbol_sector:
                sector_exposure += abs(position['size']) * position['avg_price']
        
        max_sector_exposure = self.current_capital * self.risk_params['concentration_limit']
        
        if sector_exposure + proposed_size > max_sector_exposure:
            max_allowed = max(0, max_sector_exposure - sector_exposure)
            return {
                'allowed': max_allowed > 0,
                'reason': f'Sector concentration limit exceeded for {symbol_sector}',
                'max_allowed': max_allowed
            }
        
        return {'allowed': True, 'reason': 'Concentration limits OK', 'max_allowed': proposed_size}
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        
        if market_data.empty or len(market_data) < period:
            return 0.0
        
        try:
            import ta
            atr = ta.volatility.average_true_range(
                market_data['High'], 
                market_data['Low'], 
                market_data['Close'], 
                window=period
            )
            return atr.iloc[-1] if not atr.empty else 0.0
        except:
            # Fallback calculation
            high_low = market_data['High'] - market_data['Low']
            high_close = abs(market_data['High'] - market_data['Close'].shift(1))
            low_close = abs(market_data['Low'] - market_data['Close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.tail(period).mean()
    
    def _update_portfolio_exposure(self):
        """Update total portfolio exposure"""
        
        total_exposure = 0
        for symbol, position in self.positions.items():
            total_exposure += abs(position['size']) * position['avg_price']
        
        self.portfolio_exposure = total_exposure
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        
        # Extreme loss condition
        loss_threshold = self.initial_capital * 0.10  # 10% emergency threshold
        if self.daily_pnl <= -loss_threshold:
            self.emergency_stop = True
            self.emergency_reason = f"Emergency loss threshold breached: {self.daily_pnl:.2f}"
            return
        
        # System stability check
        if self.current_capital <= self.initial_capital * 0.8:  # 20% capital loss
            self.emergency_stop = True
            self.emergency_reason = f"Critical capital loss: {self.current_capital:.2f}"
            return
        
        # Portfolio exposure check
        if self.portfolio_exposure > self.current_capital * 1.2:  # 120% exposure
            self.emergency_stop = True
            self.emergency_reason = "Excessive portfolio exposure"
            return
    
    def reset_daily_limits(self):
        """Reset daily counters (call at market open)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.symbol_trades = {}
        self.emergency_stop = False
        self.emergency_reason = ""
    
    def save_risk_state(self, filepath: str = "risk_state.json"):
        """Save current risk state"""
        state = {
            'current_capital': self.current_capital,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'positions': self.positions,
            'portfolio_exposure': self.portfolio_exposure,
            'emergency_stop': self.emergency_stop,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_risk_state(self, filepath: str = "risk_state.json"):
        """Load risk state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_capital = state.get('current_capital', self.initial_capital)
            self.daily_pnl = state.get('daily_pnl', 0.0)
            self.daily_trades = state.get('daily_trades', 0)
            self.positions = state.get('positions', {})
            self.portfolio_exposure = state.get('portfolio_exposure', 0.0)
            self.emergency_stop = state.get('emergency_stop', False)
            
        except FileNotFoundError:
            print("No existing risk state found, starting fresh")
        except Exception as e:
            print(f"Error loading risk state: {e}")

# Testing function
def test_risk_manager():
    """Test the enhanced risk management system"""
    print("🧪 Testing Enhanced Intraday Risk Manager...")
    
    # Initialize risk manager
    risk_manager = EnhancedIntradayRiskManager(initial_capital=1000000)
    
    # Create sample market data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.uniform(95, 105, 100),
        'High': np.random.uniform(105, 110, 100),
        'Low': np.random.uniform(90, 95, 100),
        'Close': np.random.uniform(95, 105, 100),
        'Volume': np.random.randint(10000, 100000, 100)
    }, index=dates)
    
    # Test trade evaluation
    print("📊 Testing trade risk evaluation...")
    risk_signal = risk_manager.evaluate_trade_risk(
        symbol='HBL',
        signal_strength=75.0,
        current_price=100.0,
        proposed_size=50000,
        market_data=sample_data
    )
    
    print(f"   Action: {risk_signal.action}")
    print(f"   Recommended Size: ${risk_signal.recommended_size:,.2f}")
    print(f"   Risk Score: {risk_signal.risk_score:.1f}")
    print(f"   Warnings: {risk_signal.warnings}")
    
    # Test stop loss calculation
    print("\n📉 Testing stop loss calculation...")
    stop_loss, take_profit = risk_manager.calculate_stop_loss(
        symbol='HBL',
        entry_price=100.0,
        direction='BUY',
        market_data=sample_data
    )
    
    print(f"   Entry Price: $100.00")
    print(f"   Stop Loss: ${stop_loss:.2f}")
    print(f"   Take Profit: ${take_profit:.2f}")
    print(f"   Risk/Reward: {(100-stop_loss)/(take_profit-100):.2f}")
    
    # Test portfolio update
    print("\n💼 Testing portfolio state update...")
    risk_manager.update_portfolio_state(
        symbol='HBL',
        action='BUY',
        size=risk_signal.recommended_size,
        price=100.0,
        pnl=0.0
    )
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics('HBL')
    print(f"   Position Size: ${metrics.position_size:,.2f}")
    print(f"   Portfolio Exposure: ${metrics.total_exposure:,.2f}")
    print(f"   Available Capital: ${metrics.available_capital:,.2f}")
    print(f"   Daily P&L: ${metrics.daily_pnl:,.2f}")
    
    print("\n✅ Enhanced Intraday Risk Manager test completed!")

if __name__ == "__main__":
    test_risk_manager()