"""
SMART RISK MANAGER - Correlation-Aware Position Management
===========================================================
Key Features:
- Sector exposure limits
- Correlation-based position sizing
- Dynamic stop management
- Portfolio heat monitoring
- Session-aware risk adjustment
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os


class RiskProfile(Enum):
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    sector: str
    entry_price: float
    current_price: float
    quantity: int
    position_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: float
    risk_amount: float
    max_loss_if_stopped: float
    days_held: int
    is_profitable: bool


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_capital: float
    cash_available: float
    total_invested: float
    total_exposure_pct: float
    unrealized_pnl: float
    realized_pnl_today: float
    total_positions: int
    sector_exposure: Dict[str, float]
    largest_position_pct: float
    portfolio_heat: float  # 0-100, higher = more risk
    daily_var_95: float  # Value at Risk
    max_drawdown: float
    can_take_new_position: bool
    position_limit_remaining: int


class SmartRiskManager:
    """
    Intelligent risk management with sector awareness and correlation limits
    """

    def __init__(self, initial_capital: float = 1000000,
                 risk_profile: RiskProfile = RiskProfile.MODERATE):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash_available = initial_capital
        self.risk_profile = risk_profile

        # Risk parameters by profile
        self.risk_params = {
            RiskProfile.CONSERVATIVE: {
                'max_position_size': 0.05,      # 5% max per position
                'max_sector_exposure': 0.15,    # 15% max per sector
                'max_total_exposure': 0.50,     # 50% max invested
                'max_positions': 8,
                'max_daily_loss': 0.02,         # 2% daily stop
                'max_correlated_exposure': 0.20,
                'min_cash_reserve': 0.30,       # Keep 30% cash
                'position_risk_pct': 0.01,      # 1% risk per trade
            },
            RiskProfile.MODERATE: {
                'max_position_size': 0.08,      # 8% max per position
                'max_sector_exposure': 0.25,    # 25% max per sector
                'max_total_exposure': 0.70,     # 70% max invested
                'max_positions': 12,
                'max_daily_loss': 0.03,         # 3% daily stop
                'max_correlated_exposure': 0.30,
                'min_cash_reserve': 0.20,       # Keep 20% cash
                'position_risk_pct': 0.02,      # 2% risk per trade
            },
            RiskProfile.AGGRESSIVE: {
                'max_position_size': 0.12,      # 12% max per position
                'max_sector_exposure': 0.35,    # 35% max per sector
                'max_total_exposure': 0.85,     # 85% max invested
                'max_positions': 15,
                'max_daily_loss': 0.05,         # 5% daily stop
                'max_correlated_exposure': 0.40,
                'min_cash_reserve': 0.10,       # Keep 10% cash
                'position_risk_pct': 0.03,      # 3% risk per trade
            }
        }

        # PSX Sector classifications
        self.sectors = {
            'BANKS': ['HBL', 'UBL', 'MCB', 'ABL', 'NBP', 'BAFL', 'BAHL', 'MEBL', 'AKBL', 'BOP', 'SNBL', 'JSBL'],
            'CEMENT': ['LUCK', 'DGKC', 'MLCF', 'FCCL', 'KOHC', 'PIOC', 'CHCC', 'ACPL', 'GWLC', 'BWCL'],
            'FERTILIZER': ['FFC', 'EFERT', 'FFBL', 'ENGRO', 'FATIMA'],
            'OIL_GAS': ['PSO', 'OGDC', 'PPL', 'POL', 'MARI', 'SSGC', 'SNGP', 'APL', 'HASCOL'],
            'POWER': ['HUBC', 'KEL', 'NCPL', 'NPL', 'KAPCO', 'PKGP', 'SPWL'],
            'TEXTILE': ['NML', 'NCL', 'GATM', 'ILP', 'KTML', 'ANL'],
            'TECH': ['TRG', 'SYS', 'NETSOL', 'AVN'],
            'AUTO': ['INDU', 'PSMC', 'HCAR', 'MTL', 'GHNL', 'ATLH'],
            'PHARMA': ['SEARL', 'GLAXO', 'FEROZ', 'ABOT', 'AGP', 'HINOON'],
            'FOOD': ['NESTLE', 'UNITY', 'FFL', 'QUICE', 'TREET'],
            'CHEMICALS': ['EPCL', 'ICI', 'LOTCHEM', 'BOC'],
        }

        # Sector correlations (simplified - higher value = more correlated)
        self.sector_correlations = {
            ('BANKS', 'BANKS'): 1.0,
            ('BANKS', 'CEMENT'): 0.3,
            ('BANKS', 'FERTILIZER'): 0.2,
            ('CEMENT', 'CEMENT'): 1.0,
            ('CEMENT', 'OIL_GAS'): 0.4,
            ('OIL_GAS', 'OIL_GAS'): 1.0,
            ('OIL_GAS', 'POWER'): 0.6,
            ('POWER', 'POWER'): 1.0,
            ('TECH', 'TECH'): 1.0,
        }

        # Active positions
        self.positions: Dict[str, dict] = {}
        self.trade_history: List[dict] = []
        self.daily_pnl = 0.0

        # Session timing
        self.market_open = time(9, 30)
        self.market_close = time(15, 30)

        # Load saved state if exists
        self._load_state()

    def get_sector(self, symbol: str) -> str:
        """Get sector for a symbol"""
        for sector, symbols in self.sectors.items():
            if symbol in symbols:
                return sector
        return 'OTHER'

    def get_sector_correlation(self, sector1: str, sector2: str) -> float:
        """Get correlation between two sectors"""
        key = (sector1, sector2)
        if key in self.sector_correlations:
            return self.sector_correlations[key]
        key_rev = (sector2, sector1)
        if key_rev in self.sector_correlations:
            return self.sector_correlations[key_rev]
        return 0.2  # Default low correlation

    def calculate_position_size(self, symbol: str, entry_price: float,
                               stop_loss: float, signal_confidence: float) -> Dict:
        """
        Calculate optimal position size considering all risk factors
        """
        params = self.risk_params[self.risk_profile]
        sector = self.get_sector(symbol)

        result = {
            'symbol': symbol,
            'recommended_shares': 0,
            'position_value': 0,
            'risk_amount': 0,
            'position_pct': 0,
            'can_trade': True,
            'warnings': [],
            'blockers': []
        }

        # Check if we already have this position
        if symbol in self.positions:
            result['blockers'].append(f"Already have position in {symbol}")
            result['can_trade'] = False
            return result

        # Check position count
        if len(self.positions) >= params['max_positions']:
            result['blockers'].append(f"Max positions reached ({params['max_positions']})")
            result['can_trade'] = False
            return result

        # Check daily loss limit
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital
            if daily_loss_pct >= params['max_daily_loss']:
                result['blockers'].append(f"Daily loss limit hit ({daily_loss_pct:.1%})")
                result['can_trade'] = False
                return result

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            result['blockers'].append("Invalid stop loss")
            result['can_trade'] = False
            return result

        # Base position size from risk management
        max_risk_amount = self.current_capital * params['position_risk_pct']
        base_shares = int(max_risk_amount / risk_per_share)

        # Apply confidence adjustment (higher confidence = larger position)
        confidence_mult = 0.5 + (signal_confidence / 100) * 0.5  # 0.5 to 1.0
        adjusted_shares = int(base_shares * confidence_mult)

        # Check max position size
        max_position_value = self.current_capital * params['max_position_size']
        if adjusted_shares * entry_price > max_position_value:
            adjusted_shares = int(max_position_value / entry_price)
            result['warnings'].append(f"Reduced to max position size ({params['max_position_size']*100:.0f}%)")

        # Check sector exposure
        current_sector_exposure = self._get_sector_exposure(sector)
        new_position_value = adjusted_shares * entry_price
        new_sector_exposure = (current_sector_exposure + new_position_value) / self.current_capital

        if new_sector_exposure > params['max_sector_exposure']:
            # Reduce position to fit within sector limit
            allowed_additional = (params['max_sector_exposure'] * self.current_capital) - current_sector_exposure
            if allowed_additional <= 0:
                result['blockers'].append(f"Sector exposure limit reached ({sector})")
                result['can_trade'] = False
                return result

            adjusted_shares = int(allowed_additional / entry_price)
            result['warnings'].append(f"Reduced due to {sector} sector limit")

        # Check correlated exposure
        correlated_exposure = self._get_correlated_exposure(sector)
        if (correlated_exposure + new_position_value) / self.current_capital > params['max_correlated_exposure']:
            result['warnings'].append("High correlation with existing positions")
            adjusted_shares = int(adjusted_shares * 0.7)  # Reduce by 30%

        # Check total exposure
        total_invested = sum(p['quantity'] * p['current_price'] for p in self.positions.values())
        new_total_exposure = (total_invested + adjusted_shares * entry_price) / self.current_capital

        if new_total_exposure > params['max_total_exposure']:
            allowed_additional = (params['max_total_exposure'] * self.current_capital) - total_invested
            if allowed_additional <= 0:
                result['blockers'].append("Total exposure limit reached")
                result['can_trade'] = False
                return result

            adjusted_shares = int(allowed_additional / entry_price)
            result['warnings'].append("Reduced due to total exposure limit")

        # Check cash reserve
        min_cash = self.current_capital * params['min_cash_reserve']
        available_for_trade = self.cash_available - min_cash

        if adjusted_shares * entry_price > available_for_trade:
            adjusted_shares = int(available_for_trade / entry_price)
            result['warnings'].append("Reduced to maintain cash reserve")

        # Session-based adjustment
        session_mult = self._get_session_multiplier()
        adjusted_shares = int(adjusted_shares * session_mult)

        # Final checks
        if adjusted_shares <= 0:
            result['blockers'].append("Position too small after adjustments")
            result['can_trade'] = False
            return result

        # Calculate final values
        position_value = adjusted_shares * entry_price
        risk_amount = adjusted_shares * risk_per_share

        result['recommended_shares'] = adjusted_shares
        result['position_value'] = position_value
        result['risk_amount'] = risk_amount
        result['position_pct'] = (position_value / self.current_capital) * 100
        result['can_trade'] = len(result['blockers']) == 0

        return result

    def _get_sector_exposure(self, sector: str) -> float:
        """Get current exposure to a sector"""
        exposure = 0
        for symbol, pos in self.positions.items():
            if self.get_sector(symbol) == sector:
                exposure += pos['quantity'] * pos['current_price']
        return exposure

    def _get_correlated_exposure(self, new_sector: str) -> float:
        """Get exposure to sectors correlated with new_sector"""
        exposure = 0
        for symbol, pos in self.positions.items():
            pos_sector = self.get_sector(symbol)
            correlation = self.get_sector_correlation(new_sector, pos_sector)
            if correlation > 0.3:  # Consider correlated if > 0.3
                exposure += pos['quantity'] * pos['current_price'] * correlation
        return exposure

    def _get_session_multiplier(self) -> float:
        """Get position size multiplier based on market session"""
        current_time = datetime.now().time()

        # Opening volatility (first 30 mins)
        if time(9, 30) <= current_time < time(10, 0):
            return 0.7  # Reduce size

        # Lunch session (low volume)
        if time(12, 30) <= current_time < time(13, 30):
            return 0.8

        # Last 30 mins (closing volatility)
        if time(15, 0) <= current_time < time(15, 30):
            return 0.7

        # Prime trading hours
        if time(10, 0) <= current_time < time(12, 30):
            return 1.0

        # Afternoon momentum
        if time(13, 30) <= current_time < time(15, 0):
            return 0.95

        return 1.0

    def enter_position(self, symbol: str, entry_price: float, quantity: int,
                      stop_loss: float, take_profit: float,
                      signal_confidence: float, signal_reasons: List[str]) -> Dict:
        """
        Enter a new position with full tracking
        """
        if symbol in self.positions:
            return {'success': False, 'error': f'Already have position in {symbol}'}

        position_value = quantity * entry_price
        if position_value > self.cash_available:
            return {'success': False, 'error': 'Insufficient cash'}

        sector = self.get_sector(symbol)

        position = {
            'symbol': symbol,
            'sector': sector,
            'entry_price': entry_price,
            'current_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now().isoformat(),
            'signal_confidence': signal_confidence,
            'signal_reasons': signal_reasons,
            'unrealized_pnl': 0,
            'max_price': entry_price,
            'min_price': entry_price,
        }

        self.positions[symbol] = position
        self.cash_available -= position_value

        # Log trade
        self.trade_history.append({
            'action': 'BUY',
            'symbol': symbol,
            'price': entry_price,
            'quantity': quantity,
            'value': position_value,
            'timestamp': datetime.now().isoformat(),
            'confidence': signal_confidence
        })

        self._save_state()

        return {
            'success': True,
            'position': position,
            'cash_remaining': self.cash_available
        }

    def update_position(self, symbol: str, current_price: float) -> Optional[Dict]:
        """
        Update position with current price, check for stop/target hits
        Returns exit signal if triggered
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pos['current_price'] = current_price

        # Track high/low
        pos['max_price'] = max(pos['max_price'], current_price)
        pos['min_price'] = min(pos['min_price'], current_price)

        # Calculate P&L
        pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['quantity']

        # Check stop loss
        if current_price <= pos['stop_loss']:
            return {
                'action': 'EXIT',
                'reason': 'STOP_LOSS',
                'symbol': symbol,
                'exit_price': current_price
            }

        # Check take profit
        if current_price >= pos['take_profit']:
            return {
                'action': 'EXIT',
                'reason': 'TAKE_PROFIT',
                'symbol': symbol,
                'exit_price': current_price
            }

        # Update trailing stop if in profit
        if pos['unrealized_pnl'] > 0:
            self._update_trailing_stop(symbol, current_price)

        return None

    def _update_trailing_stop(self, symbol: str, current_price: float):
        """Update trailing stop for profitable position"""
        pos = self.positions[symbol]
        profit_pct = (current_price - pos['entry_price']) / pos['entry_price']

        # Start trailing after 3% profit
        if profit_pct > 0.03:
            # Trail at 2% below current price
            new_stop = current_price * 0.98

            # Only move stop up, never down
            if new_stop > pos['stop_loss']:
                pos['stop_loss'] = new_stop

    def exit_position(self, symbol: str, exit_price: float, reason: str) -> Dict:
        """
        Exit a position and calculate P&L
        """
        if symbol not in self.positions:
            return {'success': False, 'error': f'No position in {symbol}'}

        pos = self.positions[symbol]

        # Calculate P&L
        realized_pnl = (exit_price - pos['entry_price']) * pos['quantity']
        position_value = exit_price * pos['quantity']

        # Update capital
        self.cash_available += position_value
        self.daily_pnl += realized_pnl

        # Log trade
        self.trade_history.append({
            'action': 'SELL',
            'symbol': symbol,
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'quantity': pos['quantity'],
            'realized_pnl': realized_pnl,
            'return_pct': (exit_price / pos['entry_price'] - 1) * 100,
            'reason': reason,
            'hold_time': pos['entry_time'],
            'timestamp': datetime.now().isoformat()
        })

        # Remove position
        del self.positions[symbol]

        self._save_state()

        return {
            'success': True,
            'symbol': symbol,
            'realized_pnl': realized_pnl,
            'return_pct': (exit_price / pos['entry_price'] - 1) * 100,
            'reason': reason
        }

    def get_portfolio_risk(self) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics
        """
        params = self.risk_params[self.risk_profile]

        # Calculate sector exposure
        sector_exposure = {}
        for symbol, pos in self.positions.items():
            sector = pos['sector']
            value = pos['quantity'] * pos['current_price']
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value

        # Total invested
        total_invested = sum(p['quantity'] * p['current_price'] for p in self.positions.values())
        total_exposure_pct = (total_invested / self.current_capital) * 100 if self.current_capital > 0 else 0

        # Largest position
        largest_position = 0
        if self.positions:
            largest_position = max(
                p['quantity'] * p['current_price'] for p in self.positions.values()
            ) / self.current_capital * 100

        # Unrealized P&L
        unrealized_pnl = sum(p['unrealized_pnl'] for p in self.positions.values())

        # Portfolio heat (composite risk score)
        heat = self._calculate_portfolio_heat()

        # Simple VaR estimate (95% daily)
        portfolio_volatility = 0.02  # Assume 2% daily volatility
        var_95 = total_invested * portfolio_volatility * 1.65

        # Can take new position?
        can_trade = (
            len(self.positions) < params['max_positions'] and
            total_exposure_pct / 100 < params['max_total_exposure'] and
            (self.daily_pnl >= 0 or abs(self.daily_pnl) / self.initial_capital < params['max_daily_loss'])
        )

        return PortfolioRisk(
            total_capital=self.current_capital,
            cash_available=self.cash_available,
            total_invested=total_invested,
            total_exposure_pct=total_exposure_pct,
            unrealized_pnl=unrealized_pnl,
            realized_pnl_today=self.daily_pnl,
            total_positions=len(self.positions),
            sector_exposure=sector_exposure,
            largest_position_pct=largest_position,
            portfolio_heat=heat,
            daily_var_95=var_95,
            max_drawdown=0,  # Would need historical data
            can_take_new_position=can_trade,
            position_limit_remaining=params['max_positions'] - len(self.positions)
        )

    def _calculate_portfolio_heat(self) -> float:
        """
        Calculate portfolio heat score (0-100)
        Higher = more risk
        """
        if not self.positions:
            return 0

        params = self.risk_params[self.risk_profile]
        heat = 0

        # Factor 1: Position count (25 points max)
        position_ratio = len(self.positions) / params['max_positions']
        heat += position_ratio * 25

        # Factor 2: Exposure (25 points max)
        total_invested = sum(p['quantity'] * p['current_price'] for p in self.positions.values())
        exposure_ratio = (total_invested / self.current_capital) / params['max_total_exposure']
        heat += min(1, exposure_ratio) * 25

        # Factor 3: Concentration (25 points max)
        if self.positions:
            values = [p['quantity'] * p['current_price'] for p in self.positions.values()]
            max_position = max(values) / self.current_capital
            concentration_ratio = max_position / params['max_position_size']
            heat += min(1, concentration_ratio) * 25

        # Factor 4: Sector concentration (25 points max)
        sector_values = {}
        for pos in self.positions.values():
            sector = pos['sector']
            value = pos['quantity'] * pos['current_price']
            sector_values[sector] = sector_values.get(sector, 0) + value

        if sector_values:
            max_sector = max(sector_values.values()) / self.current_capital
            sector_ratio = max_sector / params['max_sector_exposure']
            heat += min(1, sector_ratio) * 25

        return min(100, heat)

    def get_position_risks(self) -> List[PositionRisk]:
        """Get risk metrics for all positions"""
        risks = []
        for symbol, pos in self.positions.items():
            entry_time = datetime.fromisoformat(pos['entry_time'])
            days_held = (datetime.now() - entry_time).days

            position_value = pos['quantity'] * pos['current_price']
            unrealized_pnl_pct = (pos['current_price'] / pos['entry_price'] - 1) * 100
            max_loss = pos['quantity'] * (pos['entry_price'] - pos['stop_loss'])

            risks.append(PositionRisk(
                symbol=symbol,
                sector=pos['sector'],
                entry_price=pos['entry_price'],
                current_price=pos['current_price'],
                quantity=pos['quantity'],
                position_value=position_value,
                unrealized_pnl=pos['unrealized_pnl'],
                unrealized_pnl_pct=unrealized_pnl_pct,
                stop_loss=pos['stop_loss'],
                risk_amount=max_loss,
                max_loss_if_stopped=max_loss,
                days_held=days_held,
                is_profitable=pos['unrealized_pnl'] > 0
            ))

        return risks

    def _save_state(self):
        """Save current state to file"""
        state = {
            'positions': self.positions,
            'cash_available': self.cash_available,
            'daily_pnl': self.daily_pnl,
            'trade_history': self.trade_history[-100:],  # Keep last 100 trades
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open('risk_manager_state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except:
            pass

    def _load_state(self):
        """Load saved state from file"""
        try:
            if os.path.exists('risk_manager_state.json'):
                with open('risk_manager_state.json', 'r') as f:
                    state = json.load(f)

                # Check if state is from today
                saved_date = datetime.fromisoformat(state['timestamp']).date()
                if saved_date == datetime.now().date():
                    self.positions = state.get('positions', {})
                    self.cash_available = state.get('cash_available', self.initial_capital)
                    self.daily_pnl = state.get('daily_pnl', 0)
                    self.trade_history = state.get('trade_history', [])
                else:
                    # New day - reset daily P&L but keep positions
                    self.positions = state.get('positions', {})
                    self.trade_history = state.get('trade_history', [])
                    self.daily_pnl = 0
                    # Recalculate cash
                    invested = sum(p['quantity'] * p['entry_price'] for p in self.positions.values())
                    self.cash_available = self.current_capital - invested
        except:
            pass

    def reset_daily(self):
        """Reset daily metrics (call at start of trading day)"""
        self.daily_pnl = 0
        self._save_state()


if __name__ == "__main__":
    print("Testing Smart Risk Manager...")

    rm = SmartRiskManager(initial_capital=1000000, risk_profile=RiskProfile.MODERATE)

    # Test position sizing
    sizing = rm.calculate_position_size(
        symbol='HBL',
        entry_price=250,
        stop_loss=242,
        signal_confidence=75
    )

    print(f"\nPosition Sizing for HBL:")
    print(f"  Recommended Shares: {sizing['recommended_shares']}")
    print(f"  Position Value: {sizing['position_value']:,.0f} PKR")
    print(f"  Risk Amount: {sizing['risk_amount']:,.0f} PKR")
    print(f"  Position %: {sizing['position_pct']:.1f}%")
    print(f"  Can Trade: {sizing['can_trade']}")
    if sizing['warnings']:
        print(f"  Warnings: {', '.join(sizing['warnings'])}")

    # Test portfolio risk
    risk = rm.get_portfolio_risk()
    print(f"\nPortfolio Risk:")
    print(f"  Total Capital: {risk.total_capital:,.0f} PKR")
    print(f"  Cash Available: {risk.cash_available:,.0f} PKR")
    print(f"  Portfolio Heat: {risk.portfolio_heat:.0f}/100")
    print(f"  Can Take New Position: {risk.can_take_new_position}")
