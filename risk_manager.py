#!/usr/bin/env python3
"""
Professional Risk Management System
===================================

Advanced risk management with dynamic position sizing, multi-timeframe analysis,
and comprehensive portfolio risk controls.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config_manager import get_config

logger = logging.getLogger(__name__)

@dataclass
class PositionSize:
    """Position sizing calculation result"""
    shares: int
    investment_amount: float
    risk_amount: float
    risk_percentage: float
    stop_loss_price: float
    position_value: float
    warnings: List[str]

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a position"""
    var_1day: float  # Value at Risk (1 day, 95% confidence)
    var_1week: float  # Value at Risk (1 week, 95% confidence)
    expected_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float  # Relative to market/index

class RiskManager:
    """Professional risk management system"""
    
    def __init__(self, account_size: float = None):
        self.account_size = account_size or get_config('risk_management.default_account_size', 100000)
        self.max_position_risk = get_config('risk_management.max_position_risk_pct', 5.0)
        self.max_portfolio_risk = get_config('risk_management.max_portfolio_risk_pct', 20.0)
        
        logger.info(f"Risk Manager initialized with account size: {self.account_size:,.0f}")
    
    def calculate_position_size(self, 
                              current_price: float,
                              stop_loss_price: float,
                              risk_percentage: float = None,
                              volatility: float = None) -> PositionSize:
        """
        Calculate optimal position size using multiple methods
        """
        risk_pct = risk_percentage or get_config('risk_management.default_account_risk_pct', 2.0)
        warnings = []
        
        # Method 1: Fixed Risk Percentage
        risk_amount = self.account_size * (risk_pct / 100)
        price_risk = abs(current_price - stop_loss_price)
        
        if price_risk <= 0:
            warnings.append("Invalid stop loss - using 2% default risk")
            price_risk = current_price * 0.02
            stop_loss_price = current_price - price_risk
        
        shares_fixed_risk = int(risk_amount / price_risk)
        
        # Method 2: Volatility-based sizing (Kelly Criterion approximation)
        if volatility and volatility > 0:
            # Assume 55% win rate, 1.5:1 reward:risk ratio
            win_rate = 0.55
            avg_win = 1.5 * price_risk
            avg_loss = price_risk
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            shares_volatility = int((self.account_size * kelly_fraction) / current_price)
        else:
            shares_volatility = shares_fixed_risk
        
        # Use the more conservative approach
        shares = min(shares_fixed_risk, shares_volatility)
        
        # Position size limits
        max_position_value = self.account_size * (self.max_position_risk / 100)
        max_shares_by_value = int(max_position_value / current_price)
        
        if shares > max_shares_by_value:
            shares = max_shares_by_value
            warnings.append(f"Position size limited to {self.max_position_risk}% of account")
        
        # Minimum position check
        min_investment = get_config('risk_management.min_investment', 1000)
        if shares * current_price < min_investment:
            shares = max(1, int(min_investment / current_price))
            warnings.append(f"Position increased to minimum investment of {min_investment}")
        
        investment_amount = shares * current_price
        actual_risk = shares * price_risk
        actual_risk_pct = (actual_risk / self.account_size) * 100
        
        return PositionSize(
            shares=shares,
            investment_amount=investment_amount,
            risk_amount=actual_risk,
            risk_percentage=actual_risk_pct,
            stop_loss_price=stop_loss_price,
            position_value=investment_amount,
            warnings=warnings
        )
    
    def multi_timeframe_analysis(self, daily_data: pd.DataFrame, weekly_data: pd.DataFrame = None) -> Dict:
        """
        Analyze signals across multiple timeframes
        """
        results = {
            'daily_trend': 'neutral',
            'weekly_trend': 'neutral',
            'alignment': False,
            'strength': 0,
            'recommendation': 'hold'
        }
        
        if len(daily_data) < 50:
            return results
        
        # Daily analysis
        daily_ma20 = daily_data['Close'].rolling(20).mean()
        daily_ma50 = daily_data['Close'].rolling(50).mean()
        
        current_price = daily_data['Close'].iloc[-1]
        daily_trend_score = 0
        
        if current_price > daily_ma20.iloc[-1]:
            daily_trend_score += 1
        if current_price > daily_ma50.iloc[-1]:
            daily_trend_score += 1
        if daily_ma20.iloc[-1] > daily_ma50.iloc[-1]:
            daily_trend_score += 1
        
        if daily_trend_score >= 2:
            results['daily_trend'] = 'bullish'
        elif daily_trend_score <= 1:
            results['daily_trend'] = 'bearish'
        
        # Weekly analysis (if provided)
        if weekly_data is not None and len(weekly_data) >= 20:
            weekly_ma10 = weekly_data['Close'].rolling(10).mean()
            weekly_ma20 = weekly_data['Close'].rolling(20).mean()
            
            weekly_current = weekly_data['Close'].iloc[-1]
            weekly_trend_score = 0
            
            if weekly_current > weekly_ma10.iloc[-1]:
                weekly_trend_score += 1
            if weekly_current > weekly_ma20.iloc[-1]:
                weekly_trend_score += 1
            if weekly_ma10.iloc[-1] > weekly_ma20.iloc[-1]:
                weekly_trend_score += 1
            
            if weekly_trend_score >= 2:
                results['weekly_trend'] = 'bullish'
            elif weekly_trend_score <= 1:
                results['weekly_trend'] = 'bearish'
            
            # Check alignment
            results['alignment'] = (results['daily_trend'] == results['weekly_trend'] and 
                                  results['daily_trend'] != 'neutral')
        
        # Calculate overall strength
        strength = daily_trend_score
        if weekly_data is not None:
            strength += (weekly_trend_score if 'weekly_trend_score' in locals() else 0)
            strength = strength / 6 * 100  # Normalize to 0-100
        else:
            strength = strength / 3 * 100
        
        results['strength'] = strength
        
        # Generate recommendation
        if results['alignment'] and strength > 70:
            results['recommendation'] = 'strong_buy' if results['daily_trend'] == 'bullish' else 'strong_sell'
        elif strength > 60:
            results['recommendation'] = 'buy' if results['daily_trend'] == 'bullish' else 'sell'
        elif strength < 40:
            results['recommendation'] = 'avoid'
        else:
            results['recommendation'] = 'hold'
        
        logger.debug(f"Multi-timeframe analysis: {results}")
        return results
    
    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """
        Calculate overall portfolio risk metrics
        """
        if not positions:
            return {'total_risk': 0, 'concentration_risk': 0, 'correlation_risk': 'low'}
        
        total_value = sum(pos['value'] for pos in positions)
        total_risk = sum(pos.get('risk_amount', 0) for pos in positions)
        
        # Concentration risk (largest position as % of portfolio)
        max_position = max(pos['value'] for pos in positions)
        concentration_risk = (max_position / total_value) * 100
        
        # Sector concentration
        sectors = {}
        for pos in positions:
            sector = pos.get('sector', 'unknown')
            sectors[sector] = sectors.get(sector, 0) + pos['value']
        
        max_sector_exposure = max(sectors.values()) if sectors else 0
        sector_concentration = (max_sector_exposure / total_value) * 100
        
        # Risk assessment
        risk_level = 'low'
        if total_risk / self.account_size > 0.15:  # >15% total risk
            risk_level = 'high'
        elif concentration_risk > 25 or sector_concentration > 40:
            risk_level = 'medium'
        
        return {
            'total_risk': total_risk,
            'total_risk_pct': (total_risk / self.account_size) * 100,
            'concentration_risk': concentration_risk,
            'sector_concentration': sector_concentration,
            'risk_level': risk_level,
            'diversification_score': min(100, len(positions) * 10),  # Simple diversification score
            'recommendations': self._get_risk_recommendations(concentration_risk, sector_concentration, len(positions))
        }
    
    def _get_risk_recommendations(self, concentration: float, sector_concentration: float, num_positions: int) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if concentration > 30:
            recommendations.append("Reduce largest position size to improve diversification")
        
        if sector_concentration > 50:
            recommendations.append("Consider diversifying across different sectors")
        
        if num_positions < 5:
            recommendations.append("Increase number of positions to reduce concentration risk")
        elif num_positions > 20:
            recommendations.append("Consider consolidating positions to reduce complexity")
        
        return recommendations
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95, holding_period: int = 1) -> float:
        """
        Calculate Value at Risk (VaR)
        """
        if len(returns) < 30:
            logger.warning("Insufficient data for reliable VaR calculation")
            return 0.0
        
        # Historical simulation method
        sorted_returns = returns.sort_values()
        index = int((1 - confidence) * len(sorted_returns))
        var = abs(sorted_returns.iloc[index]) * np.sqrt(holding_period)
        
        return var
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        """
        if len(returns) < 30 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 252)  # Daily risk-free rate
        return excess_returns / returns.std() * np.sqrt(252)  # Annualized
    
    def stress_test(self, position_value: float, stress_scenarios: Dict[str, float] = None) -> Dict:
        """
        Perform stress testing on position
        """
        if stress_scenarios is None:
            stress_scenarios = {
                'market_crash_20': -0.20,
                'market_crash_30': -0.30,
                'recession': -0.35,
                'sector_rotation': -0.15,
                'flash_crash': -0.10
            }
        
        results = {}
        for scenario, decline in stress_scenarios.items():
            loss = position_value * abs(decline)
            loss_pct = (loss / self.account_size) * 100
            
            results[scenario] = {
                'position_loss': loss,
                'account_impact_pct': loss_pct,
                'survivable': loss_pct < 10  # Can survive if <10% account impact
            }
        
        return results

# Global risk manager instance
risk_manager = RiskManager()

def calculate_position_size(current_price: float, stop_loss: float, risk_pct: float = None) -> PositionSize:
    """Convenience function for position sizing"""
    return risk_manager.calculate_position_size(current_price, stop_loss, risk_pct)

def multi_timeframe_check(daily_data: pd.DataFrame, weekly_data: pd.DataFrame = None) -> Dict:
    """Convenience function for multi-timeframe analysis"""
    return risk_manager.multi_timeframe_analysis(daily_data, weekly_data)

# Test function
if __name__ == '__main__':
    print("🧪 Testing Risk Management System...")
    
    # Test position sizing
    position = calculate_position_size(
        current_price=100.0,
        stop_loss=95.0,
        risk_pct=2.0
    )
    
    print(f"Position Size: {position.shares} shares")
    print(f"Investment: {position.investment_amount:,.0f}")
    print(f"Risk: {position.risk_amount:.0f} ({position.risk_percentage:.1f}%)")
    print(f"Warnings: {position.warnings}")
    
    # Test with sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Close': 100 + np.random.normal(0, 2, 100).cumsum(),
        'Volume': np.random.randint(10000, 50000, 100)
    })
    
    mtf_analysis = multi_timeframe_check(sample_data)
    print(f"\nMulti-timeframe Analysis: {mtf_analysis}")
    
    print("✅ Risk management system test complete!")