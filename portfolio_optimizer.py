#!/usr/bin/env python3
"""
Portfolio Construction & Optimization Engine
===========================================

Professional portfolio optimization system implementing institutional
best practices for cross-sectional equity strategies:

- Alpha-driven position sizing with risk constraints
- Sector/factor exposure management
- Transaction cost optimization
- Kelly criterion and risk parity integration
- Market-neutral and long-short portfolio construction
- Turnover and capacity management
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import logging
from scipy.optimize import minimize
from scipy.stats import rankdata
import cvxpy as cp

from quant_system_config import SystemConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Advanced portfolio optimization for quantitative trading"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.portfolio_config = self.config.portfolio
        
        # Portfolio state
        self.current_positions = pd.Series()
        self.target_positions = pd.Series()
        self.position_history = []
        
        # Risk models
        self.risk_model = None
        self.sector_exposures = {}
        
        # Transaction cost model parameters
        self.tc_params = {
            'fixed_cost': self.config.execution.commission_rate,
            'linear_cost': self.config.execution.bid_ask_spread_assumption / 2,
            'market_impact': 0.001,  # Square-root impact coefficient
            'temporary_impact': 0.0005
        }
        
        # PSX sector mapping
        self.sector_mapping = {
            'UBL': 'Banking', 'MCB': 'Banking', 'HBL': 'Banking', 'ABL': 'Banking',
            'NBP': 'Banking', 'BOP': 'Banking', 'BAFL': 'Banking', 'MEBL': 'Banking',
            'PPL': 'Oil_Gas', 'OGDC': 'Oil_Gas', 'POL': 'Oil_Gas', 'MARI': 'Oil_Gas',
            'PSO': 'Oil_Gas', 'SNGP': 'Oil_Gas', 'SSGC': 'Oil_Gas',
            'LUCK': 'Cement', 'DGKC': 'Cement', 'FCCL': 'Cement', 'CHCC': 'Cement',
            'ENGRO': 'Chemical', 'FFC': 'Chemical', 'FATIMA': 'Chemical', 'ICI': 'Chemical',
            'KTML': 'Power', 'KAPCO': 'Power', 'HUBCO': 'Power', 'KEL': 'Power',
            'PTCL': 'Telecom', 'TRG': 'Technology', 'SYSTEMS': 'Technology',
            'NESTLE': 'FMCG', 'LOTTE': 'FMCG'
        }
    
    def calculate_alpha_scores(self, predictions: Union[pd.Series, Dict[str, float]], 
                              method: str = 'rank') -> pd.Series:
        """Convert ML predictions to alpha scores"""
        
        if isinstance(predictions, dict):
            predictions = pd.Series(predictions)
        
        predictions = predictions.dropna()
        
        if method == 'rank':
            # Convert to percentile ranks (0-1)
            ranks = rankdata(predictions) / len(predictions)
            alpha_scores = pd.Series(ranks, index=predictions.index)
            
        elif method == 'zscore':
            # Standardize predictions
            mean_pred = predictions.mean()
            std_pred = predictions.std()
            if std_pred > 0:
                alpha_scores = (predictions - mean_pred) / std_pred
            else:
                alpha_scores = pd.Series(0, index=predictions.index)
                
        elif method == 'raw':
            # Use raw predictions (already normalized)
            alpha_scores = predictions.copy()
            
        else:
            raise ValueError(f"Unknown alpha method: {method}")
        
        return alpha_scores
    
    def calculate_position_sizes_kelly(self, alpha_scores: pd.Series, 
                                      expected_returns: pd.Series,
                                      volatilities: pd.Series,
                                      win_rate: float = 0.55) -> pd.Series:
        """Calculate position sizes using Kelly criterion"""
        
        # Align all series
        common_index = alpha_scores.index.intersection(expected_returns.index).intersection(volatilities.index)
        alpha_aligned = alpha_scores.loc[common_index]
        returns_aligned = expected_returns.loc[common_index]
        vol_aligned = volatilities.loc[common_index]
        
        if len(common_index) == 0:
            return pd.Series()
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = lose probability
        b = np.abs(returns_aligned) / vol_aligned  # Reward-to-risk ratio
        p = win_rate  # Win probability
        q = 1 - p     # Lose probability
        
        # Kelly fractions
        kelly_fractions = (b * p - q) / b
        
        # Cap Kelly fractions to prevent over-leverage
        kelly_fractions = kelly_fractions.clip(0, 0.25)  # Max 25% Kelly
        
        # Scale by alpha confidence
        alpha_scaled = np.abs(alpha_aligned) / np.abs(alpha_aligned).max() if np.abs(alpha_aligned).max() > 0 else alpha_aligned
        position_sizes = kelly_fractions * alpha_scaled
        
        # Apply sign based on alpha direction
        position_sizes = position_sizes * np.sign(alpha_aligned)
        
        return position_sizes
    
    def calculate_position_sizes_risk_parity(self, alpha_scores: pd.Series,
                                           volatilities: pd.Series) -> pd.Series:
        """Calculate risk parity position sizes"""
        
        common_index = alpha_scores.index.intersection(volatilities.index)
        alpha_aligned = alpha_scores.loc[common_index]
        vol_aligned = volatilities.loc[common_index]
        
        if len(common_index) == 0:
            return pd.Series()
        
        # Inverse volatility weighting
        inv_vol = 1 / (vol_aligned + 1e-8)
        
        # Normalize to sum to 1 (before applying alpha direction)
        inv_vol_norm = inv_vol / inv_vol.sum()
        
        # Apply alpha direction and scaling
        alpha_magnitude = np.abs(alpha_aligned)
        risk_parity_sizes = inv_vol_norm * alpha_magnitude * np.sign(alpha_aligned)
        
        return risk_parity_sizes
    
    def calculate_target_positions(self, alpha_scores: pd.Series,
                                 market_data: Dict[str, Any] = None) -> pd.Series:
        """Calculate target portfolio positions"""
        
        if alpha_scores.empty:
            return pd.Series()
        
        # Get market data if provided
        expected_returns = market_data.get('expected_returns', pd.Series()) if market_data else pd.Series()
        volatilities = market_data.get('volatilities', pd.Series()) if market_data else pd.Series()
        
        # If no volatilities provided, estimate from alpha scores
        if volatilities.empty:
            volatilities = pd.Series(0.02, index=alpha_scores.index)  # Assume 2% daily vol
        
        # If no expected returns, use alpha as proxy
        if expected_returns.empty:
            expected_returns = alpha_scores * 0.01  # Scale alpha to reasonable return expectation
        
        # Calculate position sizes using different methods
        if self.portfolio_config.use_kelly_criterion:
            kelly_positions = self.calculate_position_sizes_kelly(
                alpha_scores, expected_returns, volatilities
            )
        else:
            kelly_positions = pd.Series(0, index=alpha_scores.index)
        
        risk_parity_positions = self.calculate_position_sizes_risk_parity(
            alpha_scores, volatilities
        )
        
        # Combine methods
        alpha_weight = self.portfolio_config.alpha_weight
        rp_weight = self.portfolio_config.risk_parity_weight
        
        if not kelly_positions.empty and self.portfolio_config.use_kelly_criterion:
            combined_positions = (alpha_weight * kelly_positions + 
                                rp_weight * risk_parity_positions)
        else:
            # Pure alpha-based sizing
            alpha_magnitude = np.abs(alpha_scores)
            alpha_norm = alpha_magnitude / alpha_magnitude.sum() if alpha_magnitude.sum() > 0 else alpha_magnitude
            combined_positions = alpha_norm * np.sign(alpha_scores)
        
        # Apply position size limits
        max_single_position = self.portfolio_config.max_single_position
        combined_positions = combined_positions.clip(-max_single_position, max_single_position)
        
        # Scale to target gross exposure
        gross_exposure = np.abs(combined_positions).sum()
        if gross_exposure > self.portfolio_config.gross_exposure_max:
            combined_positions = combined_positions * (self.portfolio_config.gross_exposure_max / gross_exposure)
        
        # Select top positions
        n_positions = min(len(combined_positions), self.portfolio_config.max_positions)
        
        # Sort by absolute alpha and take top N
        top_positions = combined_positions.reindex(
            alpha_scores.abs().nlargest(n_positions).index
        ).fillna(0)
        
        return top_positions
    
    def apply_sector_constraints(self, positions: pd.Series) -> pd.Series:
        """Apply sector exposure constraints"""
        
        if positions.empty:
            return positions
        
        constrained_positions = positions.copy()
        
        # Group by sectors
        position_sectors = pd.Series(
            [self.sector_mapping.get(symbol, 'Other') for symbol in positions.index],
            index=positions.index
        )
        
        # Check sector exposure limits
        for sector in position_sectors.unique():
            sector_mask = position_sectors == sector
            sector_positions = constrained_positions[sector_mask]
            
            # Check sector concentration
            sector_exposure = np.abs(sector_positions).sum()
            max_sector_exposure = self.portfolio_config.max_sector_exposure
            
            if sector_exposure > max_sector_exposure:
                # Scale down sector positions proportionally
                scale_factor = max_sector_exposure / sector_exposure
                constrained_positions[sector_mask] = sector_positions * scale_factor
                
                logger.info(f"Scaled down {sector} exposure by {scale_factor:.3f}")
        
        return constrained_positions
    
    def optimize_portfolio_with_constraints(self, alpha_scores: pd.Series,
                                          risk_model: Optional[np.ndarray] = None,
                                          market_data: Dict[str, Any] = None) -> pd.Series:
        """Optimize portfolio using convex optimization with constraints"""
        
        try:
            n_assets = len(alpha_scores)
            if n_assets == 0:
                return pd.Series()
            
            # Decision variables
            w = cp.Variable(n_assets)  # Portfolio weights
            
            # Objective: maximize alpha exposure - risk penalty - transaction costs
            alpha_vector = alpha_scores.values
            
            # Alpha objective (maximize)
            alpha_obj = alpha_vector @ w
            
            # Risk penalty (minimize)
            if risk_model is not None and risk_model.shape == (n_assets, n_assets):
                risk_penalty = cp.quad_form(w, risk_model)
            else:
                # Simple risk penalty based on volatility
                volatilities = market_data.get('volatilities', pd.Series(0.02, index=alpha_scores.index)) if market_data else pd.Series(0.02, index=alpha_scores.index)
                vol_vector = volatilities.reindex(alpha_scores.index).fillna(0.02).values
                risk_penalty = cp.sum_squares(cp.multiply(vol_vector, w))
            
            # Transaction cost penalty
            if not self.current_positions.empty:
                current_weights = self.current_positions.reindex(alpha_scores.index).fillna(0).values
                trades = w - current_weights
                tc_penalty = cp.norm(trades, 1) * self.tc_params['linear_cost']
            else:
                tc_penalty = 0
            
            # Combined objective
            risk_aversion = 0.5  # Risk aversion parameter
            tc_aversion = 0.1    # Transaction cost aversion
            
            objective = cp.Maximize(alpha_obj - risk_aversion * risk_penalty - tc_aversion * tc_penalty)
            
            # Constraints
            constraints = []
            
            # Position size constraints
            max_pos = self.portfolio_config.max_single_position
            constraints.extend([
                w >= -max_pos,  # Max short position
                w <= max_pos    # Max long position
            ])
            
            # Gross exposure constraint
            constraints.append(cp.norm(w, 1) <= self.portfolio_config.gross_exposure_max)
            
            # Net exposure constraint (if market neutral)
            if abs(self.portfolio_config.net_exposure_target) < 0.1:  # Market neutral
                constraints.append(cp.abs(cp.sum(w)) <= 0.1)
            else:
                net_target = self.portfolio_config.net_exposure_target
                constraints.append(cp.sum(w) >= net_target - 0.1)
                constraints.append(cp.sum(w) <= net_target + 0.1)
            
            # Sector constraints
            sectors = [self.sector_mapping.get(symbol, 'Other') for symbol in alpha_scores.index]
            unique_sectors = list(set(sectors))
            
            for sector in unique_sectors:
                sector_mask = [s == sector for s in sectors]
                sector_indices = [i for i, mask in enumerate(sector_mask) if mask]
                
                if sector_indices:
                    sector_exposure = cp.norm(w[sector_indices], 1)
                    constraints.append(sector_exposure <= self.portfolio_config.max_sector_exposure)
            
            # Solve optimization
            problem = cp.Problem(objective, constraints)
            
            # Try different solvers
            solvers = [cp.ECOS, cp.OSQP, cp.SCS]
            solution = None
            
            for solver in solvers:
                try:
                    problem.solve(solver=solver, verbose=False)
                    if problem.status == cp.OPTIMAL:
                        solution = w.value
                        break
                except:
                    continue
            
            if solution is not None:
                optimized_positions = pd.Series(solution, index=alpha_scores.index)
                
                # Clean up tiny positions
                optimized_positions = optimized_positions.where(np.abs(optimized_positions) > 1e-4, 0)
                
                logger.info(f"Portfolio optimized successfully. Gross exposure: {np.abs(optimized_positions).sum():.3f}")
                return optimized_positions
            else:
                logger.warning("Optimization failed, using simple alpha-based allocation")
                return self.calculate_target_positions(alpha_scores, market_data)
                
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return self.calculate_target_positions(alpha_scores, market_data)
    
    def calculate_turnover(self, new_positions: pd.Series) -> float:
        """Calculate portfolio turnover"""
        
        if self.current_positions.empty:
            return np.abs(new_positions).sum()
        
        # Align positions
        all_symbols = set(self.current_positions.index).union(set(new_positions.index))
        current_aligned = self.current_positions.reindex(all_symbols).fillna(0)
        new_aligned = new_positions.reindex(all_symbols).fillna(0)
        
        # Turnover = sum of absolute changes
        turnover = np.abs(new_aligned - current_aligned).sum()
        
        return turnover
    
    def apply_turnover_constraints(self, target_positions: pd.Series) -> pd.Series:
        """Apply turnover constraints to limit trading"""
        
        turnover = self.calculate_turnover(target_positions)
        max_turnover = self.portfolio_config.turnover_limit
        
        if turnover <= max_turnover:
            return target_positions
        
        # Scale down changes to meet turnover limit
        if not self.current_positions.empty:
            all_symbols = set(self.current_positions.index).union(set(target_positions.index))
            current_aligned = self.current_positions.reindex(all_symbols).fillna(0)
            target_aligned = target_positions.reindex(all_symbols).fillna(0)
            
            changes = target_aligned - current_aligned
            scale_factor = max_turnover / turnover
            
            constrained_positions = current_aligned + changes * scale_factor
            
            logger.info(f"Applied turnover constraint: scaled changes by {scale_factor:.3f}")
            
            return constrained_positions[constrained_positions != 0]
        else:
            return target_positions
    
    def construct_long_short_portfolio(self, alpha_scores: pd.Series,
                                     market_data: Dict[str, Any] = None) -> Dict[str, pd.Series]:
        """Construct long-short market neutral portfolio"""
        
        if alpha_scores.empty:
            return {'long': pd.Series(), 'short': pd.Series()}
        
        # Sort by alpha
        sorted_alpha = alpha_scores.sort_values(ascending=False)
        n_positions = min(len(sorted_alpha), self.portfolio_config.max_positions)
        
        # Split into long and short
        n_long = int(n_positions * self.portfolio_config.long_allocation)
        n_short = n_positions - n_long
        
        # Select top and bottom stocks
        long_symbols = sorted_alpha.head(n_long).index
        short_symbols = sorted_alpha.tail(n_short).index
        
        # Calculate position sizes
        max_pos = self.portfolio_config.max_single_position
        
        # Long positions (equal weight for now)
        long_weight = max_pos if n_long == 1 else min(max_pos, self.portfolio_config.long_allocation / n_long)
        long_positions = pd.Series(long_weight, index=long_symbols)
        
        # Short positions (equal weight for now) 
        short_weight = -max_pos if n_short == 1 else -min(max_pos, self.portfolio_config.short_allocation / n_short)
        short_positions = pd.Series(short_weight, index=short_symbols)
        
        # Apply sector constraints
        long_positions = self.apply_sector_constraints(long_positions)
        short_positions = self.apply_sector_constraints(short_positions)
        
        return {'long': long_positions, 'short': short_positions}
    
    def generate_portfolio_signals(self, alpha_scores: pd.Series,
                                 market_data: Dict[str, Any] = None,
                                 strategy_type: str = None) -> Dict[str, Any]:
        """Generate complete portfolio construction signals"""
        
        if alpha_scores.empty:
            return {'positions': pd.Series(), 'metadata': {}}
        
        strategy_type = strategy_type or self.portfolio_config.rebalance_frequency.value
        
        # Calculate target positions based on strategy
        if strategy_type == 'MARKET_NEUTRAL':
            long_short = self.construct_long_short_portfolio(alpha_scores, market_data)
            target_positions = pd.concat([long_short['long'], long_short['short']])
            
        elif strategy_type == 'LONG_ONLY':
            # Long-only portfolio
            positive_alpha = alpha_scores[alpha_scores > 0]
            if not positive_alpha.empty:
                target_positions = self.calculate_target_positions(positive_alpha, market_data)
                target_positions = target_positions.clip(lower=0)  # Ensure no shorts
            else:
                target_positions = pd.Series()
                
        else:
            # Default long-short with optimization
            if len(alpha_scores) > 10:  # Use optimization for larger universes
                target_positions = self.optimize_portfolio_with_constraints(
                    alpha_scores, market_data=market_data
                )
            else:
                target_positions = self.calculate_target_positions(alpha_scores, market_data)
        
        # Apply constraints
        target_positions = self.apply_sector_constraints(target_positions)
        target_positions = self.apply_turnover_constraints(target_positions)
        
        # Calculate portfolio metrics
        turnover = self.calculate_turnover(target_positions)
        gross_exposure = np.abs(target_positions).sum()
        net_exposure = target_positions.sum()
        n_positions = len(target_positions[target_positions != 0])
        
        # Sector breakdown
        sector_exposures = {}
        for symbol, position in target_positions.items():
            sector = self.sector_mapping.get(symbol, 'Other')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + abs(position)
        
        metadata = {
            'turnover': turnover,
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'n_positions': n_positions,
            'n_long': len(target_positions[target_positions > 0]),
            'n_short': len(target_positions[target_positions < 0]),
            'max_position': np.abs(target_positions).max() if not target_positions.empty else 0,
            'sector_exposures': sector_exposures,
            'construction_method': strategy_type,
            'timestamp': dt.datetime.now()
        }
        
        # Update current positions
        self.target_positions = target_positions
        
        return {'positions': target_positions, 'metadata': metadata}
    
    def update_current_positions(self, executed_positions: pd.Series):
        """Update current positions after execution"""
        self.current_positions = executed_positions
        
        # Store in history
        position_record = {
            'timestamp': dt.datetime.now(),
            'positions': executed_positions.copy(),
            'gross_exposure': np.abs(executed_positions).sum(),
            'net_exposure': executed_positions.sum()
        }
        
        self.position_history.append(position_record)
    
    def get_portfolio_analytics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio analytics"""
        
        if self.current_positions.empty:
            return {'message': 'No current positions'}
        
        analytics = {
            'current_positions': len(self.current_positions[self.current_positions != 0]),
            'gross_exposure': np.abs(self.current_positions).sum(),
            'net_exposure': self.current_positions.sum(),
            'long_exposure': self.current_positions[self.current_positions > 0].sum(),
            'short_exposure': self.current_positions[self.current_positions < 0].sum(),
            'max_position': np.abs(self.current_positions).max(),
            'position_concentration': {}
        }
        
        # Top positions
        top_positions = self.current_positions.abs().nlargest(5)
        analytics['top_positions'] = top_positions.to_dict()
        
        # Sector breakdown
        sector_breakdown = {}
        for symbol, position in self.current_positions.items():
            if position != 0:
                sector = self.sector_mapping.get(symbol, 'Other')
                sector_breakdown[sector] = sector_breakdown.get(sector, 0) + abs(position)
        
        analytics['sector_breakdown'] = sector_breakdown
        
        # Position size distribution
        pos_sizes = np.abs(self.current_positions[self.current_positions != 0])
        if not pos_sizes.empty:
            analytics['position_stats'] = {
                'mean_size': pos_sizes.mean(),
                'median_size': pos_sizes.median(),
                'std_size': pos_sizes.std(),
                'min_size': pos_sizes.min(),
                'max_size': pos_sizes.max()
            }
        
        return analytics

# Test function
def test_portfolio_optimizer():
    """Test portfolio optimization system"""
    print("🚀 Testing Portfolio Optimization System")
    print("=" * 50)
    
    # Create optimizer
    config = SystemConfig()
    optimizer = PortfolioOptimizer(config)
    
    print(f"⚙️ Portfolio Configuration:")
    print(f"   Max Positions: {config.portfolio.max_positions}")
    print(f"   Long/Short: {config.portfolio.long_allocation:.1%}/{config.portfolio.short_allocation:.1%}")
    print(f"   Max Single Position: {config.portfolio.max_single_position:.1%}")
    print(f"   Gross Exposure Max: {config.portfolio.gross_exposure_max:.1%}")
    
    # Generate sample alpha scores
    print(f"\n📊 Generating sample alpha scores...")
    
    symbols = ['UBL', 'MCB', 'HBL', 'ABL', 'LUCK', 'DGKC', 'FFC', 'ENGRO', 
               'PPL', 'OGDC', 'PSO', 'PTCL', 'TRG', 'NESTLE', 'KEL']
    
    np.random.seed(42)
    alpha_scores = pd.Series(
        np.random.normal(0, 1, len(symbols)),  # Random alpha scores
        index=symbols
    )
    
    print(f"   Generated alpha for {len(alpha_scores)} symbols")
    print(f"   Alpha range: {alpha_scores.min():.3f} to {alpha_scores.max():.3f}")
    print(f"   Top 3 alpha: {alpha_scores.nlargest(3).to_dict()}")
    
    # Test portfolio construction
    print(f"\n🏗️ Testing Portfolio Construction...")
    
    try:
        # Generate portfolio signals
        portfolio_result = optimizer.generate_portfolio_signals(alpha_scores)
        
        positions = portfolio_result['positions']
        metadata = portfolio_result['metadata']
        
        print(f"✅ Portfolio Construction Results:")
        print(f"   Total Positions: {metadata['n_positions']}")
        print(f"   Long Positions: {metadata['n_long']}")
        print(f"   Short Positions: {metadata['n_short']}")
        print(f"   Gross Exposure: {metadata['gross_exposure']:.1%}")
        print(f"   Net Exposure: {metadata['net_exposure']:.1%}")
        print(f"   Max Position: {metadata['max_position']:.1%}")
        print(f"   Turnover: {metadata['turnover']:.1%}")
        
        # Show top positions
        if not positions.empty:
            print(f"\n🔝 Top Positions:")
            top_positions = positions.abs().nlargest(5)
            for symbol, weight in top_positions.items():
                direction = "LONG" if positions[symbol] > 0 else "SHORT"
                print(f"   {symbol}: {abs(weight):.1%} {direction}")
        
        # Sector breakdown
        print(f"\n🏢 Sector Exposures:")
        for sector, exposure in metadata['sector_exposures'].items():
            print(f"   {sector}: {exposure:.1%}")
        
        # Test long-short construction
        print(f"\n📈📉 Testing Long-Short Portfolio...")
        ls_result = optimizer.construct_long_short_portfolio(alpha_scores)
        
        long_positions = ls_result['long']
        short_positions = ls_result['short']
        
        print(f"   Long Positions: {len(long_positions)}")
        if not long_positions.empty:
            print(f"   Long Exposure: {long_positions.sum():.1%}")
            print(f"   Top Long: {long_positions.idxmax()} ({long_positions.max():.1%})")
        
        print(f"   Short Positions: {len(short_positions)}")
        if not short_positions.empty:
            print(f"   Short Exposure: {short_positions.sum():.1%}")
            print(f"   Top Short: {short_positions.idxmin()} ({abs(short_positions.min()):.1%})")
        
        # Update positions and test analytics
        print(f"\n📊 Testing Portfolio Analytics...")
        optimizer.update_current_positions(positions)
        
        analytics = optimizer.get_portfolio_analytics()
        print(f"   Current Positions: {analytics['current_positions']}")
        print(f"   Gross Exposure: {analytics['gross_exposure']:.1%}")
        print(f"   Net Exposure: {analytics['net_exposure']:.1%}")
        
        if 'position_stats' in analytics:
            stats = analytics['position_stats']
            print(f"   Mean Position Size: {stats['mean_size']:.1%}")
            print(f"   Position Size Range: {stats['min_size']:.1%} - {stats['max_size']:.1%}")
        
    except Exception as e:
        print(f"❌ Error in portfolio optimization: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🏁 Portfolio optimization test completed!")

if __name__ == "__main__":
    test_portfolio_optimizer()