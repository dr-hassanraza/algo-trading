#!/usr/bin/env python3
"""
Quantitative Trading System Configuration
=========================================

Professional-grade configuration system based on institutional best practices.
Defines performance targets, risk parameters, and system architecture for
PSX algorithmic trading system.

Based on the framework:
- Target: Annualized alpha vs KSE100 ≥ 4-8%, Sharpe ≥ 1.5, max drawdown ≤ 15-20%
- Capacity: 10-50 positions, daily rebalancing
- Cross-sectional equity ranking with market-neutral capability
"""

import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import datetime as dt
import os

class StrategyType(Enum):
    LONG_SHORT = "LONG_SHORT"
    MARKET_NEUTRAL = "MARKET_NEUTRAL"  
    LONG_ONLY = "LONG_ONLY"
    EVENT_DRIVEN = "EVENT_DRIVEN"

class FrequencyType(Enum):
    DAILY = "DAILY"
    INTRADAY_4H = "INTRADAY_4H"
    INTRADAY_1H = "INTRADAY_1H"
    HIGH_FREQ = "HIGH_FREQ"

@dataclass
class PerformanceTargets:
    """Performance targets and risk limits"""
    
    # Return targets
    annual_alpha_target: float = 0.06  # 6% vs KSE100
    annual_alpha_min: float = 0.04     # Minimum acceptable
    annual_alpha_max: float = 0.12     # Upper target
    
    # Risk metrics targets
    sharpe_ratio_target: float = 1.5
    sharpe_ratio_min: float = 1.0
    max_drawdown_limit: float = 0.20   # 20% max drawdown
    max_daily_loss: float = 0.03       # 3% daily stop
    
    # Volatility targets
    target_volatility: float = 0.15    # 15% annual vol
    vol_scaling_factor: float = 1.0    # Scale positions based on vol
    
    # Hit rate targets
    win_rate_target: float = 0.55      # 55% win rate
    profit_factor_min: float = 1.5     # Profit factor minimum

@dataclass
class UniverseConfig:
    """Trading universe configuration"""
    
    # Common PSX symbols for trading
    common_symbols: List[str] = field(default_factory=lambda: [
        'UBL', 'MCB', 'OGDC', 'PPL', 'HUBCO', 'KAPCO', 'LUCK', 'ENGRO',
        'FCCL', 'DGKC', 'MLCF', 'FFBL', 'ATRL', 'SEARL', 'PIOC', 'FFC'
    ])
    
    # PSX Universe
    market_cap_min: float = 5_000_000_000    # 5B PKR minimum market cap
    avg_volume_min: float = 10_000_000       # 10M PKR daily volume minimum
    price_min: float = 10.0                  # Minimum stock price
    price_max: float = 10000.0               # Maximum stock price
    
    # Sector diversification
    max_positions_per_sector: int = 5
    max_sector_exposure: float = 0.25        # 25% max in one sector
    
    # Liquidity requirements
    min_turnover_ratio: float = 0.01         # Minimum daily turnover
    max_bid_ask_spread: float = 0.02         # 2% max spread
    
    # Exclusions
    exclude_suspended: bool = True
    exclude_penny_stocks: bool = True
    min_trading_days: int = 200              # Must trade 200+ days/year

@dataclass
class PortfolioConfig:
    """Portfolio construction parameters"""
    
    # Position limits
    max_positions: int = 50
    min_positions: int = 10
    max_single_position: float = 0.05        # 5% max per position
    
    # Long/Short allocation
    long_allocation: float = 0.6             # 60% long
    short_allocation: float = 0.4            # 40% short
    net_exposure_target: float = 0.2         # 20% net long
    gross_exposure_max: float = 1.0          # 100% gross exposure
    
    # Rebalancing
    rebalance_frequency: FrequencyType = FrequencyType.DAILY
    turnover_target: float = 0.5             # 50% daily turnover
    turnover_limit: float = 1.0              # 100% max turnover
    
    # Position sizing
    use_kelly_criterion: bool = True
    risk_parity_weight: float = 0.3          # 30% risk parity influence
    alpha_weight: float = 0.7                # 70% alpha-based sizing

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    
    # Price-based features
    momentum_periods: List[int] = field(default_factory=lambda: [21, 63, 126])  # 1m, 3m, 6m
    reversal_periods: List[int] = field(default_factory=lambda: [5, 10])        # 5d, 10d
    volatility_periods: List[int] = field(default_factory=lambda: [21, 63])     # Volatility lookbacks
    
    # Technical indicators
    enable_rsi: bool = True
    enable_macd: bool = True
    enable_bollinger: bool = True
    enable_atr: bool = True
    
    # Fundamental features
    enable_value_metrics: bool = True        # P/E, P/B, EV/EBITDA
    enable_quality_metrics: bool = True     # ROE, ROA, Debt/Equity
    enable_growth_metrics: bool = True      # Revenue growth, earnings growth
    
    # Cross-sectional ranking
    use_sector_neutralization: bool = True
    use_winsorization: bool = True
    winsorize_percentile: float = 0.02      # Winsorize at 2%/98%
    
    # Feature selection
    max_features: int = 50
    feature_selection_method: str = "mutual_info"  # or "lasso", "rfe"

@dataclass
class ModelConfig:
    """ML Model configuration"""
    
    # Model type
    primary_model: str = "lightgbm"          # LightGBM as primary
    ensemble_models: List[str] = field(default_factory=lambda: ["xgboost", "rf"])
    
    # Training parameters
    train_window_months: int = 24            # 2 years training data
    validation_months: int = 3               # 3 months validation
    retrain_frequency_days: int = 30         # Retrain monthly
    
    # Cross-validation
    cv_folds: int = 5
    use_purged_cv: bool = True
    embargo_days: int = 2                    # 2-day embargo period
    
    # Model hyperparameters
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    })
    
    # Labels
    label_forward_days: int = 5              # 5-day forward returns
    label_type: str = "vol_adjusted"         # or "raw", "rank"
    use_regime_aware_labels: bool = True

@dataclass
class RiskConfig:
    """Risk management configuration"""
    
    # Position-level risk
    stop_loss_pct: float = 0.05              # 5% stop loss
    take_profit_pct: float = 0.10             # 10% take profit
    use_trailing_stops: bool = True
    trailing_stop_pct: float = 0.03           # 3% trailing stop
    
    # Portfolio-level risk
    var_confidence: float = 0.05              # 5% VaR
    var_lookback_days: int = 252              # 1-year VaR calculation
    correlation_limit: float = 0.7            # Max correlation between positions
    
    # Kill switches
    daily_loss_limit: float = 0.03            # 3% daily loss limit
    weekly_loss_limit: float = 0.10           # 10% weekly loss limit
    drawdown_circuit_breaker: float = 0.15    # 15% drawdown circuit breaker
    
    # Live monitoring
    sharpe_monitoring_days: int = 20          # Monitor 20-day Sharpe
    sharpe_deviation_threshold: float = 1.0   # Threshold for de-risking
    min_live_days: int = 10                   # Min days before live monitoring

@dataclass
class ExecutionConfig:
    """Execution and trading configuration"""
    
    # Order management
    use_vwap: bool = True
    participation_rate: float = 0.05          # 5% of volume
    max_order_size_adv: float = 0.10          # 10% of ADV
    
    # Timing
    trading_start_time: str = "09:45"         # Start after opening volatility
    trading_end_time: str = "15:00"           # End before close
    avoid_earnings_days: bool = True
    
    # Costs
    commission_rate: float = 0.0015           # 0.15% commission
    bid_ask_spread_assumption: float = 0.002  # 0.2% spread cost
    market_impact_model: str = "sqrt"         # Square-root impact model
    
    # Paper trading
    paper_trading_enabled: bool = True
    paper_trading_duration_days: int = 30     # 30 days paper trading

@dataclass
class DataConfig:
    """Data management configuration"""
    
    # Data sources
    primary_price_source: str = "psx_dps"     # PSX DPS as primary
    backup_price_sources: List[str] = field(default_factory=lambda: ["eodhd", "yfinance"])
    
    # Storage
    data_path: str = "./data"
    use_database: bool = True
    database_type: str = "sqlite"             # or "postgresql"
    
    # Quality checks
    max_missing_data_pct: float = 0.05        # 5% max missing data
    price_change_threshold: float = 0.20      # 20% single-day change threshold
    volume_spike_threshold: float = 5.0       # 5x volume spike threshold
    
    # Updating
    data_update_time: str = "17:00"           # Update after market close
    weekend_data_processing: bool = True

@dataclass
class SystemConfig:
    """Complete system configuration"""
    
    # Meta information
    system_name: str = "PSX Quant System"
    version: str = "2.0.0"
    created_date: str = field(default_factory=lambda: dt.datetime.now().isoformat())
    
    # Configuration sections
    performance: PerformanceTargets = field(default_factory=PerformanceTargets)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # System paths
    config_path: str = "./config"
    logs_path: str = "./logs"
    models_path: str = "./models"
    results_path: str = "./results"
    
    def save_config(self, filepath: str = None):
        """Save configuration to YAML file"""
        if filepath is None:
            filepath = os.path.join(self.config_path, f"system_config_{dt.datetime.now().strftime('%Y%m%d')}.yaml")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to dict for YAML serialization
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration saved to: {filepath}")
        return filepath
    
    def load_config(self, filepath: str):
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configuration from dict
        self.from_dict(config_dict)
        print(f"✅ Configuration loaded from: {filepath}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'system_name': self.system_name,
            'version': self.version,
            'created_date': self.created_date,
            'performance': {
                'annual_alpha_target': self.performance.annual_alpha_target,
                'annual_alpha_min': self.performance.annual_alpha_min,
                'annual_alpha_max': self.performance.annual_alpha_max,
                'sharpe_ratio_target': self.performance.sharpe_ratio_target,
                'sharpe_ratio_min': self.performance.sharpe_ratio_min,
                'max_drawdown_limit': self.performance.max_drawdown_limit,
                'max_daily_loss': self.performance.max_daily_loss,
                'target_volatility': self.performance.target_volatility,
                'win_rate_target': self.performance.win_rate_target,
                'profit_factor_min': self.performance.profit_factor_min
            },
            'universe': {
                'common_symbols': self.universe.common_symbols,
                'market_cap_min': self.universe.market_cap_min,
                'avg_volume_min': self.universe.avg_volume_min,
                'price_min': self.universe.price_min,
                'price_max': self.universe.price_max,
                'max_positions_per_sector': self.universe.max_positions_per_sector,
                'max_sector_exposure': self.universe.max_sector_exposure,
                'exclude_suspended': self.universe.exclude_suspended,
                'exclude_penny_stocks': self.universe.exclude_penny_stocks
            },
            'portfolio': {
                'max_positions': self.portfolio.max_positions,
                'min_positions': self.portfolio.min_positions,
                'max_single_position': self.portfolio.max_single_position,
                'long_allocation': self.portfolio.long_allocation,
                'short_allocation': self.portfolio.short_allocation,
                'net_exposure_target': self.portfolio.net_exposure_target,
                'gross_exposure_max': self.portfolio.gross_exposure_max,
                'rebalance_frequency': self.portfolio.rebalance_frequency.value,
                'turnover_target': self.portfolio.turnover_target,
                'use_kelly_criterion': self.portfolio.use_kelly_criterion
            },
            'features': {
                'momentum_periods': self.features.momentum_periods,
                'reversal_periods': self.features.reversal_periods,
                'volatility_periods': self.features.volatility_periods,
                'enable_rsi': self.features.enable_rsi,
                'enable_macd': self.features.enable_macd,
                'enable_bollinger': self.features.enable_bollinger,
                'enable_value_metrics': self.features.enable_value_metrics,
                'enable_quality_metrics': self.features.enable_quality_metrics,
                'use_sector_neutralization': self.features.use_sector_neutralization,
                'max_features': self.features.max_features
            },
            'model': {
                'primary_model': self.model.primary_model,
                'ensemble_models': self.model.ensemble_models,
                'train_window_months': self.model.train_window_months,
                'validation_months': self.model.validation_months,
                'retrain_frequency_days': self.model.retrain_frequency_days,
                'cv_folds': self.model.cv_folds,
                'use_purged_cv': self.model.use_purged_cv,
                'embargo_days': self.model.embargo_days,
                'label_forward_days': self.model.label_forward_days,
                'lightgbm_params': self.model.lightgbm_params
            },
            'risk': {
                'stop_loss_pct': self.risk.stop_loss_pct,
                'take_profit_pct': self.risk.take_profit_pct,
                'use_trailing_stops': self.risk.use_trailing_stops,
                'daily_loss_limit': self.risk.daily_loss_limit,
                'drawdown_circuit_breaker': self.risk.drawdown_circuit_breaker,
                'var_confidence': self.risk.var_confidence,
                'correlation_limit': self.risk.correlation_limit
            },
            'execution': {
                'use_vwap': self.execution.use_vwap,
                'participation_rate': self.execution.participation_rate,
                'trading_start_time': self.execution.trading_start_time,
                'trading_end_time': self.execution.trading_end_time,
                'commission_rate': self.execution.commission_rate,
                'bid_ask_spread_assumption': self.execution.bid_ask_spread_assumption,
                'paper_trading_enabled': self.execution.paper_trading_enabled
            },
            'data': {
                'primary_price_source': self.data.primary_price_source,
                'backup_price_sources': self.data.backup_price_sources,
                'data_path': self.data.data_path,
                'use_database': self.data.use_database,
                'database_type': self.data.database_type,
                'data_update_time': self.data.data_update_time
            }
        }
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        # Meta information
        if 'system_name' in config_dict:
            self.system_name = config_dict['system_name']
        if 'version' in config_dict:
            self.version = config_dict['version']
        if 'created_date' in config_dict:
            self.created_date = config_dict['created_date']
        
        # Performance section
        if 'performance' in config_dict:
            perf = config_dict['performance']
            self.performance.annual_alpha_target = perf.get('annual_alpha_target', self.performance.annual_alpha_target)
            self.performance.annual_alpha_min = perf.get('annual_alpha_min', self.performance.annual_alpha_min)
            self.performance.annual_alpha_max = perf.get('annual_alpha_max', self.performance.annual_alpha_max)
            self.performance.sharpe_ratio_target = perf.get('sharpe_ratio_target', self.performance.sharpe_ratio_target)
            self.performance.sharpe_ratio_min = perf.get('sharpe_ratio_min', self.performance.sharpe_ratio_min)
            self.performance.max_drawdown_limit = perf.get('max_drawdown_limit', self.performance.max_drawdown_limit)
            self.performance.max_daily_loss = perf.get('max_daily_loss', self.performance.max_daily_loss)
            self.performance.target_volatility = perf.get('target_volatility', self.performance.target_volatility)
            self.performance.win_rate_target = perf.get('win_rate_target', self.performance.win_rate_target)
            self.performance.profit_factor_min = perf.get('profit_factor_min', self.performance.profit_factor_min)
        
        # Universe section
        if 'universe' in config_dict:
            univ = config_dict['universe']
            self.universe.common_symbols = univ.get('common_symbols', self.universe.common_symbols)
            self.universe.market_cap_min = univ.get('market_cap_min', self.universe.market_cap_min)
            self.universe.avg_volume_min = univ.get('avg_volume_min', self.universe.avg_volume_min)
            self.universe.price_min = univ.get('price_min', self.universe.price_min)
            self.universe.price_max = univ.get('price_max', self.universe.price_max)
            self.universe.max_positions_per_sector = univ.get('max_positions_per_sector', self.universe.max_positions_per_sector)
            self.universe.max_sector_exposure = univ.get('max_sector_exposure', self.universe.max_sector_exposure)
            self.universe.exclude_suspended = univ.get('exclude_suspended', self.universe.exclude_suspended)
            self.universe.exclude_penny_stocks = univ.get('exclude_penny_stocks', self.universe.exclude_penny_stocks)
        
        # Portfolio section
        if 'portfolio' in config_dict:
            port = config_dict['portfolio']
            self.portfolio.max_positions = port.get('max_positions', self.portfolio.max_positions)
            self.portfolio.min_positions = port.get('min_positions', self.portfolio.min_positions)
            self.portfolio.max_single_position = port.get('max_single_position', self.portfolio.max_single_position)
            self.portfolio.long_allocation = port.get('long_allocation', self.portfolio.long_allocation)
            self.portfolio.short_allocation = port.get('short_allocation', self.portfolio.short_allocation)
            self.portfolio.net_exposure_target = port.get('net_exposure_target', self.portfolio.net_exposure_target)
            self.portfolio.gross_exposure_max = port.get('gross_exposure_max', self.portfolio.gross_exposure_max)
            if 'rebalance_frequency' in port:
                self.portfolio.rebalance_frequency = FrequencyType(port['rebalance_frequency'])
            self.portfolio.turnover_target = port.get('turnover_target', self.portfolio.turnover_target)
            self.portfolio.use_kelly_criterion = port.get('use_kelly_criterion', self.portfolio.use_kelly_criterion)
        
        # Features section
        if 'features' in config_dict:
            feat = config_dict['features']
            self.features.momentum_periods = feat.get('momentum_periods', self.features.momentum_periods)
            self.features.reversal_periods = feat.get('reversal_periods', self.features.reversal_periods)
            self.features.volatility_periods = feat.get('volatility_periods', self.features.volatility_periods)
            self.features.enable_rsi = feat.get('enable_rsi', self.features.enable_rsi)
            self.features.enable_macd = feat.get('enable_macd', self.features.enable_macd)
            self.features.enable_bollinger = feat.get('enable_bollinger', self.features.enable_bollinger)
            self.features.enable_value_metrics = feat.get('enable_value_metrics', self.features.enable_value_metrics)
            self.features.enable_quality_metrics = feat.get('enable_quality_metrics', self.features.enable_quality_metrics)
            self.features.use_sector_neutralization = feat.get('use_sector_neutralization', self.features.use_sector_neutralization)
            self.features.max_features = feat.get('max_features', self.features.max_features)
        
        # Model section
        if 'model' in config_dict:
            model = config_dict['model']
            self.model.primary_model = model.get('primary_model', self.model.primary_model)
            self.model.ensemble_models = model.get('ensemble_models', self.model.ensemble_models)
            self.model.train_window_months = model.get('train_window_months', self.model.train_window_months)
            self.model.validation_months = model.get('validation_months', self.model.validation_months)
            self.model.retrain_frequency_days = model.get('retrain_frequency_days', self.model.retrain_frequency_days)
            self.model.cv_folds = model.get('cv_folds', self.model.cv_folds)
            self.model.use_purged_cv = model.get('use_purged_cv', self.model.use_purged_cv)
            self.model.embargo_days = model.get('embargo_days', self.model.embargo_days)
            self.model.label_forward_days = model.get('label_forward_days', self.model.label_forward_days)
            self.model.lightgbm_params = model.get('lightgbm_params', self.model.lightgbm_params)
        
        # Risk section
        if 'risk' in config_dict:
            risk = config_dict['risk']
            self.risk.stop_loss_pct = risk.get('stop_loss_pct', self.risk.stop_loss_pct)
            self.risk.take_profit_pct = risk.get('take_profit_pct', self.risk.stop_loss_pct)
            self.risk.use_trailing_stops = risk.get('use_trailing_stops', self.risk.use_trailing_stops)
            self.risk.daily_loss_limit = risk.get('daily_loss_limit', self.risk.daily_loss_limit)
            self.risk.drawdown_circuit_breaker = risk.get('drawdown_circuit_breaker', self.risk.drawdown_circuit_breaker)
            self.risk.var_confidence = risk.get('var_confidence', self.risk.var_confidence)
            self.risk.correlation_limit = risk.get('correlation_limit', self.risk.correlation_limit)
        
        # Execution section
        if 'execution' in config_dict:
            exec_config = config_dict['execution']
            self.execution.use_vwap = exec_config.get('use_vwap', self.execution.use_vwap)
            self.execution.participation_rate = exec_config.get('participation_rate', self.execution.participation_rate)
            self.execution.trading_start_time = exec_config.get('trading_start_time', self.execution.trading_start_time)
            self.execution.trading_end_time = exec_config.get('trading_end_time', self.execution.trading_end_time)
            self.execution.commission_rate = exec_config.get('commission_rate', self.execution.commission_rate)
            self.execution.bid_ask_spread_assumption = exec_config.get('bid_ask_spread_assumption', self.execution.bid_ask_spread_assumption)
            self.execution.paper_trading_enabled = exec_config.get('paper_trading_enabled', self.execution.paper_trading_enabled)
        
        # Data section
        if 'data' in config_dict:
            data = config_dict['data']
            self.data.primary_price_source = data.get('primary_price_source', self.data.primary_price_source)
            self.data.backup_price_sources = data.get('backup_price_sources', self.data.backup_price_sources)
            self.data.data_path = data.get('data_path', self.data.data_path)
            self.data.use_database = data.get('use_database', self.data.use_database)
            self.data.database_type = data.get('database_type', self.data.database_type)
            self.data.data_update_time = data.get('data_update_time', self.data.data_update_time)
    
    def validate_config(self) -> List[str]:
        """Validate configuration parameters"""
        errors = []
        
        # Performance validation
        if self.performance.annual_alpha_target <= 0:
            errors.append("Alpha target must be positive")
        
        if self.performance.sharpe_ratio_target < 1.0:
            errors.append("Sharpe ratio target should be >= 1.0")
        
        if self.performance.max_drawdown_limit > 0.5:
            errors.append("Max drawdown limit too high (>50%)")
        
        # Portfolio validation
        if self.portfolio.max_positions < self.portfolio.min_positions:
            errors.append("Max positions must be >= min positions")
        
        if self.portfolio.long_allocation + self.portfolio.short_allocation != 1.0:
            errors.append("Long + short allocation must equal 1.0")
        
        if self.portfolio.max_single_position > 0.2:
            errors.append("Single position limit too high (>20%)")
        
        # Risk validation
        if self.risk.daily_loss_limit > 0.1:
            errors.append("Daily loss limit too high (>10%)")
        
        if self.risk.stop_loss_pct > 0.15:
            errors.append("Stop loss too wide (>15%)")
        
        return errors
    
    def print_summary(self):
        """Print configuration summary"""
        print("🎯 PSX Quantitative Trading System Configuration")
        print("=" * 60)
        
        print(f"\n📊 Performance Targets:")
        print(f"   Alpha Target: {self.performance.annual_alpha_target:.1%} vs KSE100")
        print(f"   Sharpe Target: {self.performance.sharpe_ratio_target:.1f}")
        print(f"   Max Drawdown: {self.performance.max_drawdown_limit:.1%}")
        print(f"   Target Vol: {self.performance.target_volatility:.1%}")
        
        print(f"\n💼 Portfolio Configuration:")
        print(f"   Positions: {self.portfolio.min_positions}-{self.portfolio.max_positions}")
        print(f"   Long/Short: {self.portfolio.long_allocation:.1%}/{self.portfolio.short_allocation:.1%}")
        print(f"   Max Single Position: {self.portfolio.max_single_position:.1%}")
        print(f"   Rebalance: {self.portfolio.rebalance_frequency.value}")
        
        print(f"\n🧠 Model Configuration:")
        print(f"   Primary Model: {self.model.primary_model}")
        print(f"   Training Window: {self.model.train_window_months} months")
        print(f"   Retrain Frequency: {self.model.retrain_frequency_days} days")
        print(f"   Forward Return: {self.model.label_forward_days} days")
        
        print(f"\n⚖️ Risk Management:")
        print(f"   Stop Loss: {self.risk.stop_loss_pct:.1%}")
        print(f"   Daily Loss Limit: {self.risk.daily_loss_limit:.1%}")
        print(f"   Drawdown Circuit Breaker: {self.risk.drawdown_circuit_breaker:.1%}")
        
        print(f"\n📈 Execution:")
        print(f"   Paper Trading: {'Enabled' if self.execution.paper_trading_enabled else 'Disabled'}")
        print(f"   Commission: {self.execution.commission_rate:.2%}")
        print(f"   Trading Hours: {self.execution.trading_start_time} - {self.execution.trading_end_time}")

# Factory function to create default configuration
def create_default_config() -> SystemConfig:
    """Create default system configuration"""
    return SystemConfig()

# Factory function to create conservative configuration
def create_conservative_config() -> SystemConfig:
    """Create conservative system configuration"""
    config = SystemConfig()
    
    # Conservative performance targets
    config.performance.annual_alpha_target = 0.04      # 4% alpha target
    config.performance.max_drawdown_limit = 0.10       # 10% max drawdown
    config.performance.max_daily_loss = 0.02           # 2% daily stop
    
    # Conservative portfolio settings
    config.portfolio.max_positions = 30                # Fewer positions
    config.portfolio.max_single_position = 0.03        # 3% max per position
    config.portfolio.net_exposure_target = 0.1         # 10% net exposure
    
    # Conservative risk settings
    config.risk.stop_loss_pct = 0.03                   # 3% stop loss
    config.risk.daily_loss_limit = 0.02                # 2% daily limit
    config.risk.drawdown_circuit_breaker = 0.10        # 10% circuit breaker
    
    return config

# Factory function to create aggressive configuration
def create_aggressive_config() -> SystemConfig:
    """Create aggressive system configuration"""
    config = SystemConfig()
    
    # Aggressive performance targets
    config.performance.annual_alpha_target = 0.10      # 10% alpha target
    config.performance.max_drawdown_limit = 0.25       # 25% max drawdown
    config.performance.target_volatility = 0.20        # 20% volatility
    
    # Aggressive portfolio settings
    config.portfolio.max_positions = 50                # More positions
    config.portfolio.max_single_position = 0.08        # 8% max per position
    config.portfolio.net_exposure_target = 0.3         # 30% net exposure
    
    # Aggressive risk settings
    config.risk.stop_loss_pct = 0.08                   # 8% stop loss
    config.risk.daily_loss_limit = 0.05                # 5% daily limit
    
    return config

# Test function
def test_system_config():
    """Test system configuration"""
    print("🚀 Testing System Configuration")
    print("=" * 40)
    
    # Create default config
    config = create_default_config()
    config.print_summary()
    
    # Validate configuration
    print(f"\n🔍 Validating Configuration...")
    errors = config.validate_config()
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   • {error}")
    else:
        print("✅ Configuration is valid")
    
    # Save configuration
    print(f"\n💾 Saving Configuration...")
    config_file = config.save_config()
    
    # Test conservative config
    print(f"\n🛡️ Conservative Configuration:")
    conservative = create_conservative_config()
    print(f"   Alpha Target: {conservative.performance.annual_alpha_target:.1%}")
    print(f"   Max Drawdown: {conservative.performance.max_drawdown_limit:.1%}")
    print(f"   Max Position: {conservative.portfolio.max_single_position:.1%}")
    
    print(f"\n🔥 Aggressive Configuration:")
    aggressive = create_aggressive_config()
    print(f"   Alpha Target: {aggressive.performance.annual_alpha_target:.1%}")
    print(f"   Max Drawdown: {aggressive.performance.max_drawdown_limit:.1%}")
    print(f"   Max Position: {aggressive.portfolio.max_single_position:.1%}")
    
    print(f"\n🏁 Configuration system test completed!")

if __name__ == "__main__":
    test_system_config()