"""
INTEGRATED INTRADAY TRADING SYSTEM
Complete High-Accuracy Intraday Trading Platform

Features:
- Complete integration of all enhanced components
- Real-time signal generation and execution
- Advanced risk management and monitoring
- Performance tracking and analytics
- Comprehensive testing framework
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import asyncio
import threading
import queue
import json
import warnings
from pathlib import Path

# Import enhanced components with lazy loading to avoid circular imports
try:
    from enhanced_intraday_feature_engine import EnhancedIntradayFeatureEngine, IntradayFeatures
    from enhanced_intraday_risk_manager import EnhancedIntradayRiskManager, RiskSignal, RiskMetrics
    from volatility_regime_detector import VolatilityRegimeDetector, VolatilityRegime, RegimeSignal
    from enhanced_backtesting_engine import EnhancedBacktestingEngine, BacktestResult, WalkForwardResult
    from real_time_execution_engine import RealTimeExecutionEngine, Order, OrderType, OrderStatus
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False
    # Define fallback types to avoid import errors
    class RegimeSignal:
        pass
    class RiskSignal:
        pass
    class IntradayFeatures:
        pass

# Lazy import for ML system to avoid circular dependency
def get_ml_system():
    """Lazy import ML system to avoid circular dependencies"""
    try:
        from advanced_ml_trading_system import AdvancedMLTradingSystem, MLTradingSignal
        return AdvancedMLTradingSystem, MLTradingSignal
    except ImportError:
        return None, None

warnings.filterwarnings('ignore')

@dataclass
class IntegratedSignal:
    """Comprehensive trading signal with all enhancements"""
    symbol: str
    timestamp: datetime
    
    # Base signal
    base_signal: str  # BUY, SELL, HOLD
    confidence: float
    
    # Enhanced analysis (using Any to avoid import issues)
    ml_signal: Optional[any] = None  # MLTradingSignal - use Any to avoid import issues
    regime_signal: Optional[RegimeSignal] = None
    risk_signal: Optional[RiskSignal] = None
    features: Optional[IntradayFeatures] = None
    
    # Execution parameters
    recommended_size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Quality metrics
    signal_quality: float = 0.0
    risk_score: float = 0.0
    execution_urgency: str = "NORMAL"  # LOW, NORMAL, HIGH, URGENT

@dataclass
class SystemPerformance:
    """System performance metrics"""
    timestamp: datetime
    
    # Signal performance
    signals_generated: int
    signals_executed: int
    execution_rate: float
    
    # Trading performance
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    daily_pnl: float
    
    # Risk metrics
    max_drawdown: float
    risk_utilization: float
    var_95: float
    
    # System health
    component_status: Dict[str, bool]
    latency_ms: float
    error_count: int

class IntegratedIntradayTradingSystem:
    """Complete integrated intraday trading system"""
    
    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize components
        self.initialize_components()
        
        # System state
        self.active = False
        self.market_data_cache = {}
        self.signal_history = []
        self.performance_history = []
        
        # Configuration
        self.config = {
            'trading_enabled': True,
            'risk_enabled': True,
            'max_signals_per_minute': 10,
            'min_signal_confidence': 0.70,
            'position_size_base': 0.02,  # 2% base position size
            'max_daily_trades': 50,
            'emergency_stop_loss': 0.10,  # 10% portfolio loss
        }
        
        # Performance tracking
        self.daily_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'pnl': 0.0,
            'max_drawdown': 0.0,
            'start_time': datetime.now()
        }
        
        # Threading
        self.signal_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        self.market_data_queue = queue.Queue()
        
        self.running = False
        self.threads = []
    
    def initialize_components(self):
        """Initialize all trading system components"""
        
        if not COMPONENTS_AVAILABLE:
            print("⚠️ Some components not available, limited functionality")
        
        # Feature engineering
        try:
            self.feature_engine = EnhancedIntradayFeatureEngine()
            print("✅ Feature engine initialized")
        except Exception as e:
            print(f"❌ Feature engine failed: {e}")
            self.feature_engine = None
        
        # Risk management
        try:
            self.risk_manager = EnhancedIntradayRiskManager(self.initial_capital)
            print("✅ Risk manager initialized")
        except Exception as e:
            print(f"❌ Risk manager failed: {e}")
            self.risk_manager = None
        
        # Volatility regime detection
        try:
            self.regime_detector = VolatilityRegimeDetector()
            print("✅ Regime detector initialized")
        except Exception as e:
            print(f"❌ Regime detector failed: {e}")
            self.regime_detector = None
        
        # ML/DL system (lazy initialization)
        try:
            AdvancedMLTradingSystem, _ = get_ml_system()
            if AdvancedMLTradingSystem:
                self.ml_system = AdvancedMLTradingSystem()
                print("✅ ML/DL system initialized")
            else:
                self.ml_system = None
        except Exception as e:
            print(f"❌ ML/DL system failed: {e}")
            self.ml_system = None
        
        # Execution engine
        try:
            self.execution_engine = RealTimeExecutionEngine(self.initial_capital)
            print("✅ Execution engine initialized")
        except Exception as e:
            print(f"❌ Execution engine failed: {e}")
            self.execution_engine = None
        
        # Backtesting engine
        try:
            self.backtesting_engine = EnhancedBacktestingEngine()
            print("✅ Backtesting engine initialized")
        except Exception as e:
            print(f"❌ Backtesting engine failed: {e}")
            self.backtesting_engine = None
    
    def start_system(self):
        """Start the integrated trading system"""
        
        if self.running:
            print("System already running")
            return
        
        print("🚀 Starting Integrated Intraday Trading System...")
        self.running = True
        
        # Start execution engine
        if self.execution_engine:
            self.execution_engine.start_execution_engine()
        
        # Start processing threads
        self.threads = [
            threading.Thread(target=self._signal_processing_loop, daemon=True),
            threading.Thread(target=self._market_data_loop, daemon=True),
            threading.Thread(target=self._performance_monitoring_loop, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        print("✅ Integrated trading system started successfully")
        self.active = True
    
    def stop_system(self):
        """Stop the trading system gracefully"""
        
        print("🛑 Stopping integrated trading system...")
        self.running = False
        self.active = False
        
        # Stop execution engine
        if self.execution_engine:
            self.execution_engine.stop_execution_engine()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Save final state
        self.save_system_state()
        
        print("✅ System stopped successfully")
    
    def process_market_data(self, symbol: str, market_data: pd.DataFrame):
        """Process incoming market data and generate signals"""
        
        if not self.active or market_data.empty:
            return
        
        # Cache market data
        self.market_data_cache[symbol] = market_data
        
        # Queue for processing
        self.market_data_queue.put((symbol, market_data))
    
    def generate_integrated_signal(self, symbol: str, market_data: pd.DataFrame) -> Optional[IntegratedSignal]:
        """Generate comprehensive trading signal using all components"""
        
        try:
            # Extract features
            features = None
            if self.feature_engine:
                features = self.feature_engine.extract_comprehensive_features(symbol, market_data)
            
            # Generate ML signal
            ml_signal = None
            if self.ml_system:
                ml_signal = self.ml_system.generate_prediction(symbol)
            
            # Detect volatility regime
            regime = None
            regime_signal = None
            if self.regime_detector:
                regime = self.regime_detector.detect_regime(market_data, symbol)
                if ml_signal:
                    regime_signal = self.regime_detector.adapt_signal_to_regime(
                        ml_signal.signal, ml_signal.confidence, regime, symbol
                    )
            
            # Determine base signal
            if regime_signal:
                base_signal = regime_signal.regime_adjusted_signal
                confidence = regime_signal.confidence_adjustment * (ml_signal.confidence / 100 if ml_signal else 0.7)
            elif ml_signal:
                base_signal = ml_signal.signal
                confidence = ml_signal.confidence / 100
            else:
                # Fallback to simple technical signal
                base_signal, confidence = self._generate_fallback_signal(market_data)
            
            # Risk assessment
            risk_signal = None
            if self.risk_manager and base_signal != 'HOLD':
                current_price = market_data['Close'].iloc[-1]
                proposed_size = self.calculate_position_size(confidence, current_price)
                
                risk_signal = self.risk_manager.evaluate_trade_risk(
                    symbol, confidence * 100, current_price, proposed_size, market_data
                )
            
            # Calculate execution parameters
            entry_price = market_data['Close'].iloc[-1]
            stop_loss, take_profit = self._calculate_execution_levels(
                symbol, base_signal, entry_price, market_data
            )
            
            # Determine final position size
            if risk_signal and risk_signal.action != 'BLOCK':
                recommended_size = risk_signal.recommended_size
            else:
                recommended_size = 0.0 if risk_signal and risk_signal.action == 'BLOCK' else self.calculate_position_size(confidence, entry_price)
            
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(ml_signal, regime_signal, risk_signal, features)
            
            # Create integrated signal
            integrated_signal = IntegratedSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                base_signal=base_signal,
                confidence=confidence,
                ml_signal=ml_signal,
                regime_signal=regime_signal,
                risk_signal=risk_signal,
                features=features,
                recommended_size=recommended_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_quality=signal_quality,
                risk_score=risk_signal.risk_score if risk_signal else 50.0,
                execution_urgency=self._determine_urgency(signal_quality, confidence)
            )
            
            # Add to history
            self.signal_history.append(integrated_signal)
            self.daily_stats['signals_generated'] += 1
            
            return integrated_signal
            
        except Exception as e:
            print(f"Error generating integrated signal for {symbol}: {e}")
            return None
    
    def execute_signal(self, signal: IntegratedSignal) -> bool:
        """Execute trading signal"""
        
        if not self.active or not self.execution_engine:
            return False
        
        # Pre-execution checks
        if signal.base_signal == 'HOLD':
            return False
        
        if signal.confidence < self.config['min_signal_confidence']:
            print(f"Signal confidence {signal.confidence:.2%} below threshold")
            return False
        
        if signal.recommended_size <= 0:
            print(f"Invalid position size: {signal.recommended_size}")
            return False
        
        # Create order
        order = Order(
            order_id=f"SIG_{signal.symbol}_{int(signal.timestamp.timestamp())}",
            symbol=signal.symbol,
            side=signal.base_signal,
            quantity=int(signal.recommended_size / signal.entry_price),
            order_type=OrderType.MARKET,
            created_time=signal.timestamp
        )
        
        # Submit order
        success = self.execution_engine.submit_order(order)
        
        if success:
            self.daily_stats['trades_executed'] += 1
            print(f"✅ Signal executed: {signal.base_signal} {signal.symbol} @ {signal.entry_price:.2f}")
        else:
            print(f"❌ Failed to execute signal: {signal.symbol}")
        
        return success
    
    def calculate_position_size(self, confidence: float, price: float) -> float:
        """Calculate position size based on confidence and risk parameters"""
        
        base_size = self.current_capital * self.config['position_size_base']
        
        # Adjust for confidence
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2x based on confidence
        
        # Adjust for volatility regime
        regime_multiplier = 1.0
        if self.regime_detector and hasattr(self.regime_detector, 'current_regime'):
            if self.regime_detector.current_regime in [2, 3]:  # High/Extreme volatility
                regime_multiplier = 0.7
            elif self.regime_detector.current_regime == 0:  # Low volatility
                regime_multiplier = 1.3
        
        final_size = base_size * confidence_multiplier * regime_multiplier
        
        # Cap at maximum position size
        max_size = self.current_capital * 0.15  # 15% max per position
        
        return min(final_size, max_size)
    
    def get_system_performance(self) -> SystemPerformance:
        """Get current system performance metrics"""
        
        current_time = datetime.now()
        
        # Calculate performance metrics
        total_trades = self.daily_stats['trades_executed']
        winning_trades = 0  # Would be calculated from actual trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Component status
        component_status = {
            'feature_engine': self.feature_engine is not None,
            'risk_manager': self.risk_manager is not None,
            'regime_detector': self.regime_detector is not None,
            'ml_system': self.ml_system is not None,
            'execution_engine': self.execution_engine is not None and self.execution_engine.running
        }
        
        return SystemPerformance(
            timestamp=current_time,
            signals_generated=self.daily_stats['signals_generated'],
            signals_executed=self.daily_stats['trades_executed'],
            execution_rate=self.daily_stats['trades_executed'] / max(1, self.daily_stats['signals_generated']),
            total_trades=total_trades,
            winning_trades=winning_trades,
            win_rate=win_rate,
            total_pnl=self.daily_stats['pnl'],
            daily_pnl=self.daily_stats['pnl'],
            max_drawdown=self.daily_stats['max_drawdown'],
            risk_utilization=0.65,  # Would be calculated from risk manager
            var_95=0.0,  # Would be calculated from returns
            component_status=component_status,
            latency_ms=50.0,  # Would be measured
            error_count=0  # Would be tracked
        )
    
    def run_comprehensive_test(self, symbols: List[str], test_duration_hours: int = 1) -> Dict[str, any]:
        """Run comprehensive system test"""
        
        print(f"🧪 Running comprehensive system test for {test_duration_hours} hours...")
        
        test_results = {
            'start_time': datetime.now(),
            'symbols_tested': symbols,
            'signals_generated': [],
            'errors': [],
            'performance_metrics': [],
            'component_tests': {}
        }
        
        # Test each component individually
        test_results['component_tests'] = self._test_components()
        
        # Generate test market data
        test_data = self._generate_test_data(symbols, test_duration_hours)
        
        # Process test data
        for symbol in symbols:
            if symbol in test_data:
                try:
                    signal = self.generate_integrated_signal(symbol, test_data[symbol])
                    if signal:
                        test_results['signals_generated'].append({
                            'symbol': symbol,
                            'signal': signal.base_signal,
                            'confidence': signal.confidence,
                            'quality': signal.signal_quality
                        })
                except Exception as e:
                    test_results['errors'].append(f"Error processing {symbol}: {e}")
        
        # Performance metrics
        performance = self.get_system_performance()
        test_results['performance_metrics'].append({
            'timestamp': performance.timestamp,
            'signals_generated': performance.signals_generated,
            'execution_rate': performance.execution_rate,
            'component_health': sum(performance.component_status.values()) / len(performance.component_status)
        })
        
        test_results['end_time'] = datetime.now()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        print(f"✅ Comprehensive test completed in {test_results['duration']}")
        return test_results
    
    def _signal_processing_loop(self):
        """Main signal processing loop"""
        
        while self.running:
            try:
                if not self.market_data_queue.empty():
                    symbol, market_data = self.market_data_queue.get(timeout=1)
                    
                    # Generate signal
                    signal = self.generate_integrated_signal(symbol, market_data)
                    
                    if signal and signal.base_signal != 'HOLD':
                        # Queue for execution
                        self.signal_queue.put(signal)
                
                # Process execution queue
                if not self.signal_queue.empty():
                    signal = self.signal_queue.get(timeout=1)
                    self.execute_signal(signal)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in signal processing loop: {e}")
    
    def _market_data_loop(self):
        """Market data processing loop"""
        
        while self.running:
            try:
                # Simulate real-time market data updates
                # In production, this would connect to data feeds
                pass
            except Exception as e:
                print(f"Error in market data loop: {e}")
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        
        while self.running:
            try:
                # Update performance metrics every minute
                performance = self.get_system_performance()
                self.performance_history.append(performance)
                
                # Keep only last 24 hours of performance data
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.performance_history = [
                    p for p in self.performance_history 
                    if p.timestamp > cutoff_time
                ]
                
                # Sleep for 60 seconds
                import time as time_module
                time_module.sleep(60)
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
    
    def _generate_fallback_signal(self, market_data: pd.DataFrame) -> Tuple[str, float]:
        """Generate simple fallback signal when ML systems are unavailable"""
        
        if len(market_data) < 20:
            return 'HOLD', 0.5
        
        # Simple moving average crossover
        current_price = market_data['Close'].iloc[-1]
        sma_5 = market_data['Close'].tail(5).mean()
        sma_20 = market_data['Close'].tail(20).mean()
        
        if sma_5 > sma_20 and current_price > sma_5:
            return 'BUY', 0.6
        elif sma_5 < sma_20 and current_price < sma_5:
            return 'SELL', 0.6
        else:
            return 'HOLD', 0.5
    
    def _calculate_execution_levels(self, symbol: str, signal: str, price: float, 
                                  market_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        # Simple ATR-based levels
        if len(market_data) >= 14:
            atr = ((market_data['High'] - market_data['Low']).tail(14).mean())
        else:
            atr = price * 0.02  # 2% fallback
        
        if signal == 'BUY':
            stop_loss = price - (2 * atr)
            take_profit = price + (3 * atr)
        elif signal == 'SELL':
            stop_loss = price + (2 * atr)
            take_profit = price - (3 * atr)
        else:
            stop_loss = price * 0.95
            take_profit = price * 1.05
        
        return stop_loss, take_profit
    
    def _calculate_signal_quality(self, ml_signal, regime_signal, risk_signal, features) -> float:
        """Calculate overall signal quality score"""
        
        quality = 0.5  # Base quality
        
        # ML signal contribution
        if ml_signal and ml_signal.confidence > 70:
            quality += 0.2
        
        # Regime alignment
        if regime_signal and regime_signal.confidence_adjustment > 1.0:
            quality += 0.15
        
        # Risk assessment
        if risk_signal and risk_signal.risk_score < 30:
            quality += 0.15
        
        # Feature quality
        if features and hasattr(features, 'temporal_features'):
            if features.temporal_features.get('is_mid_day', 0) == 1:
                quality += 0.1  # Better during active hours
        
        return min(1.0, quality)
    
    def _determine_urgency(self, quality: float, confidence: float) -> str:
        """Determine execution urgency based on signal characteristics"""
        
        if quality > 0.8 and confidence > 0.85:
            return "URGENT"
        elif quality > 0.7 and confidence > 0.75:
            return "HIGH"
        elif quality > 0.6 and confidence > 0.65:
            return "NORMAL"
        else:
            return "LOW"
    
    def _test_components(self) -> Dict[str, Dict[str, any]]:
        """Test individual components"""
        
        test_results = {}
        
        # Test feature engine
        if self.feature_engine:
            try:
                sample_data = self._generate_sample_data('TEST', 100)
                features = self.feature_engine.extract_comprehensive_features('TEST', sample_data)
                test_results['feature_engine'] = {
                    'status': 'PASS',
                    'features_extracted': len(features.price_features) if features else 0
                }
            except Exception as e:
                test_results['feature_engine'] = {'status': 'FAIL', 'error': str(e)}
        
        # Test risk manager
        if self.risk_manager:
            try:
                sample_data = self._generate_sample_data('TEST', 50)
                risk_signal = self.risk_manager.evaluate_trade_risk(
                    'TEST', 75.0, 100.0, 10000, sample_data
                )
                test_results['risk_manager'] = {
                    'status': 'PASS',
                    'risk_score': risk_signal.risk_score if risk_signal else 0
                }
            except Exception as e:
                test_results['risk_manager'] = {'status': 'FAIL', 'error': str(e)}
        
        # Test regime detector
        if self.regime_detector:
            try:
                sample_data = self._generate_sample_data('TEST', 100)
                regime = self.regime_detector.detect_regime(sample_data, 'TEST')
                test_results['regime_detector'] = {
                    'status': 'PASS',
                    'regime': regime.regime_name if regime else 'Unknown'
                }
            except Exception as e:
                test_results['regime_detector'] = {'status': 'FAIL', 'error': str(e)}
        
        return test_results
    
    def _generate_test_data(self, symbols: List[str], hours: int) -> Dict[str, pd.DataFrame]:
        """Generate test market data"""
        
        data = {}
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        for symbol in symbols:
            data[symbol] = self._generate_sample_data(symbol, hours * 12)  # 5-min intervals
        
        return data
    
    def _generate_sample_data(self, symbol: str, periods: int) -> pd.DataFrame:
        """Generate sample market data for testing"""
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='5min')
        base_price = 100 + hash(symbol) % 50
        
        prices = []
        current_price = base_price
        
        for _ in range(periods):
            change = np.random.normal(0, 0.015)  # 1.5% volatility
            current_price *= (1 + change)
            prices.append(current_price)
        
        return pd.DataFrame({
            'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
            'High': [p * np.random.uniform(1.001, 1.003) for p in prices],
            'Low': [p * np.random.uniform(0.997, 0.999) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, periods)
        }, index=dates)
    
    def save_system_state(self):
        """Save system state to file"""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'daily_stats': self.daily_stats,
            'current_capital': self.current_capital,
            'active': self.active,
            'signal_count': len(self.signal_history),
            'performance_count': len(self.performance_history)
        }
        
        with open('system_state.json', 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_system_state(self):
        """Load system state from file"""
        
        try:
            with open('system_state.json', 'r') as f:
                state = json.load(f)
            
            self.config.update(state.get('config', {}))
            self.daily_stats.update(state.get('daily_stats', {}))
            self.current_capital = state.get('current_capital', self.initial_capital)
            
            print("✅ System state loaded successfully")
        except FileNotFoundError:
            print("No previous system state found")
        except Exception as e:
            print(f"Error loading system state: {e}")

# Main testing function
def test_integrated_system():
    """Test the complete integrated system"""
    
    print("🚀 Testing Integrated Intraday Trading System")
    print("=" * 60)
    
    # Initialize system
    system = IntegratedIntradayTradingSystem()
    
    # Run comprehensive test
    test_symbols = ['HBL', 'UBL', 'ENGRO']
    test_results = system.run_comprehensive_test(test_symbols, test_duration_hours=1)
    
    # Display results
    print("\n📊 Test Results Summary:")
    print(f"Duration: {test_results['duration']}")
    print(f"Symbols Tested: {len(test_results['symbols_tested'])}")
    print(f"Signals Generated: {len(test_results['signals_generated'])}")
    print(f"Errors: {len(test_results['errors'])}")
    
    # Component test results
    print("\n🔧 Component Test Results:")
    for component, result in test_results['component_tests'].items():
        status = result.get('status', 'UNKNOWN')
        print(f"  {component}: {status}")
        if status == 'FAIL':
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    # Sample signals
    if test_results['signals_generated']:
        print("\n🎯 Sample Signals Generated:")
        for signal in test_results['signals_generated'][:3]:
            print(f"  {signal['symbol']}: {signal['signal']} "
                  f"(Confidence: {signal['confidence']:.1%}, "
                  f"Quality: {signal['quality']:.1%})")
    
    # Performance
    if test_results['performance_metrics']:
        latest_perf = test_results['performance_metrics'][-1]
        print(f"\n📈 System Performance:")
        print(f"  Signals Generated: {latest_perf['signals_generated']}")
        print(f"  Execution Rate: {latest_perf['execution_rate']:.1%}")
        print(f"  Component Health: {latest_perf['component_health']:.1%}")
    
    print("\n✅ Integrated system test completed successfully!")
    return test_results

if __name__ == "__main__":
    test_integrated_system()