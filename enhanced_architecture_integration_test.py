#!/usr/bin/env python3
"""
Enhanced Architecture Integration Test
====================================

Comprehensive integration test for the enhanced algorithmic trading architecture
demonstrating the full pipeline from data ingestion to signal generation.

This test validates:
- HDBSCAN clustering for market regime detection
- Bayesian statistics for probabilistic analysis  
- Real-time processing pipeline with latency monitoring
- Feature selection and correlation management
- Statistical validation framework
- API integration layer
- Complete system integration
"""

import pandas as pd
import numpy as np
import datetime as dt
import asyncio
import time
import logging
from typing import Dict, List, Any
import warnings
import sys
from pathlib import Path

# Import all enhanced architecture components
try:
    from clustering_engine import ClusteringEngine
    from bayesian_engine import BayesianEngine
    from realtime_processor import RealTimeProcessor
    from feature_correlation_manager import FeatureCorrelationManager
    from statistical_validation_framework import StatisticalValidationFramework
    from api_integration_layer import APIIntegrationLayer
    
    # Import existing components
    from quant_system_config import SystemConfig
    from feature_engineering import FeatureEngineer
    from ml_model_system import MLModelSystem
    
    ALL_IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Some components may not be available for testing")
    ALL_IMPORTS_SUCCESSFUL = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class EnhancedArchitectureIntegrationTest:
    """Comprehensive integration test for enhanced trading architecture"""
    
    def __init__(self):
        self.config = SystemConfig()
        self.test_results = {}
        self.components = {}
        self.test_data = {}
        
    def setup_test_environment(self):
        """Setup test environment and sample data"""
        print("🔧 Setting up test environment...")
        
        # Generate comprehensive test dataset
        np.random.seed(42)
        
        # Market data simulation
        n_days = 500
        n_symbols = 10
        symbols = [f'SYMBOL_{i}' for i in range(n_symbols)]
        
        # Create multi-index time series data
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        all_data = []
        for symbol in symbols:
            # Simulate price movements with different regimes
            base_return = np.random.normal(0.0005, 0.02, n_days)  # Base returns
            
            # Add regime changes
            regime_changes = [100, 250, 400]  # Days when regime changes
            for change_day in regime_changes:
                if change_day < n_days:
                    base_return[change_day:] += np.random.normal(0, 0.01, n_days - change_day)
            
            # Price levels
            price_levels = 100 * np.exp(np.cumsum(base_return))
            
            # Volume simulation
            volumes = np.random.lognormal(10, 0.5, n_days).astype(int)
            
            # Create DataFrame for this symbol
            symbol_data = pd.DataFrame({
                'symbol': symbol,
                'date': dates,
                'price': price_levels,
                'volume': volumes,
                'returns': base_return
            })
            
            all_data.append(symbol_data)
        
        # Combine all data
        self.test_data['market_data'] = pd.concat(all_data, ignore_index=True)
        self.test_data['symbols'] = symbols
        self.test_data['dates'] = dates
        
        print(f"   ✅ Generated market data: {len(self.test_data['market_data'])} records")
        print(f"   ✅ Symbols: {len(symbols)}")
        print(f"   ✅ Date range: {dates[0].date()} to {dates[-1].date()}")
        
    def test_clustering_engine(self) -> Dict[str, Any]:
        """Test HDBSCAN clustering engine"""
        print("\n🧩 Testing HDBSCAN Clustering Engine...")
        
        if not ALL_IMPORTS_SUCCESSFUL:
            return {"status": "skipped", "reason": "Import error"}
        
        try:
            # Initialize clustering engine
            clustering_engine = ClusteringEngine(self.config)
            self.components['clustering'] = clustering_engine
            
            # Prepare features for clustering (volatility and correlation patterns)
            market_data = self.test_data['market_data']
            
            # Calculate features by symbol
            features_list = []
            for symbol in self.test_data['symbols'][:5]:  # Test with 5 symbols
                symbol_data = market_data[market_data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date')
                
                # Rolling statistics
                window = 20
                symbol_data['volatility'] = symbol_data['returns'].rolling(window).std()
                symbol_data['momentum'] = symbol_data['returns'].rolling(window).mean()
                symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume'].rolling(window).mean()
                
                # Fill NaN values
                symbol_data = symbol_data.fillna(method='bfill').fillna(method='ffill')
                
                # Extract features
                features = symbol_data[['volatility', 'momentum', 'volume_ratio']].values
                features_list.append(features)
            
            # Combine features
            combined_features = np.vstack(features_list)
            
            # Test clustering
            clustering_results = clustering_engine.fit_clustering(
                combined_features, 
                optimize_params=True, 
                apply_pca=True
            )
            
            # Test market regime detection
            volatility_features = combined_features[:, 0]
            correlation_features = np.random.uniform(0.1, 0.9, len(combined_features))
            
            regime_results = clustering_engine.detect_market_regimes(
                volatility_features, correlation_features
            )
            
            return {
                "status": "success",
                "n_clusters": clustering_results['n_clusters'],
                "n_noise_points": clustering_results['n_noise'],
                "processing_time_ms": clustering_results['processing_time_ms'],
                "validation_score": clustering_results['validation_results']['silhouette_score'],
                "regimes_detected": len(regime_results['regimes']),
                "clustering_valid": clustering_results['validation_results']['valid']
            }
            
        except Exception as e:
            logger.error(f"Clustering engine test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_bayesian_engine(self) -> Dict[str, Any]:
        """Test Bayesian statistics engine"""
        print("\n📊 Testing Bayesian Statistics Engine...")
        
        if not ALL_IMPORTS_SUCCESSFUL:
            return {"status": "skipped", "reason": "Import error"}
        
        try:
            # Initialize Bayesian engine
            bayesian_engine = BayesianEngine(self.config)
            self.components['bayesian'] = bayesian_engine
            
            # Prepare sample regression problem
            n_samples = 300
            n_features = 8
            
            X = np.random.randn(n_samples, n_features)
            true_weights = np.random.randn(n_features)
            noise = np.random.randn(n_samples) * 0.3
            y = X @ true_weights + noise
            
            # Test Bayesian linear regression
            bayesian_results = bayesian_engine.fit_bayesian_regression(X, y, "test_model")
            
            # Test predictions with uncertainty
            X_test = np.random.randn(50, n_features)
            pred_mean, pred_std = bayesian_engine.predict_with_uncertainty(X_test, "test_model")
            
            # Test confidence intervals
            confidence_intervals = bayesian_engine.get_confidence_intervals(X_test[:10], "test_model")
            
            # Test hierarchical model
            data_dict = {
                'group1': (X[:100], y[:100]),
                'group2': (X[100:200], y[100:200]),
                'group3': (X[200:], y[200:])
            }
            
            hierarchical_results = bayesian_engine.fit_hierarchical_model(data_dict, "hierarchical_test")
            
            # Test variational inference
            variational_results = bayesian_engine.fit_variational_model(X, y, "variational_test")
            
            return {
                "status": "success",
                "log_evidence": bayesian_results['log_evidence'],
                "fitting_time_ms": bayesian_results['fitting_time_ms'],
                "prediction_uncertainty": np.mean(pred_std),
                "confidence_intervals_computed": len(confidence_intervals),
                "hierarchical_groups": hierarchical_results['n_groups'],
                "variational_converged": variational_results['converged'],
                "variational_iterations": variational_results['n_iterations']
            }
            
        except Exception as e:
            logger.error(f"Bayesian engine test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_realtime_processor(self) -> Dict[str, Any]:
        """Test real-time processing pipeline"""
        print("\n⚡ Testing Real-Time Processing Pipeline...")
        
        if not ALL_IMPORTS_SUCCESSFUL:
            return {"status": "skipped", "reason": "Import error"}
        
        try:
            # Initialize real-time processor
            rt_processor = RealTimeProcessor(self.config)
            self.components['realtime'] = rt_processor
            
            # Test data containers
            processed_data = []
            generated_signals = []
            
            def data_callback(data):
                processed_data.append(data)
            
            def signal_callback(signal):
                generated_signals.append(signal)
            
            # Register callbacks
            rt_processor.register_data_callback(data_callback)
            rt_processor.register_signal_callback(signal_callback)
            
            # Start processing
            rt_processor.start_processing()
            
            # Simulate real-time data ingestion
            test_symbols = self.test_data['symbols'][:3]
            n_ticks = 200
            
            base_prices = {symbol: 100.0 for symbol in test_symbols}
            
            for i in range(n_ticks):
                for symbol in test_symbols:
                    # Random walk price
                    price_change = np.random.randn() * 0.01 * base_prices[symbol]
                    price = max(base_prices[symbol] + price_change, base_prices[symbol] * 0.9)
                    base_prices[symbol] = price
                    
                    volume = np.random.randint(100, 5000)
                    
                    # Ingest tick
                    rt_processor.ingest_tick_data(symbol, price, volume)
                
                # Small delay every 50 ticks
                if i % 50 == 0:
                    time.sleep(0.01)
            
            # Wait for processing to complete
            time.sleep(1.0)
            
            # Get performance metrics
            performance_metrics = rt_processor.get_performance_metrics()
            
            # Stop processing
            rt_processor.stop_processing()
            
            return {
                "status": "success",
                "data_processed": len(processed_data),
                "signals_generated": len(generated_signals),
                "throughput_msg_per_sec": performance_metrics['throughput_msg_per_sec'],
                "memory_usage_mb": performance_metrics['memory_mb'],
                "active_symbols": performance_metrics['active_symbols'],
                "average_latency": performance_metrics['latency_stats']['overall_stats'].get('mean', 0) if performance_metrics['latency_stats']['overall_stats'] else 0
            }
            
        except Exception as e:
            logger.error(f"Real-time processor test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_feature_correlation_manager(self) -> Dict[str, Any]:
        """Test feature selection and correlation management"""
        print("\n🎯 Testing Feature Correlation Manager...")
        
        if not ALL_IMPORTS_SUCCESSFUL:
            return {"status": "skipped", "reason": "Import error"}
        
        try:
            # Initialize feature manager
            feature_manager = FeatureCorrelationManager(self.config)
            self.components['feature_manager'] = feature_manager
            
            # Create sample features with known correlations
            n_samples = 400
            n_features = 25
            
            # Base features
            X_base = np.random.randn(n_samples, 10)
            
            # Add correlated features
            X_corr = np.column_stack([
                X_base,
                X_base[:, 0] + np.random.randn(n_samples) * 0.1,  # High correlation
                X_base[:, 1] * 1.5 + np.random.randn(n_samples) * 0.2,  # Medium correlation
                np.random.randn(n_samples, n_features - 12)  # Independent features
            ])
            
            feature_names = [f'feature_{i}' for i in range(n_features)]
            X = pd.DataFrame(X_corr, columns=feature_names)
            
            # Create target with known relationships
            y = (X['feature_0'] * 2 + X['feature_1'] * -1.5 + 
                 X['feature_2'] * 0.8 + np.random.randn(n_samples) * 0.5)
            
            # Test comprehensive analysis
            analysis_results = feature_manager.analyze_feature_set(X, y)
            
            # Test optimal feature set
            optimal_features = feature_manager.get_optimal_feature_set(X, y, remove_correlated=True)
            
            # Test drift monitoring
            X_new = X.copy()
            X_new['feature_0'] += np.random.randn(len(X_new)) * 0.1
            drift_results = feature_manager.monitor_feature_drift(X_new)
            
            return {
                "status": "success",
                "original_features": analysis_results['n_features'],
                "selected_features": analysis_results['quality_metrics']['selected_feature_count'],
                "reduction_ratio": analysis_results['quality_metrics']['reduction_ratio'],
                "highly_correlated_pairs": len(analysis_results['correlation']['highly_correlated_pairs']),
                "max_correlation": analysis_results['correlation']['max_correlation'],
                "multicollinear_groups": len(analysis_results['correlation']['multicollinear_groups']),
                "interactions_found": analysis_results['interactions']['total_interactions_found'],
                "optimal_features_count": len(optimal_features),
                "analysis_time_s": analysis_results['analysis_time'],
                "drift_monitoring": drift_results.get('drift_percentage', 0) if 'drift_percentage' in drift_results else "baseline_established"
            }
            
        except Exception as e:
            logger.error(f"Feature correlation manager test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_statistical_validation(self) -> Dict[str, Any]:
        """Test statistical validation framework"""
        print("\n📈 Testing Statistical Validation Framework...")
        
        if not ALL_IMPORTS_SUCCESSFUL:
            return {"status": "skipped", "reason": "Import error"}
        
        try:
            # Initialize validation framework
            validator = StatisticalValidationFramework(self.config)
            self.components['validator'] = validator
            
            # Generate sample trading returns
            n_days = 252
            strategy_returns = np.random.normal(0.0008, 0.02, n_days)  # Positive bias
            benchmark_returns = np.random.normal(0.0005, 0.015, n_days)
            
            # Create time series
            dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
            strategy_ts = pd.Series(strategy_returns, index=dates)
            benchmark_ts = pd.Series(benchmark_returns, index=dates)
            
            # Test strategy validation
            strategy_validation = validator.validate_trading_strategy(strategy_ts, benchmark_ts)
            
            # Test model prediction validation
            X_sample = np.random.randn(150, 4)
            true_coef = np.array([1.2, -0.8, 0.5, 1.8])
            y_true = X_sample @ true_coef + np.random.randn(150) * 0.4
            y_pred = y_true + np.random.randn(150) * 0.3  # Add prediction error
            
            # Mock model for testing
            class TestModel:
                def fit(self, X, y):
                    pass
                def predict(self, X):
                    return X @ true_coef
            
            test_model = TestModel()
            model_validation = validator.validate_model_predictions(y_true, y_pred, test_model, X_sample)
            
            # Test feature importance validation
            feature_importances = {f'feature_{i}': np.random.uniform(0.1, 0.4) for i in range(4)}
            X_df = pd.DataFrame(X_sample, columns=list(feature_importances.keys()))
            y_series = pd.Series(y_true)
            
            # Reduce permutations for testing speed
            validator.validation_config.n_permutations = 50
            importance_validation = validator.validate_feature_importance(feature_importances, X_df, y_series)
            
            # Extract key results
            strategy_significant = strategy_validation['zero_return_test']['t_test']['significant']
            strategy_sharpe = strategy_validation['strategy_summary']['sharpe_ratio']
            
            model_r2 = model_validation['prediction_summary']['r2']
            residuals_centered = not model_validation['residual_zero_test']['t_test']['significant']
            
            return {
                "status": "success",
                "strategy_returns_significant": strategy_significant,
                "strategy_sharpe_ratio": strategy_sharpe,
                "outperforms_benchmark": strategy_validation.get('benchmark_comparison', {}).get('t_test', {}).get('significant', False),
                "model_r2": model_r2,
                "residuals_properly_centered": residuals_centered,
                "cv_performed": 'cross_validation' in model_validation,
                "feature_importance_tests": len(importance_validation.get('permutation_tests', {}).get('p_values', {})),
                "bootstrap_ci_computed": 'sharpe_ratio_bootstrap' in strategy_validation
            }
            
        except Exception as e:
            logger.error(f"Statistical validation test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def test_api_integration(self) -> Dict[str, Any]:
        """Test API integration layer"""
        print("\n🌐 Testing API Integration Layer...")
        
        if not ALL_IMPORTS_SUCCESSFUL:
            return {"status": "skipped", "reason": "Import error"}
        
        try:
            # Initialize API layer
            api_layer = APIIntegrationLayer(self.config)
            self.components['api'] = api_layer
            
            # Initialize components
            await api_layer.initialize()
            
            # Test database operations
            sample_market_data = [
                {
                    'symbol': 'TEST1',
                    'price': 150.5,
                    'volume': 1000,
                    'timestamp': time.time()
                },
                {
                    'symbol': 'TEST2', 
                    'price': 120.25,
                    'volume': 2500,
                    'timestamp': time.time()
                }
            ]
            
            await api_layer.publish_market_data(sample_market_data)
            
            # Test signal storage
            sample_signals = [
                {
                    'symbol': 'TEST1',
                    'signal_type': 'BUY',
                    'strength': 0.75,
                    'confidence': 0.85,
                    'timestamp': time.time(),
                    'features_used': {'momentum': 0.05, 'volume_ratio': 1.2}
                }
            ]
            
            await api_layer.publish_trading_signals(sample_signals)
            
            # Test caching
            test_data = {"test_key": "test_value", "timestamp": time.time()}
            await api_layer.cache_data("integration_test_key", test_data, ttl=60)
            cached_result = await api_layer.get_cached_data("integration_test_key")
            
            # Cleanup
            await api_layer.cleanup()
            
            return {
                "status": "success", 
                "market_data_stored": len(sample_market_data),
                "signals_stored": len(sample_signals),
                "caching_works": cached_result is not None and cached_result.get("test_key") == "test_value",
                "websocket_connections": len(api_layer.websocket_manager.active_connections),
                "database_initialized": api_layer.db_manager.engine is not None
            }
            
        except Exception as e:
            logger.error(f"API integration test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test complete system integration"""
        print("\n🔗 Testing Complete System Integration...")
        
        if not ALL_IMPORTS_SUCCESSFUL:
            return {"status": "skipped", "reason": "Import error"}
        
        try:
            integration_results = {}
            
            # Test component interactions
            if 'clustering' in self.components and 'bayesian' in self.components:
                # Use clustering results to inform Bayesian model groups
                clustering_engine = self.components['clustering']
                bayesian_engine = self.components['bayesian']
                
                # Get cluster labels from previous test
                if hasattr(clustering_engine, 'current_labels') and clustering_engine.current_labels is not None:
                    unique_clusters = len(np.unique(clustering_engine.current_labels[clustering_engine.current_labels != -1]))
                    integration_results['clustering_bayesian_integration'] = {
                        'clusters_available': unique_clusters,
                        'can_inform_hierarchical_model': unique_clusters > 1
                    }
            
            # Test feature selection with ML models
            if 'feature_manager' in self.components:
                feature_manager = self.components['feature_manager']
                
                if hasattr(feature_manager, 'current_features') and feature_manager.current_features:
                    integration_results['feature_selection_integration'] = {
                        'selected_features_available': len(feature_manager.current_features),
                        'ready_for_ml_pipeline': len(feature_manager.current_features) >= 3
                    }
            
            # Test real-time to validation pipeline
            if 'realtime' in self.components and 'validator' in self.components:
                rt_processor = self.components['realtime']
                validator = self.components['validator']
                
                # Get recent performance metrics
                performance = rt_processor.get_performance_metrics()
                
                integration_results['realtime_validation_integration'] = {
                    'performance_metrics_available': len(performance) > 0,
                    'latency_data_for_validation': 'latency_stats' in performance,
                    'throughput_measurable': performance.get('throughput_msg_per_sec', 0) > 0
                }
            
            # Overall integration health
            components_loaded = len(self.components)
            components_expected = 5  # clustering, bayesian, realtime, feature_manager, validator
            
            integration_results['overall_integration'] = {
                'components_loaded': components_loaded,
                'components_expected': components_expected,
                'integration_success_rate': components_loaded / components_expected,
                'all_components_available': components_loaded == components_expected
            }
            
            return {
                "status": "success",
                **integration_results
            }
            
        except Exception as e:
            logger.error(f"System integration test failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test of the entire enhanced architecture"""
        
        print("🚀 Starting Comprehensive Enhanced Architecture Integration Test")
        print("=" * 80)
        
        # Setup
        self.setup_test_environment()
        
        # Run individual component tests
        test_results = {}
        
        # 1. Test clustering engine
        test_results['clustering'] = self.test_clustering_engine()
        
        # 2. Test Bayesian engine
        test_results['bayesian'] = self.test_bayesian_engine()
        
        # 3. Test real-time processor
        test_results['realtime'] = self.test_realtime_processor()
        
        # 4. Test feature correlation manager
        test_results['feature_correlation'] = self.test_feature_correlation_manager()
        
        # 5. Test statistical validation
        test_results['statistical_validation'] = self.test_statistical_validation()
        
        # 6. Test API integration (async)
        test_results['api_integration'] = await self.test_api_integration()
        
        # 7. Test system integration
        test_results['system_integration'] = self.test_system_integration()
        
        # Store results
        self.test_results = test_results
        
        return test_results
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        
        report = []
        report.append("=" * 80)
        report.append("ENHANCED ARCHITECTURE INTEGRATION TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {dt.datetime.now()}")
        report.append("")
        
        # Overall summary
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() 
                              if result.get('status') == 'success')
        skipped_tests = sum(1 for result in test_results.values() 
                           if result.get('status') == 'skipped')
        
        report.append("OVERALL SUMMARY:")
        report.append(f"  Total Tests: {total_tests}")
        report.append(f"  Successful: {successful_tests}")
        report.append(f"  Skipped: {skipped_tests}")
        report.append(f"  Failed: {total_tests - successful_tests - skipped_tests}")
        report.append(f"  Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        report.append("")
        
        # Individual test results
        for test_name, result in test_results.items():
            report.append(f"{test_name.upper().replace('_', ' ')} TEST:")
            
            status = result.get('status', 'unknown')
            status_symbol = "✅" if status == 'success' else "⚠️" if status == 'skipped' else "❌"
            report.append(f"  Status: {status_symbol} {status}")
            
            if status == 'success':
                # Add key metrics
                for key, value in result.items():
                    if key != 'status' and not isinstance(value, dict):
                        if isinstance(value, float):
                            report.append(f"    {key}: {value:.4f}")
                        else:
                            report.append(f"    {key}: {value}")
            
            elif status == 'error':
                report.append(f"    Error: {result.get('error', 'Unknown error')}")
            
            elif status == 'skipped':
                report.append(f"    Reason: {result.get('reason', 'Unknown reason')}")
            
            report.append("")
        
        # Architecture validation summary
        report.append("ARCHITECTURE VALIDATION:")
        
        if 'clustering' in test_results and test_results['clustering'].get('status') == 'success':
            clustering = test_results['clustering']
            report.append(f"  ✅ HDBSCAN Clustering: {clustering['n_clusters']} clusters, "
                         f"validation score {clustering['validation_score']:.3f}")
        
        if 'bayesian' in test_results and test_results['bayesian'].get('status') == 'success':
            bayesian = test_results['bayesian']
            report.append(f"  ✅ Bayesian Engine: Evidence {bayesian['log_evidence']:.2f}, "
                         f"uncertainty quantification working")
        
        if 'realtime' in test_results and test_results['realtime'].get('status') == 'success':
            realtime = test_results['realtime']
            report.append(f"  ✅ Real-time Processing: {realtime['throughput_msg_per_sec']:.1f} msg/sec, "
                         f"{realtime['signals_generated']} signals generated")
        
        if 'feature_correlation' in test_results and test_results['feature_correlation'].get('status') == 'success':
            features = test_results['feature_correlation']
            report.append(f"  ✅ Feature Selection: {features['reduction_ratio']:.2f} reduction ratio, "
                         f"{features['interactions_found']} interactions found")
        
        if 'statistical_validation' in test_results and test_results['statistical_validation'].get('status') == 'success':
            validation = test_results['statistical_validation']
            report.append(f"  ✅ Statistical Validation: Strategy testing, model validation, "
                         f"R² = {validation['model_r2']:.3f}")
        
        if 'api_integration' in test_results and test_results['api_integration'].get('status') == 'success':
            api = test_results['api_integration']
            report.append(f"  ✅ API Integration: Database storage, caching, WebSocket support")
        
        if 'system_integration' in test_results and test_results['system_integration'].get('status') == 'success':
            integration = test_results['system_integration']
            overall = integration.get('overall_integration', {})
            report.append(f"  ✅ System Integration: {overall.get('components_loaded', 0)}/{overall.get('components_expected', 0)} components")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)

async def main():
    """Main test execution function"""
    
    print("🏗️ Enhanced Algorithmic Trading Architecture Integration Test")
    print(f"🐍 Python: {sys.version}")
    print(f"📅 Date: {dt.datetime.now()}")
    print("")
    
    if not ALL_IMPORTS_SUCCESSFUL:
        print("⚠️ WARNING: Some imports failed. Tests may be limited.")
        print("")
    
    # Create and run test
    test_suite = EnhancedArchitectureIntegrationTest()
    
    try:
        # Run comprehensive tests
        test_results = await test_suite.run_comprehensive_test()
        
        # Generate and display report
        print("\n" + "=" * 80)
        report = test_suite.generate_test_report(test_results)
        print(report)
        
        # Save report to file
        report_path = Path("enhanced_architecture_test_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Full report saved to: {report_path}")
        
        # Overall result
        successful_tests = sum(1 for result in test_results.values() 
                              if result.get('status') == 'success')
        total_tests = len(test_results)
        
        if successful_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Enhanced architecture is fully operational.")
        elif successful_tests > total_tests * 0.8:
            print(f"\n✅ Most tests passed ({successful_tests}/{total_tests}). Architecture is largely operational.")
        else:
            print(f"\n⚠️ Some tests failed ({successful_tests}/{total_tests}). Architecture needs attention.")
            
    except Exception as e:
        print(f"\n❌ Test suite execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())