"""
INTEGRATED SYSTEM TEST RUNNER
Simple test script to verify all components work together
"""

import sys
import traceback
from datetime import datetime

def test_feature_engine():
    """Test feature engine component"""
    try:
        from enhanced_intraday_feature_engine import EnhancedIntradayFeatureEngine
        import pandas as pd
        import numpy as np
        
        engine = EnhancedIntradayFeatureEngine()
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01 09:15', periods=100, freq='5min')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 100),
            'High': np.random.uniform(105, 110, 100),
            'Low': np.random.uniform(90, 95, 100),
            'Close': np.random.uniform(95, 105, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        features = engine.extract_comprehensive_features('TEST', sample_data)
        print("✅ Feature Engine: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Feature Engine: FAIL - {e}")
        return False

def test_risk_manager():
    """Test risk manager component"""
    try:
        from enhanced_intraday_risk_manager import EnhancedIntradayRiskManager
        import pandas as pd
        import numpy as np
        
        risk_manager = EnhancedIntradayRiskManager()
        
        # Generate sample data
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 50),
            'High': np.random.uniform(105, 110, 50),
            'Low': np.random.uniform(90, 95, 50),
            'Close': np.random.uniform(95, 105, 50),
            'Volume': np.random.randint(1000, 10000, 50)
        })
        
        risk_signal = risk_manager.evaluate_trade_risk('TEST', 75.0, 100.0, 10000, sample_data)
        print("✅ Risk Manager: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Risk Manager: FAIL - {e}")
        return False

def test_regime_detector():
    """Test volatility regime detector"""
    try:
        from volatility_regime_detector import VolatilityRegimeDetector
        import pandas as pd
        import numpy as np
        
        detector = VolatilityRegimeDetector()
        
        # Generate sample data
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(95, 105, 100),
            'High': np.random.uniform(105, 110, 100),
            'Low': np.random.uniform(90, 95, 100),
            'Close': np.random.uniform(95, 105, 100),
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
        regime = detector.detect_regime(sample_data, 'TEST')
        print("✅ Regime Detector: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Regime Detector: FAIL - {e}")
        return False

def test_ml_system():
    """Test ML trading system"""
    try:
        from advanced_ml_trading_system import AdvancedMLTradingSystem
        
        ml_system = AdvancedMLTradingSystem()
        signal = ml_system.generate_prediction('TEST')
        print("✅ ML System: PASS")
        return True
        
    except Exception as e:
        print(f"❌ ML System: FAIL - {e}")
        return False

def test_execution_engine():
    """Test execution engine"""
    try:
        from real_time_execution_engine import RealTimeExecutionEngine
        
        engine = RealTimeExecutionEngine()
        print("✅ Execution Engine: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Execution Engine: FAIL - {e}")
        return False

def test_integration():
    """Test basic integration"""
    try:
        # Test importing all components together
        from enhanced_intraday_feature_engine import EnhancedIntradayFeatureEngine
        from enhanced_intraday_risk_manager import EnhancedIntradayRiskManager  
        from volatility_regime_detector import VolatilityRegimeDetector
        from advanced_ml_trading_system import AdvancedMLTradingSystem
        
        # Create instances
        feature_engine = EnhancedIntradayFeatureEngine()
        risk_manager = EnhancedIntradayRiskManager()
        regime_detector = VolatilityRegimeDetector()
        ml_system = AdvancedMLTradingSystem()
        
        print("✅ Integration: PASS")
        return True
        
    except Exception as e:
        print(f"❌ Integration: FAIL - {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Integrated System Tests")
    print("=" * 50)
    
    tests = [
        ("Feature Engine", test_feature_engine),
        ("Risk Manager", test_risk_manager),
        ("Regime Detector", test_regime_detector),
        ("ML System", test_ml_system),
        ("Execution Engine", test_execution_engine),
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📊 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: CRITICAL FAIL - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. System functional with minor issues.")
    else:
        print("❌ Multiple test failures. System needs attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)