#!/usr/bin/env python3
"""
🔧 Trading System Diagnostic Tool
================================

This script helps diagnose common issues with the trading system.
Run this before starting the Streamlit app to identify potential problems.
"""

import sys
import os
import traceback
from datetime import datetime, timedelta

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Python Version Check")
    version = sys.version_info
    print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version is compatible")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_core_dependencies():
    """Check if core dependencies are available"""
    print("\n📦 Core Dependencies Check")
    dependencies = [
        ('streamlit', 'Streamlit'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('plotly', 'Plotly'),
        ('requests', 'Requests'),
        ('sklearn', 'Scikit-learn')
    ]
    
    all_available = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - pip install {module}")
            all_available = False
    
    return all_available

def check_advanced_dependencies():
    """Check if advanced ML dependencies are available"""
    print("\n🧠 Advanced ML Dependencies Check")
    advanced_deps = [
        ('tensorflow', 'TensorFlow'),
        ('lightgbm', 'LightGBM'),
        ('transformers', 'Transformers'),
        ('torch', 'PyTorch'),
        ('ccxt', 'CCXT'),
        ('nltk', 'NLTK')
    ]
    
    available_count = 0
    for module, name in advanced_deps:
        try:
            __import__(module)
            print(f"   ✅ {name}")
            available_count += 1
        except ImportError:
            print(f"   ⚠️ {name} - pip install {module}")
    
    print(f"   📊 {available_count}/{len(advanced_deps)} advanced dependencies available")
    return available_count == len(advanced_deps)

def check_file_structure():
    """Check if required files and directories exist"""
    print("\n📁 File Structure Check")
    required_files = [
        'streamlit_app.py',
        'src/trading_system.py',
        'src/advanced_trading_system.py',
        'ADVANCED_SETUP.md'
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - Missing file")
            all_present = False
    
    return all_present

def check_import_functionality():
    """Test importing the trading systems"""
    print("\n🔄 Import Functionality Check")
    
    # Test enhanced system
    try:
        sys.path.append('.')
        from src.trading_system import PSXAlgoTradingSystem
        system = PSXAlgoTradingSystem()
        print("   ✅ Enhanced trading system imports successfully")
        enhanced_works = True
    except Exception as e:
        print(f"   ❌ Enhanced trading system error: {str(e)}")
        enhanced_works = False
    
    # Test advanced system
    try:
        from src.advanced_trading_system import AdvancedTradingSystem, create_advanced_trading_system
        print("   ✅ Advanced trading system imports successfully")
        advanced_imports = True
    except Exception as e:
        print(f"   ⚠️ Advanced trading system import error: {str(e)}")
        advanced_imports = False
    
    return enhanced_works, advanced_imports

def test_datetime_operations():
    """Test datetime operations that were causing errors"""
    print("\n⏰ DateTime Operations Test")
    
    try:
        import pandas as pd
        
        # Test problematic datetime operations
        entry_timestamp = None
        timestamp = datetime.now()
        
        # Test 1: Trade duration calculation
        if entry_timestamp is not None and pd.notna(entry_timestamp):
            trade_duration = pd.Timestamp(timestamp) - pd.Timestamp(entry_timestamp)
        else:
            trade_duration = pd.Timedelta(0)
        print("   ✅ Trade duration calculation")
        
        # Test 2: Session age calculation
        login_dt = None
        session_age = datetime.now() - login_dt if login_dt else timedelta(0)
        print("   ✅ Session age calculation")
        
        # Test 3: Login time handling
        login_time = None
        if login_time and datetime.now() - login_time > timedelta(hours=24):
            pass  # This should not execute
        print("   ✅ Login time handling")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DateTime operation failed: {str(e)}")
        traceback.print_exc()
        return False

def check_environment_variables():
    """Check for environment variables and API keys"""
    print("\n🔑 Environment Variables Check")
    
    optional_vars = [
        'NEWSAPI_KEY',
        'ALPHA_VANTAGE_KEY',
        'BINANCE_API_KEY',
        'COINBASE_API_KEY'
    ]
    
    found_vars = 0
    for var in optional_vars:
        if os.getenv(var):
            print(f"   ✅ {var} is set")
            found_vars += 1
        else:
            print(f"   ⚠️ {var} not set (optional)")
    
    print(f"   📊 {found_vars}/{len(optional_vars)} optional API keys configured")
    return True  # All are optional

def run_system_test():
    """Run a comprehensive system test"""
    print("\n🧪 System Integration Test")
    
    try:
        # Test enhanced system
        from src.trading_system import PSXAlgoTradingSystem
        system = PSXAlgoTradingSystem()
        
        # Test signal generation with mock data
        import pandas as pd
        import numpy as np
        
        mock_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'price': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Test technical indicators
        mock_data = system.calculate_technical_indicators(mock_data)
        print("   ✅ Technical indicators calculation")
        
        # Test signal generation
        signal = system.generate_ml_enhanced_trading_signals(mock_data, 'TEST')
        print(f"   ✅ Signal generation: {signal.get('signal', 'UNKNOWN')}")
        
        # Test backtesting
        signals_df = pd.DataFrame([{
            'signal': 'BUY',
            'confidence': 70,
            'entry_price': 100,
            'timestamp': datetime.now()
        }])
        
        performance = system.simulate_trade_performance_advanced(signals_df)
        print("   ✅ Backtesting engine")
        
        return True
        
    except Exception as e:
        print(f"   ❌ System test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run complete diagnostic"""
    print("🔧 TRADING SYSTEM DIAGNOSTIC TOOL")
    print("=" * 50)
    
    results = []
    
    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Core Dependencies", check_core_dependencies()))
    results.append(("Advanced Dependencies", check_advanced_dependencies()))
    results.append(("File Structure", check_file_structure()))
    
    enhanced_works, advanced_imports = check_import_functionality()
    results.append(("Enhanced System Import", enhanced_works))
    results.append(("Advanced System Import", advanced_imports))
    
    results.append(("DateTime Operations", test_datetime_operations()))
    results.append(("Environment Variables", check_environment_variables()))
    results.append(("System Integration", run_system_test()))
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to run.")
        print("   You can now start the Streamlit app with: streamlit run streamlit_app.py")
    elif passed >= total * 0.8:
        print("\n⚠️ Most tests passed. Some advanced features may not work.")
        print("   Basic trading system should work fine.")
    else:
        print("\n❌ Several issues detected. Please address the failed tests.")
        print("   Check ADVANCED_SETUP.md for installation instructions.")
    
    print(f"\n🕐 Diagnostic completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()