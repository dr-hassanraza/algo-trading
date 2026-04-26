#!/usr/bin/env python3
"""
Environment Setup Verification
==============================

Verifies that all components of the PSX Trading Chatbot are working correctly.
"""

import os
import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import requests
        print(f"✅ requests {requests.__version__}")
    except ImportError as e:
        print(f"❌ requests: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"⚠️  matplotlib: {e} (charts will be disabled)")
    
    try:
        import textblob
        print(f"✅ textblob {textblob.__version__}")
    except ImportError as e:
        print(f"⚠️  textblob: {e} (advanced NLP disabled)")
    
    return True

def test_modules():
    """Test custom modules"""
    print("\n🔍 Testing custom modules...")
    
    try:
        from psx_bbands_candle_scanner import scan, EODHDFetcher
        print("✅ PSX scanner module")
    except ImportError as e:
        print(f"❌ PSX scanner: {e}")
        return False
    
    try:
        from portfolio_manager import PortfolioManager
        print("✅ Portfolio manager module")
    except ImportError as e:
        print(f"❌ Portfolio manager: {e}")
        return False
    
    try:
        from trading_chatbot import TradingChatbot
        print("✅ Trading chatbot module")
    except ImportError as e:
        print(f"❌ Trading chatbot: {e}")
        return False
    
    return True

def test_chatbot_functionality():
    """Test basic chatbot functionality"""
    print("\n🔍 Testing chatbot functionality...")
    
    try:
        from trading_chatbot import TradingChatbot
        chatbot = TradingChatbot()
        
        # Test basic query
        response = chatbot.process_query("hello")
        if "trading assistant" in response.lower():
            print("✅ Basic chatbot responses")
        else:
            print("⚠️  Unexpected chatbot response format")
        
        # Test portfolio functionality
        response = chatbot.process_query("I bought 100 UBL at 150")
        if "trade executed" in response.lower():
            print("✅ Portfolio trade recording")
        else:
            print("⚠️  Portfolio functionality issue")
        
        # Test explanation functionality
        response = chatbot.process_query("explain bollinger bands")
        if "bollinger" in response.lower():
            print("✅ Educational explanations")
        else:
            print("⚠️  Explanation functionality issue")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatbot functionality: {e}")
        return False

def test_api_key():
    """Test API key setup"""
    print("\n🔍 Testing API key setup...")
    
    api_key = os.getenv('EODHD_API_KEY')
    if api_key:
        print(f"✅ EODHD_API_KEY is set (length: {len(api_key)})")
        return True
    else:
        print("⚠️  EODHD_API_KEY not set - live market data will not work")
        print("   Get your free API key from https://eodhd.com/")
        print("   Then run: export EODHD_API_KEY='your_key_here'")
        return False

def test_directories():
    """Test required directories"""
    print("\n🔍 Testing directories...")
    
    if os.path.exists('scan_reports'):
        print("✅ scan_reports directory exists")
    else:
        os.makedirs('scan_reports')
        print("✅ Created scan_reports directory")
    
    return True

def main():
    """Run all tests"""
    print("🚀 PSX Trading Chatbot - Environment Verification")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_modules,
        test_chatbot_functionality,
        test_directories,
        test_api_key
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("🎉 ALL TESTS PASSED! Your environment is ready.")
        print("\n🚀 Next steps:")
        print("   1. Run the demo: python3 demo.py")
        print("   2. Start interactive mode: python3 trading_chatbot.py")
        print("   3. For live data, get API key from https://eodhd.com/")
    elif passed >= 3:
        print("✅ CORE FUNCTIONALITY READY!")
        print("   Some optional features may be limited.")
        print("\n🚀 Next steps:")
        print("   1. Run the demo: python3 demo.py")
        print("   2. Start interactive mode: python3 trading_chatbot.py")
    else:
        print("❌ SETUP INCOMPLETE - Please fix the errors above")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())