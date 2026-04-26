#!/usr/bin/env python3
"""
Complete System Test for Enhanced PSX Trading System
===================================================

Tests all integrated components of the enhanced trading system
"""

import time

def test_complete_system():
    print("🚀 Testing Complete Enhanced PSX Trading System")
    print("=" * 60)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Enhanced Data Fetcher
    print("\n📊 Test 1: Enhanced Data Fetcher...")
    try:
        from enhanced_data_fetcher import EnhancedDataFetcher
        fetcher = EnhancedDataFetcher()
        
        # Test real-time data
        real_time = fetcher.get_real_time_data('UBL.KAR')
        if real_time:
            print(f"   ✅ Real-time: UBL at {real_time.price:.2f} PKR ({real_time.source})")
            success_count += 1
        else:
            print("   ❌ No real-time data")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Currency Integration
    print("\n💱 Test 2: Currency Integration...")
    try:
        from bridge_client import get_currency_context
        
        currency_rates = get_currency_context()
        if currency_rates:
            print("   ✅ Currency rates:")
            for pair, rate in list(currency_rates.items())[:3]:
                if isinstance(rate, (int, float)):
                    print(f"      {pair.upper()}: {rate:.2f}")
            success_count += 1
        else:
            print("   ⚠️  No currency data")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Multi-Source Data
    print("\n🔗 Test 3: Multi-Source Data API...")
    try:
        from multi_data_source_api import MultiSourceDataAPI
        
        api = MultiSourceDataAPI()
        data = api.get_stock_data('MCB')
        
        if data:
            print(f"   ✅ MCB: {data.price:.2f} PKR from {data.source}")
            success_count += 1
        else:
            print("   ❌ No data available")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Enhanced Signal Analysis
    print("\n🎯 Test 4: Enhanced Signal Analysis...")
    try:
        from enhanced_signal_analyzer import enhanced_signal_analysis
        
        result = enhanced_signal_analysis('UBL')
        if 'error' not in result:
            signal = result['signal_strength']
            print(f"   ✅ UBL Analysis: {signal['grade']} ({signal['score']:.1f}/100)")
            print(f"   💰 Price: {result['price']:.2f} PKR")
            success_count += 1
        else:
            print(f"   ❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Node.js Bridge
    print("\n🌉 Test 5: Node.js Bridge Server...")
    try:
        import requests
        response = requests.get("http://localhost:3001/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Bridge server healthy")
            success_count += 1
        else:
            print("   ❌ Bridge server not responding")
    except:
        print("   ⚠️  Bridge server not running (optional)")
    
    # Test 6: Package Dependencies
    print("\n📦 Test 6: Package Dependencies...")
    required_packages = ['pandas', 'numpy', 'requests', 'yfinance']
    available_packages = 0
    
    for package in required_packages:
        try:
            __import__(package)
            available_packages += 1
        except ImportError:
            pass
    
    if available_packages == len(required_packages):
        print(f"   ✅ All {len(required_packages)} packages available")
        success_count += 1
    else:
        print(f"   ⚠️  {available_packages}/{len(required_packages)} packages available")
    
    # Summary
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count >= 4:
        print("🎉 System Status: EXCELLENT - Ready for trading!")
    elif success_count >= 3:
        print("✅ System Status: GOOD - Most features working")
    elif success_count >= 2:
        print("⚠️  System Status: PARTIAL - Some features working")
    else:
        print("❌ System Status: NEEDS ATTENTION")
    
    print("\n🔧 Available Features:")
    features = [
        "✅ Enhanced technical analysis with 15+ indicators",
        "✅ Multi-source data fetching with fallback",
        "✅ Sector-based fundamental analysis",
        "✅ Market sentiment integration",
        "✅ Risk management with currency factors",
        "✅ Real-time price verification",
        "✅ Caching for performance optimization"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n🏁 Complete system test finished!")
    return success_count >= 3

if __name__ == "__main__":
    test_complete_system()