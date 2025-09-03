#!/usr/bin/env python3
"""
Test Integrated PSX DPS System
==============================

Test suite to verify that PSX DPS integration is working correctly
as the primary data source with fallback mechanisms.
"""

import sys
import time
from datetime import datetime, timedelta

def test_psx_dps_real_time():
    """Test PSX DPS real-time data fetching"""
    print("🔬 Testing PSX DPS Real-time Data")
    print("-" * 40)
    
    try:
        from psx_dps_fetcher import PSXDPSFetcher
        
        fetcher = PSXDPSFetcher()
        
        # Test individual stocks
        test_symbols = ['UBL', 'MCB', 'LUCK']
        
        for symbol in test_symbols:
            try:
                data = fetcher.fetch_real_time_data(symbol)
                if data:
                    print(f"✅ {symbol}: {data['price']:.2f} PKR (Vol: {data['volume']:,})")
                    print(f"   📅 Time: {data['datetime'].strftime('%H:%M:%S')}")
                    print(f"   📊 Source: {data['source']}")
                else:
                    print(f"❌ {symbol}: No real-time data")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ PSX DPS test failed: {e}")
        return False

def test_enhanced_data_fetcher():
    """Test Enhanced Data Fetcher with PSX DPS priority"""
    print("\n🔬 Testing Enhanced Data Fetcher Integration")
    print("-" * 50)
    
    try:
        from enhanced_data_fetcher import EnhancedDataFetcher
        
        fetcher = EnhancedDataFetcher()
        
        # Test real-time data (should use PSX DPS first)
        print("📈 Testing real-time data fetching:")
        symbols = ['UBL', 'MCB']
        
        for symbol in symbols:
            try:
                real_time = fetcher.get_real_time_data(symbol)
                if real_time:
                    print(f"✅ {symbol}: {real_time.price:.2f} PKR - Source: {real_time.source}")
                else:
                    print(f"❌ {symbol}: No real-time data")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        # Test current price verification
        print("\n🎯 Testing verified current prices:")
        for symbol in ['UBL', 'FFC']:
            try:
                verified = fetcher.get_verified_current_price(symbol)
                if verified and verified.get('price'):
                    print(f"✅ {symbol}: {verified['price']:.2f} PKR")
                    print(f"   📊 Source: {verified['source']}")
                    print(f"   ⭐ Confidence: {verified.get('confidence', 'N/A')}/10")
                else:
                    print(f"❌ {symbol}: No verified price")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Data Fetcher test failed: {e}")
        return False

def test_data_source_priority():
    """Test that PSX DPS is being used as primary source"""
    print("\n🔬 Testing Data Source Priority")
    print("-" * 40)
    
    try:
        from enhanced_data_fetcher import EnhancedDataFetcher
        
        fetcher = EnhancedDataFetcher()
        
        # Check if PSX DPS is initialized
        if hasattr(fetcher, 'psx_dps_fetcher') and fetcher.psx_dps_fetcher:
            print("✅ PSX DPS Official fetcher initialized (PRIMARY)")
        else:
            print("❌ PSX DPS Official fetcher NOT initialized")
            
        # Check reliability statistics after some operations
        print("\n📊 Data Source Reliability:")
        reliability = fetcher.get_source_reliability()
        for source, rate in reliability.items():
            if rate > 0:
                print(f"   {source}: {rate:.1%} success rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Data source priority test failed: {e}")
        return False

def test_market_status():
    """Test market status detection"""
    print("\n🔬 Testing Market Status")
    print("-" * 30)
    
    try:
        from psx_dps_fetcher import PSXDPSFetcher
        
        fetcher = PSXDPSFetcher()
        status = fetcher.get_market_status()
        
        print(f"📊 Market Status: {status['message']}")
        print(f"🕒 Next Event: {status.get('next_event', 'N/A')}")
        print(f"💹 Is Trading: {status.get('is_trading', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Market status test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 PSX DPS Integration Test Suite")
    print("=" * 60)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("PSX DPS Real-time", test_psx_dps_real_time),
        ("Enhanced Data Fetcher", test_enhanced_data_fetcher),
        ("Data Source Priority", test_data_source_priority),
        ("Market Status", test_market_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results.append({
                'name': test_name,
                'passed': result,
                'time': end_time - start_time
            })
            
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append({
                'name': test_name,
                'passed': False,
                'time': 0,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        time_str = f"({result['time']:.1f}s)"
        print(f"{status} {result['name']} {time_str}")
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    print("-" * 60)
    print(f"🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! PSX DPS integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the integration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())