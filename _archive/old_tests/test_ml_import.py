#!/usr/bin/env python3
"""
Test script to check Advanced ML system import status
"""

print("🧪 Testing Advanced ML/DL System Import")
print("=" * 50)

# Test the same logic as streamlit_app.py
try:
    from advanced_ml_trading_system import AdvancedMLTradingSystem, MLTradingSignal
    ADVANCED_ML_SYSTEM_AVAILABLE = True
    print("✅ Advanced ML/DL Trading System available")
    
    # Try to initialize
    try:
        system = AdvancedMLTradingSystem()
        print("✅ System initialized successfully")
        print(f"   ML Available: {system.ml_available}")
        print(f"   Available Models: {getattr(system, 'available_models', 'None')}")
        
        # Try to generate a basic prediction
        try:
            signal = system.generate_prediction('UBL')
            print("✅ Signal generation works")
            print(f"   Signal: {signal.signal} (confidence: {signal.confidence:.1f}%)")
            print(f"   Reasons: {len(signal.reasons)} factors")
        except Exception as e:
            print(f"⚠️ Signal generation failed: {e}")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    ADVANCED_ML_SYSTEM_AVAILABLE = False
    print(f"❌ Advanced ML/DL system not available: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🎯 Final Status: ADVANCED_ML_SYSTEM_AVAILABLE = {ADVANCED_ML_SYSTEM_AVAILABLE}")

if ADVANCED_ML_SYSTEM_AVAILABLE:
    print("✅ The 'not available' issue should be FIXED!")
else:
    print("❌ Still showing as 'not available'")