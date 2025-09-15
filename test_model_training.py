#!/usr/bin/env python3
"""
Test script to manually train LSTM and Meta models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.advanced_trading_system import AdvancedTradingSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_training():
    """Test the complete model training pipeline"""
    try:
        print("🚀 Initializing Advanced Trading System...")
        system = AdvancedTradingSystem()
        
        print(f"🔍 ML Available: {system.ml_available}")
        print(f"📊 PSX Symbols Count: {len(system.psx_symbols)}")
        
        # Check current model status
        status = system.get_system_status()
        print(f"📈 Current LSTM Model Ready: {status['lstm_model_ready']}")
        print(f"🎯 Current Meta Model Ready: {status['meta_model_ready']}")
        
        if not system.ml_available:
            print("❌ ML libraries not available. Please install:")
            print("pip install tensorflow lightgbm scikit-learn")
            return False
        
        print("\n🎓 Starting model training...")
        
        def progress_callback(message):
            print(f"   {message}")
        
        # Train models with top 20 symbols and 14 days for faster testing
        result = system.train_models_with_psx_data(
            symbols=None,  # Use default top symbols
            days_back=14,  # Smaller dataset for testing
            progress_callback=progress_callback
        )
        
        print(f"\n📊 Training Result: {result}")
        
        if result.get('status') == 'success':
            print("✅ Training completed successfully!")
            
            # Check status after training
            status = system.get_system_status()
            print(f"📈 LSTM Model Ready: {status['lstm_model_ready']}")
            print(f"🎯 Meta Model Ready: {status['meta_model_ready']}")
            
            # Test model loading
            print("\n💾 Testing model persistence...")
            system2 = AdvancedTradingSystem()
            status2 = system2.get_system_status()
            print(f"📈 New Instance LSTM Ready: {status2['lstm_model_ready']}")
            print(f"🎯 New Instance Meta Ready: {status2['meta_model_ready']}")
            
            return True
        else:
            print(f"❌ Training failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Model Training Pipeline...")
    success = test_model_training()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")