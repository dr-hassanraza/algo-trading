#!/usr/bin/env python3
"""
Test Signal Request Fix
=======================

Tests the improved signal classification and handling.
"""

from trading_chatbot import TradingChatbot

def test_signal_requests():
    """Test different ways of asking for signals"""
    
    print("🔧 Testing Signal Request Classification Fix...")
    print("=" * 60)
    
    chatbot = TradingChatbot()
    
    # Test various signal request formats
    signal_tests = [
        "Show me buy or sell signal about MCB",
        "any buy signals today?", 
        "show me signals for UBL",
        "buy signal about MCB",
        "give me recommendations for PPL",
        "what are the signals for UBL and MCB?",
        "I need buy signals",
        "show me sell signals"
    ]
    
    for query in signal_tests:
        print(f"\n🧑‍💻 Testing: '{query}'")
        print("-" * 50)
        
        # Test intent classification
        intent = chatbot._classify_intent(query.lower())
        print(f"🎯 Intent detected: {intent}")
        
        if intent == 'signals':
            print("✅ Correctly classified as signal request")
            response = chatbot.process_query(query)
            print(f"🤖 Response preview: {response[:100]}...")
        else:
            print(f"❌ Incorrectly classified as: {intent}")
    
    print("\n" + "=" * 60)
    print("🎉 Signal classification test complete!")

if __name__ == '__main__':
    test_signal_requests()