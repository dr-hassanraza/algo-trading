#!/usr/bin/env python3
"""
Test MCB Signal Request
========================

Tests the specific MCB signal request that was failing.
"""

from trading_chatbot import TradingChatbot

def test_mcb_signal():
    """Test the exact MCB signal request"""
    
    print("🔧 Testing MCB Signal Request...")
    print("=" * 50)
    
    chatbot = TradingChatbot()
    
    test_queries = [
        "Show me buy or sell signal about MCB",
        "show me signals for MCB", 
        "MCB buy signal",
        "any signals for MCB?",
        "give me MCB analysis"
    ]
    
    for query in test_queries:
        print(f"\n🧑‍💻 You: {query}")
        print("-" * 40)
        
        # Test symbol extraction
        symbols = chatbot._extract_symbols(query)
        print(f"📊 Symbols extracted: {symbols}")
        
        # Test intent
        intent = chatbot._classify_intent(query.lower())
        print(f"🎯 Intent: {intent}")
        
        # Get full response
        response = chatbot.process_query(query)
        print(f"🤖 Bot: {response}")
    
    print("\n" + "=" * 50)
    print("🎉 MCB signal test complete!")

if __name__ == '__main__':
    test_mcb_signal()