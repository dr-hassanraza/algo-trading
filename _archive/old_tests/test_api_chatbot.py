#!/usr/bin/env python3
"""
Test Chatbot with API Key
==========================

Tests the chatbot scanning functionality with the EODHD API.
"""

from trading_chatbot import TradingChatbot

def test_scanning():
    """Test the scanning functionality"""
    
    print("🚀 Testing PSX Trading Chatbot with API...")
    print("=" * 50)
    
    chatbot = TradingChatbot()
    
    # Test API key scanning
    test_queries = [
        ("hello", "👋 Greeting the bot"),
        ("scan UBL and MCB", "📊 Testing market scanning"),
        ("any buy signals today?", "🎯 Looking for trading signals"),
        ("explain why no signals", "❓ Understanding the analysis"),
        ("show my watchlist", "📋 Checking watchlist"),
    ]
    
    for query, context in test_queries:
        print(f"\n{context}")
        print(f"🧑‍💻 You: {query}")
        print("-" * 40)
        
        response = chatbot.process_query(query)
        print(f"🤖 Bot: {response}")
        
        # Pause for readability
        import time
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("🎉 API scanning test complete!")

if __name__ == '__main__':
    test_scanning()