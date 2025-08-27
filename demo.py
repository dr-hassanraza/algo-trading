#!/usr/bin/env python3
"""
PSX Trading Chatbot Demo
========================

This script demonstrates the chatbot functionality without requiring 
an EODHD API key for basic testing.
"""

from trading_chatbot import TradingChatbot

def main():
    print("🚀 PSX Trading Chatbot Demo")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = TradingChatbot()
    
    # Demo conversations
    demo_queries = [
        "hello",
        "explain bollinger bands",
        "I bought 100 UBL at 150",
        "show my portfolio",
        "add UBL to watchlist",
        "what can you do?"
    ]
    
    for query in demo_queries:
        print(f"\n🧑‍💻 User: {query}")
        print("-" * 40)
        response = chatbot.process_query(query)
        print(f"🤖 Bot: {response}")
        print()
    
    print("=" * 40)
    print("Demo completed! 🎉")
    print("\nTo start interactive mode: python3 trading_chatbot.py")
    print("To get live market data, set up your EODHD_API_KEY")

if __name__ == '__main__':
    main()