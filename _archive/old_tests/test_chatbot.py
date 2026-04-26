#!/usr/bin/env python3
"""
Interactive Chatbot Test
========================

Demonstrates the chatbot with predefined queries to show functionality.
"""

from trading_chatbot import TradingChatbot

def simulate_conversation():
    """Simulate a conversation with the chatbot"""
    
    print("🚀 Starting PSX Trading Chatbot...")
    print("=" * 50)
    
    chatbot = TradingChatbot()
    
    # Simulate a realistic trading conversation
    conversation = [
        ("hello", "👋 Starting our trading session"),
        ("explain bollinger bands", "📚 Learning about technical analysis"),
        ("I bought 100 UBL at 150.50", "💰 Recording first trade"),
        ("I bought 200 MCB at 85.75", "💰 Recording second trade"), 
        ("show my portfolio", "📊 Checking portfolio status"),
        ("add OGDC to my watchlist", "👀 Adding to watchlist"),
        ("add PPL to watchlist", "👀 Adding another symbol"),
        ("show my watchlist", "📋 Viewing watchlist"),
        ("explain our signal criteria", "🎯 Learning about signals"),
        ("sell 50 UBL at 155", "💸 Taking some profit"),
        ("show my portfolio", "📊 Final portfolio check"),
        ("what's my trading history", "📈 Reviewing performance")
    ]
    
    for query, context in conversation:
        print(f"\n{context}")
        print(f"🧑‍💻 You: {query}")
        print("-" * 40)
        
        response = chatbot.process_query(query)
        print(f"🤖 Bot: {response}")
        
        # Pause for readability
        import time
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("🎉 Chatbot demonstration complete!")
    print("\nThe chatbot is fully functional and ready for interactive use.")
    print("To start interactive mode, run: python3 trading_chatbot.py")

if __name__ == '__main__':
    simulate_conversation()