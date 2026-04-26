#!/usr/bin/env python3
"""
Test Context and Follow-up Fix
==============================

Tests the exact scenario: signal request followed by detail request.
"""

from trading_chatbot import TradingChatbot

def test_context_conversation():
    """Test the complete conversation flow"""
    
    print("🔧 Testing Context-Aware Conversation...")
    print("=" * 60)
    
    chatbot = TradingChatbot()
    
    # Simulate the exact conversation you had
    print("🧑‍💻 You: can you give me buy or sell signals of FFC.KAR")
    print("-" * 50)
    
    response1 = chatbot.process_query("can you give me buy or sell signals of FFC.KAR")
    print(f"🤖 Bot: {response1}")
    
    print("\n" + "-" * 50)
    print("🧑‍💻 You: Can you provide me detail over the above criteria")
    print("-" * 50)
    
    response2 = chatbot.process_query("Can you provide me detail over the above criteria")
    print(f"🤖 Bot: {response2}")
    
    print("\n" + "=" * 60)
    print("🎉 Context conversation test complete!")
    
    # Test other follow-up variations
    print("\n🧪 Testing other follow-up variations:")
    
    follow_ups = [
        "explain the criteria",
        "tell me more about the rules",
        "what are the details of signal criteria",
        "explain signal rules"
    ]
    
    for follow_up in follow_ups:
        print(f"\n🧑‍💻 {follow_up}")
        response = chatbot.process_query(follow_up)
        intent = chatbot._classify_intent(follow_up.lower())
        print(f"🎯 Intent: {intent}")
        print(f"🤖 Response: {'✅ Detailed explanation' if len(response) > 200 else '❌ Short response'}")

if __name__ == '__main__':
    test_context_conversation()