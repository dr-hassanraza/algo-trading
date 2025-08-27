#!/usr/bin/env python3
"""
Test Enhanced Signal System
===========================

Demonstrates the enhanced capabilities vs basic signals.
"""

from trading_chatbot import TradingChatbot

def test_enhanced_vs_basic():
    """Compare basic vs enhanced signal analysis"""
    
    print("🚀 ENHANCED SIGNAL SYSTEM COMPARISON")
    print("=" * 60)
    
    chatbot = TradingChatbot()
    
    test_scenarios = [
        ("Basic Signal", "show me signals for MCB"),
        ("Enhanced Signal", "enhanced analysis for MCB"),
        ("Context Question", "Can you provide me detail over the above criteria"),
        ("Multiple Enhanced", "advanced signals for UBL and FFC")
    ]
    
    for scenario, query in test_scenarios:
        print(f"\n🧪 **{scenario}**")
        print(f"🧑‍💻 Query: '{query}'")
        print("-" * 50)
        
        response = chatbot.process_query(query)
        print(f"🤖 Response: {response[:200]}...")
        if len(response) > 200:
            print("    [Response continues with detailed analysis...]")
        
        print()
    
    print("=" * 60)
    print("🎉 Enhanced system test complete!")
    
    print("\n✅ **NEW ENHANCED FEATURES:**")
    print("   🎯 Signal scoring: A-F grades (0-100 points)")
    print("   📊 RSI momentum analysis")
    print("   📈 Volume confirmation (surge detection)")
    print("   🎪 Support/resistance levels")
    print("   🛡️  Risk management (stop loss, targets)")
    print("   ⚡ Multi-factor signal strength")
    print("   🔍 Detailed explanations")

if __name__ == '__main__':
    test_enhanced_vs_basic()