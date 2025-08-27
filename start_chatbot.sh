#!/bin/bash

# PSX Trading Chatbot Startup Script
# ===================================

echo "🚀 Starting PSX Trading Chatbot..."
echo ""

# Load environment
export EODHD_API_KEY="68a350864b1140.05317137"

echo "✅ API Key loaded"
echo "✅ Environment ready"
echo ""
echo "🤖 Launching chatbot..."
echo "📊 Live PSX market data enabled"
echo ""
echo "Type 'quit' to exit when ready"
echo "Try: 'scan UBL and MCB' or 'hello'"
echo ""

# Start the chatbot
python3 trading_chatbot.py