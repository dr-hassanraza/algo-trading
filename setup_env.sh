#!/bin/bash

# PSX Algorithmic Trading Chatbot Environment Setup
# ==================================================

echo "🚀 Setting up PSX Trading Chatbot Environment..."
echo ""

# Check Python installation
echo "✓ Checking Python installation..."
python3 --version

# Check required packages
echo ""
echo "✓ Checking required packages..."
python3 -c "import pandas, numpy, requests, matplotlib, textblob; print('All packages installed successfully!')"

# Set up API key (you need to get this from EODHD.com)
echo ""
echo "⚠️  IMPORTANT: You need to set up your EODHD API key"
echo ""
echo "1. Go to https://eodhd.com/ and sign up for a free account"
echo "2. Get your API key from the dashboard"
echo "3. Run this command with your actual API key:"
echo ""
echo "   export EODHD_API_KEY=\"YOUR_ACTUAL_API_KEY_HERE\""
echo ""
echo "4. Or add it to your ~/.bashrc or ~/.zshrc for permanent setup:"
echo "   echo 'export EODHD_API_KEY=\"YOUR_API_KEY\"' >> ~/.zshrc"
echo ""

# Create necessary directories
echo "✓ Creating output directories..."
mkdir -p scan_reports

# Test basic functionality (without API key)
echo ""
echo "✓ Testing basic chatbot functionality..."
python3 -c "
from trading_chatbot import TradingChatbot
print('Chatbot modules loaded successfully!')
chatbot = TradingChatbot()
print('Chatbot initialized successfully!')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To start the chatbot:"
echo "   python3 trading_chatbot.py"
echo ""
echo "To test the scanner (requires API key):"
echo "   python3 psx_bbands_candle_scanner.py --tickers UBL.KAR MCB.KAR --asof today"
echo ""