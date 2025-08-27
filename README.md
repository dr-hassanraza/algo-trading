# PSX Algorithmic Trading Chatbot

A conversational AI assistant for Pakistan Stock Exchange (PSX) trading with technical analysis, portfolio management, and natural language interaction.

## Features

🤖 **Natural Language Interface**
- Chat with your trading assistant in plain English
- Understand commands like "scan UBL and MCB" or "show my portfolio"
- Context-aware conversations with memory

📊 **Technical Analysis**
- Bollinger Bands analysis with %B indicator
- 44-period Moving Average trend detection
- Candlestick pattern recognition
- Multi-criteria signal generation

💼 **Portfolio Management**
- Track positions with real-time P&L calculations
- Record trades: "I bought 100 UBL at 150"
- Cash balance management
- Transaction history and performance metrics

🎯 **Trading Signals**
- Automated buy signal detection
- Human-readable signal explanations
- Customizable risk tolerance settings

📈 **Market Scanning**
- Scan multiple PSX symbols simultaneously
- Watchlist management
- Visual charts generation (optional)

## Installation

1. **Clone or download the files**
2. **Install dependencies:**
   ```bash
   pip install pandas numpy requests matplotlib textblob
   ```

3. **Get EODHD API key:**
   - Sign up at [EODHD.com](https://eodhd.com/)
   - Set environment variable:
   ```bash
   export EODHD_API_KEY="your_api_key_here"
   ```

## Quick Start

### Basic Scanner Usage
```bash
python psx_bbands_candle_scanner.py --tickers UBL.KAR MCB.KAR OGDC.KAR --asof today --charts
```

### Interactive Chatbot
```bash
python trading_chatbot.py
```

## Chatbot Commands

### Market Analysis
- `"Scan UBL and MCB"`
- `"Check my watchlist"`
- `"Any buy signals today?"`
- `"Show me recommendations"`

### Portfolio Management
- `"Show my portfolio"`
- `"I bought 100 UBL at 150"`
- `"Sell 50 MCB at 200"`
- `"What's my P&L?"`

### Watchlist Management
- `"Add OGDC to watchlist"`
- `"Remove PPL from watchlist"`
- `"Show my watchlist"`

### Educational
- `"Explain Bollinger Bands"`
- `"How do signals work?"`
- `"What does MA44 mean?"`

## Trading Signal Criteria

The chatbot uses a sophisticated 5-criteria system for buy signals:

1. **Trend**: MA44 slope positive over 10 days (uptrend)
2. **Position**: Current price above MA44 support
3. **Momentum**: Bollinger %B between 0.35-0.85 (healthy zone)
4. **Pattern**: Green candle with real body ≥40% of day's range
5. **Entry**: Recent low within 2% of MA44 (pullback opportunity)

## File Structure

```
Algo_Trading/
├── trading_chatbot.py          # Main chatbot interface
├── psx_bbands_candle_scanner.py # Technical analysis engine
├── portfolio_manager.py        # Portfolio tracking system
├── chatbot_state.json         # Conversation memory (auto-created)
├── portfolio.json             # Portfolio data (auto-created)
└── scan_reports/              # Scanner output directory
    └── YYYY-MM-DD/
        ├── diagnostics.csv    # Detailed analysis
        ├── candidates.csv     # Buy candidates only
        └── charts/           # Optional PNG charts
```

## Examples

### Starting the Chatbot
```
🤖 PSX Trading Chatbot initialized!
Type 'quit' to exit, 'help' for commands

You: Hello
🤖 👋 Hello! I'm your PSX trading assistant. I can help you with:

• Market Scanning: "Scan UBL and MCB" or "Check my watchlist"
• Trading Signals: "Show me buy signals" or "Any recommendations?"
• Portfolio Tracking: "Show my portfolio" or "Portfolio performance"
• Trade Recording: "I bought 100 UBL at 150" or "Sell 50 MCB at 200"
• Explanations: "Explain Bollinger Bands" or "How do signals work?"
• Watchlist: "Add OGDC to watchlist" or "Show my watchlist"

What would you like to explore?
```

### Recording a Trade
```
You: I bought 100 shares of UBL at 145.50
🤖 ✅ Trade executed: Added new position: UBL 100 shares @ 145.50
```

### Getting Signals
```
You: Any buy signals today?
🤖 🎯 Buy Signals for Tomorrow:

**UBL.KAR** (Close: 147.80)
  • MA44: 142.30 | BB %B: 0.67
  • Signal: uptrend confirmed, above MA44 support, healthy momentum zone, strong green candle

⚠️ Remember: These are educational signals only. Always do your own research before trading.
```

## Configuration

### Risk Tolerance
The chatbot adapts to your risk preference:
- `Conservative`: Stricter signal criteria
- `Moderate`: Balanced approach (default)
- `Aggressive`: More permissive signals

### PSX Symbols
Use `.KAR` suffix for EODHD API (e.g., `UBL.KAR`). The system automatically adds this if you use bare symbols like `UBL`.

## Data Sources

- **Market Data**: EODHD.com (requires API key)
- **Symbols**: Pakistan Stock Exchange (PSX)
- **Timeframe**: Daily candles
- **History**: Configurable (default: 260 days ≈ 1 year)

## Limitations & Disclaimers

⚠️ **Educational Use Only**
- This is for learning and research purposes
- Not financial advice
- Always do your own research
- Past performance doesn't guarantee future results

🔒 **Technical Limitations**
- Requires internet connection for market data
- End-of-day data only (not real-time intraday)
- PSX symbols only
- Single-threaded scanning (can be slow for many symbols)

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `EODHD_API_KEY` environment variable is set
2. **No Data**: Check symbol format (use `.KAR` suffix)
3. **Import Errors**: Run `pip install textblob` for NLP features
4. **Slow Scanning**: Reduce number of symbols or use smaller date ranges

### Debug Mode
For detailed error information, check the scanner output files in `scan_reports/YYYY-MM-DD/`.

## Future Enhancements

- Real-time data integration
- Multiple exchange support
- Advanced backtesting
- Risk management tools
- Mobile app interface
- Automated trade execution
- Machine learning signal enhancement

## Contributing

This is an educational project. Feel free to:
- Add new technical indicators
- Improve the NLP processing
- Enhance portfolio analytics
- Add new chart types
- Optimize performance

## License

Educational use only. Not for commercial trading without proper licensing and disclaimers.