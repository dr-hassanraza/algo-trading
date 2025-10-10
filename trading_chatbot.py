#!/usr/bin/env python3
"""
Algorithmic Trading Chatbot
===========================

A conversational interface for the PSX trading scanner that provides:
- Natural language queries about market conditions
- Trading recommendations with explanations
- Portfolio tracking and management
- Real-time market analysis

Dependencies:
    pip install pandas numpy requests matplotlib nltk textblob

Usage:
    python trading_chatbot.py
"""

import os
import re
import json
import datetime as dt

# Load environment variables from .env file
def load_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass

load_env()
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

try:
    from textblob import TextBlob
except ImportError:
    print("Warning: TextBlob not available. Basic NLP features disabled.")
    TextBlob = None

from psx_bbands_candle_scanner import scan, EODHDFetcher, TODAY
from portfolio_manager import PortfolioManager
from enhanced_signal_analyzer import enhanced_signal_analysis


@dataclass
class ChatbotState:
    """Maintains conversation context and user preferences"""
    current_watchlist: List[str]
    last_scan_date: Optional[dt.date]
    user_risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'
    preferred_sectors: List[str]
    conversation_history: List[Dict]
    portfolio: Dict[str, Dict]  # symbol -> {quantity, avg_price, date_bought}


class TradingChatbot:
    def __init__(self):
        self.state = ChatbotState(
            current_watchlist=[],
            last_scan_date=None,
            user_risk_tolerance='moderate',
            preferred_sectors=[],
            conversation_history=[],
            portfolio={}
        )
        self.scanner_cache = {}
        self.portfolio_manager = PortfolioManager()
        
        # Common PSX symbols for quick suggestions
        self.common_symbols = [
            'UBL', 'MCB', 'FYBL', 'OGDC', 'PPL', 'HUBCO', 'KAPCO', 'LUCK', 'ENGRO',
            'FCCL', 'DGKC', 'MLCF', 'FFBL', 'ATRL', 'SEARL', 'PIOC'
        ]

    def process_query(self, user_input: str) -> str:
        """Main entry point for processing user queries"""
        user_input = user_input.strip().lower()
        
        # Store in conversation history
        self.state.conversation_history.append({
            'timestamp': dt.datetime.now().isoformat(),
            'user_input': user_input,
            'response': None  # Will be filled after generating response
        })
        
        # Determine intent and generate response
        intent = self._classify_intent(user_input)
        response = self._generate_response(intent, user_input)
        
        # Update conversation history with response
        self.state.conversation_history[-1]['response'] = response
        
        return response

    def _classify_intent(self, user_input: str) -> str:
        """Classify user intent using keyword matching and NLP"""
        
        # Keyword-based classification
        scan_keywords = ['scan', 'analyze', 'check', 'find', 'search', 'look for']
        portfolio_keywords = ['portfolio', 'holdings', 'positions', 'my stocks', 'p&l', 'profit', 'loss']
        trade_keywords = ['buy', 'sell', 'bought', 'sold', 'purchase', 'trade']
        signal_keywords = ['signal', 'recommend', 'suggest', 'candidates', 'buy signals', 'any signals', 'show me buy', 'show me sell', 'buy signal', 'sell signal', 'signals about', 'signal about', 'enhanced signal', 'advanced analysis', 'detailed signal', 'enhanced analysis', 'comprehensive analysis', 'detailed analysis']
        explain_keywords = ['explain', 'why', 'how', 'what does', 'meaning', 'detail', 'details', 'criteria', 'rules', 'above criteria', 'tell me more']
        watchlist_keywords = ['watchlist', 'add', 'remove', 'track', 'monitor', 'show my watchlist', 'my watchlist']
        
        # Prioritize explanation requests
        if any(keyword in user_input for keyword in explain_keywords):
            return 'explain'
        elif any(keyword in user_input for keyword in signal_keywords):
            return 'signals'
        elif any(keyword in user_input for keyword in scan_keywords):
            return 'scan'
        elif any(keyword in user_input for keyword in portfolio_keywords):
            return 'portfolio'
        elif any(keyword in user_input for keyword in trade_keywords):
            # Additional check: if it contains signal-related context, classify as signals
            if any(signal_word in user_input for signal_word in ['signal', 'signals', 'show me', 'recommendation']):
                return 'signals'
            return 'trade'
        elif any(keyword in user_input for keyword in watchlist_keywords):
            return 'watchlist'
        else:
            return 'general'

    def _generate_response(self, intent: str, user_input: str) -> str:
        """Generate appropriate response based on intent"""
        
        if intent == 'scan':
            return self._handle_scan_request(user_input)
        elif intent == 'portfolio':
            return self._handle_portfolio_request(user_input)
        elif intent == 'trade':
            return self._handle_trade_request(user_input)
        elif intent == 'signals':
            return self._handle_signals_request(user_input)
        elif intent == 'explain':
            return self._handle_explain_request(user_input)
        elif intent == 'watchlist':
            return self._handle_watchlist_request(user_input)
        else:
            return self._handle_general_request(user_input)

    def _handle_scan_request(self, user_input: str) -> str:
        """Handle market scanning requests"""
        
        # Extract symbols from user input
        symbols = self._extract_symbols(user_input)
        
        if not symbols:
            # Use watchlist or suggest common symbols
            if self.state.current_watchlist:
                symbols = self.state.current_watchlist
                symbol_source = "your watchlist"
            else:
                symbols = self.common_symbols[:5]  # Top 5 common symbols
                symbol_source = "popular PSX stocks"
        else:
            symbol_source = "the symbols you mentioned"
        
        try:
            # Run the scanner
            results = scan(symbols, asof=TODAY, days=260, make_charts=False)
            
            # Cache results
            self.scanner_cache[TODAY] = results
            self.state.last_scan_date = TODAY
            
            # Generate human-readable response
            return self._format_scan_results(results, symbol_source)
            
        except Exception as e:
            return f"Sorry, I encountered an error while scanning: {str(e)}. Please check your EODHD_API_KEY environment variable."

    def _handle_portfolio_request(self, user_input: str) -> str:
        """Handle portfolio-related requests"""
        
        if not self.portfolio_manager.positions:
            return "Your portfolio is currently empty. You can add positions by saying something like 'I bought 100 shares of UBL at 150'"
        
        # Get current prices for portfolio positions
        current_prices = {}
        for symbol in self.portfolio_manager.positions.keys():
            price = self._get_current_price(symbol)
            if price:
                current_prices[symbol] = price
        
        summary = self.portfolio_manager.get_portfolio_summary(current_prices)
        
        response = "📊 **Your Portfolio:**\n\n"
        
        for position in summary['positions']:
            symbol = position['symbol']
            response += f"**{symbol}**: {position['quantity']} shares @ {position['avg_price']:.2f}\n"
            response += f"   Current: {position['current_price']:.2f} | P&L: {position['unrealized_pnl']:+.2f} ({position['unrealized_pnl_pct']:+.1f}%)\n\n"
        
        response += f"**Cash Balance**: {summary['cash_balance']:.2f} PKR\n"
        response += f"**Total Portfolio Value**: {summary['total_portfolio_value']:.2f} PKR\n"
        response += f"**Total Unrealized P&L**: {summary['total_unrealized_pnl']:+.2f} PKR ({summary['total_unrealized_pnl_pct']:+.1f}%)\n"
        
        if summary['realized_pnl'] != 0:
            response += f"**Realized P&L**: {summary['realized_pnl']:+.2f} PKR"
        
        return response

    def _handle_trade_request(self, user_input: str) -> str:
        """Handle trade execution requests"""
        
        # Parse trade details from user input
        trade_details = self._parse_trade_input(user_input)
        
        if not trade_details:
            return "I couldn't understand the trade details. Please use format like 'I bought 100 UBL at 150' or 'Sell 50 MCB at 200'"
        
        symbol, action, quantity, price = trade_details
        
        try:
            if action == 'buy':
                result = self.portfolio_manager.add_position(symbol, quantity, price)
            elif action == 'sell':
                result = self.portfolio_manager.sell_position(symbol, quantity, price)
            else:
                return "Please specify 'buy' or 'sell' for the trade action."
            
            return f"✅ Trade executed: {result}"
            
        except Exception as e:
            return f"❌ Trade failed: {str(e)}"

    def _handle_signals_request(self, user_input: str) -> str:
        """Handle trading signal requests"""
        
        # Extract specific symbols if mentioned
        mentioned_symbols = self._extract_symbols(user_input)
        
        # Check if enhanced/advanced analysis is requested
        enhanced_requested = any(keyword in user_input.lower() for keyword in 
                               ['enhanced', 'advanced', 'detailed', 'comprehensive'])
        
        # Determine which symbols to analyze
        if mentioned_symbols:
            symbols_to_scan = mentioned_symbols
            symbol_source = f"the symbols you mentioned ({', '.join(mentioned_symbols)})"
        elif self.state.current_watchlist:
            symbols_to_scan = self.state.current_watchlist
            symbol_source = "your watchlist"
        else:
            symbols_to_scan = self.common_symbols[:5]  # Fewer for enhanced analysis
            symbol_source = "popular PSX stocks"
        
        # Use enhanced analysis if requested or if only 1-2 symbols
        if enhanced_requested or len(symbols_to_scan) <= 2:
            return self._enhanced_signal_analysis(symbols_to_scan, symbol_source)
        else:
            return self._basic_signal_analysis(symbols_to_scan, symbol_source)
    
    def _enhanced_signal_analysis(self, symbols: List[str], symbol_source: str) -> str:
        """Provide enhanced signal analysis with comprehensive metrics"""
        
        try:
            response = f"🚀 **Enhanced Signal Analysis for {symbol_source}:**\n\n"
            
            for symbol in symbols[:3]:  # Limit to 3 for detailed analysis
                result = enhanced_signal_analysis(symbol)
                
                if 'error' in result:
                    response += f"❌ **{symbol}**: {result['error']}\n\n"
                    continue
                
                signal = result['signal_strength']
                risk = result['risk_management']
                tech = result['technical_data']
                
                # Signal grade with emoji
                grade_emoji = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⚫"}
                
                response += f"{grade_emoji.get(signal['grade'], '⚫')} **{result['symbol']}** - Grade {signal['grade']} ({signal['score']:.0f}/100)\n"
                response += f"💰 **Price**: {result['price']:.2f} PKR | **Recommendation**: {signal['recommendation']}\n\n"
                
                # Technical metrics
                response += f"📊 **Technical Analysis:**\n"
                response += f"  • RSI: {tech['rsi']:.1f} | MA44: {tech['ma44']:.2f} | BB %B: {tech['bb_pctb']:.2f}\n"
                response += f"  • Volume: {tech['volume_ratio']:.1f}x average | ATR: {tech['atr']:.2f}\n\n"
                
                # Risk management
                response += f"🎯 **Risk Management:**\n"
                response += f"  • Stop Loss: {risk['stop_loss']:.2f} (-{risk['stop_loss_pct']:.1f}%)\n"
                response += f"  • Target 1: {risk['target1']:.2f} (+{risk['target1_pct']:.1f}%)\n"
                response += f"  • Target 2: {risk['target2']:.2f} (+{risk['target2_pct']:.1f}%)\n"
                response += f"  • Risk/Reward: {risk['risk_reward_ratio']:.2f}\n\n"
                
                # Top signal factors
                response += f"🔍 **Key Factors:**\n"
                for factor in signal['factors'][:3]:
                    response += f"  • {factor}\n"
                response += "\n"
                
                # Support/Resistance
                sr = result['support_resistance']
                if sr['nearest_support'] or sr['nearest_resistance']:
                    response += f"📈 **Levels:** "
                    if sr['nearest_support']:
                        response += f"Support: {sr['nearest_support']:.2f} "
                    if sr['nearest_resistance']:
                        response += f"Resistance: {sr['nearest_resistance']:.2f}"
                    response += "\n\n"
                
                response += "─" * 50 + "\n\n"
            
            response += "⚠️ *Enhanced analysis for educational purposes only. Always do your own research.*"
            return response
            
        except Exception as e:
            return f"Enhanced analysis error: {str(e)}"
    
    def _basic_signal_analysis(self, symbols: List[str], symbol_source: str) -> str:
        """Provide basic signal analysis (original method)"""
        
        try:
            # Run fresh scan for the requested symbols
            results = scan(symbols, asof=TODAY, days=260, make_charts=False)
            self.scanner_cache[TODAY] = results
            
            # Analyze results
            buy_candidates = results[results.get('buy_next_day', 0) == 1]
            # Filter out error rows
            error_mask = results['error'].notna() if 'error' in results.columns else pd.Series([False] * len(results), index=results.index)
            all_results = results[~error_mask]
            
            response = f"🎯 **Signal Analysis for {symbol_source}:**\n\n"
            
            if not buy_candidates.empty:
                response += "✅ **BUY SIGNALS:**\n"
                for _, row in buy_candidates.iterrows():
                    response += f"**{row['symbol']}** (Close: {row['close']:.2f})\n"
                    response += f"  • MA44: {row['ma44']:.2f} | BB %B: {row['bb_pctB']:.2f}\n"
                    response += f"  • {self._explain_buy_signal(row)}\n\n"
            
            # Show other analyzed stocks
            other_stocks = all_results[all_results.get('buy_next_day', 0) != 1]
            if not other_stocks.empty:
                response += "📊 **Other Analysis:**\n"
                for _, row in other_stocks.head(3).iterrows():
                    status = "Above MA44" if row.get('close_gt_ma44', False) else "Below MA44"
                    trend = "Uptrend" if row.get('trend_ma44_up', False) else "Downtrend"
                    response += f"• **{row['symbol']}**: {row['close']:.2f} ({status}, {trend})\n"
                response += "\n"
            
            if buy_candidates.empty and not other_stocks.empty:
                response += "🔍 **No buy signals found.** Current market conditions don't meet our 5-criteria entry rules.\n\n"
                response += "💡 *Try: 'enhanced signal analysis for [symbol]' for detailed scoring and risk management.*\n\n"
            
            response += "⚠️ *Educational signals only. Always do your own research before trading.*"
            
            return response
            
        except Exception as e:
            return f"Unable to generate signals: {str(e)}"

    def _handle_explain_request(self, user_input: str) -> str:
        """Handle explanation requests"""
        
        # Check for context-based explanations first
        if any(word in user_input for word in ['above criteria', 'criteria', 'rules', 'detail', 'details']):
            # Check recent conversation for signal context
            recent_responses = [item['response'] for item in self.state.conversation_history[-3:] if item['response']]
            signal_mentioned = any('criteria' in str(response) or 'No buy signals found' in str(response) for response in recent_responses)
            
            if signal_mentioned or 'signal' in user_input:
                return """🎯 **Detailed Buy Signal Criteria Explanation:**

Our algorithm uses a strict 5-criteria system for buy signals:

**1. 📈 Trend (MA44 Slope)**
   • MA44 must be rising over the last 10 trading days
   • Ensures we're in an established uptrend
   • Filters out sideways or declining markets

**2. 🎯 Position (Price vs MA44)**
   • Current price must be above MA44
   • Confirms we're buying strength, not weakness
   • MA44 acts as dynamic support level

**3. ⚡ Momentum (Bollinger %B)**
   • %B must be between 0.35-0.85
   • Avoids oversold (below 0.35) and overbought (above 0.85) extremes
   • Sweet spot for continued upward movement

**4. 🕯️ Candle Pattern**
   • Must be a green candle (close > open)
   • Real body ≥ 40% of the day's total range
   • Shows strong buying conviction

**5. 🎪 Entry Timing (Pullback)**
   • Recent low (last 3 sessions) within 2% of MA44
   • Ensures we're buying on a healthy pullback
   • Better risk/reward entry point

**Why So Strict?** These criteria work together to find high-probability, low-risk entry points with strong trend momentum."""
        
        elif 'bollinger' in user_input or 'bb' in user_input:
            return """🎯 **Bollinger Bands Explained:**

Bollinger Bands consist of:
• **Middle Line**: 20-period moving average
• **Upper/Lower Bands**: 2 standard deviations from middle line

**%B Indicator**: Shows where price is relative to the bands
• %B = 0: Price at lower band
• %B = 0.5: Price at middle line  
• %B = 1: Price at upper band

**Our Strategy**: We look for %B between 0.35-0.85 (avoiding extremes) to find stocks in a healthy uptrend without being overbought."""

        elif 'ma44' in user_input or 'moving average' in user_input:
            return """📈 **MA44 (44-Period Moving Average):**

• **Trend Indicator**: Shows the average price over 44 trading days
• **Slope Analysis**: We calculate if MA44 is rising (uptrend) or falling (downtrend)
• **Support Level**: Acts as dynamic support in uptrends

**Our Use**: We only consider buy signals when:
1. MA44 slope is positive (uptrend)
2. Current price is above MA44
3. Recent lows came within 2% of MA44 (healthy pullback)"""

        elif 'signal' in user_input or any(word in user_input for word in ['criteria', 'rules', 'methodology']):
            return """🎯 **Our Buy Signal Criteria:**

All conditions must be met:
1. **Trend**: MA44 rising over last 10 days
2. **Position**: Close price above MA44
3. **Momentum**: Bollinger %B between 0.35-0.85
4. **Candle**: Green candle with real body ≥40% of range
5. **Entry**: Recent low within 2% of MA44 (last 3 sessions)

This combines trend following, momentum, and pattern recognition for high-probability setups."""

        else:
            return "I can explain Bollinger Bands, MA44, or our signal criteria. What would you like to know more about?"

    def _handle_watchlist_request(self, user_input: str) -> str:
        """Handle watchlist management"""
        
        if 'add' in user_input:
            symbols = self._extract_symbols(user_input)
            if symbols:
                for symbol in symbols:
                    if symbol not in self.state.current_watchlist:
                        self.state.current_watchlist.append(symbol)
                return f"Added {', '.join(symbols)} to your watchlist. Current watchlist: {', '.join(self.state.current_watchlist)}"
            else:
                return "Please specify which symbols to add. For example: 'add UBL and MCB to watchlist'"
        
        elif 'remove' in user_input:
            symbols = self._extract_symbols(user_input)
            if symbols:
                for symbol in symbols:
                    if symbol in self.state.current_watchlist:
                        self.state.current_watchlist.remove(symbol)
                return f"Removed {', '.join(symbols)} from watchlist. Current watchlist: {', '.join(self.state.current_watchlist)}"
            else:
                return "Please specify which symbols to remove."
        
        elif 'show' in user_input or 'list' in user_input:
            if self.state.current_watchlist:
                return f"Your current watchlist: {', '.join(self.state.current_watchlist)}"
            else:
                return "Your watchlist is empty. Add symbols by saying 'add UBL to watchlist'"
        
        else:
            return "I can help you add, remove, or show your watchlist. What would you like to do?"

    def _handle_general_request(self, user_input: str) -> str:
        """Handle general queries and greetings"""
        
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if any(greeting in user_input for greeting in greetings):
            return """👋 Hello! I'm your enhanced PSX trading assistant. I can help you with:

• **Market Scanning**: "Scan UBL and MCB" or "Check my watchlist"
• **Basic Signals**: "Show me buy signals" or "Any recommendations?"
• **🚀 Enhanced Signals**: "Enhanced analysis for MCB" or "Advanced signals for UBL"
• **Portfolio Tracking**: "Show my portfolio" or "Portfolio performance"
• **Trade Recording**: "I bought 100 UBL at 150" or "Sell 50 MCB at 200"
• **Explanations**: "Explain Bollinger Bands" or "How do signals work?"
• **Watchlist**: "Add OGDC to watchlist" or "Show my watchlist"

🆕 **NEW**: Enhanced signals include RSI, volume analysis, signal scoring (A-F grades), support/resistance levels, and complete risk management with stop losses and targets!

What would you like to explore?"""
        
        help_keywords = ['help', 'what can you do', 'commands']
        if any(keyword in user_input for keyword in help_keywords):
            return self._handle_general_request('hello')  # Return help message
        
        return "I'm not sure I understand. You can ask me to scan stocks, show signals, manage your portfolio, or explain trading concepts. Try saying 'help' for more options."

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from user input"""
        
        # First check against known PSX symbols (more accurate)
        known_matches = []
        text_upper = text.upper()
        for symbol in self.common_symbols:
            # Look for symbol as whole word
            if re.search(r'\b' + symbol + r'\b', text_upper):
                known_matches.append(symbol)
        
        # Look for explicit PSX symbols (3-4 letters, possibly with .KAR)
        # but exclude common English words
        exclude_words = {'BUY', 'SELL', 'THE', 'AND', 'FOR', 'ARE', 'WHAT', 'SHOW', 'GIVE', 'NEED', 'SCAN', 'ABOUT', 'CAN', 'YOU', 'GET', 'ANY', 'NOT', 'BUT', 'ALL', 'HOW', 'WHY', 'ADD', 'OUR'}
        symbol_pattern = r'\b([A-Z]{3,4})(?:\.KAR)?\b'
        found_symbols = re.findall(symbol_pattern, text.upper())
        found_symbols = [s for s in found_symbols if s not in exclude_words]
        
        # Combine and deduplicate
        all_symbols = list(set(known_matches + found_symbols))
        
        return all_symbols

    def _parse_trade_input(self, text: str) -> Optional[Tuple[str, str, int, float]]:
        """Parse trade details from natural language input"""
        
        text = text.lower()
        
        # Determine action
        action = None
        if any(word in text for word in ['buy', 'bought', 'purchase', 'purchased']):
            action = 'buy'
        elif any(word in text for word in ['sell', 'sold']):
            action = 'sell'
        
        if not action:
            return None
        
        # Extract symbol
        symbols = self._extract_symbols(text)
        if not symbols:
            return None
        symbol = symbols[0]  # Take first symbol found
        
        # Extract quantity (look for numbers before 'shares' or just standalone numbers)
        quantity_pattern = r'(\d+)\s*(?:shares?)?'
        quantity_matches = re.findall(quantity_pattern, text)
        
        # Extract price (look for 'at' followed by number or just numbers with decimal)
        price_pattern = r'(?:at|@|price)\s*(\d+(?:\.\d+)?)|(\d+\.\d+)'
        price_matches = re.findall(price_pattern, text)
        
        if not quantity_matches:
            return None
        
        quantity = int(quantity_matches[0])
        
        # For price, try to find the most likely candidate
        price = None
        if price_matches:
            # Flatten the tuples and get first non-empty match
            all_prices = [match for group in price_matches for match in group if match]
            if all_prices:
                price = float(all_prices[0])
        
        # If no explicit price found, look for any decimal number that could be a price
        if price is None:
            decimal_pattern = r'\b(\d{2,4}\.\d{1,2})\b'
            decimal_matches = re.findall(decimal_pattern, text)
            if decimal_matches:
                price = float(decimal_matches[0])
        
        # If still no price, look for any reasonable integer that could be a price
        if price is None:
            number_pattern = r'\b(\d{2,4})\b'
            number_matches = re.findall(number_pattern, text)
            # Filter out the quantity number
            price_candidates = [int(n) for n in number_matches if int(n) != quantity and int(n) > 10]
            if price_candidates:
                price = float(price_candidates[0])
        
        if price is None:
            return None
        
        return symbol, action, quantity, price

    def _format_scan_results(self, results: pd.DataFrame, symbol_source: str) -> str:
        """Format scan results in a user-friendly way"""
        
        response = f"📊 **Market Scan Results** (from {symbol_source}):\n\n"
        
        # Separate buy candidates from others
        buy_candidates = results[results['buy_next_day'] == 1]
        others = results[results['buy_next_day'] != 1]
        
        if not buy_candidates.empty:
            response += "🎯 **Buy Candidates:**\n"
            for _, row in buy_candidates.iterrows():
                response += f"• **{row['symbol']}**: {row['close']:.2f} (MA44: {row['ma44']:.2f})\n"
            response += "\n"
        
        if not others.empty:
            response += "📈 **Other Scanned Stocks:**\n"
            for _, row in others.head(5).iterrows():  # Limit to 5 for brevity
                if 'error' in row and pd.notna(row['error']):
                    response += f"• **{row['symbol']}**: Error - {row['error']}\n"
                else:
                    status = "Above MA44" if row.get('close_gt_ma44', False) else "Below MA44"
                    response += f"• **{row['symbol']}**: {row['close']:.2f} ({status})\n"
        
        response += f"\n*Scanned {len(results)} symbols on {TODAY}*"
        
        return response

    def _explain_buy_signal(self, row: pd.Series) -> str:
        """Generate explanation for why a stock is a buy candidate"""
        
        reasons = []
        if row.get('trend_ma44_up', False):
            reasons.append("uptrend confirmed")
        if row.get('close_gt_ma44', False):
            reasons.append("above MA44 support")
        if row.get('bb_pctB_midzone', False):
            reasons.append("healthy momentum zone")
        if row.get('green_candle_body40', False):
            reasons.append("strong green candle")
        
        return "Signal: " + ", ".join(reasons)

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol (simplified - would need real-time data)"""
        
        # For now, use cached scan data if available
        if TODAY in self.scanner_cache:
            results = self.scanner_cache[TODAY]
            symbol_data = results[results['symbol'] == symbol]
            if not symbol_data.empty:
                return float(symbol_data.iloc[0]['close'])
        
        return None

    def save_state(self, filepath: str = 'chatbot_state.json'):
        """Save chatbot state to file"""
        state_dict = asdict(self.state)
        # Convert date objects to strings for JSON serialization
        if state_dict['last_scan_date']:
            state_dict['last_scan_date'] = state_dict['last_scan_date'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def load_state(self, filepath: str = 'chatbot_state.json'):
        """Load chatbot state from file"""
        try:
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            
            # Convert date strings back to date objects
            if state_dict['last_scan_date']:
                state_dict['last_scan_date'] = dt.datetime.fromisoformat(state_dict['last_scan_date']).date()
            
            self.state = ChatbotState(**state_dict)
        except FileNotFoundError:
            pass  # Use default state


def main():
    """Interactive chatbot loop"""
    
    print("🤖 PSX Trading Chatbot initialized!")
    print("Type 'quit' to exit, 'help' for commands\n")
    
    chatbot = TradingChatbot()
    chatbot.load_state()  # Load previous state if exists
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                chatbot.save_state()
                print("🤖 Goodbye! Your state has been saved.")
                break
            
            if not user_input:
                continue
            
            response = chatbot.process_query(user_input)
            print(f"🤖 {response}\n")
    
    except KeyboardInterrupt:
        chatbot.save_state()
        print("\n🤖 Session saved. Goodbye!")


if __name__ == '__main__':
    main()