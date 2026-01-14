#!/usr/bin/env python3
"""
PSX PROFESSIONAL HITLIST - Main Trading Dashboard
==================================================
Clean, actionable trading signals for Pakistan Stock Exchange

Run this file to get today's top trading opportunities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, time
from typing import List, Dict, Tuple
import concurrent.futures

from enhanced_signal_engine import EnhancedSignalEngine, EnhancedSignal
from smart_risk_manager import SmartRiskManager, RiskProfile
from signal_tracker import get_tracker

# Try to import all_psx_tickers, fallback to default list
try:
    from all_psx_tickers import STOCK_SYMBOLS_ONLY
    PSX_SYMBOLS = STOCK_SYMBOLS_ONLY[:100]  # Top 100 stocks
except ImportError:
    PSX_SYMBOLS = [
        # Banks
        'HBL', 'UBL', 'MCB', 'ABL', 'NBP', 'BAFL', 'BAHL', 'MEBL',
        # Cement
        'LUCK', 'DGKC', 'MLCF', 'FCCL', 'KOHC', 'PIOC', 'CHCC',
        # Fertilizer
        'FFC', 'EFERT', 'FFBL', 'ENGRO',
        # Oil & Gas
        'PSO', 'OGDC', 'PPL', 'POL', 'MARI', 'SSGC', 'SNGP',
        # Power
        'HUBC', 'KEL', 'NCPL', 'KAPCO',
        # Technology
        'TRG', 'SYS', 'NETSOL',
        # Autos
        'INDU', 'PSMC', 'HCAR', 'MTL',
        # Pharma
        'SEARL', 'GLAXO', 'FEROZ',
        # Others
        'NESTLE', 'ICI', 'EPCL', 'LOTCHEM'
    ]


class PSXHitlistGenerator:
    """
    Generate professional-grade trading hitlist
    """

    def __init__(self, capital: float = 1000000, risk_profile: str = 'MODERATE'):
        self.signal_engine = EnhancedSignalEngine()
        profile = RiskProfile[risk_profile.upper()]
        self.risk_manager = SmartRiskManager(initial_capital=capital, risk_profile=profile)
        self.tracker = get_tracker()

    def analyze_symbol(self, symbol: str) -> Tuple[str, EnhancedSignal]:
        """Analyze a single symbol"""
        try:
            signal = self.signal_engine.generate_signal(symbol)
            return (symbol, signal)
        except Exception as e:
            return (symbol, None)

    def generate_hitlist(self, symbols: List[str] = None,
                        max_workers: int = 10) -> Dict:
        """
        Generate complete trading hitlist with parallel processing
        """
        if symbols is None:
            symbols = PSX_SYMBOLS

        # Analyze all symbols in parallel
        signals = []
        failed = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.analyze_symbol, sym): sym for sym in symbols}

            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    sym, signal = future.result()
                    if signal:
                        signals.append(signal)
                    else:
                        failed.append(sym)
                except Exception as e:
                    failed.append(symbol)

        # Categorize and rank signals
        buy_signals = [s for s in signals if s.signal == 'BUY' and not s.avoid_entry]
        sell_signals = [s for s in signals if s.signal == 'SELL' and not s.avoid_entry]
        hold_signals = [s for s in signals if s.signal == 'HOLD']

        # Sort by confidence and quality
        def signal_score(s):
            quality_scores = {'A': 40, 'B': 30, 'C': 20, 'D': 10}
            return s.confidence + quality_scores.get(s.signal_quality, 0) + s.risk_reward_ratio * 5

        buy_signals.sort(key=signal_score, reverse=True)
        sell_signals.sort(key=signal_score, reverse=True)

        # Get portfolio risk status
        portfolio_risk = self.risk_manager.get_portfolio_risk()

        # Build hitlist result
        hitlist = {
            'generated_at': datetime.now().isoformat(),
            'market_status': self._get_market_status(),
            'symbols_analyzed': len(symbols),
            'signals_generated': len(signals),
            'failed_symbols': len(failed),

            'summary': {
                'buy_signals': len(buy_signals),
                'sell_signals': len(sell_signals),
                'hold_signals': len(hold_signals),
                'high_confidence': len([s for s in signals if s.confidence >= 75]),
                'grade_a_signals': len([s for s in signals if s.signal_quality == 'A']),
            },

            'portfolio': {
                'capital': portfolio_risk.total_capital,
                'cash_available': portfolio_risk.cash_available,
                'positions': portfolio_risk.total_positions,
                'heat': portfolio_risk.portfolio_heat,
                'can_trade': portfolio_risk.can_take_new_position,
            },

            'top_buys': buy_signals[:10],
            'top_sells': sell_signals[:5],
            'all_signals': signals,
        }

        return hitlist

    def _get_market_status(self) -> Dict:
        """Get current market status"""
        now = datetime.now()
        current_time = now.time()

        market_open = time(9, 30)
        market_close = time(15, 30)

        is_weekday = now.weekday() < 5
        is_market_hours = market_open <= current_time <= market_close

        if not is_weekday:
            status = "CLOSED (Weekend)"
            session = "WEEKEND"
        elif current_time < market_open:
            status = "PRE-MARKET"
            session = "PRE_MARKET"
        elif current_time > market_close:
            status = "AFTER-HOURS"
            session = "AFTER_HOURS"
        elif time(9, 30) <= current_time < time(10, 0):
            status = "OPEN (Opening Session)"
            session = "OPENING"
        elif time(10, 0) <= current_time < time(12, 30):
            status = "OPEN (Morning Session)"
            session = "MORNING"
        elif time(12, 30) <= current_time < time(13, 30):
            status = "OPEN (Lunch Session)"
            session = "LUNCH"
        elif time(13, 30) <= current_time < time(15, 0):
            status = "OPEN (Afternoon Session)"
            session = "AFTERNOON"
        else:
            status = "OPEN (Closing Session)"
            session = "CLOSING"

        return {
            'status': status,
            'session': session,
            'is_open': is_weekday and is_market_hours,
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S')
        }

    def format_hitlist(self, hitlist: Dict) -> str:
        """Format hitlist as clean text output"""

        output = []

        # Header
        output.append("")
        output.append("=" * 70)
        output.append("              PSX PROFESSIONAL TRADING HITLIST")
        output.append("=" * 70)
        output.append(f"  Generated: {hitlist['generated_at'][:19]}")
        output.append(f"  Market: {hitlist['market_status']['status']}")
        output.append("=" * 70)
        output.append("")

        # Quick Summary
        summary = hitlist['summary']
        output.append("MARKET OVERVIEW")
        output.append("-" * 70)
        output.append(f"  Stocks Analyzed: {hitlist['symbols_analyzed']}")
        output.append(f"  BUY Signals: {summary['buy_signals']} | SELL Signals: {summary['sell_signals']} | HOLD: {summary['hold_signals']}")
        output.append(f"  High Confidence (75%+): {summary['high_confidence']} | Grade A Signals: {summary['grade_a_signals']}")
        output.append("")

        # Portfolio Status
        portfolio = hitlist['portfolio']
        output.append("PORTFOLIO STATUS")
        output.append("-" * 70)
        output.append(f"  Capital: {portfolio['capital']:,.0f} PKR | Cash: {portfolio['cash_available']:,.0f} PKR")
        output.append(f"  Active Positions: {portfolio['positions']} | Risk Heat: {portfolio['heat']:.0f}/100")
        output.append(f"  Can Take New Position: {'Yes' if portfolio['can_trade'] else 'No - Check Limits'}")
        output.append("")

        # Top BUY Signals
        output.append("=" * 70)
        output.append("                    TOP BUY OPPORTUNITIES")
        output.append("=" * 70)
        output.append("")

        buy_signals = hitlist['top_buys']

        if not buy_signals:
            output.append("  No strong BUY signals at this time.")
        else:
            output.append(f"  {'#':<3} {'SYMBOL':<8} {'GRADE':<6} {'CONF':<6} {'ENTRY':<10} {'STOP':<10} {'TARGET':<10} {'R/R':<5} {'POS%':<5}")
            output.append("  " + "-" * 68)

            for i, signal in enumerate(buy_signals[:8], 1):
                grade_display = f"[{signal.signal_quality}]"
                output.append(
                    f"  {i:<3} {signal.symbol:<8} {grade_display:<6} {signal.confidence:>5.0f}% "
                    f"{signal.entry_price:>9.2f} {signal.stop_loss:>9.2f} {signal.take_profit_1:>9.2f} "
                    f"{signal.risk_reward_ratio:>4.1f} {signal.position_size_pct:>4.1f}%"
                )

                # Show reasons for top 3
                if i <= 3 and signal.reasons:
                    reasons_str = " | ".join(signal.reasons[:2])
                    output.append(f"      Reasons: {reasons_str}")

                    if signal.warnings:
                        output.append(f"      Warning: {signal.warnings[0]}")

                    output.append("")

        output.append("")

        # Top SELL Signals
        output.append("=" * 70)
        output.append("                    TOP SELL OPPORTUNITIES")
        output.append("=" * 70)
        output.append("")

        sell_signals = hitlist['top_sells']

        if not sell_signals:
            output.append("  No strong SELL signals at this time.")
        else:
            output.append(f"  {'#':<3} {'SYMBOL':<8} {'GRADE':<6} {'CONF':<6} {'ENTRY':<10} {'STOP':<10} {'TARGET':<10} {'R/R':<5}")
            output.append("  " + "-" * 63)

            for i, signal in enumerate(sell_signals[:5], 1):
                grade_display = f"[{signal.signal_quality}]"
                output.append(
                    f"  {i:<3} {signal.symbol:<8} {grade_display:<6} {signal.confidence:>5.0f}% "
                    f"{signal.entry_price:>9.2f} {signal.stop_loss:>9.2f} {signal.take_profit_1:>9.2f} "
                    f"{signal.risk_reward_ratio:>4.1f}"
                )

        output.append("")

        # Detailed Analysis of Top Pick
        if buy_signals:
            output.append("=" * 70)
            output.append("                    DETAILED: TOP PICK")
            output.append("=" * 70)
            output.append("")

            top = buy_signals[0]
            output.append(f"  Symbol: {top.symbol}")
            output.append(f"  Signal: {top.signal} ({top.strength.value})")
            output.append(f"  Quality Grade: {top.signal_quality}")
            output.append(f"  Confidence: {top.confidence:.1f}%")
            output.append("")
            output.append(f"  ENTRY LEVELS:")
            output.append(f"    Entry Price:     {top.entry_price:,.2f} PKR")
            output.append(f"    Stop Loss:       {top.stop_loss:,.2f} PKR ({((top.stop_loss/top.entry_price)-1)*100:+.1f}%)")
            output.append(f"    Target 1:        {top.take_profit_1:,.2f} PKR ({((top.take_profit_1/top.entry_price)-1)*100:+.1f}%)")
            output.append(f"    Target 2:        {top.take_profit_2:,.2f} PKR ({((top.take_profit_2/top.entry_price)-1)*100:+.1f}%)")
            output.append(f"    Risk/Reward:     {top.risk_reward_ratio:.2f}")
            output.append("")
            output.append(f"  TECHNICAL ANALYSIS:")
            output.append(f"    RSI:             {top.rsi:.1f}")
            output.append(f"    ATR:             {top.atr:.2f} ({top.atr_percent:.1f}%)")
            output.append(f"    Volume Ratio:    {top.volume_ratio:.2f}x")
            output.append(f"    Support:         {top.nearest_support:,.2f} ({top.distance_to_support_pct:.1f}% away)")
            output.append(f"    Resistance:      {top.nearest_resistance:,.2f} ({top.distance_to_resistance_pct:.1f}% away)")
            output.append("")
            output.append(f"  SCORES:")
            output.append(f"    Technical:       {top.technical_score:.0f}/100")
            output.append(f"    Momentum:        {top.momentum_score:.0f}/100")
            output.append(f"    Volume:          {top.volume_score:.0f}/100")
            output.append(f"    Overall:         {top.trend_score:.0f}/100")
            output.append("")

            if top.bullish_divergence:
                output.append("  ** BULLISH DIVERGENCE DETECTED **")
            if top.bearish_divergence:
                output.append("  ** BEARISH DIVERGENCE DETECTED **")

            output.append(f"  REASONS:")
            for reason in top.reasons:
                output.append(f"    - {reason}")

            if top.warnings:
                output.append("")
                output.append(f"  WARNINGS:")
                for warning in top.warnings:
                    output.append(f"    ! {warning}")

            output.append("")
            output.append(f"  POSITION SIZING:")
            output.append(f"    Recommended: {top.position_size_pct:.1f}% of portfolio")
            output.append(f"    Entry Window: {top.optimal_entry_window}")

        output.append("")

        # Market Sentiment
        all_signals = hitlist['all_signals']
        if all_signals:
            bullish = len([s for s in all_signals if s.signal == 'BUY'])
            bearish = len([s for s in all_signals if s.signal == 'SELL'])
            total = len(all_signals)

            sentiment_pct = (bullish - bearish) / total * 100 if total > 0 else 0

            output.append("=" * 70)
            output.append("                    MARKET SENTIMENT")
            output.append("=" * 70)
            output.append("")

            if sentiment_pct > 20:
                sentiment = "BULLISH"
                indicator = "[>>>>>>>>  ]"
            elif sentiment_pct > 5:
                sentiment = "SLIGHTLY BULLISH"
                indicator = "[>>>>>     ]"
            elif sentiment_pct < -20:
                sentiment = "BEARISH"
                indicator = "[  <<<<<<<<]"
            elif sentiment_pct < -5:
                sentiment = "SLIGHTLY BEARISH"
                indicator = "[     <<<<<]"
            else:
                sentiment = "NEUTRAL"
                indicator = "[    <>    ]"

            output.append(f"  Overall Sentiment: {sentiment}")
            output.append(f"  {indicator}")
            output.append(f"  Bullish Signals: {bullish} | Bearish Signals: {bearish}")
            output.append("")

            # Average confidence by sector
            sector_signals = {}
            for s in all_signals:
                sector = self.signal_engine.get_sector(s.symbol)
                if sector not in sector_signals:
                    sector_signals[sector] = []
                sector_signals[sector].append(s)

            output.append("  SECTOR BREAKDOWN:")
            for sector in sorted(sector_signals.keys()):
                sigs = sector_signals[sector]
                buys = len([s for s in sigs if s.signal == 'BUY'])
                sells = len([s for s in sigs if s.signal == 'SELL'])
                output.append(f"    {sector:<12}: {buys} BUY, {sells} SELL")

        output.append("")

        # Risk Disclaimer
        output.append("=" * 70)
        output.append("                       DISCLAIMER")
        output.append("=" * 70)
        output.append("  This is algorithmic analysis for educational purposes only.")
        output.append("  - Always conduct your own research before trading")
        output.append("  - Past performance does not guarantee future results")
        output.append("  - Only trade with capital you can afford to lose")
        output.append("  - Consider your personal risk tolerance")
        output.append("=" * 70)
        output.append("")

        return "\n".join(output)


def main():
    """Main entry point - generate and display hitlist"""
    print("\nInitializing PSX Professional Hitlist Generator...")
    print("Analyzing market data. Please wait...\n")

    # Create generator with default settings
    generator = PSXHitlistGenerator(
        capital=1000000,  # 1 Million PKR
        risk_profile='MODERATE'
    )

    # Generate hitlist
    hitlist = generator.generate_hitlist()

    # Format and display
    formatted = generator.format_hitlist(hitlist)
    print(formatted)

    # Save to file
    output_file = 'PSX_HITLIST_CURRENT.md'
    with open(output_file, 'w') as f:
        f.write(formatted)

    print(f"\nHitlist saved to {output_file}")

    # Track signals for performance monitoring
    tracker = get_tracker()
    for signal in hitlist['top_buys'][:5]:
        if signal.signal_quality in ['A', 'B']:
            tracker.record_signal(
                symbol=signal.symbol,
                signal_type=signal.signal,
                confidence=signal.confidence,
                quality_grade=signal.signal_quality,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit_1
            )

    return hitlist


if __name__ == "__main__":
    main()
