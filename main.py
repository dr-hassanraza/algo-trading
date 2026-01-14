#!/usr/bin/env python3
"""
PSX ALGO TRADING SYSTEM - Main Entry Point
============================================
Professional algorithmic trading system for Pakistan Stock Exchange

Usage:
    python main.py                    # Show today's hitlist (default)
    python main.py --hitlist          # Generate trading hitlist
    python main.py --analyze SYMBOL   # Analyze specific stock
    python main.py --portfolio        # Show portfolio status
    python main.py --performance      # Show signal performance
    python main.py --help             # Show help
"""

import sys
import os
import argparse
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Print application banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║      ██████╗ ███████╗██╗  ██╗    ████████╗██████╗  █████╗ ██████╗ ║
    ║      ██╔══██╗██╔════╝╚██╗██╔╝    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗║
    ║      ██████╔╝███████╗ ╚███╔╝        ██║   ██████╔╝███████║██║  ██║║
    ║      ██╔═══╝ ╚════██║ ██╔██╗        ██║   ██╔══██╗██╔══██║██║  ██║║
    ║      ██║     ███████║██╔╝ ██╗       ██║   ██║  ██║██║  ██║██████╔╝║
    ║      ╚═╝     ╚══════╝╚═╝  ╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ║
    ║                                                                   ║
    ║              Professional Algorithmic Trading System              ║
    ║                   Pakistan Stock Exchange (PSX)                   ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def show_hitlist():
    """Generate and display trading hitlist"""
    from psx_hitlist_pro import PSXHitlistGenerator

    print("\n  Analyzing market data...")
    print("  This may take a moment...\n")

    generator = PSXHitlistGenerator(capital=1000000, risk_profile='MODERATE')
    hitlist = generator.generate_hitlist()
    formatted = generator.format_hitlist(hitlist)

    print(formatted)

    # Save to file
    with open('PSX_HITLIST_CURRENT.md', 'w') as f:
        f.write(formatted)

    print(f"\n  Hitlist saved to PSX_HITLIST_CURRENT.md")

    return hitlist


def analyze_symbol(symbol: str):
    """Detailed analysis of a specific symbol"""
    from enhanced_signal_engine import EnhancedSignalEngine
    from smart_risk_manager import SmartRiskManager, RiskProfile

    print(f"\n  Analyzing {symbol}...")
    print("  " + "=" * 60)

    engine = EnhancedSignalEngine()
    signal = engine.generate_signal(symbol)

    if not signal:
        print(f"\n  Unable to analyze {symbol}. No data available.")
        return

    # Display detailed analysis
    print(f"""
  SYMBOL: {signal.symbol}
  {'=' * 60}

  SIGNAL SUMMARY
  --------------
  Signal:          {signal.signal} ({signal.strength.value})
  Confidence:      {signal.confidence:.1f}%
  Quality Grade:   {signal.signal_quality}

  ENTRY LEVELS
  ------------
  Entry Price:     {signal.entry_price:,.2f} PKR
  Stop Loss:       {signal.stop_loss:,.2f} PKR ({((signal.stop_loss/signal.entry_price)-1)*100:+.1f}%)
  Target 1:        {signal.take_profit_1:,.2f} PKR ({((signal.take_profit_1/signal.entry_price)-1)*100:+.1f}%)
  Target 2:        {signal.take_profit_2:,.2f} PKR ({((signal.take_profit_2/signal.entry_price)-1)*100:+.1f}%)
  Risk/Reward:     {signal.risk_reward_ratio:.2f}

  TECHNICAL INDICATORS
  --------------------
  RSI:             {signal.rsi:.1f}
  ATR:             {signal.atr:.2f} ({signal.atr_percent:.1f}%)
  Volume Ratio:    {signal.volume_ratio:.2f}x average

  SUPPORT/RESISTANCE
  ------------------
  Nearest Support:    {signal.nearest_support:,.2f} ({signal.distance_to_support_pct:.1f}% away)
  Nearest Resistance: {signal.nearest_resistance:,.2f} ({signal.distance_to_resistance_pct:.1f}% away)

  SCORES (0-100)
  --------------
  Technical Score:  {signal.technical_score:.0f}
  Momentum Score:   {signal.momentum_score:.0f}
  Volume Score:     {signal.volume_score:.0f}
  Overall Score:    {signal.trend_score:.0f}

  DIVERGENCE DETECTION
  --------------------
  Bullish Divergence: {'Yes - Potential reversal UP' if signal.bullish_divergence else 'No'}
  Bearish Divergence: {'Yes - Potential reversal DOWN' if signal.bearish_divergence else 'No'}

  SIGNAL REASONS
  --------------""")

    for reason in signal.reasons:
        print(f"  - {reason}")

    if signal.warnings:
        print("\n  WARNINGS")
        print("  --------")
        for warning in signal.warnings:
            print(f"  ! {warning}")

    print(f"""
  POSITION RECOMMENDATION
  -----------------------
  Suggested Size:   {signal.position_size_pct:.1f}% of portfolio
  Entry Window:     {signal.optimal_entry_window}
  Avoid Entry:      {'Yes - Wait for better setup' if signal.avoid_entry else 'No - Signal is actionable'}
  {'=' * 60}
""")

    # Position sizing
    rm = SmartRiskManager(initial_capital=1000000, risk_profile=RiskProfile.MODERATE)
    sizing = rm.calculate_position_size(
        symbol=symbol,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        signal_confidence=signal.confidence
    )

    print(f"""
  POSITION SIZING (Based on 1M PKR capital)
  -----------------------------------------
  Recommended Shares: {sizing['recommended_shares']:,}
  Position Value:     {sizing['position_value']:,.0f} PKR
  Risk Amount:        {sizing['risk_amount']:,.0f} PKR
  Position %:         {sizing['position_pct']:.1f}%
  Can Trade:          {'Yes' if sizing['can_trade'] else 'No'}
""")

    if sizing['warnings']:
        print("  Sizing Warnings:")
        for w in sizing['warnings']:
            print(f"    - {w}")

    if sizing['blockers']:
        print("  Sizing Blockers:")
        for b in sizing['blockers']:
            print(f"    ! {b}")

    print()


def show_portfolio():
    """Show current portfolio status"""
    from smart_risk_manager import SmartRiskManager, RiskProfile

    rm = SmartRiskManager(initial_capital=1000000, risk_profile=RiskProfile.MODERATE)
    risk = rm.get_portfolio_risk()
    positions = rm.get_position_risks()

    print(f"""
  PORTFOLIO STATUS
  {'=' * 60}

  CAPITAL
  -------
  Total Capital:     {risk.total_capital:,.0f} PKR
  Cash Available:    {risk.cash_available:,.0f} PKR
  Total Invested:    {risk.total_invested:,.0f} PKR
  Exposure:          {risk.total_exposure_pct:.1f}%

  RISK METRICS
  ------------
  Portfolio Heat:    {risk.portfolio_heat:.0f}/100
  Unrealized P&L:    {risk.unrealized_pnl:+,.0f} PKR
  Today's P&L:       {risk.realized_pnl_today:+,.0f} PKR
  Daily VaR (95%):   {risk.daily_var_95:,.0f} PKR

  POSITIONS
  ---------
  Active Positions:  {risk.total_positions}
  Can Take New:      {'Yes' if risk.can_take_new_position else 'No'}
  Slots Remaining:   {risk.position_limit_remaining}
""")

    if positions:
        print("  ACTIVE POSITIONS")
        print("  " + "-" * 58)
        print(f"  {'Symbol':<8} {'Sector':<10} {'Entry':<10} {'Current':<10} {'P&L':<10} {'P&L%':<8}")
        print("  " + "-" * 58)

        for pos in positions:
            pnl_color = '+' if pos.unrealized_pnl >= 0 else ''
            print(f"  {pos.symbol:<8} {pos.sector:<10} {pos.entry_price:>9.2f} "
                  f"{pos.current_price:>9.2f} {pnl_color}{pos.unrealized_pnl:>9.0f} "
                  f"{pnl_color}{pos.unrealized_pnl_pct:>7.1f}%")

        print()

    if risk.sector_exposure:
        print("  SECTOR EXPOSURE")
        print("  " + "-" * 40)
        for sector, value in sorted(risk.sector_exposure.items(), key=lambda x: x[1], reverse=True):
            pct = (value / risk.total_capital) * 100
            print(f"  {sector:<15} {value:>12,.0f} PKR ({pct:>5.1f}%)")

    print()


def show_performance():
    """Show signal performance statistics"""
    from signal_tracker import get_tracker

    tracker = get_tracker()
    report = tracker.generate_report()
    print(report)


def show_market_status():
    """Show current market status"""
    from psx_market_filter import PSXMarketFilter

    filter = PSXMarketFilter()
    session = filter.get_trading_session()

    print(f"""
  MARKET STATUS
  {'=' * 60}

  Session:          {session['name']}
  Market Open:      {'Yes' if session['is_market_open'] else 'No'}
  Time:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

  RECOMMENDATION
  --------------
  {session['recommendation']}

  Position Size Modifier: {session['position_size_modifier']:.0%}
  Avoid New Positions:    {'Yes' if session['avoid_new_positions'] else 'No'}
  {'=' * 60}
""")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='PSX Professional Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Show today's trading hitlist
  python main.py --analyze HBL      Analyze HBL stock in detail
  python main.py --portfolio        View portfolio status
  python main.py --performance      View signal performance stats
  python main.py --market           View market status
        """
    )

    parser.add_argument('--hitlist', '-l', action='store_true',
                       help='Generate trading hitlist (default)')
    parser.add_argument('--analyze', '-a', type=str, metavar='SYMBOL',
                       help='Analyze specific stock symbol')
    parser.add_argument('--portfolio', '-p', action='store_true',
                       help='Show portfolio status')
    parser.add_argument('--performance', '-s', action='store_true',
                       help='Show signal performance stats')
    parser.add_argument('--market', '-m', action='store_true',
                       help='Show market status')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output (no banner)')

    args = parser.parse_args()

    # Show banner unless quiet mode
    if not args.quiet:
        print_banner()

    # Execute requested action
    if args.analyze:
        analyze_symbol(args.analyze.upper())
    elif args.portfolio:
        show_portfolio()
    elif args.performance:
        show_performance()
    elif args.market:
        show_market_status()
    else:
        # Default: show hitlist
        show_hitlist()


if __name__ == "__main__":
    main()
