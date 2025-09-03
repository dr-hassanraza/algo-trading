#!/usr/bin/env python3
"""
PSX Intraday Trading System - Complete Demo
===========================================

Comprehensive demonstration of the complete intraday trading system:
1. PSX DPS tick-by-tick data integration
2. Real-time signal analysis
3. Volume profile and momentum indicators
4. Risk management and position sizing
5. Live trading alerts and notifications

This shows the complete workflow from data fetching to trade execution.
"""

import datetime as dt
import time
import json
from typing import List, Dict

# Import our intraday trading components
from psx_dps_fetcher import PSXDPSFetcher
from intraday_signal_analyzer import IntradaySignalAnalyzer, SignalType
from intraday_risk_manager import IntradayRiskManager, RiskLevel

def demo_header():
    """Display demo header"""
    print("🚀 PSX Intraday Trading System - Complete Demo")
    print("=" * 65)
    print(f"📅 Demo Time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S PKT')}")
    print("=" * 65)

def demo_psx_dps_integration():
    """Demo PSX DPS tick-by-tick data integration"""
    print("\n🏛️ STEP 1: PSX DPS Tick-by-Tick Data Integration")
    print("-" * 55)
    
    fetcher = PSXDPSFetcher()
    
    # Test symbols
    symbols = ['FFC', 'UBL', 'LUCK']
    
    for symbol in symbols:
        print(f"\n📊 {symbol} - Live Tick Data:")
        try:
            # Get intraday ticks
            ticks = fetcher.fetch_intraday_ticks(symbol, limit=5)
            
            if not ticks.empty:
                print(f"   Total Trades Available: {len(ticks)}")
                print(f"   Latest Tick: {ticks.iloc[0]['time_pkt']} - {ticks.iloc[0]['price']:.2f} PKR (Vol: {ticks.iloc[0]['volume']:,})")
                print(f"   Price Range: {ticks['price'].min():.2f} - {ticks['price'].max():.2f} PKR")
                
                # Volume profile analysis
                volume_profile = fetcher.get_volume_profile(symbol, time_window_minutes=30)
                if volume_profile:
                    print(f"   VWAP (30min): {volume_profile['vwap']:.2f} PKR")
                    print(f"   Total Volume: {volume_profile['total_volume']:,} shares")
                
                # Price momentum
                momentum = fetcher.get_price_momentum(symbol, lookback_minutes=15)
                if momentum:
                    print(f"   Momentum: {momentum['momentum_direction']} ({momentum['momentum_strength']})")
                    print(f"   Price Velocity: {momentum['price_velocity']:.2f} PKR/min")
                
                # Liquidity analysis
                liquidity = fetcher.get_liquidity_analysis(symbol)
                if liquidity:
                    print(f"   Liquidity Level: {liquidity['liquidity_level']}")
                    print(f"   Avg Time Between Trades: {liquidity['avg_time_between_trades']:.1f} seconds")
            else:
                print(f"   ❌ No tick data available")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def demo_signal_analysis():
    """Demo real-time signal analysis"""
    print("\n\n📈 STEP 2: Real-Time Signal Analysis")
    print("-" * 45)
    
    analyzer = IntradaySignalAnalyzer()
    
    # Analyze multiple symbols
    test_symbols = ['UBL', 'FFC', 'LUCK', 'MCB']
    print(f"\n🔍 Analyzing {len(test_symbols)} symbols for trading opportunities...")
    
    signals = analyzer.analyze_multiple_symbols(test_symbols, analysis_period_minutes=20)
    
    # Display results
    actionable_signals = 0
    
    for signal in signals:
        signal_emoji = {
            SignalType.STRONG_BUY: "🟢🟢",
            SignalType.BUY: "🟢",
            SignalType.HOLD: "🟡",
            SignalType.SELL: "🔴",
            SignalType.STRONG_SELL: "🔴🔴"
        }.get(signal.signal_type, "⚪")
        
        print(f"\n{signal_emoji} {signal.symbol} - {signal.signal_type.value}")
        print(f"   Confidence: {signal.confidence:.1f}%")
        print(f"   Entry: {signal.entry_price:.2f} PKR")
        print(f"   Target: {signal.target_price:.2f} PKR")
        print(f"   Stop: {signal.stop_loss:.2f} PKR")
        print(f"   R/R Ratio: {signal.risk_reward_ratio:.2f}")
        print(f"   Reasoning: {'; '.join(signal.reasoning[:2])}")
        
        if signal.signal_type != SignalType.HOLD:
            actionable_signals += 1
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Total Symbols Analyzed: {len(signals)}")
    print(f"   Actionable Signals: {actionable_signals}")
    print(f"   Hold Recommendations: {len(signals) - actionable_signals}")
    
    return signals

def demo_risk_management(signals):
    """Demo risk management and position sizing"""
    print("\n\n⚖️ STEP 3: Risk Management & Position Sizing")
    print("-" * 50)
    
    # Initialize risk manager with moderate settings
    risk_manager = IntradayRiskManager(
        initial_capital=1000000,  # 10 lakh PKR
        risk_level=RiskLevel.MODERATE
    )
    
    print(f"💰 Initial Capital: {risk_manager.initial_capital:,} PKR")
    print(f"⚖️ Risk Level: {risk_manager.risk_level.value}")
    
    # Find actionable signals
    actionable_signals = [s for s in signals if s.signal_type != SignalType.HOLD and s.confidence >= 65]
    
    if not actionable_signals:
        print("\n❌ No actionable signals for risk analysis")
        return risk_manager
    
    print(f"\n🔍 Risk Analysis for {len(actionable_signals)} actionable signals:")
    
    successful_trades = 0
    
    for signal in actionable_signals:
        print(f"\n📊 {signal.symbol} - {signal.signal_type.value} ({signal.confidence:.1f}%)")
        
        # Calculate position size
        sizing = risk_manager.calculate_position_size(signal)
        
        if sizing['recommended_quantity'] > 0:
            print(f"   Recommended Quantity: {sizing['recommended_quantity']:,} shares")
            print(f"   Position Value: {sizing['position_value']:,.0f} PKR")
            print(f"   Risk Amount: {sizing['risk_amount']:,.0f} PKR")
            print(f"   Risk Percentage: {sizing['risk_percentage']:.2f}%")
            
            # Validate trade
            validation = risk_manager.validate_trade_entry(signal, sizing['recommended_quantity'])
            
            print(f"   Trade Valid: {'✅' if validation['is_valid'] else '❌'}")
            
            if validation['warnings']:
                print(f"   Warnings: {'; '.join(validation['warnings'][:2])}")
            
            if validation['blocking_issues']:
                print(f"   Issues: {'; '.join(validation['blocking_issues'][:2])}")
            
            # Execute trade if valid
            if validation['is_valid']:
                entry_result = risk_manager.enter_position(signal, sizing['recommended_quantity'])
                
                if entry_result['success']:
                    print(f"   ✅ Trade Executed!")
                    print(f"   Remaining Capital: {entry_result['capital_remaining']:,.0f} PKR")
                    successful_trades += 1
                else:
                    print(f"   ❌ Trade Failed: {entry_result['reason']}")
        else:
            print(f"   ❌ No position recommended: {sizing.get('reason', 'Unknown')}")
    
    print(f"\n📊 Trade Execution Summary:")
    print(f"   Successful Trades: {successful_trades}")
    print(f"   Active Positions: {len(risk_manager.positions)}")
    
    return risk_manager

def demo_position_monitoring(risk_manager):
    """Demo position monitoring and updates"""
    print("\n\n💼 STEP 4: Position Monitoring & Risk Updates")
    print("-" * 50)
    
    if not risk_manager.positions:
        print("❌ No active positions to monitor")
        return
    
    print(f"📊 Monitoring {len(risk_manager.positions)} active positions:")
    
    # Get position summary
    summary = risk_manager.get_position_summary()
    
    for pos in summary['positions']:
        print(f"\n📈 {pos['symbol']} ({pos['position_type']})")
        print(f"   Entry: {pos['entry_price']:.2f} PKR")
        print(f"   Current: {pos['current_price']:.2f} PKR")
        print(f"   Quantity: {pos['quantity']:,} shares")
        print(f"   Unrealized P&L: {pos['unrealized_pnl']:,.0f} PKR ({pos['pnl_pct']:+.2f}%)")
        print(f"   Stop Loss: {pos['stop_loss']:.2f} PKR")
        print(f"   Target: {pos['take_profit']:.2f} PKR")
        print(f"   Holding Time: {pos['holding_minutes']:.0f} minutes")
    
    print(f"\n📊 Portfolio Summary:")
    print(f"   Total Unrealized P&L: {summary['total_unrealized_pnl']:,.0f} PKR")
    
    # Update positions (check stops, targets, etc.)
    print(f"\n🔄 Updating positions with current market prices...")
    update_result = risk_manager.update_positions()
    
    print(f"   Positions Updated: {update_result['positions_updated']}")
    
    if update_result['stops_triggered']:
        for stop in update_result['stops_triggered']:
            print(f"   🛑 Stop Loss Triggered: {stop['symbol']} - P&L: {stop.get('realized_pnl', 0):,.0f} PKR")
    
    if update_result['targets_hit']:
        for target in update_result['targets_hit']:
            print(f"   🎯 Target Hit: {target['symbol']} - P&L: {target.get('realized_pnl', 0):,.0f} PKR")
    
    if update_result['trailing_stops_adjusted']:
        for trail in update_result['trailing_stops_adjusted']:
            print(f"   📈 Trailing Stop Updated: {trail['symbol']} - New Stop: {trail['new_trailing_stop']:.2f} PKR")

def demo_performance_metrics(risk_manager):
    """Demo performance metrics and analytics"""
    print("\n\n📊 STEP 5: Performance Metrics & Analytics")
    print("-" * 50)
    
    metrics = risk_manager.get_risk_metrics()
    
    print(f"💰 Capital Metrics:")
    print(f"   Initial Capital: {metrics.total_capital:,.0f} PKR")
    print(f"   Available Capital: {metrics.available_capital:,.0f} PKR")
    print(f"   Used Capital: {metrics.used_capital:,.0f} PKR")
    print(f"   Capital Utilization: {(metrics.used_capital / metrics.total_capital) * 100:.1f}%")
    
    print(f"\n⚖️ Risk Metrics:")
    print(f"   Total Exposure: {metrics.total_exposure:,.0f} PKR")
    print(f"   Current Drawdown: {metrics.current_drawdown:.1%}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.1%}")
    print(f"   Active Positions: {metrics.positions_count}")
    
    print(f"\n📈 Trading Performance:")
    print(f"   Daily P&L: {metrics.daily_pnl:,.0f} PKR")
    print(f"   Win Rate: {metrics.win_rate:.1%}")
    print(f"   Avg Win: {metrics.avg_win:,.0f} PKR")
    print(f"   Avg Loss: {metrics.avg_loss:,.0f} PKR")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    
    # Risk assessment
    risk_assessment = "LOW" if metrics.current_drawdown < 0.02 else \
                     "MODERATE" if metrics.current_drawdown < 0.05 else "HIGH"
    
    print(f"\n🚨 Risk Assessment: {risk_assessment}")
    
    # Trading recommendations
    print(f"\n💡 System Recommendations:")
    
    if metrics.win_rate > 0.6:
        print("   ✅ Good win rate - consider increasing position sizes")
    elif metrics.win_rate < 0.4:
        print("   ⚠️ Low win rate - review signal quality")
    
    if metrics.current_drawdown > 0.05:
        print("   ⚠️ High drawdown - consider reducing risk")
    
    if metrics.positions_count == 0:
        print("   💡 No active positions - look for new opportunities")
    elif metrics.positions_count > 5:
        print("   ⚠️ Many positions - monitor concentration risk")

def demo_live_alerts():
    """Demo live trading alerts"""
    print("\n\n🔔 STEP 6: Live Trading Alerts")
    print("-" * 40)
    
    analyzer = IntradaySignalAnalyzer()
    
    # Get live alerts
    watchlist = ['UBL', 'FFC', 'LUCK', 'MCB', 'HBL']
    print(f"📢 Checking for live alerts on {len(watchlist)} symbols...")
    
    alerts = analyzer.get_live_alerts(watchlist, min_confidence=70.0)
    
    if alerts:
        print(f"\n🚨 {len(alerts)} HIGH-CONFIDENCE ALERTS:")
        
        for i, alert in enumerate(alerts, 1):
            alert_emoji = "🟢🔥" if alert.signal_type == SignalType.STRONG_BUY else \
                         "🟢" if alert.signal_type == SignalType.BUY else \
                         "🔴" if alert.signal_type == SignalType.SELL else "🔴🔥"
            
            print(f"\n   {i}. {alert_emoji} {alert.symbol} - {alert.signal_type.value}")
            print(f"      Confidence: {alert.confidence:.1f}%")
            print(f"      Entry: {alert.entry_price:.2f} PKR")
            print(f"      Target: {alert.target_price:.2f} PKR")
            print(f"      Stop: {alert.stop_loss:.2f} PKR")
            print(f"      Expected Hold: {alert.holding_period_minutes} minutes")
            print(f"      Reasoning: {'; '.join(alert.reasoning[:1])}")
    else:
        print("\n📝 No high-confidence alerts at this time")
        print("   Market conditions may not favor aggressive trading")
        print("   Consider waiting for better setups")

def demo_market_conditions():
    """Demo market conditions analysis"""
    print("\n\n🌡️ STEP 7: Current Market Conditions")
    print("-" * 45)
    
    fetcher = PSXDPSFetcher()
    
    # Market status
    market_status = fetcher.get_market_status()
    status_emoji = "🟢" if market_status['is_trading'] else "🔴"
    
    print(f"{status_emoji} Market Status: {market_status['message']}")
    print(f"   Next Event: {market_status.get('next_event', 'Unknown')}")
    
    # Multi-symbol analysis
    print(f"\n📊 Multi-Symbol Market Analysis:")
    
    key_symbols = ['UBL', 'MCB', 'FFC', 'LUCK']
    total_volume = 0
    active_symbols = 0
    bullish_momentum = 0
    
    for symbol in key_symbols:
        try:
            momentum = fetcher.get_price_momentum(symbol, lookback_minutes=20)
            liquidity = fetcher.get_liquidity_analysis(symbol)
            
            if momentum and liquidity:
                active_symbols += 1
                
                momentum_emoji = "📈" if momentum['momentum_direction'] == 'Bullish' else \
                               "📉" if momentum['momentum_direction'] == 'Bearish' else "➡️"
                
                print(f"   {momentum_emoji} {symbol}: {momentum['momentum_direction']} "
                      f"({momentum['momentum_strength']}) - {liquidity['liquidity_level']} Liquidity")
                
                if momentum['momentum_direction'] == 'Bullish':
                    bullish_momentum += 1
                
        except Exception as e:
            print(f"   ❌ {symbol}: Analysis error")
    
    print(f"\n📈 Market Sentiment Analysis:")
    bullish_pct = (bullish_momentum / active_symbols * 100) if active_symbols > 0 else 0
    
    if bullish_pct > 60:
        sentiment = "🟢 BULLISH"
    elif bullish_pct > 40:
        sentiment = "🟡 NEUTRAL"
    else:
        sentiment = "🔴 BEARISH"
    
    print(f"   Overall Sentiment: {sentiment} ({bullish_pct:.0f}% bullish)")
    print(f"   Active Symbols: {active_symbols}/{len(key_symbols)}")
    
    # Trading recommendations based on market
    print(f"\n💡 Market-Based Recommendations:")
    
    if market_status['is_trading']:
        if bullish_pct > 60:
            print("   🟢 Market favors long positions")
            print("   🟢 Consider scaling into positions")
        elif bullish_pct < 40:
            print("   🔴 Market shows bearish sentiment")
            print("   🔴 Consider defensive strategies")
        else:
            print("   🟡 Mixed market conditions")
            print("   🟡 Trade selectively with tight stops")
    else:
        print("   ⏰ Market closed - prepare for next session")
        print("   📋 Review positions and plan trades")

def demo_conclusion():
    """Demo conclusion and system summary"""
    print("\n\n🎯 SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 65)
    
    print("\n✅ Successfully Demonstrated:")
    print("   🏛️ PSX DPS Real-time tick-by-tick data integration")
    print("   📈 Advanced intraday signal analysis with volume profile")
    print("   ⚖️ Comprehensive risk management and position sizing")
    print("   💼 Real-time position monitoring and stop management")
    print("   📊 Performance metrics and risk analytics")
    print("   🔔 Live trading alerts with confidence scoring")
    print("   🌡️ Market condition analysis and sentiment tracking")
    
    print("\n🚀 Key System Capabilities:")
    print("   • Real-time data: Official PSX DPS tick data")
    print("   • Signal quality: 65%+ confidence actionable signals")
    print("   • Risk control: Dynamic position sizing with stops")
    print("   • Speed: Sub-second signal generation")
    print("   • Automation: Auto stop-loss and profit taking")
    print("   • Monitoring: Live position and P&L tracking")
    
    print("\n💡 Next Steps:")
    print("   1. Run 'streamlit run live_trading_dashboard.py' for GUI")
    print("   2. Customize watchlist and risk parameters")
    print("   3. Monitor live signals during market hours")
    print("   4. Start with paper trading to validate strategies")
    print("   5. Scale up with real capital after testing")
    
    print("\n🎉 Your PSX intraday trading system is ready!")
    print("   Time to catch those intraday moves! 📈💰")

def main():
    """Run complete system demonstration"""
    
    demo_header()
    
    try:
        # Step 1: Data Integration
        demo_psx_dps_integration()
        
        # Step 2: Signal Analysis
        signals = demo_signal_analysis()
        
        # Step 3: Risk Management
        risk_manager = demo_risk_management(signals)
        
        # Step 4: Position Monitoring
        demo_position_monitoring(risk_manager)
        
        # Step 5: Performance Metrics
        demo_performance_metrics(risk_manager)
        
        # Step 6: Live Alerts
        demo_live_alerts()
        
        # Step 7: Market Conditions
        demo_market_conditions()
        
        # Conclusion
        demo_conclusion()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        print("   Check system components and try again")

if __name__ == "__main__":
    main()