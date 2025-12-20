#!/usr/bin/env python3
"""
Intraday Signal Analyzer for PSX High-Frequency Trading
=======================================================

Advanced intraday trading signals using PSX DPS tick-by-tick data.
Provides real-time buy/sell signals based on:
- Volume profile analysis
- Price momentum indicators  
- Liquidity analysis
- Support/resistance levels
- Trade frequency patterns

Perfect for scalping and day trading strategies.
"""

import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Import our PSX DPS fetcher
from psx_dps_fetcher import PSXDPSFetcher

logger = logging.getLogger(__name__)

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY" 
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class IntradaySignal:
    """Structured intraday trading signal"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-100
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    volume_support: bool
    momentum_direction: str
    liquidity_level: str
    analysis_time: dt.datetime
    holding_period_minutes: int
    reasoning: List[str]

class IntradaySignalAnalyzer:
    """Real-time intraday signal analyzer using PSX DPS tick data"""
    
    def __init__(self, ml_biases: Optional[Dict[str, float]] = None):
        self.fetcher = PSXDPSFetcher()
        self.ml_biases = ml_biases if ml_biases is not None else {}
        self.min_trades_required = 5  # Minimum trades for analysis
        self.min_volume_required = 1000  # Minimum volume for signals
        
        # Signal thresholds
        self.strong_momentum_threshold = 1.5  # % price change
        self.moderate_momentum_threshold = 0.5  # % price change
        self.high_volume_threshold = 2.0  # Multiple of average volume
        self.liquidity_threshold = 5.0  # Liquidity score threshold

    def _get_ml_adjustment(self, symbol: str) -> float:
        """Get an adjustment factor based on the ML bias."""
        bias = self.ml_biases.get(symbol, 0.0)  # e.g., predicted return
        
        # Say a 1% predicted return maps to a 15 point adjustment.
        # Cap the adjustment at +/- 25 points.
        adjustment = np.clip(bias * 1500, -25, 25)
        return adjustment
        
    def analyze_symbol(self, symbol: str, analysis_period_minutes: int = 30) -> IntradaySignal:
        """
        Comprehensive intraday analysis for a single symbol
        
        Args:
            symbol: Stock symbol to analyze
            analysis_period_minutes: Period for analysis (default 30 minutes)
            
        Returns:
            IntradaySignal with complete analysis
        """
        try:
            # Fetch tick data
            ticks_df = self.fetcher.fetch_intraday_ticks(symbol, limit=200)
            
            if ticks_df.empty or len(ticks_df) < self.min_trades_required:
                return self._create_hold_signal(symbol, "Insufficient tick data")
            
            # Get various analyses
            momentum = self.fetcher.get_price_momentum(symbol, analysis_period_minutes)
            volume_profile = self.fetcher.get_volume_profile(symbol, analysis_period_minutes)
            liquidity = self.fetcher.get_liquidity_analysis(symbol)
            
            if not momentum or not volume_profile or not liquidity:
                return self._create_hold_signal(symbol, "Analysis data incomplete")
            
            # Current market conditions
            current_price = momentum['current_price']
            
            # Check minimum volume requirement
            if volume_profile.get('total_volume', 0) < self.min_volume_required:
                return self._create_hold_signal(symbol, "Volume too low for reliable signals")
            
            # Analyze signal components
            momentum_score = self._analyze_momentum(momentum)
            volume_score = self._analyze_volume(volume_profile, ticks_df)
            liquidity_score = self._analyze_liquidity(liquidity)
            technical_score = self._analyze_technical_levels(ticks_df, current_price)
            
            # Combine scores
            base_score = (momentum_score * 0.35 + 
                           volume_score * 0.25 + 
                           liquidity_score * 0.20 + 
                           technical_score * 0.20)
            
            # Get ML adjustment
            ml_adjustment = self._get_ml_adjustment(symbol)
            momentum_direction = momentum.get('momentum_direction', 'Neutral')
            
            if momentum_direction == 'Bullish':
                overall_score = base_score + ml_adjustment
            elif momentum_direction == 'Bearish':
                overall_score = base_score - ml_adjustment
            else:
                overall_score = base_score
            
            overall_score = np.clip(overall_score, 0, 100)

            # Generate signal
            signal_type, confidence = self._determine_signal_type(
                overall_score, momentum, volume_profile, liquidity
            )
            
            # Calculate entry/exit levels
            entry_price = current_price
            target_price, stop_loss = self._calculate_levels(
                current_price, signal_type, momentum, volume_profile
            )
            
            # Risk/reward calculation
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                risk_reward = abs(target_price - entry_price) / abs(entry_price - stop_loss)
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                risk_reward = abs(entry_price - target_price) / abs(stop_loss - entry_price)
            else:
                risk_reward = 0.0
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                momentum, volume_profile, liquidity, momentum_score, volume_score, ml_adjustment
            )
            
            # Determine holding period
            holding_period = self._determine_holding_period(signal_type, momentum, liquidity)
            
            return IntradaySignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                risk_reward_ratio=risk_reward,
                volume_support=volume_score > 60,
                momentum_direction=momentum['momentum_direction'],
                liquidity_level=liquidity['liquidity_level'],
                analysis_time=dt.datetime.now(),
                holding_period_minutes=holding_period,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Intraday analysis error for {symbol}: {e}")
            return self._create_hold_signal(symbol, f"Analysis error: {str(e)}")
    
    def _analyze_momentum(self, momentum: Dict) -> float:
        """Analyze price momentum and return score 0-100"""
        try:
            price_change_pct = abs(momentum.get('price_change_pct', 0))
            momentum_direction = momentum.get('momentum_direction', 'Neutral')
            price_velocity = abs(momentum.get('price_velocity', 0))
            trade_frequency = momentum.get('trade_frequency', 0)
            
            score = 0
            
            # Price change component (40 points)
            if price_change_pct > self.strong_momentum_threshold:
                score += 40
            elif price_change_pct > self.moderate_momentum_threshold:
                score += 25
            else:
                score += 10
            
            # Direction clarity (20 points)
            if momentum_direction != 'Neutral':
                score += 20
            
            # Velocity component (25 points)
            if price_velocity > 1.0:
                score += 25
            elif price_velocity > 0.5:
                score += 15
            else:
                score += 5
            
            # Trade frequency (15 points)
            if trade_frequency > 2.0:  # 2+ trades per minute
                score += 15
            elif trade_frequency > 1.0:
                score += 10
            else:
                score += 5
            
            return min(score, 100)
            
        except Exception:
            return 50  # Neutral score on error
    
    def _analyze_volume(self, volume_profile: Dict, ticks_df: pd.DataFrame) -> float:
        """Analyze volume patterns and return score 0-100"""
        try:
            total_volume = volume_profile.get('total_volume', 0)
            recent_avg_volume = ticks_df['volume'].mean() if not ticks_df.empty else 0
            large_trades_count = len(ticks_df[ticks_df['volume'] > recent_avg_volume * 2]) if not ticks_df.empty else 0
            
            score = 0
            
            # Volume size (40 points)
            if total_volume > 50000:
                score += 40
            elif total_volume > 20000:
                score += 30
            elif total_volume > 10000:
                score += 20
            else:
                score += 10
            
            # Volume consistency (30 points)
            if large_trades_count > len(ticks_df) * 0.2:  # 20%+ large trades
                score += 30
            elif large_trades_count > len(ticks_df) * 0.1:
                score += 20
            else:
                score += 10
            
            # Volume profile distribution (30 points)
            vwap = volume_profile.get('vwap', 0)
            poc_price = volume_profile.get('poc_price', 0)
            
            if abs(vwap - poc_price) / vwap < 0.02:  # VWAP close to POC
                score += 30
            elif abs(vwap - poc_price) / vwap < 0.05:
                score += 20
            else:
                score += 10
            
            return min(score, 100)
            
        except Exception:
            return 50  # Neutral score on error
    
    def _analyze_liquidity(self, liquidity: Dict) -> float:
        """Analyze market liquidity and return score 0-100"""
        try:
            liquidity_level = liquidity.get('liquidity_level', 'Low')
            avg_time_between_trades = liquidity.get('avg_time_between_trades', 999)
            price_volatility = liquidity.get('price_volatility', 0)
            
            score = 0
            
            # Liquidity level (50 points)
            if liquidity_level == 'High':
                score += 50
            elif liquidity_level == 'Medium':
                score += 30
            else:
                score += 10
            
            # Trade frequency (30 points)
            if avg_time_between_trades < 30:  # Less than 30 seconds between trades
                score += 30
            elif avg_time_between_trades < 60:
                score += 20
            else:
                score += 10
            
            # Price stability (20 points)
            if price_volatility < 1.0:
                score += 20
            elif price_volatility < 2.0:
                score += 15
            else:
                score += 10
            
            return min(score, 100)
            
        except Exception:
            return 50  # Neutral score on error
    
    def _analyze_technical_levels(self, ticks_df: pd.DataFrame, current_price: float) -> float:
        """Analyze technical support/resistance levels"""
        try:
            if ticks_df.empty:
                return 50
            
            high = ticks_df['price'].max()
            low = ticks_df['price'].min()
            price_range = high - low
            
            score = 0
            
            # Position within range (40 points)
            if price_range > 0:
                position_pct = (current_price - low) / price_range
                
                if 0.4 <= position_pct <= 0.6:  # Middle range
                    score += 40
                elif 0.2 <= position_pct <= 0.8:  # Good range
                    score += 30
                else:  # Near extremes
                    score += 20
            else:
                score += 30  # No range, neutral
            
            # Price action pattern (35 points)
            recent_prices = ticks_df.head(10)['price'].tolist()
            if len(recent_prices) >= 3:
                if recent_prices[0] > recent_prices[1] > recent_prices[2]:  # Uptrend
                    score += 35
                elif recent_prices[0] < recent_prices[1] < recent_prices[2]:  # Downtrend
                    score += 35
                else:
                    score += 20  # Sideways
            
            # Volume at levels (25 points)
            high_volume_trades = ticks_df[ticks_df['volume'] > ticks_df['volume'].quantile(0.8)]
            if not high_volume_trades.empty:
                volume_price_avg = high_volume_trades['price'].mean()
                if abs(volume_price_avg - current_price) / current_price < 0.01:  # Within 1%
                    score += 25
                else:
                    score += 15
            
            return min(score, 100)
            
        except Exception:
            return 50
    
    def _determine_signal_type(self, overall_score: float, momentum: Dict, 
                              volume_profile: Dict, liquidity: Dict) -> Tuple[SignalType, float]:
        """Determine signal type and confidence based on analysis"""
        
        momentum_direction = momentum.get('momentum_direction', 'Neutral')
        momentum_strength = momentum.get('momentum_strength', 'Weak')
        price_change_pct = momentum.get('price_change_pct', 0)
        
        # Base confidence from overall score
        confidence = min(overall_score, 95)  # Cap at 95%
        
        # Strong bullish signals
        if (overall_score > 80 and 
            momentum_direction == 'Bullish' and 
            momentum_strength in ['Strong', 'Moderate'] and
            price_change_pct > 0.5):
            return SignalType.STRONG_BUY, confidence
        
        # Bullish signals
        elif (overall_score > 65 and 
              momentum_direction == 'Bullish' and
              price_change_pct > 0.2):
            return SignalType.BUY, confidence
        
        # Strong bearish signals
        elif (overall_score > 80 and 
              momentum_direction == 'Bearish' and 
              momentum_strength in ['Strong', 'Moderate'] and
              price_change_pct < -0.5):
            return SignalType.STRONG_SELL, confidence
        
        # Bearish signals
        elif (overall_score > 65 and 
              momentum_direction == 'Bearish' and
              price_change_pct < -0.2):
            return SignalType.SELL, confidence
        
        # Default to hold
        else:
            return SignalType.HOLD, min(confidence, 60)
    
    def _calculate_levels(self, current_price: float, signal_type: SignalType,
                         momentum: Dict, volume_profile: Dict) -> Tuple[float, float]:
        """Calculate target and stop loss levels"""
        
        support_level = momentum.get('support_level', current_price * 0.99)
        resistance_level = momentum.get('resistance_level', current_price * 1.01)
        price_range = volume_profile.get('price_range', current_price * 0.02)
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            # For buy signals
            target_price = current_price + (price_range * 0.6)  # 60% of recent range
            stop_loss = max(support_level, current_price - (price_range * 0.3))
            
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            # For sell signals
            target_price = current_price - (price_range * 0.6)
            stop_loss = min(resistance_level, current_price + (price_range * 0.3))
            
        else:
            # Hold signal
            target_price = current_price
            stop_loss = current_price
        
        return target_price, stop_loss
    
    def _generate_reasoning(self, momentum: Dict, volume_profile: Dict, 
                           liquidity: Dict, momentum_score: float, volume_score: float, ml_adjustment: float) -> List[str]:
        """Generate human-readable reasoning for the signal"""
        reasoning = []
        
        # ML reasoning
        if abs(ml_adjustment) > 5:
            adj_dir = "positive" if ml_adjustment > 0 else "negative"
            reasoning.append(f"ML daily forecast provided a {adj_dir} bias ({ml_adjustment:.1f} pts)")

        # Momentum reasoning
        if momentum_score > 75:
            reasoning.append(f"Strong price momentum: {momentum['momentum_direction']}")
            if momentum.get('price_velocity', 0) > 1:
                reasoning.append(f"High price velocity: {momentum['price_velocity']:.2f} PKR/min")
        
        # Volume reasoning
        if volume_score > 70:
            reasoning.append(f"Strong volume support: {volume_profile['total_volume']:,} shares")
            if volume_profile.get('total_trades', 0) > 20:
                reasoning.append(f"High trading activity: {volume_profile['total_trades']} trades")
        
        # Liquidity reasoning
        if liquidity.get('liquidity_level') == 'High':
            reasoning.append("High liquidity environment")
        elif liquidity.get('avg_time_between_trades', 999) < 60:
            reasoning.append("Frequent trading activity")
        
        # Technical reasoning
        poc_price = volume_profile.get('poc_price', 0)
        vwap = volume_profile.get('vwap', 0)
        if poc_price and vwap and abs(poc_price - vwap) / vwap < 0.01:
            reasoning.append("Price near high-volume area (POC)")
        
        if not reasoning:
            reasoning.append("Mixed signals - holding recommended")
        
        return reasoning
    
    def _determine_holding_period(self, signal_type: SignalType, momentum: Dict, liquidity: Dict) -> int:
        """Determine recommended holding period in minutes"""
        
        base_period = 15  # Base 15 minutes
        
        # Adjust based on signal strength
        if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            base_period = 30  # Stronger signals held longer
        elif signal_type == SignalType.HOLD:
            return 5  # Short hold periods
        
        # Adjust based on momentum strength
        momentum_strength = momentum.get('momentum_strength', 'Weak')
        if momentum_strength == 'Strong':
            base_period += 15
        elif momentum_strength == 'Moderate':
            base_period += 5
        
        # Adjust based on liquidity
        liquidity_level = liquidity.get('liquidity_level', 'Low')
        if liquidity_level == 'High':
            base_period -= 5  # Can exit faster in liquid markets
        elif liquidity_level == 'Low':
            base_period += 10  # Hold longer in illiquid markets
        
        return max(base_period, 5)  # Minimum 5 minutes
    
    def _create_hold_signal(self, symbol: str, reason: str) -> IntradaySignal:
        """Create a HOLD signal with given reason"""
        
        # Try to get current price
        current_price = 0.0
        try:
            real_time_data = self.fetcher.fetch_real_time_data(symbol)
            if real_time_data:
                current_price = real_time_data['price']
        except:
            pass
        
        return IntradaySignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            confidence=30.0,
            entry_price=current_price,
            target_price=current_price,
            stop_loss=current_price,
            risk_reward_ratio=0.0,
            volume_support=False,
            momentum_direction='Neutral',
            liquidity_level='Unknown',
            analysis_time=dt.datetime.now(),
            holding_period_minutes=5,
            reasoning=[reason]
        )
    
    def analyze_multiple_symbols(self, symbols: List[str], 
                                analysis_period_minutes: int = 30) -> List[IntradaySignal]:
        """Analyze multiple symbols and return sorted by signal strength"""
        
        signals = []
        for symbol in symbols:
            try:
                signal = self.analyze_symbol(symbol, analysis_period_minutes)
                signals.append(signal)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                continue
        
        # Sort by confidence (strongest signals first)
        signals.sort(key=lambda s: (s.signal_type != SignalType.HOLD, s.confidence), reverse=True)
        
        return signals
    
    def get_live_alerts(self, symbols: List[str], min_confidence: float = 70.0) -> List[IntradaySignal]:
        """Get live trading alerts for symbols above confidence threshold"""
        
        signals = self.analyze_multiple_symbols(symbols)
        
        # Filter for actionable signals above confidence threshold
        alerts = [
            signal for signal in signals 
            if signal.signal_type != SignalType.HOLD and signal.confidence >= min_confidence
        ]
        
        return alerts

# Test function
def test_intraday_analyzer():
    """Test the intraday signal analyzer"""
    print("🚀 Testing Intraday Signal Analyzer")
    print("=" * 50)
    
    analyzer = IntradaySignalAnalyzer()
    
    # Test symbols
    test_symbols = ['FFC', 'UBL', 'LUCK']
    
    print(f"📊 Testing {len(test_symbols)} symbols...")
    
    for symbol in test_symbols:
        try:
            print(f"\n🔍 Analyzing {symbol}...")
            signal = analyzer.analyze_symbol(symbol, analysis_period_minutes=20)
            
            print(f"   Signal: {signal.signal_type.value}")
            print(f"   Confidence: {signal.confidence:.1f}%")
            print(f"   Entry: {signal.entry_price:.2f} PKR")
            print(f"   Target: {signal.target_price:.2f} PKR")
            print(f"   Stop Loss: {signal.stop_loss:.2f} PKR")
            print(f"   R/R Ratio: {signal.risk_reward_ratio:.2f}")
            print(f"   Holding Period: {signal.holding_period_minutes} minutes")
            print(f"   Reasoning: {'; '.join(signal.reasoning[:2])}")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {symbol}: {e}")
    
    print(f"\n🎯 Testing live alerts...")
    try:
        alerts = analyzer.get_live_alerts(test_symbols, min_confidence=60.0)
        
        if alerts:
            print(f"   Found {len(alerts)} trading alerts:")
            for alert in alerts:
                print(f"   📢 {alert.symbol}: {alert.signal_type.value} "
                      f"({alert.confidence:.1f}% confidence)")
        else:
            print("   No trading alerts at this time")
            
    except Exception as e:
        print(f"   ❌ Error getting alerts: {e}")
    
    print(f"\n🏁 Intraday analyzer test completed!")

if __name__ == "__main__":
    test_intraday_analyzer()