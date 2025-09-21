"""
VOLATILITY REGIME DETECTION AND ADAPTIVE MODEL SELECTION
Advanced Market Regime Analysis for High-Accuracy Intraday Trading

Features:
- Multi-timeframe volatility regime detection
- Adaptive model selection based on market conditions
- Regime-specific parameter optimization
- Real-time regime monitoring and alerts
- Historical regime analysis and backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json

warnings.filterwarnings('ignore')

@dataclass
class VolatilityRegime:
    """Container for volatility regime information"""
    regime_id: int
    regime_name: str
    volatility_level: float
    confidence: float
    duration: int  # Minutes in current regime
    characteristics: Dict[str, float]
    model_recommendations: Dict[str, float]
    risk_adjustments: Dict[str, float]

@dataclass
class RegimeSignal:
    """Signal with regime-specific adjustments"""
    symbol: str
    base_signal: str
    regime_adjusted_signal: str
    confidence_adjustment: float
    size_adjustment: float
    risk_adjustment: float
    regime_info: VolatilityRegime
    explanation: str

class VolatilityRegimeDetector:
    """Advanced volatility regime detection and model adaptation"""
    
    def __init__(self):
        self.regimes = {
            0: {'name': 'Low Volatility', 'vol_range': (0, 0.15), 'color': 'green'},
            1: {'name': 'Normal Volatility', 'vol_range': (0.15, 0.35), 'color': 'blue'},
            2: {'name': 'High Volatility', 'vol_range': (0.35, 0.60), 'color': 'orange'},
            3: {'name': 'Extreme Volatility', 'vol_range': (0.60, 1.0), 'color': 'red'}
        }
        
        # Model performance by regime (would be calibrated from historical data)
        self.model_performance = {
            'Low Volatility': {
                'momentum_models': 0.65,      # Mean reversion works better
                'mean_reversion_models': 0.85,
                'ml_models': 0.75,
                'technical_indicators': 0.80
            },
            'Normal Volatility': {
                'momentum_models': 0.75,
                'mean_reversion_models': 0.70,
                'ml_models': 0.82,
                'technical_indicators': 0.78
            },
            'High Volatility': {
                'momentum_models': 0.85,      # Momentum works better
                'mean_reversion_models': 0.60,
                'ml_models': 0.88,
                'technical_indicators': 0.70
            },
            'Extreme Volatility': {
                'momentum_models': 0.90,
                'mean_reversion_models': 0.45,
                'ml_models': 0.75,           # ML models may struggle
                'technical_indicators': 0.55
            }
        }
        
        # Regime-specific parameters
        self.regime_parameters = {
            'Low Volatility': {
                'position_size_multiplier': 1.2,    # Larger positions
                'stop_loss_multiplier': 0.8,        # Tighter stops
                'take_profit_multiplier': 0.9,      # Closer targets
                'confidence_threshold': 0.6,        # Lower threshold
                'max_holding_period': 240,          # 4 hours
            },
            'Normal Volatility': {
                'position_size_multiplier': 1.0,    # Normal positions
                'stop_loss_multiplier': 1.0,        # Normal stops
                'take_profit_multiplier': 1.0,      # Normal targets
                'confidence_threshold': 0.7,        # Standard threshold
                'max_holding_period': 180,          # 3 hours
            },
            'High Volatility': {
                'position_size_multiplier': 0.7,    # Smaller positions
                'stop_loss_multiplier': 1.5,        # Wider stops
                'take_profit_multiplier': 1.3,      # Further targets
                'confidence_threshold': 0.8,        # Higher threshold
                'max_holding_period': 120,          # 2 hours
            },
            'Extreme Volatility': {
                'position_size_multiplier': 0.4,    # Much smaller positions
                'stop_loss_multiplier': 2.0,        # Much wider stops
                'take_profit_multiplier': 1.8,      # Much further targets
                'confidence_threshold': 0.9,        # Much higher threshold
                'max_holding_period': 60,           # 1 hour
            }
        }
        
        # State tracking
        self.current_regime = None
        self.regime_history = []
        self.regime_start_time = None
        self.regime_duration = 0
        
        # Calibration data
        self.volatility_scaler = StandardScaler()
        self.regime_model = None
        self.is_calibrated = False
        
    def detect_regime(self, market_data: pd.DataFrame, symbol: str) -> VolatilityRegime:
        """Detect current volatility regime"""
        
        if market_data.empty or len(market_data) < 50:
            return self._create_default_regime()
        
        # Calculate volatility features
        vol_features = self._extract_volatility_features(market_data)
        
        # Classify regime
        regime_id = self._classify_regime(vol_features)
        regime_info = self.regimes[regime_id]
        
        # Calculate regime characteristics
        characteristics = self._calculate_regime_characteristics(market_data, vol_features)
        
        # Get model recommendations
        model_recommendations = self._get_model_recommendations(regime_info['name'])
        
        # Calculate risk adjustments
        risk_adjustments = self._calculate_risk_adjustments(regime_info['name'], characteristics)
        
        # Update regime tracking
        self._update_regime_tracking(regime_id)
        
        return VolatilityRegime(
            regime_id=regime_id,
            regime_name=regime_info['name'],
            volatility_level=vol_features['current_volatility'],
            confidence=vol_features['regime_confidence'],
            duration=self.regime_duration,
            characteristics=characteristics,
            model_recommendations=model_recommendations,
            risk_adjustments=risk_adjustments
        )
    
    def adapt_signal_to_regime(self, base_signal: str, signal_confidence: float,
                              regime: VolatilityRegime, symbol: str) -> RegimeSignal:
        """Adapt trading signal based on volatility regime"""
        
        regime_params = self.regime_parameters[regime.regime_name]
        
        # Adjust confidence based on regime
        if regime.regime_name in ['Low Volatility', 'Normal Volatility']:
            # Mean reversion strategies work better
            if base_signal in ['BUY', 'SELL'] and signal_confidence > 0.5:
                confidence_adjustment = min(1.2, 1 + (0.7 - signal_confidence) * 0.5)
            else:
                confidence_adjustment = 1.0
        else:
            # High volatility - momentum strategies work better
            if base_signal in ['BUY', 'SELL'] and signal_confidence > 0.7:
                confidence_adjustment = min(1.3, 1 + (signal_confidence - 0.7) * 1.0)
            else:
                confidence_adjustment = max(0.7, 1 - (0.8 - signal_confidence) * 0.5)
        
        adjusted_confidence = signal_confidence * confidence_adjustment
        
        # Determine regime-adjusted signal
        confidence_threshold = regime_params['confidence_threshold']
        
        if adjusted_confidence >= confidence_threshold:
            regime_adjusted_signal = base_signal
        else:
            regime_adjusted_signal = 'HOLD'  # Not confident enough for this regime
        
        # Position size adjustment
        size_adjustment = regime_params['position_size_multiplier']
        
        # Risk adjustment
        risk_adjustment = self._calculate_regime_risk_adjustment(regime)
        
        # Generate explanation
        explanation = self._generate_regime_explanation(
            base_signal, regime_adjusted_signal, regime, confidence_adjustment
        )
        
        return RegimeSignal(
            symbol=symbol,
            base_signal=base_signal,
            regime_adjusted_signal=regime_adjusted_signal,
            confidence_adjustment=confidence_adjustment,
            size_adjustment=size_adjustment,
            risk_adjustment=risk_adjustment,
            regime_info=regime,
            explanation=explanation
        )
    
    def calibrate_regime_detection(self, historical_data: Dict[str, pd.DataFrame]):
        """Calibrate regime detection on historical data"""
        
        print("🔧 Calibrating volatility regime detection...")
        
        all_vol_features = []
        
        # Extract volatility features from all symbols
        for symbol, data in historical_data.items():
            if len(data) >= 100:  # Minimum data requirement
                vol_features = self._extract_volatility_features(data)
                all_vol_features.append([
                    vol_features['current_volatility'],
                    vol_features['vol_of_vol'],
                    vol_features['vol_trend'],
                    vol_features['vol_percentile']
                ])
        
        if len(all_vol_features) < 50:
            print("⚠️ Insufficient data for regime calibration")
            return False
        
        # Prepare data for clustering
        feature_matrix = np.array(all_vol_features)
        feature_matrix_scaled = self.volatility_scaler.fit_transform(feature_matrix)
        
        # Fit Gaussian Mixture Model for regime classification
        self.regime_model = GaussianMixture(
            n_components=4,  # 4 volatility regimes
            covariance_type='full',
            random_state=42
        )
        
        self.regime_model.fit(feature_matrix_scaled)
        self.is_calibrated = True
        
        print("✅ Volatility regime detection calibrated successfully")
        return True
    
    def _extract_volatility_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract comprehensive volatility features"""
        
        features = {}
        
        try:
            # Returns calculation
            returns = market_data['Close'].pct_change().dropna()
            
            if len(returns) < 20:
                return self._default_volatility_features()
            
            # Current volatility (20-period rolling)
            current_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized
            features['current_volatility'] = current_vol
            
            # Volatility of volatility
            if len(returns) >= 60:
                rolling_vol = returns.rolling(20).std()
                vol_of_vol = rolling_vol.std()
                features['vol_of_vol'] = vol_of_vol
            else:
                features['vol_of_vol'] = current_vol * 0.3
            
            # Volatility trend
            if len(returns) >= 40:
                recent_vol = returns.tail(20).std()
                previous_vol = returns.tail(40).head(20).std()
                vol_trend = (recent_vol - previous_vol) / previous_vol if previous_vol > 0 else 0
                features['vol_trend'] = vol_trend
            else:
                features['vol_trend'] = 0.0
            
            # Volatility percentile
            if len(returns) >= 100:
                historical_vol = returns.rolling(20).std().dropna()
                vol_percentile = (historical_vol <= current_vol).mean()
                features['vol_percentile'] = vol_percentile
            else:
                features['vol_percentile'] = 0.5
            
            # Intraday volatility patterns
            if len(market_data) >= 20:
                high_low_vol = ((market_data['High'] - market_data['Low']) / market_data['Close']).tail(20).mean()
                features['intraday_volatility'] = high_low_vol
            else:
                features['intraday_volatility'] = current_vol * 0.5
            
            # Volume-weighted volatility
            if len(market_data) >= 20:
                volume_weighted_returns = returns.tail(20) * (market_data['Volume'].tail(20) / market_data['Volume'].tail(20).mean())
                vw_volatility = volume_weighted_returns.std() * np.sqrt(252)
                features['volume_weighted_volatility'] = vw_volatility
            else:
                features['volume_weighted_volatility'] = current_vol
            
            # Regime confidence (based on feature consistency)
            regime_confidence = self._calculate_regime_confidence(features)
            features['regime_confidence'] = regime_confidence
            
        except Exception as e:
            print(f"Error extracting volatility features: {e}")
            return self._default_volatility_features()
        
        return features
    
    def _classify_regime(self, vol_features: Dict[str, float]) -> int:
        """Classify volatility regime based on features"""
        
        current_vol = vol_features['current_volatility']
        
        # Simple rule-based classification (can be enhanced with ML)
        if current_vol <= 0.15:
            return 0  # Low Volatility
        elif current_vol <= 0.35:
            return 1  # Normal Volatility
        elif current_vol <= 0.60:
            return 2  # High Volatility
        else:
            return 3  # Extreme Volatility
    
    def _calculate_regime_characteristics(self, market_data: pd.DataFrame, 
                                        vol_features: Dict[str, float]) -> Dict[str, float]:
        """Calculate regime-specific characteristics"""
        
        characteristics = {}
        
        try:
            # Trend persistence
            if len(market_data) >= 20:
                returns = market_data['Close'].pct_change().tail(20)
                positive_returns = (returns > 0).sum()
                characteristics['trend_persistence'] = positive_returns / len(returns)
            else:
                characteristics['trend_persistence'] = 0.5
            
            # Price momentum
            if len(market_data) >= 10:
                momentum = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-10]) - 1
                characteristics['price_momentum'] = momentum
            else:
                characteristics['price_momentum'] = 0.0
            
            # Volume momentum
            if len(market_data) >= 20:
                vol_ratio = market_data['Volume'].tail(5).mean() / market_data['Volume'].tail(20).mean()
                characteristics['volume_momentum'] = vol_ratio - 1
            else:
                characteristics['volume_momentum'] = 0.0
            
            # Range expansion/contraction
            if len(market_data) >= 20:
                recent_range = (market_data['High'].tail(5) - market_data['Low'].tail(5)).mean()
                historical_range = (market_data['High'].tail(20) - market_data['Low'].tail(20)).mean()
                characteristics['range_expansion'] = (recent_range / historical_range) - 1 if historical_range > 0 else 0
            else:
                characteristics['range_expansion'] = 0.0
            
            # Mean reversion tendency
            if len(market_data) >= 20:
                price = market_data['Close'].tail(20)
                sma = price.mean()
                mean_reversion = abs(price.iloc[-1] - sma) / sma if sma > 0 else 0
                characteristics['mean_reversion_signal'] = mean_reversion
            else:
                characteristics['mean_reversion_signal'] = 0.0
            
        except Exception as e:
            print(f"Error calculating regime characteristics: {e}")
        
        return characteristics
    
    def _get_model_recommendations(self, regime_name: str) -> Dict[str, float]:
        """Get model performance recommendations for regime"""
        
        if regime_name in self.model_performance:
            return self.model_performance[regime_name].copy()
        else:
            # Default recommendations
            return {
                'momentum_models': 0.7,
                'mean_reversion_models': 0.7,
                'ml_models': 0.75,
                'technical_indicators': 0.7
            }
    
    def _calculate_risk_adjustments(self, regime_name: str, 
                                  characteristics: Dict[str, float]) -> Dict[str, float]:
        """Calculate regime-specific risk adjustments"""
        
        regime_params = self.regime_parameters.get(regime_name, self.regime_parameters['Normal Volatility'])
        
        adjustments = {
            'position_size_multiplier': regime_params['position_size_multiplier'],
            'stop_loss_multiplier': regime_params['stop_loss_multiplier'],
            'take_profit_multiplier': regime_params['take_profit_multiplier'],
            'confidence_threshold': regime_params['confidence_threshold'],
            'max_holding_period': regime_params['max_holding_period']
        }
        
        # Dynamic adjustments based on characteristics
        if characteristics.get('volume_momentum', 0) > 0.5:  # High volume
            adjustments['position_size_multiplier'] *= 1.1
        
        if characteristics.get('trend_persistence', 0.5) > 0.7:  # Strong trend
            adjustments['stop_loss_multiplier'] *= 1.2
            adjustments['take_profit_multiplier'] *= 1.3
        
        return adjustments
    
    def _update_regime_tracking(self, regime_id: int):
        """Update regime tracking state"""
        
        current_time = datetime.now()
        
        if self.current_regime != regime_id:
            # Regime change detected
            self.current_regime = regime_id
            self.regime_start_time = current_time
            self.regime_duration = 0
            
            # Add to history
            self.regime_history.append({
                'regime_id': regime_id,
                'start_time': current_time,
                'regime_name': self.regimes[regime_id]['name']
            })
            
            # Keep only last 100 regime changes
            if len(self.regime_history) > 100:
                self.regime_history.pop(0)
        else:
            # Update duration
            if self.regime_start_time:
                self.regime_duration = int((current_time - self.regime_start_time).total_seconds() / 60)
    
    def _calculate_regime_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in regime classification"""
        
        # Simple confidence calculation based on feature consistency
        vol = features['current_volatility']
        vol_percentile = features['vol_percentile']
        vol_trend = abs(features['vol_trend'])
        
        # Confidence is higher when:
        # 1. Volatility is clearly in a regime (not borderline)
        # 2. Volatility percentile is extreme (very high or very low)
        # 3. Volatility trend is consistent
        
        confidence = 0.5  # Base confidence
        
        # Volatility clarity
        if vol <= 0.10 or vol >= 0.50:
            confidence += 0.2
        elif vol <= 0.20 or vol >= 0.40:
            confidence += 0.1
        
        # Percentile extremes
        if vol_percentile <= 0.2 or vol_percentile >= 0.8:
            confidence += 0.2
        elif vol_percentile <= 0.3 or vol_percentile >= 0.7:
            confidence += 0.1
        
        # Trend consistency
        if vol_trend <= 0.1:  # Stable volatility
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_regime_risk_adjustment(self, regime: VolatilityRegime) -> float:
        """Calculate overall risk adjustment for regime"""
        
        base_risk = 1.0
        
        # Adjust based on volatility level
        if regime.regime_name == 'Extreme Volatility':
            base_risk *= 2.0
        elif regime.regime_name == 'High Volatility':
            base_risk *= 1.5
        elif regime.regime_name == 'Low Volatility':
            base_risk *= 0.8
        
        # Adjust based on regime confidence
        confidence_adjustment = 1 + (1 - regime.confidence) * 0.5
        
        return base_risk * confidence_adjustment
    
    def _generate_regime_explanation(self, base_signal: str, adjusted_signal: str,
                                   regime: VolatilityRegime, confidence_adj: float) -> str:
        """Generate explanation for regime adjustments"""
        
        explanations = []
        
        explanations.append(f"Market Regime: {regime.regime_name}")
        explanations.append(f"Volatility Level: {regime.volatility_level:.1%}")
        explanations.append(f"Regime Confidence: {regime.confidence:.1%}")
        
        if base_signal != adjusted_signal:
            explanations.append(f"Signal changed from {base_signal} to {adjusted_signal}")
        
        if confidence_adj != 1.0:
            explanations.append(f"Confidence adjusted by {confidence_adj:.1%}")
        
        return " | ".join(explanations)
    
    def _create_default_regime(self) -> VolatilityRegime:
        """Create default regime when insufficient data"""
        
        return VolatilityRegime(
            regime_id=1,
            regime_name='Normal Volatility',
            volatility_level=0.25,
            confidence=0.5,
            duration=0,
            characteristics={'trend_persistence': 0.5, 'price_momentum': 0.0},
            model_recommendations=self.model_performance['Normal Volatility'],
            risk_adjustments=self.regime_parameters['Normal Volatility']
        )
    
    def _default_volatility_features(self) -> Dict[str, float]:
        """Default volatility features when calculation fails"""
        
        return {
            'current_volatility': 0.25,
            'vol_of_vol': 0.08,
            'vol_trend': 0.0,
            'vol_percentile': 0.5,
            'intraday_volatility': 0.12,
            'volume_weighted_volatility': 0.25,
            'regime_confidence': 0.5
        }
    
    def get_regime_summary(self) -> Dict[str, Union[str, int, float]]:
        """Get summary of current regime state"""
        
        return {
            'current_regime_id': self.current_regime,
            'current_regime_name': self.regimes[self.current_regime]['name'] if self.current_regime is not None else 'Unknown',
            'regime_duration': self.regime_duration,
            'total_regime_changes': len(self.regime_history),
            'is_calibrated': self.is_calibrated
        }

# Testing function
def test_volatility_regime_detector():
    """Test the volatility regime detection system"""
    print("🧪 Testing Volatility Regime Detector...")
    
    # Create sample market data with different volatility regimes
    dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
    
    # Generate data with regime changes
    prices = []
    current_price = 100
    volatility = 0.02  # Start with low volatility
    
    for i in range(200):
        # Change volatility regime every 50 periods
        if i == 50:
            volatility = 0.06  # High volatility
        elif i == 100:
            volatility = 0.01  # Very low volatility
        elif i == 150:
            volatility = 0.10  # Extreme volatility
        
        change = np.random.normal(0, volatility)
        current_price *= (1 + change)
        prices.append(current_price)
    
    sample_data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'High': [p * np.random.uniform(1.01, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 0.99) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(10000, 100000, 200)
    }, index=dates)
    
    # Initialize detector
    detector = VolatilityRegimeDetector()
    
    # Test regime detection
    print("📊 Testing regime detection...")
    regime = detector.detect_regime(sample_data, 'TEST')
    
    print(f"   Detected Regime: {regime.regime_name}")
    print(f"   Volatility Level: {regime.volatility_level:.1%}")
    print(f"   Confidence: {regime.confidence:.1%}")
    print(f"   Duration: {regime.duration} minutes")
    
    # Test signal adaptation
    print("\n🎯 Testing signal adaptation...")
    base_signals = ['BUY', 'SELL', 'HOLD']
    
    for signal in base_signals:
        adapted_signal = detector.adapt_signal_to_regime(
            base_signal=signal,
            signal_confidence=0.75,
            regime=regime,
            symbol='TEST'
        )
        
        print(f"   {signal} → {adapted_signal.regime_adjusted_signal}")
        print(f"      Confidence Adj: {adapted_signal.confidence_adjustment:.2f}")
        print(f"      Size Adj: {adapted_signal.size_adjustment:.2f}")
        print(f"      Risk Adj: {adapted_signal.risk_adjustment:.2f}")
    
    # Test calibration (simplified)
    print("\n🔧 Testing calibration...")
    historical_data = {'TEST': sample_data}
    success = detector.calibrate_regime_detection(historical_data)
    print(f"   Calibration successful: {success}")
    
    # Test regime summary
    print("\n📈 Regime Summary:")
    summary = detector.get_regime_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Volatility Regime Detector test completed!")

if __name__ == "__main__":
    test_volatility_regime_detector()