"""
ENHANCED INTRADAY FEATURE ENGINEERING ENGINE
High-Accuracy Multi-Timeframe Analysis for Intraday Trading

Features:
- Multi-timeframe analysis (1m, 5m, 15m, 1h)
- Microstructure features (bid-ask, order flow)
- Time-of-day pattern detection
- Volatility regime detection
- PSX-specific adaptations
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio

warnings.filterwarnings('ignore')

@dataclass
class IntradayFeatures:
    """Container for comprehensive intraday features"""
    symbol: str
    timestamp: datetime
    timeframe: str
    
    # Price-based features
    price_features: Dict[str, float]
    
    # Technical indicators across timeframes
    technical_features: Dict[str, float]
    
    # Microstructure features
    microstructure_features: Dict[str, float]
    
    # Time-based features
    temporal_features: Dict[str, float]
    
    # Volatility regime features
    volatility_features: Dict[str, float]
    
    # Market session features
    session_features: Dict[str, float]
    
    # PSX-specific features
    psx_features: Dict[str, float]

class EnhancedIntradayFeatureEngine:
    """Advanced feature engineering for high-accuracy intraday trading"""
    
    def __init__(self):
        self.timeframes = ['1min', '5min', '15min', '1H']
        self.psx_market_hours = {
            'open': time(9, 15),
            'close': time(15, 30),
            'lunch_start': time(12, 30),
            'lunch_end': time(13, 30)
        }
        
        # PSX-specific parameters
        self.psx_params = {
            'min_volume_threshold': 1000,
            'liquidity_tiers': {
                'high': 100000,
                'medium': 50000,
                'low': 10000
            },
            'tick_sizes': {
                'below_5': 0.01,
                'below_25': 0.05,
                'below_100': 0.10,
                'above_100': 0.25
            }
        }
        
        # Cache for multi-timeframe data
        self.data_cache = {}
        
    def extract_comprehensive_features(self, symbol: str, data_1m: pd.DataFrame) -> IntradayFeatures:
        """Extract comprehensive intraday features from 1-minute data"""
        
        # Generate multi-timeframe data
        multi_tf_data = self.create_multi_timeframe_data(data_1m)
        
        # Extract features from each category
        price_features = self.extract_price_features(data_1m, multi_tf_data)
        technical_features = self.extract_multi_timeframe_technical(multi_tf_data)
        microstructure_features = self.extract_microstructure_features(data_1m)
        temporal_features = self.extract_temporal_features(data_1m)
        volatility_features = self.extract_volatility_regime_features(data_1m, multi_tf_data)
        session_features = self.extract_market_session_features(data_1m)
        psx_features = self.extract_psx_specific_features(data_1m, symbol)
        
        return IntradayFeatures(
            symbol=symbol,
            timestamp=data_1m.index[-1] if not data_1m.empty else datetime.now(),
            timeframe='1min',
            price_features=price_features,
            technical_features=technical_features,
            microstructure_features=microstructure_features,
            temporal_features=temporal_features,
            volatility_features=volatility_features,
            session_features=session_features,
            psx_features=psx_features
        )
    
    def create_multi_timeframe_data(self, data_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create multi-timeframe datasets from 1-minute data"""
        if data_1m.empty:
            return {tf: pd.DataFrame() for tf in self.timeframes}
        
        multi_tf_data = {'1min': data_1m.copy()}
        
        # Create higher timeframe data
        for tf in ['5min', '15min', '1H']:
            try:
                resampled = data_1m.resample(tf).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                multi_tf_data[tf] = resampled
            except Exception as e:
                print(f"Error creating {tf} data: {e}")
                multi_tf_data[tf] = pd.DataFrame()
        
        return multi_tf_data
    
    def extract_price_features(self, data_1m: pd.DataFrame, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract advanced price-based features"""
        if data_1m.empty:
            return {}
        
        features = {}
        
        try:
            # Current price metrics
            current_price = data_1m['Close'].iloc[-1]
            features['current_price'] = current_price
            
            # Multi-timeframe returns
            for tf, data in multi_tf_data.items():
                if not data.empty and len(data) > 1:
                    returns = data['Close'].pct_change()
                    features[f'return_{tf}'] = returns.iloc[-1] if not pd.isna(returns.iloc[-1]) else 0.0
                    features[f'return_std_{tf}'] = returns.std() if len(returns) > 1 else 0.0
                    features[f'return_skew_{tf}'] = returns.skew() if len(returns) > 2 else 0.0
            
            # Price momentum across timeframes
            for periods in [5, 10, 20, 60]:
                if len(data_1m) > periods:
                    momentum = (current_price / data_1m['Close'].iloc[-periods-1]) - 1
                    features[f'momentum_{periods}min'] = momentum
            
            # Price positioning
            if len(data_1m) >= 20:
                high_20 = data_1m['High'].tail(20).max()
                low_20 = data_1m['Low'].tail(20).min()
                if high_20 != low_20:
                    features['price_position_20min'] = (current_price - low_20) / (high_20 - low_20)
                else:
                    features['price_position_20min'] = 0.5
            
            # Gap analysis
            if len(data_1m) > 1:
                prev_close = data_1m['Close'].iloc[-2]
                gap = (current_price - prev_close) / prev_close
                features['price_gap'] = gap
                features['gap_magnitude'] = abs(gap)
            
        except Exception as e:
            print(f"Error extracting price features: {e}")
        
        return features
    
    def extract_multi_timeframe_technical(self, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract technical indicators across multiple timeframes"""
        features = {}
        
        for tf, data in multi_tf_data.items():
            if data.empty or len(data) < 20:
                continue
                
            try:
                # RSI across timeframes
                rsi = ta.momentum.rsi(data['Close'], window=14)
                if not rsi.empty:
                    features[f'rsi_{tf}'] = rsi.iloc[-1]
                    features[f'rsi_slope_{tf}'] = rsi.iloc[-1] - rsi.iloc[-2] if len(rsi) > 1 else 0
                
                # MACD across timeframes
                macd = ta.trend.macd(data['Close'])
                macd_signal = ta.trend.macd_signal(data['Close'])
                if not macd.empty and not macd_signal.empty:
                    features[f'macd_{tf}'] = macd.iloc[-1]
                    features[f'macd_signal_{tf}'] = macd_signal.iloc[-1]
                    features[f'macd_histogram_{tf}'] = macd.iloc[-1] - macd_signal.iloc[-1]
                
                # Bollinger Bands
                bb_upper = ta.volatility.bollinger_hband(data['Close'])
                bb_lower = ta.volatility.bollinger_lband(data['Close'])
                bb_middle = ta.volatility.bollinger_mavg(data['Close'])
                
                if not bb_upper.empty and not bb_lower.empty:
                    current_price = data['Close'].iloc[-1]
                    features[f'bb_position_{tf}'] = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                    features[f'bb_width_{tf}'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                
                # Moving averages and alignment
                sma_5 = ta.trend.sma_indicator(data['Close'], window=5)
                sma_20 = ta.trend.sma_indicator(data['Close'], window=20)
                ema_9 = ta.trend.ema_indicator(data['Close'], window=9)
                
                if not sma_5.empty and not sma_20.empty:
                    features[f'sma_alignment_{tf}'] = 1 if sma_5.iloc[-1] > sma_20.iloc[-1] else 0
                    features[f'price_vs_sma20_{tf}'] = (current_price / sma_20.iloc[-1]) - 1
                
                # ADX for trend strength
                adx = ta.trend.adx(data['High'], data['Low'], data['Close'])
                if not adx.empty:
                    features[f'adx_{tf}'] = adx.iloc[-1]
                
                # Stochastic
                stoch = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
                if not stoch.empty:
                    features[f'stoch_{tf}'] = stoch.iloc[-1]
                
            except Exception as e:
                print(f"Error extracting technical features for {tf}: {e}")
                continue
        
        return features
    
    def extract_microstructure_features(self, data_1m: pd.DataFrame) -> Dict[str, float]:
        """Extract market microstructure features"""
        features = {}
        
        if data_1m.empty or len(data_1m) < 10:
            return features
        
        try:
            # Bid-Ask Spread Proxy (using high-low)
            recent_data = data_1m.tail(10)
            spread_proxy = (recent_data['High'] - recent_data['Low']) / recent_data['Close']
            features['avg_spread_proxy'] = spread_proxy.mean()
            features['spread_volatility'] = spread_proxy.std()
            
            # Volume-Price Analysis
            features['volume_weighted_price'] = (recent_data['Close'] * recent_data['Volume']).sum() / recent_data['Volume'].sum()
            
            # Price Impact (volume vs price change relationship)
            if len(recent_data) >= 2:
                price_changes = recent_data['Close'].pct_change().dropna()
                volume_changes = recent_data['Volume'].pct_change().dropna()
                
                if len(price_changes) > 0 and len(volume_changes) > 0:
                    # Correlation between volume and absolute price changes
                    features['volume_price_correlation'] = np.corrcoef(
                        volume_changes, abs(price_changes)
                    )[0, 1] if len(price_changes) == len(volume_changes) else 0
            
            # Order Flow Approximation
            # Uptick vs downtick volume
            uptick_volume = 0
            downtick_volume = 0
            
            for i in range(1, len(recent_data)):
                if recent_data['Close'].iloc[i] > recent_data['Close'].iloc[i-1]:
                    uptick_volume += recent_data['Volume'].iloc[i]
                elif recent_data['Close'].iloc[i] < recent_data['Close'].iloc[i-1]:
                    downtick_volume += recent_data['Volume'].iloc[i]
            
            total_volume = uptick_volume + downtick_volume
            if total_volume > 0:
                features['order_flow_ratio'] = (uptick_volume - downtick_volume) / total_volume
            else:
                features['order_flow_ratio'] = 0
            
            # Volume surge detection
            avg_volume = data_1m['Volume'].tail(20).mean()
            current_volume = data_1m['Volume'].iloc[-1]
            features['volume_surge'] = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Liquidity measures
            features['volume_volatility'] = data_1m['Volume'].tail(20).std() / avg_volume if avg_volume > 0 else 0
            
        except Exception as e:
            print(f"Error extracting microstructure features: {e}")
        
        return features
    
    def extract_temporal_features(self, data_1m: pd.DataFrame) -> Dict[str, float]:
        """Extract time-of-day and calendar-based features"""
        features = {}
        
        if data_1m.empty:
            return features
        
        try:
            current_time = data_1m.index[-1]
            
            # Time of day features
            hour = current_time.hour
            minute = current_time.minute
            
            features['hour'] = hour
            features['minute'] = minute
            features['time_decimal'] = hour + minute/60
            
            # Market session indicators
            current_time_only = current_time.time()
            
            # Opening session (first 30 minutes)
            features['is_opening_session'] = 1 if time(9, 15) <= current_time_only <= time(9, 45) else 0
            
            # Lunch time
            features['is_lunch_time'] = 1 if time(12, 30) <= current_time_only <= time(13, 30) else 0
            
            # Closing session (last 30 minutes)
            features['is_closing_session'] = 1 if time(15, 0) <= current_time_only <= time(15, 30) else 0
            
            # Mid-day session
            features['is_mid_day'] = 1 if time(10, 30) <= current_time_only <= time(12, 30) else 0
            
            # Day of week
            features['day_of_week'] = current_time.weekday()
            features['is_monday'] = 1 if current_time.weekday() == 0 else 0
            features['is_friday'] = 1 if current_time.weekday() == 4 else 0
            
            # Minutes since market open
            market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
            if current_time >= market_open:
                minutes_since_open = (current_time - market_open).total_seconds() / 60
                features['minutes_since_open'] = minutes_since_open
            else:
                features['minutes_since_open'] = 0
            
            # Minutes until market close
            market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
            if current_time <= market_close:
                minutes_until_close = (market_close - current_time).total_seconds() / 60
                features['minutes_until_close'] = minutes_until_close
            else:
                features['minutes_until_close'] = 0
            
        except Exception as e:
            print(f"Error extracting temporal features: {e}")
        
        return features
    
    def extract_volatility_regime_features(self, data_1m: pd.DataFrame, multi_tf_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract volatility regime and risk features"""
        features = {}
        
        if data_1m.empty:
            return features
        
        try:
            # Realized volatility across timeframes
            for tf, data in multi_tf_data.items():
                if not data.empty and len(data) >= 20:
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) > 1:
                        features[f'realized_vol_{tf}'] = returns.std() * np.sqrt(252)  # Annualized
                        
                        # Volatility of volatility
                        rolling_vol = returns.rolling(window=10).std()
                        if len(rolling_vol.dropna()) > 1:
                            features[f'vol_of_vol_{tf}'] = rolling_vol.std()
            
            # ATR-based volatility
            if len(data_1m) >= 14:
                atr = ta.volatility.average_true_range(data_1m['High'], data_1m['Low'], data_1m['Close'], window=14)
                if not atr.empty:
                    features['atr_normalized'] = atr.iloc[-1] / data_1m['Close'].iloc[-1]
            
            # Volatility regime detection
            if len(data_1m) >= 60:
                recent_vol = data_1m['Close'].pct_change().tail(20).std()
                historical_vol = data_1m['Close'].pct_change().tail(60).std()
                
                if historical_vol > 0:
                    vol_regime = recent_vol / historical_vol
                    features['volatility_regime'] = vol_regime
                    features['high_vol_regime'] = 1 if vol_regime > 1.5 else 0
                    features['low_vol_regime'] = 1 if vol_regime < 0.7 else 0
            
            # Price range analysis
            if len(data_1m) >= 20:
                recent_data = data_1m.tail(20)
                price_range = (recent_data['High'].max() - recent_data['Low'].min()) / recent_data['Close'].mean()
                features['price_range_normalized'] = price_range
            
        except Exception as e:
            print(f"Error extracting volatility features: {e}")
        
        return features
    
    def extract_market_session_features(self, data_1m: pd.DataFrame) -> Dict[str, float]:
        """Extract market session-specific features"""
        features = {}
        
        if data_1m.empty:
            return features
        
        try:
            # Session volume analysis
            current_time = data_1m.index[-1].time()
            
            # Calculate session-specific metrics
            if len(data_1m) >= 60:  # At least 1 hour of data
                
                # Opening session statistics (9:15-10:15)
                opening_data = data_1m.between_time('09:15', '10:15')
                if not opening_data.empty:
                    features['opening_volume_avg'] = opening_data['Volume'].mean()
                    features['opening_volatility'] = opening_data['Close'].pct_change().std()
                    features['opening_return'] = (opening_data['Close'].iloc[-1] / opening_data['Close'].iloc[0]) - 1 if len(opening_data) > 1 else 0
                
                # Mid-day session (10:15-14:00)
                midday_data = data_1m.between_time('10:15', '14:00')
                if not midday_data.empty:
                    features['midday_volume_avg'] = midday_data['Volume'].mean()
                    features['midday_volatility'] = midday_data['Close'].pct_change().std()
                
                # Closing session (14:00-15:30)
                closing_data = data_1m.between_time('14:00', '15:30')
                if not closing_data.empty:
                    features['closing_volume_avg'] = closing_data['Volume'].mean()
                    features['closing_volatility'] = closing_data['Close'].pct_change().std()
            
            # Current session performance
            today_data = data_1m[data_1m.index.date == data_1m.index[-1].date()]
            if not today_data.empty and len(today_data) > 1:
                features['session_return'] = (today_data['Close'].iloc[-1] / today_data['Close'].iloc[0]) - 1
                features['session_high'] = today_data['High'].max()
                features['session_low'] = today_data['Low'].min()
                features['session_volume'] = today_data['Volume'].sum()
                
                # Position within session range
                session_range = features['session_high'] - features['session_low']
                if session_range > 0:
                    features['session_position'] = (today_data['Close'].iloc[-1] - features['session_low']) / session_range
            
        except Exception as e:
            print(f"Error extracting session features: {e}")
        
        return features
    
    def extract_psx_specific_features(self, data_1m: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Extract PSX market-specific features"""
        features = {}
        
        if data_1m.empty:
            return features
        
        try:
            current_price = data_1m['Close'].iloc[-1]
            
            # Tick size analysis
            if current_price < 5:
                features['tick_size'] = 0.01
            elif current_price < 25:
                features['tick_size'] = 0.05
            elif current_price < 100:
                features['tick_size'] = 0.10
            else:
                features['tick_size'] = 0.25
            
            features['tick_size_normalized'] = features['tick_size'] / current_price
            
            # Liquidity tier classification
            avg_volume = data_1m['Volume'].tail(20).mean()
            if avg_volume >= self.psx_params['liquidity_tiers']['high']:
                features['liquidity_tier'] = 3  # High
            elif avg_volume >= self.psx_params['liquidity_tiers']['medium']:
                features['liquidity_tier'] = 2  # Medium
            elif avg_volume >= self.psx_params['liquidity_tiers']['low']:
                features['liquidity_tier'] = 1  # Low
            else:
                features['liquidity_tier'] = 0  # Very low
            
            # Volume adequacy
            features['volume_adequate'] = 1 if avg_volume >= self.psx_params['min_volume_threshold'] else 0
            
            # Price movement granularity
            if len(data_1m) >= 10:
                recent_prices = data_1m['Close'].tail(10)
                price_changes = recent_prices.diff().dropna()
                
                # Count of minimal price movements (tick-level moves)
                tick_moves = (abs(price_changes) <= features['tick_size'] * 2).sum()
                features['tick_level_moves_ratio'] = tick_moves / len(price_changes) if len(price_changes) > 0 else 0
            
            # Market cap estimation (rough)
            # This would ideally come from fundamental data
            features['estimated_market_cap_tier'] = hash(symbol) % 5  # Placeholder: 0-4 (small to large cap)
            
            # Sector classification (simplified)
            sector_mapping = {
                'HBL': 'banking', 'UBL': 'banking', 'MCB': 'banking',
                'ENGRO': 'chemicals', 'FFC': 'fertilizer', 'EFERT': 'fertilizer',
                'LUCK': 'cement', 'MLCF': 'cement', 'DGKC': 'cement',
                'TRG': 'technology', 'SYSTEMS': 'technology',
                'PSO': 'oil_gas', 'OGDC': 'oil_gas', 'PPL': 'oil_gas'
            }
            
            features['sector_code'] = hash(sector_mapping.get(symbol, 'other')) % 10
            
            # Trading session intensity
            current_time = data_1m.index[-1].time()
            if time(9, 15) <= current_time <= time(10, 0):
                features['session_intensity'] = 3  # High (opening)
            elif time(14, 30) <= current_time <= time(15, 30):
                features['session_intensity'] = 3  # High (closing)
            elif time(12, 30) <= current_time <= time(13, 30):
                features['session_intensity'] = 1  # Low (lunch)
            else:
                features['session_intensity'] = 2  # Medium
            
        except Exception as e:
            print(f"Error extracting PSX features: {e}")
        
        return features
    
    def get_feature_vector(self, features: IntradayFeatures) -> np.ndarray:
        """Convert IntradayFeatures to numerical feature vector"""
        feature_vector = []
        
        # Combine all feature dictionaries
        all_features = {
            **features.price_features,
            **features.technical_features,
            **features.microstructure_features,
            **features.temporal_features,
            **features.volatility_features,
            **features.session_features,
            **features.psx_features
        }
        
        # Convert to numerical array
        for key, value in sorted(all_features.items()):
            if isinstance(value, (int, float)) and not np.isnan(value):
                feature_vector.append(value)
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def get_feature_names(self, features: IntradayFeatures) -> List[str]:
        """Get ordered list of feature names"""
        all_features = {
            **features.price_features,
            **features.technical_features,
            **features.microstructure_features,
            **features.temporal_features,
            **features.volatility_features,
            **features.session_features,
            **features.psx_features
        }
        
        return sorted(all_features.keys())

# Testing function
def test_feature_engine():
    """Test the enhanced intraday feature engine"""
    print("🧪 Testing Enhanced Intraday Feature Engine...")
    
    # Create sample 1-minute data
    dates = pd.date_range(start='2024-01-01 09:15', end='2024-01-01 15:30', freq='1min')
    n_samples = len(dates)
    
    # Generate realistic price data
    base_price = 100
    price_data = []
    current_price = base_price
    
    for i in range(n_samples):
        # Add some realistic intraday patterns
        hour = dates[i].hour
        minute = dates[i].minute
        
        # Opening volatility
        if hour == 9 and minute < 45:
            volatility = 0.003
        # Lunch time low volatility
        elif 12 <= hour <= 13:
            volatility = 0.001
        # Closing volatility
        elif hour >= 15:
            volatility = 0.002
        else:
            volatility = 0.0015
        
        change = np.random.normal(0, volatility)
        current_price *= (1 + change)
        price_data.append(current_price)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.999, 1.001) for p in price_data],
        'High': [p * np.random.uniform(1.001, 1.003) for p in price_data],
        'Low': [p * np.random.uniform(0.997, 0.999) for p in price_data],
        'Close': price_data,
        'Volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # Test feature extraction
    engine = EnhancedIntradayFeatureEngine()
    features = engine.extract_comprehensive_features('HBL', sample_data)
    
    print(f"✅ Extracted features for {features.symbol}")
    print(f"   Timestamp: {features.timestamp}")
    print(f"   Price features: {len(features.price_features)}")
    print(f"   Technical features: {len(features.technical_features)}")
    print(f"   Microstructure features: {len(features.microstructure_features)}")
    print(f"   Temporal features: {len(features.temporal_features)}")
    print(f"   Volatility features: {len(features.volatility_features)}")
    print(f"   Session features: {len(features.session_features)}")
    print(f"   PSX features: {len(features.psx_features)}")
    
    # Test feature vector conversion
    feature_vector = engine.get_feature_vector(features)
    feature_names = engine.get_feature_names(features)
    
    print(f"   Total features: {len(feature_vector)}")
    print(f"   Feature names: {len(feature_names)}")
    
    # Show sample features
    print("\n📊 Sample Features:")
    for i, (name, value) in enumerate(zip(feature_names[:10], feature_vector[:10])):
        print(f"   {name}: {value:.4f}")
    
    print("\n✅ Enhanced Intraday Feature Engine test completed!")

if __name__ == "__main__":
    test_feature_engine()