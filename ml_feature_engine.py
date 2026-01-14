"""
ML FEATURE ENGINE - Advanced Feature Engineering for Trading
=============================================================
Creates 30+ features proven to have predictive power for stock movements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MLFeatureEngine:
    """
    Advanced feature engineering for ML-based trading signals
    """

    def __init__(self):
        self.feature_names = []
        self.scaler = None

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all ML features from OHLCV data
        Returns DataFrame with all features
        """
        if df.empty or len(df) < 50:
            return pd.DataFrame()

        df = df.copy()
        price_col = 'close' if 'close' in df.columns else 'price'

        # === PRICE-BASED FEATURES ===
        features = pd.DataFrame(index=df.index)

        # Returns at different horizons
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'return_{period}d'] = df[price_col].pct_change(period) * 100

        # Moving average features
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                sma = df[price_col].rolling(period).mean()
                features[f'price_vs_sma_{period}'] = (df[price_col] / sma - 1) * 100
                features[f'sma_{period}_slope'] = sma.pct_change(5) * 100

        # EMA crossovers
        if len(df) >= 20:
            ema_5 = df[price_col].ewm(span=5).mean()
            ema_20 = df[price_col].ewm(span=20).mean()
            features['ema_5_20_diff'] = (ema_5 / ema_20 - 1) * 100

        # === MOMENTUM FEATURES ===

        # RSI
        delta = df[price_col].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi'] = 100 - (100 / (1 + rs))

        # RSI momentum (change in RSI)
        features['rsi_momentum'] = features['rsi'].diff(5)

        # RSI zones (binary features)
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)

        # MACD
        if len(df) >= 26:
            ema12 = df[price_col].ewm(span=12).mean()
            ema26 = df[price_col].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            features['macd'] = macd
            features['macd_signal'] = signal_line
            features['macd_histogram'] = macd - signal_line
            features['macd_histogram_change'] = features['macd_histogram'].diff()

        # Stochastic
        if len(df) >= 14:
            low_14 = df[price_col].rolling(14).min()
            high_14 = df[price_col].rolling(14).max()
            features['stoch_k'] = 100 * (df[price_col] - low_14) / (high_14 - low_14)
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # Williams %R
        features['williams_r'] = -100 * (high_14 - df[price_col]) / (high_14 - low_14) if len(df) >= 14 else 0

        # Rate of Change
        features['roc_5'] = df[price_col].pct_change(5) * 100
        features['roc_10'] = df[price_col].pct_change(10) * 100
        features['roc_20'] = df[price_col].pct_change(20) * 100

        # === VOLATILITY FEATURES ===

        # ATR
        if 'high' in df.columns and 'low' in df.columns:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df[price_col].shift())
            low_close = abs(df['low'] - df[price_col].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        else:
            tr = df[price_col].diff().abs()

        features['atr'] = tr.rolling(14).mean()
        features['atr_percent'] = (features['atr'] / df[price_col]) * 100

        # Volatility (rolling std of returns)
        features['volatility_5'] = df[price_col].pct_change().rolling(5).std() * 100
        features['volatility_20'] = df[price_col].pct_change().rolling(20).std() * 100
        features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']

        # Bollinger Bands
        if len(df) >= 20:
            bb_middle = df[price_col].rolling(20).mean()
            bb_std = df[price_col].rolling(20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            features['bb_position'] = (df[price_col] - bb_lower) / (bb_upper - bb_lower)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100

        # === VOLUME FEATURES ===
        if 'volume' in df.columns:
            features['volume_sma_10'] = df['volume'].rolling(10).mean()
            features['volume_sma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_20']

            # Volume momentum
            features['volume_change'] = df['volume'].pct_change(5) * 100

            # Price-volume relationship
            features['pv_trend'] = (df[price_col].pct_change() * df['volume']).rolling(10).sum()

            # On-Balance Volume trend
            obv = (np.sign(df[price_col].diff()) * df['volume']).cumsum()
            features['obv_slope'] = obv.diff(10) / obv.shift(10) * 100

            # Volume spikes
            features['volume_spike'] = (features['volume_ratio'] > 2.0).astype(int)

        # === TREND FEATURES ===

        # ADX (simplified)
        if len(df) >= 14:
            plus_dm = df[price_col].diff().clip(lower=0)
            minus_dm = (-df[price_col].diff()).clip(lower=0)

            atr_14 = features['atr']
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
            features['adx'] = dx.rolling(14).mean()
            features['di_diff'] = plus_di - minus_di

        # Trend strength
        if len(df) >= 50:
            features['trend_50'] = (df[price_col] > df[price_col].rolling(50).mean()).astype(int)
            features['above_sma_count'] = (
                (df[price_col] > df[price_col].rolling(5).mean()).astype(int) +
                (df[price_col] > df[price_col].rolling(10).mean()).astype(int) +
                (df[price_col] > df[price_col].rolling(20).mean()).astype(int) +
                (df[price_col] > df[price_col].rolling(50).mean()).astype(int)
            )

        # === PRICE PATTERNS ===

        # Distance from recent high/low
        features['dist_from_high_20'] = (df[price_col] / df[price_col].rolling(20).max() - 1) * 100
        features['dist_from_low_20'] = (df[price_col] / df[price_col].rolling(20).min() - 1) * 100

        # Distance from 52-week high/low (if enough data)
        if len(df) >= 252:
            features['dist_from_52w_high'] = (df[price_col] / df[price_col].rolling(252).max() - 1) * 100
            features['dist_from_52w_low'] = (df[price_col] / df[price_col].rolling(252).min() - 1) * 100
        else:
            # Use available data
            max_period = min(len(df), 252)
            features['dist_from_52w_high'] = (df[price_col] / df[price_col].rolling(max_period).max() - 1) * 100
            features['dist_from_52w_low'] = (df[price_col] / df[price_col].rolling(max_period).min() - 1) * 100

        # Consecutive up/down days
        up_days = (df[price_col].diff() > 0).astype(int)
        down_days = (df[price_col].diff() < 0).astype(int)
        features['consecutive_up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumsum()
        features['consecutive_down'] = down_days.groupby((down_days != down_days.shift()).cumsum()).cumsum()

        # Gap detection
        if 'open' in df.columns:
            features['gap_percent'] = (df['open'] - df[price_col].shift()) / df[price_col].shift() * 100
        else:
            features['gap_percent'] = 0

        # === MARKET STRUCTURE ===

        # Support/Resistance features
        features['dist_to_round_number'] = self._distance_to_round_number(df[price_col])

        # Price position in range
        if len(df) >= 20:
            range_high = df[price_col].rolling(20).max()
            range_low = df[price_col].rolling(20).min()
            features['price_position'] = (df[price_col] - range_low) / (range_high - range_low)

        # Store feature names
        self.feature_names = features.columns.tolist()

        return features

    def _distance_to_round_number(self, prices: pd.Series) -> pd.Series:
        """Calculate distance to nearest round number (psychological level)"""
        def nearest_round(price):
            if price < 10:
                return round(price)
            elif price < 100:
                return round(price / 5) * 5
            elif price < 1000:
                return round(price / 10) * 10
            else:
                return round(price / 50) * 50

        round_prices = prices.apply(nearest_round)
        return (prices - round_prices) / prices * 100

    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(self.feature_names, importance))
        return {}

    def prepare_features_for_prediction(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for ML prediction
        Returns: (feature_array, feature_names)
        """
        features = self.calculate_all_features(df)

        if features.empty:
            return np.array([]), []

        # Get latest row
        latest = features.iloc[-1]

        # Handle missing values
        latest = latest.fillna(0)

        # Replace infinites
        latest = latest.replace([np.inf, -np.inf], 0)

        return latest.values, self.feature_names

    def create_training_dataset(self, df: pd.DataFrame,
                               forward_period: int = 5,
                               threshold_pct: float = 2.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training dataset with labels
        Labels: 1 (BUY) if price goes up > threshold, -1 (SELL) if down > threshold, 0 (HOLD)
        """
        features = self.calculate_all_features(df)
        price_col = 'close' if 'close' in df.columns else 'price'

        # Calculate forward returns
        forward_returns = df[price_col].pct_change(forward_period).shift(-forward_period) * 100

        # Create labels
        labels = pd.Series(0, index=df.index)
        labels[forward_returns > threshold_pct] = 1  # BUY
        labels[forward_returns < -threshold_pct] = -1  # SELL

        # Align features and labels
        valid_idx = features.dropna().index.intersection(labels.dropna().index)

        X = features.loc[valid_idx]
        y = labels.loc[valid_idx]

        # Remove last forward_period rows (no labels)
        X = X.iloc[:-forward_period]
        y = y.iloc[:-forward_period]

        return X, y


class MLModelTrainer:
    """
    Train and manage ML models for signal prediction
    """

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.feature_engine = MLFeatureEngine()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def train_model(self, historical_data: Dict[str, pd.DataFrame],
                   model_type: str = 'lightgbm') -> Dict:
        """
        Train ML model on historical data from multiple symbols
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import LabelEncoder

            # Combine data from all symbols
            all_X = []
            all_y = []

            for symbol, df in historical_data.items():
                X, y = self.feature_engine.create_training_dataset(df)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)

            if not all_X:
                return {'success': False, 'error': 'No training data'}

            X = pd.concat(all_X, ignore_index=True)
            y = pd.concat(all_y, ignore_index=True)

            # Handle missing values
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)

            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, shuffle=False
            )

            # Train model
            if model_type == 'lightgbm':
                import lightgbm as lgb

                model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            elif model_type == 'xgboost':
                import xgboost as xgb

                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )

            # Fit model
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Save model
            model_path = os.path.join(self.model_dir, 'enhanced_ml_model.pkl')
            encoder_path = os.path.join(self.model_dir, 'enhanced_label_encoder.pkl')
            features_path = os.path.join(self.model_dir, 'feature_names.pkl')

            joblib.dump(model, model_path)
            joblib.dump(label_encoder, encoder_path)
            joblib.dump(self.feature_engine.feature_names, features_path)

            # Feature importance
            importance = self.feature_engine.get_feature_importance(model)
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                'success': True,
                'accuracy': accuracy,
                'model_path': model_path,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'top_features': top_features,
                'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
            }

        except ImportError as e:
            return {'success': False, 'error': f'Missing library: {e}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def load_model(self):
        """Load trained model"""
        try:
            model_path = os.path.join(self.model_dir, 'enhanced_ml_model.pkl')
            encoder_path = os.path.join(self.model_dir, 'enhanced_label_encoder.pkl')
            features_path = os.path.join(self.model_dir, 'feature_names.pkl')

            if all(os.path.exists(p) for p in [model_path, encoder_path, features_path]):
                model = joblib.load(model_path)
                encoder = joblib.load(encoder_path)
                feature_names = joblib.load(features_path)
                return model, encoder, feature_names
            return None, None, None
        except:
            return None, None, None

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Make prediction using trained model
        """
        model, encoder, feature_names = self.load_model()

        if model is None:
            return {'prediction': 'HOLD', 'confidence': 0, 'error': 'Model not loaded'}

        try:
            features = self.feature_engine.calculate_all_features(df)

            if features.empty:
                return {'prediction': 'HOLD', 'confidence': 0, 'error': 'No features'}

            latest = features.iloc[-1:].copy()

            # Ensure same features as training
            for f in feature_names:
                if f not in latest.columns:
                    latest[f] = 0

            latest = latest[feature_names]
            latest = latest.fillna(0)
            latest = latest.replace([np.inf, -np.inf], 0)

            # Predict
            proba = model.predict_proba(latest)[0]
            pred_class = model.predict(latest)[0]
            pred_label = encoder.inverse_transform([pred_class])[0]

            # Map numeric labels to strings
            label_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            prediction = label_map.get(pred_label, 'HOLD')

            confidence = proba.max() * 100

            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': dict(zip(encoder.classes_, proba)),
                'error': None
            }

        except Exception as e:
            return {'prediction': 'HOLD', 'confidence': 0, 'error': str(e)}


if __name__ == "__main__":
    print("Testing ML Feature Engine...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    prices = 100 + np.cumsum(np.random.randn(200) * 2)
    volumes = np.random.randint(50000, 200000, 200)

    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': volumes
    })

    engine = MLFeatureEngine()
    features = engine.calculate_all_features(df)

    print(f"Generated {len(engine.feature_names)} features:")
    for i, name in enumerate(engine.feature_names[:10]):
        print(f"  {i+1}. {name}")
    print(f"  ... and {len(engine.feature_names) - 10} more")

    print(f"\nLatest feature values:")
    latest = features.iloc[-1]
    for name in engine.feature_names[:5]:
        print(f"  {name}: {latest[name]:.2f}")
