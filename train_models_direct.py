#!/usr/bin/env python3
"""
Direct model training script - bypasses complex initialization
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required ML libraries are available"""
    try:
        import tensorflow as tf
        import lightgbm as lgb
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        logger.info("✅ All ML dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing ML dependencies: {e}")
        print("Please install required packages:")
        print("pip install tensorflow lightgbm scikit-learn pandas numpy requests")
        return False

def create_sample_training_data():
    """Create sample training data for demonstration"""
    logger.info("📊 Creating sample training data...")
    
    # Generate synthetic PSX-like data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: price, volume, rsi, macd, bb_position, volatility
    features = []
    targets = []
    
    for i in range(n_samples):
        # Simulate market data with some pattern
        price = 100 + np.random.normal(0, 10)
        volume = np.random.exponential(10000)
        rsi = np.random.uniform(20, 80)
        macd = np.random.normal(0, 2)
        bb_position = np.random.uniform(0, 1)
        volatility = np.random.exponential(0.02)
        
        # Simple target logic: 0=SELL, 1=HOLD, 2=BUY
        if rsi > 70 and macd < 0:
            target = 0  # SELL
        elif rsi < 30 and macd > 0:
            target = 2  # BUY  
        else:
            target = 1  # HOLD
            
        features.append([price, volume, rsi, macd, bb_position, volatility])
        targets.append(target)
    
    return np.array(features), np.array(targets)

def prepare_lstm_sequences(features, targets, sequence_length=60):
    """Prepare sequences for LSTM training"""
    logger.info("🔧 Preparing LSTM sequences...")
    
    X_lstm = []
    y_lstm = []
    
    for i in range(sequence_length, len(features)):
        X_lstm.append(features[i-sequence_length:i])
        y_lstm.append(targets[i])
    
    return np.array(X_lstm), np.array(y_lstm)

def train_lstm_model(X_train, y_train, X_val, y_val):
    """Train LSTM model"""
    logger.info("🧠 Training LSTM model...")
    
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: SELL, HOLD, BUY
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    # Get best accuracy
    best_accuracy = max(history.history['val_accuracy'])
    logger.info(f"✅ LSTM training complete. Best accuracy: {best_accuracy:.3f}")
    
    return model, best_accuracy

def train_meta_model(X_train, y_train, X_val, y_val):
    """Train LightGBM meta model"""
    logger.info("🌟 Training LightGBM meta model...")
    
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Parameters
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }
    
    # Train
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    # Calculate accuracy
    val_pred = model.predict(X_val)
    val_pred_class = np.argmax(val_pred, axis=1)
    accuracy = accuracy_score(y_val, val_pred_class)
    
    logger.info(f"✅ Meta model training complete. Accuracy: {accuracy:.3f}")
    return model, accuracy

def save_models(lstm_model, meta_model, scaler):
    """Save trained models"""
    logger.info("💾 Saving trained models...")
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save LSTM
    lstm_model.save(os.path.join(models_dir, "lstm_model.h5"))
    
    # Save Meta model
    import pickle
    with open(os.path.join(models_dir, "meta_model.pkl"), 'wb') as f:
        pickle.dump(meta_model, f)
    
    # Save scaler
    with open(os.path.join(models_dir, "feature_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    logger.info("✅ All models saved successfully!")

def main():
    """Main training function"""
    print("🚀 Direct Model Training Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create training data
    features, targets = create_sample_training_data()
    print(f"📊 Created {len(features)} training samples")
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Prepare LSTM sequences
    X_lstm, y_lstm = prepare_lstm_sequences(features_scaled, targets, 60)
    
    # Split data
    split_idx = int(0.8 * len(X_lstm))
    X_lstm_train, X_lstm_val = X_lstm[:split_idx], X_lstm[split_idx:]
    y_lstm_train, y_lstm_val = y_lstm[:split_idx], y_lstm[split_idx:]
    
    # Also prepare regular features for meta model
    from sklearn.model_selection import train_test_split
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        features_scaled, targets, test_size=0.2, random_state=42
    )
    
    print(f"🔧 LSTM training data: {X_lstm_train.shape}")
    print(f"🔧 Meta training data: {X_meta_train.shape}")
    
    # Train LSTM
    lstm_model, lstm_acc = train_lstm_model(X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val)
    
    # Train Meta model
    meta_model, meta_acc = train_meta_model(X_meta_train, y_meta_train, X_meta_val, y_meta_val)
    
    # Save models
    save_models(lstm_model, meta_model, scaler)
    
    print("\n🎉 Training Complete!")
    print(f"📈 LSTM Accuracy: {lstm_acc:.3f}")
    print(f"🎯 Meta Accuracy: {meta_acc:.3f}")
    print("💾 Models saved to ./models/ directory")
    print("\nNow restart your Streamlit app to see the models as 'Ready'!")

if __name__ == "__main__":
    main()