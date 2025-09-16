"""
Quick ML Fix for PSX Trading System
Creates a better ML model quickly to replace uniform buy signals
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def create_diverse_training_data():
    """Create diverse synthetic training data with realistic patterns"""
    print("📊 Creating diverse training dataset...")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Create different market scenarios
    scenarios = []
    
    # Scenario 1: Bull Market (30% of data)
    bull_samples = int(n_samples * 0.3)
    bull_data = {
        'rsi': np.random.uniform(20, 80, bull_samples),
        'sma_5': np.random.uniform(0.98, 1.05, bull_samples),
        'sma_10': np.random.uniform(0.95, 1.03, bull_samples), 
        'sma_20': np.random.uniform(0.90, 1.01, bull_samples),
        'volume_ratio': np.random.uniform(0.8, 3.0, bull_samples),
        'momentum': np.random.uniform(-0.02, 0.05, bull_samples),
        'volatility': np.random.uniform(0.01, 0.04, bull_samples),
        'macd_histogram': np.random.uniform(-0.5, 2.0, bull_samples),
        'bb_position': np.random.uniform(0.2, 0.9, bull_samples),
        'adx': np.random.uniform(20, 70, bull_samples),
        'scenario': ['bull'] * bull_samples
    }
    scenarios.append(bull_data)
    
    # Scenario 2: Bear Market (25% of data)  
    bear_samples = int(n_samples * 0.25)
    bear_data = {
        'rsi': np.random.uniform(20, 80, bear_samples),
        'sma_5': np.random.uniform(0.92, 1.02, bear_samples),
        'sma_10': np.random.uniform(0.90, 1.01, bear_samples),
        'sma_20': np.random.uniform(0.95, 1.05, bear_samples),
        'volume_ratio': np.random.uniform(0.5, 2.5, bear_samples),
        'momentum': np.random.uniform(-0.05, 0.02, bear_samples),
        'volatility': np.random.uniform(0.02, 0.08, bear_samples),
        'macd_histogram': np.random.uniform(-2.0, 0.5, bear_samples),
        'bb_position': np.random.uniform(0.1, 0.8, bear_samples),
        'adx': np.random.uniform(15, 60, bear_samples),
        'scenario': ['bear'] * bear_samples
    }
    scenarios.append(bear_data)
    
    # Scenario 3: Sideways/Range Market (45% of data)
    range_samples = n_samples - bull_samples - bear_samples
    range_data = {
        'rsi': np.random.uniform(25, 75, range_samples),
        'sma_5': np.random.uniform(0.97, 1.03, range_samples),
        'sma_10': np.random.uniform(0.96, 1.04, range_samples),
        'sma_20': np.random.uniform(0.95, 1.05, range_samples),
        'volume_ratio': np.random.uniform(0.6, 2.0, range_samples),
        'momentum': np.random.uniform(-0.03, 0.03, range_samples),
        'volatility': np.random.uniform(0.01, 0.06, range_samples),
        'macd_histogram': np.random.uniform(-1.0, 1.0, range_samples),
        'bb_position': np.random.uniform(0.15, 0.85, range_samples),
        'adx': np.random.uniform(10, 50, range_samples),
        'scenario': ['range'] * range_samples
    }
    scenarios.append(range_data)
    
    # Combine all scenarios
    combined_data = {}
    for key in bull_data.keys():
        combined_data[key] = np.concatenate([scenario[key] for scenario in scenarios])
    
    df = pd.DataFrame(combined_data)
    
    # Create realistic labels based on technical conditions
    labels = []
    for _, row in df.iterrows():
        rsi = row['rsi']
        momentum = row['momentum'] 
        sma_trend = (row['sma_5'] - row['sma_20'])
        bb_pos = row['bb_position']
        scenario = row['scenario']
        
        # More nuanced labeling
        buy_score = 0
        sell_score = 0
        
        # RSI conditions
        if rsi < 30:
            buy_score += 2
        elif rsi < 40:
            buy_score += 1
        elif rsi > 70:
            sell_score += 2
        elif rsi > 60:
            sell_score += 1
            
        # Momentum conditions
        if momentum > 0.02:
            buy_score += 2
        elif momentum > 0.01:
            buy_score += 1
        elif momentum < -0.02:
            sell_score += 2
        elif momentum < -0.01:
            sell_score += 1
            
        # Trend conditions
        if sma_trend > 0.02:
            buy_score += 1
        elif sma_trend < -0.02:
            sell_score += 1
            
        # Bollinger Band position
        if bb_pos < 0.2:
            buy_score += 1
        elif bb_pos > 0.8:
            sell_score += 1
            
        # Market scenario influence
        if scenario == 'bull':
            buy_score += 1
        elif scenario == 'bear':
            sell_score += 1
            
        # Add randomness for diversity
        random_factor = np.random.uniform(-0.5, 0.5)
        buy_score += random_factor
        sell_score -= random_factor
        
        # Final decision with balanced distribution
        if buy_score > sell_score + 1.5:
            labels.append('BUY')
        elif sell_score > buy_score + 1.5:
            labels.append('SELL')
        else:
            labels.append('HOLD')
    
    df['target'] = labels
    
    # Remove scenario column
    df = df.drop('scenario', axis=1)
    
    print(f"Label distribution: {df['target'].value_counts().to_dict()}")
    return df

def train_quick_model():
    """Train a quick but effective model"""
    print("🤖 Training quick ML model...")
    
    # Create training data
    df = create_diverse_training_data()
    
    # Prepare features
    feature_cols = [
        'rsi', 'sma_5', 'sma_10', 'sma_20', 'volume_ratio',
        'momentum', 'volatility', 'macd_histogram', 'bb_position', 'adx'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest with balanced classes
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    model.fit(X_train, y_train)
    
    # Test accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    print(f"Top features: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/quick_ml_model.pkl')
    joblib.dump(le, 'models/quick_label_encoder.pkl')
    
    with open('models/quick_feature_names.txt', 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\\n")
    
    print("✅ Quick ML model saved successfully!")
    
    # Test predictions on sample data
    test_samples = [
        [25, 1.02, 1.01, 0.98, 1.5, 0.02, 0.03, 0.5, 0.3, 35],  # Should be BUY
        [75, 0.98, 0.99, 1.02, 1.2, -0.02, 0.04, -0.5, 0.8, 40],  # Should be SELL  
        [50, 1.00, 1.00, 1.00, 1.0, 0.00, 0.02, 0.0, 0.5, 25],   # Should be HOLD
    ]
    
    predictions = model.predict(test_samples)
    probabilities = model.predict_proba(test_samples)
    
    print("\\n📊 Sample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        pred_label = le.inverse_transform([pred])[0]
        confidence = prob.max() * 100
        print(f"  Sample {i+1}: {pred_label} ({confidence:.1f}% confidence)")
    
    return model, le, feature_cols

if __name__ == "__main__":
    print("🚀 Quick ML Model Training for PSX Trading System")
    train_quick_model()
    print("\\n🎉 Training completed! The new model should provide diverse signals.")