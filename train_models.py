import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import argparse
import os
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import TensorFlow and its components
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ Warning: TensorFlow is not installed. The LSTM primary model cannot be trained.")

def load_data_from_csv(file_path):
    """Loads historical data from a CSV file."""
    print(f"💾 Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        # --- Data Validation ---
        required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ Error: CSV file must contain the following columns: {required_columns}")
            return None
        
        # Convert to datetime and set as index
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        print(f"✅ Loaded {len(df)} data points.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ An error occurred while reading the CSV file: {e}")
        return None

def calculate_features(df):
    """Calculates a rich set of technical indicators."""
    print("🧮 Calculating technical features...")
    df['return'] = df['Close'].pct_change()
    
    # Momentum
    df['rsi'] = 100 - (100 / (1 + (df['return'].clip(lower=0).rolling(14).mean() / -df['return'].clip(upper=0).rolling(14).mean())))
    df['macd'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Volatility
    df['atr'] = pd.DataFrame(np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift()))), index=df.index).ewm(span=14, adjust=False).mean()
    df['bb_width'] = (df['Close'].rolling(20).std() * 4) / df['Close'].rolling(20).mean()

    # Volume
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma'] = df['Volume'].rolling(20).mean()

    df.dropna(inplace=True)
    print("✅ Features calculated.")
    return df

def get_triple_barrier_labels(df, t_final, upper_pct, lower_pct):
    """Applies the Triple-Barrier Method for labeling."""
    print("🏷️ Applying Triple-Barrier Method for labeling...")
    results = pd.Series(index=df.index, dtype='float64').fillna(0)
    
    for i in range(len(df) - t_final):
        entry_price = df['Close'].iloc[i]
        upper_barrier = entry_price * (1 + upper_pct)
        lower_barrier = entry_price * (1 - lower_pct)
        
        for j in range(1, t_final + 1):
            future_price = df['Close'].iloc[i + j]
            
            if future_price >= upper_barrier:
                results.iloc[i] = 1  # Upper barrier touched
                break
            elif future_price <= lower_barrier:
                results.iloc[i] = -1 # Lower barrier touched
                break
    
    print(f"✅ Labeling complete. Found {len(results[results==1])} buy, {len(results[results==-1])} sell, {len(results[results==0])} hold labels.")
    return results

def create_sequences(X, y, time_steps=60):
    """Creates sequences for LSTM model."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_primary_model(df, features, labels, output_dir, symbol):
    """Trains and saves the primary LSTM model."""
    if not TENSORFLOW_AVAILABLE:
        return None, None
        
    print("\n--- Phase 1: Training Primary LSTM Model ---")
    
    X = df[features]
    y = labels.copy()
    y[y == -1] = 0 # For binary classification (0=SELL/HOLD, 1=BUY)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_seq, y_seq = create_sequences(X_scaled, y.values, time_steps=60)
    
    if len(X_seq) == 0:
        print("❌ Not enough data to create sequences. Aborting primary model training.")
        return None, None

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print(f"🧠 Training LSTM model on {len(X_seq)} sequences...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_seq, y_seq, batch_size=32, epochs=50, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    model_path = os.path.join(output_dir, f'{symbol}_primary_model.h5')
    scaler_path = os.path.join(output_dir, f'{symbol}_primary_scaler.joblib')
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"✅ Primary model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    
    return model, scaler

def train_meta_model(df, features, labels, primary_model, scaler, output_dir, symbol):
    """Trains and saves the LightGBM meta-model."""
    print("\n--- Phase 2: Training Meta-Model (LightGBM) ---")
    
    if primary_model is None:
        print("❌ Primary model not available. Skipping meta-model training.")
        return

    X = df[features]
    X_scaled = scaler.transform(X)
    X_seq, _ = create_sequences(X_scaled, labels.values, time_steps=60)

    if len(X_seq) == 0:
        print("❌ Not enough data to create sequences for meta-model. Aborting.")
        return

    primary_predictions = primary_model.predict(X_seq, verbose=0)
    
    df_meta = df.iloc[60:].copy()
    df_meta['primary_pred'] = primary_predictions.flatten()

    actual_outcomes = labels.iloc[60:]
    correct_long = (df_meta['primary_pred'] > 0.5) & (actual_outcomes == 1)
    correct_short = (df_meta['primary_pred'] <= 0.5) & (actual_outcomes == -1)
    df_meta['meta_label'] = (correct_long | correct_short).astype(int)

    meta_features = features + ['primary_pred']
    X_meta = df_meta[meta_features]
    y_meta = df_meta['meta_label']

    print(f"🧠 Training LightGBM meta-model on {len(X_meta)} samples...")
    X_train, X_test, y_train, y_test = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42, stratify=y_meta)
    
    lgb_model = lgb.LGBMClassifier(objective='binary', random_state=42)
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)
    
    meta_model_path = os.path.join(output_dir, f'{symbol}_meta_model.joblib')
    joblib.dump(lgb_model, meta_model_path)
    print(f"✅ Meta-model saved to {meta_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Expert training script for the Advanced Trading System.")
    parser.add_argument("--symbol", type=str, required=True, help="The symbol to train on (e.g., 'HBL', 'ENGRO'). This is used for naming the output files.")
    parser.add_argument("--datafile", type=str, required=True, help="Path to the CSV file containing historical OHLCV data.")
    parser.add_argument("--output_dir", type=str, default="models/enhanced", help="Directory to save trained models.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    df = load_data_from_csv(args.datafile)
    if df is None:
        return
        
    # 2. Feature Engineering
    df = calculate_features(df)
    
    # 3. Labeling
    labels = get_triple_barrier_labels(df, t_final=20, upper_pct=0.005, lower_pct=0.005)
    
    # 4. Train Primary Model
    features = ['rsi', 'macd', 'macd_signal', 'atr', 'bb_width', 'volume_change', 'volume_ma']
    primary_model, scaler = train_primary_model(df, features, labels, args.output_dir, args.symbol)
    
    # 5. Train Meta Model
    if primary_model and scaler:
        train_meta_model(df, features, labels, primary_model, scaler, args.output_dir, args.symbol)
    
    print("\n🎉 Training complete! Models are saved in the 'models/enhanced' directory.")

if __name__ == "__main__":
    main()