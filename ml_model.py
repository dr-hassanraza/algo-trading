
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

MODEL_FILE = Path("stock_predictor.joblib")

def create_features_and_target(df: pd.DataFrame, future_days: int = 5, percent_change: float = 0.02):
    """Create features and target variable for the model."""
    df['future_price'] = df['Close'].shift(-future_days)
    df['price_change'] = (df['future_price'] - df['Close']) / df['Close']
    df['target'] = (df['price_change'] > percent_change).astype(int)

    # Select features
    features = [
        'MA44',
        'BB_pctB',
        'RSI',
        'ATR',
        'MACD_Hist',
        'Stoch',
        'ADX',
        'Volume_MA_10',
        'Price_Change_5d',
        'Price_Volatility'
    ]

    df = df.dropna()
    X = df[features]
    y = df['target']

    return X, y

def train_model(df: pd.DataFrame):
    """Train the machine learning model and save it."""
    X, y = create_features_and_target(df)

    if len(X) < 100:
        return {"status": "error", "message": "Not enough data to train the model."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save the trained model
    joblib.dump(model, MODEL_FILE)

    return {"status": "success", "accuracy": accuracy}

def predict_signal(df: pd.DataFrame) -> dict:
    """Predict the signal for the latest data point."""
    if not MODEL_FILE.exists():
        return {"status": "error", "message": "Model not found. Please train the model first."}

    model = joblib.load(MODEL_FILE)

    # Select features for prediction
    features = [
        'MA44',
        'BB_pctB',
        'RSI',
        'ATR',
        'MACD_Hist',
        'Stoch',
        'ADX',
        'Volume_MA_10',
        'Price_Change_5d',
        'Price_Volatility'
    ]

    latest_data = df[features].iloc[-1:].dropna()

    if latest_data.empty:
        return {"status": "error", "message": "Not enough data for prediction."}

    prediction = model.predict(latest_data)[0]
    probability = model.predict_proba(latest_data)[0][1]  # Probability of price increase

    return {"status": "success", "prediction": prediction, "probability": probability}
