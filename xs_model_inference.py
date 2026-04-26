"""
Cross-sectional model inference for the Streamlit app.

Loads the Phase 3 LightGBM model and runs universe-wide ranking on the
cached EODHD OHLCV data. Returns per-symbol predictions:
    {symbol: {prob_outperform, decile, rank_pct}}

The model needs the full daily cross-section to compute rank features,
so this module always runs inference on the entire PSX universe at once
and returns a dict keyed by symbol. Single-symbol callers should look
their symbol up in this dict.

Refresh by re-running `python fetch_eodhd_psx.py` (pulls latest EODHD
bars), then the next call here will pick up the new data after the
cache TTL expires.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data_cache/ohlcv")
MODEL_DIR = Path("models/v2")
MODEL_PATH = MODEL_DIR / "lightgbm_psx.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.json"

RAW_FEATURES = ["ret_1d", "ret_5d", "ret_20d", "rsi_14", "sma_ratio",
                "macd_diff", "atr_pct", "bb_pos", "vol_z"]
MIN_ROWS = 60  # need at least 60 days for features (RSI/SMA need warmup)


# -------- feature builders (mirror train_psx_ml.py) --------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _macd_diff(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    return line - line.ewm(span=9, adjust=False).mean()


def _features_for_latest(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < MIN_ROWS:
        return None
    df = df.sort_index()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    log_close = np.log(c.replace(0, np.nan))
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    bb_std = c.rolling(20).std()
    vol_mean = v.rolling(20).mean()
    vol_std = v.rolling(20).std().replace(0, np.nan)

    feats = {
        "ret_1d": log_close.diff(1).iloc[-1],
        "ret_5d": log_close.diff(5).iloc[-1],
        "ret_20d": log_close.diff(20).iloc[-1],
        "rsi_14": _rsi(c, 14).iloc[-1],
        "sma_ratio": (sma20 / sma50 - 1).iloc[-1],
        "macd_diff": _macd_diff(c).iloc[-1],
        "atr_pct": (_atr(h, l, c, 14) / c).iloc[-1],
        "bb_pos": ((c - (sma20 - 2 * bb_std)) /
                   ((sma20 + 2 * bb_std) - (sma20 - 2 * bb_std)).replace(0, np.nan)).iloc[-1],
        "vol_z": ((v - vol_mean) / vol_std).iloc[-1],
        "asof": df.index[-1],
    }
    if any(pd.isna(v) for k, v in feats.items() if k != "asof"):
        return None
    return feats


def model_available() -> bool:
    return MODEL_PATH.exists() and FEATURES_PATH.exists() and DATA_DIR.exists()


def load_model():
    """Load the LightGBM model + feature name list. Returns (model, feature_names)."""
    model = joblib.load(MODEL_PATH)
    feature_names = json.loads(FEATURES_PATH.read_text())
    return model, feature_names


def compute_universe_predictions(model=None, feature_names=None) -> pd.DataFrame:
    """Run inference across the entire PSX universe.

    Returns a DataFrame indexed by symbol with columns:
        prob_outperform, decile (1..10, 10 = best), rank_pct, asof
    """
    if model is None or feature_names is None:
        model, feature_names = load_model()

    rows = []
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        feats = _features_for_latest(df)
        if feats is None:
            continue
        feats["symbol"] = fp.stem
        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).set_index("symbol")
    asof = panel["asof"]
    panel = panel.drop(columns=["asof"])

    # Cross-sectional rank features (matches training pipeline)
    for feat in RAW_FEATURES:
        panel[f"{feat}_rank"] = panel[feat].rank(pct=True)

    X = panel[feature_names]
    panel["prob_outperform"] = model.predict_proba(X)[:, 1]
    panel["rank_pct"] = panel["prob_outperform"].rank(pct=True)
    panel["decile"] = (panel["rank_pct"] * 10).clip(upper=10).astype(int).clip(lower=1)
    panel["asof"] = asof
    return panel[["prob_outperform", "rank_pct", "decile", "asof"]].sort_values(
        "prob_outperform", ascending=False
    )


def get_prediction(symbol: str, predictions: pd.DataFrame) -> Optional[dict]:
    """Look up a single symbol's prediction. Returns None if not in universe."""
    if predictions is None or predictions.empty or symbol not in predictions.index:
        return None
    row = predictions.loc[symbol]
    return {
        "prob_outperform": float(row["prob_outperform"]),
        "decile": int(row["decile"]),
        "rank_pct": float(row["rank_pct"]),
        "asof": row["asof"],
    }
