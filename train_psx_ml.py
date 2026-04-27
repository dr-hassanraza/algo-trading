"""
PSX cross-sectional ranking model — honest version.

WHY THIS APPROACH (instead of absolute direction prediction):
With only 1 year of data, the PSX index moved through several regimes
(neutral → bear → strong bull). A model trained to predict absolute
direction sees train/val/test as totally different distributions and
fails to generalize. The cross-sectional approach asks a *regime-invariant*
question: "given two stocks on the same day, which will outperform the
other over the next 5 days?" Class balance is always ~50/50 by construction
because labels are defined within each day's cross-section.

- Inputs:  data_cache/ohlcv/{TICKER}.parquet  (Phase 2 EODHD pulls)
- Output:  models/v2/{lightgbm_psx.pkl, feature_names.json, REPORT.md}

Pipeline:
1. Load all OHLCV parquets, compute features per symbol (time-series).
2. For each date, compute cross-sectional ranks of features and label.
3. Label = 1 if 5d forward return is in TOP quartile of that day's cross-section,
   0 if in BOTTOM quartile. Drop the middle 50% (ambiguous).
4. Chronological train/val/test split with embargo gap.
5. LightGBM binary classifier with early stopping on val AUC.
6. Evaluate vs (a) random and (b) "long top-quartile, short bottom-quartile"
   buy-hold strategy.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------------------- config (overridable via CLI) --------------------------
DATA_DIR = Path("data_cache/ohlcv")
MODEL_DIR = Path("models/v2")
LABEL_HORIZON = 5
MIN_ROWS_PER_SYMBOL = 200
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
EMBARGO_DAYS = LABEL_HORIZON
MIN_STOCKS_PER_DAY = 50  # need enough cross-sectional breadth
TOP_Q = 0.25
BOT_Q = 0.25
RANDOM_SEED = 42


# -------------------------- per-symbol features --------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    dn = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def macd_diff(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line - signal


def build_per_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    if len(df) < MIN_ROWS_PER_SYMBOL:
        return None
    df = df.copy().sort_index()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    log_close = np.log(c.replace(0, np.nan))
    f = pd.DataFrame(index=df.index)
    f["ret_1d"] = log_close.diff(1)
    f["ret_5d"] = log_close.diff(5)
    f["ret_20d"] = log_close.diff(20)
    f["rsi_14"] = rsi(c, 14)
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    f["sma_ratio"] = (sma20 / sma50) - 1
    f["macd_diff"] = macd_diff(c)
    f["atr_pct"] = atr(h, l, c, 14) / c
    bb_mid, bb_std = sma20, c.rolling(20).std()
    f["bb_pos"] = (c - (bb_mid - 2 * bb_std)) / ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)).replace(0, np.nan)
    vol_mean = v.rolling(20).mean()
    vol_std = v.rolling(20).std().replace(0, np.nan)
    f["vol_z"] = (v - vol_mean) / vol_std
    # Forward 5-day log return (used to make the label)
    f["fwd_ret"] = log_close.shift(-LABEL_HORIZON) - log_close
    f["symbol"] = symbol
    return f.dropna(subset=["ret_1d", "ret_5d", "ret_20d", "rsi_14", "sma_ratio",
                            "macd_diff", "atr_pct", "bb_pos", "vol_z"])


# -------------------------- cross-sectional transforms --------------------------
RAW_FEATURES = ["ret_1d", "ret_5d", "ret_20d", "rsi_14", "sma_ratio",
                "macd_diff", "atr_pct", "bb_pos", "vol_z"]


def add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """For each (date), rank each feature within that day's cross-section to (0,1).
    Adds <feat>_rank columns. Rank features are regime-invariant."""
    out = panel.copy()
    for feat in RAW_FEATURES:
        out[f"{feat}_rank"] = out.groupby(level=0)[feat].rank(pct=True)
    return out


def make_xs_label(panel: pd.DataFrame) -> pd.DataFrame:
    """Within each day: label=1 if fwd_ret in top quartile, 0 if bottom quartile,
    NaN otherwise. Drop NaN rows. Also drops days with too few stocks."""
    g = panel.groupby(level=0)
    counts = g["fwd_ret"].transform("count")
    panel = panel[counts >= MIN_STOCKS_PER_DAY].copy()
    g = panel.groupby(level=0)
    panel["fwd_rank"] = g["fwd_ret"].rank(pct=True)
    panel["label"] = np.nan
    panel.loc[panel["fwd_rank"] >= 1 - TOP_Q, "label"] = 1.0
    panel.loc[panel["fwd_rank"] <= BOT_Q, "label"] = 0.0
    return panel.dropna(subset=["label"])


# -------------------------- main --------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory of per-symbol parquet OHLCV files")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR,
                        help="Output directory for model + report")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    model_dir: Path = args.model_dir

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print("ERROR: no parquet files in data_cache/ohlcv/", file=sys.stderr)
        return 1

    print(f"Loading {len(files)} symbol files…")
    panels, skipped = [], 0
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            skipped += 1
            continue
        feat = build_per_symbol(df, fp.stem)
        if feat is not None:
            panels.append(feat)
        else:
            skipped += 1

    panel = pd.concat(panels).sort_index()
    print(f"Per-symbol panel: {len(panel):,} rows, {panel['symbol'].nunique()} symbols, {skipped} skipped")

    # Add cross-sectional rank features (regime-invariant)
    panel = add_cross_sectional_ranks(panel)

    # Build cross-sectional label (top quartile vs bottom quartile each day)
    labeled = make_xs_label(panel)
    print(f"Labeled rows (top + bottom quartile only): {len(labeled):,}")
    print(f"Date range: {labeled.index.min().date()} → {labeled.index.max().date()}")

    feature_cols = RAW_FEATURES + [f"{f}_rank" for f in RAW_FEATURES]
    print(f"Features: {len(feature_cols)} ({len(RAW_FEATURES)} raw + {len(RAW_FEATURES)} cross-sectional ranks)")

    # Chronological splits with embargo
    unique_dates = np.array(sorted(labeled.index.unique()))
    n = len(unique_dates)
    tr_end = unique_dates[int(n * TRAIN_FRAC)]
    va_start = unique_dates[int(n * TRAIN_FRAC) + EMBARGO_DAYS]
    va_end = unique_dates[int(n * (TRAIN_FRAC + VAL_FRAC))]
    te_start = unique_dates[int(n * (TRAIN_FRAC + VAL_FRAC)) + EMBARGO_DAYS]

    train = labeled[labeled.index <= tr_end]
    val = labeled[(labeled.index >= va_start) & (labeled.index <= va_end)]
    test = labeled[labeled.index >= te_start]

    print(f"\nSplits (with {EMBARGO_DAYS}-day embargo):")
    print(f"  train: {train.index.min().date()} → {train.index.max().date()}  ({len(train):,} rows)")
    print(f"  val:   {val.index.min().date()} → {val.index.max().date()}  ({len(val):,} rows)")
    print(f"  test:  {test.index.min().date()} → {test.index.max().date()}  ({len(test):,} rows)")
    print(f"\nClass balance — train: {train['label'].mean():.3f}, val: {val['label'].mean():.3f}, test: {test['label'].mean():.3f}")

    X_tr, y_tr = train[feature_cols], train["label"].astype(int)
    X_va, y_va = val[feature_cols], val["label"].astype(int)
    X_te, y_te = test[feature_cols], test["label"].astype(int)

    # ----- train -----
    print("\nTraining LightGBM…")
    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_child_samples=200,
        reg_alpha=0.1,
        reg_lambda=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)])

    # ----- evaluate -----
    p_te = model.predict_proba(X_te)[:, 1]
    pred_te = (p_te > 0.5).astype(int)
    auc = roc_auc_score(y_te, p_te)
    acc = accuracy_score(y_te, pred_te)

    fwd_te = test["fwd_ret"].values
    # Long top-decile, short bottom-decile by model probability per day
    test_eval = test.copy()
    test_eval["pred_p"] = p_te
    grp = test_eval.groupby(level=0)
    top10 = grp["pred_p"].rank(pct=True) >= 0.9
    bot10 = grp["pred_p"].rank(pct=True) <= 0.1
    long_short_pnl = test_eval.loc[top10, "fwd_ret"].mean() - test_eval.loc[bot10, "fwd_ret"].mean()
    long_only_pnl = test_eval.loc[top10, "fwd_ret"].mean()

    # Equal-weight cross-sectional baseline (mean fwd_ret of all labeled stocks)
    baseline_mean_ret = test_eval["fwd_ret"].mean()

    metrics = {
        "n_test_obs": int(len(test)),
        "test_period": f"{test.index.min().date()} → {test.index.max().date()}",
        "accuracy": float(acc),
        "auc": float(auc),
        "test_class_balance": float(y_te.mean()),
        "long_top_decile_mean_5d_ret": float(long_only_pnl),
        "short_bot_decile_implied_mean_5d_ret": float(test_eval.loc[bot10, "fwd_ret"].mean()),
        "long_short_decile_spread_5d_ret": float(long_short_pnl),
        "equal_weight_baseline_mean_5d_ret": float(baseline_mean_ret),
        "best_iter": int(model.best_iteration_) if model.best_iteration_ else int(model.n_estimators),
    }
    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ----- verdict -----
    has_edge = (metrics["auc"] > 0.55) and (metrics["long_short_decile_spread_5d_ret"] > 0.005)
    verdict = "USABLE" if has_edge else "NO_EDGE"
    print(f"\nVerdict: {verdict}  (auc>0.55 AND ls_spread>0.5% per 5d trade required)")

    # ----- save artifacts -----
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "lightgbm_psx.pkl")
    (model_dir / "feature_names.json").write_text(json.dumps(feature_cols, indent=2))

    fi = pd.DataFrame({"feature": feature_cols,
                       "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    report = f"""# PSX Cross-Sectional ML Model — Phase 3 Report

**Verdict:** **{verdict}**

## Why cross-sectional ranking (not direct up/down)
A first attempt at predicting 5-day absolute direction (up vs down) failed —
train/val/test had wildly different class balances (48% / 24% / 61% positive)
because PSX moved through three different regimes in just one year. The model
couldn't generalize and was beaten by a "always predict up" baseline.

This v2 model asks a *regime-invariant* question instead: **"which stocks will
outperform the cross-section over the next 5 days?"** Labels are defined per
day relative to that day's universe — top quartile = 1, bottom quartile = 0,
middle 50% dropped. Class balance is ~50/50 by construction, so the model
isn't fooled by overall market drift.

## Setup
- Data: {len(files)} EODHD parquet files, ~1y daily OHLCV per symbol
- Symbols used: {panel['symbol'].nunique()} (filtered to ≥{MIN_ROWS_PER_SYMBOL} rows)
- Per-symbol panel rows: {len(panel):,}
- Labeled rows after top/bottom quartile filter: {len(labeled):,}
- Features ({len(feature_cols)}): {len(RAW_FEATURES)} raw + {len(RAW_FEATURES)} cross-sectional rank versions
- Label: 1 if 5d-forward return in top quartile of that day, 0 if bottom quartile
- Split: chronological train/val/test with {EMBARGO_DAYS}-day embargo
- Model: LightGBM binary classifier, early-stopped at iter {metrics['best_iter']}

## Splits
| Split | Period | Rows | Class balance |
|---|---|---|---|
| Train | {train.index.min().date()} → {train.index.max().date()} | {len(train):,} | {train['label'].mean():.3f} |
| Val | {val.index.min().date()} → {val.index.max().date()} | {len(val):,} | {val['label'].mean():.3f} |
| Test | {test.index.min().date()} → {test.index.max().date()} | {len(test):,} | {test['label'].mean():.3f} |

## Test-set performance
| Metric | Value |
|---|---|
| Accuracy | {metrics['accuracy']:.4f} (vs 0.5 random) |
| AUC | {metrics['auc']:.4f} (vs 0.5 random) |
| Mean 5d return — long top decile | {metrics['long_top_decile_mean_5d_ret']:+.5f} |
| Mean 5d return — short bottom decile (implied) | {metrics['short_bot_decile_implied_mean_5d_ret']:+.5f} |
| Long/short decile spread per 5d | **{metrics['long_short_decile_spread_5d_ret']:+.5f}** |
| Equal-weight baseline (mean of all labeled) | {metrics['equal_weight_baseline_mean_5d_ret']:+.5f} |

## Feature importance (LightGBM split-count)
{fi.to_markdown(index=False)}

## Honest interpretation
"""
    if has_edge:
        report += f"""
The model crosses the minimum thresholds (AUC > 0.55 AND long/short decile
spread > 0.5% per 5-day trade). It's a **real but small edge** on this 1-year
sample. Important caveats:

1. **Test set is small** ({metrics['n_test_obs']:,} stock-days, ~{len(test.index.unique())} trading days). Confidence intervals on
   these numbers are wide. Run live in paper-trading mode for at least 3 months
   before sizing up.
2. **No transaction costs modeled.** PSX brokerage (~0.25%) plus slippage on
   small/illiquid names will erase a meaningful chunk of the long/short spread.
   Net edge after costs is what matters.
3. **One year of data only.** This model has not seen a full bull/bear cycle
   and may degrade in regimes it wasn't trained on.
4. **Suggested usage:** treat the top-decile model probability as a *ranking
   signal*, not a buy/sell decision. Combine with the rule-based engine and
   risk manager already in the repo.
"""
    else:
        report += f"""
The model does not cross the minimum thresholds (AUC > 0.55, L/S spread > 0.5%).
The cross-sectional approach was the right methodology for this dataset, but the
1-year sample with ~{len(test.index.unique())} test days is genuinely too small to extract a robust signal
with the conservative feature set used here.

This is the **honest** outcome — the system is saved for inspection but should
**not be deployed with real money**. To make further progress:

1. **Get more history** — EODHD All-World plan ($19.99/mo) gives 5+ years.
   This is the single highest-leverage change you can make.
2. **Add fundamentals/sector features** — book/price, sector dummies, which
   would add information beyond pure technicals.
3. **Until then**, the rule-based signal engine combined with strict risk
   management (position limits, hard stops) is the more honest tool.
"""
    (model_dir / "REPORT.md").write_text(report)
    print(f"\nArtifacts written to {model_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
