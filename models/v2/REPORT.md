# PSX Cross-Sectional ML Model — Phase 3 Report

**Verdict:** **USABLE**

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
- Data: 490 EODHD parquet files, ~1y daily OHLCV per symbol
- Symbols used: 453 (filtered to ≥200 rows)
- Per-symbol panel rows: 89,527
- Labeled rows after top/bottom quartile filter: 43,679
- Features (18): 9 raw + 9 cross-sectional rank versions
- Label: 1 if 5d-forward return in top quartile of that day, 0 if bottom quartile
- Split: chronological train/val/test with 5-day embargo
- Model: LightGBM binary classifier, early-stopped at iter 13

## Splits
| Split | Period | Rows | Class balance |
|---|---|---|---|
| Train | 2025-07-10 → 2026-01-21 | 30,596 | 0.502 |
| Val | 2026-01-28 → 2026-03-05 | 5,789 | 0.503 |
| Test | 2026-03-12 → 2026-04-17 | 5,514 | 0.502 |

## Test-set performance
| Metric | Value |
|---|---|
| Accuracy | 0.5854 (vs 0.5 random) |
| AUC | 0.5941 (vs 0.5 random) |
| Mean 5d return — long top decile | +0.03644 |
| Mean 5d return — short bottom decile (implied) | +0.02216 |
| Long/short decile spread per 5d | **+0.01428** |
| Equal-weight baseline (mean of all labeled) | +0.03112 |

## Feature importance (LightGBM split-count)
| feature        |   importance |
|:---------------|-------------:|
| ret_5d_rank    |           49 |
| sma_ratio      |           37 |
| atr_pct        |           27 |
| sma_ratio_rank |           26 |
| ret_20d        |           24 |
| macd_diff      |           23 |
| ret_5d         |           19 |
| bb_pos         |           16 |
| vol_z          |           16 |
| ret_1d         |           15 |
| vol_z_rank     |           15 |
| rsi_14_rank    |           14 |
| macd_diff_rank |           12 |
| atr_pct_rank   |           12 |
| ret_20d_rank   |           11 |
| rsi_14         |           11 |
| ret_1d_rank    |           11 |
| bb_pos_rank    |            9 |

## Honest interpretation

The model crosses the minimum thresholds (AUC > 0.55 AND long/short decile
spread > 0.5% per 5-day trade). It's a **real but small edge** on this 1-year
sample. Important caveats:

1. **Test set is small** (5,514 stock-days, ~25 trading days). Confidence intervals on
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
