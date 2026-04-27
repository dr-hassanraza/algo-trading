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
- Data: 521 EODHD parquet files, ~1y daily OHLCV per symbol
- Symbols used: 485 (filtered to ≥200 rows)
- Per-symbol panel rows: 514,732
- Labeled rows after top/bottom quartile filter: 255,836
- Features (18): 9 raw + 9 cross-sectional rank versions
- Label: 1 if 5d-forward return in top quartile of that day, 0 if bottom quartile
- Split: chronological train/val/test with 5-day embargo
- Model: LightGBM binary classifier, early-stopped at iter 446

## Splits
| Split | Period | Rows | Class balance |
|---|---|---|---|
| Train | 2021-07-02 → 2024-11-05 | 171,715 | 0.502 |
| Val | 2024-11-12 → 2025-07-29 | 40,517 | 0.502 |
| Test | 2025-08-05 → 2026-04-17 | 41,780 | 0.502 |

## Test-set performance
| Metric | Value |
|---|---|
| Accuracy | 0.5802 (vs 0.5 random) |
| AUC | 0.6077 (vs 0.5 random) |
| Mean 5d return — long top decile | +0.03159 |
| Mean 5d return — short bottom decile (implied) | -0.02323 |
| Long/short decile spread per 5d | **+0.05482** |
| Equal-weight baseline (mean of all labeled) | +0.00620 |

## Feature importance (LightGBM split-count)
| feature        |   importance |
|:---------------|-------------:|
| ret_1d         |          969 |
| sma_ratio      |          851 |
| macd_diff      |          821 |
| atr_pct_rank   |          714 |
| sma_ratio_rank |          689 |
| vol_z          |          625 |
| atr_pct        |          578 |
| macd_diff_rank |          533 |
| bb_pos_rank    |          510 |
| ret_20d        |          503 |
| ret_1d_rank    |          477 |
| ret_5d         |          462 |
| ret_20d_rank   |          458 |
| bb_pos         |          449 |
| rsi_14         |          447 |
| ret_5d_rank    |          373 |
| vol_z_rank     |          340 |
| rsi_14_rank    |          313 |

## Honest interpretation

The model crosses the minimum thresholds (AUC > 0.55 AND long/short decile
spread > 0.5% per 5-day trade). It's a **real but small edge** on this 1-year
sample. Important caveats:

1. **Test set is small** (41,780 stock-days, ~181 trading days). Confidence intervals on
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
