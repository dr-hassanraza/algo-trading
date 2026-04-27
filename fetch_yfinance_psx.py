"""
yfinance PSX bulk OHLCV fetcher (5y history).

Pulls 5 years of daily OHLCV for all PSX equity symbols via Yahoo Finance
using the .KA exchange suffix. Free, unofficial — Yahoo can break this
without notice. EODHD remains as a fallback in `data_cache/ohlcv/` if
this directory has no entry for a symbol.

Output:  data_cache/ohlcv_yf/{TICKER}.parquet
         data_cache/yf_fetch_summary.parquet

Run:  python3 fetch_yfinance_psx.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from all_psx_tickers import STOCK_SYMBOLS_ONLY

CACHE_DIR = Path("data_cache/ohlcv_yf")
SUMMARY_PATH = Path("data_cache/yf_fetch_summary.parquet")
SUFFIX = ".KA"
PERIOD = "5y"


def fetch_one(symbol: str) -> tuple[pd.DataFrame | None, str]:
    """Fetch one symbol from Yahoo. Returns (df, status)."""
    yf_sym = f"{symbol}{SUFFIX}"
    try:
        # auto_adjust=True applies splits/dividends so 'Close' is total-return adjusted.
        # actions=False drops dividend/split columns we don't need here.
        hist = yf.Ticker(yf_sym).history(period=PERIOD, auto_adjust=True, actions=False)
    except Exception as e:
        return None, f"error:{type(e).__name__}"
    if hist is None or hist.empty:
        return None, "empty"
    # Normalize to lowercase columns to match the EODHD parquet schema
    hist = hist.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    # Drop tz info to match EODHD parquet style (which used naive datetime index)
    hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
    hist.index.name = "date"
    return hist[["open", "high", "low", "close", "volume"]], "ok"


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(set(STOCK_SYMBOLS_ONLY))
    print(f"Fetching {len(symbols)} PSX symbols from Yahoo Finance ({PERIOD} period, suffix {SUFFIX})")
    print(f"Output: {CACHE_DIR}/")

    summary_rows = []
    for sym in tqdm(symbols, desc="yfinance pull"):
        out_path = CACHE_DIR / f"{sym}.parquet"
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                summary_rows.append({"symbol": sym, "status": "cached", "rows": len(existing),
                                     "first": existing.index.min(), "last": existing.index.max()})
                continue
            except Exception:
                pass

        df, status = fetch_one(sym)
        if df is not None:
            df.to_parquet(out_path)
            summary_rows.append({"symbol": sym, "status": status, "rows": len(df),
                                 "first": df.index.min(), "last": df.index.max()})
        else:
            summary_rows.append({"symbol": sym, "status": status, "rows": 0,
                                 "first": pd.NaT, "last": pd.NaT})
        # Yahoo doesn't publish a strict rate limit; be polite.
        time.sleep(0.05)

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(SUMMARY_PATH)

    print("\n=== Summary ===")
    print(summary["status"].value_counts().to_string())
    ok = summary[summary["status"].isin(["ok", "cached"])]
    print(f"\nUsable symbols: {len(ok)} / {len(symbols)}")
    if not ok.empty:
        print(f"Median rows per symbol: {int(ok['rows'].median())}")
        print(f"Min/Max rows: {int(ok['rows'].min())} / {int(ok['rows'].max())}")
        print(f"Date range coverage: {ok['first'].min().date()} → {ok['last'].max().date()}")
    print(f"\nSummary written to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
