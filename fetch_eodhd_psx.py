"""
EODHD PSX bulk OHLCV fetcher.

Pulls 1y of daily OHLCV (free-tier limit) for all PSX equity symbols and
writes one Parquet file per symbol to data_cache/ohlcv/.

Run:  python3 fetch_eodhd_psx.py
"""
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from all_psx_tickers import STOCK_SYMBOLS_ONLY

CACHE_DIR = Path("data_cache/ohlcv")
SUMMARY_PATH = Path("data_cache/fetch_summary.parquet")
EXCHANGE = "KAR"
BASE_URL = "https://eodhd.com/api/eod/{symbol}"
RATE_LIMIT_SLEEP = 0.15  # be polite; ~6/sec, well under EODHD limits
TIMEOUT = 30


def fetch_one(symbol: str, api_key: str, frm: str, to: str) -> tuple[pd.DataFrame | None, str]:
    """Fetch one symbol. Returns (df, status). status in {ok, empty, http_<code>, error}."""
    url = BASE_URL.format(symbol=f"{symbol}.{EXCHANGE}")
    params = {"api_token": api_key, "fmt": "json", "from": frm, "to": to}
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        return None, f"error:{type(e).__name__}"
    if r.status_code != 200:
        return None, f"http_{r.status_code}"
    try:
        data = r.json()
    except ValueError:
        return None, "error:bad_json"
    if not isinstance(data, list):
        return None, "error:unexpected_shape"
    # Drop the trailing warning row that has no OHLCV
    rows = [d for d in data if "open" in d and "close" in d]
    if not rows:
        return None, "empty"
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df, "ok"


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        print("ERROR: EODHD_API_KEY missing from environment / .env", file=sys.stderr)
        return 1

    today = date.today()
    frm = (today - timedelta(days=400)).isoformat()  # ~1y, ask slightly more to maximize
    to = today.isoformat()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(set(STOCK_SYMBOLS_ONLY))
    print(f"Fetching {len(symbols)} PSX symbols ({frm} → {to})")
    print(f"Output: {CACHE_DIR}/")

    summary_rows = []
    for sym in tqdm(symbols, desc="EODHD pull"):
        out_path = CACHE_DIR / f"{sym}.parquet"
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                summary_rows.append({"symbol": sym, "status": "cached", "rows": len(existing),
                                     "first": existing.index.min(), "last": existing.index.max()})
                continue
            except Exception:
                pass  # corrupt cache, refetch

        df, status = fetch_one(sym, api_key, frm, to)
        if df is not None:
            df.to_parquet(out_path)
            summary_rows.append({"symbol": sym, "status": status, "rows": len(df),
                                 "first": df.index.min(), "last": df.index.max()})
        else:
            summary_rows.append({"symbol": sym, "status": status, "rows": 0,
                                 "first": pd.NaT, "last": pd.NaT})
        time.sleep(RATE_LIMIT_SLEEP)

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(SUMMARY_PATH)

    print("\n=== Summary ===")
    print(summary["status"].value_counts().to_string())
    ok = summary[summary["status"].isin(["ok", "cached"])]
    print(f"\nUsable symbols: {len(ok)} / {len(symbols)}")
    if not ok.empty:
        print(f"Median rows per symbol: {int(ok['rows'].median())}")
        print(f"Min/Max rows: {int(ok['rows'].min())} / {int(ok['rows'].max())}")
    print(f"\nFull summary written to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
