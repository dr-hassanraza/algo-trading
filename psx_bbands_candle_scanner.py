#!/usr/bin/env python3
"""
PSX Bollinger + MA44 + Candle Scanner (Tickers Mode)
====================================================

What this does
--------------
Scans one or more **PSX tickers** (you pass them on the command line) and flags
**next‑day BUY candidates** using a robust version of your MA44 + green-candle idea:

Entry rules (all must be true on the latest bar):
  1) Trend:      MA44 slope over last 10 trading days > 0 (uptrend)
  2) Location:   Close > MA44
  3) Bands:      Bollinger %B in [0.35, 0.85] (avoid extremes)
  4) Candle:     Green candle AND real body ≥ 40% of day range
  5) Pullback:   Low within 2% of MA44 in the last 3 sessions

Outputs
-------
  scan_reports/YYYY-MM-DD/diagnostics.csv  (per‑ticker snapshot + reasons)
  scan_reports/YYYY-MM-DD/candidates.csv   (filtered BUY candidates only)
  Optional charts with --charts (PNG per ticker)

Run it (EODHD only; no CSV, no Yahoo):
  pip install pandas numpy requests matplotlib
  export EODHD_API_KEY="YOUR_KEY"   # https://eodhd.com/
  python psx_bbands_candle_scanner.py --tickers UBL.KAR MCB.KAR OGDC.KAR --asof today --days 260 --charts

Notes
-----
• Symbols should include the **.KAR** suffix for EODHD (e.g., UBL.KAR). If you pass
  a bare name like UBL, the script will automatically append .KAR.
• Educational only. No financial advice.
"""

import os, argparse, pathlib, datetime as dt

# Load environment variables from .env file
def load_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except FileNotFoundError:
        pass

load_env()
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import requests
import math

# -----------------------------
# Globals
# -----------------------------
OUT = pathlib.Path("scan_reports")
TZ = dt.timezone(dt.timedelta(hours=5))  # Asia/Karachi for date stamping
TODAY = dt.datetime.now(tz=TZ).date()

# -----------------------------
# Data fetcher (EODHD)
# -----------------------------
class EODHDFetcher:
    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise RuntimeError("EODHD_API_KEY not set. export EODHD_API_KEY=YOUR_KEY")
        self.key = api_key

    def fetch(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        sym = symbol if symbol.upper().endswith('.KAR') else f"{symbol}.KAR"
        url = (
            f"https://eodhd.com/api/eod/{sym}?from={start:%Y-%m-%d}&to={end:%Y-%m-%d}&period=d&fmt=json&api_token={self.key}"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"No data for {sym}")
        df = pd.DataFrame(data)
        df.rename(columns={
            'date':'Date','open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume',
            'adjusted_close':'AdjClose'
        }, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        if 'AdjClose' not in df or df['AdjClose'].isna().all():
            df['AdjClose'] = df['Close']
        for c in ['Open','High','Low','Close','AdjClose','Volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna().sort_values('Date')
        return df[['Date','Open','High','Low','Close','AdjClose','Volume']]

# -----------------------------
# Indicators
# -----------------------------

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()


def bollinger(close: pd.Series, w=20, n=2.0):
    mid = sma(close, w)
    std = close.rolling(w, min_periods=w).std()
    up = mid + n*std
    lo = mid - n*std
    pctb = (close - lo) / (up - lo + 1e-12)
    return mid, up, lo, pctb


def slope(series: pd.Series, win: int = 10) -> pd.Series:
    return (series - series.shift(win)) / win

# -----------------------------
# Signal logic
# -----------------------------

def last_green_candle(df: pd.DataFrame) -> bool:
    oc = df['Close'].iloc[-1] - df['Open'].iloc[-1]
    rng = max(df['High'].iloc[-1] - df['Low'].iloc[-1], 1e-9)
    body = abs(df['Close'].iloc[-1] - df['Open'].iloc[-1])
    return (oc > 0) and (body / rng >= 0.40)


def near_ma44(df: pd.DataFrame, pct: float = 0.02, lookback: int = 3) -> bool:
    sub = df.tail(lookback)
    return any(sub['Low'] <= sub['MA44'] * (1 + pct))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['MA44'] = sma(out['Close'], 44)
    mid, up, lo, pctb = bollinger(out['Close'], 20, 2.0)
    out['BB_mid'] = mid
    out['BB_up'] = up
    out['BB_lo'] = lo
    out['BB_pctB'] = pctb
    out['MA44_slope10'] = slope(out['MA44'], 10)
    out['BBmid_slope5'] = slope(out['BB_mid'], 5)
    out.dropna(inplace=True)
    return out


def signal_row(df: pd.DataFrame) -> Dict:
    latest = df.iloc[-1]
    conds = {
        'trend_ma44_up': latest['MA44_slope10'] > 0,
        'close_gt_ma44': latest['Close'] > latest['MA44'],
        'bb_pctB_midzone': 0.35 <= latest['BB_pctB'] <= 0.85,
        'green_candle_body40': last_green_candle(df),
        'bb_mid_slope_up': latest['BBmid_slope5'] > 0,
        'near_ma44': near_ma44(df, pct=0.02, lookback=3)
    }
    buy = all(conds.values())
    return {
        'buy_next_day': int(buy),
        'reasons': {k: bool(v) for k, v in conds.items()},
        'snapshot': {
            'close': float(latest['Close']),
            'ma44': float(latest['MA44']),
            'bb_pctB': float(latest['BB_pctB']),
            'ma44_slope10': float(latest['MA44_slope10']),
            'bbmid_slope5': float(latest['BBmid_slope5'])
        }
    }

# -----------------------------
# Plotting (optional)
# -----------------------------

def plot_chart(df: pd.DataFrame, sym: str, outdir: pathlib.Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Warning: matplotlib not available for charting: {e}")
        return
    plt.figure(figsize=(11,5))
    plt.plot(df['Date'], df['Close'], label='Close')
    plt.plot(df['Date'], df['MA44'], label='MA44')
    plt.plot(df['Date'], df['BB_mid'], label='BB mid')
    plt.fill_between(df['Date'], df['BB_lo'], df['BB_up'], alpha=0.15, label='BB(20,2)')
    # mark last candle
    plt.scatter([df['Date'].iloc[-1]], [df['Close'].iloc[-1]], marker='o', s=40)
    plt.title(f"{sym} — MA44 + Bollinger + Candle")
    plt.legend()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / f"{sym.replace('.','_')}.png", dpi=130)
    plt.close()

# -----------------------------
# Runner
# -----------------------------

def parse_asof(asof: str) -> dt.date:
    if asof.lower() == 'today':
        return TODAY
    return dt.datetime.strptime(asof, '%Y-%m-%d').date()


def scan(symbols: List[str], asof: dt.date, days: int = 260, make_charts: bool = False, return_dataframes: bool = False) -> pd.DataFrame:
    end = asof
    start = end - dt.timedelta(days=days+40)
    fetcher = EODHDFetcher(os.getenv('EODHD_API_KEY'))

    rows = []
    charts_df_list = []
    dataframes = {}  # Store raw dataframes for chatbot use
    for raw_sym in symbols:
        sym = raw_sym if raw_sym.upper().endswith('.KAR') else f"{raw_sym}.KAR"
        try:
            bars = fetcher.fetch(sym, start, end)
            ind = compute_indicators(bars)
            if ind.empty:
                rows.append({'symbol': sym, 'date': str(end), 'error': 'insufficient data'})
                continue
            sig = signal_row(ind)
            rows.append({
                'symbol': sym,
                'date': str(ind['Date'].iloc[-1].date()),
                'buy_next_day': sig['buy_next_day'],
                **sig['reasons'],
                **sig['snapshot']
            })
            if return_dataframes:
                dataframes[sym] = ind
            if make_charts:
                charts_df_list.append((sym, ind))
        except Exception as e:
            rows.append({'symbol': sym, 'date': str(end), 'error': str(e)})

    res = pd.DataFrame(rows)

    # charts
    if make_charts and charts_df_list:
        chartdir = OUT / f"{asof:%Y-%m-%d}" / "charts"
        for sym, ind in charts_df_list:
            plot_chart(ind, sym, chartdir)

    if return_dataframes:
        return res, dataframes
    return res


def main():
    ap = argparse.ArgumentParser(description='PSX Bollinger + MA44 + Candle scanner (tickers mode)')
    ap.add_argument('--tickers', nargs='+', required=True, help='List of PSX tickers (use .KAR suffix, e.g., UBL.KAR)')
    ap.add_argument('--asof', default='today', help='YYYY-MM-DD or "today" (default)')
    ap.add_argument('--days', type=int, default=260, help='Lookback window in calendar days (~1Y)')
    ap.add_argument('--charts', action='store_true', help='Save per-ticker PNG charts')
    args = ap.parse_args()

    asof = parse_asof(args.asof)
    OUTDIR = OUT / f"{asof:%Y-%m-%d}"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    res = scan(args.tickers, asof=asof, days=args.days, make_charts=args.charts)
    res.to_csv(OUTDIR / 'diagnostics.csv', index=False)

    cands = res[(res['buy_next_day']==1) & (~res.get('error', pd.Series(dtype=object)).notna())]
    cands[['symbol','date','buy_next_day','close','ma44','bb_pctB']].to_csv(OUTDIR / 'candidates.csv', index=False)

    print("\nScan done. Candidates (next-day buy intent):\n")
    if not cands.empty:
        print(cands[['symbol','date','close','ma44','bb_pctB']].to_string(index=False))
    else:
        print("(none)")

if __name__ == '__main__':
    main()
