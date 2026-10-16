"""Rebuild past option-chain states from Deribit's free trade history.

Every trade record carries the traded instrument's mark price, Deribit's IV for the trade and the index price.
Deribit's IV is computed on the forward of the expiry, so each trade pins down that forward (black.implied_forward).
Per daily window we take, for every instrument, its latest trade: mark price (coin), forward and time. Marks come
from Deribit's own smooth smile, so they are clean inputs for calibration, but they are not executable prices.
"""
import json
import ssl
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import black

TRADES = ("https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
          "?currency={ccy}&kind=option&start_timestamp={start}&end_timestamp={end}&count=1000"
          "&include_old=true&sorting=asc")
YEAR_S = 365 * 86400


def _get(url, retries=5):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=ctx) as r:
                return json.load(r)["result"]
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def fetch_trades(ccy, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s, e = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    out = []
    while True:
        r = _get(TRADES.format(ccy=ccy, start=s, end=e))
        out += r["trades"]
        if not r["has_more"] or not r["trades"]:
            break
        s = r["trades"][-1]["timestamp"] + 1
    cols = ["timestamp", "instrument_name", "price", "mark_price", "iv", "index_price", "amount"]
    return pd.DataFrame(out, columns=cols).drop_duplicates()


def window_trades(ccy, day: pd.Timestamp, cache_dir: Path, start_h=7, end_h=9) -> pd.DataFrame:
    """Trades in [day start_h:00, day end_h:00) UTC, cached as one gzipped CSV per day."""
    path = Path(cache_dir) / ccy / f"{day:%Y-%m-%d}.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    t = fetch_trades(ccy, day + pd.Timedelta(hours=start_h), day + pd.Timedelta(hours=end_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path, index=False)
    return t


def panel(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per instrument: its latest trade in the window, with forward, mark IV and log-moneyness."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    p = t["instrument_name"].str.split("-", expand=True)
    t["expiry"] = pd.to_datetime(p[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)
    t["K"] = p[2].astype(float)
    t["is_call"] = p[3].eq("C")
    t["ts"] = pd.to_datetime(t["timestamp"], unit="ms", utc=True)
    t["T"] = (t["expiry"] - t["ts"]).dt.total_seconds() / YEAR_S
    t = t[t["T"] > 1.0 / 365]
    # forward per expiry: annualised basis b = ln(F/S)/T from near-the-money trades with enough premium for precision
    near = t[(np.log(t["K"] / t["index_price"]).abs() < 0.1) & (t["price"] >= 0.002) & (t["iv"] > 0)]
    F = black.implied_forward(near["price"], near["K"], near["T"], near["iv"] / 100, near["is_call"], near["index_price"])
    basis = pd.Series(np.log(F / near["index_price"].values) / near["T"].values, index=near.index)
    b = basis.groupby(near["expiry"]).median()
    t = t[t["expiry"].isin(b.index)]
    last = t.sort_values("ts").groupby("instrument_name").tail(1).copy()
    last["F"] = last["index_price"] * np.exp(last["expiry"].map(b) * last["T"])
    last["k"] = np.log(last["K"] / last["F"])
    last["V"] = last["mark_price"] * last["F"]  # USD value at the trade time
    last["mark_iv"] = black.implied_vol(last["V"], last["F"], last["K"], last["T"], last["is_call"])
    cols = ["instrument_name", "ts", "expiry", "T", "K", "is_call", "index_price", "F", "k", "V", "mark_iv"]
    return last[cols].dropna(subset=["mark_iv"]).reset_index(drop=True)
