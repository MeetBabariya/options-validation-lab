"""Snapshot Deribit option chains (public API, no key) into a tidy quote table."""
import json
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import black

API = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={ccy}&kind=option"
YEAR = 365.0 * 24 * 3600 * 1000  # ms


def _get(url):
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=30, context=ctx) as r:
        return json.load(r)["result"]


def fetch(ccy="BTC"):
    """Raw book summaries for every listed option on `ccy`."""
    return pd.DataFrame(_get(API.format(ccy=ccy)))


def parse(raw: pd.DataFrame) -> pd.DataFrame:
    """Instrument names look like BTC-27NOV26-73000-P. Prices are quoted in coin; convert to USD on the forward."""
    df = raw.copy()
    parts = df["instrument_name"].str.split("-", expand=True)
    df["expiry"] = pd.to_datetime(parts[1], format="%d%b%y", utc=True) + pd.Timedelta(hours=8)  # 08:00 UTC
    df["K"] = parts[2].astype(float)
    df["is_call"] = parts[3].eq("C")
    df["ts"] = pd.to_datetime(df["creation_timestamp"], unit="ms", utc=True)
    df["T"] = (df["expiry"] - df["ts"]).dt.total_seconds() * 1000 / YEAR
    df["F"] = df["underlying_price"]
    for side in ("bid", "ask", "mark"):
        df[f"{side}_usd"] = df[f"{side}_price"] * df["F"]
    df["k"] = np.log(df["K"] / df["F"])
    df["deribit_mark_iv"] = df["mark_iv"] / 100  # keep Deribit's own mark IV as a cross-check
    df = df[df["T"] > 1 / 365]  # drop options in their last day: IV is noise there
    for side in ("bid", "ask", "mark"):
        df[f"{side}_iv"] = black.implied_vol(df[f"{side}_usd"], df["F"], df["K"], df["T"], df["is_call"])
    cols = ["instrument_name", "ts", "expiry", "T", "F", "K", "k", "is_call", "bid_usd", "ask_usd", "mark_usd",
            "bid_iv", "ask_iv", "mark_iv", "deribit_mark_iv", "open_interest", "volume_usd"]
    out = df[cols]
    return out.sort_values(["expiry", "K", "is_call"]).reset_index(drop=True)


def otm_quotes(q: pd.DataFrame, min_abs_delta: float = 0.05) -> pd.DataFrame:
    """One quote per strike: the out-of-the-money option (calls above F, puts below), with a two-sided market."""
    otm = q[(q["is_call"] & (q["K"] >= q["F"])) | (~q["is_call"] & (q["K"] < q["F"]))]
    otm = otm[otm["bid_iv"].notna() & otm["ask_iv"].notna() & (otm["ask_iv"] > otm["bid_iv"])].copy()
    otm["mid_iv"] = 0.5 * (otm["bid_iv"] + otm["ask_iv"])
    # Deep wings (|delta| < 5%) trade at a few ticks, so their IVs are mostly rounding noise. Desks fit on 5-95 delta.
    otm["delta"] = black.delta(otm["F"], otm["K"], otm["T"], otm["mid_iv"], otm["is_call"])
    otm = otm[otm["delta"].abs() >= min_abs_delta]
    return otm


def snapshot(ccy="BTC", out_dir="data/snapshots") -> Path:
    q = parse(fetch(ccy))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    path = Path(out_dir) / f"{ccy}_{stamp}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(path, index=False)
    return path
