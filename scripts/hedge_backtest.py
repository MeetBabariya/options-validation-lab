"""Build the one-day hedging pairs (model deltas on day d, option and forward on day d+1) for a date range.

Usage: python scripts/hedge_backtest.py BTC 2025-10-16 2026-10-15 [--refit]
Needs the cached trade windows from scripts/fetch_history.py; analyse with scripts/analyze_hedging.py.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovl import hedging, history  # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "BTC"
HIST = ROOT / "data" / "history"


def load_panel(day):
    f = HIST / CCY / f"{day:%Y-%m-%d}.csv.gz"
    return history.panel(pd.read_csv(f)) if f.exists() else pd.DataFrame()


def one_day(day):
    try:
        p0, p1 = load_panel(day), load_panel(day + pd.Timedelta(days=1))
        if p0.empty or p1.empty:
            return None
        d = hedging.model_deltas(p0)
        if d.empty:
            return None
        pr = hedging.pairs(d, p1)
        pr["day"] = day
        return pr.drop(columns=["heston_params"]).assign(
            **{f"h_{n}": v for n, v in zip(("v0", "kappa", "theta", "xi", "rho"), d["heston_params"].iloc[0])})
    except Exception as e:  # one bad day must not kill a year-long run
        print(f"{day:%Y-%m-%d} failed: {e}", flush=True)
        return None


if __name__ == "__main__":
    start, end = pd.Timestamp(sys.argv[2], tz="UTC"), pd.Timestamp(sys.argv[3], tz="UTC")
    days = list(pd.date_range(start, end - pd.Timedelta(days=1), freq="D"))
    out = ROOT / "reports" / "hedging"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"{CCY}_pairs.csv.gz"
    if cache.exists() and "--refit" not in sys.argv:
        df = pd.read_csv(cache, parse_dates=["ts", "ts_1", "expiry", "day"])
    else:
        res = []
        with Pool(8) as pool:
            for i, r in enumerate(pool.imap(one_day, days, chunksize=2)):
                res.append(r)
                if (i + 1) % 30 == 0 or i + 1 == len(days):
                    print(f"  {i + 1}/{len(days)} days done", flush=True)
        df = pd.concat([r for r in res if r is not None], ignore_index=True)
        df.to_csv(cache, index=False)
    print(f"{df['day'].nunique()} days, {len(df):,} option pairs saved to {cache.relative_to(ROOT)}")
    print("next: python scripts/analyze_hedging.py " + CCY)
