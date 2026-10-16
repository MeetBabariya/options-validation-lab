"""Fit Bates on every backtest day and merge its delta into the cached hedging pairs.

Usage: python scripts/add_bates.py BTC
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovl import hedging, history  # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "BTC"
PAIRS = ROOT / "reports" / "hedging" / f"{CCY}_pairs.csv.gz"


def one_day(args):
    day, hstart = args
    try:
        p = history.panel(pd.read_csv(ROOT / "data" / "history" / CCY / f"{day:%Y-%m-%d}.csv.gz"))
        b = hedging.bates_deltas(p, heston_start=hstart)
        b["day"] = day
        return b
    except Exception as e:
        print(f"{day:%Y-%m-%d} failed: {e}", flush=True)
        return None


if __name__ == "__main__":
    pairs = pd.read_csv(PAIRS, parse_dates=["ts", "ts_1", "expiry", "day"])
    pairs = pairs.drop(columns=[c for c in pairs if c == "delta_bates" or c.startswith("b_")])
    hs = pairs.groupby("day")[["h_v0", "h_kappa", "h_theta", "h_xi", "h_rho"]].first()
    jobs = [(d, tuple(r)) for d, r in hs.iterrows()]
    res = []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap(one_day, jobs, chunksize=2)):
            res.append(r)
            if (i + 1) % 30 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} days done", flush=True)
    b = pd.concat([r for r in res if r is not None], ignore_index=True)
    out = pairs.merge(b, on=["day", "instrument_name"], how="left")
    print(f"Bates delta for {out['delta_bates'].notna().sum():,} of {len(out):,} pairs")
    out.to_csv(PAIRS, index=False)
