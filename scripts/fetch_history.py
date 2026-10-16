"""Download (and cache) the 07:00-09:00 UTC option-trade window for every day in a date range.

Usage: python scripts/fetch_history.py BTC 2025-10-16 2026-10-15
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovl import history  # noqa: E402

ccy, start, end = sys.argv[1], pd.Timestamp(sys.argv[2], tz="UTC"), pd.Timestamp(sys.argv[3], tz="UTC")
days = pd.date_range(start, end, freq="D")
for i, d in enumerate(days):
    t = history.window_trades(ccy, d, ROOT / "data" / "history")
    if i % 20 == 0 or i == len(days) - 1:
        print(f"{d:%Y-%m-%d}: {len(t)} trades ({i + 1}/{len(days)})", flush=True)
