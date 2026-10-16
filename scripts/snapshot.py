"""Save one Deribit option-chain snapshot per currency. Run daily by GitHub Actions to build the history."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl import data  # noqa: E402

for ccy in sys.argv[1:] or ["BTC", "ETH"]:
    print(data.snapshot(ccy))
