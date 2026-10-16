# Options Model Validation Lab

An independent validation of option-pricing models on live market data. It answers the question a model-validation
team asks: does the model fit the market, is it free of arbitrage, and does it hedge well?

Data: free public Deribit option chains for BTC and ETH, with no API key. One snapshot is about 900 listed options
across 11 expiries.

## Status

- [x] **Step 1: implied-vol surface.** Black-76 implied vols from bid, ask and mark prices. Raw SVI is fitted per
  expiry using the quasi-explicit method (Zeliade 2009), with Lee's moment bound and a Durrleman butterfly-arbitrage
  check. SSVI (Gatheral & Jacquier 2014) is fitted as a globally arbitrage-free benchmark, and calendar arbitrage is
  checked across expiries.
- [ ] Step 2: Black-Scholes, Heston and SABR calibrated to the same chains, with stability of the parameters over time.
- [ ] Step 3: delta-hedging backtest on daily snapshots, to see which model hedges best.
- [ ] Step 4: an automatically generated model-validation report, plus a live dashboard.

## First results (BTC, 16 Oct 2026)

| | SVI per expiry | SSVI (global) |
|---|---|---|
| IV error (RMSE, vol points) | 0.09 to 0.25 | 0.22 to 2.26 |
| Fitted IV inside the bid-ask spread | 92 to 100% | 20 to 85% |
| Butterfly / calendar arbitrage | none found | none, by construction |

SSVI is arbitrage-free by construction, but its single skew parameter misses the steep short-dated BTC put skew by
1 to 2 vol points. That trade-off between flexibility and guaranteed no-arbitrage is the central question of the
validation.

![smiles](reports/BTC_smiles.png)

## Data choices

- Only out-of-the-money options with a two-sided market are fitted, and only those with |delta| of at least 5%.
  Deep-wing quotes trade at a few ticks, so their implied vols are mostly rounding noise.
- Fit weights are 1/(bid-ask spread in IV)², so tight markets count more.
- The SVI width parameter has a floor of a quarter of one at-the-money standard deviation. Without it, the
  short-dated fits collapse into a V shape that only fits quote noise.

## Run

```bash
pip install -r requirements.txt
python scripts/build_surface.py BTC   # snapshot, fit, arbitrage checks, reports/BTC_smiles.png
pytest -q
```

Each run saves a gzipped snapshot in `data/snapshots/`. A GitHub Actions job (`.github/workflows/snapshot.yml`)
saves BTC and ETH snapshots every day at 08:00 UTC and refits the surface. This builds the history for the hedging
backtest.
