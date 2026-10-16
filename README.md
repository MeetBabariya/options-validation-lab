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
- [x] **Step 2: model calibration.** Black-Scholes, SABR and Heston are calibrated to the same chains as SVI and SSVI.
  Heston is priced with the COS method, which matches the published Fang & Oosterlee reference price to 1e-6. Each
  calibration is checked on synthetic data, where it must recover the known parameters, and with a bootstrap that
  redraws every quote inside its bid-ask spread. Parameters are saved for every snapshot, so their stability over
  time builds up day by day.
- [x] **Step 3: hedging backtest.** One year of BTC option-chain states is rebuilt from Deribit's free trade history,
  and six deltas are compared on 1-day hedges out of sample, following Hull & White (2017).
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

## Model comparison (BTC, 16 Oct 2026, 284 quotes, 2 to 343 days)

| Model | Parameters | Median IV error (vol points) | Quotes priced inside bid-ask |
|---|---|---|---|
| Black-Scholes | 1 per expiry | 3.07 | 11% |
| SVI | 5 per expiry | 0.15 | 100% |
| SABR (beta = 1) | 3 per expiry | 0.41 | 79% |
| SSVI | 3 global + ATM curve | 1.12 | 53% |
| Heston | 5 global | 1.50 | 23% |

![models](reports/BTC_models.png)

**Validation findings**

1. **Heston needs implausible dynamics and still misses.** The best fit has mean reversion kappa = 57 (variance
   half-life of about 4.5 days) and vol-of-vol xi = 8.6 (865%), with a Feller ratio of 0.26. Even so, it misprices
   7-day at-the-money options by about 3.5 vol points and leaves 1.5 vol points of error at both ends of the term
   structure. With kappa capped at 30 the fit only moves error from short to long maturities.
2. **Stable is not the same as correct.** Redrawing every quote inside its bid-ask spread moves the Heston parameters
   by only 1 to 2% (30 refits). The data pins them down tightly, so the misfit is model risk, not estimation noise.
   Pure diffusion cannot produce steep short-dated crypto smiles; that calls for jumps (Bates) or a per-expiry
   model.
3. **SABR is the practical choice per expiry.** Three parameters fit within the spread for 79% of quotes. Its
   vol-of-vol nu falls with maturity roughly like 1/sqrt(T) (nu * sqrt(T) is about 0.6 to 0.76 from 2 to 343 days).
   Only the correlation rho of the 2 to 3 day expiries is poorly identified (10 to 12% variation in the bootstrap).
4. **SVI is the best marking tool, but not a dynamic model.** It fits every quote inside the spread, but it describes
   one day's smile and says nothing about how the smile moves. Step 3's hedging test decides which model is actually
   useful for risk.

## Which delta hedges best? (BTC, Jan to Oct 2026, out of sample)

**Data.** There is no free history of full option chains, so I rebuilt one. Every Deribit trade record carries the
instrument's mark price, the trade's implied vol and the index price. Deribit computes that IV on the forward of the
expiry, so each trade pins down the forward: one equation, one unknown, with a unique root. The median over
near-the-money trades gives each expiry's forward. Checked against live forwards, the error is under 0.5 basis points
of the forward. From a 07:00 to 09:00 UTC window each day (365 days, 7 MB) I take every instrument's latest mark.

**Test.** For every out-of-the-money option seen on day d and again about 24 hours later, the hedge error is
`dV - delta * dF`, with the hedge in the forward of the same expiry. The score is Hull & White's gain,
`1 - SSE(model) / SSE(BS)`. Intervals come from a bootstrap over whole days, because options on the same day share the
same BTC move. The empirical deltas are fitted only on data before each test month. In total: 287 test days and
18,555 option-days.

| Delta | Assumption | Gain vs BS | 95% interval | Days beating BS |
|---|---|---|---|---|
| Black-Scholes (sticky strike) | each option keeps its own IV | 0 | | |
| Sticky moneyness (SVI) | smile moves with the forward | **-12.4%** | -24.7 to -4.4 | 45% |
| Heston | stochastic vol, calibrated daily | **-15.1%** | -27.8 to -6.7 | 42% |
| SABR, Bartlett delta | vol-of-vol co-moves with spot | -2.8% | -15.3 to +15.6 | 55% |
| Min-variance (Hull & White) | regression on past months | -12.2% | -34.6 to +15.1 | 44% |
| Adaptive spot-vol (30 days) | IV slope re-estimated monthly | **+8.5%** | -0.7 to +15.4 | 46% |

![hedging](reports/BTC_hedging.png)

**Findings**

1. **The plain sticky-strike delta is hard to beat on BTC.** Sticky moneyness and Heston are significantly worse.
   Both assume IV rises when BTC rises (+0.31 and +0.33 vol per 100% move). In fact IV fell, by 0.21 vol per 100%
   move over the year.
2. **The spot-vol relationship flips sign between months,** from a correlation of -0.36 in February to +0.60 in
   August. Deltas built for a negative correlation (Bartlett, min-variance) win by 25 to 39% when it is negative and
   lose by 23 to 52% when it turns positive. Sticky moneyness and Heston do the mirror image. No fixed smile model
   wins consistently.
3. **Adapting is the only thing that helps overall,** and only a little. Re-estimating the spot-vol slope from the
   last 30 days gives the best result, +8.5%, but its interval touches zero. The median error does not improve and it
   beats BS on only 46% of days, so the gain comes from a few large-move days. It is a lead, not yet a production
   hedge.
4. **Hedging errors are heavy-tailed.** The worst 1% of option-days make up 45% of the squared error, so any single
   ranking can turn on a handful of crash days. That is why I also report day-level and vega-scaled results
   (`reports/hedging/`).
5. **Daily Heston calibration is not usable for risk.** The median kappa is 47, and on 15% of days it hits the
   upper bound of 200. This confirms step 2's finding over a full year.

**Limitations.** Marks are Deribit's model prices, not executable bids and offers. The test uses daily rebalancing
and ignores transaction costs. Option and forward are measured at the instrument's last trade in each window.
Only options that trade on both days are included, which biases the sample towards liquid strikes.

## Data choices

- Only out-of-the-money options with a two-sided market are fitted, and only those with |delta| of at least 5%.
  Deep-wing quotes trade at a few ticks, so their implied vols are mostly rounding noise.
- Fit weights are 1/(bid-ask spread in IV)², so tight markets count more.
- The SVI width parameter has a floor of a quarter of one at-the-money standard deviation. Without it, the
  short-dated fits collapse into a V shape that only fits quote noise.

## Run

```bash
pip install -r requirements.txt
python scripts/build_surface.py BTC      # snapshot, fit, arbitrage checks, reports/BTC_smiles.png
python scripts/calibrate_models.py BTC   # BS / SVI / SSVI / SABR / Heston on the latest snapshot
python scripts/fetch_history.py BTC 2025-10-16 2026-10-15   # one year of daily trade windows (cached)
python scripts/hedge_backtest.py BTC 2025-10-16 2026-10-15  # model deltas and 1-day pairs (8 processes)
python scripts/analyze_hedging.py BTC                       # gains, robust views, monthly regimes, chart
pytest -q
```

Each run saves a gzipped snapshot in `data/snapshots/`. A GitHub Actions job (`.github/workflows/snapshot.yml`)
saves BTC and ETH snapshots every day at 08:00 UTC, refits the surface and recalibrates all models
(`reports/params/`). This builds the history for the hedging
backtest.
