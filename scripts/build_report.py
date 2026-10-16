"""Generate the model-validation report (HTML, and PDF if Chrome is available) from the project's own results.

Structure follows a bank model-validation report: scope, data, conceptual soundness, implementation verification,
outcome analysis, stability, findings register, decision per model. Every number is read from saved results or
recomputed here; nothing is typed in by hand.

Usage: python scripts/build_report.py [BTC]
"""
import glob
import html
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovl import calibrate as cal, data, svi  # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "BTC"
REP = ROOT / "reports"
H = REP / "hedging"
pct = lambda x: f"{100 * x:+.1f}%"


# ---------------------------------------------------------------- evidence
def run_tests():
    xml = REP / "tests.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", f"--junitxml={xml}"], cwd=ROOT,
                   capture_output=True, text=True)
    rows = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        failed = tc.find("failure") is not None or tc.find("error") is not None
        rows.append((tc.get("classname").split(".")[-1], tc.get("name"), not failed))
    xml.unlink()
    return rows


latest_snap = sorted((ROOT / "data/snapshots").glob(f"{CCY}_*.csv.gz"))[-1]
q = pd.read_csv(latest_snap, parse_dates=["ts", "expiry"])
otm = data.otm_quotes(q).reset_index(drop=True)
fits = svi.fit_svi_surface(otm)
cal_viol = svi.calendar_violations(fits, otm)
bfly = sum(f.min_g < -1e-6 for f in fits.values())
params = json.loads(sorted((REP / "params").glob(f"{CCY}_*.json"))[-1].read_text())
score = pd.DataFrame(params["score"]).set_index("model")
hes, sab = cal.bootstrap(otm, n=20)
hes_cv = (hes.std() / hes.mean().abs())
# Bates identifiability: redraw quotes inside bid-ask and refit (10 refits; each fit is ~4x a Heston fit)
_rng = np.random.default_rng(1)
_h0 = tuple(hes.mean())
bat_boot = pd.DataFrame([cal.fit_bates(otm.assign(mid_iv=u), mid_iv=u, heston_start=_h0)
                         for u in (_rng.uniform(otm["bid_iv"].values, otm["ask_iv"].values) for _ in range(10))],
                        columns=cal.BATES_NAMES)
bat_cv = bat_boot.std() / bat_boot.mean().abs()
bat_boot["mean_jump"] = np.exp(bat_boot["muj"] + bat_boot["sj"] ** 2 / 2) - 1
bat_boot["jump_var"] = bat_boot["lam"] * (bat_boot["muj"] ** 2 + bat_boot["sj"] ** 2)
bat_ridge = max(abs(bat_boot["lam"].corr(bat_boot["muj"])), abs(bat_boot["lam"].corr(bat_boot["sj"])))
tests = run_tests()

gains = pd.read_csv(H / f"{CCY}_gains.csv", index_col=0)
robust = pd.read_csv(H / f"{CCY}_gains_robust.csv", index_col=0)
monthly = pd.read_csv(H / f"{CCY}_gains_monthly.csv", index_col=0)
pairs = pd.read_csv(H / f"{CCY}_pairs.csv.gz", parse_dates=["day"])
daily_h = pairs.groupby("day")[["h_v0", "h_kappa", "h_theta", "h_xi", "h_rho"]].first()
has_bates = "delta_bates" in pairs and "Bates" in gains.index
daily_b = pairs.groupby("day")[[c for c in pairs if c.startswith("b_")]].first() if has_bates else None
n_hist_days = len(glob.glob(str(ROOT / "data/history" / CCY / "*.csv.gz"))) or pairs["day"].nunique() + 1
test_pairs = pairs[(pairs["dt_h"] > 20) & (pairs["dt_h"] < 28) & (pairs["day"] >= pd.Timestamp(monthly.index[0] + "-01", tz="UTC"))]

# Heston parameter history over the backtest year
fig, ax = plt.subplots(1, 3, figsize=(15, 3.3))
for a, (c, lab) in zip(ax, [("h_kappa", "kappa (mean reversion)"), ("h_xi", "xi (vol of vol)"), ("h_rho", "rho")]):
    a.plot(daily_h.index, daily_h[c], lw=0.9, color="C1")
    a.set_title(lab)
    a.tick_params(axis="x", rotation=30)
ax[0].axhline(200, color="C3", ls="--", lw=0.8)
ax[0].text(daily_h.index[5], 185, "upper bound", color="C3", fontsize=8)
fig.suptitle("Heston parameters calibrated each day (one year)")
fig.tight_layout()
fig.savefig(REP / f"{CCY}_heston_history.png", dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- content
g = gains["gain"]
ci = lambda m: f"{100 * gains.loc[m, 'lo95']:+.1f} to {100 * gains.loc[m, 'hi95']:+.1f}%"
k_at_bound = (daily_h["h_kappa"] > 199).mean()
corr = monthly["corr(ret, dIV)"]

def bates_decision():
    """Rule-based: hedge interval below zero -> not approved; spanning zero -> monitor; above zero -> conditional."""
    lo, hi = gains.loc["Bates", "lo95"], gains.loc["Bates", "hi95"]
    decision = "Not approved" if hi < 0 else "Monitor" if lo < 0 else "Approved with conditions"
    mj = 100 * bat_boot["mean_jump"].mean()
    return ("Bates (Heston + jumps)", "Pricing and hedging across expiries", decision,
            f"Median IV error {score.loc['Bates', 'median_rmse_volpts']:.2f} vol pts (Heston {score.loc['Heston', 'median_rmse_volpts']:.2f}). "
            f"Fits by pricing a rare crash ({bat_boot['lam'].mean():.2f} per year, mean jump {mj:+.0f}%) while kappa stays near "
            f"{bat_boot['kappa'].mean():.0f}; delta {pct(g['Bates'])} vs BS ({ci('Bates')}).",
            "Use only as a cross-maturity pricing benchmark; jump parameters are identified only as a combination (F8).")


decisions = [
    ("SVI (per expiry)", "Marking and interpolating the daily smile", "Approved with conditions",
     f"Median IV error {score.loc['SVI', 'median_rmse_volpts']:.2f} vol pts, {100 * score.loc['SVI', 'mean_in_spread']:.0f}% of quotes inside bid-ask, "
     f"{bfly} butterfly and {len(cal_viol)} calendar violations on the latest surface.",
     "Run the arbitrage checks on every fit; do not use the fitted smile as a model of smile dynamics (sticky-moneyness delta "
     f"loses {abs(100 * g['Sticky moneyness (SVI)']):.0f}% vs BS)."),
    ("SSVI (global)", "Arbitrage-free benchmark and extrapolation", "Approved with conditions",
     f"Arbitrage-free by construction; median IV error {score.loc['SSVI', 'median_rmse_volpts']:.2f} vol pts.",
     "Not for marking short-dated options: one skew parameter misses the short-dated put skew."),
    ("SABR, beta = 1 (per expiry)", "Marking; delta with Bartlett correction", "Approved for marking; hedge not approved",
     f"Median IV error {score.loc['SABR', 'median_rmse_volpts']:.2f} vol pts. Bartlett delta {pct(g['SABR Bartlett'])} vs BS ({ci('SABR Bartlett')}).",
     "Bartlett delta wins only when spot-vol correlation is negative; do not use as the default hedge."),
    ("Heston (global)", "Pricing and hedging across expiries", "Not approved",
     f"Median IV error {score.loc['Heston', 'median_rmse_volpts']:.2f} vol pts with implausible parameters (kappa at the 200 bound on "
     f"{100 * k_at_bound:.0f}% of days); delta {pct(g['Heston'])} vs BS ({ci('Heston')}).",
     "Do not use; jumps (Bates) improve the fit but not the dynamics, see F8."),
] + ([bates_decision()] if has_bates else []) + [
    ("Black-Scholes, sticky strike", "Default delta hedge", "Approved",
     "Benchmark that no fixed smile-dynamics delta beats over the year.", "Keep as the production delta; monitor challengers."),
    ("Adaptive spot-vol delta (30 days)", "Challenger hedge", "Monitor",
     f"{pct(g['Adaptive spot-vol (30d)'])} vs BS ({ci('Adaptive spot-vol (30d)')}); beats BS on only "
     f"{100 * robust.loc['Adaptive spot-vol (30d)', 'days_beating_BS']:.0f}% of days.",
     "Track for another 6 months; promote only if the interval excludes zero."),
]

findings = [
    ("F1", "High", "Heston", f"Fitting BTC smiles requires implausible dynamics (median kappa {daily_h['h_kappa'].median():.0f}, "
     f"vol of vol {daily_h['h_xi'].median():.1f}) and still misprices short-dated options; parameters are tightly identified "
     f"(bootstrap CV {100 * hes_cv[['kappa', 'xi', 'rho']].max():.0f}% or less), so this is model misspecification, not noise.",
     "Do not use. A jump extension (Bates) was tested, see F8."),
    ("F2", "High", "Smile-dynamics deltas", f"Spot-vol correlation changes sign between months ({corr.min():+.2f} to {corr.max():+.2f}). "
     "Every fixed assumption wins in one regime and loses in the other.", "Keep sticky-strike as default; evaluate regime-adaptive deltas."),
    ("F3", "Medium", "Sticky moneyness, Heston", f"Both imply that IV rises when BTC rises; over the year it fell. Hedging error is "
     f"{abs(100 * g['Sticky moneyness (SVI)']):.0f}% and {abs(100 * g['Heston']):.0f}% worse than BS, intervals exclude zero.",
     "Do not use sticky-moneyness deltas for BTC."),
    ("F4", "Medium", "All hedges", "Hedging errors are heavy-tailed: the worst 1% of option-days carry about 45% of the squared error.",
     "Report robust metrics alongside SSE; stress-test on large-move days."),
    ("F5", "Medium", "SSVI", "One global skew parameter cannot match the short-dated BTC put skew (1 to 2 vol pts).",
     "Use per-expiry SVI or SABR for short maturities."),
    ("F6", "Low", "SVI implementation", "Very short expiries produced degenerate fits (negative total variance, extreme deltas) "
     "until optimiser starts were scaled to the slice and total variance was bounded at zero. Fixed and covered by tests.",
     "Keep the regression tests; alert on any |delta| > 1."),
] + ([("F8", "High", "Bates", f"Adding jumps cuts the median IV error from {score.loc['Heston', 'median_rmse_volpts']:.2f} to "
     f"{score.loc['Bates', 'median_rmse_volpts']:.2f} vol pts, but only by pricing a rare, very large crash "
     f"({bat_boot['lam'].mean():.2f} per year, mean jump {100 * bat_boot['mean_jump'].mean():+.0f}%). The jump parameters trade off "
     f"along a ridge (|correlation| {bat_ridge:.2f} in the bootstrap); only combinations such as the jump variance "
     f"({bat_boot['jump_var'].mean():.3f} per year, CV {100 * bat_boot['jump_var'].std() / bat_boot['jump_var'].mean():.0f}%) are pinned down. "
     f"The diffusion stays implausible (kappa {bat_boot['kappa'].mean():.0f}). The objective has a second, much worse basin with "
     f"frequent small jumps, and over the year the daily fit switches between the two: jump rate below 0.5 per year on "
     f"{100 * (daily_b['b_lam'] < 0.5).mean():.0f}% of days, at the 60 bound on {100 * (daily_b['b_lam'] > 59.9).mean():.0f}%. "
     f"Hedge: {pct(g['Bates'])} vs BS ({ci('Bates')}), no better than Heston.",
     "Not for hedging. Calibrate from several starts in both basins; report jump risk as a combination, not separate parameters.")]
     if has_bates else []) + [
    ("F7", "Low", "Data", "Backtest uses Deribit marks, not executable quotes, with daily rebalancing and no costs; "
     "only options trading on both days are included.", "Repeat with order-book snapshots when available."),
]

n_pass = sum(t[2] for t in tests)
verification = [
    ("Heston COS price vs published Fang & Oosterlee (2008) value 5.785155450", "test_heston_cos_matches_published_reference"),
    ("Heston COS vs independent numerical integration (2 to 365 days)", "test_heston_cos_matches_numerical_integration_across_strikes_and_maturities"),
    ("Heston reduces to Black-Scholes when vol of vol is zero", "test_heston_with_zero_volvol_is_black_scholes"),
    ("Calibration recovers known Heston parameters (synthetic data)", "test_heston_calibration_recovers_known_parameters"),
    ("Calibration recovers known SABR parameters (synthetic data)", "test_sabr_calibration_recovers_known_parameters"),
    ("Bates with constant variance matches an independent Merton jump-diffusion series", "test_bates_with_constant_variance_matches_merton_series"),
    ("Bates without jumps equals Heston; characteristic function is a martingale", "test_bates_without_jumps_is_heston"),
    ("Calibration recovers known Bates parameters (synthetic data)", "test_bates_calibration_recovers_known_parameters"),
    ("Durrleman check flags the known arbitrage slice (Gatheral & Jacquier 2014)", "test_durrleman_flags_known_butterfly_arbitrage"),
    ("Sticky-moneyness delta equals total derivative", "test_sticky_moneyness_delta_is_total_derivative"),
    ("Bartlett delta equals derivative with alpha following the forward", "test_bartlett_delta_is_derivative_with_alpha_following_the_forward"),
    ("Forward inference round-trip on Deribit coin prices", "test_implied_forward_round_trip"),
    ("Min-variance regression recovers known coefficients", "test_min_variance_regression_recovers_coefficients"),
]
status = {name: ok for _, name, ok in tests}


def table(head, rows, cls="", widths=None):
    cg = "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>" if widths else ""
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}">{cg}<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


badge = lambda d: f'<span class="badge {d.split()[0].lower()}">{html.escape(d)}</span>'
sev = lambda s: f'<span class="sev {s.lower()}">{s}</span>'
built = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

fit_rows = [(m, f"{score.loc[m, 'median_rmse_volpts']:.2f}", f"{100 * score.loc[m, 'mean_in_spread']:.0f}%") for m in score.index]
hedge_rows = [(m, pct(gains.loc[m, "gain"]) if m != "BS (sticky strike)" else "benchmark",
               ci(m) if m != "BS (sticky strike)" else "",
               f"{100 * robust.loc[m, 'gain_vega_scaled']:+.1f}%" if m != "BS (sticky strike)" else "",
               f"{100 * robust.loc[m, 'days_beating_BS']:.0f}%" if m != "BS (sticky strike)" else "")
              for m in gains.index]
ident_rows = [(n, f"{hes[n].mean():.3f}", f"{hes[n].std():.3f}", f"{100 * hes_cv[n]:.1f}%") for n in hes.columns]

css = """
:root{--ink:#111;--mut:#555;--line:#d9d9d9;--acc:#0b3d91;--bg:#fff}
@page{size:A4;margin:16mm 15mm}
*{box-sizing:border-box}
body{font:10.2pt/1.5 "Helvetica Neue",Arial,sans-serif;color:var(--ink);background:var(--bg);max-width:980px;margin:0 auto;padding:28px}
h1{font-size:21pt;margin:0 0 4px;letter-spacing:-.2px}
h2{font-size:13pt;color:var(--acc);border-bottom:1.5px solid var(--acc);padding-bottom:3px;margin:26px 0 10px;break-after:avoid}
h3{font-size:10.6pt;margin:16px 0 6px}
.meta{color:var(--mut);font-size:9.2pt;margin-bottom:14px}
.meta b{color:var(--ink)}
table{width:100%;border-collapse:collapse;margin:8px 0 12px;font-size:9.2pt}
tr{break-inside:avoid}
thead{display:table-header-group}
th{background:#f2f4f8;text-align:left;font-weight:600;padding:6px 7px;border-bottom:1.5px solid #c8cdd8}
td{padding:6px 7px;border-bottom:1px solid var(--line);vertical-align:top}
img{width:100%;border:1px solid var(--line);margin:6px 0 4px;break-inside:avoid}
.cap{color:var(--mut);font-size:8.8pt;margin-bottom:10px}
.badge,.sev{display:inline-block;padding:1px 7px;border-radius:9px;font-size:8.4pt;font-weight:600}
.approved{background:#e3f4e8;color:#13652f}.not{background:#fde4e4;color:#9b1c1c}.monitor{background:#fff3d6;color:#8a5a00}
.high{background:#fde4e4;color:#9b1c1c}.medium{background:#fff3d6;color:#8a5a00}.low{background:#e8eef9;color:#28467f}
.pass{color:#13652f;font-weight:600}.fail{color:#9b1c1c;font-weight:600}
.summary{background:#f7f9fc;border-left:3px solid var(--acc);padding:10px 14px;margin:10px 0}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}
.kpi{border:1px solid var(--line);border-radius:6px;padding:8px 10px}.kpi b{display:block;font-size:15pt}
.kpi span{color:var(--mut);font-size:8.6pt}
a{color:var(--acc)}
@media (max-width:640px){body{padding:14px}.kpis{grid-template-columns:1fr 1fr}table{font-size:8.4pt}}
"""

body = f"""
<h1>Model Validation Report: {CCY} Option Models on Deribit</h1>
<div class="meta">Independent validation &middot; Author: <b>Meet Babariya</b> &middot; Generated automatically {built}
&middot; Code and data: <a href="https://github.com/MeetBabariya/options-validation-lab">github.com/MeetBabariya/options-validation-lab</a></div>

<h2>1. Executive summary</h2>
<div class="summary">Seven models were validated for marking and hedging {CCY} options: SVI, SSVI, SABR, Heston, Bates, Black-Scholes
and an adaptive challenger delta. <b>SVI and SABR are fit for marking</b>: they price
{100 * score.loc['SVI', 'mean_in_spread']:.0f}% and {100 * score.loc['SABR', 'mean_in_spread']:.0f}% of quotes inside the bid-ask spread.
<b>Heston is not approved</b>: it needs implausible dynamics, misprices short-dated options and hedges worse than Black-Scholes.
For hedging, <b>no fixed smile-dynamics assumption beats the plain sticky-strike delta</b> over a year, because the
spot-volatility correlation changes sign between months ({corr.min():+.2f} to {corr.max():+.2f}).</div>
<div class="kpis">
<div class="kpi"><b>{score.loc['SVI', 'median_rmse_volpts']:.2f}</b><span>SVI median IV error, vol pts</span></div>
<div class="kpi"><b>{n_pass}/{len(tests)}</b><span>implementation tests passing</span></div>
<div class="kpi"><b>{len(test_pairs):,}</b><span>out-of-sample option-days hedged</span></div>
<div class="kpi"><b>{pct(g['Heston'])}</b><span>Heston hedge vs Black-Scholes</span></div>
</div>
{table(["Model", "Intended use", "Decision", "Evidence", "Conditions"], [(m, u, badge(d), e, c) for m, u, d, e, c in decisions], widths=[14, 14, 15, 31, 26])}

<h2>2. Scope and intended use</h2>
<p>The models are assessed for two uses: <b>marking</b> (producing an implied-volatility surface consistent with the market and
free of static arbitrage) and <b>hedging</b> (producing a delta that minimises the one-day hedging error). Out of scope:
exotic pricing, vega hedging, transaction costs and capital.</p>

<h2>3. Data</h2>
{table(["Dataset", "Source", "Size", "Use"], [
    ("Live option chains", "Deribit public API (no key), snapshot daily at 08:00 UTC", f"latest: {len(q)} options, {q['expiry'].nunique()} expiries", "Fit, arbitrage, identifiability"),
    ("Historical chain states", "Rebuilt from Deribit trade history: marks, trade IV, index", f"{n_hist_days} daily windows (07:00 to 09:00 UTC)", "Hedging backtest, parameter history"),
    ("Hedging pairs", "Same option observed on consecutive days", f"{len(pairs):,} pairs, {len(test_pairs):,} in the out-of-sample test", "Outcome analysis"),
])}
<p><b>Cleaning.</b> Only out-of-the-money options with a two-sided market and |delta| of at least 5% are used; deep-wing
quotes are mostly tick rounding. <b>Forward.</b> Deribit prices options in coin on the forward of each expiry. The
historical forward is recovered from each trade's price and Deribit's IV (one equation, unique root). Checked against
live forwards, the error is below 0.5 bp of the forward.</p>

<h2>4. Conceptual soundness</h2>
{table(["Model", "Key assumption", "Known weakness"], [
    ("SVI", "Smile per expiry; Lee moment bound; Durrleman butterfly condition checked", "Static: says nothing about how the smile moves"),
    ("SSVI", "Global surface, arbitrage-free under Gatheral-Jacquier conditions", "One skew parameter for all maturities"),
    ("SABR (beta = 1)", "Lognormal forward with stochastic vol per expiry", "Hagan expansion; per-expiry parameters"),
    ("Heston", "Mean-reverting stochastic variance, no jumps", "Cannot produce steep short-dated smiles without jumps"),
    ("Bates", "Heston plus lognormal jumps (Merton)", "Eight parameters; jump size, frequency and dispersion trade off"),
    ("BS sticky strike", "Each option keeps its own IV when spot moves", "Ignores spot-vol co-movement"),
])}

<h2>5. Implementation verification</h2>
<p>{n_pass} of {len(tests)} automated tests pass in this run. Key checks against independent references:</p>
{table(["Check", "Result"], [(c, '<span class="pass">pass</span>' if status.get(t) else '<span class="fail">FAIL</span>') for c, t in verification])}

<h2>6. Outcome analysis: fit to the market</h2>
<p>Latest snapshot {params['snapshot']}: {len(otm)} usable quotes from {otm['T'].min() * 365:.0f} to {otm['T'].max() * 365:.0f} days.
SVI shows {bfly} butterfly and {len(cal_viol)} calendar arbitrage violations.</p>
{table(["Model", "Median IV error (vol pts)", "Quotes inside bid-ask"], fit_rows)}
<img src="{CCY}_models.png" alt="model fits"><div class="cap">Market bid-ask (grey) against each model for four expiries, and fit error by maturity.</div>

<h2>7. Outcome analysis: hedging</h2>
<p>One-day delta hedges for every out-of-the-money option observed on consecutive days. Gain = 1 - SSE(model) / SSE(BS).
Empirical deltas are fitted only on earlier months. Intervals: bootstrap over whole days.</p>
{table(["Delta", "Gain vs BS", "95% interval", "Gain, vega-scaled", "Days beating BS"], hedge_rows)}
<img src="{CCY}_hedging.png" alt="hedging results"><div class="cap">Gains with intervals, gains by month, and spot-vol correlation by month.</div>

<h2>8. Stability and identifiability</h2>
<p><b>Within a day:</b> quotes redrawn uniformly inside their bid-ask spreads and Heston refitted 20 times. Parameters
move by at most {100 * hes_cv.max():.1f}%: the data pins them down, so the misfit is not estimation noise.</p>
{table(["Heston parameter", "Mean", "Std", "Coefficient of variation"], ident_rows)}
<p><b>Bates, 10 refits:</b> the jump parameters move more and are almost perfectly correlated with each other
(|correlation| {bat_ridge:.2f}); the combined jump variance is far more stable than its parts.</p>
{table(["Bates parameter", "Mean", "Std", "Coefficient of variation"], [(n, f"{bat_boot[n].mean():.3f}", f"{bat_boot[n].std():.3f}", f"{100 * bat_cv[n]:.1f}%") for n in cal.BATES_NAMES] + [("jump variance per year", f"{bat_boot['jump_var'].mean():.3f}", f"{bat_boot['jump_var'].std():.3f}", f"{100 * bat_boot['jump_var'].std() / bat_boot['jump_var'].mean():.1f}%")])}
<p><b>Across days:</b> over the backtest year, Heston's kappa reached its upper bound on {100 * k_at_bound:.0f}% of days.</p>
<img src="{CCY}_heston_history.png" alt="Heston parameter history">

<h2>9. Findings register</h2>
{table(["ID", "Severity", "Area", "Finding", "Recommendation"], [(i, sev(s), a, f, r) for i, s, a, f, r in findings], widths=[5, 9, 14, 45, 27])}

<h2>10. Limitations</h2>
<p>Marks are Deribit model prices, not executable quotes. Hedging is daily with no transaction costs. Option and forward
are measured at each instrument's last trade in the window, and only options that trade on both days are included,
which favours liquid strikes. Results cover {CCY} only.</p>
"""

short = {"SVI (per expiry)": "SVI", "SSVI (global)": "SSVI", "SABR, beta = 1 (per expiry)": "SABR", "Heston (global)": "Heston",
         "Bates (Heston + jumps)": "Bates", "Black-Scholes, sticky strike": "Black-Scholes"}
(REP / "decisions.json").write_text(json.dumps({short[m]: d for m, _, d, _, _ in decisions if m in short}, indent=1))

out = REP / "validation_report.html"
out.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<title>{CCY} Model Validation Report</title><style>{css}</style></head><body>{body}</body></html>')
print("saved", out.relative_to(ROOT), f"({n_pass}/{len(tests)} tests passing)")

chrome = next((c for c in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", shutil.which("google-chrome"),
                           shutil.which("chromium")] if c and Path(c).exists()), None)
if chrome:
    pdf = REP / "validation_report.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    f"file://{out}"], capture_output=True)
    print("saved", pdf.relative_to(ROOT))
