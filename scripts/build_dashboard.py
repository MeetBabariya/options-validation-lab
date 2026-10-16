"""Export everything the live dashboard shows into docs/ (served by GitHub Pages).

docs/data.json: latest smiles with every model's curve, term structure, model scores, hedging results, Heston
parameter history. docs/index.html is the hand-written page (site/index.html) that renders it.
Usage: python scripts/build_dashboard.py [BTC]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovl import calibrate as cal, data, heston, sabr, svi, black  # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "BTC"
REP, DOCS = ROOT / "reports", ROOT / "docs"
DOCS.mkdir(exist_ok=True)
r = lambda a, n=4: [None if not np.isfinite(v) else round(float(v), n) for v in np.asarray(a, float)]

snap = sorted((ROOT / "data/snapshots").glob(f"{CCY}_*.csv.gz"))[-1]
q = pd.read_csv(snap, parse_dates=["ts", "expiry"])
otm = data.otm_quotes(q).reset_index(drop=True)
fits = svi.fit_svi_surface(otm)
theta, ssvi_p = svi.fit_ssvi(otm)
sabr_p = cal.fit_sabr(otm)
bs = cal.fit_bs(otm)
hp = cal.fit_heston(otm)
bp = cal.fit_bates(otm, heston_start=hp)

smiles = []
for exp, s in otm.groupby("expiry"):
    T = s["T"].iloc[0]
    kk = np.linspace(s["k"].min() - 0.03, s["k"].max() + 0.03, 60)
    c = kk >= 0
    hv = black.implied_vol(heston.price(np.exp(kk), T, hp, c, N=1024), 1.0, np.exp(kk), T, c)
    bv = black.implied_vol(heston.bates_price(np.exp(kk), T, bp, c, N=1024), 1.0, np.exp(kk), T, c)
    smiles.append(dict(
        expiry=str(exp.date()), days=round(T * 365, 1), F=round(float(s["F"].iloc[0]), 1),
        k=r(s["k"]), strike=r(s["K"], 0), bid=r(100 * s["bid_iv"], 2), ask=r(100 * s["ask_iv"], 2),
        grid=r(kk), models={
            "SVI": r(100 * np.sqrt(np.maximum(svi.svi_w(kk, *fits[exp].params), 0) / T), 2),
            "SSVI": r(100 * np.sqrt(svi.ssvi_w(kk, theta[exp], *ssvi_p) / T), 2),
            "SABR": r(100 * sabr.implied_vol(kk, T, *sabr_p[exp]), 2),
            "Heston": r(100 * hv, 2),
            "Bates": r(100 * bv, 2),
            "Black-Scholes": r(np.full(len(kk), 100 * bs[exp]), 2)}))

atm = [dict(days=s["days"], expiry=s["expiry"],
            atm=round(float(np.interp(0, s["grid"], s["models"]["SVI"])), 2)) for s in smiles]

params = json.loads(sorted((REP / "params").glob(f"{CCY}_*.json"))[-1].read_text())
live_hist = []
for f in sorted((REP / "params").glob(f"{CCY}_*.json")):
    p = json.loads(f.read_text())
    live_hist.append(dict(ts=p["ts"][:16], **{k: round(v, 4) for k, v in p["heston"].items()}))

pairs = pd.read_csv(REP / "hedging" / f"{CCY}_pairs.csv.gz", parse_dates=["day"])
dh = pairs.groupby("day")[["h_kappa", "h_xi", "h_rho"]].first()
gains = pd.read_csv(REP / "hedging" / f"{CCY}_gains.csv", index_col=0)
robust = pd.read_csv(REP / "hedging" / f"{CCY}_gains_robust.csv", index_col=0)
monthly = pd.read_csv(REP / "hedging" / f"{CCY}_gains_monthly.csv", index_col=0)

out = dict(
    ccy=CCY, snapshot=snap.name, ts=str(q["ts"].max())[:16] + " UTC", n_options=len(q), n_quotes=len(otm),
    smiles=smiles, atm=atm,
    scores=params["score"],
    heston_now=dict(zip(cal.HESTON_NAMES, (round(v, 4) for v in hp)), feller=round(cal.feller(hp), 3)),
    bates_now=dict(zip(cal.BATES_NAMES, (round(v, 4) for v in bp))),
    decisions=json.loads((REP / "decisions.json").read_text()) if (REP / "decisions.json").exists() else {},
    heston_live_history=live_hist,
    heston_year=dict(day=[d.strftime("%Y-%m-%d") for d in dh.index], kappa=r(dh["h_kappa"], 2), xi=r(dh["h_xi"], 3),
                     rho=r(dh["h_rho"], 3)),
    hedging=dict(
        models=[m for m in gains.index if m != "BS (sticky strike)"],
        gain=r(100 * gains["gain"].drop("BS (sticky strike)"), 1),
        lo=r(100 * gains["lo95"].drop("BS (sticky strike)"), 1), hi=r(100 * gains["hi95"].drop("BS (sticky strike)"), 1),
        days_beating=r(100 * robust["days_beating_BS"].drop("BS (sticky strike)"), 0),
        months=list(monthly.index), corr=r(monthly["corr(ret, dIV)"], 3),
        monthly={m: r(100 * monthly[m], 1) for m in monthly.columns if m not in ("corr(ret, dIV)", "BS (sticky strike)")},
        n_days=int(pairs[(pairs["day"] >= pd.Timestamp(monthly.index[0] + "-01", tz="UTC"))]["day"].nunique()),
    ),
)
(DOCS / "data.json").write_text(json.dumps(out, separators=(",", ":")))
shutil.copy(ROOT / "site" / "index.html", DOCS / "index.html")
for f in ["validation_report.html", "validation_report.pdf", f"{CCY}_models.png", f"{CCY}_hedging.png",
          f"{CCY}_heston_history.png", f"{CCY}_smiles.png"]:
    if (REP / f).exists():
        shutil.copy(REP / f, DOCS / f)
(DOCS / ".nojekyll").touch()
print(f"docs/ built from {snap.name}: {len(smiles)} expiries, data.json {(DOCS / 'data.json').stat().st_size // 1024} KB")
