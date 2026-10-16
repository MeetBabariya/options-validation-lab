"""Calibrate BS, SVI, SSVI, SABR and Heston to the latest (or a given) snapshot and compare them.

Usage: python scripts/calibrate_models.py [BTC|ETH] [path/to/snapshot.csv.gz]
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovl import calibrate as cal, data, svi  # noqa: E402

ccy = sys.argv[1] if len(sys.argv) > 1 else "BTC"
path = Path(sys.argv[2]) if len(sys.argv) > 2 else sorted((ROOT / "data/snapshots").glob(f"{ccy}_*.csv.gz"))[-1]
q = pd.read_csv(path, parse_dates=["ts", "expiry"])
otm = data.otm_quotes(q).reset_index(drop=True)
print(f"{path.name}: {len(otm)} quotes, {otm['expiry'].nunique()} expiries")

t0 = time.time()
bs = cal.fit_bs(otm)
svi_fits = svi.fit_svi_surface(otm)
theta, ssvi_p = svi.fit_ssvi(otm)
sabr_p = cal.fit_sabr(otm)
t1 = time.time()
hes = cal.fit_heston(otm)
t2 = time.time()
print(f"fit times: BS/SVI/SSVI/SABR {t1 - t0:.1f}s, Heston {t2 - t1:.1f}s")

iv = {
    "Black-Scholes": cal.bs_iv(otm, bs),
    "SVI": np.sqrt(np.maximum([svi.svi_w(k, *svi_fits[e].params) for k, e in zip(otm["k"], otm["expiry"])], 0) / otm["T"]),
    "SSVI": np.sqrt(svi.ssvi_w(otm["k"].values, otm["expiry"].map(theta).values, *ssvi_p) / otm["T"].values),
    "SABR": cal.sabr_iv(otm, sabr_p),
    "Heston": cal.heston_iv(otm, hes),
}
iv = {k: np.asarray(v, float) for k, v in iv.items()}
sc = cal.score(otm, iv)

pd.set_option("display.width", 200)
print("\nIV RMSE by expiry (vol points)")
print(sc.pivot(index="days", columns="model", values="rmse_volpts")[list(iv)].round(2).to_string())
print("\nShare of quotes priced inside the bid-ask spread")
print(sc.pivot(index="days", columns="model", values="in_spread")[list(iv)].round(2).to_string())
overall = sc.groupby("model").agg(median_rmse_volpts=("rmse_volpts", "median"),
                                  mean_in_spread=("in_spread", "mean")).loc[list(iv)]
print("\nOverall\n" + overall.round(3).to_string())
print("\nHeston: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(cal.HESTON_NAMES, hes))
      + f"  (Feller ratio {cal.feller(hes):.2f}; < 1 means variance can hit zero)")
print("SABR per expiry (alpha, rho, nu):")
for e, p in sabr_p.items():
    print(f"  {e.date()}  {p[0]:.3f} {p[1]:+.3f} {p[2]:.2f}")

# save parameters: one JSON per snapshot builds the time series for the stability analysis
stamp = path.name.split("_")[1].split(".")[0]
out = ROOT / "reports" / "params" / f"{ccy}_{stamp}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "snapshot": path.name, "ts": str(q["ts"].max()),
    "heston": dict(zip(cal.HESTON_NAMES, hes)),
    "ssvi": dict(zip(("rho", "eta", "gamma"), ssvi_p)),
    "sabr": {str(e.date()): dict(zip(("alpha", "rho", "nu"), p)) for e, p in sabr_p.items()},
    "svi": {str(e.date()): dict(zip(("a", "b", "rho", "m", "sigma"), f.params)) for e, f in svi_fits.items()},
    "bs": {str(e.date()): v for e, v in bs.items()},
    "score": overall.reset_index().to_dict(orient="records"),
}, indent=1, default=float))
print("saved", out.relative_to(ROOT))

# plot: smiles for 4 representative expiries + RMSE by maturity
exps = sorted(otm["expiry"].unique())
pick = [exps[i] for i in sorted({0, len(exps) // 3, 2 * len(exps) // 3, len(exps) - 1})]
colors = {"Black-Scholes": "0.5", "SVI": "C0", "SSVI": "C3", "SABR": "C2", "Heston": "C1"}
styles = {"Black-Scholes": ":", "SVI": "-", "SSVI": "--", "SABR": "-.", "Heston": "-"}
fig, axes = plt.subplots(1, len(pick) + 1, figsize=(5 * (len(pick) + 1), 4.2))
for ax, e in zip(axes, pick):
    s = otm[otm["expiry"] == e].sort_values("k")
    idx = s.index.values
    ax.vlines(s["k"], 100 * s["bid_iv"], 100 * s["ask_iv"], color="0.75", lw=3, label="market bid-ask")
    for name, v in iv.items():
        ax.plot(s["k"], 100 * v[idx], styles[name], color=colors[name], lw=1.6, label=name)
    ax.set_title(f"{pd.Timestamp(e).date()} ({s['T'].iloc[0] * 365:.0f} days)")
    ax.set_xlabel("log-moneyness k = ln(K/F)")
axes[0].set_ylabel("implied vol (%)")
axes[0].legend(fontsize=8)
ax = axes[-1]
for name in iv:
    g = sc[sc.model == name]
    ax.plot(g["days"], g["rmse_volpts"], "o" + styles[name], color=colors[name], label=name, ms=4)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("days to expiry")
ax.set_ylabel("IV RMSE (vol points)")
ax.set_title("Fit error by maturity")
ax.legend(fontsize=8)
fig.suptitle(f"{ccy}: model calibration on Deribit, {q['ts'].max():%Y-%m-%d %H:%M} UTC")
fig.tight_layout()
fig.savefig(ROOT / "reports" / f"{ccy}_models.png", dpi=110)
print("saved", f"reports/{ccy}_models.png")
