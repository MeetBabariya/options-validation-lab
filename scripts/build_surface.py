"""Snapshot Deribit, fit SVI + SSVI surfaces, print arbitrage checks and save a surface plot."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl import data, svi  # noqa: E402

ccy = sys.argv[1] if len(sys.argv) > 1 else "BTC"
path = data.snapshot(ccy)
q = pd.read_csv(path, parse_dates=["ts", "expiry"])
otm = data.otm_quotes(q)
print(f"{path.name}: {len(q)} options, {len(otm)} usable OTM quotes, {otm['expiry'].nunique()} expiries")

fits = svi.fit_svi_surface(otm)
theta, (rho, eta, gamma) = svi.fit_ssvi(otm)
print(f"SSVI: rho={rho:.3f} eta={eta:.3f} gamma={gamma:.3f}  no-butterfly condition eta(1+|rho|)={eta*(1+abs(rho)):.3f} <= 2")

rows = []
for exp, f in fits.items():
    s = otm[otm["expiry"] == exp]
    iv_ssvi = np.sqrt(svi.ssvi_w(s["k"].values, theta[exp], rho, eta, gamma) / f.T)
    rows.append(dict(expiry=exp.date(), days=round(f.T * 365, 1), n=len(s),
                     svi_rmse_volpts=100 * f.rmse_iv, svi_in_spread=f.inside_spread, svi_min_g=f.min_g,
                     ssvi_rmse_volpts=100 * float(np.sqrt(np.mean((iv_ssvi - s["mid_iv"]) ** 2))),
                     ssvi_in_spread=float(np.mean((iv_ssvi >= s["bid_iv"]) & (iv_ssvi <= s["ask_iv"])))))
summary = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(summary.round(3).to_string(index=False))
cal = svi.calendar_violations(fits, otm)
print(f"SVI butterfly arbitrage: {(summary.svi_min_g < -1e-6).sum()} slices; calendar arbitrage: {len(cal)} adjacent pairs")
for e1, e2, d, k in cal:
    print(f"  {e1.date()} -> {e2.date()}: total variance falls by {d:.5f} at k={k:+.2f}")

# plot: market bid/ask IV vs SVI and SSVI per expiry
exps = sorted(fits)
n = len(exps)
cols = 4
fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(16, 3.4 * np.ceil(n / cols)), squeeze=False)
for ax, exp in zip(axes.flat, exps):
    s = otm[otm["expiry"] == exp]
    T = fits[exp].T
    kk = np.linspace(s["k"].min() - 0.05, s["k"].max() + 0.05, 200)
    ax.vlines(s["k"], 100 * s["bid_iv"], 100 * s["ask_iv"], color="0.6", lw=2, label="bid-ask")
    ax.plot(kk, 100 * np.sqrt(svi.svi_w(kk, *fits[exp].params) / T), "C0", label="SVI")
    ax.plot(kk, 100 * np.sqrt(svi.ssvi_w(kk, theta[exp], rho, eta, gamma) / T), "C3--", label="SSVI")
    ax.set_title(f"{exp.date()}  ({T*365:.0f}d)", fontsize=10)
    ax.set_xlabel("log-moneyness k")
for ax in list(axes.flat)[n:]:
    ax.axis("off")
axes.flat[0].set_ylabel("implied vol (%)")
axes.flat[0].legend(fontsize=8)
fig.suptitle(f"{ccy} implied-vol smiles, Deribit {q['ts'].max():%Y-%m-%d %H:%M} UTC", fontsize=12)
fig.tight_layout()
out = Path("reports") / f"{ccy}_smiles.png"
fig.savefig(out, dpi=110)
print("saved", out)
