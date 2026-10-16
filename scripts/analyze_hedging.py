"""Analyse the cached hedging pairs: all deltas, robust metrics, regime dependence, and the summary chart.

Usage: python scripts/analyze_hedging.py BTC [pairs.csv.gz]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ovl import black, hedging  # noqa: E402

CCY = sys.argv[1] if len(sys.argv) > 1 else "BTC"
src = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports" / "hedging" / f"{CCY}_pairs.csv.gz"
df = pd.read_csv(src, parse_dates=["ts", "ts_1", "expiry", "day"])
df = df[(df["dt_h"] > 20) & (df["dt_h"] < 28)].reset_index(drop=True)  # observed about one day apart
df["month"] = df["day"].dt.strftime("%Y-%m")
df["tenor"] = pd.cut(df["T"] * 365, [0, 14, 60, 400], labels=["<14d", "14-60d", ">60d"])
df["ret"] = df["dF"] / df["F"]
# the option's own IV change over the day (marks), for the realised spot-vol slope
df["dsig"] = black.implied_vol(df["V_1"], df["F_1"], df["K"], df["T"] - df["dt_h"] / 24 / 365, df["is_call"]) - df["mark_iv"]

# --- Hull & White min-variance delta: expanding window, refit monthly, strictly out of sample ---
months = sorted(df["month"].unique())
df["delta_mv"] = np.nan
for m in months[3:]:
    idx = df["month"] == m
    df.loc[idx, "delta_mv"] = hedging.mv_delta(df[idx], hedging.fit_mv(df[df["month"] < m]))

# --- adaptive spot-vol delta: d sigma = beta * dF/F, beta re-estimated from the previous 30 days per tenor ---
# Uses only pairs whose day-1 observation is at least one day before the hedge starts (day <= d - 2).
df["delta_adaptive"] = np.nan
ok = df["dsig"].notna()
for d in sorted(df["day"].unique()):
    hist = df[ok & (df["day"] <= d - pd.Timedelta(days=2)) & (df["day"] > d - pd.Timedelta(days=32))]
    today = df["day"] == d
    for tenor, g in hist.groupby("tenor", observed=True):
        if len(g) < 100:
            continue
        beta = np.polyfit(g["ret"], g["dsig"], 1)[0]
        idx = today & (df["tenor"] == tenor)
        df.loc[idx, "delta_adaptive"] = df.loc[idx, "delta_bs"] + df.loc[idx, "vega"] * beta / df.loc[idx, "F"]

hedging.DELTA_COLS["Adaptive spot-vol (30d)"] = "delta_adaptive"
cols = {m: c for m, c in hedging.DELTA_COLS.items() if c in df}
test = df[df[list(cols.values())].notna().all(axis=1)].copy()
print(f"{df['day'].nunique()} days, {len(df):,} pairs; common out-of-sample test set: "
      f"{test['day'].nunique()} days, {len(test):,} pairs ({test['day'].min():%Y-%m-%d} to {test['day'].max():%Y-%m-%d})")

pd.set_option("display.width", 220)
print("\n1) Gain over BS delta: 1 - SSE/SSE_BS (USD errors), 95% day-block bootstrap")
boot = hedging.day_block_bootstrap(test)
print(boot.round(3).to_string())

# robust view 1: errors scaled by vega (in vol points), so long-dated ATM options do not dominate
scaled = test.copy()
for c in cols.values():
    scaled[c + "_e"] = (scaled["dV"] - scaled[c] * scaled["dF"]) / scaled["vega"]
sse_v = {m: np.sum(scaled[c + "_e"] ** 2) for m, c in cols.items()}
# robust view 2: share of days on which each delta beats BS (day-level SSE)
day_sse = {m: test.assign(e2=(test["dV"] - test[c] * test["dF"]) ** 2).groupby("day")["e2"].sum() for m, c in cols.items()}
robust = pd.DataFrame({
    "gain_vega_scaled": {m: 1 - v / sse_v["BS (sticky strike)"] for m, v in sse_v.items()},
    "gain_median_abs": {m: 1 - np.median(np.abs(test["dV"] - test[c] * test["dF"]))
                        / np.median(np.abs(test["dV"] - test["delta_bs"] * test["dF"])) for m, c in cols.items()},
    "days_beating_BS": {m: (day_sse[m] < day_sse["BS (sticky strike)"]).mean() for m in cols},
})
print("\n2) Robust views: vega-scaled SSE gain, median |error| gain, share of days beating BS")
print(robust.round(3).to_string())

print("\n3) Gain by month (USD SSE)")
monthly = pd.DataFrame({m: hedging.gains(g) for m, g in test.groupby("month")}).T
monthly.insert(0, "corr(ret, dIV)", test[test["dsig"].notna()].groupby("month").apply(
    lambda g: np.corrcoef(g["ret"], g["dsig"])[0, 1], include_groups=False))
print(monthly.round(3).to_string())

print("\n4) Spot-vol slope d(IV)/d(ln F): realised vs what each delta assumes (median, vol per 100% move)")
slopes = {"realised (all pairs)": np.polyfit(df.loc[ok, "ret"], df.loc[ok, "dsig"], 1)[0]}
for m, c in cols.items():
    if m != "BS (sticky strike)":
        slopes[m] = float(((test[c] - test["delta_bs"]) * test["F"] / test["vega"]).median())
print(pd.Series(slopes).round(3).to_string())

bucket = hedging.bucketed_gains(test)
out = ROOT / "reports" / "hedging"
boot.to_csv(out / f"{CCY}_gains.csv")
robust.to_csv(out / f"{CCY}_gains_robust.csv")
monthly.to_csv(out / f"{CCY}_gains_monthly.csv")
bucket.to_csv(out / f"{CCY}_gains_by_bucket.csv", index=False)

# chart
models = [m for m in cols if m != "BS (sticky strike)"]
colors = dict(zip(models, ["C0", "C2", "C1", "C4", "C3"]))
fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
b = boot.loc[models]
axes[0].barh(models, 100 * b["gain"], xerr=[100 * (b["gain"] - b["lo95"]), 100 * (b["hi95"] - b["gain"])],
             color=[colors[m] for m in models], capsize=4)
axes[0].axvline(0, color="k", lw=0.8)
axes[0].invert_yaxis()
axes[0].set_xlabel("reduction in squared hedging error vs BS delta (%)")
axes[0].set_title(f"1-day hedge, out of sample: {test['day'].nunique()} days, {len(test):,} option-days")
mm = monthly[models] * 100
for m in models:
    axes[1].plot(mm.index, mm[m], "o-", color=colors[m], label=m, ms=4)
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_ylabel("gain vs BS (%)")
axes[1].set_title("Gain by month: no smile model wins consistently")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend(fontsize=7)
c = monthly["corr(ret, dIV)"]
axes[2].bar(c.index, c.values, color=np.where(c.values < 0, "C3", "C2"))
axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_title("Spot-vol correlation by month flips sign")
axes[2].set_ylabel("corr(BTC daily return, IV change)")
axes[2].tick_params(axis="x", rotation=45)
fig.suptitle(f"{CCY} options on Deribit: which delta hedges best?")
fig.tight_layout()
fig.savefig(ROOT / "reports" / f"{CCY}_hedging.png", dpi=110)
print("\nsaved", f"reports/{CCY}_hedging.png")
