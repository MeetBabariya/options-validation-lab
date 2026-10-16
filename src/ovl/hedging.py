"""Which model's delta hedges best? One-day hedging test on rebuilt Deribit chain states.

For every out-of-the-money option observed on day d and again on day d+1:
    hedge error  e = dV - delta * dF
where dV is the change in its USD value and dF the change in the forward of its expiry (the hedge is a forward
or future on that expiry). The model whose delta gives the smallest sum of squared errors hedges best. Results are
reported as Hull & White (2017)'s gain over Black-Scholes: 1 - SSE(model) / SSE(BS).

Only OTM options are used: by put-call parity an ITM call hedges exactly like the OTM put at the same strike.
"""
import numpy as np
import pandas as pd

from . import black, calibrate as cal, heston, sabr, svi

DELTAS = ["BS (sticky strike)", "Sticky moneyness (SVI)", "SABR Bartlett", "Heston", "Min-variance (empirical)"]


def otm(p: pd.DataFrame) -> pd.DataFrame:
    o = p[(p["is_call"] & (p["k"] >= 0)) | (~p["is_call"] & (p["k"] < 0))].copy()
    o["mid_iv"] = o["mark_iv"]
    o["bid_iv"], o["ask_iv"] = o["mark_iv"] - 0.005, o["mark_iv"] + 0.005  # marks have no spread: uniform weights
    d = black.delta(o["F"], o["K"], o["T"], o["mark_iv"], o["is_call"])
    return o[(np.abs(d) >= 0.05)]


def model_deltas(p: pd.DataFrame, min_strikes=6, with_heston=True) -> pd.DataFrame:
    """Fit SVI and SABR per expiry (and Heston globally) to one day's marks; return each OTM option's deltas."""
    o = otm(p)
    counts = o.groupby("expiry")["K"].transform("size")
    o = o[counts >= min_strikes].reset_index(drop=True)
    if o.empty:
        return o
    F, K, T, s, c = o["F"].values, o["K"].values, o["T"].values, o["mark_iv"].values, o["is_call"].values
    d_bs = black.delta(F, K, T, s, c)
    vega = black.vega(F, K, T, s)
    sm_slope = np.full(len(o), np.nan)
    sabr_adj = np.full(len(o), np.nan)
    for exp, g in o.groupby("expiry"):
        i = g.index.values
        k, Tg = g["k"].values, g["T"].iloc[0]
        # sticky moneyness: smile fixed in k = ln(K/F), so d sigma / dF = -sigma'(k) / F
        sp = svi.fit_svi_slice(k, g["mark_iv"].values ** 2 * Tg, np.ones(len(g)), Tg)
        w, w1, _ = svi.svi_derivs(k, *sp)
        sig = np.sqrt(np.maximum(w, 1e-12) / Tg)
        sm_slope[i] = w1 / (2 * sig * Tg)
        # SABR (beta = 1) with Bartlett's correction: alpha co-moves with F, E[d alpha | dF] = rho * nu * dF / F
        a, r, n = cal.fit_sabr(g)[exp]
        h = 1e-4
        ds_dk = (sabr.implied_vol(k + h, Tg, a, r, n) - sabr.implied_vol(k - h, Tg, a, r, n)) / (2 * h)
        ds_da = (sabr.implied_vol(k, Tg, a * (1 + h), r, n) - sabr.implied_vol(k, Tg, a * (1 - h), r, n)) / (2 * a * h)
        sabr_adj[i] = -ds_dk + ds_da * r * n
    o["delta_bs"] = d_bs
    o["vega"] = vega
    o["delta_sm"] = d_bs + vega * (-sm_slope) / F
    o["delta_bartlett"] = d_bs + vega * sabr_adj / F
    if with_heston:
        hp = cal.fit_heston(o)
        dh = np.empty(len(o))
        for exp, g in o.groupby("expiry"):
            i = g.index.values
            Fg, Kg, Tg, cg = g["F"].values, g["K"].values, g["T"].iloc[0], g["is_call"].values
            up = np.array([heston.price(np.array([kk]), Tg, hp, cc, F=ff * 1.001, N=1024)[0] for kk, cc, ff in zip(Kg, cg, Fg)])
            dn = np.array([heston.price(np.array([kk]), Tg, hp, cc, F=ff * 0.999, N=1024)[0] for kk, cc, ff in zip(Kg, cg, Fg)])
            dh[i] = (up - dn) / (0.002 * Fg)
        o["delta_heston"] = dh
        o["heston_params"] = [hp] * len(o)
    return o


def pairs(day0: pd.DataFrame, panel1: pd.DataFrame) -> pd.DataFrame:
    """Join an option's day-0 deltas with its day-1 observation."""
    m = day0.merge(panel1[["instrument_name", "ts", "F", "V"]], on="instrument_name", suffixes=("", "_1"))
    m["dV"] = m["V_1"] - m["V"]
    m["dF"] = m["F_1"] - m["F"]
    m["dt_h"] = (m["ts_1"] - m["ts"]).dt.total_seconds() / 3600
    return m


# ---------- Hull & White (2017) empirical minimum-variance delta ----------
def mv_features(df):
    """dV - delta_BS dF = vega / sqrt(T) * (a + b delta + c delta^2) * dF / F  ->  linear regression in (a, b, c)."""
    d = df["delta_bs"].values
    base = (df["vega"] / np.sqrt(df["T"]) * df["dF"] / df["F"]).values
    return np.column_stack([base, base * d, base * d**2]), (df["dV"] - df["delta_bs"] * df["dF"]).values


def fit_mv(train: pd.DataFrame) -> dict:
    """Separate coefficients for calls and puts (Hull & White fit them per option type)."""
    out = {}
    for is_call, g in train.groupby("is_call"):
        X, y = mv_features(g)
        out[bool(is_call)] = np.linalg.lstsq(X, y, rcond=None)[0]
    return out


def mv_delta(df: pd.DataFrame, coef: dict) -> np.ndarray:
    d = df["delta_bs"].values
    a = np.array([coef[bool(c)] for c in df["is_call"]])
    return d + (df["vega"] / (df["F"] * np.sqrt(df["T"]))).values * (a[:, 0] + a[:, 1] * d + a[:, 2] * d**2)


# ---------- scoring ----------
DELTA_COLS = {"BS (sticky strike)": "delta_bs", "Sticky moneyness (SVI)": "delta_sm", "SABR Bartlett": "delta_bartlett",
              "Heston": "delta_heston", "Min-variance (empirical)": "delta_mv"}


def gains(df: pd.DataFrame) -> pd.Series:
    sse = {m: np.sum((df["dV"] - df[c] * df["dF"]) ** 2) for m, c in DELTA_COLS.items() if c in df}
    return pd.Series({m: 1 - v / sse["BS (sticky strike)"] for m, v in sse.items()})


def bucketed_gains(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        side=np.where(df["is_call"], "OTM calls", "OTM puts"),
        absdelta=pd.cut(df["delta_bs"].abs(), [0.05, 0.25, 0.5], labels=["5-25 delta", "25-50 delta"]),
        tenor=pd.cut(df["T"] * 365, [0, 14, 60, 400], labels=["<14d", "14-60d", ">60d"]))
    rows = []
    for keys, g in df.groupby(["side", "absdelta", "tenor"], observed=True):
        if len(g) >= 30:
            rows.append(dict(zip(["side", "delta", "tenor"], keys), n=len(g), **gains(g)))
    return pd.DataFrame(rows)


def day_block_bootstrap(df: pd.DataFrame, n=500, seed=0) -> pd.DataFrame:
    """95% intervals for each gain by resampling whole days: errors within a day share the same BTC move."""
    rng = np.random.default_rng(seed)
    days = df["day"].unique()
    by_day = {d: g for d, g in df.groupby("day")}
    res = [gains(pd.concat([by_day[d] for d in rng.choice(days, len(days))])) for _ in range(n)]
    res = pd.DataFrame(res)
    return pd.DataFrame({"gain": gains(df), "lo95": res.quantile(0.025), "hi95": res.quantile(0.975)})
