"""Implied-vol surface fits in total-variance space w(k, T) = sigma_imp^2 * T.

- Raw SVI per expiry (Gatheral 2004): flexible, but each slice is fitted alone, so the surface can contain
  butterfly or calendar arbitrage. We penalise butterfly arbitrage and *check* calendar arbitrage.
- SSVI (Gatheral & Jacquier 2014): one global fit that is arbitrage-free by construction when its parameter
  conditions hold. Less flexible, so it usually fits worse. Comparing the two is part of the validation.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

K_GRID = np.linspace(-1.5, 1.5, 301)


# ---------- raw SVI ----------
def svi_w(k, a, b, rho, m, sig):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig**2))


def svi_derivs(k, a, b, rho, m, sig):
    r = np.sqrt((k - m) ** 2 + sig**2)
    w = a + b * (rho * (k - m) + r)
    w1 = b * (rho + (k - m) / r)
    w2 = b * sig**2 / r**3
    return w, w1, w2


def durrleman_g(w, w1, w2, k):
    """Risk-neutral density is proportional to g(k) * (...) > 0, so g(k) < 0 means butterfly arbitrage."""
    return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


@dataclass
class SliceFit:
    T: float
    params: tuple
    rmse_iv: float      # fit error in implied-vol points
    inside_spread: float  # share of quotes where the fitted IV lies within the bid-ask IV range
    min_g: float        # most negative Durrleman g on the grid (>= 0 means no butterfly arbitrage)


def fit_svi_slice(k, w, weights, T, penalty=1e3):
    """Quasi-explicit SVI: outer Nelder-Mead over (m, log sig); inner constrained linear LS over (a, d, c).
    A final L-BFGS polish adds the Durrleman butterfly penalty. Errors are scaled by the slice's mean total
    variance so short and long expiries are fitted with the same effort."""
    k, w, weights = (np.asarray(v, float) for v in (k, w, weights))
    scale = w.mean()
    wn, wt = w / scale, weights / weights.sum()

    def inner(m, sig):
        y = (k - m) / sig
        X = np.column_stack([np.ones_like(y), y, np.sqrt(y**2 + 1)])
        sw = np.sqrt(wt)
        lee = 4 * sig / scale  # Lee: b(1+|rho|) <= 4 in raw units; we fit w/scale, so the bound scales too
        cons = [{"type": "ineq", "fun": lambda z: z[2]},                  # c >= 0
                {"type": "ineq", "fun": lambda z: z[2] - abs(z[1])},      # |d| <= c   (|rho| <= 1)
                {"type": "ineq", "fun": lambda z: lee - z[2] - abs(z[1])},  # c + |d| <= 4 sig (Lee's moment bound)
                {"type": "ineq", "fun": lambda z: z[0]}]                  # a >= 0 (in this parametrisation)
        z0 = np.linalg.lstsq(X * sw[:, None], wn * sw, rcond=None)[0]
        r = minimize(lambda z: np.sum(wt * (X @ z - wn) ** 2), z0, constraints=cons, method="SLSQP")
        return r.x, r.fun

    # Floor on sig: a smile kink narrower than ~1/4 of one ATM standard deviation cannot be identified from
    # strikes this far apart, and letting sig -> 0 turns the smile into a V that just fits quote noise.
    sig_min = 0.25 * np.sqrt(np.interp(0.0, np.sort(k), w[np.argsort(k)]))

    def outer(x):
        return inner(x[0], max(np.exp(x[1]), sig_min))[1]

    best = None
    for m0 in (-0.3, 0.0, 0.3):
        for s0 in (0.05, 0.2, 0.5):
            r = minimize(outer, (m0, np.log(s0)), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-12})
            if best is None or r.fun < best.fun:
                best = r
    m, sig = best.x[0], max(np.exp(best.x[1]), sig_min)
    (a_, d, c), _ = inner(m, sig)
    b = c / sig
    rho = np.clip(d / c, -0.999, 0.999) if c > 0 else 0.0
    x = np.array([a_ * scale, b * scale, rho, m, sig])  # back to raw SVI (a, b, rho, m, sig)

    def loss(z):
        ww, w1, w2 = svi_derivs(K_GRID, *z)
        g = durrleman_g(np.maximum(ww, 1e-12), w1, w2, K_GRID)
        arb = np.sum(np.minimum(g, 0) ** 2)
        return np.sum(wt * ((svi_w(k, *z) - w) / scale) ** 2) + penalty * arb

    ww, w1, w2 = svi_derivs(K_GRID, *x)
    if durrleman_g(np.maximum(ww, 1e-12), w1, w2, K_GRID).min() < 0:
        bounds = [(-1, 2), (1e-8, 5), (-0.999, 0.999), (-1.5, 1.5), (sig_min, 3)]
        x = minimize(loss, x, bounds=bounds, method="L-BFGS-B").x
    return x


def fit_svi_surface(otm: pd.DataFrame) -> dict:
    fits = {}
    for exp, s in otm.groupby("expiry"):
        T = s["T"].iloc[0]
        w = s["mid_iv"] ** 2 * T
        spread = (s["ask_iv"] - s["bid_iv"]).clip(lower=0.005)
        weights = 1 / spread**2  # trust tight markets more
        p = fit_svi_slice(s["k"], w, weights, T)
        iv_fit = np.sqrt(np.maximum(svi_w(s["k"].values, *p), 0) / T)
        ww, w1, w2 = svi_derivs(K_GRID, *p)
        fits[exp] = SliceFit(
            T=T, params=tuple(p),
            rmse_iv=float(np.sqrt(np.mean((iv_fit - s["mid_iv"]) ** 2))),
            inside_spread=float(np.mean((iv_fit >= s["bid_iv"]) & (iv_fit <= s["ask_iv"]))),
            min_g=float(durrleman_g(np.maximum(ww, 1e-8), w1, w2, K_GRID).min()),
        )
    return fits


def calendar_violations(fits: dict, otm: pd.DataFrame, n=101):
    """Total variance must not decrease with maturity at any fixed log-moneyness. Checked only on the k range
    quoted in *both* expiries: beyond that the comparison tests extrapolation, not the market."""
    exps = sorted(fits)
    out = []
    for e1, e2 in zip(exps, exps[1:]):
        k1, k2 = otm.loc[otm["expiry"] == e1, "k"], otm.loc[otm["expiry"] == e2, "k"]
        lo, hi = max(k1.min(), k2.min()), min(k1.max(), k2.max())
        if lo >= hi:
            continue
        k_grid = np.linspace(lo, hi, n)
        d = svi_w(k_grid, *fits[e2].params) - svi_w(k_grid, *fits[e1].params)
        if d.min() < 0:
            out.append((e1, e2, float(d.min()), float(k_grid[d.argmin()])))
    return out


# ---------- SSVI ----------
def ssvi_w(k, theta, rho, eta, gamma):
    phi = eta / (theta**gamma * (1 + theta) ** (1 - gamma))
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2))


def atm_total_variance(otm: pd.DataFrame) -> pd.Series:
    """theta_T: total variance at k = 0 per expiry (linear interpolation in k), forced non-decreasing in T."""
    th = {}
    for exp, s in otm.groupby("expiry"):
        s = s.sort_values("k")
        th[exp] = np.interp(0.0, s["k"], s["mid_iv"] ** 2 * s["T"])
    th = pd.Series(th).sort_index()
    return th.cummax()


def fit_ssvi(otm: pd.DataFrame):
    theta = atm_total_variance(otm)
    th = otm["expiry"].map(theta).values
    k = otm["k"].values
    w = (otm["mid_iv"] ** 2 * otm["T"]).values
    spread = (otm["ask_iv"] - otm["bid_iv"]).clip(lower=0.005).values
    wt = 1 / spread**2

    def loss(x):
        rho, eta, gamma = x
        pen = max(0.0, eta * (1 + abs(rho)) - 2) ** 2  # Gatheral-Jacquier no-butterfly condition
        return np.sum(wt * (ssvi_w(k, th, rho, eta, gamma) - w) ** 2) / wt.sum() + 1e3 * pen

    r = minimize(loss, (-0.1, 1.0, 0.4), bounds=[(-0.99, 0.99), (1e-3, 4), (0.01, 0.5)], method="L-BFGS-B")
    return theta, tuple(r.x)
