"""Heston (1993) stochastic volatility, priced on the forward (r = 0, F = 1 after normalising strikes).

dS/S = sqrt(v) dW1,  dv = kappa (theta - v) dt + xi sqrt(v) dW2,  d<W1, W2> = rho dt

Prices use the COS method (Fang & Oosterlee 2008) with the "little Heston trap" form of the characteristic function
(Albrecher et al. 2007), which stays continuous for long maturities.
"""
import numpy as np


def charfn(u, T, v0, kappa, theta, xi, rho):
    """E[exp(i u log(S_T / F))]."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    edt = np.exp(-d * T)
    C = kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * edt) / (1 - g)))
    D = (b - d) / xi**2 * (1 - edt) / (1 - g * edt)
    return np.exp(C + D * v0)


def _chi_psi(uk, a, c, d):
    chi = (np.cos(uk * (d - a)) * np.exp(d) - np.cos(uk * (c - a)) * np.exp(c)
           + uk * np.sin(uk * (d - a)) * np.exp(d) - uk * np.sin(uk * (c - a)) * np.exp(c)) / (1 + uk**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = (np.sin(uk * (d - a)) - np.sin(uk * (c - a))) / uk
    psi = np.where(uk == 0, d - c, psi)
    return chi, psi


def put_price(K, T, params, F=1.0, N=256, L=12.0):
    """European puts for an array of strikes K at one maturity T. Puts are priced (bounded payoff, stable),
    calls follow from parity."""
    v0, kappa, theta, xi, rho = params
    K = np.atleast_1d(np.asarray(K, float))
    vbar = max(v0, theta)
    half = L * np.sqrt(vbar * T) * (1 + xi)  # truncation range for log(S_T/F): wide enough for fat Heston tails
    a, b = -half, half
    k = np.arange(N)
    uk = k * np.pi / (b - a)
    chi, psi = _chi_psi(uk, a, a, 0.0)
    Uk = 2 / (b - a) * (-chi + psi)          # put payoff coefficients on y = log(S_T/K)
    phi = charfn(uk, T, v0, kappa, theta, xi, rho)
    x = np.log(F / K)                        # shift from log(S_T/F) to log(S_T/K)
    terms = (phi * Uk)[None, :] * np.exp(1j * np.outer(x - a, uk))
    terms[:, 0] *= 0.5
    return np.maximum(K * terms.real.sum(axis=1), 0.0)


def price(K, T, params, is_call, F=1.0, **kw):
    p = put_price(K, T, params, F, **kw)
    return np.where(is_call, p + F - np.asarray(K), p)


# ---------- Bates (1996): Heston + Merton lognormal jumps ----------
# Jumps arrive at rate lam; each multiplies S by exp(J) with J ~ N(muj, sj^2). The drift is compensated so S/F
# stays a martingale. params = (v0, kappa, theta, xi, rho, lam, muj, sj).
def bates_charfn(u, T, v0, kappa, theta, xi, rho, lam, muj, sj):
    kbar = np.exp(muj + 0.5 * sj**2) - 1
    jump = lam * T * (np.exp(1j * u * muj - 0.5 * sj**2 * u**2) - 1 - 1j * u * kbar)
    return charfn(u, T, v0, kappa, theta, xi, rho) * np.exp(jump)


def bates_put_price(K, T, params, F=1.0, N=1024, L=12.0):
    v0, kappa, theta, xi, rho, lam, muj, sj = params
    K = np.atleast_1d(np.asarray(K, float))
    var = max(v0, theta) * T * (1 + xi) ** 2 + lam * T * (muj**2 + sj**2)  # diffusion + jump variance
    half = L * np.sqrt(var) + abs(lam * T * muj)
    a, b = -half, half
    uk = np.arange(N) * np.pi / (b - a)
    chi, psi = _chi_psi(uk, a, a, 0.0)
    Uk = 2 / (b - a) * (-chi + psi)
    phi = bates_charfn(uk, T, *params)
    x = np.log(F / K)
    terms = (phi * Uk)[None, :] * np.exp(1j * np.outer(x - a, uk))
    terms[:, 0] *= 0.5
    return np.maximum(K * terms.real.sum(axis=1), 0.0)


def bates_price(K, T, params, is_call, F=1.0, **kw):
    p = bates_put_price(K, T, params, F, **kw)
    return np.where(is_call, p + F - np.asarray(K), p)
