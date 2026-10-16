"""Black-76 pricing on the forward, with zero discounting (Deribit options settle on the forward)."""
import numpy as np
from scipy.stats import norm


def price(F, K, T, vol, is_call):
    F, K, T, vol = map(np.asarray, (F, K, T, vol))
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    return np.where(is_call, call, call - F + K)  # put-call parity


def vega(F, K, T, vol):
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return F * norm.pdf(d1) * np.sqrt(T)


def delta(F, K, T, vol, is_call):
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)


def implied_vol(p, F, K, T, is_call, lo=1e-4, hi=10.0, iters=100):
    """Vectorised bisection: robust for deep OTM quotes where Newton fails. NaN if price is outside no-arb bounds."""
    p, F, K, T, is_call = np.broadcast_arrays(*map(np.asarray, (p, F, K, T, is_call)))
    intrinsic = np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = np.where(is_call, F, K)
    ok = (p > intrinsic + 1e-12) & (p < upper)
    a = np.full(p.shape, lo, float)
    b = np.full(p.shape, hi, float)
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_high = price(F, K, T, m, is_call) > p
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    return np.where(ok, 0.5 * (a + b), np.nan)
