import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl import black, heston, sabr  # noqa: E402

# Fang & Oosterlee (2008), Table 4 test case: S0 = K = 100, T = 1, r = q = 0
FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
FO_PARAMS = (FO["v0"], FO["kappa"], FO["theta"], FO["xi"], FO["rho"])


def test_heston_cos_matches_published_reference():
    call = heston.price(np.array([100.0]), 1.0, FO_PARAMS, True, F=100.0)[0]
    assert abs(call - 5.785155450) < 1e-6


def gil_pelaez_call(F, K, T, p):
    """Independent pricer: Heston (1993) P1/P2 integrals by adaptive quadrature."""
    k = np.log(K / F)
    phi = lambda u: heston.charfn(u, T, *p)
    P2 = 0.5 + quad(lambda u: (np.exp(-1j * u * k) * phi(u) / (1j * u)).real, 1e-10, 200, limit=500)[0] / np.pi
    P1 = 0.5 + quad(lambda u: (np.exp(-1j * u * k) * phi(u - 1j) / (1j * u)).real, 1e-10, 200, limit=500)[0] / np.pi
    return F * P1 - K * P2


def test_heston_cos_matches_numerical_integration_across_strikes_and_maturities():
    p = (0.4, 2.0, 0.5, 1.5, -0.3)  # crypto-like: 60-70% vol, high vol-of-vol
    for T in (0.02, 0.25, 1.0):
        K = np.exp(np.array([-0.3, -0.1, 0.0, 0.1, 0.3]) * np.sqrt(T) * 3)
        cos = heston.price(K, T, p, True)
        ref = np.array([gil_pelaez_call(1.0, k, T, p) for k in K])
        assert np.max(np.abs(cos - ref)) < 1e-6, (T, cos, ref)


def test_heston_put_call_parity_and_bounds():
    K = np.exp(np.linspace(-0.5, 0.5, 11))
    c = heston.price(K, 0.5, FO_PARAMS, True)
    p = heston.price(K, 0.5, FO_PARAMS, False)
    assert np.allclose(c - p, 1 - K, atol=1e-10)
    assert np.all(c >= np.maximum(1 - K, 0) - 1e-12) and np.all(c <= 1)


def test_heston_with_zero_volvol_is_black_scholes():
    # xi -> 0, v0 = theta: variance is constant, so Heston collapses to Black-Scholes with vol sqrt(theta)
    K = np.exp(np.linspace(-0.3, 0.3, 7))
    h = heston.price(K, 0.5, (0.09, 1.0, 0.09, 1e-4, 0.0), True)
    bs = black.price(1.0, K, 0.5, 0.3, True)
    assert np.max(np.abs(h - bs)) < 1e-5


def test_sabr_atm_limit_and_zero_volvol():
    T, alpha, rho, nu = 0.5, 0.6, -0.3, 1.2
    atm = alpha * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T)
    assert np.isclose(sabr.implied_vol(0.0, T, alpha, rho, nu), atm)
    assert np.isclose(sabr.implied_vol(1e-9, T, alpha, rho, nu), atm, atol=1e-6)  # continuous through ATM
    assert np.allclose(sabr.implied_vol(np.linspace(-0.5, 0.5, 5), T, 0.6, 0.0, 1e-8), 0.6, atol=1e-6)


def test_sabr_negative_rho_gives_put_skew():
    v = sabr.implied_vol(np.array([-0.2, 0.0, 0.2]), 0.25, 0.5, -0.5, 1.0)
    assert v[0] > v[1] and v[0] > v[2]


def _synthetic_chain(iv_fn):
    import pandas as pd
    rows = []
    for days in (7, 30, 90, 270):
        T = days / 365
        for k in np.linspace(-0.4, 0.4, 15) * np.sqrt(T / 0.25):
            v = float(iv_fn(k, T))
            rows.append(dict(expiry=pd.Timestamp("2027-01-01") + pd.to_timedelta(int(days), unit="D"), T=T, k=k, is_call=k >= 0,
                             mid_iv=v, bid_iv=v - 0.005, ask_iv=v + 0.005))
    return pd.DataFrame(rows)


def test_heston_calibration_recovers_known_parameters():
    from ovl import calibrate as cal
    true = (0.36, 3.0, 0.49, 1.8, -0.35)
    otm = _synthetic_chain(lambda k, T: black.implied_vol(
        heston.price(np.array([np.exp(k)]), T, true, k >= 0, N=1024)[0], 1.0, np.exp(k), T, k >= 0))
    fit = cal.fit_heston(otm)
    assert np.allclose(fit, true, rtol=0.02, atol=0.005), fit


def test_sabr_calibration_recovers_known_parameters():
    from ovl import calibrate as cal
    true = {7: (0.55, -0.3, 4.0), 30: (0.6, -0.25, 2.0), 90: (0.62, -0.2, 1.2), 270: (0.65, -0.15, 0.7)}
    otm = _synthetic_chain(lambda k, T: sabr.implied_vol(k, T, *true[round(T * 365)]))
    fit = cal.fit_sabr(otm)
    for e, p in fit.items():
        days = (e - __import__("pandas").Timestamp("2027-01-01")).days
        assert np.allclose(p, true[days], rtol=0.01, atol=1e-3), (days, p)


# ---------- Bates ----------
def test_bates_without_jumps_is_heston():
    K = np.exp(np.linspace(-0.4, 0.4, 9))
    p = (0.36, 3.0, 0.49, 1.8, -0.35)
    for T in (0.02, 0.3, 1.0):
        assert np.allclose(heston.bates_price(K, T, p + (0.0, -0.1, 0.2), True), heston.price(K, T, p, True, N=1024), atol=1e-9)


def test_bates_charfn_is_martingale():
    # E[S_T / F] = phi(-i) = 1 for any parameters
    phi = heston.bates_charfn(np.array([-1j]), 0.5, 0.3, 2.0, 0.4, 1.5, -0.4, 8.0, -0.08, 0.15)
    assert np.isclose(phi[0], 1.0, atol=1e-12)


def test_bates_with_constant_variance_matches_merton_series():
    # xi -> 0, v0 = theta: Bates is Merton (1976) jump-diffusion, priced independently as a Poisson-weighted BS sum
    from math import factorial
    sig, lam, muj, sj, T = 0.5, 6.0, -0.1, 0.2, 0.25
    K = np.exp(np.linspace(-0.4, 0.3, 8))
    kbar = np.exp(muj + 0.5 * sj**2) - 1
    merton = np.zeros_like(K)
    for n in range(60):
        w = np.exp(-lam * T) * (lam * T) ** n / factorial(n)
        Fn = np.exp(n * (muj + 0.5 * sj**2) - lam * kbar * T)  # forward conditional on n jumps
        vn = np.sqrt(sig**2 + n * sj**2 / T)
        merton += w * black.price(Fn, K, T, vn, True)
    bates = heston.bates_price(K, T, (sig**2, 1.0, sig**2, 1e-5, 0.0, lam, muj, sj), True)
    assert np.max(np.abs(bates - merton)) < 1e-6


def test_bates_calibration_recovers_known_parameters():
    from ovl import calibrate as cal
    true = (0.3, 2.5, 0.4, 1.2, -0.3, 6.0, -0.08, 0.12)
    otm = _synthetic_chain(lambda k, T: black.implied_vol(
        heston.bates_price(np.array([np.exp(k)]), T, true, k >= 0)[0], 1.0, np.exp(k), T, k >= 0))
    fit = cal.fit_bates(otm)
    iv_fit = cal.bates_iv(otm, fit)
    assert np.max(np.abs(iv_fit - otm["mid_iv"].values)) < 1e-3  # prices recovered to 0.1 vol pt
    assert np.allclose(fit[5:], true[5:], rtol=0.15, atol=0.02), fit  # jump parameters close (they trade off)
