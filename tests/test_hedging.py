import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl import black, hedging, sabr, svi  # noqa: E402


def test_implied_forward_round_trip():
    S, T = 80_000.0, np.array([0.01, 0.1, 0.5, 1.0])
    F_true = S * np.exp(0.06 * T)
    K = np.array([78_000, 90_000, 70_000, 100_000.0])
    vol = np.array([0.5, 0.45, 0.6, 0.55])
    for is_call in (True, False):
        coin = black.price(F_true, K, T, vol, is_call) / F_true  # Deribit quotes in coin on the forward
        F = black.implied_forward(coin, K, T, vol, is_call, S)
        assert np.allclose(F, F_true, rtol=1e-9)


def test_sticky_moneyness_delta_is_total_derivative():
    # V(F) = BS(F, K, T, sigma(ln(K/F))): the analytic delta must match a bump of F that moves the smile with it
    p = (0.02, 0.1, -0.4, 0.05, 0.15)
    T, K, F = 0.25, 90.0, 100.0
    sig = lambda F: np.sqrt(svi.svi_w(np.log(K / F), *p) / T)
    V = lambda F: black.price(F, K, T, sig(F), False)
    h = 1e-4
    numeric = (V(F + h) - V(F - h)) / (2 * h)
    k = np.log(K / F)
    w, w1, _ = svi.svi_derivs(k, *p)
    s = np.sqrt(w / T)
    analytic = black.delta(F, K, T, s, False) + black.vega(F, K, T, s) * (-(w1 / (2 * s * T))) / F
    assert np.isclose(analytic, numeric, rtol=1e-6)


def test_bartlett_delta_is_derivative_with_alpha_following_the_forward():
    # SABR beta = 1: E[d alpha | dF] = rho * nu * dF / F, i.e. alpha(F) = alpha0 + rho * nu * ln(F / F0)
    T, K, F0, a0, r, n = 0.1, 105.0, 100.0, 0.5, -0.4, 2.5
    vol = lambda F: sabr.implied_vol(np.log(K / F), T, a0 + r * n * np.log(F / F0), r, n)
    V = lambda F: black.price(F, K, T, vol(F), True)
    h = 1e-4
    numeric = (V(F0 + h) - V(F0 - h)) / (2 * h)
    k, e = np.log(K / F0), 1e-6
    ds_dk = (sabr.implied_vol(k + e, T, a0, r, n) - sabr.implied_vol(k - e, T, a0, r, n)) / (2 * e)
    ds_da = (sabr.implied_vol(k, T, a0 + e, r, n) - sabr.implied_vol(k, T, a0 - e, r, n)) / (2 * e)
    s0 = sabr.implied_vol(k, T, a0, r, n)
    analytic = black.delta(F0, K, T, s0, True) + black.vega(F0, K, T, s0) * (-ds_dk + ds_da * r * n) / F0
    assert np.isclose(analytic, numeric, rtol=1e-5)


def test_min_variance_regression_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame(dict(delta_bs=rng.uniform(0.05, 0.5, n), vega=rng.uniform(1e3, 2e4, n),
                           T=rng.uniform(0.02, 0.5, n), F=80_000.0, dF=rng.normal(0, 2000, n),
                           is_call=rng.random(n) < 0.5))
    true = {True: np.array([-0.2, 0.3, -0.1]), False: np.array([0.1, -0.4, 0.2])}
    df["dV"] = 0.0  # placeholder: the features themselves do not depend on dV
    X, _ = hedging.mv_features(df)
    coef = np.array([true[c] for c in df["is_call"]])
    df["dV"] = df["delta_bs"] * df["dF"] + np.sum(X * coef, axis=1) + rng.normal(0, 1, n)
    fit = hedging.fit_mv(df)
    for c in (True, False):
        assert np.allclose(fit[c], true[c], atol=0.01), (c, fit[c])


def test_scaled_delta_formula_matches_direct_bump():
    # delta = c(m) - m c'(m) with V = F c(K/F) must equal a direct bump of F
    from ovl import heston
    p, T, K, F = (0.3, 2.0, 0.4, 1.5, -0.3, 0.5, -0.3, 0.4), 0.1, 105.0, 100.0
    V = lambda f: heston.bates_price(np.array([K]), T, p, True, F=f)[0]
    direct = (V(F * 1.0001) - V(F * 0.9999)) / (0.0002 * F)
    m, h = K / F, 1e-3
    c = lambda x: heston.bates_price(np.array([x]), T, p, True)[0]
    scaled = c(m) - m * (c(m * (1 + h)) - c(m * (1 - h))) / (2 * m * h)
    assert np.isclose(scaled, direct, rtol=1e-5)
