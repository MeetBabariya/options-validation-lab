import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl import black, svi  # noqa: E402


def test_implied_vol_round_trip():
    F, T = 80_000.0, 0.25
    K = np.array([50_000, 70_000, 80_000, 90_000, 120_000.0])
    vol = np.array([0.7, 0.5, 0.4, 0.45, 0.6])
    for is_call in (True, False):
        p = black.price(F, K, T, vol, is_call)
        assert np.allclose(black.implied_vol(p, F, K, T, is_call), vol, atol=1e-6)


def test_put_call_parity():
    F, K, T, v = 100.0, 110.0, 0.5, 0.3
    assert np.isclose(black.price(F, K, T, v, True) - black.price(F, K, T, v, False), F - K)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(black.implied_vol(5.0, 100.0, 90.0, 1.0, True))  # below intrinsic 10


def test_svi_recovers_known_slice():
    true = (0.02, 0.1, -0.4, 0.05, 0.15)
    k = np.linspace(-0.6, 0.6, 41)
    w = svi.svi_w(k, *true)
    fit = svi.fit_svi_slice(k, w, np.ones_like(k), T=0.5)
    assert np.max(np.abs(svi.svi_w(k, *fit) - w)) < 1e-4


def test_durrleman_flags_known_butterfly_arbitrage():
    # Raw SVI slice from Gatheral & Jacquier (2014), Example 3.1 (Axel Vogt): has butterfly arbitrage.
    p = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    w, w1, w2 = svi.svi_derivs(svi.K_GRID, *p)
    assert durrleman_min(p) < 0
    # and a well-behaved slice does not
    assert durrleman_min((0.02, 0.1, -0.4, 0.05, 0.15)) > 0


def durrleman_min(p):
    w, w1, w2 = svi.svi_derivs(svi.K_GRID, *p)
    return svi.durrleman_g(w, w1, w2, svi.K_GRID).min()


def test_ssvi_parameters_satisfy_no_butterfly_condition_example():
    # eta(1+|rho|) <= 2 with gamma in (0, 1/2]  =>  every slice has g >= 0
    theta, rho, eta, gamma = 0.04, -0.5, 1.2, 0.4
    k = svi.K_GRID
    w = svi.ssvi_w(k, theta, rho, eta, gamma)
    h = k[1] - k[0]
    w1, w2 = np.gradient(w, h), np.gradient(np.gradient(w, h), h)
    assert svi.durrleman_g(w, w1, w2, k)[2:-2].min() > -1e-6
