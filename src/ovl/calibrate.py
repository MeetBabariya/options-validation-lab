"""Calibrate Black-Scholes, SABR and Heston to one option-chain snapshot.

All fits minimise implied-vol errors scaled by the bid-ask spread in IV, so a model is penalised in proportion to how
far outside the market it prices: an error of 1.0 means one full bid-ask spread.
Strikes are normalised by the forward (F = 1), which is exact under Black-76 and both models with r = 0.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import black, heston, sabr

SPREAD_FLOOR = 0.005  # half a vol point: stops a few razor-thin quotes dominating the fit

# Wide bounds on purpose: on BTC the optimum sits at kappa ~ 57 and xi ~ 9. Clipping them would hide that the model
# needs implausible dynamics to fit short-dated smiles, which is a finding of the validation.
HESTON_BOUNDS = ([1e-3, 0.05, 1e-3, 0.05, -0.99], [4.0, 200.0, 4.0, 50.0, 0.99])  # v0, kappa, theta, xi, rho
HESTON_NAMES = ("v0", "kappa", "theta", "xi", "rho")


def _spread(s):
    return (s["ask_iv"] - s["bid_iv"]).clip(lower=SPREAD_FLOOR).values


# ---------- Black-Scholes: one flat vol per expiry ----------
def fit_bs(otm: pd.DataFrame) -> dict:
    out = {}
    for exp, s in otm.groupby("expiry"):
        w = 1 / _spread(s) ** 2
        out[exp] = float(np.sum(w * s["mid_iv"]) / w.sum())
    return out


def bs_iv(otm, params):
    return otm["expiry"].map(params).values


# ---------- SABR: (alpha, rho, nu) per expiry ----------
def fit_sabr(otm: pd.DataFrame) -> dict:
    out = {}
    for exp, s in otm.groupby("expiry"):
        k, T, mid, sp = s["k"].values, s["T"].iloc[0], s["mid_iv"].values, _spread(s)
        atm = float(np.interp(0.0, np.sort(k), mid[np.argsort(k)]))
        best = None
        for rho0 in (-0.5, 0.0, 0.3):
            for nu0 in (0.5, 2.0, 6.0):
                r = least_squares(lambda x: (sabr.implied_vol(k, T, *x) - mid) / sp, (atm, rho0, nu0),
                                  bounds=([1e-3, -0.999, 1e-3], [5.0, 0.999, 50.0]))
                if best is None or r.cost < best.cost:
                    best = r
        out[exp] = tuple(float(v) for v in best.x)
    return out


def sabr_iv(otm, params):
    iv = np.empty(len(otm))
    for i, (exp, s) in enumerate(otm.groupby("expiry", sort=False)):
        iv[otm.index.get_indexer(s.index)] = sabr.implied_vol(s["k"].values, s["T"].iloc[0], *params[exp])
    return iv


# ---------- Heston: 5 parameters shared by all expiries ----------
class _HestonProblem:
    price_fn = staticmethod(lambda K, T, x, c: heston.price(K, T, x, c, N=1024))

    def __init__(self, otm: pd.DataFrame, mid_iv=None):
        self.groups = []
        mid_all = otm["mid_iv"].values if mid_iv is None else mid_iv
        for exp, s in otm.groupby("expiry"):
            idx = otm.index.get_indexer(s.index)
            K, T, c = np.exp(s["k"].values), s["T"].iloc[0], s["is_call"].values
            mid = mid_all[idx]
            p_mid = black.price(1.0, K, T, mid, c)
            # price error / vega ~ IV error; dividing by the spread puts it in "spreads"
            scale = black.vega(1.0, K, T, mid) * _spread(s)
            self.groups.append((K, T, c, p_mid, scale))

    def residuals(self, x):
        return np.concatenate([(self.price_fn(K, T, x, c) - p) / sc for K, T, c, p, sc in self.groups])


def fit_heston(otm: pd.DataFrame, x0=None, mid_iv=None) -> tuple:
    prob = _HestonProblem(otm, mid_iv)
    atm_var = float(np.median(otm["mid_iv"]) ** 2)
    starts = [x0] if x0 is not None else [
        (atm_var, 2.0, atm_var, 1.0, -0.2), (atm_var, 8.0, atm_var, 3.0, 0.0), (0.5 * atm_var, 1.0, atm_var, 0.5, -0.5),
        (0.5 * atm_var, 50.0, 1.3 * atm_var, 8.0, -0.2)]
    best = None
    for s in starts:
        r = least_squares(prob.residuals, s, bounds=HESTON_BOUNDS, x_scale="jac", max_nfev=2000)
        if best is None or r.cost < best.cost:
            best = r
    return tuple(float(v) for v in best.x)


def heston_iv(otm, params):
    iv = np.empty(len(otm))
    for exp, s in otm.groupby("expiry", sort=False):
        idx = otm.index.get_indexer(s.index)
        K, T, c = np.exp(s["k"].values), s["T"].iloc[0], s["is_call"].values
        iv[idx] = black.implied_vol(heston.price(K, T, params, c, N=1024), 1.0, K, T, c)
    return iv


# ---------- Bates: Heston + lognormal jumps, 8 parameters shared by all expiries ----------
BATES_NAMES = HESTON_NAMES + ("lam", "muj", "sj")
# Wide jump bounds on purpose: on BTC the best fit is a rare, very large crash (mean log-jump below -0.6), and the
# objective has a second, much worse basin with frequent small jumps. Both kinds of start are tried.
BATES_BOUNDS = (HESTON_BOUNDS[0] + [0.0, -2.0, 0.01], HESTON_BOUNDS[1] + [60.0, 0.5, 2.0])


class _BatesProblem(_HestonProblem):
    price_fn = staticmethod(lambda K, T, x, c: heston.bates_price(K, T, x, c, N=1024))


def fit_bates(otm: pd.DataFrame, x0=None, mid_iv=None, heston_start=None) -> tuple:
    """Starts from the Heston solution plus jump settings from both basins, so the jump part only has to explain what
    the diffusion cannot, and the optimiser is not trapped in the frequent-small-jump basin."""
    prob = _BatesProblem(otm, mid_iv)
    h = tuple(heston_start) if heston_start is not None else fit_heston(otm, mid_iv=mid_iv)
    h = tuple(np.clip(h, HESTON_BOUNDS[0], HESTON_BOUNDS[1]))
    atm_var = float(np.median(otm["mid_iv"]) ** 2)
    starts = [x0] if x0 is not None else [
        h + (1.0, -0.15, 0.15), h + (8.0, -0.03, 0.08),                   # frequent, small jumps
        h + (0.1, -0.6, 0.9), h + (0.3, -0.3, 0.5),                         # rare, large crashes
        (atm_var, 2.0, atm_var, 1.0, -0.2, 4.0, -0.05, 0.1)]
    best = None
    for s0 in starts:
        r = least_squares(prob.residuals, s0, bounds=BATES_BOUNDS, x_scale="jac", max_nfev=2000)
        if best is None or r.cost < best.cost:
            best = r
    return tuple(float(v) for v in best.x)


def bates_iv(otm, params):
    iv = np.empty(len(otm))
    for exp, s in otm.groupby("expiry", sort=False):
        idx = otm.index.get_indexer(s.index)
        K, T, c = np.exp(s["k"].values), s["T"].iloc[0], s["is_call"].values
        iv[idx] = black.implied_vol(heston.bates_price(K, T, params, c, N=1024), 1.0, K, T, c)
    return iv


def feller(params) -> float:
    """2 kappa theta / xi^2. Below 1 the variance process can hit zero (common in fitted crypto parameters)."""
    v0, kappa, theta, xi, rho = params
    return 2 * kappa * theta / xi**2


# ---------- scoring ----------
def score(otm: pd.DataFrame, model_iv: dict) -> pd.DataFrame:
    """Per expiry and model: IV RMSE (vol points), share of quotes priced inside bid-ask, mean error in spreads."""
    rows = []
    for exp, s in otm.groupby("expiry"):
        idx = otm.index.get_indexer(s.index)
        for name, iv in model_iv.items():
            m = iv[idx]
            err = m - s["mid_iv"].values
            rows.append(dict(expiry=exp, days=s["T"].iloc[0] * 365, model=name,
                             rmse_volpts=100 * np.sqrt(np.nanmean(err**2)),
                             in_spread=np.nanmean((m >= s["bid_iv"].values) & (m <= s["ask_iv"].values)),
                             mean_abs_spreads=np.nanmean(np.abs(err) / _spread(s))))
    return pd.DataFrame(rows)


# ---------- parameter identifiability ----------
def bootstrap(otm: pd.DataFrame, n=30, seed=0):
    """Redraw every mid uniformly inside its bid-ask IV range and refit. If parameters move a lot while the quotes
    stay inside the market, the data cannot pin them down, so they should not be trusted for hedging or risk."""
    rng = np.random.default_rng(seed)
    hes, sab = [], []
    for _ in range(n):
        u = rng.uniform(otm["bid_iv"].values, otm["ask_iv"].values)
        o = otm.assign(mid_iv=u)
        hes.append(fit_heston(o, mid_iv=u))
        sab.append(fit_sabr(o))
    hes = pd.DataFrame(hes, columns=HESTON_NAMES)
    rows = []
    for e in sab[0]:
        arr = np.array([b[e] for b in sab])
        rows.append(dict(expiry=e, **{f"{n}_mean": arr[:, i].mean() for i, n in enumerate(("alpha", "rho", "nu"))},
                         **{f"{n}_cv": arr[:, i].std() / abs(arr[:, i].mean()) for i, n in enumerate(("alpha", "rho", "nu"))}))
    return hes, pd.DataFrame(rows)
