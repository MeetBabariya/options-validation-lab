"""SABR (Hagan et al. 2002) lognormal implied-vol approximation with beta = 1, the usual choice for crypto,
where the forward is a price that scales multiplicatively.

dF = alpha F dW1,  dalpha = nu alpha dW2,  d<W1, W2> = rho dt
"""
import numpy as np


def implied_vol(k, T, alpha, rho, nu):
    """k = log(K/F). With beta = 1 the formula depends on F and K only through k."""
    k = np.asarray(k, float)
    z = -nu / alpha * k  # Hagan's z uses log(F/K) = -k
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        ratio = np.where(np.abs(z) < 1e-7, 1 - 0.5 * rho * z, z / x)  # z/x(z) -> 1 as z -> 0
    return alpha * ratio * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho**2) * nu**2 / 24) * T)
