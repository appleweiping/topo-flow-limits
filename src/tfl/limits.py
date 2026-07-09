"""Closed-form information-theoretic limits for triangle identifiability.

The atoms of the theory live in the *curl domain*. For an isolated candidate
triangle ``tau`` the curl scalar ``c = B2_tau.T f`` is zero-mean Gaussian with

    H0 (tau inactive) :  var v0 = 3 * sigma_noise^2
    H1 (tau active)   :  var v1 = 9 * sigma_curl^2 + 3 * sigma_noise^2
                            = v0 * (1 + rho),    rho = 3 * sigma_curl^2 / sigma_noise^2

Detecting ``tau`` from ``T`` i.i.d. snapshots is therefore a **two-variance
Gaussian test**. Two quantities govern it:

* the exact minimum Bayes error :func:`two_variance_bayes_error` (finite T);
* its exponential rate, the Gaussian **Chernoff information**
  :func:`gaussian_chernoff_information`, so that the min error decays like
  ``exp(-T * C)``.

Because ``C(v0, v1) -> 0`` as ``rho -> 0`` (with ``C ~ rho^2 / 16`` to leading
order), for any fixed budget ``T`` there is a curl-SNR floor ``rho*(T)`` below
which no estimator can identify the triangle: the **curl-invisibility phase**.
Every closed form here is cross-checked against Monte-Carlo simulation in the
test-suite before it enters the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def curl_variances(sigma_curl: float, sigma_noise: float) -> tuple[float, float]:
    """Return ``(v0, v1)`` for the isolated-triangle curl scalar."""
    v0 = 3.0 * sigma_noise**2
    v1 = 9.0 * sigma_curl**2 + 3.0 * sigma_noise**2
    return v0, v1


def curl_snr_scalar(sigma_curl: float, sigma_noise: float) -> float:
    """Per-triangle curl-SNR ``rho = 3 sigma_curl^2 / sigma_noise^2`` (so
    ``v1 = v0 (1 + rho)``)."""
    return 3.0 * sigma_curl**2 / sigma_noise**2


def _chernoff_g(s: float, v0: float, v1: float) -> float:
    """Per-sample exponent ``g(s) = -log \\int p0^s p1^{1-s}`` for zero-mean
    Gaussians with variances ``v0, v1``. Convex in ``s`` and non-negative."""
    w0, w1 = 1.0 / v0, 1.0 / v1
    mix = s * w0 + (1.0 - s) * w1
    return 0.5 * (np.log(mix) - s * np.log(w0) - (1.0 - s) * np.log(w1))


def gaussian_chernoff_information(v0: float, v1: float) -> tuple[float, float]:
    """Chernoff information ``C = max_{s in [0,1]} g(s)`` and the optimizer ``s*``.

    This is the best achievable exponent of the minimum Bayes error for the
    single-triangle test: ``P_err(T) ~ exp(-T * C)``.
    """
    if abs(v0 - v1) < 1e-15:
        return 0.0, 0.5
    res = minimize_scalar(lambda s: -_chernoff_g(s, v0, v1), bounds=(0.0, 1.0), method="bounded")
    return float(-res.fun), float(res.x)


def single_triangle_chernoff(sigma_curl: float, sigma_noise: float) -> float:
    """Chernoff information for detecting one isolated triangle."""
    v0, v1 = curl_variances(sigma_curl, sigma_noise)
    C, _ = gaussian_chernoff_information(v0, v1)
    return C


def two_variance_bayes_error(v0: float, v1: float, T: int, n_grid: int = 4000) -> float:
    """Exact minimum Bayes error (equal priors) of the energy test on ``T``
    i.i.d. samples.

    The optimal statistic is the energy ``E = sum_t c_t^2``; under H0
    ``E / v0 ~ chi2_T`` and under H1 ``E / v1 ~ chi2_T``. The Bayes-optimal rule
    thresholds ``E``; we minimize the total error over the threshold ``gamma``.
    """
    if v1 <= v0:
        v0, v1 = min(v0, v1), max(v0, v1)
    lo = 0.0
    hi = chi2.ppf(1 - 1e-9, df=T) * v1
    gammas = np.linspace(lo, hi, n_grid)
    # H0 error: declare H1 when E > gamma  -> P0(E > gamma)
    p0_err = chi2.sf(gammas / v0, df=T)
    # H1 error: declare H0 when E < gamma  -> P1(E < gamma)
    p1_err = chi2.cdf(gammas / v1, df=T)
    total = 0.5 * (p0_err + p1_err)
    return float(total.min())


def single_triangle_bayes_error(sigma_curl: float, sigma_noise: float, T: int) -> float:
    v0, v1 = curl_variances(sigma_curl, sigma_noise)
    return two_variance_bayes_error(v0, v1, T)


def snapshots_for_target_error(
    sigma_curl: float, sigma_noise: float, target_error: float = 0.05
) -> float:
    """Chernoff-rate estimate of the snapshots needed for a target single-triangle
    error: ``T ~ log(1 / target_error) / C``."""
    C = single_triangle_chernoff(sigma_curl, sigma_noise)
    if C <= 0:
        return np.inf
    return np.log(1.0 / target_error) / C


def invisibility_curl_snr_floor(
    sigma_noise: float, T: int, target_error: float = 0.05
) -> float:
    """Curl-SNR floor ``rho*`` at budget ``T``: the smallest ``rho`` whose Chernoff
    rate still meets ``exp(-T C) <= target_error``. Below ``rho*`` the triangle is
    invisible. Solved by a monotone bisection on ``rho``."""
    need_C = np.log(1.0 / target_error) / T

    def C_of_rho(rho: float) -> float:
        v0 = 3.0 * sigma_noise**2
        v1 = v0 * (1.0 + rho)
        C, _ = gaussian_chernoff_information(v0, v1)
        return C

    lo, hi = 1e-6, 1e6
    if C_of_rho(hi) < need_C:
        return np.inf
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if C_of_rho(mid) < need_C:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))
