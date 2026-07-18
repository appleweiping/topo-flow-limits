"""Non-oracle support selection for the lifted-covariance NNLS estimator.

The paper's headline rule ``S_hat = {tau : w_hat_tau > w_min/2}`` uses the true
minimum weight ``w_min`` and the true ``sigma_n`` -- an ORACLE theorem setting.
This module provides reproducible NON-oracle rules that use neither:

  * :func:`estimate_sigma_n_curl` -- ``sigma_n`` from the median eigenvalue of
    the curl-coordinate sample covariance.  This is a *noise* eigenvalue ONLY
    when the active curl-covariance rank is below ~``r/2`` (a noise-
    identifiability / sparsity ceiling); beyond it the median eigenvalue is
    signal-inflated and ``sigma_n`` is overestimated.
  * :func:`bic_support_path` -- walk the NNLS solution path (nest the support by
    decreasing ``w_hat``) and pick the support minimizing the Gaussian-model
    BIC ``N[log det Sigma_S + tr(Sigma_S^{-1} Sigma_hat)] + |S| log N``.

IMPORTANT (round-9 retraction of an earlier overclaim).  The BIC selector is
NOT unconditionally consistent.  What is true:

  * GIVEN an identifiable ``sigma_n`` (e.g. the true value), classical Gaussian-
    BIC selection consistency holds under fixed ``p``, lifted-atom injectivity
    on ``S*`` (``rank(G o G)|_{S*}`` full), a beta-min gap, and a unique
    minimizer.  Numerically the oracle-``sigma_n`` path recovers even the fully
    DENSE ``k=p`` support on ``K6`` (``experiments/run_bic_boundary.py``).
  * The FULLY non-oracle rule (median-eigenvalue ``sigma_n``) is therefore an
    EMPIRICAL selector, valid only inside the noise-identifiable regime.  When
    the support is dense enough that the active curl covariance approaches full
    rank ``r``, the median rule FAILS even as ``N -> infinity`` -- a counter-
    example (``K6``, ``k>=12``, ``N=1e5`` -> 0 recovery) is reproduced in
    ``run_bic_boundary.py`` and locked by ``tests/test_bic_boundary.py``.  Do
    NOT describe the median-sigma_n selector as "consistent" for dense supports.

  * :func:`sample_split_support` -- assumption-light held-out-residual variant.
"""
from __future__ import annotations

import numpy as np


def estimate_sigma_n_curl(Sigma_hat: np.ndarray) -> float:
    """Non-oracle ``sigma_n`` from the curl covariance: ``sqrt(median
    eigenvalue)``.  The population ``Sigma_z = sigma_n^2 I + U_S Gamma U_S^T``
    has ``sigma_n^2`` on all directions orthogonal to the (low-dim) signal
    range, so the median eigenvalue is a noise eigenvalue whenever the active
    range spans fewer than half the ``r`` curl directions."""
    ev = np.linalg.eigvalsh(Sigma_hat)
    return float(np.sqrt(max(np.median(ev), 1e-12)))


def _sigma_model(U: np.ndarray, support: np.ndarray, w: np.ndarray,
                 sigma_n: float) -> np.ndarray:
    r = U.shape[0]
    Us = U[:, support]
    return sigma_n ** 2 * np.eye(r) + (Us * w[support][None, :]) @ Us.T


def gaussian_cov_bic(Sigma_hat: np.ndarray, Sigma_model: np.ndarray,
                     N: int, dof: int) -> float:
    """BIC (up to an additive constant in ``N r log 2pi``) of a zero-mean
    Gaussian with covariance ``Sigma_model`` given sample covariance
    ``Sigma_hat``: ``N[log det Sigma + tr(Sigma^{-1} Sigma_hat)] + dof log N``."""
    sign, logdet = np.linalg.slogdet(Sigma_model)
    if sign <= 0:
        return np.inf
    tr = float(np.trace(np.linalg.solve(Sigma_model, Sigma_hat)))
    return float(N * (logdet + tr) + dof * np.log(max(N, 2)))


def bic_support_path(Sigma_hat: np.ndarray, U: np.ndarray, w_hat: np.ndarray,
                     sigma_n: float, N: int, max_support: int | None = None
                     ) -> tuple[np.ndarray, dict]:
    """Select the support along the NNLS path by minimum Gaussian-model BIC.

    Nest supports ``S_0 = {} subset S_1 subset ...`` by adding candidates in
    decreasing ``w_hat``; refit is unnecessary (weights come from ``w_hat``).
    Returns ``(support_mask, info)`` with the BIC curve and chosen size."""
    p = len(w_hat)
    order = np.argsort(-w_hat)
    pos = int(np.sum(w_hat > 0))
    kmax = pos if max_support is None else min(pos, max_support)
    bic_curve = []
    best_bic, best_j = np.inf, 0
    for j in range(0, kmax + 1):
        support = np.zeros(p, bool)
        if j > 0:
            support[order[:j]] = True
        Sigma_S = _sigma_model(U, support, w_hat, sigma_n)
        bic = gaussian_cov_bic(Sigma_hat, Sigma_S, N, dof=j)
        bic_curve.append(bic)
        if bic < best_bic:
            best_bic, best_j = bic, j
    mask = np.zeros(p, bool)
    if best_j > 0:
        mask[order[:best_j]] = True
    return mask, {"bic_curve": bic_curve, "selected_size": best_j,
                  "sigma_n_used": float(sigma_n)}


def bic_nonoracle_support(Z: np.ndarray, U: np.ndarray, N: int | None = None,
                          sigma_n: float | None = None, solver: str = "fista",
                          max_support: int | None = None
                          ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fully non-oracle estimator: estimate ``sigma_n`` (if not given), solve
    the matrix-free NNLS, then pick the support by BIC.  Returns
    ``(support_mask, w_hat, info)``."""
    from tfl.estimators_mf import nnls_lifted_fista, nnls_lifted_active_set
    r, Nz = Z.shape
    if N is None:
        N = Nz
    Sigma_hat = (Z @ Z.T) / N
    if sigma_n is None:
        sigma_n = estimate_sigma_n_curl(Sigma_hat)
    C = Sigma_hat - sigma_n ** 2 * np.eye(r)
    if solver == "active_set":
        w_hat = nnls_lifted_active_set(C, U)
    else:
        w_hat = nnls_lifted_fista(C, U)
    mask, info = bic_support_path(Sigma_hat, U, w_hat, sigma_n, N,
                                  max_support=max_support)
    info["sigma_n_estimated"] = sigma_n
    return mask, w_hat, info


def sample_split_support(Z: np.ndarray, U: np.ndarray, sigma_n: float | None = None,
                         frac: float = 0.5, solver: str = "fista"
                         ) -> tuple[np.ndarray, dict]:
    """Assumption-light held-out-residual selection: fit ``w_hat`` on the first
    ``frac`` of snapshots; along the sorted-``w_hat`` path, pick the support
    minimizing the Frobenius residual ``||Sigma_hat_2 - sigma_n^2 I -
    sum_{tau in S} w_tau u_tau u_tau^T||_F`` on the held-out half."""
    from tfl.estimators_mf import nnls_lifted_fista, nnls_lifted_active_set
    r, N = Z.shape
    n1 = max(2, int(frac * N))
    Z1, Z2 = Z[:, :n1], Z[:, n1:]
    S1 = (Z1 @ Z1.T) / Z1.shape[1]
    S2 = (Z2 @ Z2.T) / Z2.shape[1]
    if sigma_n is None:
        sigma_n = estimate_sigma_n_curl(S1)
    C1 = S1 - sigma_n ** 2 * np.eye(r)
    w_hat = (nnls_lifted_active_set(C1, U) if solver == "active_set"
             else nnls_lifted_fista(C1, U))
    order = np.argsort(-w_hat)
    pos = int(np.sum(w_hat > 0))
    C2 = S2 - sigma_n ** 2 * np.eye(r)
    best, best_j = np.inf, 0
    p = len(w_hat)
    for j in range(0, pos + 1):
        support = np.zeros(p, bool)
        if j > 0:
            support[order[:j]] = True
        Us = U[:, support]
        resid = C2 - (Us * w_hat[support][None, :]) @ Us.T
        val = float(np.sum(resid * resid))
        if val < best:
            best, best_j = val, j
    mask = np.zeros(p, bool)
    if best_j > 0:
        mask[order[:best_j]] = True
    return mask, {"selected_size": best_j, "sigma_n_used": float(sigma_n)}
