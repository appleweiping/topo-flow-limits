"""Estimators for the latent filled-triangle support ``S``.

Mechanism (the crux of the whole approach)
-------------------------------------------
Take the curl of the observed flow, ``c_t = B2.T f_t``. Because ``B2.T B1.T = 0``
and the harmonic space is both curl- and divergence-free, the **gradient and
harmonic nuisances are annihilated**:

    c_t = B2.T f_t = (B2.T B2_S) y_t + B2.T n_t .

So the curl statistic sees only the active-triangle signal plus projected noise.
Writing the triangle Gram matrix ``G = B2.T B2`` (``G_{στ}=3`` if ``σ==τ``, ``±1``
if the triangles share exactly one edge, ``0`` otherwise), the curl covariance is

    Sigma_c = sigma_curl^2 * G[:,S] G[:,S].T  +  sigma_noise^2 * G .

Recovering ``S`` from ``Sigma_c`` is a sparse PSD dictionary-selection problem in
the atoms ``{g_tau g_tau.T}`` (``g_tau`` = column ``tau`` of ``G``) — a lifted
non-negative lasso, which is where the achievability guarantee comes from.

This module provides:
  * ``curl_statistics`` / ``curl_energy_scores`` — the sufficient statistics;
  * ``energy_detector_support`` — per-triangle variance test (naive baseline &
    the estimator analyzed in the well-separated regime);
  * ``sparse_curl_covariance_support`` — the proposed convex estimator (handles
    edge-sharing confusability);
  * ``greedy_support`` — a greedy baseline standing in for greedy topology
    learning.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from tfl.generative import FlowDataset


def curl_statistics(dataset: FlowDataset) -> np.ndarray:
    """Per-snapshot curls ``C`` of shape ``(n_candidate_triangles, T)``."""
    return dataset.B2_all.T @ dataset.F


def curl_energy_scores(dataset: FlowDataset) -> np.ndarray:
    """Empirical curl energy ``(1/T) sum_t c_{tau,t}^2`` per candidate triangle."""
    C = curl_statistics(dataset)
    return np.mean(C**2, axis=1)


def triangle_gram(dataset: FlowDataset) -> np.ndarray:
    """Triangle Gram matrix ``G = B2.T B2``."""
    return dataset.B2_all.T @ dataset.B2_all


def energy_detector_support(
    dataset: FlowDataset,
    sigma_noise: float,
    fpr: float = 0.05,
) -> np.ndarray:
    """Recover ``S`` by testing each triangle's curl energy against the
    noise-only null.

    Under H0 (triangle inactive, no active edge-neighbours) the curl scalar has
    variance ``3 * sigma_noise^2`` and ``T * energy / (3 sigma_noise^2) ~ chi2_T``.
    We flag a triangle active when its energy exceeds the upper ``fpr`` quantile.
    This detector is exactly the analyzed estimator in the well-separated regime
    and the naive baseline otherwise.
    """
    scores = curl_energy_scores(dataset)
    T = dataset.T
    null_var = 3.0 * sigma_noise**2
    thresh = null_var * chi2.ppf(1.0 - fpr, df=T) / T
    return scores > thresh


def whitened_curl_scores(dataset: FlowDataset, center: bool = False) -> np.ndarray:
    """Whitened triangle scores ``(1/T) sum_t yhat_{tau,t}^2`` where
    ``yhat = G^+ c`` decorrelates the curl statistic.

    Since ``c = G[:,S] y + B2.T n``, applying the Moore-Penrose inverse of the
    triangle Gram matrix gives ``yhat = (G^+ G)[:,S] y + G^+ B2.T n``. On the
    identifiable part ``G^+ G`` acts as identity, so ``yhat_tau ~ y_tau`` for
    active triangles and pure noise (covariance ``sigma_noise^2 G^+``) otherwise.
    This removes edge-sharing leakage that the raw curl-energy detector suffers.

    ``center=True`` subtracts the temporal mean of the whitened scores first,
    which removes ANY constant background flow (e.g. a real equilibrium flow
    underlying the snapshots) at the cost of one degree of freedom: centered
    sums of squares follow ``v * chi^2_{T-1}`` instead of ``v * chi^2_T``.
    """
    C = curl_statistics(dataset)
    G = triangle_gram(dataset)
    Gp = np.linalg.pinv(G)
    Yhat = Gp @ C
    if center:
        Yhat = Yhat - Yhat.mean(axis=1, keepdims=True)
    return np.mean(Yhat**2, axis=1)


def whitened_curl_detector_support(
    dataset: FlowDataset, sigma_curl: float, sigma_noise: float,
    mode: str = "bayes", alpha: float = 0.05, center: bool = False,
) -> np.ndarray:
    """Geometry-aware detector: per-triangle two-variance test on the whitened
    scores, each with its own noise level ``v0_tau = sigma_noise^2 (G^+)_{tau tau}``
    and signal level ``v1_tau = sigma_curl^2 + v0_tau``. This is the matching
    estimator for the edge-sharing (confusable) regime.

    ``mode="bayes"`` (equal-prior threshold) suits a constant active fraction;
    ``mode="fwer"`` (Bonferroni noise-quantile) suits the sparse regime with many
    candidate triangles and few active.

    ``center=True`` makes the detector invariant to any constant background flow
    (real-data regime); thresholds then use ``T - 1`` degrees of freedom.
    """
    from tfl.limits import per_triangle_threshold

    scores = whitened_curl_scores(dataset, center=center)
    G = triangle_gram(dataset)
    Gp_diag = np.clip(np.diag(np.linalg.pinv(G)), 1e-12, None)
    T = dataset.T
    df = T - 1 if center else T
    p = len(scores)
    support = np.zeros(p, dtype=bool)
    for tau, s in enumerate(scores):
        v0 = sigma_noise**2 * Gp_diag[tau]
        v1 = sigma_curl**2 + v0
        gamma_sum = per_triangle_threshold(v0, v1, df, mode=mode, alpha=alpha, p=p)
        support[tau] = s * T > gamma_sum
    return support


def effective_curl_snr(dataset: FlowDataset, sigma_curl: float, sigma_noise: float) -> np.ndarray:
    """Per-triangle effective curl-SNR ``rho^eff_tau = sigma_curl^2 /
    (sigma_noise^2 (G^+)_{tau tau})`` governing whitened-domain identifiability."""
    G = triangle_gram(dataset)
    Gp_diag = np.clip(np.diag(np.linalg.pinv(G)), 1e-12, None)
    return sigma_curl**2 / (sigma_noise**2 * Gp_diag)


def energy_detector_bayes_support(
    dataset: FlowDataset, sigma_curl: float, sigma_noise: float
) -> np.ndarray:
    """Energy detector using the Bayes-optimal two-variance threshold.

    In the well-separated (edge-disjoint) regime every triangle shares the same
    ``(v0, v1)``, so a single optimal threshold applies and this estimator meets
    the Chernoff detection limit — the achievability side of the phase transition.
    """
    from tfl.limits import curl_variances, two_variance_bayes_threshold

    scores = curl_energy_scores(dataset)  # mean of c^2 over snapshots
    v0, v1 = curl_variances(sigma_curl, sigma_noise)
    gamma_sum = two_variance_bayes_threshold(v0, v1, dataset.T)  # threshold on SUM c^2
    return scores > gamma_sum / dataset.T


def sparse_curl_covariance_support(
    dataset: FlowDataset,
    sigma_noise: float,
    lam: float | None = None,
    tol: float = 1e-4,
) -> np.ndarray:
    """Proposed convex estimator.

    Fit the empirical curl covariance with a non-negative combination of the
    rank-one atoms ``g_tau g_tau.T`` plus the known noise floor ``sigma_noise^2 G``:

        min_{w >= 0}  || Sigmahat_c - sigma_noise^2 G - sum_tau w_tau g_tau g_tau.T ||_F^2
                       + lam * sum_tau w_tau

    Support estimate is ``{tau : w_tau > tol * max(w)}``. Solved with cvxpy; the
    ``lam`` default is a mild data-driven value. Handles edge-sharing
    confusability that the plain energy detector cannot.
    """
    import cvxpy as cp

    C = curl_statistics(dataset)
    T = dataset.T
    Sig = (C @ C.T) / T
    G = triangle_gram(dataset)
    p = G.shape[0]

    residual_target = Sig - sigma_noise**2 * G
    atoms = [np.outer(G[:, t], G[:, t]) for t in range(p)]

    if lam is None:
        lam = 0.5 * np.linalg.norm(residual_target, "fro") / max(p, 1)

    w = cp.Variable(p, nonneg=True)
    approx = sum(w[t] * atoms[t] for t in range(p))
    obj = cp.Minimize(cp.sum_squares(residual_target - approx) + lam * cp.sum(w))
    prob = cp.Problem(obj)
    prob.solve(solver=cp.CLARABEL)

    wv = np.asarray(w.value).ravel() if w.value is not None else np.zeros(p)
    wv = np.clip(wv, 0, None)
    if wv.max() <= 0:
        return np.zeros(p, dtype=bool)
    return wv > tol * wv.max()


def greedy_support(
    dataset: FlowDataset,
    sigma_noise: float,
    max_k: int | None = None,
    improve_frac: float = 0.02,
) -> np.ndarray:
    """Greedy baseline (stand-in for greedy topology learning).

    Repeatedly add the candidate triangle whose rank-one atom best reduces the
    Frobenius residual to the empirical curl covariance, stopping when the
    relative improvement falls below ``improve_frac``.
    """
    C = curl_statistics(dataset)
    T = dataset.T
    Sig = (C @ C.T) / T
    G = triangle_gram(dataset)
    p = G.shape[0]
    if max_k is None:
        max_k = p

    target = Sig - sigma_noise**2 * G
    chosen: list[int] = []
    residual = target.copy()
    prev_norm = np.linalg.norm(residual, "fro")

    for _ in range(max_k):
        best_t, best_coef, best_new = -1, 0.0, prev_norm
        for t in range(p):
            if t in chosen:
                continue
            A = np.outer(G[:, t], G[:, t])
            denom = float(np.sum(A * A))
            if denom <= 0:
                continue
            coef = max(0.0, float(np.sum(residual * A)) / denom)
            new_norm = np.linalg.norm(residual - coef * A, "fro")
            if new_norm < best_new:
                best_new, best_t, best_coef = new_norm, t, coef
        if best_t < 0 or (prev_norm - best_new) < improve_frac * prev_norm:
            break
        chosen.append(best_t)
        residual = residual - best_coef * np.outer(G[:, best_t], G[:, best_t])
        prev_norm = best_new

    support = np.zeros(p, dtype=bool)
    support[chosen] = True
    return support


def hamming_error(estimate: np.ndarray, truth: np.ndarray) -> int:
    """Number of mislabeled triangles (false positives + false negatives)."""
    return int(np.sum(np.asarray(estimate, bool) != np.asarray(truth, bool)))


def exact_recovery(estimate: np.ndarray, truth: np.ndarray) -> bool:
    return hamming_error(estimate, truth) == 0
