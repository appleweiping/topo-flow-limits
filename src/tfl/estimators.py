"""Estimators for the latent filled-triangle support ``S``.

Mechanism (the reduction everything rests on; the identity itself is
elementary — the detection-theoretic USE of HodgeRank's curl-as-inconsistency
observation)
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
non-negative lasso. Because the lifted atoms are ALWAYS linearly independent
(limits.lifted_atoms_linearly_independent — a pair of edges lies in at most one
triangle), these second-order estimators identify supports even when ``B2`` is
rank-deficient, i.e. beyond what any first-order/subspace argument allows: they
are the achievers of the second-order regime of the identifiability dichotomy.

Naming note: ``whitened_*`` below decorrelates the MEAN of the curl statistic
(GLS/BLUE inversion ``yhat = G^+ c``); the residual noise covariance
``sigma_n^2 G^+`` is generally NOT white. The paper calls this "geometry-aware
decorrelation"; the function names keep the historical ``whitened_`` prefix
for API stability.

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


def nnls_lifted_support(
    Z: np.ndarray,
    U: np.ndarray,
    sigma_noise: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The lifted-covariance NNLS estimator (the achievability workhorse of
    the excitation-dependent theory, diagonal-excitation class).

    Model: curl-coordinate snapshots ``z_t ~ N(0, sigma_n^2 I + sum_tau
    w_tau u_tau u_tau^T)`` with ``w >= 0`` supported on ``S``. Estimator:

        w_hat = argmin_{w >= 0} || Sigma_hat - sigma_n^2 I
                                   - sum_tau w_tau u_tau u_tau^T ||_F^2,
        S_hat = { tau : w_hat_tau > threshold }.

    Consistency: Sigma_hat -> Sigma a.s.; the atom map is linear and
    injective (spark lemma), so the cone-constrained LS minimizer is unique
    and continuous in Sigma_hat, whence w_hat -> w a.s. and thresholding at
    any 0 < threshold < w_min recovers S exactly w.p. -> 1. A fully explicit
    O(1/N) failure bound (threshold = w_min/2) is
    :func:`tfl.limits.nnls_recovery_bound`.

    Parameters: ``Z`` (r, N) curl-coordinate snapshots (``Q^T F``); ``U``
    (r, p) signatures; returns ``(support_bool, w_hat)``.
    """
    from scipy.optimize import nnls as _nnls

    from tfl.limits import lifted_atom_matrix

    r, N = Z.shape
    Sig_hat = (Z @ Z.T) / N
    A = lifted_atom_matrix(U)
    s = (Sig_hat - sigma_noise**2 * np.eye(r)).ravel()
    w_hat, _ = _nnls(A, s)
    return w_hat > threshold, w_hat


def subspace_matched_support(
    Z: np.ndarray,
    U: np.ndarray,
    k: int,
    subspace_dim: int,
) -> np.ndarray:
    """First-order / matched-subspace baseline (oracle-aided: true ``k`` and
    true ``dim im B_{2,S}`` are given).

    Estimates the signal subspace as the top-``subspace_dim`` eigenvectors of
    the sample covariance and scores each candidate by the fraction of its
    signature energy inside that subspace, ``||P_hat u_tau||^2 / ||u_tau||^2``,
    selecting the top ``k``. This is the natural subspace method in the vein
    of (Hodge-aware) matched subspace detection. Its POPULATION scores depend
    on S only through ``im B_{2,S}`` — every dependent candidate ties at
    score exactly 1 — so it has no population-level margin on equal-image
    supports. At finite N it can still break ties through the eigen-
    anisotropy of the sample covariance (a second-order side channel), but
    that margin is fragile: under projector excitation (limits.py case (c))
    the within-subspace anisotropy vanishes and the method provably drops to
    chance, as does every method. NNLS retains a margin whenever the
    excitation is (approximately) diagonal. Both behaviours are pinned in
    tests/test_excitation.py.
    """
    N = Z.shape[1]
    Sig_hat = (Z @ Z.T) / N
    vals, vecs = np.linalg.eigh(Sig_hat)
    P = vecs[:, -subspace_dim:] if subspace_dim > 0 else vecs[:, :0]
    num = np.sum((P.T @ U) ** 2, axis=0)
    den = np.sum(U**2, axis=0)
    scores = num / den
    support = np.zeros(U.shape[1], dtype=bool)
    support[np.argsort(-scores)[:k]] = True
    return support


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


# ---------------------------------------------------------------------------
# Plug-in variance estimation + fully adaptive detector (supplement, Sec. S3)
# ---------------------------------------------------------------------------

def estimate_noise_sigma(dataset: FlowDataset, center: bool = False) -> float:
    """Median-based plug-in estimate of ``sigma_noise``.

    For an INACTIVE triangle the whitened mean-score satisfies
    ``score_tau * T / v0_tau ~ chi2_df`` with ``v0_tau = sigma_n^2 (G^+)_{tau tau}``
    (``df = T-1`` if centered). Normalizing each score by ``(G^+)_{tau tau}`` and
    taking the MEDIAN over candidates is therefore robust whenever fewer than
    half the candidates are active (active scores are inflated by
    ``sigma_c^2``, pushing them above the median):

        sigma_n_hat^2 = median_tau( score_tau / (G^+)_{tau tau} ) * T / chi2_median(df).
    """
    from scipy.stats import chi2

    scores = whitened_curl_scores(dataset, center=center)
    G = triangle_gram(dataset)
    Gp_diag = np.clip(np.diag(np.linalg.pinv(G)), 1e-12, None)
    T = dataset.T
    df = T - 1 if center else T
    med = float(np.median(scores / Gp_diag))
    return float(np.sqrt(med * T / chi2.median(df)))


def estimate_curl_sigma(
    dataset: FlowDataset, sigma_noise_hat: float,
    center: bool = False, alpha: float = 0.01,
) -> float:
    """Plug-in estimate of ``sigma_curl`` from the excess energy of triangles
    flagged by a conservative (Bonferroni, level ``alpha``) noise-only screen:
    on the whitened scale ``E[score_tau] = sigma_c^2 + v0_tau`` for active
    triangles, so ``sigma_c_hat^2 = mean(score_tau - v0_tau)`` over the screened
    set. Returns 0.0 when the screen flags nothing (no evidence of signal)."""
    from scipy.stats import chi2

    scores = whitened_curl_scores(dataset, center=center)
    G = triangle_gram(dataset)
    Gp_diag = np.clip(np.diag(np.linalg.pinv(G)), 1e-12, None)
    T = dataset.T
    df = T - 1 if center else T
    p = len(scores)
    v0 = sigma_noise_hat**2 * Gp_diag
    # conservative noise-only screen at FWER alpha
    gamma = v0 * chi2.ppf(1.0 - alpha / p, df=df) / T
    flagged = scores > gamma
    if not flagged.any():
        return 0.0
    excess = scores[flagged] - v0[flagged]
    return float(np.sqrt(max(float(np.mean(excess)), 0.0)))


def adaptive_whitened_detector_support(
    dataset: FlowDataset, mode: str = "bayes", alpha: float = 0.05,
    center: bool = False, refine: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Fully adaptive geometry-aware detector: estimates ``(sigma_n, sigma_c)``
    from the data (:func:`estimate_noise_sigma`, :func:`estimate_curl_sigma`)
    and runs :func:`whitened_curl_detector_support` with the plug-in values.
    Returns ``(support, sigma_curl_hat, sigma_noise_hat)``. If the screen finds
    no signal evidence (``sigma_c_hat = 0``) the returned support is empty.

    ``refine=True`` adds one refit pass: after the first detection, sigma_n is
    re-estimated on the OFF-support scores only (removing most of the
    active-triangle contamination that biases the initial median upward at
    small ``T``), and sigma_c on the on-support excess; the detector is then
    re-run once. Note the refit set is selected by the same data, so the refit
    is CONDITIONALLY BIASED (missed active triangles inflate it — measured
    ~+5% at T=10 on the strip benchmark, vanishing for T >= 20); its accuracy
    is established empirically, not by an unbiasedness argument. Requires
    fewer than half the candidates active (median breakdown otherwise: the
    screen then finds nothing and an empty support is returned, exactly as in
    the genuine no-signal case).
    """
    sn_hat = estimate_noise_sigma(dataset, center=center)
    sc_hat = estimate_curl_sigma(dataset, sn_hat, center=center)
    if sc_hat <= 0:
        return np.zeros(dataset.B2_all.shape[1], dtype=bool), 0.0, sn_hat
    support = whitened_curl_detector_support(
        dataset, sc_hat, sn_hat, mode=mode, alpha=alpha, center=center)

    if refine and support.any() and not support.all():
        scores = whitened_curl_scores(dataset, center=center)
        G = triangle_gram(dataset)
        Gp_diag = np.clip(np.diag(np.linalg.pinv(G)), 1e-12, None)
        T = dataset.T
        df = T - 1 if center else T
        # mean-based refit on the DETECTED inactive set; conditionally biased
        # by the selection (miss contamination at small T), see docstring
        off = scores[~support] / Gp_diag[~support]
        sn_hat = float(np.sqrt(np.mean(off) * T / df))
        v0 = sn_hat**2 * Gp_diag[support]
        sc_hat = float(np.sqrt(max(float(np.mean(scores[support] - v0)), 1e-12)))
        support = whitened_curl_detector_support(
            dataset, sc_hat, sn_hat, mode=mode, alpha=alpha, center=center)
    return support, sc_hat, sn_hat
