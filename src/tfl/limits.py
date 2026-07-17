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
order), for any fixed budget ``T`` there is a curl-SNR scale ``rho*(T)`` below
which no estimator attains the target error exponent: the
**curl-invisibility phase**. (``rho*`` solves the exponent equation; the exact
finite-``T`` boundary sits an order-one factor below it — see
:func:`invisibility_curl_snr_floor`.) Every closed form here is cross-checked
against Monte-Carlo simulation in the test-suite before it enters the paper.
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


def two_variance_bayes_threshold(v0: float, v1: float, T: int, n_grid: int = 4000) -> float:
    """Energy threshold ``gamma*`` achieving the minimum Bayes error for the
    two-variance test (declare active when energy ``> gamma*``)."""
    if v1 <= v0:
        v0, v1 = min(v0, v1), max(v0, v1)
    hi = chi2.ppf(1 - 1e-9, df=T) * v1
    gammas = np.linspace(0.0, hi, n_grid)
    total = 0.5 * (chi2.sf(gammas / v0, df=T) + chi2.cdf(gammas / v1, df=T))
    return float(gammas[int(np.argmin(total))])


def single_triangle_bayes_error(sigma_curl: float, sigma_noise: float, T: int) -> float:
    v0, v1 = curl_variances(sigma_curl, sigma_noise)
    return two_variance_bayes_error(v0, v1, T)


def exact_recovery_probability(
    sigma_curl: float, sigma_noise: float, T: int, n_active: int, n_inactive: int
) -> float:
    """Exact P(exact support recovery) in the well-separated (edge-disjoint)
    regime, where triangles are classified independently by the common
    Bayes-optimal energy threshold ``gamma*``.

    ``P = (1 - P_miss)^{n_active} * (1 - P_fa)^{n_inactive}`` with
    ``P_miss = chi2_T(gamma*/v1)`` and ``P_fa = 1 - chi2_T(gamma*/v0)``. This is
    the finite-sample achievability curve the simulation validates.
    """
    v0, v1 = curl_variances(sigma_curl, sigma_noise)
    gamma = two_variance_bayes_threshold(v0, v1, T)
    p_miss = chi2.cdf(gamma / v1, df=T)
    p_fa = chi2.sf(gamma / v0, df=T)
    return float((1.0 - p_miss) ** n_active * (1.0 - p_fa) ** n_inactive)


def whitened_variances(
    Gp_diag: np.ndarray, sigma_curl: float, sigma_noise: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle whitened-domain variances.

    After whitening ``yhat = G^+ c``, triangle ``tau`` has
    ``v0_tau = sigma_noise^2 (G^+)_{tau tau}`` (inactive) and
    ``v1_tau = sigma_curl^2 + v0_tau`` (active). The effective per-triangle
    curl-SNR is ``rho^eff_tau = sigma_curl^2 / v0_tau``; ``(G^+)_{tau tau}`` is the
    geometry/confusability factor (``1/3`` for an isolated triangle, larger when
    curl signatures overlap).
    """
    v0 = sigma_noise**2 * np.asarray(Gp_diag, float)
    v1 = sigma_curl**2 + v0
    return v0, v1


def per_triangle_threshold(
    v0: float, v1: float, T: int, mode: str = "bayes", alpha: float = 0.05, p: int = 1
) -> float:
    """Energy threshold (on the SUM statistic) for one triangle.

    * ``mode="bayes"``: equal-prior Bayes-optimal threshold — right when active
      triangles are a constant fraction of the candidates.
    * ``mode="fwer"``: noise-only upper quantile at per-test level ``alpha/p``
      (Bonferroni) — right for the sparse regime (few active among many
      candidates), where the equal-prior rule over-detects.
    """
    if mode == "bayes":
        return two_variance_bayes_threshold(v0, v1, T)
    if mode == "fwer":
        return v0 * chi2.ppf(1.0 - alpha / max(p, 1), df=T)
    raise ValueError(f"unknown threshold mode {mode!r}")


def _per_triangle_error_probs(
    v0s: np.ndarray, v1s: np.ndarray, active: np.ndarray, T: int,
    mode: str, alpha: float,
) -> np.ndarray:
    """Marginal per-triangle error probabilities of the whitened detector.

    The *marginal* law of each whitened score is an exact two-variance test
    (rigorous whenever ``B2`` has full column rank, since then ``G`` is
    invertible and ``yhat = G^{-1} c`` has mean exactly ``y_tau 1{tau in S}``
    with noise variance ``sigma_n^2 (G^{-1})_{tau tau}``). Correlations across
    triangles affect only the JOINT law, not these marginals.
    """
    active = np.asarray(active, bool)
    p = len(v0s)
    errs = np.zeros(p)
    for tau in range(p):
        v0, v1 = float(v0s[tau]), float(v1s[tau])
        gamma = per_triangle_threshold(v0, v1, T, mode=mode, alpha=alpha, p=p)
        if active[tau]:
            errs[tau] = chi2.cdf(gamma / v1, df=T)      # P(miss)
        else:
            errs[tau] = chi2.sf(gamma / v0, df=T)       # P(false alarm)
    return errs


def heterogeneous_exact_recovery_probability(
    v0s: np.ndarray, v1s: np.ndarray, active: np.ndarray, T: int,
    mode: str = "bayes", alpha: float = 0.05,
) -> float:
    """Independence APPROXIMATION to the whitened detector's exact-recovery
    probability: the product of the (exact) per-triangle marginal success
    probabilities.

    This product is exact when the whitened scores are independent — e.g. for
    edge-disjoint candidates — and is an approximation otherwise, because the
    whitened noise covariance ``sigma_n^2 G^{+}`` retains off-diagonal terms.
    For a bound that is rigorous under arbitrary correlations use
    :func:`heterogeneous_recovery_union_bound`. Empirically the approximation
    tracks Monte-Carlo closely when ``G`` is diagonally dominant (see the
    confusability experiment)."""
    errs = _per_triangle_error_probs(v0s, v1s, active, T, mode, alpha)
    return float(np.prod(1.0 - errs))


def heterogeneous_recovery_union_bound(
    v0s: np.ndarray, v1s: np.ndarray, active: np.ndarray, T: int,
    mode: str = "bayes", alpha: float = 0.05,
) -> float:
    """RIGOROUS lower bound on the whitened detector's exact-recovery
    probability: ``P(exact) >= 1 - sum_tau P_err,tau`` (union bound over the
    exact per-triangle marginal error probabilities). Valid regardless of the
    correlation structure of the whitened scores."""
    errs = _per_triangle_error_probs(v0s, v1s, active, T, mode, alpha)
    return float(max(0.0, 1.0 - errs.sum()))


def recovery_contour_rho(
    sigma_noise: float, T: int, n_active: int, n_inactive: int, level: float = 0.5
) -> float:
    """Curl-SNR ``rho`` at which exact-recovery probability equals ``level``
    (the theoretical phase-transition contour). Monotone bisection on ``rho``."""
    def P_of_rho(rho: float) -> float:
        sc = np.sqrt(rho * sigma_noise**2 / 3.0)
        return exact_recovery_probability(sc, sigma_noise, T, n_active, n_inactive)

    lo, hi = 1e-4, 1e4
    if P_of_rho(hi) < level:
        return np.inf
    if P_of_rho(lo) > level:
        return 0.0
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if P_of_rho(mid) < level:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


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
    """Curl-SNR scale ``rho*`` at budget ``T``: the ``rho`` at which the optimal
    error EXPONENT meets the budget, i.e. ``exp(-T C(rho)) = target_error``.

    Calibration note: this solves the exponent equation, which is the
    Chernoff (achievability-side) calibration of the phase boundary; it is a
    scale, not a hard impossibility threshold. The converse content is at the
    exponent level (no test achieves an error exponent better than ``C``), so
    the exact finite-``T`` boundary sits below this value by an
    order-one factor (the exact Bayes error can meet ``target_error`` at
    ``rho`` roughly 40-50% smaller at moderate ``T``). Both interpretations
    share the ``rho* ~ sqrt(log(1/target_error)/T)`` law. Solved by a
    monotone bisection on ``rho``."""
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


# ---------------------------------------------------------------------------
# Fano-type converses for JOINT support recovery (supplement, Sec. S1)
# ---------------------------------------------------------------------------

def gaussian_kl_two_variance(v_from: float, v_to: float) -> float:
    """``KL( N(0, v_from) || N(0, v_to) )`` for zero-mean scalar Gaussians."""
    r = v_from / v_to
    return 0.5 * (r - 1.0 - np.log(r))


def _log_binom(p: int, k: int) -> float:
    from scipy.special import gammaln
    return float(gammaln(p + 1) - gammaln(k + 1) - gammaln(p - k + 1))


def fano_min_snapshots(p: int, k: int, rho: float, err: float = 0.5) -> float:
    """Fano lower bound on the number of snapshots needed to recover a
    uniformly random size-``k`` support among ``p`` edge-disjoint candidates
    with error probability at most ``err`` (GAUSSIAN signals and noise).

    Mechanics: with edge-disjoint candidates the curl coordinates are
    independent, and taking the all-inactive law ``P_0`` as the Fano reference
    gives ``I(S; C^N) <= N * k * KL(N(0,v1) || N(0,v0))`` exactly, with
    ``KL_1 = (rho - log(1+rho))/2 ~ rho^2/4``. Fano then yields

        N  >=  ( (1-err) * log C(p,k) - log 2 ) / ( k * KL_1 ) .

    The bound scales as ``N >~ 4 (1-err) log(p/k) / rho^2``: JOINT recovery
    pays a ``log p`` factor on top of the single-triangle Chernoff budget.
    """
    if not (0 < k < p):
        raise ValueError("need 0 < k < p")
    kl1 = gaussian_kl_two_variance(1.0 + rho, 1.0)  # KL(v1||v0), v0-normalized
    numer = (1.0 - err) * _log_binom(p, k) - np.log(2.0)
    if numer <= 0:
        return 0.0
    return float(numer / (k * kl1))


def signal_agnostic_fano_min_snapshots(
    p: int, k: int, rho: float, err: float = 0.5
) -> float:
    """Fano lower bound valid for ARBITRARY zero-mean triangle-signal
    distributions with per-coordinate variance ``sigma_c^2`` (Gaussian noise).

    Mechanics (max-entropy budget): under the uniform size-``k`` prior each
    curl coordinate has marginal variance ``v0 (1 + rho k/p)``; per snapshot,
    each coordinate's entropy is at most Gaussian at that variance while its
    conditional entropy given the support is at least the Gaussian-noise
    entropy, so ``I(S; c) <= (p/2) log(1 + rho k/p)``. Hence

        N  >=  ( (1-err) * log C(p,k) - log 2 ) / ( (p/2) log(1 + rho k/p) ) .

    Weaker than :func:`fano_min_snapshots` for small ``rho`` (by a factor
    ``rho``), but it holds with no distributional assumption on the signal and
    is the TIGHTER of the two for large ``rho``; a valid converse may take the
    pointwise maximum of both.
    """
    if not (0 < k < p):
        raise ValueError("need 0 < k < p")
    cap = 0.5 * p * np.log1p(rho * k / p)
    numer = (1.0 - err) * _log_binom(p, k) - np.log(2.0)
    if numer <= 0:
        return 0.0
    return float(numer / cap)


def fano_rho_floor(p: int, k: int, N: int, err: float = 0.5) -> float:
    """Invert :func:`fano_min_snapshots` in ``rho``: the curl-SNR below which
    NO estimator recovers a random size-``k`` support from ``N`` snapshots with
    error <= ``err``. Monotone bisection on ``rho``."""
    numer = (1.0 - err) * _log_binom(p, k) - np.log(2.0)
    if numer <= 0:
        return 0.0
    need_kl = numer / (k * N)

    def kl_of(rho: float) -> float:
        return gaussian_kl_two_variance(1.0 + rho, 1.0)

    lo, hi = 1e-9, 1e9
    if kl_of(hi) < need_kl:
        return float("inf")
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if kl_of(mid) < need_kl:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


# ---------------------------------------------------------------------------
# Isotropic-excitation identifiability and its price
# ---------------------------------------------------------------------------
#
# SCOPE NOTE: the statements in this section live inside the DIAGONAL /
# isotropic excitation regime (a) of the hierarchy above. What the excitation
# section makes precise is that "random signals" per se do NOT remove the
# rank obstruction — only excitation STRUCTURE does:
#
#   FIRST ORDER  (a single snapshot / unknown deterministic signals): the flow
#   mean set is  im B_{2,S}. Supports with equal column-image are genuinely
#   indistinguishable — identifiable only modulo ker B2. On K_n the curl
#   subspace has dim C(n-1, 2) against C(n, 3) candidates: the 3/n DoF ratio.
#
#   ISOTROPIC/DIAGONAL EXCITATION (regime (a)): the flow covariance
#   carries sigma_c^2 * sum_{tau in S} b_tau b_tau^T, strictly finer than the
#   column space; the lifted atoms are always linearly independent (see
#   `lifted_atoms_linearly_independent`), so S -> Sigma(S) is injective and
#   every support is identifiable at any rank deficiency. Under ARBITRARY PSD
#   excitation a single covariance identifies only the realized range
#   R = im(U_S Gamma_S^{1/2}) (= im U_S whenever Gamma_S > 0; a singular
#   Gamma can hide part of the image and even the support, case (b)), and the
#   projector excitation makes equal-image supports exactly indistinguishable
#   (case (c) above).
#
# Within case (a) the price of geometry is a SAMPLE-COMPLEXITY separation,
# not an impossibility: distinguishing an equal-image confuser pair costs
# N ~ log(1/delta) / C_G snapshots, where C_G is the Gaussian Chernoff
# information between the two covariances (computed in the r-dimensional curl
# coordinate z = Q^T f, which annihilates the gradient nuisance and the
# candidate-orthogonal harmonic nuisance h in ker B2^T).
# Write D_S = sum_{tau in S} u_tau u_tau^T (DIMENSIONLESS; ||u||^2 = 3), so the
# covariance signal is M_S = sigma_c^2 D_S and the leading Chernoff exponent is
#     C_G = rho_2^2 || Delta D ||_F^2 / 16 + o(rho_2^2)
#         = || Delta M ||_F^2 / (16 sigma_n^4) + o(rho_2^2),   rho_2 = sigma_c^2/sigma_n^2
# (the two forms are identical; do NOT combine rho_2^2 with ||Delta M||, which
# would double-count sigma_c^4). For the minimal EQUAL-CARDINALITY confusers —
# single-triangle swaps inside a tetrahedron — ||Delta D||_F^2 = 16 exactly
# (9 + 9 - 2, from ||u||^2 = 3 and u_a.u_b = ±1), giving C_G = rho_2^2 (1+o(1)).
# The UNRESTRICTED worst case is the subset confuser (S vs S plus a hosted
# tetrahedron's fourth face), ||Delta D||_F^2 = 9, C_G = (9/16) rho_2^2 (1+o(1))
# — exactly the isolated-triangle exponent.
# All constants below are validated in tests/test_second_order.py, incl.
# exhaustively over all 2^10 supports of K5.


# ---------------------------------------------------------------------------
# Excitation-dependent identifiability: three excitation regimes (the core
# theory). The regimes are nested/overlapping constraints on Gamma_S, not a
# partition -- e.g. sigma_c^2 I is both diagonal (regime a) and a special
# full-rank PSD (regime b).
# ---------------------------------------------------------------------------
#
# Triangle excitation is y_S ~ N(0, Gamma_S) with Gamma_S PSD. The POPULATION
# curl covariance is
#     Sigma_z(S, Gamma_S) = sigma_n^2 I_r + M,   M = U_S Gamma_S U_S^T,
# with U = Q^T B2 (sigma_n known throughout). The population covariance
# determines M EXACTLY (M = Sigma_z - sigma_n^2 I); the question is what M
# then reveals about (S, Gamma_S). This depends on the regime:
#
#  (a) Gamma_S POSITIVE DIAGONAL (known or unknown): the weighted support
#      (S, diag weights) is identifiable at ANY rank deficiency, because the
#      lifted atoms {u_tau u_tau^T} are always linearly independent (spark
#      lemma) and M = sum_tau w_tau u_tau u_tau^T with w_tau = gamma_tau
#      1{tau in S}. (On K_n with sigma_n UNKNOWN there is exactly one ambiguous
#      direction: sum_ALL u_tau u_tau^T = n I_r, so a uniform weight shift
#      trades off against the noise floor.)
#  (b) Gamma_S ARBITRARY PSD. STRUCTURAL CHARACTERIZATION: the achievable-
#      covariance set of a support is
#          C_S = {U_S Gamma U_S^T : Gamma PSD} = {M PSD : im M ⊆ im U_S},
#      so, given the population covariance (equivalently M), the set of
#      supports that can have produced it is exactly
#          {S : im M ⊆ im U_S}.
#      Consequences:
#        * The REALIZED RANGE R = im M = im(U_S Gamma_S^{1/2}) ⊆ im U_S is
#          always readable from the population covariance. It need NOT equal
#          the candidate image im U_S: arbitrary PSD does NOT determine
#          im B_{2,S}. A support with a LARGER candidate image and a suitable
#          Gamma can reproduce the same M, so the support is not identified.
#        * If Gamma_S > 0 then R = im U_S (full-rank excitation reveals the
#          image). The converse holds only when B_{2,S} has full column rank
#          (U_S injective); if U_S has a kernel, some singular Gamma_S still
#          attain im U_S. So R = im U_S is NOT equivalent to Gamma_S > 0 in
#          general.
#        * STRONG COUNTEREXAMPLE (strict-positive-diagonal, on K4 all 4 faces):
#          c = (1,-1,1,-1) in ker U; v = e1 - 0.25 c = (.75,.25,-.25,.25) has
#          all entries nonzero; Gamma' = v v^T has POSITIVE DIAGONAL and rank 1,
#          and U Gamma' U^T = (Uv)(Uv)^T = u0 u0^T (since Uc=0). So S'=all 4
#          faces with Gamma' gives the SAME M as S={face0} with Gamma=[1], yet
#          dim im B_{2,S}=1 != 3=dim im B_{2,S'}. Positive per-triangle variance
#          is NOT enough; what regime (a) additionally requires is that Gamma be
#          DIAGONAL (uncorrelated). Gamma' is correlated (off-diagonal ~0.19),
#          which is exactly what breaks identifiability here.
#  (c) Gamma_S = sigma_c^2 (B_{2,S}^T B_{2,S})^+ (projector excitation): a
#      SPECIFIC full-rank-on-its-own-image excitation for which
#      U_S Gamma_S U_S^T = sigma_c^2 P_im(U_S) EXACTLY, so DISTINCT supports
#      with the SAME image induce IDENTICAL population covariances at every SNR
#      and N -- the sharp equal-image indistinguishability. (This is the
#      Hodge-smoothness prior Gamma = L_2^+ used by prior topology learning.)
#
# All statements are verified numerically in tests/test_excitation.py.


def excitation_covariance(
    U: np.ndarray, support: np.ndarray, Gamma: np.ndarray, sigma_noise: float
) -> np.ndarray:
    """Curl-coordinate covariance ``sigma_n^2 I + U_S Gamma U_S^T`` for a
    general PSD triangle-excitation covariance ``Gamma`` ((k, k), ordered as
    the active columns of ``U``)."""
    support = np.asarray(support, bool)
    Us = U[:, support]
    return sigma_noise**2 * np.eye(U.shape[0]) + Us @ Gamma @ Us.T


def realized_range_dim(U: np.ndarray, support: np.ndarray,
                       Gamma: np.ndarray, tol: float = 1e-9) -> int:
    """Dimension of the REALIZED RANGE ``R = im(U_S Gamma U_S^T)
    = im(Sigma_z - sigma_n^2 I)`` -- the subspace the population covariance
    always reveals (regime (b)). ``R ⊆ im U_S`` always; ``Gamma > 0`` gives
    ``R = im U_S``. The converse (``R = im U_S => Gamma > 0``) holds only when
    ``U_S`` is injective (``B_{2,S}`` full column rank); if ``U_S`` has a
    kernel, some singular ``Gamma`` still attain ``im U_S``, so a singular
    ``Gamma`` does NOT always shrink the range."""
    support = np.asarray(support, bool)
    Us = U[:, support]
    M = Us @ Gamma @ Us.T
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * max(1.0, s[0])).sum())


def feasible_supports(U: np.ndarray, M: np.ndarray,
                      candidate_supports, tol: float = 1e-8) -> list:
    """Regime (b) structural characterization: the supports whose achievable-
    covariance set ``C_S = {M PSD : im M ⊆ im U_S}`` contains a given population
    covariance-signal ``M = Sigma_z - sigma_n^2 I`` are exactly
    ``{S : im M ⊆ im U_S}``. Returns the sublist of ``candidate_supports`` (each
    a boolean mask) that could have produced ``M``. When this set has more than
    one element the support is NOT identified by the population covariance."""
    if M.shape[0] == 0 or np.linalg.norm(M) < tol:
        imM = np.zeros((U.shape[0], 0))
    else:
        Uu, s, _ = np.linalg.svd(M, full_matrices=False)
        imM = Uu[:, s > tol * max(1.0, s[0])]
    out = []
    for S in candidate_supports:
        S = np.asarray(S, bool)
        Us = U[:, S]
        if Us.shape[1] == 0:
            if imM.shape[1] == 0:
                out.append(S)
            continue
        P = Us @ np.linalg.pinv(Us)
        if imM.shape[1] == 0 or np.linalg.norm(imM - P @ imM) < tol * max(1.0, np.linalg.norm(imM)):
            out.append(S)
    return out


def strict_positive_diagonal_witness(
    U: np.ndarray, tau_keep: int, kernel_support: np.ndarray,
    sigma_noise: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regime-(b) STRONG witness (strictly positive diagonal excitation).

    On a support ``kernel_support`` whose signatures ``U_S`` have a nontrivial
    kernel (e.g. all four faces of a K4 tetrahedron), pick ``c in ker U_S`` and
    ``v = e_j + t c`` (``j`` = local index of ``tau_keep``) with ``t`` chosen so
    every entry of ``v`` is nonzero. Then ``Gamma' = v v^T`` has POSITIVE
    DIAGONAL and rank 1, and ``U_S Gamma' U_S^T = (U_S v)(U_S v)^T =
    u_{tau_keep} u_{tau_keep}^T`` (since ``U_S c = 0``). So the large support
    ``kernel_support`` with the strictly-positive-diagonal (but CORRELATED)
    ``Gamma'`` gives the same population covariance as the singleton
    ``{tau_keep}`` with unit variance -- positive per-triangle variance does
    NOT identify the support; regime (a) additionally needs ``Gamma`` DIAGONAL.

    Returns ``(Sigma_big, Sigma_single, support_big, support_single)``.
    """
    S = np.asarray(kernel_support, bool)
    idx = np.where(S)[0]
    j = int(np.where(idx == tau_keep)[0][0])   # local index of tau_keep in S
    Us = U[:, S]
    # kernel direction of U_S: trailing right-singular vectors past the rank
    _, sv, Vt = np.linalg.svd(Us, full_matrices=True)   # Vt is (k, k)
    rank = int((sv > 1e-9 * max(1.0, sv[0])).sum())
    if rank >= Vt.shape[0]:
        raise ValueError("kernel_support must have a rank-deficient U_S (nontrivial kernel)")
    c = Vt[rank]
    e = np.zeros(len(idx)); e[j] = 1.0
    # choose t so that v = e + t c has all-nonzero entries; scan a few values
    for t in (-e[j] / c[j] * 0.25 if abs(c[j]) > 1e-12 else 0.25, 0.3, -0.3, 0.5, -0.5, 0.7):
        v = e + t * c
        if np.all(np.abs(v) > 1e-9):
            break
    else:
        raise ValueError("could not find t with all-nonzero v")
    Gam = np.outer(v, v)
    Sig_big = excitation_covariance(U, S, Gam, sigma_noise)
    Ssingle = np.zeros(U.shape[1], bool); Ssingle[tau_keep] = True
    Sig_single = excitation_covariance(U, Ssingle, np.array([[1.0]]), sigma_noise)
    return Sig_big, Sig_single, S, Ssingle


def singular_gamma_equal_covariance_witness(
    U: np.ndarray, tau_in: int, tau_extra: int, sigma_noise: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple DEGENERATE regime-(b) example (a zero-variance triangle): supports
    ``S={tau_in}`` and ``S'={tau_in, tau_extra}`` with ``Gamma_S=[[1]]`` and
    ``Gamma_S'=diag(1,0)`` give IDENTICAL curl covariances although
    ``dim im B_{2,S'} > dim im B_{2,S}``. This is the easy case (one triangle
    inactive); the STRONG statement -- that even a strictly-positive-diagonal
    (but correlated) excitation fails -- is
    :func:`strict_positive_diagonal_witness`. Returns
    ``(Sigma_S, Sigma_Sp, support_S, support_Sp)``."""
    p = U.shape[1]
    sS = np.zeros(p, bool); sS[tau_in] = True
    sSp = np.zeros(p, bool); sSp[tau_in] = True; sSp[tau_extra] = True
    Sig_S = excitation_covariance(U, sS, np.array([[1.0]]), sigma_noise)
    Sig_Sp = excitation_covariance(U, sSp, np.diag([1.0, 0.0]), sigma_noise)
    return Sig_S, Sig_Sp, sS, sSp


def projector_excitation_gamma(B2: np.ndarray, support: np.ndarray,
                               sigma_curl: float = 1.0) -> np.ndarray:
    """The excitation ``Gamma_S = sigma_c^2 (B_{2,S}^T B_{2,S})^+`` for which
    ``B_{2,S} Gamma_S B_{2,S}^T = sigma_c^2 P_{im B_{2,S}}`` exactly: the
    projector excitation that makes equal-image supports second-order
    indistinguishable."""
    support = np.asarray(support, bool)
    Bs = B2[:, support]
    return sigma_curl**2 * np.linalg.pinv(Bs.T @ Bs)


def interpolated_excitation_gamma(B2: np.ndarray, support: np.ndarray,
                                  alpha: float, sigma_curl: float = 1.0
                                  ) -> np.ndarray:
    """``Gamma_alpha = sigma_c^2 [(1-alpha) I + alpha (B_{2,S}^T B_{2,S})^+]``:
    interpolates from isotropic (alpha=0, identifiable) to projector
    excitation (alpha=1, equal-image indistinguishable)."""
    support = np.asarray(support, bool)
    k = int(support.sum())
    Bs = B2[:, support]
    return sigma_curl**2 * ((1 - alpha) * np.eye(k)
                            + alpha * np.linalg.pinv(Bs.T @ Bs))


def lifted_atom_matrix(U: np.ndarray) -> np.ndarray:
    """The ``(r^2, p)`` design matrix ``A`` with columns
    ``vec(u_tau u_tau^T)`` — the lifted dictionary of the second-order
    problem. Full column rank on every clique complex (spark lemma), with
    smallest singular value ``sigma_min(A) > 0`` entering the estimator
    bound."""
    r, p = U.shape
    return np.stack([np.outer(U[:, t], U[:, t]).ravel() for t in range(p)],
                    axis=1)


def nnls_recovery_bound(
    Sigma: np.ndarray, sigma_min_A: float, w_min: float, N: int
) -> float:
    """Fully explicit non-asymptotic failure bound for the lifted-covariance
    NNLS estimator with threshold ``w_min / 2`` (Theorem: estimator
    consistency + O(1/N) failure probability):

        P(support error)
          <= 16 [ (tr Sigma)^2 + ||Sigma||_F^2 ] / ( N sigma_min(A)^2 w_min^2 ).

    Derivation (no hidden constants; each step is verified in
    tests/test_excitation.py):
      1. cone-constrained least squares obeys the deterministic perturbation
         bound ||w_hat - w||_2 <= 2 ||Sigma_hat - Sigma||_F / sigma_min(A)
         (optimality of w_hat + feasibility of w + Cauchy-Schwarz);
      2. for i.i.d. Gaussian snapshots with known mean,
         E ||Sigma_hat - Sigma||_F^2 = [ (tr Sigma)^2 + ||Sigma||_F^2 ] / N
         EXACTLY (Wishart second moments);
      3. thresholding at w_min/2 fails only if ||w_hat - w||_inf >= w_min/2;
         Markov's inequality on step 2 through step 1 gives the bound.
    Markov makes it conservative: the empirical transition happens roughly an
    order of magnitude earlier in N; the value is in the explicit O(1/N) rate
    with computable constants, not in tightness.
    """
    num = 16.0 * ((np.trace(Sigma)) ** 2 + np.linalg.norm(Sigma, "fro") ** 2)
    return float(min(1.0, num / (N * sigma_min_A**2 * w_min**2)))


def share_edge_adjacency(B2: np.ndarray) -> np.ndarray:
    """Adjacency matrix ``A`` of the share-an-edge graph on candidate
    triangles (``A_{sigma tau} = 1`` iff the triangles share an edge). The
    binary-support separation identity is
    ``|| sum_tau c_tau b_tau b_tau^T ||_F^2 = c^T (9 I + A) c`` for ternary
    ``c`` (since ``(G o G)`` has diagonal 9 and off-diagonal ``G^2 in {0,1}``).
    ``A`` is an induced subgraph of the Johnson graph J(n, 3), whose least
    eigenvalue is >= -3; Cauchy interlacing then gives the GLOBAL analytic
    separations: unrestricted binary equal-image minimum 9 (attained exactly
    by dependent single-face additions), equal-cardinality minimum 16
    (attained exactly by tetrahedral swaps). See supplement S4."""
    G = B2.T @ B2
    A = (np.abs(G) > 1e-9).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def curl_domain_signatures(B2: np.ndarray) -> np.ndarray:
    """Curl-domain triangle signatures ``U = Q.T B2`` (shape ``(r, p)``),
    where ``Q`` is an orthonormal basis of ``im(B2)``.

    Column ``u_tau`` satisfies ``||u_tau||^2 = 3`` and
    ``u_sigma . u_tau = G_{sigma tau}`` (the triangle Gram matrix) — the
    signatures preserve all curl-domain geometry in minimal coordinates.
    """
    from tfl.hodge import curl_subspace_basis

    Q = curl_subspace_basis(B2)
    return Q.T @ B2


def second_order_covariance(
    U: np.ndarray, support: np.ndarray, sigma_curl: float, sigma_noise: float
) -> np.ndarray:
    """Curl-coordinate snapshot covariance ``Sigma_z(S) = sigma_n^2 I_r +
    sigma_c^2 sum_{tau in S} u_tau u_tau^T`` for signatures ``U = Q.T B2``.

    This is the exact law of ``z_t = Q.T f_t`` under the generative model:
    the gradient component (and the candidate-orthogonal harmonic component,
    ``h in ker B2.T``) is annihilated by ``Q.T`` and the white edge noise
    projects to white ``r``-dimensional noise.
    """
    support = np.asarray(support, bool)
    r = U.shape[0]
    Us = U[:, support]
    return sigma_noise**2 * np.eye(r) + sigma_curl**2 * (Us @ Us.T)


def lifted_atoms_linearly_independent(B2: np.ndarray, tol: float = 1e-8) -> bool:
    """Certify the spark condition: the lifted atoms ``{b_tau b_tau^T}`` are
    linearly independent.

    This always holds for distinct 3-cliques of a simple graph: a PAIR of
    distinct edges lies in at most one common triangle, so the off-diagonal
    entry ``(e, e')`` of ``sum_tau w_tau b_tau b_tau^T`` equals ``± w_tau`` for
    that unique triangle — every coefficient is directly readable off the
    matrix. The numerical rank check below certifies it for any given ``B2``
    (and guards against degenerate candidate sets).
    """
    p = B2.shape[1]
    if p == 0:
        return True
    atoms = np.stack([np.outer(B2[:, t], B2[:, t]).ravel() for t in range(p)])
    return int(np.linalg.matrix_rank(atoms, tol=tol)) == p


def matrix_gaussian_chernoff(
    Sigma0: np.ndarray, Sigma1: np.ndarray
) -> tuple[float, float]:
    """Chernoff information between ``N(0, Sigma0)`` and ``N(0, Sigma1)``.

    ``C = max_s (1/2) [ log det( (1-s) W0 + s W1 ) - (1-s) log det W0
    - s log det W1 ]`` with ``W_i = Sigma_i^{-1}`` — the optimal error exponent
    of the binary test between the two covariances from i.i.d. snapshots
    (``P_err ~ exp(-N C)``). Returns ``(C, s*)``.
    """
    W0 = np.linalg.inv(Sigma0)
    W1 = np.linalg.inv(Sigma1)
    ld0 = np.linalg.slogdet(W0)[1]
    ld1 = np.linalg.slogdet(W1)[1]

    def neg_g(s: float) -> float:
        _, ld = np.linalg.slogdet((1.0 - s) * W0 + s * W1)
        return -0.5 * (ld - (1.0 - s) * ld0 - s * ld1)

    res = minimize_scalar(neg_g, bounds=(1e-6, 1 - 1e-6), method="bounded")
    return float(-res.fun), float(res.x)


def candidate_tetrahedra(
    triangles: list[tuple[int, int, int]]
) -> list[tuple[int, int, int, int]]:
    """Indices (into ``triangles``) of the four faces of every tetrahedron —
    every 4-clique of vertices whose four triangular faces are ALL candidates.

    The four face signatures of a tetrahedron satisfy the signed relation
    ``b_{012} - b_{013} + b_{023} - b_{123} = 0`` (boundary of the solid
    3-simplex), so any three faces span the same 3-space as any other three:
    tetrahedra host the MINIMAL equal-image confuser pairs.
    """
    from itertools import combinations

    tri_index = {t: i for i, t in enumerate(triangles)}
    verts = sorted({v for t in triangles for v in t})
    quads: list[tuple[int, int, int, int]] = []
    for quad in combinations(verts, 4):
        faces = [tuple(sorted(c)) for c in combinations(quad, 3)]
        if all(f in tri_index for f in faces):
            quads.append(tuple(tri_index[f] for f in faces))  # type: ignore[arg-type]
    return quads


def equal_image_single_swap_pairs(
    B2: np.ndarray, triangles: list[tuple[int, int, int]]
) -> list[tuple[int, int]]:
    """All UNORDERED candidate pairs ``(a, b)`` lying in a common tetrahedron:
    swapping ``a`` for ``b`` preserves the column image of any support that
    contains the other two faces. Swaps are the minimal equal-image confusers
    AMONG SUPPORTS OF EQUAL CARDINALITY (exhaustively verified on K5), with
    curl-domain separation
    ``|| u_a u_a^T - u_b u_b^T ||_F^2 = 9 + 9 - 2 (u_a . u_b)^2 = 16`` exactly
    (tetrahedron faces share exactly one edge, so ``u_a . u_b = ±1``).

    The UNRESTRICTED worst case is the SUBSET confuser ``S`` vs
    ``S ∪ {fourth face}`` (equal image because the four face signatures are
    linearly dependent), whose separation is ``||u_4||^4 = 9`` — giving
    Chernoff exponent ``(9/16) rho_2^2 (1+o(1))``, which EQUALS the
    isolated-triangle detection exponent ``C(rho)=rho^2/16`` at
    ``rho = 3 rho_2``: at the exponent level, deciding the hardest
    equal-image question costs no more than detecting one isolated triangle.
    See ``tests/test_second_order.py::test_exhaustive_k5_confuser_separations``.
    """
    from itertools import combinations

    pairs: set[tuple[int, int]] = set()
    for quad in candidate_tetrahedra(triangles):
        for a, b in combinations(quad, 2):
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def confuser_pair_chernoff(
    B2: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
    sigma_curl: float,
    sigma_noise: float,
) -> float:
    """Chernoff information ``C_G`` between the curl-coordinate covariances of
    two supports — the exact error exponent for telling them apart. Finite and
    positive whenever the supports differ (second-order identifiability);
    the sample complexity of the pair is ``N* ~ log(1/delta) / C_G``.
    """
    U = curl_domain_signatures(B2)
    S0 = second_order_covariance(U, support_a, sigma_curl, sigma_noise)
    S1 = second_order_covariance(U, support_b, sigma_curl, sigma_noise)
    C, _ = matrix_gaussian_chernoff(S0, S1)
    return C


def second_order_snr(sigma_curl: float, sigma_noise: float) -> float:
    """Second-order SNR ``rho_2 = sigma_c^2 / sigma_n^2``. (The per-triangle
    curl-SNR of the first-order theory is ``rho = 3 rho_2``.)"""
    return sigma_curl**2 / sigma_noise**2


def tetra_confuser_chernoff_small_snr(sigma_curl: float, sigma_noise: float) -> float:
    """Leading-order Chernoff information of the tetrahedral SWAP confuser
    pair: ``C_G = rho_2^2 (1 + o(1))`` as ``rho_2 -> 0``.

    Derivation: with ``E = Sigma_0^{-1/2}(Sigma_1 - Sigma_0) Sigma_0^{-1/2}``
    and ``D = sum u_tau u_tau^T`` (DIMENSIONLESS), the Bhattacharyya expansion
    gives ``C = (1/16) tr(E^2)(1+o(1)) = (rho_2^2/16) ||Delta D||_F^2 (1+o(1))``
    ``= ||Delta M||_F^2/(16 sigma_n^4)(1+o(1))`` with ``M = sigma_c^2 D`` (the
    two forms coincide; do NOT write ``rho_2^2 ||Delta M||^2``). The swap
    separation is ``||Delta D||_F^2 = 9 + 9 - 2 = 16`` (faces share exactly one
    edge), so the ``16``s cancel. The swap is the minimal confuser among
    EQUAL-CARDINALITY supports. See
    :func:`subset_confuser_chernoff_small_snr` for the unrestricted worst
    case.
    """
    return second_order_snr(sigma_curl, sigma_noise) ** 2


def subset_confuser_chernoff_small_snr(sigma_curl: float, sigma_noise: float) -> float:
    """Leading-order Chernoff information of the SUBSET confuser
    (``S`` vs ``S ∪ {fourth face of a hosted tetrahedron}``): with separation
    ``||u_4 u_4^T||_F^2 = ||u_4||^4 = 9``,
    ``C_G = (9/16) rho_2^2 (1+o(1))`` — the UNRESTRICTED worst case over
    equal-image pairs (exhaustively verified on K5), and EXACTLY the
    isolated-triangle detection exponent ``C(rho) = rho^2/16`` at
    ``rho = 3 rho_2``. Consequence: geometry-induced rank deficiency is FREE
    at the exponent level — the hardest equal-image decision costs the same
    exponent as detecting one isolated triangle; only the log-multiplicity
    (Fano) factor reflects the geometry.
    """
    return 9.0 / 16.0 * second_order_snr(sigma_curl, sigma_noise) ** 2


def second_order_min_snapshots(
    B2: np.ndarray,
    support_a: np.ndarray,
    support_b: np.ndarray,
    sigma_curl: float,
    sigma_noise: float,
    target_error: float = 0.05,
) -> float:
    """Chernoff-rate snapshot budget for distinguishing two supports:
    ``N* = log(1/delta) / C_G``. For minimal tetrahedral confusers at small
    SNR this is ``N* ~ log(1/delta) / rho_2^2`` — the SAME order as the
    single-triangle detection budget ``log(1/delta)/C(rho)`` with
    ``C(rho)=rho^2/16``, ``rho=3 rho_2`` (also ``Theta(1/rho_2^2)``): the two
    differ only by a CONSTANT factor (the subset confuser matches the
    single-triangle exponent exactly; the swap is smaller by 9/16). It is
    finite (second-order identifiable) but INFINITE for the first-order
    (deterministic-signal) model. That gap — finite vs infinite, not an extra
    1/rho_2 — is the corrected content of the rank obstruction."""
    C = confuser_pair_chernoff(B2, support_a, support_b, sigma_curl, sigma_noise)
    if C <= 0:
        return np.inf
    return float(np.log(1.0 / target_error) / C)


def confuser_family_fano_min_snapshots(
    B2: np.ndarray,
    triangles: list[tuple[int, int, int]],
    sigma_curl: float,
    sigma_noise: float,
    err: float = 0.5,
    max_hypotheses: int = 400,
) -> float:
    """Fano converse over the tetrahedral-confuser family.

    Prior: pick one tetrahedron uniformly among the ``n_tetra`` hosted by the
    candidate set and one of its 4 faces to leave inactive (the other 3
    active) — ``M = 4 n_tetra`` hypotheses. Fano with the pairwise-KL bound
    ``I(S; z^N) <= N max_{i != j} KL(P_i || P_j)`` gives
    ``N >= ((1-err) log M - log 2) / KL_max``.

    VALIDITY NOTE: ``KL_max`` must be the maximum over ALL ordered hypothesis
    pairs — cross-tetrahedron pairs have larger covariance separation than
    within-tetrahedron ones, so restricting the scan would UNDERSTATE
    ``KL_max`` and hence OVERSTATE the lower bound (invalid as a converse).
    This function therefore scans every ordered pair; if ``M`` exceeds
    ``max_hypotheses`` it raises rather than silently truncating.
    """
    quads = candidate_tetrahedra(triangles)
    if not quads:
        return 0.0
    M = 4 * len(quads)
    if M > max_hypotheses:
        raise ValueError(
            f"M={M} hypotheses exceed max_hypotheses={max_hypotheses}; "
            "a partial KL scan would not yield a valid converse")
    U = curl_domain_signatures(B2)
    p = B2.shape[1]

    covs = []
    for quad in quads:
        for leave in range(4):
            s = np.zeros(p, bool)
            s[[quad[i] for i in range(4) if i != leave]] = True
            covs.append(second_order_covariance(U, s, sigma_curl, sigma_noise))

    # precompute inverses/logdets once; max KL over all ordered pairs
    invs = [np.linalg.inv(S) for S in covs]
    lds = [np.linalg.slogdet(S)[1] for S in covs]
    r = covs[0].shape[0]
    kl_max = 0.0
    for i in range(len(covs)):
        for j in range(len(covs)):
            if i == j:
                continue
            kl = 0.5 * (float(np.trace(invs[j] @ covs[i])) - r + lds[j] - lds[i])
            if kl > kl_max:
                kl_max = kl
    if kl_max <= 0:
        return np.inf
    numer = (1.0 - err) * np.log(M) - np.log(2.0)
    return float(max(numer, 0.0) / kl_max)


def median_sigma_envelope(
    d: int, p: int, active_fraction: float, delta: float = 0.05,
) -> tuple[float, float]:
    """Finite-sample envelope for the median-based noise-variance estimate
    (supplement, Lemma S3.1; requires EDGE-DISJOINT candidates so the inactive
    normalized scores are i.i.d. — the DKW step needs independence): with
    probability >= 1 - delta,

        sigma_n_hat^2 / sigma_n^2  in  [q_d(a_lo), q_d(a_hi)] / q_d(1/2),

    where ``q_d`` is the chi2_d quantile function,
    ``a_lo = (1/2 - pi - eps)/(1 - pi)``, ``a_hi = (1/2 + eps)/(1 - pi)``,
    ``eps = sqrt(log(2/delta) / (2 (1-pi) p))`` (DKW on the inactive scores),
    valid whenever ``pi + eps < 1/2``. Raises ``ValueError`` outside that
    regime (too few candidates / too many active for the median to be robust).
    """
    pi = float(active_fraction)
    eps = float(np.sqrt(np.log(2.0 / delta) / (2.0 * (1.0 - pi) * p)))
    if pi + eps >= 0.5:
        raise ValueError(
            f"median envelope needs active_fraction + eps < 1/2 "
            f"(got pi={pi:.3f}, eps={eps:.3f})")
    a_lo = (0.5 - pi - eps) / (1.0 - pi)
    a_hi = (0.5 + eps) / (1.0 - pi)
    q_mid = chi2.ppf(0.5, df=d)
    return float(chi2.ppf(a_lo, df=d) / q_mid), float(chi2.ppf(a_hi, df=d) / q_mid)
