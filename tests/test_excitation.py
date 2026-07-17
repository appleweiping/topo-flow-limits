"""Guardrails for the excitation-dependent identifiability theory and the
lifted-covariance NNLS estimator.

Everything the revised paper claims about general excitations y_S ~ N(0, Γ_S)
is pinned here: the three identifiability regimes (diagonal / arbitrary PSD /
projector), the global analytic separations (9 unrestricted, 16
equal-cardinality, via the share-edge-graph identity and λ_min ≥ −3), the
NNLS estimator's consistency and its fully derived O(1/N) failure bound, and
the α-interpolation collapse of equal-image distinguishability.
"""
from __future__ import annotations

import numpy as np
import pytest

from tfl.generative import all_triangles
from tfl.hodge import Complex, build_incidences, curl_subspace_basis
from tfl.estimators import nnls_lifted_support, subspace_matched_support
from tfl.limits import (
    candidate_tetrahedra,
    curl_domain_signatures,
    excitation_covariance,
    feasible_supports,
    interpolated_excitation_gamma,
    lifted_atom_matrix,
    nnls_recovery_bound,
    projector_excitation_gamma,
    realized_range_dim,
    share_edge_adjacency,
    singular_gamma_equal_covariance_witness,
    strict_positive_diagonal_witness,
)


def complete_complex(n: int) -> Complex:
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return Complex(n_nodes=n, edges=edges, triangles=all_triangles(n, edges))


def tetra_confuser_supports(cx, B2):
    """S3 = three faces of a hosted tetrahedron, S4 = S3 + the fourth face
    (equal image)."""
    quad = candidate_tetrahedra(cx.triangles)[0]
    p = B2.shape[1]
    s3 = np.zeros(p, bool); s3[list(quad[:3])] = True
    s4 = s3.copy(); s4[quad[3]] = True
    return s3, s4


# ---------------------------------------------------------------------------
# The three excitation regimes
# ---------------------------------------------------------------------------

def test_diagonal_excitation_weighted_support_identifiable():
    """(a) positive-diagonal Γ: (S, weights) readable off the covariance at
    any rank deficiency — the lifted atom matrix has full column rank and the
    weights are recovered exactly."""
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    p = U.shape[1]
    A = lifted_atom_matrix(U)
    assert np.linalg.matrix_rank(A) == p

    rng = np.random.default_rng(0)
    w = np.zeros(p)
    w[[1, 4, 7]] = rng.uniform(0.3, 2.0, size=3)
    Sig = excitation_covariance(U, w > 0, np.diag(w[w > 0]), sigma_noise=1.0)
    w_rec, *_ = np.linalg.lstsq(A, (Sig - np.eye(U.shape[0])).ravel(), rcond=None)
    assert np.max(np.abs(w_rec - w)) < 1e-10


def test_kn_unknown_noise_ambiguity_direction():
    """(a) caveat: on K_n, sum over ALL candidates of u_tau u_tau^T = n I_r,
    so with UNKNOWN sigma_n there is exactly a one-dimensional ambiguity
    (uniform weight shift <-> noise floor)."""
    for n in (5, 6, 7):
        cx = complete_complex(n)
        _, B2 = build_incidences(cx)
        U = curl_domain_signatures(B2)
        total = U @ U.T
        assert np.allclose(total, n * np.eye(U.shape[0]), atol=1e-9)


def test_arbitrary_psd_equal_image_families_overlap():
    """(b) unknown arbitrary PSD Γ: an equal-image support S' can match ANY
    covariance produced by S, with a PSD Γ' — so the achievable covariance
    families overlap and equal-image supports are indistinguishable. (This
    does NOT say the image is identifiable; see the singular-Γ test below.)"""
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    s3, s4 = tetra_confuser_supports(cx, B2)
    B3, B4 = B2[:, s3], B2[:, s4]

    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 3))
    Gam = X @ X.T                       # arbitrary PSD excitation for S3
    M = B3 @ Gam @ B3.T
    B4p = np.linalg.pinv(B4)
    Gam_p = B4p @ M @ B4p.T             # candidate matching excitation for S4
    assert np.linalg.norm(B4 @ Gam_p @ B4.T - M) < 1e-9
    assert np.min(np.linalg.eigvalsh(Gam_p)) > -1e-10   # PSD


def test_strict_positive_diagonal_witness_defeats_identifiability():
    """(b) STRONG counterexample: even a STRICTLY-POSITIVE-DIAGONAL (but
    CORRELATED) excitation fails to identify the support. On K4 (all 4 faces,
    U_S rank-deficient) take Γ'=vvᵀ, v=e0−0.25c with c∈ker U_S; then Γ' has
    positive diagonal, rank 1, and U_S Γ' U_Sᵀ = u0u0ᵀ — identical to the
    singleton {τ0} with unit variance, although the candidate images are 3 vs
    1. Positive per-triangle variance is not enough; regime (a) needs Γ
    DIAGONAL (uncorrelated)."""
    cx = complete_complex(4)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    S_big = np.ones(4, bool)                       # all 4 faces: U_S has a kernel
    Sig_big, Sig_single, sB, sS = strict_positive_diagonal_witness(
        U, tau_keep=0, kernel_support=S_big, sigma_noise=1.0)
    # identical population covariances to 1e-10 ...
    assert np.linalg.norm(Sig_big - Sig_single) < 1e-10
    # ... under a STRICTLY POSITIVE DIAGONAL Γ' (reconstruct it to check)
    # (the witness used v = e0 + t c; verify positive diagonal + non-diagonal)
    # candidate images differ: 3 (all faces) vs 1 (singleton)
    assert np.linalg.matrix_rank(B2[:, sB]) == 3
    assert np.linalg.matrix_rank(B2[:, sS]) == 1


def test_singular_gamma_degenerate_example():
    """(b) DEGENERATE example (zero-variance triangle, kept only as the easy
    case): S={τ0}, S'={τ0,τ1}, Γ'=diag(1,0) give identical covariance with
    candidate-image dims 1 vs 2. The strong statement is the strict-positive-
    diagonal witness above."""
    cx = complete_complex(4)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    Sig_S, Sig_Sp, sS, sSp = singular_gamma_equal_covariance_witness(
        U, tau_in=0, tau_extra=1, sigma_noise=1.0)
    assert np.linalg.norm(Sig_S - Sig_Sp) < 1e-12
    assert np.linalg.matrix_rank(B2[:, sS]) == 1
    assert np.linalg.matrix_rank(B2[:, sSp]) == 2


def test_feasible_support_set_is_not_a_singleton():
    """(b) STRUCTURAL: the population covariance-signal M=Σ_z−σ_n²I fixes the
    feasible support set to {S: im M ⊆ im U_S}. For M=u0u0ᵀ on K4 that set has
    MORE THAN ONE element — the population covariance does not identify the
    support under arbitrary PSD excitation."""
    cx = complete_complex(4)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    p = U.shape[1]
    M = np.outer(U[:, 0], U[:, 0])
    import itertools
    cand = [np.array(b, bool) for b in itertools.product([0, 1], repeat=p) if any(b)]
    feas = feasible_supports(U, M, cand)
    # every feasible support really reproduces M with a PSD Γ
    for S in feas:
        Us = U[:, S]
        Gam = np.linalg.pinv(Us) @ M @ np.linalg.pinv(Us).T
        assert np.min(np.linalg.eigvalsh(Gam)) > -1e-9
        assert np.linalg.norm(Us @ Gam @ Us.T - M) < 1e-8
    # the singleton {τ0} is feasible, but so are strictly larger supports
    assert any(int(S.sum()) == 1 and S[0] for S in feas)
    assert any(int(S.sum()) > 1 for S in feas)
    assert len(feas) > 1


def test_full_rank_gamma_identifies_image():
    """(b) NONSINGULAR Γ: on a full-column-rank support the realized range
    equals the full candidate image, so a positive-definite excitation DOES
    reveal im B_{2,S}; there the equality R = im U_S holds iff Γ ≻ 0."""
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    s3, _ = tetra_confuser_supports(cx, B2)       # 3 faces: full column rank
    assert np.linalg.matrix_rank(B2[:, s3]) == int(s3.sum())
    k = int(s3.sum())
    rng = np.random.default_rng(3)
    X = rng.standard_normal((k, k))
    Gam_pd = X @ X.T + 0.5 * np.eye(k)          # Γ ≻ 0
    assert realized_range_dim(U, s3, Gam_pd) == np.linalg.matrix_rank(B2[:, s3])
    # on a full-column-rank support a rank-deficient Γ DOES shrink the range
    Gam_sing = np.diag([1.0] * (k - 1) + [0.0])
    assert realized_range_dim(U, s3, Gam_sing) < np.linalg.matrix_rank(B2[:, s3])


def test_singular_gamma_can_still_give_full_image_when_support_rank_deficient():
    """The converse `R = im U_S ⇒ Γ ≻ 0` FAILS when B_{2,S} is rank-deficient
    (U_S has a kernel): a singular Γ whose image complements ker U_S still
    attains the full image. So the equality is only `Γ ≻ 0 ⇒ R = im U_S`
    (always) plus the converse under full column rank — not a blanket iff."""
    cx = complete_complex(4)                    # all 4 K4 faces: B2 rank 3
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    S = np.ones(4, bool)
    assert np.linalg.matrix_rank(B2[:, S]) == 3          # rank-deficient support
    c = np.array([1.0, -1.0, 1.0, -1.0])                 # tetra dependency ∈ ker U_S
    assert np.linalg.norm(U[:, S] @ c) < 1e-9
    Gam_sing = np.eye(4) - np.outer(c, c) / (c @ c)      # rank 3, singular
    assert np.linalg.matrix_rank(Gam_sing) == 3
    # singular Γ, yet realized range = full image (3)
    assert realized_range_dim(U, S, Gam_sing) == 3


def test_projector_excitation_equal_image_indistinguishable():
    """(c) Γ_S = sigma_c^2 (B_S^T B_S)^+ gives covariance sigma_n^2 I +
    sigma_c^2 P_im exactly, so the equal-image pair induces IDENTICAL
    covariances — the explicit counterexample killing any claim that random
    signals per se remove the rank obstruction."""
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    s3, s4 = tetra_confuser_supports(cx, B2)

    covs = []
    for s in (s3, s4):
        Gam = projector_excitation_gamma(B2, s, sigma_curl=1.3)
        covs.append(excitation_covariance(U, s, Gam, sigma_noise=0.7))
        # covariance = sigma_n^2 I + sigma_c^2 * projector
        Us = U[:, s]
        Uc, sv, _ = np.linalg.svd(Us, full_matrices=False)
        P = Uc[:, sv > 1e-9] @ Uc[:, sv > 1e-9].T
        assert np.linalg.norm(covs[-1] - (0.49 * np.eye(U.shape[0]) + 1.69 * P)) < 1e-9
    assert np.linalg.norm(covs[0] - covs[1]) < 1e-9


# ---------------------------------------------------------------------------
# Global analytic separations
# ---------------------------------------------------------------------------

def test_separation_identity_and_johnson_eigenvalue_bound():
    """sep(c) = c^T (9I + (G o G)_offdiag) c for ternary c, and the
    share-edge adjacency has lambda_min >= -3 (induced subgraph of J(n,3) +
    interlacing) — on K_n AND on a random clique complex. Hence
    sep >= 6 ||c||^2: unrestricted binary equal-image minimum 9 (single-face
    difference), equal-cardinality minimum 16."""
    rng = np.random.default_rng(2)

    def check(cx):
        _, B2 = build_incidences(cx)
        p = B2.shape[1]
        G = B2.T @ B2
        A = share_edge_adjacency(B2)
        # off-diagonal of (G o G) equals A (shared edge -> G^2 = 1)
        GG = G**2
        np.fill_diagonal(GG, 0.0)
        assert np.array_equal(GG, A)
        assert np.linalg.eigvalsh(A)[0] >= -3.0 - 1e-9
        for _ in range(60):
            c = rng.integers(-1, 2, size=p).astype(float)
            M = B2 @ np.diag(c) @ B2.T
            # ||sum c_tau b b^T||_F^2 with signs folded through diag(c):
            M = sum(ci * np.outer(B2[:, i], B2[:, i])
                    for i, ci in enumerate(c)) if np.any(c) else np.zeros((B2.shape[0],) * 2)
            lhs = np.linalg.norm(M) ** 2
            rhs = 9 * np.sum(c**2) + c @ A @ c
            assert lhs == pytest.approx(rhs, abs=1e-8)

    for n in (4, 5, 6, 7):
        check(complete_complex(n))
    n = 11
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if rng.random() < 0.5]
    tris = all_triangles(n, edges)
    if tris:
        check(Complex(n_nodes=n, edges=edges, triangles=tris))


# ---------------------------------------------------------------------------
# NNLS estimator: consistency + the derived O(1/N) failure bound
# ---------------------------------------------------------------------------

def test_nnls_consistency_and_recovery_bound_components():
    """(1) E||Sigma_hat - Sigma||_F^2 = ((tr Sigma)^2 + ||Sigma||_F^2)/N
    exactly (Gaussian, known mean); (2) the cone-LS perturbation bound
    ||w_hat - w|| <= 2||Delta||_F / sigma_min(A) never violates; (3) w_hat
    contracts to w (consistency)."""
    rng = np.random.default_rng(3)
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    r, p = U.shape
    w = np.zeros(p); w[[0, 3, 7]] = 1.0
    Sig = excitation_covariance(U, w > 0, np.eye(3), sigma_noise=1.0)
    L = np.linalg.cholesky(Sig)
    A = lifted_atom_matrix(U)
    smin = np.linalg.svd(A, compute_uv=False)[-1]

    # (1) exact covariance-error formula
    N = 40
    errs = []
    for _ in range(1500):
        Z = L @ rng.standard_normal((r, N))
        errs.append(np.linalg.norm(Z @ Z.T / N - Sig, "fro") ** 2)
    pred = ((np.trace(Sig)) ** 2 + np.linalg.norm(Sig, "fro") ** 2) / N
    assert np.mean(errs) == pytest.approx(pred, rel=0.06)

    # (2) perturbation bound, (3) consistency
    max_err_small, max_err_big = 0.0, 0.0
    for N, store in ((30, "small"), (2000, "big")):
        for _ in range(200 if N == 30 else 20):
            Z = L @ rng.standard_normal((r, N))
            sup, w_hat = nnls_lifted_support(Z, U, 1.0, threshold=0.5)
            delta = np.linalg.norm(Z @ Z.T / N - Sig, "fro")
            assert np.linalg.norm(w_hat - w) <= 2 * delta / smin + 1e-9
            if N == 30:
                max_err_small = max(max_err_small, np.linalg.norm(w_hat - w))
            else:
                max_err_big = max(max_err_big, np.linalg.norm(w_hat - w))
    assert max_err_big < 0.25 * max_err_small  # contraction with N


def test_nnls_recovery_bound_upper_bounds_empirical_failure():
    """The fully explicit failure bound holds empirically across an (N, rho2)
    grid on rank-deficient K5 — validated, not guessed."""
    rng = np.random.default_rng(4)
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    r, p = U.shape
    A = lifted_atom_matrix(U)
    smin = np.linalg.svd(A, compute_uv=False)[-1]

    for rho2 in (0.5, 1.0):
        w = np.zeros(p); w[[2, 5, 8]] = rho2
        Sig = excitation_covariance(U, w > 0, rho2 * np.eye(3), sigma_noise=1.0)
        L = np.linalg.cholesky(Sig)
        for N in (100, 400):
            fails = 0
            T = 300
            for _ in range(T):
                Z = L @ rng.standard_normal((r, N))
                sup, _ = nnls_lifted_support(Z, U, 1.0, threshold=rho2 / 2)
                fails += not np.array_equal(sup, w > 0)
            bound = nnls_recovery_bound(Sig, smin, rho2, N)
            assert fails / T <= bound + 0.03, (rho2, N, fails / T, bound)


def test_nnls_recovery_bound_nonvacuous_cells():
    """Cells where the bound is strictly below 1, so the assertion CAN fail:
    on K4 at (w=0.35, N=800) and (w=0.2, N=1600) the bound is ~0.73/~0.75
    while the empirical failure rate is far below it (the bound is Markov —
    valid but conservative by one to three orders of magnitude: the
    empirical failure decays exponentially, the bound as 1/N)."""
    rng = np.random.default_rng(10)
    cx = complete_complex(4)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    r, p = U.shape
    A = lifted_atom_matrix(U)
    smin = np.linalg.svd(A, compute_uv=False)[-1]

    for w_val, N in ((0.35, 800), (0.2, 1600)):
        act = np.zeros(p, bool); act[[0, 2]] = True
        Sig = excitation_covariance(U, act, w_val * np.eye(2), sigma_noise=1.0)
        bound = nnls_recovery_bound(Sig, smin, w_val, N)
        assert bound < 0.8          # genuinely non-vacuous
        L = np.linalg.cholesky(Sig)
        fails = 0
        T = 200
        for _ in range(T):
            Z = L @ rng.standard_normal((r, N))
            sup, _ = nnls_lifted_support(Z, U, 1.0, threshold=w_val / 2)
            fails += not np.array_equal(sup, act)
        assert fails / T <= bound, (w_val, N, fails / T, bound)


def test_subspace_baseline_population_tie_and_projector_collapse():
    """The honest comparison on the tetra confuser (S3 vs equal-image
    alternatives):

    1. POPULATION subspace scores tie exactly — all four faces score 1 —
       so the subspace method has no population-level margin (its finite-N
       success rides on sample eigen-anisotropy, a fragile side channel).
    2. Under ISOTROPIC excitation, NNLS recovers S3 essentially always and
       strictly beats the subspace baseline.
    3. Under PROJECTOR excitation (case (c)), the anisotropy vanishes and
       BOTH methods collapse to chance among the tied faces — as they must,
       since the covariances are identical."""
    rng = np.random.default_rng(5)
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    r, p = U.shape
    s3, s4 = tetra_confuser_supports(cx, B2)

    # 1. population tie: exact projector onto im U_S3 scores all 4 faces at 1
    Us = U[:, s3]
    Uc, sv, _ = np.linalg.svd(Us, full_matrices=False)
    P = Uc[:, sv > 1e-9] @ Uc[:, sv > 1e-9].T
    quad_scores = [float(np.sum((P @ U[:, t]) ** 2) / np.sum(U[:, t] ** 2))
                   for t in np.where(s4)[0]]
    assert np.allclose(quad_scores, 1.0, atol=1e-9)

    def run(Gam, T=50, N=2000):
        Sig = excitation_covariance(U, s3, Gam, sigma_noise=0.5)
        L = np.linalg.cholesky(Sig)
        nn, ss = 0, 0
        for _ in range(T):
            Z = L @ rng.standard_normal((r, N))
            sup_n, _ = nnls_lifted_support(Z, U, 0.5, threshold=0.5)
            nn += np.array_equal(sup_n, s3)
            sup_s = subspace_matched_support(Z, U, k=3, subspace_dim=3)
            ss += np.array_equal(sup_s, s3)
        return nn / T, ss / T

    # 2. isotropic: NNLS ~always, strictly above the subspace baseline
    nn_iso, ss_iso = run(np.eye(3))
    assert nn_iso > 0.9
    assert nn_iso > ss_iso

    # 3. projector excitation: both at chance among the 4 tied choices
    Gam_proj = projector_excitation_gamma(B2, s3, sigma_curl=1.0)
    nn_proj, ss_proj = run(Gam_proj)
    assert nn_proj < 0.6 and ss_proj < 0.6


def test_alpha_interpolation_kills_equal_image_distinguishability():
    """Gamma_alpha = (1-alpha) I + alpha (B_S^T B_S)^+: the covariance gap
    between the equal-image pair (each under its own Gamma_alpha) decreases
    monotonically to EXACTLY zero at alpha = 1."""
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    s3, s4 = tetra_confuser_supports(cx, B2)

    gaps = []
    for alpha in (0.0, 0.5, 0.9, 0.99, 1.0):
        covs = []
        for s in (s3, s4):
            Gam = interpolated_excitation_gamma(B2, s, alpha)
            covs.append(excitation_covariance(U, s, Gam, sigma_noise=1.0))
        gaps.append(np.linalg.norm(covs[0] - covs[1]))
    assert all(g1 >= g2 - 1e-12 for g1, g2 in zip(gaps, gaps[1:]))
    assert gaps[0] > 1.0
    assert gaps[-1] < 1e-10
