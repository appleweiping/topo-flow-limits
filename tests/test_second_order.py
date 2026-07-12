"""Guardrails for the first- vs second-order identifiability theory.

These tests pin down the corrected "rank obstruction": equal-image supports
are indistinguishable at FIRST order (deterministic/adversarial signals) but
identifiable at SECOND order (random signals), at a sample-complexity price
whose constants are validated here against exact algebra and Monte-Carlo.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from tfl.generative import (
    FlowParams,
    all_triangles,
    disjoint_triangle_complex,
    sample_flows,
    triangle_strip_complex,
)
from tfl.hodge import Complex, build_incidences, curl_subspace_basis
from tfl.limits import (
    candidate_tetrahedra,
    confuser_family_fano_min_snapshots,
    confuser_pair_chernoff,
    curl_domain_signatures,
    equal_image_single_swap_pairs,
    lifted_atoms_linearly_independent,
    matrix_gaussian_chernoff,
    second_order_covariance,
    second_order_min_snapshots,
    second_order_snr,
    tetra_confuser_chernoff_small_snr,
)


def complete_complex(n: int) -> Complex:
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return Complex(n_nodes=n, edges=edges, triangles=all_triangles(n, edges))


# ---------------------------------------------------------------------------
# The spark lemma: lifted atoms are linearly independent on every clique complex
# ---------------------------------------------------------------------------

def test_lifted_atoms_independent_on_standard_complexes():
    for cx in [complete_complex(4), complete_complex(6), complete_complex(9),
               triangle_strip_complex(9), disjoint_triangle_complex(6)]:
        _, B2 = build_incidences(cx)
        assert lifted_atoms_linearly_independent(B2), cx
    # random Erdos-Renyi clique complex
    rng = np.random.default_rng(7)
    n = 12
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
    tris = all_triangles(n, edges)
    if tris:
        cx = Complex(n_nodes=n, edges=edges, triangles=tris)
        _, B2 = build_incidences(cx)
        assert lifted_atoms_linearly_independent(B2)


def test_second_order_covariance_map_is_injective_on_k4_exhaustively():
    """On K4 (maximal rank deficiency among small K_n), every pair of distinct
    supports has DIFFERENT curl-coordinate covariance — the covariance map is
    injective even though rank(B2) = 3 < p = 4."""
    cx = complete_complex(4)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    p = B2.shape[1]
    assert np.linalg.matrix_rank(B2) == 3 and p == 4

    supports = []
    for k in range(p + 1):
        for c in combinations(range(p), k):
            s = np.zeros(p, bool)
            s[list(c)] = True
            supports.append(s)
    covs = [second_order_covariance(U, s, 1.0, 1.0) for s in supports]
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            assert np.linalg.norm(covs[i] - covs[j]) > 1e-9, (
                f"supports {supports[i].nonzero()} and {supports[j].nonzero()} "
                "share a covariance")


# ---------------------------------------------------------------------------
# First-order impossibility: equal-image confusers exist and are exactly
# indistinguishable under deterministic signals
# ---------------------------------------------------------------------------

def test_tetrahedral_confusers_have_equal_image_and_separation_16():
    cx = complete_complex(6)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    pairs = equal_image_single_swap_pairs(B2, cx.triangles)
    assert len(pairs) == 6 * len(candidate_tetrahedra(cx.triangles))
    assert len(candidate_tetrahedra(cx.triangles)) == 15  # C(6,4)

    for a, b in pairs[:12]:
        # equal image of the two swap supports inside the host tetrahedron
        quad = next(q for q in candidate_tetrahedra(cx.triangles) if a in q and b in q)
        others = [t for t in quad if t not in (a, b)]
        Sa = B2[:, others + [a]]
        Sb = B2[:, others + [b]]
        ra, rb = np.linalg.matrix_rank(Sa), np.linalg.matrix_rank(Sb)
        rab = np.linalg.matrix_rank(np.hstack([Sa, Sb]))
        assert ra == rb == rab == 3
        # universal curl-domain separation of the swap: exactly 16
        dM = np.outer(U[:, a], U[:, a]) - np.outer(U[:, b], U[:, b])
        assert np.linalg.norm(dM) ** 2 == pytest.approx(16.0, abs=1e-9)


def test_first_order_indistinguishability_deterministic_signals():
    """For equal-image supports S, S': ANY deterministic signal sequence under
    S can be replicated exactly by one under S' — the induced flow
    distributions coincide, so no estimator distinguishes them at any SNR/N.
    (This is the correctly-scoped impossibility half of the old Thm 4.)"""
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    quad = candidate_tetrahedra(cx.triangles)[0]
    a, b = quad[0], quad[1]
    others = [t for t in quad if t not in (a, b)]
    Sa = sorted(others + [a])
    Sb = sorted(others + [b])

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.standard_normal(len(Sa))
        target = B2[:, Sa] @ y
        y_prime, res, *_ = np.linalg.lstsq(B2[:, Sb], target, rcond=None)
        assert np.linalg.norm(B2[:, Sb] @ y_prime - target) < 1e-9


# ---------------------------------------------------------------------------
# Second-order price: validated Chernoff constants and sample complexity
# ---------------------------------------------------------------------------

def test_chernoff_small_snr_constant_is_rho2_squared():
    """C_G for the minimal tetrahedral swap converges to rho_2^2 as
    rho_2 -> 0 (the two 16s cancel: (rho_2^2/16) * ||dM||^2 = rho_2^2)."""
    cx = complete_complex(6)
    _, B2 = build_incidences(cx)
    quad = candidate_tetrahedra(cx.triangles)[0]
    a, b = quad[0], quad[1]
    others = [t for t in quad if t not in (a, b)]
    p = B2.shape[1]
    s_a = np.zeros(p, bool); s_a[others + [a]] = True
    s_b = np.zeros(p, bool); s_b[others + [b]] = True

    for sigma_c, rtol in [(0.1, 0.25), (0.03, 0.05)]:
        C = confuser_pair_chernoff(B2, s_a, s_b, sigma_c, 1.0)
        pred = tetra_confuser_chernoff_small_snr(sigma_c, 1.0)
        assert C == pytest.approx(pred, rel=rtol), (sigma_c, C, pred)


def test_binary_confuser_test_error_matches_chernoff_exponent():
    """Monte-Carlo: the optimal (likelihood-ratio) test between the two
    covariances of a tetrahedral confuser pair has error <= exp(-N C_G)
    (Chernoff bound, equal priors) and its empirical exponent decreases
    toward C_G as N grows."""
    rng = np.random.default_rng(3)
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    U = curl_domain_signatures(B2)
    quad = candidate_tetrahedra(cx.triangles)[0]
    a, b = quad[0], quad[1]
    others = [t for t in quad if t not in (a, b)]
    p = B2.shape[1]
    s_a = np.zeros(p, bool); s_a[others + [a]] = True
    s_b = np.zeros(p, bool); s_b[others + [b]] = True

    sigma_c = sigma_n = 1.0
    S0 = second_order_covariance(U, s_a, sigma_c, sigma_n)
    S1 = second_order_covariance(U, s_b, sigma_c, sigma_n)
    C, s_star = matrix_gaussian_chernoff(S0, S1)
    assert 0 < C < 1 and 0.3 < s_star < 0.7

    W0, W1 = np.linalg.inv(S0), np.linalg.inv(S1)
    ld0, ld1 = np.linalg.slogdet(S0)[1], np.linalg.slogdet(S1)[1]
    L0, L1 = np.linalg.cholesky(S0), np.linalg.cholesky(S1)
    r = S0.shape[0]

    def mc_error(N: int, trials: int = 4000) -> float:
        def err_under(L: np.ndarray, truth: int) -> float:
            Z = (L @ rng.standard_normal((r, N * trials))).reshape(r, trials, N)
            q0 = np.einsum("itn,ij,jtn->tn", Z, W0, Z).sum(axis=1)
            q1 = np.einsum("itn,ij,jtn->tn", Z, W1, Z).sum(axis=1)
            llr = 0.5 * (q0 - q1) - 0.5 * N * (ld1 - ld0)
            dec1 = llr > 0
            return float(np.mean(dec1) if truth == 0 else np.mean(~dec1))
        return 0.5 * (err_under(L0, 0) + err_under(L1, 1))

    exps = []
    for N in (10, 20, 40):
        pe = mc_error(N)
        assert pe <= np.exp(-N * C) * 1.15 + 3e-3, (N, pe, np.exp(-N * C))
        if pe > 0:
            exps.append(-np.log(pe) / N)
    # empirical exponent stays above C_G and decreases toward it
    assert all(e >= C * 0.95 for e in exps), (exps, C)
    assert exps == sorted(exps, reverse=True), exps


def test_second_order_recovery_succeeds_in_rank_deficient_regime():
    """Achievability: on K5 (rank B2 = 6 < p = 10 — the regime where the old
    Thm 4 claimed impossibility), the second-order greedy estimator recovers
    the exact support from enough snapshots, with gradient nuisance present."""
    from tfl.estimators import greedy_support, hamming_error

    rng = np.random.default_rng(11)
    cx = complete_complex(5)
    _, B2 = build_incidences(cx)
    assert np.linalg.matrix_rank(B2) == 6 and B2.shape[1] == 10

    p = B2.shape[1]
    active = np.zeros(p, bool)
    active[[0, 3, 7, 8]] = True
    params = FlowParams(sigma_curl=2.0, sigma_grad=1.0, sigma_harm=0.0,
                        sigma_noise=0.3)
    ds = sample_flows(cx, active, params, T=1200, rng=rng)
    est = greedy_support(ds, sigma_noise=params.sigma_noise)
    assert hamming_error(est, active) == 0


def test_sample_complexity_and_fano_are_consistent():
    cx = complete_complex(6)
    _, B2 = build_incidences(cx)
    quad = candidate_tetrahedra(cx.triangles)[0]
    a, b = quad[0], quad[1]
    others = [t for t in quad if t not in (a, b)]
    p = B2.shape[1]
    s_a = np.zeros(p, bool); s_a[others + [a]] = True
    s_b = np.zeros(p, bool); s_b[others + [b]] = True

    n_pair = second_order_min_snapshots(B2, s_a, s_b, 0.5, 1.0, target_error=0.05)
    C = confuser_pair_chernoff(B2, s_a, s_b, 0.5, 1.0)
    assert n_pair == pytest.approx(np.log(20.0) / C, rel=1e-6)

    # Fano over the confuser family: positive, finite, increases as SNR drops
    n_lo = confuser_family_fano_min_snapshots(B2, cx.triangles, 0.5, 1.0)
    n_hi = confuser_family_fano_min_snapshots(B2, cx.triangles, 0.1, 1.0)
    assert 0 < n_lo < n_hi < np.inf

    # no tetrahedra on the strip complex -> vacuous bound
    strip = triangle_strip_complex(9)
    _, B2s = build_incidences(strip)
    assert confuser_family_fano_min_snapshots(B2s, strip.triangles, 0.5, 1.0) == 0.0


def test_rho2_definition_links_first_and_second_order_snr():
    assert second_order_snr(2.0, 1.0) == pytest.approx(4.0)
    assert tetra_confuser_chernoff_small_snr(0.1, 1.0) == pytest.approx(1e-4)
