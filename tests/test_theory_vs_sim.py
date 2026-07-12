"""Cross-check every closed-form limit against Monte-Carlo simulation.

If a theorem's constant is wrong these tests fail, so nothing enters the paper
un-validated. Runtime is kept to a few seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tfl.generative import Complex, FlowParams, sample_flows
from tfl.estimators import curl_statistics
from tfl.limits import (
    curl_variances,
    gaussian_chernoff_information,
    single_triangle_chernoff,
    two_variance_bayes_error,
    invisibility_curl_snr_floor,
)


def _min_bayes_error_from_samples(E0: np.ndarray, E1: np.ndarray) -> float:
    """Empirical minimum Bayes error over all thresholds, from energy samples
    under H0 (``E0``) and H1 (``E1``)."""
    gammas = np.unique(np.concatenate([E0, E1]))
    best = 1.0
    for g in gammas:
        err = 0.5 * (np.mean(E0 > g) + np.mean(E1 < g))
        best = min(best, err)
    return best


def test_analytic_bayes_error_matches_gaussian_montecarlo():
    """Exact two-variance Bayes error matches direct Gaussian MC."""
    rng = np.random.default_rng(0)
    v0, v1 = 3.0, 3.0 * (1 + 1.5)
    T, R = 15, 20000
    E0 = np.sum(rng.normal(0, np.sqrt(v0), size=(R, T)) ** 2, axis=1)
    E1 = np.sum(rng.normal(0, np.sqrt(v1), size=(R, T)) ** 2, axis=1)
    emp = _min_bayes_error_from_samples(E0, E1)
    ana = two_variance_bayes_error(v0, v1, T)
    assert abs(emp - ana) < 0.02


def _isolated_triangle_complex() -> Complex:
    """Triangle (0,1,2) as its own component, plus a disjoint 4-cycle 3-4-5-6
    (a hole -> harmonic dimension 1). The only candidate triangle is (0,1,2),
    so it is isolated among candidates."""
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (5, 6), (3, 6)]
    triangles = [(0, 1, 2)]
    return Complex(n_nodes=7, edges=edges, triangles=triangles)


def test_curl_detection_is_immune_to_gradient_and_harmonic_nuisance():
    """End-to-end: with HUGE gradient/harmonic nuisance, single-triangle
    detection error still matches the noise-only two-variance theory. This is
    the paper's core mechanism (curl annihilates grad & harmonic)."""
    cx = _isolated_triangle_complex()
    sigma_curl, sigma_noise = 0.7, 1.0
    params = FlowParams(sigma_curl=sigma_curl, sigma_grad=25.0, sigma_harm=25.0, sigma_noise=sigma_noise)

    T_block, R = 18, 5000
    rng = np.random.default_rng(1)
    d1 = sample_flows(cx, active=np.array([True]), params=params, T=T_block * R, rng=rng)
    d0 = sample_flows(cx, active=np.array([False]), params=params, T=T_block * R, rng=rng)

    c1 = curl_statistics(d1)[0].reshape(R, T_block)
    c0 = curl_statistics(d0)[0].reshape(R, T_block)
    E1 = np.sum(c1**2, axis=1)
    E0 = np.sum(c0**2, axis=1)

    emp = _min_bayes_error_from_samples(E0, E1)
    v0, v1 = curl_variances(sigma_curl, sigma_noise)
    ana = two_variance_bayes_error(v0, v1, T_block)
    assert abs(emp - ana) < 0.03, f"empirical {emp:.3f} vs analytic {ana:.3f}"


def test_chernoff_is_error_exponent():
    """-log P_err(T) / T converges to the Chernoff information as T grows.

    Because P_err ~ (c / sqrt(T)) exp(-T C), the finite-T rate approaches C from
    *above* and monotonically decreases toward it."""
    sc, sn = 0.8, 1.0
    C = single_triangle_chernoff(sc, sn)
    v0, v1 = curl_variances(sc, sn)
    rates = []
    for T in (160, 320, 640):
        err = two_variance_bayes_error(v0, v1, T)
        rates.append(-np.log(err) / T)
    assert rates[0] > rates[1] > rates[2] > C            # monotone descent to C from above
    assert (rates[-1] - C) / C < 0.10                    # within 10% at T=640
    # distance to C is shrinking
    assert (rates[2] - C) < (rates[0] - C)


def test_small_rho_chernoff_scaling():
    """C(rho) ~ rho^2 / 16 as rho -> 0 (the sharp curl-invisibility rate)."""
    sn = 1.0
    for rho in (0.02, 0.01, 0.005):
        sc = np.sqrt(rho * sn**2 / 3.0)  # rho = 3 sc^2 / sn^2
        C = single_triangle_chernoff(sc, sn)
        assert abs(C / (rho**2 / 16.0) - 1.0) < 0.05


def test_exact_recovery_probability_matches_simulation():
    """The finite-sample exact-recovery formula matches end-to-end simulation on
    an edge-disjoint complex (with gradient+harmonic nuisance present)."""
    from tfl.generative import disjoint_triangle_complex
    from tfl.estimators import energy_detector_bayes_support, exact_recovery
    from tfl.limits import exact_recovery_probability

    cx = disjoint_triangle_complex(n_tri=6, n_cycles=1, cycle_len=5)
    p = cx.n_triangles
    active = np.zeros(p, dtype=bool)
    active[::2] = True
    n_active = int(active.sum())
    n_inactive = p - n_active

    sigma_noise = 1.0
    rng = np.random.default_rng(3)
    for rho, T in ((1.5, 30), (3.0, 20)):
        sc = np.sqrt(rho * sigma_noise**2 / 3.0)
        params = FlowParams(sigma_curl=sc, sigma_grad=2.0, sigma_harm=1.0, sigma_noise=sigma_noise)
        R = 800
        hits = 0
        for _ in range(R):
            ds = sample_flows(cx, active, params, T, rng)
            est = energy_detector_bayes_support(ds, sc, sigma_noise)
            hits += exact_recovery(est, active)
        emp = hits / R
        ana = exact_recovery_probability(sc, sigma_noise, T, n_active, n_inactive)
        assert abs(emp - ana) < 0.06, f"rho={rho} T={T}: emp {emp:.3f} vs ana {ana:.3f}"


def test_whitened_detector_beats_naive_under_confusability():
    """On an edge-sharing strip the whitened detector recovers the support while
    the naive curl-energy detector does not, and it matches the geometry-aware
    theory."""
    from tfl.generative import triangle_strip_complex
    from tfl.estimators import (
        energy_detector_bayes_support,
        whitened_curl_detector_support,
        hamming_error,
        exact_recovery,
        triangle_gram,
    )
    from tfl.limits import whitened_variances, heterogeneous_exact_recovery_probability

    cx = triangle_strip_complex(9)
    p = cx.n_triangles
    active = np.zeros(p, dtype=bool)
    active[1::2] = True
    sn, rho = 1.0, 8.0
    sc = float(np.sqrt(rho / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=2.0, sigma_harm=1.0, sigma_noise=sn)

    Gp_diag = np.diag(np.linalg.pinv(triangle_gram(_ds := sample_flows(cx, active, params, 5, np.random.default_rng(0)))))
    v0s, v1s = whitened_variances(Gp_diag, sc, sn)

    rng = np.random.default_rng(4)
    T = 60
    R = 400
    naive_h = white_h = white_exact = 0
    for _ in range(R):
        ds = sample_flows(cx, active, params, T, rng)
        naive_h += hamming_error(energy_detector_bayes_support(ds, sc, sn), active)
        est_w = whitened_curl_detector_support(ds, sc, sn)
        white_h += hamming_error(est_w, active)
        white_exact += exact_recovery(est_w, active)
    naive_h /= R
    white_h /= R
    white_exact /= R

    assert white_h < 0.1 < naive_h, f"naive {naive_h:.2f} should fail, whitened {white_h:.2f} should win"
    theory = heterogeneous_exact_recovery_probability(v0s, v1s, active, T)
    assert abs(white_exact - theory) < 0.05


def test_road_network_geometry_and_centered_recovery():
    """Guards the traffic study (R6): (a) the vendored TNTP road networks have
    FULL-column-rank B2 (curl DoF ratio exactly 1, unlike K_n's 3/n) and
    edge-disjoint triangles (G = 3I); (b) with a constant real-flow background,
    the CENTERED whitened detector matches the exact df=N-1 product law, while
    theory would be mis-calibrated without centering."""
    from pathlib import Path
    from tfl.tntp import load_tntp_network
    from tfl.hodge import build_incidences
    from tfl.estimators import whitened_curl_detector_support, exact_recovery
    from tfl.limits import whitened_variances, heterogeneous_exact_recovery_probability

    data = Path(__file__).resolve().parent.parent / "data" / "traffic"
    # (nodes, edges, triangles, pairwise edge-disjoint?) — EMA has 10 sharing pairs
    expected = {
        "SiouxFalls": (24, 38, 2, True),
        "EMA": (74, 129, 33, False),
        "Anaheim": (416, 634, 54, True),
    }
    for name, (nn, ne, p, disjoint) in expected.items():
        net = load_tntp_network(data / f"{name}_net.tntp", name=name)
        cx = net.complex
        assert (cx.n_nodes, len(cx.edges), len(cx.triangles)) == (nn, ne, p)
        _, B2 = build_incidences(cx)
        assert np.linalg.matrix_rank(B2) == p          # DoF ratio 1 on ALL of them
        G = B2.T @ B2
        assert np.allclose(G, 3.0 * np.eye(p)) == disjoint

    # (b) centered recovery on SiouxFalls (2 triangles -> fast MC)
    net = load_tntp_network(data / "SiouxFalls_net.tntp", data / "SiouxFalls_flow.tntp")
    cx = net.complex
    from tfl.hodge import build_incidences as _bi
    B1, B2 = _bi(cx)
    f_bg = net.real_flow / np.sqrt(np.mean(net.real_flow**2)) * 5.0  # strong background
    active = np.array([True, False])
    sn, sc = 1.0, 1.0                                   # rho = 3
    params = FlowParams(sigma_curl=sc, sigma_grad=1.0, sigma_harm=0.5, sigma_noise=sn)
    v0s, v1s = whitened_variances(np.full(2, 1.0 / 3.0), sc, sn)
    rng = np.random.default_rng(3)
    N, R = 20, 400
    hits = 0
    for _ in range(R):
        ds = sample_flows(cx, active, params, N, rng)
        ds.F += f_bg[:, None]
        est = whitened_curl_detector_support(ds, sc, sn, mode="bayes", center=True)
        hits += exact_recovery(est, active)
    emp = hits / R
    th = heterogeneous_exact_recovery_probability(v0s, v1s, active, N - 1)
    assert abs(emp - th) < 0.07, f"centered detector {emp:.3f} vs df=N-1 theory {th:.3f}"


def test_union_bound_is_rigorous_lower_bound():
    """The union bound must lower-bound both the independence approximation
    (algebraically: 1 - sum(e) <= prod(1-e)) and the empirical exact-recovery
    rate of the whitened detector under correlated (edge-sharing) noise."""
    from tfl.generative import triangle_strip_complex
    from tfl.estimators import whitened_curl_detector_support, exact_recovery, triangle_gram
    from tfl.limits import (
        whitened_variances,
        heterogeneous_exact_recovery_probability,
        heterogeneous_recovery_union_bound,
    )

    cx = triangle_strip_complex(9)
    p = cx.n_triangles
    active = np.zeros(p, dtype=bool)
    active[1::2] = True
    sn, rho = 1.0, 8.0
    sc = float(np.sqrt(rho / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=2.0, sigma_harm=1.0, sigma_noise=sn)

    ds0 = sample_flows(cx, active, params, 5, np.random.default_rng(0))
    Gp_diag = np.diag(np.linalg.pinv(triangle_gram(ds0)))
    v0s, v1s = whitened_variances(Gp_diag, sc, sn)

    rng = np.random.default_rng(7)
    for T in (12, 25, 60):
        indep = heterogeneous_exact_recovery_probability(v0s, v1s, active, T)
        union = heterogeneous_recovery_union_bound(v0s, v1s, active, T)
        assert union <= indep + 1e-12
        R = 300
        hits = sum(
            exact_recovery(whitened_curl_detector_support(
                sample_flows(cx, active, params, T, rng), sc, sn), active)
            for _ in range(R)
        )
        emp = hits / R
        # rigorous bound: empirical rate must not fall below it (MC tolerance)
        assert emp >= union - 0.05, f"T={T}: emp {emp:.3f} < union {union:.3f}"


def test_fano_bounds_are_valid_converses():
    """Supplement S1: the Fano snapshot lower bounds must (a) lower-bound the
    empirical 50%-recovery budget measured in the Fig. 2 (phase-transition)
    experiment, (b) be
    monotone decreasing in rho, and (c) exhibit the documented sparse/dense
    regime ordering between the Gaussian-KL and signal-agnostic variants."""
    import json
    from pathlib import Path
    from tfl.limits import fano_min_snapshots, signal_agnostic_fano_min_snapshots

    res = json.loads((Path(__file__).resolve().parent.parent
                      / "results" / "phase_transition.json").read_text())
    T_grid = np.array(res["T_grid"], float)
    rec = np.array(res["recovery"], float)          # (n_rho, n_T)
    rho_grid = np.array(res["rho_grid"], float)
    p, k = res["n_tri"], res["n_active"]
    for i, rho in enumerate(rho_grid):
        crossing = np.where(rec[i] >= 0.5)[0]
        if len(crossing) == 0:
            continue                                 # never recovers on this grid
        n_emp = T_grid[crossing[0]]                  # empirical 50% budget
        for bound in (fano_min_snapshots, signal_agnostic_fano_min_snapshots):
            assert bound(p, k, float(rho), err=0.5) <= n_emp + 1e-9

    rhos = np.geomspace(0.1, 30, 12)
    for bound in (fano_min_snapshots, signal_agnostic_fano_min_snapshots):
        vals = [bound(100, 10, float(r)) for r in rhos]
        assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))  # decreasing

    # tight anchor: pins the exact constant of the Gaussian Fano bound
    from scipy.special import gammaln
    from tfl.limits import gaussian_kl_two_variance
    p_a, k_a, rho_a = 20, 5, 1.5
    log_binom = gammaln(p_a + 1) - gammaln(k_a + 1) - gammaln(p_a - k_a + 1)
    expect = (0.5 * log_binom - np.log(2)) / (k_a * gaussian_kl_two_variance(1 + rho_a, 1))
    assert abs(fano_min_snapshots(p_a, k_a, rho_a, err=0.5) - expect) < 1e-9
    # sparse + small rho: Gaussian-KL is the tighter (larger) bound
    assert fano_min_snapshots(10_000, 10, 0.2) > signal_agnostic_fano_min_snapshots(10_000, 10, 0.2)
    # dense + large rho: the signal-agnostic bound overtakes
    assert signal_agnostic_fano_min_snapshots(100, 50, 100.0) > fano_min_snapshots(100, 50, 100.0)


def test_partial_sampling_closed_form_matches_simulation():
    """Supplement S2: on an edge-disjoint complex, exact recovery under
    Bernoulli(q) edge sampling matches the closed form
    [q^3 (1-P_miss)]^k [1 - q^3 P_fa]^(p-k)."""
    from tfl.generative import (
        disjoint_triangle_complex, edge_subsample_mask, observable_triangles,
    )
    from tfl.estimators import whitened_curl_detector_support, exact_recovery
    from tfl.limits import whitened_variances, _per_triangle_error_probs

    cx = disjoint_triangle_complex(6)
    p = 6
    active = np.zeros(p, bool)
    active[:2] = True          # asymmetric split (k=2 vs p-k=4): pins the
    sn, rho = 1.0, 8.0         # exponent placement in the closed form
    sc = float(np.sqrt(rho / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=1.0, sigma_harm=0.5, sigma_noise=sn)
    N = 25
    v0s, v1s = whitened_variances(np.full(p, 1.0 / 3.0), sc, sn)
    errs = _per_triangle_error_probs(v0s, v1s, active, N, "bayes", 0.05)
    p_miss, p_fa = float(errs[active][0]), float(errs[~active][0])

    from tfl.hodge import build_incidences
    B1, B2 = build_incidences(cx)
    rng = np.random.default_rng(5)
    for q in (0.8, 0.95):
        hits, R = 0, 400
        for _ in range(R):
            ds = sample_flows(cx, active, params, N, rng)
            emask = edge_subsample_mask(B2.shape[0], q, rng)
            omask = observable_triangles(B2, emask)
            ds.F *= emask[:, None]
            est = np.zeros(p, bool)
            if omask.any():
                ds_obs = type(ds)(F=ds.F, B1=ds.B1, B2_all=ds.B2_all[:, omask],
                                  active=active[omask], params=params,
                                  candidate_triangles=[t for t, m in
                                                       zip(cx.triangles, omask) if m])
                est[omask] = whitened_curl_detector_support(ds_obs, sc, sn, mode="bayes")
            hits += exact_recovery(est, active)
        emp = hits / R
        theory = (q**3 * (1 - p_miss))**2 * (1 - q**3 * p_fa)**4
        assert abs(emp - theory) < 0.07, f"q={q}: emp {emp:.3f} vs theory {theory:.3f}"


def test_median_sigma_envelope_and_plugin_consistency():
    """Supplement S3: (a) the median-based noise estimate falls inside the
    Lemma envelope at the advertised confidence; (b) the fully adaptive
    detector matches the known-parameter detector at moderate N."""
    from tfl.generative import disjoint_triangle_complex
    from tfl.estimators import (
        estimate_noise_sigma, adaptive_whitened_detector_support,
        whitened_curl_detector_support, exact_recovery,
    )
    from tfl.limits import median_sigma_envelope

    # (a) envelope coverage on a complex large enough for DKW to bite
    # (pre-factorized sampling: build incidences/harmonic basis ONCE)
    from tfl.hodge import build_incidences
    from tfl.generative import harmonic_basis, FlowDataset

    cx = disjoint_triangle_complex(200, n_cycles=1)
    p = 200
    active = np.zeros(p, bool)
    active[:50] = True                                  # pi = 0.25
    sn, sc = 1.0, 1.0                                   # rho = 3
    params = FlowParams(sigma_curl=sc, sigma_grad=1.0, sigma_harm=0.5, sigma_noise=sn)
    N = 50
    lo, hi = median_sigma_envelope(N, p, 0.25, delta=0.05)
    B1, B2 = build_incidences(cx)
    H = harmonic_basis(B1, B2)
    B2S = B2[:, active]
    rng = np.random.default_rng(9)
    inside = 0
    R = 120
    for _ in range(R):
        F = B1.T @ (params.sigma_grad * rng.standard_normal((B1.shape[0], N)))
        F += B2S @ (sc * rng.standard_normal((B2S.shape[1], N)))
        F += H @ (params.sigma_harm * rng.standard_normal((H.shape[1], N)))
        F += sn * rng.standard_normal((B1.shape[1], N))
        ds = FlowDataset(F=F, B1=B1, B2_all=B2, active=active, params=params,
                         candidate_triangles=list(cx.triangles))
        ratio = estimate_noise_sigma(ds)**2 / sn**2
        inside += (lo - 1e-12 <= ratio <= hi + 1e-12)
    assert inside / R >= 0.93, f"envelope coverage {inside/R:.2f} < 0.93 [{lo:.3f},{hi:.3f}]"

    # (b) adaptive == known at moderate N on the confusable strip
    from tfl.generative import triangle_strip_complex
    cx2 = triangle_strip_complex(9)
    active2 = np.zeros(9, bool)
    active2[1::2] = True
    sc2 = float(np.sqrt(8.0 / 3.0))
    params2 = FlowParams(sigma_curl=sc2, sigma_grad=2.0, sigma_harm=1.0, sigma_noise=1.0)
    hk = hp = 0
    R2, N2 = 250, 45
    for _ in range(R2):
        ds = sample_flows(cx2, active2, params2, N2, rng)
        hk += exact_recovery(
            whitened_curl_detector_support(ds, sc2, 1.0, mode="bayes"), active2)
        est, _, _ = adaptive_whitened_detector_support(ds, mode="bayes")
        hp += exact_recovery(est, active2)
    assert abs(hk - hp) / R2 < 0.08, f"known {hk/R2:.2f} vs plugin {hp/R2:.2f}"


def test_invisibility_floor_decreases_with_budget():
    """The curl-SNR floor rho*(T) shrinks as the snapshot budget grows, and
    scales like 1/sqrt(T) in the small-rho regime."""
    sn = 1.0
    floors = [invisibility_curl_snr_floor(sn, T, target_error=0.05) for T in (25, 100, 400)]
    assert floors[0] > floors[1] > floors[2] > 0
    # small-rho regime: quadrupling T should roughly halve rho* (1/sqrt(T) law)
    f_lo = invisibility_curl_snr_floor(sn, 1600, target_error=0.05)
    f_hi = invisibility_curl_snr_floor(sn, 6400, target_error=0.05)
    assert 1.7 < f_lo / f_hi < 2.4
