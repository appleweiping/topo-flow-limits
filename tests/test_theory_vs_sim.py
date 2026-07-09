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
