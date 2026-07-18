"""Guardrails for the scalable (matrix-free / GPU-portable) machinery:
matrix-free lifted-NNLS, random-complex generators, fast signatures, and the
non-oracle BIC selection rule.  The underlying math is verified independently
in the round-8 rule-1 scripts; these are the regression tests.
"""
from __future__ import annotations

import numpy as np

from tfl.hodge import Complex, build_incidences
from tfl.limits import curl_domain_signatures, lifted_atom_matrix
from tfl.estimators import nnls_lifted_support
from tfl import estimators_mf as mf
from tfl.complexes import (
    planted_clique_complex, complex_report, curl_signatures_fast,
    planted_signatures, erdos_renyi_clique_complex,
)
from tfl.selection import bic_nonoracle_support, estimate_sigma_n_curl


def _kn(n):
    E = [(i, j) for i in range(n) for j in range(i + 1, n)]
    T = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    cx = Complex(n_nodes=n, edges=E, triangles=T)
    _, B2 = build_incidences(cx)
    return B2, curl_domain_signatures(B2)


def test_matrix_free_kernels_match_dense():
    rng = np.random.default_rng(0)
    for n in (4, 5, 6):
        _, U = _kn(n)
        A = lifted_atom_matrix(U)
        r, p = U.shape
        w = rng.random(p)
        R = rng.standard_normal((r, r)); R = (R + R.T) / 2
        assert np.linalg.norm(mf.lifted_apply(U, w).ravel() - A @ w) < 1e-10
        assert np.linalg.norm(mf.lifted_adjoint(U, R) - A.T @ R.ravel()) < 1e-10
        # A^T A = G o G
        G = U.T @ U
        assert np.linalg.norm(mf.gram_hadamard(U) - G * G) < 1e-10
        # lipschitz / sigma_min match dense spectrum
        GG = mf.gram_hadamard(U)
        ev = np.linalg.eigvalsh(GG)
        assert abs(mf.lipschitz_ATA(U) - ev[-1]) < 1e-4 * ev[-1]
        assert abs(mf.sigma_min_A(U) - np.sqrt(max(ev[0], 0))) < 1e-6


def test_matrix_free_nnls_support_equals_dense():
    """Correctness gate: matrix-free FISTA & active-set == dense scipy NNLS."""
    rng = np.random.default_rng(1)
    sn = 1.0
    mism_f = mism_a = 0
    for n in (4, 5, 6, 7):
        _, U = _kn(n)
        r, p = U.shape
        for _ in range(4):
            k = rng.integers(1, max(2, p // 3) + 1)
            a = np.zeros(p, bool); a[rng.choice(p, k, replace=False)] = True
            rho2 = float(rng.choice([0.5, 1.0, 2.0])); sc = np.sqrt(rho2) * sn; N = 400
            Z = U[:, a] @ (sc * rng.standard_normal((k, N))) + sn * rng.standard_normal((r, N))
            thr = rho2 * sn ** 2 / 2
            sup_d, _ = nnls_lifted_support(Z, U, sn, thr)
            sup_f, _ = mf.nnls_lifted_support_mf(Z, U, sn, thr, solver="fista")
            sup_a, _ = mf.nnls_lifted_support_mf(Z, U, sn, thr, solver="active_set")
            mism_f += int(not np.array_equal(sup_f, sup_d))
            mism_a += int(not np.array_equal(sup_a, sup_d))
    assert mism_a == 0
    assert mism_f == 0


def test_planted_generator_dof_control():
    for s in (4, 5, 6, 8):
        cx = planted_clique_complex(100, block_size=s, overlap="edge", seed=0)
        rep = complex_report(cx)
        assert abs(rep["dof_ratio"] - 3.0 / s) < 1e-9
    # p targeting
    assert complex_report(planted_clique_complex(1000, block_size=6, seed=0))["p"] == 1000


def test_fast_signatures_exact_gram():
    cx = planted_clique_complex(300, block_size=6, seed=0)
    _, B2 = build_incidences(cx)
    G = B2.T @ B2
    Uf = curl_signatures_fast(B2, rank_hint=200)
    assert np.linalg.norm(Uf.T @ Uf - G) < 1e-8 * np.linalg.norm(G)
    # rank equals the SVD rank
    assert Uf.shape[0] == int(np.linalg.matrix_rank(B2))
    _cx, U, rep = planted_signatures(600, block_size=6, seed=1)
    assert abs(rep["dof_ratio"] - 0.5) < 1e-9


def test_bic_nonoracle_is_selection_consistent():
    """Non-oracle BIC (no w_min, no true sigma_n) recovers the support as N
    grows; sigma_n estimate is within ~15%."""
    rng = np.random.default_rng(2)
    _, U = _kn(6); r, p = U.shape
    sn = 1.0
    for N, floor in ((100, 0.9), (400, 0.95)):
        ntr = 60; hits = 0; sn_err = []
        for _ in range(ntr):
            a = np.zeros(p, bool); a[rng.choice(p, 3, replace=False)] = True
            Z = U[:, a] @ (rng.standard_normal((3, N))) + sn * rng.standard_normal((r, N))
            m, _w, info = bic_nonoracle_support(Z, U, N)
            hits += int(np.array_equal(m, a))
            sn_err.append(abs(info["sigma_n_estimated"] - sn))
        assert hits / ntr >= floor
        assert np.median(sn_err) < 0.15


def test_erdos_renyi_generator_runs():
    cx = erdos_renyi_clique_complex(24, 0.5, seed=0)
    rep = complex_report(cx)
    assert rep["p"] > 0 and 0 < rep["dof_ratio"] <= 1.0
