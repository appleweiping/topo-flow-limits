"""Round-9 regression: the non-oracle median-eigenvalue BIC selector has a
NOISE-IDENTIFIABILITY boundary and is NOT unconditionally consistent.

This test LOCKS the retraction: it asserts that on K6 (r=10, p=20) the shipped
median-sigma_n rule FAILS on a dense support even at large N (a bias, not
variance -- so it does not vanish as N grows), while the SAME BIC path given the
true sigma_n recovers the dense support. Any change that lets the median rule be
described as "consistent" for dense supports must break this test.
"""
import numpy as np

from tfl.hodge import Complex, build_incidences
from tfl.limits import curl_domain_signatures
from tfl.estimators_mf import nnls_lifted_fista
from tfl.selection import bic_nonoracle_support, bic_support_path


def _k6_U():
    n = 6
    E = [(i, j) for i in range(n) for j in range(i + 1, n)]
    T = [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)]
    _, B2 = build_incidences(Complex(n_nodes=n, edges=E, triangles=T))
    return curl_domain_signatures(B2)


def _recovery(U, k, N, oracle, seed, trials=6):
    r, p = U.shape
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(trials):
        a = np.zeros(p, bool)
        a[rng.choice(p, k, replace=False)] = True
        Z = U[:, a] @ rng.standard_normal((k, N)) + rng.standard_normal((r, N))
        if oracle:
            Sig = (Z @ Z.T) / N
            w = nnls_lifted_fista(Sig - np.eye(r), U)
            m, _ = bic_support_path(Sig, U, w, 1.0, N)
        else:
            m, _, _ = bic_nonoracle_support(Z, U, N)
        hits += int(np.array_equal(m, a))
    return hits / trials


def test_bic_median_noise_fails_on_dense_support_even_at_large_N():
    """Counterexample: K6, dense k=15, N=8000. Median-sigma_n BIC must FAIL
    (noise-identifiability broken); oracle-sigma_n BIC must still recover."""
    U = _k6_U()
    N = 8000
    dense_median = _recovery(U, k=15, N=N, oracle=False, seed=2)
    dense_oracle = _recovery(U, k=15, N=N, oracle=True, seed=2)
    # The retraction: dense + median-noise is a persistent failure.
    assert dense_median <= 0.34, (
        f"median-sigma_n BIC recovered {dense_median} on a dense support -- if "
        "this rises, the noise-identifiability boundary changed; do NOT relabel "
        "the dense regime as consistent without re-deriving it.")
    # The retained (conditional) claim: with an identifiable sigma_n it recovers.
    assert dense_oracle >= 0.8, (
        f"oracle-sigma_n BIC only recovered {dense_oracle}; population "
        "identifiability of dense supports should hold (atoms independent).")


def test_bic_median_noise_recovers_in_sparse_regime():
    """Inside the noise-identifiable regime (sparse k=3) the shipped rule works
    -- this is the regime the paper's experiments actually use."""
    U = _k6_U()
    sparse_median = _recovery(U, k=3, N=8000, oracle=False, seed=1)
    assert sparse_median >= 0.8, (
        f"median-sigma_n BIC only recovered {sparse_median} in the sparse "
        "regime it is designed for.")
