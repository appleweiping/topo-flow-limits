"""Tests that pin down the Hodge geometry the whole theory rests on."""

from __future__ import annotations

import numpy as np
import pytest

from tfl.hodge import (
    Complex,
    build_incidences,
    curl,
    hodge_1_laplacian,
    hodge_decomposition,
)


def _tetrahedron() -> Complex:
    """K4 with all 4 triangular faces filled (a hollow tetrahedron surface)."""
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    return Complex(n_nodes=4, edges=edges, triangles=triangles)


def _two_triangles_sharing_edge() -> Complex:
    """Nodes 0-1-2 and 1-2-3 sharing edge (1,2)."""
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    triangles = [(0, 1, 2), (1, 2, 3)]
    return Complex(n_nodes=4, edges=edges, triangles=triangles)


@pytest.mark.parametrize("factory", [_tetrahedron, _two_triangles_sharing_edge])
def test_boundary_of_boundary_is_zero(factory):
    """d1 . d2 = 0, i.e. B1 @ B2 == 0. This is what makes grad ⟂ curl."""
    cx = factory()
    B1, B2 = build_incidences(cx)
    assert np.allclose(B1 @ B2, 0.0, atol=1e-12)


@pytest.mark.parametrize("factory", [_tetrahedron, _two_triangles_sharing_edge])
def test_hodge_decomposition_reconstructs_and_is_orthogonal(factory):
    cx = factory()
    B1, B2 = build_incidences(cx)
    rng = np.random.default_rng(0)
    f = rng.standard_normal(cx.n_edges)

    g, c, h = hodge_decomposition(f, B1, B2)

    # reconstruction is exact
    assert np.allclose(g + c + h, f, atol=1e-10)
    # the three components are mutually orthogonal
    assert abs(g @ c) < 1e-9
    assert abs(g @ h) < 1e-9
    assert abs(c @ h) < 1e-9


@pytest.mark.parametrize("factory", [_tetrahedron, _two_triangles_sharing_edge])
def test_gradient_is_curl_free_and_curl_is_divergence_free(factory):
    cx = factory()
    B1, B2 = build_incidences(cx)
    rng = np.random.default_rng(1)
    f = rng.standard_normal(cx.n_edges)
    g, c, h = hodge_decomposition(f, B1, B2)

    # a gradient flow has zero curl
    assert np.allclose(curl(g, B2), 0.0, atol=1e-9)
    # a curl flow is divergence-free
    assert np.allclose(B1 @ c, 0.0, atol=1e-9)


def test_curl_flow_lives_entirely_in_curl_component():
    """A flow synthesized as B2 @ y has no gradient/harmonic part: the curl
    component captures *all* triangle-circulation energy. This is the mechanism
    behind curl-invisibility."""
    cx = _two_triangles_sharing_edge()
    B1, B2 = build_incidences(cx)
    rng = np.random.default_rng(2)
    y = rng.standard_normal(cx.n_triangles)
    f = B2 @ y

    g, c, h = hodge_decomposition(f, B1, B2)
    assert np.linalg.norm(g) < 1e-9
    assert np.linalg.norm(h) < 1e-9
    assert np.allclose(c, f, atol=1e-9)


def test_curl_dimension_ratio_on_complete_graph():
    """Geometry-side identifiability: on K_n the curl subspace dimension is
    rank(B2) = C(n-1, 2), so the identifiable fraction of the C(n,3) candidate
    triangles is exactly 3/n."""
    from itertools import combinations
    from math import comb

    for n in range(5, 13):
        edges = [(i, j) for i, j in combinations(range(n), 2)]
        triangles = [(i, j, k) for i, j, k in combinations(range(n), 3)]
        cx = Complex(n_nodes=n, edges=edges, triangles=triangles)
        _, B2 = build_incidences(cx)
        rank = int(np.linalg.matrix_rank(B2))
        assert rank == comb(n - 1, 2)
        assert abs(rank / comb(n, 3) - 3.0 / n) < 1e-9


def test_harmonic_space_dimension_matches_betti_1():
    """dim ker(L1) = first Betti number b1. A filled tetrahedron surface is
    simply connected at the 1-level: b1 = 0. Remove all triangles and the same
    1-skeleton (K4) has b1 = |E| - |V| + 1 = 6 - 4 + 1 = 3 independent cycles."""
    filled = _tetrahedron()
    B1, B2 = build_incidences(filled)
    L1, _, _ = hodge_1_laplacian(B1, B2)
    eig = np.linalg.eigvalsh(L1)
    b1_filled = int(np.sum(eig < 1e-9))
    assert b1_filled == 0

    hollow = Complex(n_nodes=4, edges=filled.edges, triangles=[])
    B1h, B2h = build_incidences(hollow)
    L1h, _, _ = hodge_1_laplacian(B1h, B2h)
    eigh = np.linalg.eigvalsh(L1h)
    b1_hollow = int(np.sum(eigh < 1e-9))
    assert b1_hollow == 3
