"""Hodge machinery for simplicial 2-complexes.

Conventions
-----------
* Vertices are ``0..n-1``.
* An edge is an ordered pair ``(i, j)`` with ``i < j``, oriented ``i -> j``.
  Boundary  ``d1[i,j] = j - i``  =>  ``B1[j, e] = +1``, ``B1[i, e] = -1``.
* A triangle is an ordered triple ``(i, j, k)`` with ``i < j < k``.
  Boundary  ``d2[i,j,k] = [j,k] - [i,k] + [i,j]``  =>
      B2[(i,j), t] = +1,  B2[(i,k), t] = -1,  B2[(j,k), t] = +1.

With these orientations the fundamental identity ``B1 @ B2 == 0`` holds
(``d1 . d2 = 0``), so the gradient subspace ``im(B1.T)`` and the curl subspace
``im(B2)`` are orthogonal and the Hodge decomposition is an orthogonal split
``R^E = grad  (+)  curl  (+)  harmonic`` with ``harmonic = ker(L1)``.

Matrix shapes
-------------
* ``B1`` : ``(n_nodes, n_edges)``     node-edge incidence  (boundary d1)
* ``B2`` : ``(n_edges, n_triangles)`` edge-triangle incidence (boundary d2)
* ``L1 = B1.T @ B1 + B2 @ B2.T`` : ``(n_edges, n_edges)`` Hodge 1-Laplacian
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Edge = tuple[int, int]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class Complex:
    """A simplicial 2-complex: nodes, oriented edges, oriented triangles."""

    n_nodes: int
    edges: list[Edge]
    triangles: list[Triangle]

    def __post_init__(self) -> None:
        for (i, j) in self.edges:
            if not (0 <= i < j < self.n_nodes):
                raise ValueError(f"edge {(i, j)} must satisfy 0 <= i < j < n_nodes")
        for tri in self.triangles:
            i, j, k = tri
            if not (0 <= i < j < k < self.n_nodes):
                raise ValueError(f"triangle {tri} must satisfy 0 <= i < j < k < n_nodes")
            for e in ((i, j), (i, k), (j, k)):
                if e not in self._edge_index:
                    raise ValueError(f"triangle {tri} needs edge {e} in the 1-skeleton")

    @property
    def _edge_index(self) -> dict[Edge, int]:
        return {e: idx for idx, e in enumerate(self.edges)}

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)


def build_incidences(cx: Complex) -> tuple[np.ndarray, np.ndarray]:
    """Return dense ``(B1, B2)`` for the complex ``cx``.

    ``B1`` is ``(n_nodes, n_edges)``, ``B2`` is ``(n_edges, n_triangles)``.
    Dense is intentional: this project targets small/medium CPU-scale complexes
    where dense linear algebra (SVD, pinv, eig) is the right tool.
    """
    n, m = cx.n_nodes, cx.n_edges
    edge_idx = cx._edge_index

    B1 = np.zeros((n, m))
    for e, (i, j) in enumerate(cx.edges):
        B1[i, e] = -1.0
        B1[j, e] = +1.0

    B2 = np.zeros((m, cx.n_triangles))
    for t, (i, j, k) in enumerate(cx.triangles):
        B2[edge_idx[(i, j)], t] = +1.0
        B2[edge_idx[(i, k)], t] = -1.0
        B2[edge_idx[(j, k)], t] = +1.0
    return B1, B2


def hodge_1_laplacian(
    B1: np.ndarray, B2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(L1, L1_lower, L1_upper)``.

    ``L1_lower = B1.T @ B1`` couples edges through shared vertices;
    ``L1_upper = B2 @ B2.T`` couples edges through shared triangles;
    ``L1 = L1_lower + L1_upper``.
    """
    L1_lower = B1.T @ B1
    L1_upper = B2 @ B2.T
    return L1_lower + L1_upper, L1_lower, L1_upper


def divergence(f: np.ndarray, B1: np.ndarray) -> np.ndarray:
    """Node divergence of an edge flow: ``div f = B1 @ f`` (shape ``n_nodes``)."""
    return B1 @ f


def curl(f: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """Triangle curl of an edge flow: ``curl f = B2.T @ f`` (shape ``n_triangles``)."""
    return B2.T @ f


def project_onto_columns(f: np.ndarray, A: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Orthogonal projection of ``f`` onto the column space of ``A``.

    Uses the economy SVD of ``A`` for numerical stability (rank-deficient ``A``
    is expected: ``B1.T`` and ``B2`` are typically not full column rank).
    """
    if A.shape[1] == 0:
        return np.zeros_like(f)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = rcond * (s[0] if s.size else 0.0)
    U_r = U[:, s > tol]
    return U_r @ (U_r.T @ f)


def curl_subspace_basis(B2: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Orthonormal basis ``Q`` (columns) of the curl subspace ``im(B2)``.

    ``Q`` has shape ``(n_edges, r)`` with ``r = rank(B2)``. Projecting flows to
    ``z = Q.T f`` annihilates the gradient component exactly (``B2.T B1.T=0``)
    and the candidate-orthogonal harmonic component (``h in ker B2.T``), and
    yields the minimal full-rank coordinate in which the second-order
    identifiability theory (limits.py, "first- vs second-order" section) lives.
    """
    if B2.shape[1] == 0:
        return np.zeros((B2.shape[0], 0))
    U, s, _ = np.linalg.svd(B2, full_matrices=False)
    tol = rcond * (s[0] if s.size else 0.0)
    return U[:, s > tol]


def hodge_decomposition(
    f: np.ndarray, B1: np.ndarray, B2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an edge flow ``f`` into ``(gradient, curl, harmonic)`` components.

    * ``gradient`` lives in ``im(B1.T)`` (the "potential-driven" part);
    * ``curl`` lives in ``im(B2)`` (the "triangle-circulation" part — the *only*
      part that carries information about which triangles are filled);
    * ``harmonic = f - gradient - curl`` lives in ``ker(L1)`` (flows around holes).

    Because ``B1 @ B2 == 0``, ``im(B1.T) ⟂ im(B2)``, so these are genuine
    orthogonal components and ``gradient + curl + harmonic == f`` exactly.
    """
    g = project_onto_columns(f, B1.T)
    c = project_onto_columns(f, B2)
    h = f - g - c
    return g, c, h
