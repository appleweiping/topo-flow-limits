"""Random / planted clique-complex generators for the scaling experiments.

The scaling study needs candidate sets of controllable size ``p`` (number of
3-cliques) and controllable rank deficiency ``dof = rank(B2)/p``.  A single
``K_s`` block has ``C(s,3)`` triangles and ``rank(B2) = C(s-1,2)``, so its dof
ratio is exactly ``3/s``.  A **union of overlapping** ``K_s`` blocks then hits
any target ``p ~ m C(s,3)`` at dof ``~ 3/s``, while the overlap couples the
per-block NNLS (disjoint blocks would decouple and make the large-``p``
benchmark trivial/dishonest).  Two extra generators (Erdos-Renyi, random
geometric) give distinct, "naturally random" geometries for the GNN
out-of-distribution test.
"""
from __future__ import annotations

from itertools import combinations
from math import comb, ceil

import numpy as np

from tfl.hodge import Complex, build_incidences


def _complex_from_triangles(n_nodes, triangles):
    """Build a Complex whose edge set is exactly the edges spanned by the
    given triangles (candidate curl geometry)."""
    tri = sorted({tuple(sorted(t)) for t in triangles})
    edges = sorted({tuple(sorted(e)) for t in tri for e in combinations(t, 2)})
    # relabel nodes to a compact 0..n-1 range actually used
    used = sorted({v for t in tri for v in t})
    remap = {v: i for i, v in enumerate(used)}
    tri = [tuple(remap[v] for v in t) for t in tri]
    edges = [tuple(remap[v] for v in e) for e in edges]
    return Complex(n_nodes=len(used), edges=edges, triangles=tri)


def planted_clique_complex(p_target: int, block_size: int = 6,
                           overlap: str = "edge", seed: int = 0) -> Complex:
    """Union of ``m`` overlapping ``K_s`` blocks (``s = block_size``) with
    ``p = m C(s,3) >= p_target`` triangles and dof ``~ 3/s``.  Consecutive
    blocks share ``overlap in {'none','vertex','edge'}`` vertices, coupling the
    problem through shared edges without merging triangles."""
    s = block_size
    tpb = comb(s, 3)
    m = max(1, ceil(p_target / tpb))
    ov = {"none": 0, "vertex": 1, "edge": 2}[overlap]
    blocks, prev, nxt = [], [], 0
    for i in range(m):
        if i == 0 or ov == 0:
            vs = list(range(nxt, nxt + s)); nxt += s
        else:
            shared = prev[-ov:]
            new = list(range(nxt, nxt + (s - ov))); nxt += (s - ov)
            vs = shared + new
        blocks.append(vs); prev = vs
    tri = set()
    for vs in blocks:
        for c in combinations(sorted(vs), 3):
            tri.add(c)
    return _complex_from_triangles(nxt, tri)


def erdos_renyi_clique_complex(n: int, q: float, seed: int = 0) -> Complex:
    """All 3-cliques of an Erdos-Renyi ``G(n,q)`` graph.  Sparse ``q`` -> few
    tetrahedra -> dof ~ 1; denser ``q`` -> stronger rank deficiency."""
    rng = np.random.default_rng(seed)
    adj = rng.random((n, n)) < q
    adj = np.triu(adj, 1)
    adj = adj | adj.T
    tri = []
    for i in range(n):
        Ni = np.where(adj[i])[0]
        Ni = Ni[Ni > i]
        for a_idx in range(len(Ni)):
            for b_idx in range(a_idx + 1, len(Ni)):
                a, b = int(Ni[a_idx]), int(Ni[b_idx])
                if adj[a, b]:
                    tri.append((i, a, b))
    if not tri:
        raise ValueError("empty complex; increase n or q")
    return _complex_from_triangles(n, tri)


def random_geometric_clique_complex(n: int, radius: float, dim: int = 2,
                                    seed: int = 0) -> Complex:
    """All 3-cliques of a random geometric graph (nodes in the unit cube,
    connect within ``radius``): locally clustered tetrahedra."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, dim))
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    adj = (d2 <= radius ** 2)
    np.fill_diagonal(adj, False)
    tri = []
    for i in range(n):
        Ni = np.where(adj[i])[0]; Ni = Ni[Ni > i]
        for a_idx in range(len(Ni)):
            for b_idx in range(a_idx + 1, len(Ni)):
                a, b = int(Ni[a_idx]), int(Ni[b_idx])
                if adj[a, b]:
                    tri.append((i, a, b))
    if not tri:
        raise ValueError("empty complex; increase n or radius")
    return _complex_from_triangles(n, tri)


def curl_signatures_fast(B2: np.ndarray, rank_hint: int | None = None,
                         oversample: int = 30, n_power: int = 2,
                         tol: float = 1e-9, seed: int = 0) -> np.ndarray:
    """Curl-domain signatures ``U = Q^T B2`` (``r x p``) via a GEMM-based
    randomized range finder instead of a full SVD -- ``O(E p l)`` GEMMs +
    a small ``O(E l^2)`` QR, vs the ``O(E^2 p)`` LAPACK ``gesdd`` that does not
    thread well and dominates at ``p ~ 1e4``.

    With ``l >= rank(B2)`` and a couple of power iterations, ``im(B2) subset
    im(Q)`` exactly, so ``U^T U = B2^T Q Q^T B2 = B2^T B2`` EXACTLY (verified in
    the round-8 rule-1 script) -- the estimator only sees the Gram, so this is
    an exact substitute for :func:`tfl.hodge.curl_subspace_basis` composed with
    ``Q^T B2``.  Rows with (near-)zero norm are trimmed to give ``r = rank``."""
    E, p = B2.shape
    if rank_hint is None:
        rank_hint = min(E, p)
    l = int(min(E, rank_hint + oversample))
    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((p, l))
    Q, _ = np.linalg.qr(B2 @ Omega)                 # (E, l)
    for _ in range(n_power):
        Q, _ = np.linalg.qr(B2 @ (B2.T @ Q))        # subspace (power) iteration
    U = Q.T @ B2                                     # (l, p)
    rn = np.linalg.norm(U, axis=1)
    keep = rn > tol * (rn.max() if rn.size else 1.0)
    return U[keep]


def planted_signatures(p_target: int, block_size: int = 6, overlap: str = "edge",
                       seed: int = 0):
    """Build a planted overlapping-``K_s`` complex and its curl signatures
    ``U`` (via the fast range finder, sized from the known planted rank
    ``m C(s-1,2)``).  Returns ``(cx, U, report)``."""
    from math import comb, ceil
    cx = planted_clique_complex(p_target, block_size=block_size, overlap=overlap, seed=seed)
    _, B2 = build_incidences(cx)
    s = block_size
    m = max(1, ceil(p_target / comb(s, 3)))
    rank_hint = m * comb(s - 1, 2) + 5 * s          # generous upper bound
    U = curl_signatures_fast(B2, rank_hint=rank_hint, seed=seed)
    r, p = U.shape
    rep = {"p": int(p), "rank_B2": int(r), "dof_ratio": r / p,
           "n_nodes": int(cx.n_nodes), "n_edges": int(B2.shape[0]),
           "block_size": s, "m_blocks": m}
    return cx, U, rep


def complex_report(cx: Complex, compute_rank: bool = True) -> dict:
    """Summary stats for a candidate complex: ``p`` (triangles), ``rank_B2``,
    ``dof_ratio``, ``n_nodes``, ``n_edges``, and the share-edge density
    (fraction of candidate pairs sharing an edge)."""
    B1, B2 = build_incidences(cx)
    p = B2.shape[1]
    out = {"p": int(p), "n_nodes": int(cx.n_nodes), "n_edges": int(B1.shape[1])}
    if compute_rank:
        r = int(np.linalg.matrix_rank(B2))
        out["rank_B2"] = r
        out["dof_ratio"] = r / p if p else float("nan")
    if p <= 4000:
        G = B2.T @ B2
        off = (np.abs(G) - np.diag(np.diag(G)))
        n_pairs = p * (p - 1)
        out["share_edge_density"] = float((np.abs(off) > 0.5).sum() / n_pairs) if n_pairs else 0.0
    return out
