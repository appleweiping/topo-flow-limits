"""Edge-flow generative model with a latent filled-triangle support.

We observe ``T`` i.i.d. edge-flow snapshots

    f_t = B1.T a_t  +  B2_S y_t  +  h_t  +  n_t ,   t = 1..T

over a *known* 1-skeleton whose set of candidate triangles (all 3-cliques) is
``cx_all.triangles``. A latent boolean mask ``S`` says which candidates are
actually filled. Only the curl term ``B2_S y_t`` carries information about ``S``;
the gradient term ``B1.T a_t``, the harmonic term ``h_t`` and the noise ``n_t``
are nuisances. The estimator sees ``{f_t}`` and the full candidate list, and must
recover ``S``.

This is the model whose identifiability limits the project characterizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from tfl.hodge import Complex, build_incidences, hodge_1_laplacian


def all_triangles(n_nodes: int, edges: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """All 3-cliques (i<j<k) of the 1-skeleton — the candidate fillable triangles."""
    edge_set = set(edges)
    adj: dict[int, set[int]] = {v: set() for v in range(n_nodes)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    tris: list[tuple[int, int, int]] = []
    for i, j, k in combinations(range(n_nodes), 3):
        if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
            tris.append((i, j, k))
    return tris


@dataclass(frozen=True)
class FlowParams:
    """Component standard deviations of the flow generative model.

    * ``sigma_curl``  : triangle-signal scale (the *signal* — drives the curl term)
    * ``sigma_grad``  : node-potential scale (gradient nuisance)
    * ``sigma_harm``  : harmonic nuisance scale (flows around holes)
    * ``sigma_noise`` : per-edge white noise scale
    """

    sigma_curl: float
    sigma_grad: float = 1.0
    sigma_harm: float = 0.0
    sigma_noise: float = 1.0


def harmonic_basis(B1: np.ndarray, B2: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthonormal basis (columns) of the harmonic space ``ker(L1)``."""
    L1, _, _ = hodge_1_laplacian(B1, B2)
    w, V = np.linalg.eigh(L1)
    return V[:, w < tol]


@dataclass
class FlowDataset:
    """A sampled dataset plus everything an estimator/oracle may condition on."""

    F: np.ndarray            # (n_edges, T) observed flows
    B1: np.ndarray           # (n_nodes, n_edges)
    B2_all: np.ndarray       # (n_edges, n_candidate_triangles)  all candidate columns
    active: np.ndarray       # (n_candidate_triangles,) bool  -- the latent support S
    params: FlowParams
    candidate_triangles: list[tuple[int, int, int]]

    @property
    def T(self) -> int:
        return self.F.shape[1]


def sample_flows(
    cx_all: Complex,
    active: np.ndarray,
    params: FlowParams,
    T: int,
    rng: np.random.Generator,
) -> FlowDataset:
    """Draw ``T`` edge-flow snapshots. ``active`` is a boolean mask over
    ``cx_all.triangles`` marking the filled (signal-bearing) triangles."""
    B1, B2_all = build_incidences(cx_all)
    n_edges = B1.shape[1]
    active = np.asarray(active, dtype=bool)
    if active.shape[0] != B2_all.shape[1]:
        raise ValueError("active mask length must equal number of candidate triangles")

    B2_S = B2_all[:, active]
    H = harmonic_basis(B1, B2_all)

    F = np.zeros((n_edges, T))
    for t in range(T):
        grad = B1.T @ (params.sigma_grad * rng.standard_normal(B1.shape[0]))
        if B2_S.shape[1] > 0 and params.sigma_curl > 0:
            curl = B2_S @ (params.sigma_curl * rng.standard_normal(B2_S.shape[1]))
        else:
            curl = np.zeros(n_edges)
        if H.shape[1] > 0 and params.sigma_harm > 0:
            harm = H @ (params.sigma_harm * rng.standard_normal(H.shape[1]))
        else:
            harm = np.zeros(n_edges)
        noise = params.sigma_noise * rng.standard_normal(n_edges)
        F[:, t] = grad + curl + harm + noise

    return FlowDataset(
        F=F, B1=B1, B2_all=B2_all, active=active, params=params,
        candidate_triangles=list(cx_all.triangles),
    )


def curl_snr(cx_all: Complex, active: np.ndarray, params: FlowParams) -> float:
    """Population curl-SNR  ``rho = E||B2_S y||^2 / E||grad + harm + noise||^2``.

    Each triangle column of ``B2`` has squared norm 3 (three ±1 entries), so
    ``E||B2_S y||^2 = 3 * sigma_curl^2 * |S|``. The gradient nuisance energy is
    ``sigma_grad^2 * ||B1.T||_F^2 = sigma_grad^2 * tr(B1 B1.T) = sigma_grad^2 * sum(deg)``
    (``= 2 * sigma_grad^2 * n_edges``), harmonic energy ``sigma_harm^2 * b1``, and
    noise energy ``sigma_noise^2 * n_edges``.
    """
    B1, B2_all = build_incidences(cx_all)
    active = np.asarray(active, dtype=bool)
    n_edges = B1.shape[1]

    signal = 3.0 * params.sigma_curl**2 * int(active.sum())

    grad_energy = params.sigma_grad**2 * float((B1.T ** 2).sum())  # = sigma^2 * sum(deg)
    b1 = harmonic_basis(B1, B2_all).shape[1]
    harm_energy = params.sigma_harm**2 * b1
    noise_energy = params.sigma_noise**2 * n_edges
    denom = grad_energy + harm_energy + noise_energy
    return signal / denom if denom > 0 else np.inf
