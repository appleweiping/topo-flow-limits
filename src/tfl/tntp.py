"""Loader for TNTP road networks (github.com/bstabler/TransportationNetworks).

Road networks are the *achievability-friendly* real geometry for latent-triangle
identification: planar-ish street graphs contain few 3-cliques, those cliques
rarely share edges, and ``B2`` has full column rank — the opposite regime from
the dense complete graphs (``3/n`` degrees-of-freedom ratio) of the FX study.

Files vendored under ``data/traffic/`` (see ``data/fetch_traffic.py``):
  * ``*_net.tntp``  — directed link list; we form the simple undirected graph.
  * ``*_flow.tntp`` — a user-equilibrium link-flow solution; we form the
    *net* undirected flow (difference of the two directions, sign convention
    ``i -> j`` positive for ``i < j``, matching ``build_incidences``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tfl.hodge import Complex


def _enumerate_triangles(n_nodes: int, edges: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """All 3-cliques (i<j<k) via neighbour intersection (fast on sparse graphs)."""
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    tris: list[tuple[int, int, int]] = []
    for i, j in edges:  # i < j by construction
        for k in sorted(adj[i] & adj[j]):
            if k > j:
                tris.append((i, j, k))
    return sorted(tris)


@dataclass(frozen=True)
class TrafficNetwork:
    """A TNTP road network as a simplicial 2-complex plus (optionally) the
    real equilibrium net flow on its undirected edges."""

    name: str
    complex: Complex
    node_ids: list[int]          # original TNTP node numbers, index-aligned
    real_flow: np.ndarray | None  # (n_edges,) net UE flow, or None if no flow file


def load_tntp_network(net_path: str | Path, flow_path: str | Path | None = None,
                      name: str | None = None) -> TrafficNetwork:
    net_path = Path(net_path)
    directed: set[tuple[int, int]] = set()
    with net_path.open(encoding="utf-8", errors="replace") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not in_data:
                if line.startswith("<END OF METADATA>"):
                    in_data = True
                continue
            if not line or line.startswith("~") or line.startswith("<"):
                continue
            parts = line.rstrip(";").split()
            try:
                i, j = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue
            if i != j:
                directed.add((i, j))

    und = sorted({(min(i, j), max(i, j)) for i, j in directed})
    nodes = sorted({v for e in und for v in e})
    idx = {v: k for k, v in enumerate(nodes)}
    edges = [(idx[i], idx[j]) for i, j in und]
    triangles = _enumerate_triangles(len(nodes), edges)
    cx = Complex(n_nodes=len(nodes), edges=edges, triangles=triangles)

    real_flow = None
    if flow_path is not None:
        vol: dict[tuple[int, int], float] = {}
        with Path(flow_path).open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
                except ValueError:
                    continue
                a, b = (i, j) if i < j else (j, i)
                vol[(a, b)] = vol.get((a, b), 0.0) + (v if i < j else -v)
        real_flow = np.array([vol.get((nodes[i], nodes[j]), 0.0) for i, j in edges])

    return TrafficNetwork(
        name=name or net_path.stem.replace("_net", ""),
        complex=cx, node_ids=nodes, real_flow=real_flow,
    )
