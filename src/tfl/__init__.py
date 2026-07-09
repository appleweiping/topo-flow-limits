"""topo-flow-limits (tfl)

Fundamental limits of identifying the latent filled-triangle set of a simplicial
2-complex from edge-flow observations. The higher-order structure is observed
*only* through the curl component of edge flows; this package builds the Hodge
machinery, the flow generative model, estimators, and the closed-form
information-theoretic thresholds that the experiments validate.
"""

from tfl.hodge import (
    build_incidences,
    hodge_1_laplacian,
    hodge_decomposition,
    curl,
    divergence,
    project_onto_columns,
)

__all__ = [
    "build_incidences",
    "hodge_1_laplacian",
    "hodge_decomposition",
    "curl",
    "divergence",
    "project_onto_columns",
]
