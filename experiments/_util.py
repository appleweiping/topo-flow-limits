"""Shared experiment utilities: paths, fast sampling, plotting defaults."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no window (respects window-management rules)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGDIR = RESULTS / "figures"


def ensure_dirs() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)


def save_json(name: str, obj: dict) -> Path:
    ensure_dirs()
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2))
    return p


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (shared helper)."""
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def savefig(fig, name: str) -> Path:
    ensure_dirs()
    p = FIGDIR / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return p


class FastFlowSampler:
    """Pre-factorized sampler for one complex: builds ``B1, B2_all, H`` once and
    draws snapshots without recomputing the Hodge basis each call."""

    def __init__(self, cx):
        from tfl.generative import harmonic_basis
        from tfl.hodge import build_incidences

        self.B1, self.B2_all = build_incidences(cx)
        self.n_edges = self.B1.shape[1]
        self.n_nodes = self.B1.shape[0]
        self.H = harmonic_basis(self.B1, self.B2_all)

    def sample(self, active: np.ndarray, params, T: int, rng,
               gamma_sqrt: np.ndarray | None = None) -> np.ndarray:
        """Return an ``(n_edges, T)`` flow matrix.

        ``gamma_sqrt`` (optional): a ``(k, k)`` square root of the triangle
        excitation covariance ``Gamma_S`` (``y_t = gamma_sqrt @ g_t`` with
        ``g_t`` standard normal), overriding the isotropic
        ``params.sigma_curl``. Pass ``sqrtm``/Cholesky/eigh-based roots — only
        ``gamma_sqrt @ gamma_sqrt.T = Gamma_S`` matters.
        """
        B2S = self.B2_all[:, np.asarray(active, bool)]
        F = self.B1.T @ (params.sigma_grad * rng.standard_normal((self.n_nodes, T)))
        if B2S.shape[1] > 0:
            if gamma_sqrt is not None:
                F += B2S @ (gamma_sqrt @ rng.standard_normal((B2S.shape[1], T)))
            elif params.sigma_curl > 0:
                F += B2S @ (params.sigma_curl * rng.standard_normal((B2S.shape[1], T)))
        if self.H.shape[1] > 0 and params.sigma_harm > 0:
            F += self.H @ (params.sigma_harm * rng.standard_normal((self.H.shape[1], T)))
        F += params.sigma_noise * rng.standard_normal((self.n_edges, T))
        return F
