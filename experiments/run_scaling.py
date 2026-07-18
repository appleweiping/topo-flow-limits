"""Scaling study: the matrix-free lifted-covariance NNLS estimator recovers
supports at ``p`` well beyond the dense ``(r^2, p)`` operator's reach.

For a range of target ``p`` on planted overlapping-``K_s`` complexes (dof
``= 3/s``), we run the Theorem-3 estimator in matrix-free form (FISTA; active-
set for a small-``p`` exact cross-check), measure exact-recovery rate (Wilson
CI), median solve time, and peak RSS, and record the size the DENSE operator
would have needed.  Everything is CPU here; the GPU-batched Monte-Carlo is
``run_gpu_mc.py``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import save_json, wilson_ci  # noqa: E402
from tfl.complexes import planted_clique_complex, complex_report  # noqa: E402
from tfl.hodge import build_incidences, curl_subspace_basis  # noqa: E402
from tfl.estimators_mf import (  # noqa: E402
    lipschitz_ATA, sigma_min_A, nnls_lifted_support_mf,
)
from tfl.limits import excitation_covariance, nnls_recovery_bound  # noqa: E402


def _peak_rss_mb() -> float:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS bytes
        return v / 1024.0 if v > 1e6 else v / 1024.0 if v < 1e9 else v / 1e6
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except Exception:
            return float("nan")


# (p_target, block_size s, N, k, n_trials)
CONFIGS = [
    (100, 6, 200, 4, 60),
    (300, 6, 300, 8, 40),
    (1000, 6, 400, 16, 30),
    (1000, 8, 400, 16, 30),
]
SIGMA_N = 1.0
RHO2 = 1.0
SEED = 0


def run() -> dict:
    rng = np.random.default_rng(SEED)
    sc = float(np.sqrt(RHO2)) * SIGMA_N
    thr = RHO2 * SIGMA_N ** 2 / 2.0
    cells = []
    for (p_t, s, N, k, n_tr) in CONFIGS:
        cx = planted_clique_complex(p_t, block_size=s, overlap="edge", seed=SEED)
        _, B2 = build_incidences(cx)
        Q = curl_subspace_basis(B2)
        U = Q.T @ B2
        r, p = U.shape
        rep = complex_report(cx, compute_rank=True)
        L = lipschitz_ATA(U)
        smin = sigma_min_A(U)
        # one representative bound (worst-case handled in run_second_order for the K-grid)
        a0 = np.zeros(p, bool); a0[:k] = True
        Sig0 = excitation_covariance(U, a0, RHO2 * np.eye(k), SIGMA_N)
        bound = nnls_recovery_bound(Sig0, smin, RHO2 * SIGMA_N ** 2, N)
        dense_gb = (r * r * p * 8) / 1e9   # size of the (r^2,p) dense operator

        hits = 0
        times = []
        use_as = (p <= 120)
        hits_as = 0
        for _ in range(n_tr):
            active = np.zeros(p, bool)
            active[rng.choice(p, k, replace=False)] = True
            Y = rng.standard_normal((k, N))
            Z = U[:, active] @ (sc * Y) + SIGMA_N * rng.standard_normal((r, N))
            t0 = time.perf_counter()
            sup, _w = nnls_lifted_support_mf(Z, U, SIGMA_N, thr, solver="fista", L=L)
            times.append(time.perf_counter() - t0)
            hits += int(np.array_equal(sup, active))
            if use_as:
                sup_a, _ = nnls_lifted_support_mf(Z, U, SIGMA_N, thr, solver="active_set")
                hits_as += int(np.array_equal(sup_a, active))
        lo, hi = wilson_ci(hits, n_tr)
        cell = {
            "p_target": p_t, "block_size": s, "p": int(p), "rank_B2": rep["rank_B2"],
            "dof_ratio": rep["dof_ratio"], "n_nodes": rep["n_nodes"], "N": N, "k": k,
            "n_trials": n_tr, "recovery_fista": hits / n_tr, "recovery_ci95": [lo, hi],
            "median_solve_s": float(np.median(times)),
            "lipschitz": float(L), "sigma_min_A": float(smin),
            "nnls_theory_bound": float(bound),
            "dense_operator_gb": float(dense_gb),
            "peak_rss_mb": _peak_rss_mb(),
        }
        if use_as:
            cell["recovery_active_set"] = hits_as / n_tr
        cells.append(cell)
        print(f"p={p} (s={s}) rank={rep['rank_B2']} dof={rep['dof_ratio']:.3f} N={N} k={k}: "
              f"rec {hits}/{n_tr} | {np.median(times):.3f}s/solve | dense would be {dense_gb:.1f} GB",
              flush=True)
    out = {"config": {"sigma_n": SIGMA_N, "rho2": RHO2, "threshold": thr, "seed": SEED},
           "cells": cells}
    save_json("scaling.json", out)
    return out


if __name__ == "__main__":
    run()
