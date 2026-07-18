"""GPU-batched Monte-Carlo for the lifted-covariance NNLS estimator.

Genuine GPU workload: sample ``B`` curl-domain covariances on-device and solve
all ``B`` NNLS problems with batched einsum GEMMs (``tfl.gpu``).  Headline
``p~=1000`` (fully batched) and stretch ``p~=10000`` (tiled), on planted
overlapping-``K_s`` complexes.  Cross-checks the batched GPU FISTA against the
CPU matrix-free FISTA on a small complex, and logs GPU name / VRAM / peak
memory / wall time for the manifest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import save_json  # noqa: E402
from tfl.complexes import planted_signatures  # noqa: E402
from tfl.gpu import mc_recovery_grid, same_input_crosscheck  # noqa: E402
from tfl.estimators_mf import nnls_lifted_support_mf  # noqa: E402

SIGMA_N = 1.0


def _signatures(p_target, s, seed=0):
    # fast GEMM-based range finder (exact U^T U = B2^T B2); avoids the O(E^2 p)
    # LAPACK SVD that dominates at p ~ 1e4.
    _cx, U, _rep = planted_signatures(p_target, block_size=s, overlap="edge", seed=seed)
    return U


def _gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception:
        return "nvidia-smi unavailable"


def _same_input_crosscheck():
    """SAME U, C, support, observations on CPU (numpy) and GPU (torch).

    Generate ONE dataset ``Z`` (fixed seed), form ``C = Sigma_hat - sigma_n^2 I``
    once, and hand the identical ``(U, C)`` to both solvers. Compare weights,
    objective, KKT residual, and thresholded support -- run in float64 so any
    gap is algorithmic, not precision. Also report an aggregate recovery
    cross-check over a batch for context."""
    U = _signatures(60, 5, seed=1)
    r, p = U.shape
    k, N, rho2 = 3, 300, 1.0
    sc = np.sqrt(rho2) * SIGMA_N
    rng = np.random.default_rng(5)
    a = np.zeros(p, bool); a[rng.choice(p, k, replace=False)] = True
    Z = U[:, a] @ (sc * rng.standard_normal((k, N))) + SIGMA_N * rng.standard_normal((r, N))
    Sigma = (Z @ Z.T) / N
    C = Sigma - SIGMA_N ** 2 * np.eye(r)
    cc = same_input_crosscheck(U, C, device="cuda", dtype="float64")
    cc["true_support_size"] = int(a.sum())
    # aggregate recovery agreement over a batch (context, float32 GPU MC path)
    thr = rho2 * SIGMA_N ** 2 / 2
    B = 200
    rng2 = np.random.default_rng(7)
    cpu_hits = 0
    for _ in range(B):
        aa = np.zeros(p, bool); aa[rng2.choice(p, k, replace=False)] = True
        Zb = U[:, aa] @ (sc * rng2.standard_normal((k, N))) + SIGMA_N * rng2.standard_normal((r, N))
        sup, _ = nnls_lifted_support_mf(Zb, U, SIGMA_N, thr, solver="fista")
        cpu_hits += int(np.array_equal(sup, aa))
    g = mc_recovery_grid(U, k, rho2, N, B_total=B, B_tile=B, device="cuda")
    cc["batch_cpu_recovery"] = cpu_hits / B
    cc["batch_gpu_recovery"] = g["recovery"]
    cc["batch_gpu_converged_frac"] = g["converged_frac"]
    return cc


def run() -> dict:
    import torch
    gpu = _gpu_info()
    print("GPU:", gpu, "| torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)

    cross = _same_input_crosscheck()
    print("same-input cross-check p=%d (float64): |dw|=%.2e obj_diff=%.2e "
          "kkt_cpu=%.2e kkt_gpu=%.2e support_agree=%s (iters=%d)" %
          (cross["p"], cross["max_abs_weight_diff"], cross["abs_obj_diff"],
           cross["kkt_cpu"], cross["kkt_gpu"], cross["support_agree"],
           cross["gpu_iters"]), flush=True)
    print("  batch recovery: cpu %.3f vs gpu %.3f (gpu converged frac %.3f)" %
          (cross["batch_cpu_recovery"], cross["batch_gpu_recovery"],
           cross["batch_gpu_converged_frac"]), flush=True)

    runs = []

    def _flush():
        save_json("gpu_mc.json", {"gpu": gpu, "torch": torch.__version__,
                                  "crosscheck": cross, "runs": runs, "sigma_n": SIGMA_N})

    # headline p~1000 fully batched: recovery vs N (tight CIs, B=1000)
    U1 = _signatures(1000, 6, seed=0); r1 = U1.shape[0]
    print("built p=%d r=%d (dof %.3f)" % (U1.shape[1], r1, r1 / U1.shape[1]), flush=True)
    for N in [100, 200, 400]:
        g = mc_recovery_grid(U1, k=16, rho2=1.0, N=N, B_total=1000, B_tile=1000, device="cuda")
        g["regime"] = "headline_fully_batched"
        runs.append(g); _flush()
        print("p=%d N=%d: rec %.3f [%.3f,%.3f] | %.1fs | peak %.0f MB | "
              "kkt_max %.1e conv %.2f iters %.0f" %
              (g["p"], N, g["recovery"], g["recovery_ci95"][0], g["recovery_ci95"][1],
               g["wall_s"], g["peak_gpu_mb"], g["kkt_max"], g["converged_frac"],
               g["mean_iters"]), flush=True)

    # stretch p~10000 tiled: a FEASIBILITY measurement (small B, capped iters)
    # in the strongly-rank-deficient regime (s=16, dof~0.19, r~1890), which is
    # both the interesting (most confusers) and the affordable regime.
    U2 = _signatures(10000, 16, seed=0); r2 = U2.shape[0]
    print("built p=%d r=%d (dof %.3f)" % (U2.shape[1], r2, r2 / U2.shape[1]), flush=True)
    g = mc_recovery_grid(U2, k=24, rho2=1.0, N=400, B_total=64, B_tile=16,
                         device="cuda", max_iter=400)
    # This is a FEASIBILITY point unless the per-solve KKT residual actually
    # converged; the label is set from the diagnostics, not asserted a priori.
    g["regime"] = ("stretch_tiled_converged" if g["converged"]
                   else "stretch_tiled_feasibility")
    runs.append(g); _flush()
    print("p=%d N=400 (%s): rec %.3f [%.3f,%.3f] | %.1fs | peak %.0f MB | "
          "kkt_max %.1e conv %.2f iters %.0f/%d" %
          (g["p"], g["regime"], g["recovery"], g["recovery_ci95"][0],
           g["recovery_ci95"][1], g["wall_s"], g["peak_gpu_mb"], g["kkt_max"],
           g["converged_frac"], g["mean_iters"], g["max_iter"]), flush=True)

    _flush()
    return {"gpu": gpu, "runs": runs}


if __name__ == "__main__":
    run()
