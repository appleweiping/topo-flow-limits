"""Non-oracle support selection: BIC (headline) vs the oracle ``w_min/2`` rule,
the sample-split rule, and a faithful convex-sparse (non-negative lifted-LASSO)
baseline, as a function of the snapshot budget ``N``.

Shows (i) the BIC rule is selection-consistent and competitive with -- even
beating at small ``N`` -- the oracle it replaces, using no ``w_min`` and an
estimated ``sigma_n``; (ii) the NNLS estimator beats the generic convex-sparse
baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import save_json, wilson_ci  # noqa: E402
from tfl.hodge import Complex, build_incidences  # noqa: E402
from tfl.limits import curl_domain_signatures  # noqa: E402
from tfl.estimators_mf import nnls_lifted_support_mf, lasso_lifted_support  # noqa: E402
from tfl.selection import bic_nonoracle_support, sample_split_support  # noqa: E402

SIGMA_N = 1.0
RHO2 = 1.0
N_GRID = [50, 100, 200, 400, 800, 1600]
N_TRIALS = 200
SEED = 0


def _kn(n):
    E = [(i, j) for i in range(n) for j in range(i + 1, n)]
    T = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    cx = Complex(n_nodes=n, edges=E, triangles=T)
    _, B2 = build_incidences(cx)
    return curl_domain_signatures(B2)


def run() -> dict:
    rng = np.random.default_rng(SEED)
    sc = np.sqrt(RHO2) * SIGMA_N
    thr = RHO2 * SIGMA_N ** 2 / 2
    out = {"config": {"sigma_n": SIGMA_N, "rho2": RHO2, "n_trials": N_TRIALS,
                      "complex": "K6", "k": 3}, "curves": []}
    U = _kn(6); r, p = U.shape
    for N in N_GRID:
        h = {"bic": 0, "oracle": 0, "split": 0, "lasso": 0}
        sigma_rel_errs = []
        for _ in range(N_TRIALS):
            a = np.zeros(p, bool); a[rng.choice(p, 3, replace=False)] = True
            Z = U[:, a] @ (sc * rng.standard_normal((3, N))) + SIGMA_N * rng.standard_normal((r, N))
            mb, _, info = bic_nonoracle_support(Z, U, N); h["bic"] += int(np.array_equal(mb, a))
            # non-oracle sigma_n accuracy (true sigma_n = SIGMA_N): relative error
            sigma_rel_errs.append(abs(info["sigma_n_estimated"] - SIGMA_N) / SIGMA_N)
            mo, _ = nnls_lifted_support_mf(Z, U, SIGMA_N, thr); h["oracle"] += int(np.array_equal(mo, a))
            ms, _ = sample_split_support(Z, U); h["split"] += int(np.array_equal(ms, a))
            ml, _ = lasso_lifted_support(Z, U, SIGMA_N, N); h["lasso"] += int(np.array_equal(ml, a))
        row = {"N": N}
        for key, hits in h.items():
            lo, hi = wilson_ci(hits, N_TRIALS)
            row[key] = hits / N_TRIALS
            row[key + "_ci95"] = [lo, hi]
        row["sigma_n_median_rel_err"] = float(np.median(sigma_rel_errs))
        out["curves"].append(row)
        print(f"N={N:4d}: BIC {row['bic']:.3f} | oracle {row['oracle']:.3f} | "
              f"split {row['split']:.3f} | LASSO {row['lasso']:.3f} | "
              f"sigma_n relerr {row['sigma_n_median_rel_err']:.3f}", flush=True)
    save_json("selection.json", out)
    return out


if __name__ == "__main__":
    run()
