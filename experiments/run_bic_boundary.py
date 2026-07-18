"""The NON-ORACLE BIC selector has a NOISE-IDENTIFIABILITY BOUNDARY.

Round-9 honesty experiment. The paper's non-oracle rule estimates ``sigma_n``
from the median eigenvalue of the curl sample covariance, which is a *noise*
eigenvalue only while the active curl-covariance rank stays below ~r/2. Beyond
that the median eigenvalue is signal-inflated, ``sigma_n`` is overestimated, and
the selector FAILS even as ``N -> infinity`` -- it is therefore an EMPIRICAL
selector valid under noise-identifiability, NOT an unconditionally consistent
one.

This sweep documents the boundary on ``K6`` (r=10, p=20): median-eigenvalue
sigma_n vs an oracle sigma_n, over support size ``k`` at large ``N``. The oracle
column stays at 1.0 for every ``k`` (population identifiability holds -- the
lifted atoms are independent, ``rank(G o G)=p``); the median column collapses
for dense supports. Outputs results/bic_boundary.json.
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
from tfl.estimators_mf import nnls_lifted_fista  # noqa: E402
from tfl.selection import bic_nonoracle_support, bic_support_path  # noqa: E402

SIGMA_N = 1.0
RHO2 = 1.0
N = 100_000
N_TRIALS = 40
K_GRID = [3, 5, 6, 8, 10, 12, 15, 18, 20]
SEED = 0


def _kn(n):
    E = [(i, j) for i in range(n) for j in range(i + 1, n)]
    T = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    cx = Complex(n_nodes=n, edges=E, triangles=T)
    _, B2 = build_incidences(cx)
    return curl_domain_signatures(B2)


def run() -> dict:
    U = _kn(6)
    r, p = U.shape
    GoG = (U.T @ U) ** 2
    rank_atoms = int(np.linalg.matrix_rank(GoG))
    sc = np.sqrt(RHO2) * SIGMA_N
    out = {
        "config": {"complex": "K6", "r": r, "p": p, "sigma_n": SIGMA_N,
                   "rho2": RHO2, "N": N, "n_trials": N_TRIALS, "seed": SEED},
        "rank_lifted_atoms": rank_atoms,
        "atoms_independent": rank_atoms == p,
        "note": "median = shipped non-oracle rule (median-eigenvalue sigma_n); "
                "oracle = same BIC path given the true sigma_n. The median rule "
                "fails for dense supports (noise-identifiability boundary); the "
                "oracle stays at 1.0 (population identifiability holds). This is "
                "the retraction of the unconditional-consistency claim.",
        "curves": [],
    }
    rng = np.random.default_rng(SEED)
    for k in K_GRID:
        hitsA = hitsB = 0
        med_eigs, act_ranks = [], []
        for _ in range(N_TRIALS):
            a = np.zeros(p, bool); a[rng.choice(p, k, replace=False)] = True
            Z = U[:, a] @ (sc * rng.standard_normal((k, N))) + SIGMA_N * rng.standard_normal((r, N))
            Sig = (Z @ Z.T) / N
            med_eigs.append(float(np.median(np.linalg.eigvalsh(Sig))))
            act = (U[:, a] * (RHO2 * SIGMA_N ** 2)) @ U[:, a].T
            act_ranks.append(int(np.linalg.matrix_rank(act, tol=1e-9)))
            mA, _, _ = bic_nonoracle_support(Z, U, N)
            hitsA += int(np.array_equal(mA, a))
            C = Sig - SIGMA_N ** 2 * np.eye(r)
            w = nnls_lifted_fista(C, U)
            mB, _ = bic_support_path(Sig, U, w, SIGMA_N, N)
            hitsB += int(np.array_equal(mB, a))
        loA, hiA = wilson_ci(hitsA, N_TRIALS)
        loB, hiB = wilson_ci(hitsB, N_TRIALS)
        row = {"k": k, "median_sigma_n": hitsA / N_TRIALS,
               "median_ci95": [loA, hiA], "oracle_sigma_n": hitsB / N_TRIALS,
               "oracle_ci95": [loB, hiB],
               "median_eig_mean": float(np.mean(med_eigs)),
               "active_cov_rank_mean": float(np.mean(act_ranks))}
        out["curves"].append(row)
        print(f"k={k:>3}: median {row['median_sigma_n']:.3f} | "
              f"oracle {row['oracle_sigma_n']:.3f} | "
              f"med_eig {row['median_eig_mean']:.2f} (true sigma_n^2=1) | "
              f"act_rank {row['active_cov_rank_mean']:.1f}/{r}", flush=True)
    save_json("bic_boundary.json", out)
    return out


if __name__ == "__main__":
    run()
