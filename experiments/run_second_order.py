"""Second-order achievability: the lifted-covariance NNLS estimator across
K4-K8, versus first-order/subspace and greedy baselines, plus the
excitation-interpolation (alpha-sweep) study.

This is the estimator-side companion of the excitation-dependent
identifiability theory (limits.py):

  * MAIN GRID  — K4..K8, per-trial RANDOM supports at several sparsities k,
    a full (N, rho2) grid, isotropic excitation Gamma = rho2 * I (the
    diagonal class where the theory says every support is identifiable at
    any rank deficiency). Estimators compared per cell, with exact-recovery
    rates and 95% Wilson confidence intervals:
      - NNLS   : lifted-covariance non-negative least squares, threshold
                 w_min/2 (the rule analysed by the consistency theorem and
                 the explicit O(1/N) failure bound);
      - SUBSPACE: oracle-aided matched-subspace baseline (true k and true
                 dim im B_{2,S} given) — population scores see only the
                 image, finite-N tie-breaking rides on eigen-anisotropy;
      - GREEDY : the repository's greedy covariance-atom fitter.
    The full edge-domain pipeline is used (gradient nuisance included,
    annihilated by the curl projection).

  * ALPHA SWEEP — Gamma_alpha = rho2 [(1-alpha) I + alpha (B_S^T B_S)^+] on
    the K5 tetrahedron confuser: as alpha -> 1 the excitation approaches the
    projector case and equal-image distinguishability provably vanishes.
    Reported: the analytic covariance gap between the equal-image pair
    (normalised by its alpha=0 value) and the empirical NNLS exact-recovery
    probability; with the declared rho2/2 threshold, recovery collapses
    BEFORE the gap reaches zero (only the alpha=1 endpoint is threshold-free).

Outputs: results/second_order.json + results/figures/second_order.png
Runtime: ~15-25 min CPU (greedy dominates; every cap is logged, none silent).
Run:     python experiments/run_second_order.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import FastFlowSampler, save_json, savefig  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from tfl.estimators import (  # noqa: E402
    greedy_support,
    nnls_lifted_support,
    subspace_matched_support,
)
from tfl.generative import FlowDataset, FlowParams, all_triangles  # noqa: E402
from tfl.hodge import Complex, curl_subspace_basis  # noqa: E402
from tfl.limits import (  # noqa: E402
    excitation_covariance,
    interpolated_excitation_gamma,
    lifted_atom_matrix,
    nnls_recovery_bound,
)

N_GRID = [25, 50, 100, 200, 400]
RHO2_GRID = [0.25, 0.5, 1.0, 2.0]
N_TRIALS = 200
SIGMA_N = 1.0
SEED = 42

ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0]
ALPHA_N = 400
ALPHA_TRIALS = 200


def complete_complex(n: int) -> Complex:
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return Complex(n_nodes=n, edges=edges, triangles=all_triangles(n, edges))


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def k_values(p: int) -> list[int]:
    ks = sorted({2, max(2, int(round(p / 4))), max(3, int(round(p / 2)))})
    return [k for k in ks if k < p]


def run_main_grid(rng: np.random.Generator) -> dict:
    out = {}
    for n in range(4, 9):
        cx = complete_complex(n)
        sampler = FastFlowSampler(cx)
        B2 = sampler.B2_all
        Q = curl_subspace_basis(B2)
        U = Q.T @ B2
        p = B2.shape[1]
        rank = int(np.linalg.matrix_rank(B2))
        graph = {"p": p, "rank_B2": rank, "dof_ratio": rank / p, "cells": []}
        t0 = time.perf_counter()
        for k in k_values(p):
            for rho2 in RHO2_GRID:
                sc = float(np.sqrt(rho2)) * SIGMA_N
                params = FlowParams(sigma_curl=sc, sigma_grad=1.0,
                                    sigma_harm=0.0, sigma_noise=SIGMA_N)
                smin = np.linalg.svd(lifted_atom_matrix(U),
                                     compute_uv=False)[-1]
                for N in N_GRID:
                    hits = {"nnls": 0, "subspace": 0, "greedy": 0}
                    hits["nnls_gap"] = hits.get("nnls_gap", 0)
                    bound_worst = 0.0   # worst-case bound over the trial supports
                    for _ in range(N_TRIALS):
                        active = np.zeros(p, bool)
                        active[rng.choice(p, k, replace=False)] = True
                        # per-trial (per-support) bound; ||Sigma||_F^2 varies
                        # with the support's share-edge count, so we report the
                        # WORST case over supports, not one representative.
                        Sig = excitation_covariance(
                            U, active, rho2 * np.eye(k), SIGMA_N)
                        b = nnls_recovery_bound(Sig, smin, rho2 * SIGMA_N**2, N)
                        bound_worst = max(bound_worst, b)
                        F = sampler.sample(active, params, N, rng)
                        Z = Q.T @ F
                        sup, w_hat = nnls_lifted_support(
                            Z, U, SIGMA_N, threshold=rho2 * SIGMA_N**2 / 2)
                        hits["nnls"] += np.array_equal(sup, active)
                        # oracle-free variant: threshold at the midpoint of the
                        # LARGEST GAP in the sorted coefficients (no w_min)
                        ws = np.sort(w_hat)
                        gaps = np.diff(ws)
                        gi = int(np.argmax(gaps)) if len(gaps) else 0
                        thr_gap = (ws[gi] + ws[gi + 1]) / 2 if len(gaps) else 0.0
                        hits["nnls_gap"] += np.array_equal(w_hat > thr_gap,
                                                           active)
                        m = int(np.linalg.matrix_rank(B2[:, active]))
                        sup_s = subspace_matched_support(Z, U, k=k,
                                                         subspace_dim=m)
                        hits["subspace"] += np.array_equal(sup_s, active)
                        ds = FlowDataset(
                            F=F, B1=sampler.B1, B2_all=B2, active=active,
                            params=params,
                            candidate_triangles=list(cx.triangles))
                        sup_g = greedy_support(ds, sigma_noise=SIGMA_N)
                        hits["greedy"] += np.array_equal(sup_g, active)
                    cell = {"k": k, "rho2": rho2, "N": N, "n_trials": N_TRIALS}
                    for name, h in hits.items():
                        lo, hi = wilson_ci(h, N_TRIALS)
                        cell[name] = {"p_exact": h / N_TRIALS,
                                      "ci95": [lo, hi]}
                    # WORST-CASE derived bound over the trial supports: a valid
                    # per-cell upper bound on every trial's failure probability
                    # (not a single representative support).
                    cell["nnls_theory_bound"] = bound_worst
                    graph["cells"].append(cell)
        graph["runtime_s"] = round(time.perf_counter() - t0, 1)
        out[f"K{n}"] = graph
        print(f"  K{n}: p={p} rank={rank} cells={len(graph['cells'])} "
              f"({graph['runtime_s']}s)", flush=True)
    return out


def run_alpha_sweep(rng: np.random.Generator) -> dict:
    from tfl.limits import candidate_tetrahedra

    cx = complete_complex(5)
    sampler = FastFlowSampler(cx)
    B2 = sampler.B2_all
    Q = curl_subspace_basis(B2)
    U = Q.T @ B2
    p = B2.shape[1]
    quad = candidate_tetrahedra(cx.triangles)[0]
    s3 = np.zeros(p, bool); s3[list(quad[:3])] = True
    s4 = s3.copy(); s4[quad[3]] = True

    rho2 = 1.0
    gaps, recovery, ci = [], [], []
    params = FlowParams(sigma_curl=0.0, sigma_grad=1.0, sigma_harm=0.0,
                        sigma_noise=SIGMA_N)
    for alpha in ALPHAS:
        covs = []
        for s in (s3, s4):
            Gam = interpolated_excitation_gamma(B2, s, alpha, np.sqrt(rho2))
            covs.append(excitation_covariance(U, s, Gam, SIGMA_N))
        gaps.append(float(np.linalg.norm(covs[0] - covs[1])))

        Gam3 = interpolated_excitation_gamma(B2, s3, alpha, np.sqrt(rho2))
        evals, evecs = np.linalg.eigh(Gam3)
        gam_sqrt = evecs @ np.diag(np.sqrt(np.clip(evals, 0, None))) @ evecs.T
        hits = 0
        for _ in range(ALPHA_TRIALS):
            F = sampler.sample(s3, params, ALPHA_N, rng, gamma_sqrt=gam_sqrt)
            Z = Q.T @ F
            # SAME threshold rule as the main grid (w_min/2 at the alpha=0
            # weights). For alpha > 0 the diagonal NNLS model is deliberately
            # misspecified — that is the point of the experiment — so the
            # recovery of the specific S3 can collapse BEFORE the analytic
            # gap reaches zero; only the alpha=1 endpoint (identical
            # covariances) is threshold-free.
            sup, _ = nnls_lifted_support(Z, U, SIGMA_N,
                                         threshold=rho2 * SIGMA_N**2 / 2)
            hits += np.array_equal(sup, s3)
        recovery.append(hits / ALPHA_TRIALS)
        ci.append(list(wilson_ci(hits, ALPHA_TRIALS)))
        print(f"  alpha={alpha}: gap={gaps[-1]:.3f} "
              f"P(exact)={recovery[-1]:.3f}", flush=True)

    return {"alphas": ALPHAS, "gap_frobenius": gaps,
            "gap_normalised": [g / gaps[0] for g in gaps],
            "nnls_recovery": recovery, "nnls_recovery_ci95": ci,
            "N": ALPHA_N, "rho2": rho2, "n_trials": ALPHA_TRIALS,
            "note": "S = 3 faces of a K5 tetrahedron; equal-image "
                    "alternative = S + fourth face; Gamma_alpha = rho2 "
                    "[(1-alpha) I + alpha (B_S^T B_S)^+]; at alpha=1 the m=4 "
                    "equal-image supports (3-subsets of the tetra faces) have "
                    "identical distributions, so under a uniform prior no "
                    "estimator exceeds chance 1/m=1/4; the plotted value is "
                    "recovery of the SPECIFIC planted S (the thresholded NNLS "
                    "returns the wrong cardinality, giving 0), which is <= 1/m"}


def _plot(grid: dict, alpha: dict) -> None:
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 14,
                         "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    avail_N = sorted({c["N"] for g in grid.values() for c in g["cells"]})
    avail_r = sorted({c["rho2"] for g in grid.values() for c in g["cells"]})
    N_fix = 200 if 200 in avail_N else avail_N[-1]
    rho_fix = 1.0 if 1.0 in avail_r else avail_r[-1]

    import matplotlib.lines as mlines
    # DUAL ENCODING: color = complex K_n, linestyle = estimator (so the legend
    # factorizes cleanly instead of one entry per (K_n, estimator) pair).
    kn_colors = {n: plt.cm.viridis((n - 4) / 4.0) for n in range(4, 9)}
    est_styles = (("nnls", "-"), ("subspace", "--"), ("greedy", ":"))
    est_names = {"nnls": "NNLS", "subspace": "subspace", "greedy": "greedy"}
    ax = axes[0]
    for n in range(4, 9):
        g = grid[f"K{n}"]
        k_mid = k_values(g["p"])[min(1, len(k_values(g["p"])) - 1)]
        cells = [c for c in g["cells"] if c["rho2"] == rho_fix
                 and c["k"] == k_mid]
        cells.sort(key=lambda c: c["N"])
        Ns = [c["N"] for c in cells]
        col = kn_colors[n]
        for est, ls in est_styles:
            y = [c[est]["p_exact"] for c in cells]
            lo = [c[est]["ci95"][0] for c in cells]
            hi = [c[est]["ci95"][1] for c in cells]
            ax.plot(Ns, y, ls, color=col,
                    marker="o" if est == "nnls" else None, ms=3,
                    alpha=0.9 if est == "nnls" else 0.5)
            if est == "nnls":
                ax.fill_between(Ns, lo, hi, alpha=0.10, color=col)
    ax.set_xscale("log")
    ax.set_xticks(avail_N)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("snapshots N")
    ax.set_ylabel("P(exact recovery)")
    ax.set_title(f"(A) NNLS vs baselines ($\\rho_2$={rho_fix}, mid-$k$)")
    color_handles = [mlines.Line2D([], [], color=kn_colors[n], lw=2.2,
                                   label=f"$K_{n}$") for n in range(4, 9)]
    style_handles = [mlines.Line2D([], [], color="0.35", ls=ls, lw=1.6,
                                   marker="o" if est == "nnls" else None, ms=3,
                                   label=est_names[est]) for est, ls in est_styles]
    leg_color = ax.legend(handles=color_handles, fontsize=6.5, ncol=2,
                          loc="lower right", title="complex", title_fontsize=6.5)
    ax.add_artist(leg_color)
    ax.legend(handles=style_handles, fontsize=6.5, loc="upper left",
              title="estimator", title_fontsize=6.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for n in range(4, 9):
        g = grid[f"K{n}"]
        k_mid = k_values(g["p"])[min(1, len(k_values(g["p"])) - 1)]
        cells = [c for c in g["cells"] if c["N"] == N_fix and c["k"] == k_mid]
        cells.sort(key=lambda c: c["rho2"])
        xs = [c["rho2"] for c in cells]
        y = [c["nnls"]["p_exact"] for c in cells]
        lo = [c["nnls"]["ci95"][0] for c in cells]
        hi = [c["nnls"]["ci95"][1] for c in cells]
        line, = ax.plot(xs, y, "-o", ms=3, label=f"K{n} (rank {g['rank_B2']}/{g['p']})")
        ax.fill_between(xs, lo, hi, alpha=0.12, color=line.get_color())
    ax.set_xscale("log")
    ax.set_xticks(avail_r)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"excitation strength $\rho_2$")
    ax.set_ylabel("P(exact recovery)")
    ax.set_title(f"(B) NNLS across rank deficiency (N={N_fix})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(alpha["alphas"], alpha["gap_normalised"], "k--",
            label="analytic equal-image covariance gap (norm.)")
    y = alpha["nnls_recovery"]
    lo = [c[0] for c in alpha["nnls_recovery_ci95"]]
    hi = [c[1] for c in alpha["nnls_recovery_ci95"]]
    ax.plot(alpha["alphas"], y, "-o", ms=4, color="tab:red",
            label="NNLS P(exact), N=400")
    ax.fill_between(alpha["alphas"], lo, hi, alpha=0.15, color="tab:red")
    ax.set_xlabel(r"$\alpha$  in  $\Gamma_\alpha=(1-\alpha)I+\alpha(B_S^\top B_S)^+$")
    ax.set_ylabel("normalised gap  /  P(exact)")
    ax.set_title("(C) excitation interpolation: equal-image\n"
                 "distinguishability vanishes as $\\alpha\\to1$")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "second_order.png")


def main() -> None:
    rng = np.random.default_rng(SEED)
    print("[second-order] main grid (K4-K8) ...", flush=True)
    grid = run_main_grid(rng)
    print("[second-order] alpha sweep ...", flush=True)
    alpha = run_alpha_sweep(rng)
    save_json("second_order.json", {
        "config": {"N_grid": N_GRID, "rho2_grid": RHO2_GRID,
                   "n_trials": N_TRIALS, "sigma_n": SIGMA_N, "seed": SEED,
                   "support": "random k-subset per trial",
                   "threshold_rule": "w_min/2 (theorem's rule)"},
        "grid": grid, "alpha_sweep": alpha,
    })
    _plot(grid, alpha)
    # console summary
    for n in (5, 8):
        g = grid[f"K{n}"]
        c = [c for c in g["cells"]
             if c["rho2"] == 1.0 and c["N"] == 400][0]
        print(f"K{n} (rank {g['rank_B2']}/{g['p']}), k={c['k']}, rho2=1, "
              f"N=400: NNLS={c['nnls']['p_exact']:.2f} "
              f"subspace={c['subspace']['p_exact']:.2f} "
              f"greedy={c['greedy']['p_exact']:.2f}")
    print(f"alpha sweep: P(exact) {alpha['nnls_recovery'][0]:.2f} -> "
          f"{alpha['nnls_recovery'][-1]:.2f} as alpha 0 -> 1")


if __name__ == "__main__":
    main()
