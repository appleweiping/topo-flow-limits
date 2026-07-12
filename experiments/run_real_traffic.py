"""Real road-network traffic study — the ACHIEVABILITY companion to the FX converse.

Two panels, honest labels:

(A) Geometry (fully real). Planar-ish road networks (TNTP: Sioux Falls,
    Eastern-Massachusetts, Anaheim) contain few 3-cliques and B2 has full
    column rank in all three: the curl degrees-of-freedom ratio is exactly 1 —
    every candidate triangle is identifiable, the opposite regime from the
    complete graph K9 (ratio 1/3). Sioux Falls and Anaheim triangles are even
    pairwise edge-disjoint (G = 3I); EMA has 10 edge-sharing pairs yet stays
    full rank.
    For networks with a published user-equilibrium flow we also report the Hodge
    energy split of the REAL flow: unlike arbitrage-free FX (curl ~ 1e-31 of
    gradient), real traffic carries a few percent of genuine curl energy.

(B) Recovery on Anaheim (planted signals on real geometry + real-flow
    background). Honest scope: Anaheim's triangles are pairwise edge-disjoint,
    so G = 3I and the recovery law COLLAPSES TO the edge-disjoint product law
    — this panel checks that law on a real-derived geometry, it does not
    exercise the geometry-aware machinery. The real UE background is a
    constant across snapshots, which temporal centering removes exactly (one
    degree of freedom, df = N-1); the gradient/harmonic nuisances are
    annihilated by the curl map. Those invariances are the POINT of the
    detector, but they also mean the surviving statistical problem is fully
    synthetic — stated plainly here and in the paper.

(C) Recovery on EMA (planted signals on real NON-TRIVIAL geometry). EMA has
    10 edge-sharing triangle pairs (G genuinely non-diagonal, yet full column
    rank), so the whitened detector's heterogeneous per-triangle laws
    v0_tau = sigma_n^2 (G^{-1})_{tau tau} do real work: this is the panel that
    exercises the geometry-aware theory on real infrastructure geometry. EMA
    has no published UE flow; no background is added (the Anaheim panel shows
    constant backgrounds are exactly removed anyway).

Ground truth caveat (stated in the paper): real traffic has no labelled
triangle support, so panels (B)/(C) use planted signals; the real ingredient
is the network geometry. The genuinely UNPLANTED real recovery lives in the
cyclone experiment (run_real_cyclone.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tfl.tntp import load_tntp_network  # noqa: E402
from tfl.hodge import build_incidences, hodge_decomposition  # noqa: E402
from tfl.generative import FlowParams, FlowDataset  # noqa: E402
from tfl.estimators import whitened_curl_detector_support, exact_recovery  # noqa: E402
from tfl.limits import (  # noqa: E402
    whitened_variances,
    heterogeneous_exact_recovery_probability,
    heterogeneous_recovery_union_bound,
)
from _util import save_json, savefig, FastFlowSampler  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "traffic"

NETWORKS = [
    ("SiouxFalls", "SiouxFalls_net.tntp", "SiouxFalls_flow.tntp"),
    ("EMA", "EMA_net.tntp", None),  # no published flow file for EMA
    ("Anaheim", "Anaheim_net.tntp", "Anaheim_flow.tntp"),
]

# Recovery-panel parameters
RHO = 3.0            # curl SNR  => sigma_curl = sigma_noise = 1
N_ACTIVE = 18        # Anaheim (54 candidates)
N_ACTIVE_EMA = 11    # EMA (33 candidates)
N_GRID = [6, 10, 15, 22, 32, 46, 66, 95]
N_TRIALS = 200
SEED = 7


def geometry_panel() -> list[dict]:
    rows = []
    for name, net_file, flow_file in NETWORKS:
        net = load_tntp_network(DATA / net_file, DATA / flow_file if flow_file else None,
                                name=name)
        cx = net.complex
        B1, B2 = build_incidences(cx)
        p = len(cx.triangles)
        rank = int(np.linalg.matrix_rank(B2)) if p else 0
        G = B2.T @ B2
        row = {
            "network": name,
            "n_nodes": cx.n_nodes,
            "n_edges": len(cx.edges),
            "n_triangles": p,
            "rank_B2": rank,
            "dof_ratio": (rank / p) if p else None,
            "triangles_edge_disjoint": bool(np.allclose(G, 3.0 * np.eye(p))) if p else None,
        }
        if net.real_flow is not None:
            f = net.real_flow
            g, c, h = hodge_decomposition(f, B1, B2)
            tot = float((f**2).sum())
            row["real_flow_energy_frac"] = {
                "gradient": float((g**2).sum() / tot),
                "curl": float((c**2).sum() / tot),
                "harmonic": float((h**2).sum() / tot),
            }
        rows.append(row)
    # dense-graph contrast (from the FX study): K9 DoF ratio = 3/9
    rows.append({
        "network": "K9 (FX, contrast)", "n_nodes": 9, "n_edges": 36,
        "n_triangles": 84, "rank_B2": 28, "dof_ratio": 28 / 84,
        "triangles_edge_disjoint": False,
    })
    return rows


def recovery_panel(seed: int = SEED) -> dict:
    net = load_tntp_network(DATA / "Anaheim_net.tntp", DATA / "Anaheim_flow.tntp",
                            name="Anaheim")
    cx = net.complex
    p = len(cx.triangles)
    sampler = FastFlowSampler(cx)
    G = sampler.B2_all.T @ sampler.B2_all
    assert np.allclose(G, 3.0 * np.eye(p)), "Anaheim triangles must be edge-disjoint"

    f_bg = net.real_flow / np.sqrt(np.mean(net.real_flow**2))  # unit-RMS background
    rng = np.random.default_rng(seed)
    active = np.zeros(p, dtype=bool)
    active[rng.choice(p, N_ACTIVE, replace=False)] = True

    sn = 1.0
    sc = float(np.sqrt(RHO / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=1.0, sigma_harm=0.5, sigma_noise=sn)
    v0s, v1s = whitened_variances(np.full(p, 1.0 / 3.0), sc, sn)

    emp, theory, union = [], [], []
    for N in N_GRID:
        hits = 0
        for _ in range(N_TRIALS):
            F = sampler.sample(active, params, N, rng) + f_bg[:, None]
            ds = FlowDataset(F=F, B1=sampler.B1, B2_all=sampler.B2_all,
                             active=active, params=params,
                             candidate_triangles=list(cx.triangles))
            est = whitened_curl_detector_support(ds, sc, sn, mode="bayes", center=True)
            hits += exact_recovery(est, active)
        emp.append(hits / N_TRIALS)
        # centering costs one degree of freedom; G = 3I makes the product law exact
        theory.append(heterogeneous_exact_recovery_probability(v0s, v1s, active, N - 1))
        union.append(heterogeneous_recovery_union_bound(v0s, v1s, active, N - 1))

    return {
        "network": "Anaheim", "n_candidates": p, "n_active": N_ACTIVE,
        "rho": RHO, "sigma_noise": sn, "sigma_grad": 1.0, "sigma_harm": 0.5,
        "background": "real UE flow, unit RMS", "detector": "whitened, centered (df=N-1)",
        "N_grid": N_GRID, "empirical": emp,
        "theory_product_exact": theory, "union_bound": union,
        "n_trials": N_TRIALS, "seed": seed,
    }


def recovery_panel_ema(seed: int = SEED) -> dict:
    """Planted recovery on EMA: full-column-rank B2 with 10 edge-sharing
    triangle pairs — non-diagonal G, heterogeneous per-triangle whitened laws.
    The product law is an independence APPROXIMATION here; the union bound is
    the rigorous guarantee."""
    net = load_tntp_network(DATA / "EMA_net.tntp", None, name="EMA")
    cx = net.complex
    p = len(cx.triangles)
    sampler = FastFlowSampler(cx)
    G = sampler.B2_all.T @ sampler.B2_all
    assert not np.allclose(G, 3.0 * np.eye(p)), "EMA must be edge-sharing"
    Gp_diag = np.clip(np.diag(np.linalg.pinv(G)), 1e-12, None)
    n_sharing_pairs = int((np.abs(np.triu(G, 1)) > 0).sum())

    rng = np.random.default_rng(seed)
    active = np.zeros(p, dtype=bool)
    active[rng.choice(p, N_ACTIVE_EMA, replace=False)] = True

    sn = 1.0
    sc = float(np.sqrt(RHO / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=1.0, sigma_harm=0.5, sigma_noise=sn)
    v0s, v1s = whitened_variances(Gp_diag, sc, sn)

    emp, theory, union = [], [], []
    for N in N_GRID:
        hits = 0
        for _ in range(N_TRIALS):
            F = sampler.sample(active, params, N, rng)
            ds = FlowDataset(F=F, B1=sampler.B1, B2_all=sampler.B2_all,
                             active=active, params=params,
                             candidate_triangles=list(cx.triangles))
            est = whitened_curl_detector_support(ds, sc, sn, mode="bayes")
            hits += exact_recovery(est, active)
        emp.append(hits / N_TRIALS)
        theory.append(heterogeneous_exact_recovery_probability(v0s, v1s, active, N))
        union.append(heterogeneous_recovery_union_bound(v0s, v1s, active, N))

    return {
        "network": "EMA", "n_candidates": p, "n_active": N_ACTIVE_EMA,
        "n_edge_sharing_pairs": n_sharing_pairs,
        "Gp_diag_range": [float(Gp_diag.min()), float(Gp_diag.max())],
        "rho": RHO, "sigma_noise": sn, "sigma_grad": 1.0, "sigma_harm": 0.5,
        "detector": "whitened (heterogeneous v0_tau), uncentered (df=N)",
        "N_grid": N_GRID, "empirical": emp,
        "theory_product_approx": theory, "union_bound": union,
        "n_trials": N_TRIALS, "seed": seed,
    }


def _plot(geo: list[dict], rec: dict, rec_ema: dict):
    import matplotlib.pyplot as plt

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.0, 3.0))

    names = [r["network"].replace(" (FX, contrast)", "\n(FX)") for r in geo]
    ratios = [r["dof_ratio"] for r in geo]
    colors = ["tab:green"] * (len(geo) - 1) + ["tab:red"]
    a0.bar(names, ratios, color=colors, width=0.55)
    for x, r in zip(range(len(geo)), geo):
        a0.text(x, ratios[x] + 0.02, f"{r['rank_B2']}/{r['n_triangles']}",
                ha="center", fontsize=9)
    a0.set_ylim(0, 1.15)
    a0.set_ylabel("curl DoF ratio  rank$(B_2)/p$")
    a0.set_title("(A) Road networks: every triangle\nidentifiable (ratio 1) vs dense $K_9$")
    a0.grid(alpha=0.3, axis="y")

    N = rec["N_grid"]
    a1.plot(N, rec["empirical"], "^-", label="whitened (centered): empirical")
    a1.plot(N, rec["theory_product_exact"], "k--",
            label="exact product law (df=$N{-}1$)")
    a1.plot(N, rec["union_bound"], "k:", label="union bound")
    a1.set_xlabel("number of snapshots  N")
    a1.set_ylabel("P(exact recovery)")
    a1.set_title(f"(B) Anaheim ($G{{=}}3I$: edge-disjoint\nproduct law is exact; "
                 f"$\\rho$={rec['rho']:.0f}, {rec['n_active']}/{rec['n_candidates']})")
    a1.set_ylim(-0.03, 1.03)
    a1.legend(loc="lower right", fontsize=8)
    a1.grid(alpha=0.3)

    Ne = rec_ema["N_grid"]
    a2.plot(Ne, rec_ema["empirical"], "^-", color="tab:purple",
            label="whitened: empirical")
    a2.plot(Ne, rec_ema["theory_product_approx"], "k--",
            label="product law (indep. approx.)")
    a2.plot(Ne, rec_ema["union_bound"], "k:", label="union bound (rigorous)")
    a2.set_xlabel("number of snapshots  N")
    a2.set_ylabel("P(exact recovery)")
    a2.set_title(f"(C) EMA: non-diagonal $G$\n({rec_ema['n_edge_sharing_pairs']} edge-sharing pairs, "
                 f"{rec_ema['n_active']}/{rec_ema['n_candidates']})")
    a2.set_ylim(-0.03, 1.03)
    a2.legend(loc="lower right", fontsize=8)
    a2.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "real_traffic.png")


def run() -> dict:
    geo = geometry_panel()
    rec = recovery_panel()
    rec_ema = recovery_panel_ema()
    out = {"geometry": geo, "recovery": rec, "recovery_ema": rec_ema}
    save_json("real_traffic.json", out)
    _plot(geo, rec, rec_ema)
    return out


if __name__ == "__main__":
    res = run()
    an = [r for r in res["geometry"] if r["network"] == "Anaheim"][0]
    print("traffic geometry: Anaheim %d/%d curl DoF (ratio %.2f), G=3I edge-disjoint=%s"
          % (an["rank_B2"], an["n_triangles"], an["dof_ratio"],
             an["triangles_edge_disjoint"]))
    rec = res["recovery"]
    i50 = next((i for i, v in enumerate(rec["empirical"]) if v >= 0.5), None)
    print("recovery at N=%d: empirical=%.2f theory=%.2f"
          % (rec["N_grid"][-1], rec["empirical"][-1], rec["theory_product_exact"][-1]))
