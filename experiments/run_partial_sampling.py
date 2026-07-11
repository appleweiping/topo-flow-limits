"""Partial edge observation (supplement Sec. S2).

Setting: only a Bernoulli(q) random subset of edges is observed. A triangle's
curl statistic is computable iff all three of its edges are observed, so under
uniform sampling each triangle "survives" with probability q^3 — and an
unobservable ACTIVE triangle is an automatic miss, while an unobservable
INACTIVE triangle is an automatic correct rejection.

On Anaheim the 54 candidate triangles are pairwise edge-disjoint, so survival
events are independent across triangles and exact recovery admits the CLOSED
FORM (df = N-1, centered detector, real UE background as in Fig. 4):

    P(exact | q) = [ q^3 (1 - P_miss) ]^k  *  [ 1 - q^3 P_fa ]^(p-k) .

The experiment overlays this closed form on Monte-Carlo, plus the pure
geometry curve (fraction of observable triangles = q^3).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tfl.tntp import load_tntp_network  # noqa: E402
from tfl.generative import (  # noqa: E402
    FlowParams, FlowDataset, edge_subsample_mask, observable_triangles,
)
from tfl.estimators import whitened_curl_detector_support, exact_recovery  # noqa: E402
from tfl.limits import whitened_variances, _per_triangle_error_probs  # noqa: E402
from _util import save_json, savefig, FastFlowSampler  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "traffic"

RHO = 3.0
N_SNAP = 46          # a budget where full observation recovers w.h.p. (Fig. 4)
N_ACTIVE = 18
Q_GRID = [0.50, 0.60, 0.70, 0.80, 0.875, 0.925, 0.95, 0.975, 0.99, 1.00]
N_TRIALS = 200
SEED = 11


def run(seed: int = SEED) -> dict:
    net = load_tntp_network(DATA / "Anaheim_net.tntp", DATA / "Anaheim_flow.tntp",
                            name="Anaheim")
    cx = net.complex
    p = len(cx.triangles)
    sampler = FastFlowSampler(cx)
    f_bg = net.real_flow / np.sqrt(np.mean(net.real_flow**2))

    rng = np.random.default_rng(seed)
    active = np.zeros(p, dtype=bool)
    active[rng.choice(p, N_ACTIVE, replace=False)] = True

    sn = 1.0
    sc = float(np.sqrt(RHO / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=1.0, sigma_harm=0.5, sigma_noise=sn)

    # per-triangle marginal errors at df = N-1 (G = 3I: exact, homogeneous)
    v0s, v1s = whitened_variances(np.full(p, 1.0 / 3.0), sc, sn)
    errs = _per_triangle_error_probs(v0s, v1s, active, N_SNAP - 1, "bayes", 0.05)
    p_miss = float(errs[active][0])
    p_fa = float(errs[~active][0])

    emp, theory, obs_frac, emp_restricted = [], [], [], []
    k = int(active.sum())
    for q in Q_GRID:
        hits, hits_restricted, obs_count = 0, 0, 0
        for _ in range(N_TRIALS):
            emask = edge_subsample_mask(sampler.n_edges, q, rng)
            omask = observable_triangles(sampler.B2_all, emask)
            obs_count += int(omask.sum())
            F = sampler.sample(active, params, N_SNAP, rng) + f_bg[:, None]
            F = F * emask[:, None]     # unobserved edge flows are unavailable
            est = np.zeros(p, dtype=bool)
            if omask.any():
                ds = FlowDataset(F=F, B1=sampler.B1, B2_all=sampler.B2_all[:, omask],
                                 active=active[omask], params=params,
                                 candidate_triangles=[t for t, m in
                                                      zip(cx.triangles, omask) if m])
                est[omask] = whitened_curl_detector_support(
                    ds, sc, sn, mode="bayes", center=True)
                # restricted metric: recovery on the observable candidates only
                hits_restricted += int((est[omask] == active[omask]).all())
            hits += exact_recovery(est, active)
        emp.append(hits / N_TRIALS)
        emp_restricted.append(hits_restricted / N_TRIALS)
        obs_frac.append(obs_count / (N_TRIALS * p))
        theory.append((q**3 * (1.0 - p_miss))**k * (1.0 - q**3 * p_fa)**(p - k))

    out = {
        "network": "Anaheim", "n_candidates": p, "n_active": k,
        "rho": RHO, "N": N_SNAP, "q_grid": Q_GRID,
        "empirical": emp, "theory_closed_form": theory,
        "empirical_restricted_to_observable": emp_restricted,
        "observable_fraction_empirical": obs_frac,
        "observable_fraction_theory": [q**3 for q in Q_GRID],
        "p_miss": p_miss, "p_fa": p_fa,
        "n_trials": N_TRIALS, "seed": seed,
    }
    save_json("partial_sampling.json", out)
    _plot(out)
    return out


def _plot(out: dict):
    import matplotlib.pyplot as plt

    q = out["q_grid"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.0, 3.3))

    a0.plot(q, out["observable_fraction_empirical"], "o", label="empirical")
    a0.plot(q, out["observable_fraction_theory"], "-", label=r"theory  $q^3$")
    a0.set_xlabel("edge sampling rate  q")
    a0.set_ylabel("fraction of observable triangles")
    a0.set_title("(A) Triangle survival under\nBernoulli(q) edge sampling")
    a0.legend()
    a0.grid(alpha=0.3)

    a1.plot(q, out["empirical"], "^-", label="full support: empirical")
    a1.plot(q, out["theory_closed_form"], "k--", label="full support: exact closed form")
    a1.plot(q, out["empirical_restricted_to_observable"], "s:", color="tab:green",
            label="restricted to observable candidates")
    a1.set_xlabel("edge sampling rate  q")
    a1.set_ylabel("P(exact recovery)")
    a1.set_title(f"(B) Anaheim recovery vs sampling rate\n"
                 f"($\\rho$={out['rho']:.0f}, N={out['N']}, "
                 f"{out['n_active']}/{out['n_candidates']} active)")
    a1.set_ylim(-0.03, 1.03)
    a1.legend(loc="center left", fontsize=8)
    a1.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "partial_sampling.png")


if __name__ == "__main__":
    res = run()
    print("partial sampling: P(exact) at q=%.2f -> emp=%.2f theory=%.2f ; at q=1: emp=%.2f"
          % (res["q_grid"][2], res["empirical"][2], res["theory_closed_form"][2],
             res["empirical"][-1]))
