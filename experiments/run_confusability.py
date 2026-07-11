"""Confusability experiment: why geometry-aware whitening is necessary.

On an edge-sharing triangle strip, active triangles leak curl energy onto their
inactive neighbours (Gram off-diagonals ``G_{sigma tau}=+-1``). The raw
curl-energy detector then false-alarms — and *worsens* as the curl-SNR grows,
because stronger signals leak harder. The whitened detector ``yhat = G^+ c``
decorrelates the triangles and recovers the support, matching the geometry-aware
finite-sample theory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tfl.generative import FlowParams, triangle_strip_complex, sample_flows  # noqa: E402
from tfl.estimators import (  # noqa: E402
    energy_detector_bayes_support,
    whitened_curl_detector_support,
    greedy_support,
    hamming_error,
    exact_recovery,
    triangle_gram,
)
from tfl.limits import (  # noqa: E402
    whitened_variances,
    heterogeneous_exact_recovery_probability,
    heterogeneous_recovery_union_bound,
)
from _util import save_json, savefig  # noqa: E402


def run(
    n_tri: int = 9,
    rho: float = 8.0,
    sigma_noise: float = 1.0,
    sigma_grad: float = 2.0,
    sigma_harm: float = 1.0,
    T_grid=None,
    n_trials: int = 600,
    seed: int = 0,
) -> dict:
    cx = triangle_strip_complex(n_tri)
    p = cx.n_triangles
    active = np.zeros(p, dtype=bool)
    active[1::2] = True  # alternate: every inactive triangle is flanked by actives
    sc = float(np.sqrt(rho * sigma_noise**2 / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=sigma_grad,
                        sigma_harm=sigma_harm, sigma_noise=sigma_noise)

    if T_grid is None:
        # start below the transition so Fig. 2 (right) actually stresses the
        # marginal theory + union bound through the whole S-curve
        T_grid = np.unique(np.round(np.geomspace(3, 200, 12)).astype(int))

    Gp_diag = np.diag(np.linalg.pinv(triangle_gram_of(cx)))
    v0s, v1s = whitened_variances(Gp_diag, sc, sigma_noise)

    ham = {"naive": [], "whitened": [], "greedy": []}
    exact_w = []
    theory_w = []
    union_w = []
    rng = np.random.default_rng(seed)
    for T in T_grid:
        h = {"naive": 0.0, "whitened": 0.0, "greedy": 0.0}
        ex = 0
        for _ in range(n_trials):
            ds = sample_flows(cx, active, params, int(T), rng)
            e_naive = energy_detector_bayes_support(ds, sc, sigma_noise)
            e_white = whitened_curl_detector_support(ds, sc, sigma_noise)
            e_greedy = greedy_support(ds, sigma_noise)
            h["naive"] += hamming_error(e_naive, active)
            h["whitened"] += hamming_error(e_white, active)
            h["greedy"] += hamming_error(e_greedy, active)
            ex += exact_recovery(e_white, active)
        for k in ham:
            ham[k].append(h[k] / n_trials)
        exact_w.append(ex / n_trials)
        theory_w.append(
            heterogeneous_exact_recovery_probability(v0s, v1s, active, int(T))
        )
        union_w.append(
            heterogeneous_recovery_union_bound(v0s, v1s, active, int(T))
        )

    out = {
        "n_tri": p, "n_active": int(active.sum()), "rho": rho, "sigma_noise": sigma_noise,
        "T_grid": [int(x) for x in T_grid], "hamming": ham,
        "exact_whitened": exact_w, "theory_whitened": theory_w,
        "union_bound_whitened": union_w,
        "rho_eff": (sc**2 / v0s).tolist(), "n_trials": n_trials,
    }
    save_json("confusability.json", out)
    _plot(out)
    return out


def triangle_gram_of(cx):
    from tfl.hodge import build_incidences

    _, B2 = build_incidences(cx)
    return B2.T @ B2


def _plot(out: dict) -> None:
    import matplotlib.pyplot as plt

    T = np.array(out["T_grid"])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    a0.plot(T, out["hamming"]["naive"], "o-", label="naive curl-energy")
    a0.plot(T, out["hamming"]["greedy"], "s-", label="greedy")
    a0.plot(T, out["hamming"]["whitened"], "^-", label="whitened (proposed)")
    a0.set_xlabel("number of snapshots  N")
    a0.set_ylabel("mean Hamming error")
    a0.set_title(f"Edge-sharing strip, curl-SNR $\\rho$={out['rho']}\n"
                 f"(p={out['n_tri']} candidates, {out['n_active']} active)")
    a0.legend()
    a0.grid(alpha=0.3)

    a1.plot(T, out["exact_whitened"], "^-", label="whitened: empirical")
    a1.plot(T, out["theory_whitened"], "k--",
            label="marginal law, indep. approx.")
    a1.plot(T, out["union_bound_whitened"], "k:",
            label="rigorous union bound")
    a1.set_xlabel("number of snapshots  N")
    a1.set_ylabel("P(exact recovery)")
    a1.set_title("Whitened detector vs\ngeometry-aware theory")
    a1.set_ylim(-0.03, 1.03)
    a1.legend(loc="lower right")
    a1.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "confusability.png")


if __name__ == "__main__":
    res = run()
    print("confusability done; naive worsens with leakage, whitened tracks theory")
    print("Hamming at N=%d: naive=%.2f whitened=%.2f greedy=%.2f" % (
        res["T_grid"][-1], res["hamming"]["naive"][-1],
        res["hamming"]["whitened"][-1], res["hamming"]["greedy"][-1]))
