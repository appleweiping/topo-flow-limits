"""Paper Fig. 2: the curl-invisibility phase transition (isotropic class).

Sweep snapshot budget ``T`` and per-triangle curl-SNR ``rho`` on a well-separated
(edge-disjoint) planted 2-complex; measure the empirical probability of *exact*
filled-triangle recovery by the Bayes energy detector; overlay the theoretical
50%-recovery contour ``rho*(T)`` (Chernoff limit with a Bonferroni union bound
over the ``p`` candidate triangles). The empirical transition tracks the theory
curve — evidence that the identifiability threshold is correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tfl.generative import FlowParams, disjoint_triangle_complex, all_triangles  # noqa: E402
from tfl.estimators import energy_detector_bayes_support, exact_recovery  # noqa: E402
from tfl.limits import invisibility_curl_snr_floor, recovery_contour_rho  # noqa: E402
from tfl.generative import FlowDataset  # noqa: E402
from _util import FastFlowSampler, save_json, savefig  # noqa: E402


def rho_to_sigma_curl(rho: float, sigma_noise: float) -> float:
    """Invert rho = 3 sigma_curl^2 / sigma_noise^2."""
    return float(np.sqrt(rho * sigma_noise**2 / 3.0))


def run(
    n_tri: int = 8,
    sigma_noise: float = 1.0,
    sigma_grad: float = 2.0,
    sigma_harm: float = 1.0,
    T_grid=None,
    rho_grid=None,
    n_trials: int = 200,
    seed: int = 0,
) -> dict:
    cx = disjoint_triangle_complex(n_tri=n_tri, n_cycles=2, cycle_len=5)
    # sanity: the candidate triangles are exactly the planted edge-disjoint ones
    assert set(all_triangles(cx.n_nodes, cx.edges)) == set(cx.triangles)
    p = cx.n_triangles
    active = np.zeros(p, dtype=bool)
    active[::2] = True  # plant every other triangle active

    if T_grid is None:
        T_grid = np.unique(np.round(np.geomspace(5, 90, 12)).astype(int))
    if rho_grid is None:
        rho_grid = np.geomspace(0.3, 30.0, 14)

    sampler = FastFlowSampler(cx)
    rng = np.random.default_rng(seed)
    recov = np.zeros((len(rho_grid), len(T_grid)))

    for ri, rho in enumerate(rho_grid):
        sc = rho_to_sigma_curl(rho, sigma_noise)
        params = FlowParams(sigma_curl=sc, sigma_grad=sigma_grad,
                            sigma_harm=sigma_harm, sigma_noise=sigma_noise)
        for ti, T in enumerate(T_grid):
            hits = 0
            for _ in range(n_trials):
                F = sampler.sample(active, params, int(T), rng)
                ds = FlowDataset(F=F, B1=sampler.B1, B2_all=sampler.B2_all,
                                 active=active, params=params,
                                 candidate_triangles=cx.triangles)
                est = energy_detector_bayes_support(ds, sc, sigma_noise)
                hits += exact_recovery(est, active)
            recov[ri, ti] = hits / n_trials

    n_active = int(active.sum())
    n_inactive = p - n_active
    # exact finite-sample 50%-recovery contour (the tight theoretical prediction)
    theory_exact = [recovery_contour_rho(sigma_noise, int(T), n_active, n_inactive, 0.5)
                    for T in T_grid]
    # asymptotic Chernoff+Bonferroni floor (the interpretable ~1/sqrt(T) scaling law)
    theory_floor = [invisibility_curl_snr_floor(sigma_noise, int(T), target_error=0.5 / p)
                    for T in T_grid]

    out = {
        "n_tri": p, "n_active": n_active, "sigma_noise": sigma_noise,
        "sigma_grad": sigma_grad, "sigma_harm": sigma_harm,
        "T_grid": [int(x) for x in T_grid], "rho_grid": [float(x) for x in rho_grid],
        "recovery": recov.tolist(), "theory_exact": theory_exact,
        "theory_floor": theory_floor, "n_trials": n_trials,
    }
    save_json("phase_transition.json", out)
    _plot(out)
    return out


def _plot(out: dict) -> None:
    import matplotlib.pyplot as plt

    T_grid = np.array(out["T_grid"])
    rho_grid = np.array(out["rho_grid"])
    recov = np.array(out["recovery"])

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.pcolormesh(T_grid, rho_grid, recov, shading="auto", cmap="viridis",
                       vmin=0, vmax=1)
    ax.plot(T_grid, out["theory_exact"], "w-", lw=2.4,
            label="exact theory (50% contour)")
    ax.plot(T_grid, out["theory_floor"], "w:", lw=1.8,
            label=r"asymptotic floor $\rho^\star\!\sim\!1/\sqrt{N}$")
    ax.set_yscale("log")
    ax.set_xlabel("number of snapshots  N")
    ax.set_ylabel(r"curl-SNR  $\rho$")
    ax.set_title("Curl-invisibility phase transition\n(exact triangle recovery)")
    ax.legend(loc="upper right", framealpha=0.85)
    fig.colorbar(im, ax=ax, label="P(exact recovery)")
    savefig(fig, "phase_transition.png")


if __name__ == "__main__":
    res = run()
    print("phase transition done; p=%d, active=%d" % (res["n_tri"], res["n_active"]))
    print("exact-theory rho* at N=%d: %.3f" % (res["T_grid"][-1], res["theory_exact"][-1]))
