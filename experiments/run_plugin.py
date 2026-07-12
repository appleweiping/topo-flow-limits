"""Plug-in variance estimation + fully adaptive detector (supplement Sec. S3).

The detectors of the paper take (sigma_c, sigma_n) as known. Here both are
estimated from the SAME data: sigma_n by a chi-square-median-calibrated MEDIAN
of the normalized whitened scores (robust while fewer than half the candidates
are active), sigma_c from the mean excess energy of a conservative
Bonferroni-screened set. The experiment shows (A) both estimates are
consistent, and (B) the fully adaptive detector's recovery curve is
indistinguishable from the known-parameter detector on the confusable strip
benchmark of the repo confusability figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tfl.generative import triangle_strip_complex, FlowParams, FlowDataset  # noqa: E402
from tfl.estimators import (  # noqa: E402
    whitened_curl_detector_support, adaptive_whitened_detector_support,
    estimate_noise_sigma, estimate_curl_sigma, exact_recovery,
)
from tfl.limits import whitened_variances, heterogeneous_exact_recovery_probability  # noqa: E402
from _util import save_json, savefig, FastFlowSampler  # noqa: E402

RHO = 8.0
N_TRI = 9
N_GRID = [10, 14, 20, 30, 45, 65, 95, 140, 200]
N_TRIALS = 300
SEED = 13


def run(seed: int = SEED) -> dict:
    cx = triangle_strip_complex(N_TRI)
    p = N_TRI
    active = np.zeros(p, dtype=bool)
    active[1::2] = True

    sn = 1.0
    sc = float(np.sqrt(RHO / 3.0))
    params = FlowParams(sigma_curl=sc, sigma_grad=2.0, sigma_harm=1.0, sigma_noise=sn)
    sampler = FastFlowSampler(cx)

    G = sampler.B2_all.T @ sampler.B2_all
    Gp_diag = np.diag(np.linalg.pinv(G))
    v0s, v1s = whitened_variances(Gp_diag, sc, sn)

    rng = np.random.default_rng(seed)
    rec_known, rec_plugin, theory = [], [], []
    err_sn_mean, err_sc_mean = [], []
    for N in N_GRID:
        hk = hp = 0
        errs_n, errs_c = [], []
        for _ in range(N_TRIALS):
            F = sampler.sample(active, params, N, rng)
            ds = FlowDataset(F=F, B1=sampler.B1, B2_all=sampler.B2_all,
                             active=active, params=params,
                             candidate_triangles=list(cx.triangles))
            hk += exact_recovery(
                whitened_curl_detector_support(ds, sc, sn, mode="bayes"), active)
            est, sc_hat, sn_hat = adaptive_whitened_detector_support(ds, mode="bayes")
            hp += exact_recovery(est, active)
            errs_n.append(abs(sn_hat - sn) / sn)
            errs_c.append(abs(sc_hat - sc) / sc if sc_hat > 0 else 1.0)
        rec_known.append(hk / N_TRIALS)
        rec_plugin.append(hp / N_TRIALS)
        theory.append(heterogeneous_exact_recovery_probability(v0s, v1s, active, N))
        err_sn_mean.append(float(np.mean(errs_n)))
        err_sc_mean.append(float(np.mean(errs_c)))

    out = {
        "benchmark": "edge-sharing strip (repo confusability-figure setting)",
        "n_tri": p, "n_active": int(active.sum()), "rho": RHO,
        "N_grid": N_GRID,
        "recovery_known": rec_known, "recovery_plugin": rec_plugin,
        "theory_known": theory,
        "rel_err_sigma_noise": err_sn_mean, "rel_err_sigma_curl": err_sc_mean,
        "n_trials": N_TRIALS, "seed": seed,
    }
    save_json("plugin.json", out)
    _plot(out)
    return out


def _plot(out: dict):
    import matplotlib.pyplot as plt

    N = out["N_grid"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.0, 3.3))

    a0.loglog(N, out["rel_err_sigma_noise"], "o-",
              label=r"$|\hat\sigma_n-\sigma_n|/\sigma_n$")
    a0.loglog(N, out["rel_err_sigma_curl"], "s-",
              label=r"$|\hat\sigma_c-\sigma_c|/\sigma_c$")
    a0.loglog(N, [out["rel_err_sigma_noise"][0] * np.sqrt(N[0] / n) for n in N],
              "k:", label=r"$\propto 1/\sqrt{N}$")
    a0.set_xlabel("number of snapshots  N")
    a0.set_ylabel("mean relative error")
    a0.set_title("(A) Plug-in variance estimates\nare consistent")
    a0.legend(fontsize=8)
    a0.grid(alpha=0.3, which="both")

    a1.plot(N, out["recovery_known"], "^-", label="known $(\\sigma_c,\\sigma_n)$")
    a1.plot(N, out["recovery_plugin"], "o--", label="fully adaptive (plug-in)")
    a1.plot(N, out["theory_known"], "k:", label="theory (known params)")
    a1.set_xlabel("number of snapshots  N")
    a1.set_ylabel("P(exact recovery)")
    a1.set_title("(B) Adaptive detector matches\nknown-parameter detector")
    a1.set_ylim(-0.03, 1.03)
    a1.legend(loc="lower right", fontsize=8)
    a1.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "plugin.png")


if __name__ == "__main__":
    res = run()
    gap = max(abs(a - b) for a, b in zip(res["recovery_known"], res["recovery_plugin"]))
    print("plugin: max |known - plugin| recovery gap = %.3f ; sigma_n rel-err @N=%d: %.3f"
          % (gap, res["N_grid"][-1], res["rel_err_sigma_noise"][-1]))
