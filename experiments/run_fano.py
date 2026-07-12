"""Fano-type converse curves for JOINT support recovery (supplement Sec. S1).

Plots the curl-SNR floor implied by the two Fano converses (Gaussian-KL and
signal-agnostic max-entropy) against the single-triangle Chernoff floor of the
main paper, for a small problem (the paper-Fig.-2 setting, p=8) and a large one
(p=10^4): the log(p) factor that joint recovery pays only becomes visible at
scale. Note the error normalizations differ (Fano at error 1/2 vs Chernoff at
delta=0.05), so the curves are not directly comparable in level: at p=10^4 the
Fano floor narrows the gap to ~0.5x the Chernoff floor but does not cross it
on this grid (crossing would need log(p/k) >~ 24).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tfl.limits import (  # noqa: E402
    invisibility_curl_snr_floor,
    fano_rho_floor,
    signal_agnostic_fano_min_snapshots,
    fano_min_snapshots,
)
from _util import save_json, savefig  # noqa: E402

N_GRID = [int(n) for n in np.unique(np.geomspace(5, 2000, 24).astype(int))]
CASES = [
    {"p": 8, "k": 4, "label": "$p=8,\\ k=4$ (Fig. 2 setting)"},
    {"p": 10_000, "k": 10, "label": "$p=10^4,\\ k=10$"},
]
ERR = 0.5   # Fano: error probability 1/2 (the standard converse normalization)


def run() -> dict:
    out = {"N_grid": N_GRID, "err": ERR, "cases": []}
    # single-triangle Chernoff floor (delta = 0.05, as in the paper)
    chern = [invisibility_curl_snr_floor(1.0, n, target_error=0.05) for n in N_GRID]
    out["chernoff_floor"] = chern
    for case in CASES:
        p, k = case["p"], case["k"]
        fano_g = [fano_rho_floor(p, k, n, err=ERR) for n in N_GRID]
        out["cases"].append({**case, "fano_gauss_floor": fano_g})
    # crossing behaviour of the two Fano N-bounds in rho: in the DENSE-support
    # regime (k = p/2) the signal-agnostic budget (p/2) log(1 + rho k/p) grows
    # only logarithmically in rho, so it beats the Gaussian-KL bound at high SNR
    rho_grid = list(np.geomspace(0.05, 200, 44))
    out["rho_grid"] = rho_grid
    out["dense_p"], out["dense_k"] = 100, 50
    out["fano_gauss_N"] = [fano_min_snapshots(100, 50, r, err=ERR) for r in rho_grid]
    out["fano_agnostic_N"] = [
        signal_agnostic_fano_min_snapshots(100, 50, r, err=ERR) for r in rho_grid]
    save_json("fano.json", out)
    _plot(out)
    return out


def _plot(out: dict):
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.0, 3.3))

    N = out["N_grid"]
    a0.loglog(N, out["chernoff_floor"], "k-",
              label="single-triangle Chernoff floor ($\\delta$=.05)")
    for case, style in zip(out["cases"], ("--", ":")):
        a0.loglog(N, case["fano_gauss_floor"], style,
                  label=f"Fano floor (err 1/2), {case['label']}")
    a0.set_xlabel("number of snapshots  N")
    a0.set_ylabel(r"curl-SNR floor  $\rho^\star$")
    a0.set_title("(A) Joint recovery pays a log $p$ factor:\nFano floors vs the per-triangle floor\n(different error normalizations)")
    a0.legend(fontsize=7)
    a0.grid(alpha=0.3, which="both")

    r = np.asarray(out["rho_grid"])
    g = np.asarray(out["fano_gauss_N"])
    a = np.asarray(out["fano_agnostic_N"])
    keep = (g >= 1.0) | (a >= 1.0)     # N < 1 is vacuous
    a1.loglog(r[keep], np.maximum(g[keep], 1.0), "-", label="Gaussian-KL Fano bound")
    a1.loglog(r[keep], np.maximum(a[keep], 1.0), "--", label="signal-agnostic Fano bound")
    a1.set_xlabel(r"curl-SNR  $\rho$")
    a1.set_ylabel("minimum snapshots  N (lower bound)")
    a1.set_title(f"(B) Dense supports ($p={out['dense_p']},k={out['dense_k']}$): "
                 "the two\nconverses cross — take the pointwise max")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3, which="both")

    fig.tight_layout()
    savefig(fig, "fano_bounds.png")


if __name__ == "__main__":
    res = run()
    print("fano: floors at N=%d -> chernoff=%.3f, fano(p=1e4)=%.3f"
          % (res["N_grid"][-1], res["chernoff_floor"][-1],
             res["cases"][1]["fano_gauss_floor"][-1]))
