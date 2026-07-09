"""Real-data study on daily foreign-exchange flows (ECB reference rates).

Currencies are the nodes of a complete graph; the edge flow on day ``t`` is the
log-price difference ``p_i,t - p_j,t``. Real FX flows land in the curl-invisible
regime along BOTH axes of the theory:

(A) SNR-side invisibility. An arbitrage-free market's flow is a pure gradient,
    so its curl energy is ~machine-zero relative to the gradient energy: there is
    no higher-order signal to detect at any snapshot budget.

(B) Geometry-side invisibility. On the complete currency graph K_n the number of
    candidate triangles C(n,3) far exceeds the curl-subspace dimension
    rank(B2) = C(n-1,2). Their ratio is exactly 3/n, so most triangle patterns
    are unidentifiable from edge flows regardless of SNR. On K9 (this dataset)
    only a 28-dimensional shadow of 84 candidate triangles is observable.

Data are cached in ``data/fx_rates.json`` (fetched once from api.frankfurter.app),
so the experiment is fully offline-reproducible.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tfl.hodge import Complex, build_incidences, hodge_decomposition  # noqa: E402
from _util import save_json, savefig  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "fx_rates.json"


def load_fx() -> tuple[list[str], np.ndarray]:
    raw = json.loads(DATA.read_text(encoding="utf-8-sig"))
    rates = raw["rates"]
    dates = sorted(rates.keys())
    symbols = sorted({s for d in dates for s in rates[d].keys()})
    currencies = [raw["base"]] + symbols
    logp = np.zeros((len(dates), len(currencies)))
    for ti, d in enumerate(dates):
        for si, s in enumerate(symbols, start=1):
            logp[ti, si] = -np.log(rates[d][s])
    return currencies, logp


def complete_complex(n: int) -> Complex:
    edges = [(i, j) for i, j in combinations(range(n), 2)]
    triangles = [(i, j, k) for i, j, k in combinations(range(n), 3)]
    return Complex(n_nodes=n, edges=edges, triangles=triangles)


def curl_dimension(cx: Complex) -> int:
    _, B2 = build_incidences(cx)
    return int(np.linalg.matrix_rank(B2))


def run() -> dict:
    currencies, logp = load_fx()
    n = len(currencies)
    cx = complete_complex(n)
    B1, B2 = build_incidences(cx)
    F_real = B1.T @ logp.T  # edge flows over days
    T = F_real.shape[1]

    # (A) SNR-side: Hodge energies of the real market flow
    grad_e = curl_e = harm_e = 0.0
    for t in range(T):
        g, c, h = hodge_decomposition(F_real[:, t], B1, B2)
        grad_e += float(g @ g)
        curl_e += float(c @ c)
        harm_e += float(h @ h)
    curl_ratio = curl_e / (grad_e + 1e-30)

    # (B) geometry-side: identifiability ratio 3/n, verified numerically for K5..K12
    ns = list(range(5, 13))
    ratios_num, ratios_theory = [], []
    for m in ns:
        cxm = complete_complex(m)
        dim = curl_dimension(cxm)
        p = cxm.n_triangles
        ratios_num.append(dim / p)
        ratios_theory.append(3.0 / m)

    out = {
        "currencies": currencies, "n_days": T, "n_edges": int(B1.shape[1]),
        "n_candidate_triangles": cx.n_triangles, "curl_dimension": curl_dimension(cx),
        "gradient_energy": grad_e, "curl_energy": curl_e, "harmonic_energy": harm_e,
        "curl_to_gradient_ratio": curl_ratio,
        "Kn_grid": ns, "identifiability_ratio_numeric": ratios_num,
        "identifiability_ratio_theory": ratios_theory,
    }
    save_json("real_fx.json", out)
    _plot(out)
    return out


def _plot(out: dict) -> None:
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.6, 4.2))

    labels = ["gradient", "curl", "harmonic"]
    energies = [max(out["gradient_energy"], 1e-30),
                max(out["curl_energy"], 1e-30),
                max(out["harmonic_energy"], 1e-30)]
    a0.bar(labels, energies, color=["#4477aa", "#ee6677", "#228833"])
    a0.set_yscale("log")
    a0.set_ylabel("Hodge component energy (log)")
    a0.set_title("(A) SNR-side invisibility\n"
                 f"real FX is curl-free: curl/grad = {out['curl_to_gradient_ratio']:.1e}\n"
                 f"({out['n_days']} days, {len(out['currencies'])} currencies)")

    ns = np.array(out["Kn_grid"])
    a1.plot(ns, out["identifiability_ratio_numeric"], "o", ms=7, label="numeric  rank(B2)/#triangles")
    a1.plot(ns, out["identifiability_ratio_theory"], "-", label=r"theory  $3/n$")
    a1.axvline(len(out["currencies"]), color="grey", ls=":", label=f"this dataset  K{len(out['currencies'])}")
    a1.set_xlabel("complete graph size  n")
    a1.set_ylabel("identifiable fraction")
    a1.set_title("(B) Geometry-side invisibility\n"
                 f"K{len(out['currencies'])}: {out['curl_dimension']} curl dims vs "
                 f"{out['n_candidate_triangles']} triangles")
    a1.legend()
    a1.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "real_fx.png")


if __name__ == "__main__":
    res = run()
    print("real FX: curl/gradient = %.2e (SNR-side curl-invisibility)" % res["curl_to_gradient_ratio"])
    print("K%d geometry: %d curl dims vs %d candidate triangles (ratio %.3f = 3/n)" % (
        len(res["currencies"]), res["curl_dimension"], res["n_candidate_triangles"],
        res["curl_dimension"] / res["n_candidate_triangles"]))
