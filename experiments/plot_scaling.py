"""Figure for the round-8 scaling + non-oracle-selection supplement section.
Reads results/{scaling,gpu_mc,selection}.json and writes scaling.png."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import RESULTS, savefig  # noqa: E402


def run() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 12.5,
                         "axes.labelsize": 13, "legend.fontsize": 9})
    sc = json.loads((RESULTS / "scaling.json").read_text())
    gp = json.loads((RESULTS / "gpu_mc.json").read_text())
    se = json.loads((RESULTS / "selection.json").read_text())

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (A) CPU matrix-free solve time vs p; dense operator size annotated
    cells = [c for c in sc["cells"] if c["block_size"] == 6]
    ps = [c["p"] for c in cells]
    tt = [c["median_solve_s"] for c in cells]
    a0.loglog(ps, tt, "o-", color="#4477aa", ms=6)
    for c in cells:
        if c["dense_operator_gb"] >= 0.05:
            a0.annotate(f"dense {c['dense_operator_gb']:.1f} GB",
                        (c["p"], c["median_solve_s"]), fontsize=8,
                        xytext=(4, -12), textcoords="offset points")
    a0.set_xlabel("candidate triangles $p$")
    a0.set_ylabel("matrix-free solve time (s)")
    a0.set_title("(A) matrix-free NNLS scales\n(dense $(r^2,p)$ operator OOMs)")
    a0.grid(alpha=0.3, which="both")

    # (B) GPU recovery vs N at p~1000, and the p~1e4 feasibility point
    head = [r for r in gp["runs"] if r.get("regime") == "headline_fully_batched"]
    Ns = [r["N"] for r in head]; rec = [r["recovery"] for r in head]
    lo = [r["recovery_ci95"][0] for r in head]; hi = [r["recovery_ci95"][1] for r in head]
    a1.errorbar(Ns, rec, yerr=[[rec[i]-lo[i] for i in range(len(rec))],
                               [hi[i]-rec[i] for i in range(len(rec))]],
                fmt="s-", color="#ee6677", capsize=3,
                label=f"p={head[0]['p']} (GPU, B={head[0]['B_total']})")
    st = [r for r in gp["runs"] if r.get("regime") == "stretch_tiled_feasibility"]
    if st:
        s = st[0]
        a1.errorbar([s["N"]], [s["recovery"]],
                    yerr=[[s["recovery"]-s["recovery_ci95"][0]], [s["recovery_ci95"][1]-s["recovery"]]],
                    fmt="D", color="#228833", capsize=3,
                    label=f"p={s['p']} (GPU tiled, B={s['B_total']})")
    a1.set_xlabel("snapshots $N$")
    a1.set_ylabel("exact-recovery rate")
    a1.set_title("(B) GPU-batched Monte-Carlo\n(RTX 4080, matrix-free FISTA)")
    a1.set_ylim(0.4, 1.03); a1.legend(loc="lower right"); a1.grid(alpha=0.3)

    # (C) non-oracle selection vs oracle / baselines
    cur = se["curves"]; Ns = [c["N"] for c in cur]
    for key, lab, col, ls in [("bic", "BIC (non-oracle)", "#4477aa", "-o"),
                              ("oracle", "oracle $w_{\\min}/2$", "#000000", "--"),
                              ("split", "sample-split", "#ee6677", ":s"),
                              ("lasso", "convex LASSO", "#228833", "-.^")]:
        a2.plot(Ns, [c[key] for c in cur], ls, color=col, ms=4, label=lab)
    a2.set_xscale("log")
    a2.set_xlabel("snapshots $N$")
    a2.set_ylabel("exact-recovery rate")
    a2.set_title("(C) non-oracle threshold (K6)\nBIC matches the oracle")
    a2.legend(loc="lower right"); a2.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, "scaling.png")
    print("wrote scaling.png")


if __name__ == "__main__":
    run()
