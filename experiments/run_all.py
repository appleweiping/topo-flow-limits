"""One-click reproduction of every figure and number in the paper.

Runs all experiments from fixed seeds and writes to ``results/`` (+ figures).
CPU-only; completes in a couple of minutes on a laptop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_phase_transition
import run_confusability
import run_real_fx
import run_real_traffic


def main() -> None:
    t0 = time.perf_counter()
    print("[1/4] phase transition (curl-invisibility, exact recovery vs theory) ...")
    pt = run_phase_transition.run()
    print("      p=%d active=%d; exact-theory rho* at N=%d -> %.3f"
          % (pt["n_tri"], pt["n_active"], pt["T_grid"][-1], pt["theory_exact"][-1]))

    print("[2/4] confusability (naive vs whitened on edge-sharing strip) ...")
    cf = run_confusability.run()
    print("      Hamming@N=%d: naive=%.2f whitened=%.2f greedy=%.2f"
          % (cf["T_grid"][-1], cf["hamming"]["naive"][-1],
             cf["hamming"]["whitened"][-1], cf["hamming"]["greedy"][-1]))

    print("[3/4] real FX (curl-invisibility in the wild + rank obstruction) ...")
    fx = run_real_fx.run()
    print("      curl/grad=%.1e ; K%d: %d curl dims vs %d triangle coefficients"
          % (fx["curl_to_gradient_ratio"], len(fx["currencies"]),
             fx["curl_dimension"], fx["n_candidate_triangles"]))

    print("[4/4] real road-network traffic (favorable geometry + achievability) ...")
    tr = run_real_traffic.run()
    an = [r for r in tr["geometry"] if r["network"] == "Anaheim"][0]
    rec = tr["recovery"]
    print("      Anaheim DoF %d/%d=1.0 ; recovery@N=%d: emp=%.2f theory=%.2f"
          % (an["rank_B2"], an["n_triangles"], rec["N_grid"][-1],
             rec["empirical"][-1], rec["theory_product_exact"][-1]))

    print("done in %.1fs. figures in results/figures/, metrics in results/*.json"
          % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
