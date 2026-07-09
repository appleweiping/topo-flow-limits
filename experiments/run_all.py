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


def main() -> None:
    t0 = time.perf_counter()
    print("[1/3] phase transition (curl-invisibility, exact recovery vs theory) ...")
    pt = run_phase_transition.run()
    print("      p=%d active=%d; exact-theory rho* at T=%d -> %.3f"
          % (pt["n_tri"], pt["n_active"], pt["T_grid"][-1], pt["theory_exact"][-1]))

    print("[2/3] confusability (naive vs whitened on edge-sharing strip) ...")
    cf = run_confusability.run()
    print("      Hamming@T=%d: naive=%.2f whitened=%.2f greedy=%.2f"
          % (cf["T_grid"][-1], cf["hamming"]["naive"][-1],
             cf["hamming"]["whitened"][-1], cf["hamming"]["greedy"][-1]))

    print("[3/3] real FX (curl-invisibility in the wild + geometry limit) ...")
    fx = run_real_fx.run()
    print("      curl/grad=%.1e ; K%d: %d curl dims vs %d triangles"
          % (fx["curl_to_gradient_ratio"], len(fx["currencies"]),
             fx["curl_dimension"], fx["n_candidate_triangles"]))

    print("done in %.1fs. figures in results/figures/, metrics in results/*.json"
          % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
