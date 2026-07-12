"""One-click reproduction of every figure and number in the paper + supplement.

Runs all experiments from fixed seeds and writes to ``results/`` (+ figures).
CPU-only; completes in roughly 15-20 minutes on a laptop. The real-cyclone
experiment [7/7] needs the cached ERA5/IBTrACS files (ship with the repo via
``data/fetch_era5.py`` / ``data/fetch_ibtracs.py``); it is skipped with a
notice if they are absent.
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
import run_fano
import run_partial_sampling
import run_plugin


def main() -> None:
    t0 = time.perf_counter()
    print("[1/7] phase transition (curl-invisibility, exact recovery vs theory) ...")
    pt = run_phase_transition.run()
    print("      p=%d active=%d; exact-theory rho* at N=%d -> %.3f"
          % (pt["n_tri"], pt["n_active"], pt["T_grid"][-1], pt["theory_exact"][-1]))

    print("[2/7] confusability (naive vs whitened on edge-sharing strip) ...")
    cf = run_confusability.run()
    print("      Hamming@N=%d: naive=%.2f whitened=%.2f greedy=%.2f"
          % (cf["T_grid"][-1], cf["hamming"]["naive"][-1],
             cf["hamming"]["whitened"][-1], cf["hamming"]["greedy"][-1]))

    print("[3/7] real FX (curl-invisibility in the wild + rank obstruction) ...")
    fx = run_real_fx.run()
    print("      curl/grad=%.1e ; K%d: %d curl dims vs %d triangle coefficients"
          % (fx["curl_to_gradient_ratio"], len(fx["currencies"]),
             fx["curl_dimension"], fx["n_candidate_triangles"]))

    print("[4/7] real road-network traffic (favorable geometry + achievability) ...")
    tr = run_real_traffic.run()
    an = [r for r in tr["geometry"] if r["network"] == "Anaheim"][0]
    rec = tr["recovery"]
    print("      Anaheim DoF %d/%d=1.0 ; recovery@N=%d: emp=%.2f theory=%.2f"
          % (an["rank_B2"], an["n_triangles"], rec["N_grid"][-1],
             rec["empirical"][-1], rec["theory_product_exact"][-1]))

    print("[5/7] Fano converse curves (supplement S1) ...")
    fa = run_fano.run()
    print("      floors at N=%d: chernoff=%.3f fano(p=1e4)=%.3f"
          % (fa["N_grid"][-1], fa["chernoff_floor"][-1],
             fa["cases"][1]["fano_gauss_floor"][-1]))

    print("[6/7] partial edge sampling + plug-in estimation (supplement S2-S3) ...")
    ps = run_partial_sampling.run()
    pl = run_plugin.run()
    gap = max(abs(a - b) for a, b in
              zip(pl["recovery_known"], pl["recovery_plugin"]))
    print("      sampling: P(exact) emp=%.2f theory=%.2f at q=%.2f ; "
          "plugin max gap=%.3f"
          % (ps["empirical"][-2], ps["theory_closed_form"][-2],
             ps["q_grid"][-2], gap))

    print("[7/7] real cyclone recovery (unplanted; ERA5 + IBTrACS) ...")
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if (data_dir / "era5_wnp_2020.npz").exists() and \
            (data_dir / "ibtracs_wp_2020.csv").exists():
        import run_real_cyclone
        run_real_cyclone.main()
    else:
        print("      SKIPPED: cached ERA5/IBTrACS files missing "
              "(run data/fetch_era5.py and data/fetch_ibtracs.py first)")

    print("done in %.1fs. figures in results/figures/, metrics in results/*.json"
          % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
