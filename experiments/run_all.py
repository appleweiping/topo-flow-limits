"""One-click reproduction of every figure and JSON in the paper + supplement.

Round-9 rewrite: layered, subprocess-based, provenance-aware.

  * CPU tier (default): every experiment behind the MAIN paper AND the
    supplement's CPU results -- theory grids, real-data studies, the BIC
    noise-identifiability boundary, the matrix-free scaling, the non-oracle
    selection. No GPU, no torch. Reproduces every figure in the paper.
  * GPU tier (``--gpu``): additionally the batched Monte-Carlo of supplement
    S6 (needs CUDA + torch; ``pip install -e '.[gpu]'``).

Each experiment runs as its OWN subprocess, so (a) its ``results/*.json`` gets a
clean same-process ``_provenance`` block, and (b) run_all records its exit code
and wall time in ``results/run_all_manifest.json``. A non-zero exit is reported,
not swallowed.

    python experiments/run_all.py            # CPU tier
    python experiments/run_all.py --gpu      # CPU + GPU tiers
    python experiments/run_all.py --only run_selection.py run_scaling.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
DATA = ROOT / "data"

# (script, tier, needs_data) in run order. tier in {"cpu","gpu"}.
EXPERIMENTS = [
    ("run_phase_transition.py", "cpu", False),
    ("run_confusability.py", "cpu", False),
    ("run_real_fx.py", "cpu", False),
    ("run_real_traffic.py", "cpu", False),
    ("run_fano.py", "cpu", False),
    ("run_partial_sampling.py", "cpu", False),
    ("run_plugin.py", "cpu", False),
    ("run_second_order.py", "cpu", False),
    ("run_bic_boundary.py", "cpu", False),        # round-9: BIC noise-id boundary
    ("run_scaling.py", "cpu", False),             # matrix-free scaling (CPU)
    ("run_selection.py", "cpu", False),           # non-oracle BIC selection
    ("run_real_cyclone.py", "cpu", True),         # ERA5 + IBTrACS (cached)
    ("run_gpu_mc.py", "gpu", False),              # GPU-only batched Monte-Carlo
    ("plot_scaling.py", "cpu", False),            # figure from the 3 JSONs above
]


def _data_present() -> bool:
    return (DATA / "era5_wnp_2020.npz").exists() and \
        (DATA / "ibtracs_wp_2020.csv").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also run the GPU tier")
    ap.add_argument("--only", nargs="*", help="run only these script names")
    args = ap.parse_args()

    t_all = time.perf_counter()
    records = []
    for script, tier, needs_data in EXPERIMENTS:
        if args.only and script not in args.only:
            continue
        if tier == "gpu" and not args.gpu and not (args.only and script in args.only):
            print(f"[skip] {script} (GPU tier; pass --gpu)")
            records.append({"script": script, "tier": tier, "status": "skipped_gpu"})
            continue
        if needs_data and not _data_present():
            print(f"[skip] {script} (cached ERA5/IBTrACS missing)")
            records.append({"script": script, "tier": tier, "status": "skipped_no_data"})
            continue
        print(f"[run ] {script} ({tier}) ...", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(EXP / script)], cwd=str(ROOT))
        dt = time.perf_counter() - t0
        rec = {"script": script, "tier": tier, "exit_code": proc.returncode,
               "wall_s": round(dt, 2),
               "status": "ok" if proc.returncode == 0 else "FAILED"}
        records.append(rec)
        print(f"       -> exit {proc.returncode} in {dt:.1f}s", flush=True)
        if proc.returncode != 0:
            print(f"       !! {script} FAILED (exit {proc.returncode})", flush=True)

    manifest = {
        "generated_by": "experiments/run_all.py",
        "total_wall_s": round(time.perf_counter() - t_all, 2),
        "gpu_tier_included": bool(args.gpu),
        "records": records,
        "n_failed": sum(1 for r in records if r.get("status") == "FAILED"),
    }
    (ROOT / "results" / "run_all_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    n_fail = manifest["n_failed"]
    print(f"\ndone in {manifest['total_wall_s']:.1f}s; {n_fail} failed. "
          f"run manifest -> results/run_all_manifest.json")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
