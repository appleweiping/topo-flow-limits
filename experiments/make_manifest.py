"""Generate results/manifest.json: a reproducibility record binding every
released artifact to its commit, command, seed, data hash, environment, and
(local) hardware. Run from the repo root: ``python experiments/make_manifest.py``.

Honesty note: the ``hardware`` block records the machine THIS manifest was
generated on. If you regenerate the experiments elsewhere (e.g. a GPU server),
re-run this script there so the record matches; do not hand-edit it.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sh(args: list[str]) -> str:
    """Run a fixed command (no shell, list form) and return stdout, or ''."""
    try:
        return subprocess.check_output(args, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _sha256(path: Path, n: int = 16) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:n]


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "matplotlib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts")
        blas = (cfg.get("Build Dependencies", {}).get("blas", {}) or {})
        out["blas"] = {k: blas.get(k) for k in ("name", "version",
                                                "openblas configuration")}
    except Exception:
        out["blas"] = None
    return out


def _hardware() -> dict:
    return {
        "note": "LOCAL machine this manifest was generated on (not a server).",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "blas_threads_env": os.environ.get("OMP_NUM_THREADS")
        or os.environ.get("OPENBLAS_NUM_THREADS") or "(default)",
    }


# (script, seed, output artifacts) for each released experiment.
EXPERIMENTS = [
    ("experiments/run_second_order.py", 42,
     ["results/second_order.json", "results/figures/second_order.png"]),
    ("experiments/run_real_cyclone.py", 7,
     ["results/real_cyclone.json", "results/figures/real_cyclone.png"]),
    ("experiments/run_phase_transition.py", 0,
     ["results/phase_transition.json", "results/figures/phase_transition.png"]),
    ("experiments/run_confusability.py", 0,
     ["results/confusability.json", "results/figures/confusability.png"]),
    ("experiments/run_fano.py", 0,
     ["results/fano.json", "results/figures/fano_bounds.png"]),
    ("experiments/run_partial_sampling.py", 0,
     ["results/partial_sampling.json", "results/figures/partial_sampling.png"]),
    ("experiments/run_plugin.py", 0,
     ["results/plugin.json", "results/figures/plugin.png"]),
    ("experiments/run_real_traffic.py", 0,
     ["results/real_traffic.json", "results/figures/real_traffic.png"]),
    ("experiments/run_real_fx.py", 0,
     ["results/real_fx.json", "results/figures/real_fx.png"]),
]

DATA_FILES = [
    "data/fx_rates.json", "data/era5_wnp_2020.npz", "data/ibtracs_wp_2020.csv",
]


def main() -> None:
    os.chdir(ROOT)
    manifest = {
        "commit": _sh(["git", "rev-parse", "HEAD"]) or "(no git)",
        "commit_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "commit_note": "This is HEAD at generation time. When the manifest is "
        "committed it becomes the PARENT of the commit that carries it, so "
        "manifest.commit == HEAD~1 for the commit that adds/updates it. "
        "Regenerate on a clean tree so commit_dirty is false.",
        "generated_by": "experiments/make_manifest.py",
        "environment": _versions(),
        "hardware": _hardware(),
        "data": {f: {"sha256_16": _sha256(ROOT / f),
                     "bytes": (ROOT / f).stat().st_size if (ROOT / f).exists()
                     else None}
                 for f in DATA_FILES},
        "experiments": [
            {"script": s, "seed": seed,
             "command": f"python {s}",
             "artifacts": {a: {"sha256_16": _sha256(ROOT / a),
                               "bytes": (ROOT / a).stat().st_size
                               if (ROOT / a).exists() else None}
                           for a in arts}}
            for s, seed, arts in EXPERIMENTS
        ],
        "wall_time_and_peak_rss": "not captured here; run each script with "
        "/usr/bin/time -v (Linux) or a psutil wrapper to record per-experiment "
        "wall time and peak RSS on your machine.",
    }
    out = ROOT / "results" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out} (commit {manifest['commit'][:10]}, "
          f"dirty={manifest['commit_dirty']})")


if __name__ == "__main__":
    main()
