"""Same-process provenance for released result JSON.

Round-9 requirement: every result file must carry provenance gathered by the
SAME process that produced it -- git SHA/dirty, hostname, command, timestamp,
Python/dependency/BLAS-thread environment, hardware (CPU + GPU if present),
wall time, peak RSS (and VRAM if CUDA). This forecloses "generate locally, then
staple a server probe on top" fakery: if a JSON claims a server hostname/GPU,
the same run measured it.

Usage: :func:`tfl.provenance.collect` returns the dict; ``experiments/_util``'s
``save_json`` embeds it automatically under the ``_provenance`` key. Wall time
is measured from the OS process-start time, so it is the whole-process runtime
(each experiment is its own process under ``run_all.py``).
"""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone


def _sh(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git() -> tuple[str, bool]:
    sha = _sh(["git", "rev-parse", "HEAD"]) or "(no git)"
    dirty = bool(_sh(["git", "status", "--porcelain"]))
    return sha, dirty


def _deps() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "matplotlib", "torch"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def _blas_threads() -> dict:
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


def _peak_rss_mb() -> float | None:
    try:
        import resource  # Linux/macOS
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KB on Linux, bytes on macOS
        return ru / 1024.0 if sys.platform != "darwin" else ru / 1e6
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except Exception:
            return None


def _proc_wall_s() -> float | None:
    try:
        import psutil
        return time.time() - psutil.Process().create_time()
    except Exception:
        return None


def _hardware() -> dict:
    hw = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for ln in f:
                if ln.lower().startswith("model name"):
                    hw["cpu_model"] = ln.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_mem_total_mb"] = \
                torch.cuda.get_device_properties(0).total_memory / 1e6
            hw["cuda"] = torch.version.cuda
            hw["driver"] = _sh(["nvidia-smi",
                                "--query-gpu=driver_version",
                                "--format=csv,noheader"]) or None
    except Exception:
        pass
    return hw


def peak_vram_mb() -> float | None:
    """Peak CUDA memory this process allocated (MB), or None if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return None


def collect(seed=None, extra: dict | None = None, wall_start=None) -> dict:
    """Provenance dict for the current process. ``status='completed_ok'`` is
    present only because this runs at save time -- a crashed run writes no JSON,
    so its absence is the failure signal. ``seed`` is the experiment's master
    seed if it has one (also usually recorded in the JSON's own ``config``).
    ``wall_start`` (a ``time.time()`` from experiment start) gives a psutil-free
    wall time; falls back to the OS process-start time."""
    sha, dirty = _git()
    wall = (time.time() - wall_start) if wall_start else _proc_wall_s()
    prov = {
        "git_sha": sha,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
        "seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _deps(),
        "blas_threads": _blas_threads(),
        "hardware": _hardware(),
        "wall_time_s": wall,
        "peak_rss_mb": _peak_rss_mb(),
        "peak_vram_mb": peak_vram_mb(),
        "exit_status": "completed_ok",
    }
    if extra:
        prov.update(extra)
    return prov
