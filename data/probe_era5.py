"""Probe the ARCO-ERA5 public GCS store: list candidate zarr stores, inspect
chunking of the wind variables, and estimate the download cost of a basin/season
slice BEFORE committing to it. Falls back to reporting NCEP PSL OPeNDAP as the
alternative if chunking makes ERA5 impractical over this network.

Run:  python data/probe_era5.py
"""
from __future__ import annotations

import json
import sys

import gcsfs


def main() -> None:
    fs = gcsfs.GCSFileSystem(token="anon")
    root = "gcp-public-data-arco-era5"

    print("=== bucket top-level ===")
    for p in fs.ls(root):
        print(" ", p)

    print("\n=== ar/ stores ===")
    for p in fs.ls(f"{root}/ar"):
        print(" ", p)

    # The full 37-level analysis-ready store
    store = f"{root}/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
    meta = json.loads(fs.cat(f"{store}/.zmetadata"))["metadata"]
    for var in ["u_component_of_wind", "v_component_of_wind"]:
        arr = meta.get(f"{var}/.zarray")
        if arr:
            print(f"\n{var}: shape={arr['shape']} chunks={arr['chunks']} dtype={arr['dtype']}")
            # bytes per chunk (uncompressed estimate)
            import math
            per = math.prod(arr["chunks"]) * 4
            print(f"  uncompressed chunk ~ {per/1e6:.1f} MB")
    # coordinate metadata
    for coord in ["time", "level", "latitude", "longitude"]:
        arr = meta.get(f"{coord}/.zarray")
        if arr:
            print(f"{coord}: shape={arr['shape']} chunks={arr['chunks']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"PROBE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
