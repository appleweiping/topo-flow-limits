"""Assemble data/era5_wnp_2020.npz from pre-downloaded raw zarr chunk files.

Companion to fetch_era5_fast.py for very slow links: download the 62 chunk
objects with any parallel HTTP tool (curl -P / aria2), naming them
``<var>__<chunkid>.0.0`` in one directory, then run

    python -u data/assemble_era5.py <chunk_dir>

Only the tiny .zmetadata + coordinate arrays are fetched over the network here
(plain HTTPS via requests — no gcsfs/aiohttp, which proved crash-prone on
some Windows setups).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numcodecs
import numpy as np
import requests

STORE = "gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr"
HTTP = "https://storage.googleapis.com/" + STORE
OUT = Path(__file__).parent / "era5_wnp_2020.npz"

T0, T1 = np.datetime64("2020-08-01T00"), np.datetime64("2020-09-30T18")
LAT_RANGE = (0.0, 45.0)
LON_RANGE = (100.0, 180.0)
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind"]
C0, C1 = 11246, 11276  # chunk ids covering the season (8 timesteps per chunk)


def http_get(path: str, tries: int = 4) -> bytes:
    last: Exception | None = None
    for _ in range(tries):
        try:
            r = requests.get(f"{HTTP}/{path}", timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"failed to GET {path}: {last}")


def read_coord(meta, name: str) -> np.ndarray:
    arr = meta[f"{name}/.zarray"]
    comp = numcodecs.get_codec(arr["compressor"]) if arr["compressor"] else None
    n, cs = arr["shape"][0], arr["chunks"][0]
    out = np.empty(n, dtype=arr["dtype"])
    for i in range((n + cs - 1) // cs):
        raw = http_get(f"{name}/{i}")
        buf = comp.decode(raw) if comp else raw
        chunk = np.frombuffer(buf, dtype=arr["dtype"])
        out[i * cs: i * cs + len(chunk)] = chunk[: n - i * cs]
    return out


def main(chunk_dir: str) -> None:
    cdir = Path(chunk_dir)
    print("reading metadata + coords ...", flush=True)
    meta = json.loads(http_get(".zmetadata"))["metadata"]

    units = meta["time/.zattrs"]["units"]
    epoch = np.datetime64(units.split("since")[1].strip()[:19].replace(" ", "T"))
    times = epoch + read_coord(meta, "time").astype("timedelta64[h]")
    lat = read_coord(meta, "latitude").astype(np.float64)
    lon = read_coord(meta, "longitude").astype(np.float64)

    t_idx = np.nonzero((times >= T0) & (times <= T1))[0]
    lat_idx = np.nonzero((lat >= LAT_RANGE[0]) & (lat <= LAT_RANGE[1]))[0]
    lon_idx = np.nonzero((lon >= LON_RANGE[0]) & (lon <= LON_RANGE[1]))[0]
    assert t_idx[0] // 8 == C0 and t_idx[-1] // 8 == C1, (t_idx[0] // 8, t_idx[-1] // 8)

    arr0 = meta[f"{VARS[0]}/.zarray"]
    # NOTE the ARCO equiangular stores are laid out (time, LONGITUDE, LATITUDE)
    # — shape [T, 512, 256] with len(lon)=512, len(lat)=256.
    tchunk, nlon, nlat = arr0["chunks"][0], arr0["shape"][1], arr0["shape"][2]
    assert nlon == len(lon) and nlat == len(lat), (arr0["shape"], len(lon), len(lat))
    comp = numcodecs.get_codec(arr0["compressor"]) if arr0["compressor"] else None
    dtype = np.dtype(arr0["dtype"])

    def blosc_frame_ok(raw: bytes, expect_nbytes: int) -> bool:
        """Validate a blosc1 frame header WITHOUT decoding (a truncated frame
        can segfault the C decoder): bytes 4-8 = uncompressed size, bytes
        12-16 = compressed size, which must equal the file length."""
        if len(raw) < 16:
            return False
        nbytes = int.from_bytes(raw[4:8], "little")
        cbytes = int.from_bytes(raw[12:16], "little")
        return nbytes == expect_nbytes and cbytes == len(raw)

    expect = tchunk * nlat * nlon * dtype.itemsize
    bad = []
    for var in VARS:
        for ci in range(C0, C1 + 1):
            f = cdir / f"{var}__{ci}.0.0"
            if not f.exists() or (comp is not None
                                  and not blosc_frame_ok(f.read_bytes(), expect)):
                bad.append(f.name)
    if bad:
        raise RuntimeError(
            f"{len(bad)} chunk files are missing or truncated (re-download them "
            f"before assembling): {bad}")

    out = {}
    for var in VARS:
        full = np.empty(((C1 - C0 + 1) * tchunk, nlon, nlat), dtype=dtype)
        for ci in range(C0, C1 + 1):
            f = cdir / f"{var}__{ci}.0.0"
            raw = f.read_bytes()
            buf = comp.decode(raw) if comp else raw
            full[(ci - C0) * tchunk:(ci - C0 + 1) * tchunk] = \
                np.frombuffer(buf, dtype=dtype).reshape(tchunk, nlon, nlat)
        rel_t = t_idx - C0 * tchunk
        # slice (time, lon, lat) then transpose to the (T, n_lat, n_lon)
        # convention used by tfl.geo
        sliced = full[rel_t][:, lon_idx][:, :, lat_idx]
        out[var] = np.transpose(sliced, (0, 2, 1)).astype(np.float32)
        print(f"{var}: {out[var].shape} (T, n_lat, n_lon)", flush=True)

    np.savez_compressed(OUT, u=out[VARS[0]], v=out[VARS[1]],
                        lat=lat[lat_idx], lon=lon[lon_idx], time=times[t_idx])
    u = out[VARS[0]]
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB); "
          f"|u| mean {np.abs(u).mean():.2f} max {np.abs(u).max():.1f} m/s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/era5_chunks")
