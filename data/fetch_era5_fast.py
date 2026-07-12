"""Concurrent-chunk ERA5 fetcher (GFW-tolerant variant of fetch_era5.py).

Reads the zarr chunk index directly and fetches all needed chunk objects with
gcsfs's batched async `cat` (concurrent range requests), then decodes with
numcodecs and assembles the array — an order of magnitude faster than
sequential per-chunk reads on high-latency links. Output identical to
fetch_era5.py: data/era5_wnp_2020.npz.

Run:  python -u data/fetch_era5_fast.py
"""
from __future__ import annotations

import json
from pathlib import Path

import gcsfs
import numcodecs
import numpy as np

STORE = "gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr"
OUT = Path(__file__).parent / "era5_wnp_2020.npz"

T0, T1 = np.datetime64("2020-08-01T00"), np.datetime64("2020-09-30T18")
LAT_RANGE = (0.0, 45.0)
LON_RANGE = (100.0, 180.0)
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind"]


def read_coord(fs, meta, name: str) -> np.ndarray:
    arr = meta[f"{name}/.zarray"]
    comp = numcodecs.get_codec(arr["compressor"]) if arr["compressor"] else None
    n = arr["shape"][0]
    cs = arr["chunks"][0]
    out = np.empty(n, dtype=arr["dtype"])
    paths = [f"{STORE}/{name}/{i}" for i in range((n + cs - 1) // cs)]
    blobs = fs.cat(paths)
    for i, p in enumerate(paths):
        raw = blobs[p]
        buf = comp.decode(raw) if comp else raw
        chunk = np.frombuffer(buf, dtype=arr["dtype"])
        out[i * cs: i * cs + len(chunk)] = chunk[: n - i * cs]
    return out


def main() -> None:
    fs = gcsfs.GCSFileSystem(token="anon")
    print("reading consolidated metadata ...", flush=True)
    meta = json.loads(fs.cat(f"{STORE}/.zmetadata"))["metadata"]

    time_attrs = meta["time/.zattrs"]
    units = time_attrs["units"]          # e.g. "hours since 1900-01-01"
    assert units.startswith("hours since"), units
    epoch = np.datetime64(units.split("since")[1].strip()[:19].replace(" ", "T"))
    time_raw = read_coord(fs, meta, "time")
    times = epoch + time_raw.astype("timedelta64[h]")
    lat = read_coord(fs, meta, "latitude").astype(np.float64)
    lon = read_coord(fs, meta, "longitude").astype(np.float64)
    print(f"time[0]={times[0]}, time[-1]={times[-1]}, lat {lat.min():.2f}..{lat.max():.2f}, "
          f"lon {lon.min():.2f}..{lon.max():.2f}", flush=True)

    t_idx = np.nonzero((times >= T0) & (times <= T1))[0]
    lat_idx = np.nonzero((lat >= LAT_RANGE[0]) & (lat <= LAT_RANGE[1]))[0]
    lon_idx = np.nonzero((lon >= LON_RANGE[0]) & (lon <= LON_RANGE[1]))[0]
    print(f"selection: {len(t_idx)} times, {len(lat_idx)} lats, {len(lon_idx)} lons", flush=True)

    arr0 = meta[f"{VARS[0]}/.zarray"]
    tchunk = arr0["chunks"][0]
    assert arr0["chunks"][1] == arr0["shape"][1] and arr0["chunks"][2] == arr0["shape"][2], \
        "expected full-plane chunks"
    # NOTE: ARCO equiangular stores are (time, LONGITUDE, LATITUDE)
    nlon_full, nlat_full = arr0["shape"][1], arr0["shape"][2]
    assert nlon_full == len(lon) and nlat_full == len(lat)
    comp = numcodecs.get_codec(arr0["compressor"]) if arr0["compressor"] else None
    dtype = np.dtype(arr0["dtype"])

    c0, c1 = t_idx[0] // tchunk, t_idx[-1] // tchunk
    chunk_ids = list(range(c0, c1 + 1))
    print(f"fetching {len(chunk_ids)} chunks x {len(VARS)} vars "
          f"(~{len(chunk_ids)*len(VARS)*4.2:.0f} MB raw) ...", flush=True)

    def blosc_frame_ok(raw: bytes, expect_nbytes: int) -> bool:
        """Truncation check without decoding (a truncated blosc frame can
        segfault the C decoder): header bytes 4-8 = uncompressed size,
        12-16 = compressed size, which must equal the blob length."""
        return (len(raw) >= 16
                and int.from_bytes(raw[4:8], "little") == expect_nbytes
                and int.from_bytes(raw[12:16], "little") == len(raw))

    expect_nbytes = tchunk * nlat_full * nlon_full * dtype.itemsize
    out = {}
    for var in VARS:
        paths = [f"{STORE}/{var}/{ci}.0.0" for ci in chunk_ids]
        blobs = {}
        BATCH = 6
        for b in range(0, len(paths), BATCH):
            batch = paths[b: b + BATCH]
            for attempt in range(6):
                try:
                    got = fs.cat(batch)
                    bad = [p for p, raw in got.items()
                           if comp is not None and not blosc_frame_ok(raw, expect_nbytes)]
                    blobs.update({p: raw for p, raw in got.items() if p not in bad})
                    if not bad:
                        break
                    batch = bad  # refetch only the truncated ones
                    print(f"  batch {b//BATCH}: {len(bad)} truncated, refetching", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"  batch {b//BATCH} retry {attempt+1}: {type(e).__name__}", flush=True)
            else:
                raise RuntimeError(f"failed to fetch batch starting {batch[0]}")
            print(f"  {var}: {min(b + BATCH, len(paths))}/{len(paths)} chunks", flush=True)
        full = np.empty((len(chunk_ids) * tchunk, nlon_full, nlat_full), dtype=dtype)
        for i, ci in enumerate(chunk_ids):
            buf = comp.decode(blobs[paths[i]]) if comp else blobs[paths[i]]
            full[i * tchunk:(i + 1) * tchunk] = np.frombuffer(buf, dtype=dtype).reshape(
                tchunk, nlon_full, nlat_full)
        # slice (time, lon, lat), then transpose to (T, n_lat, n_lon)
        rel_t = t_idx - c0 * tchunk
        sliced = full[rel_t][:, lon_idx][:, :, lat_idx]
        out[var] = np.transpose(sliced, (0, 2, 1)).astype(np.float32)
        print(f"  {var}: sliced -> {out[var].shape} (T, n_lat, n_lon)", flush=True)

    np.savez_compressed(
        OUT,
        u=out[VARS[0]], v=out[VARS[1]],
        lat=lat[lat_idx], lon=lon[lon_idx], time=times[t_idx],
    )
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)", flush=True)
    u = out[VARS[0]]
    print(f"sanity: |u| mean {np.abs(u).mean():.2f} m/s, max {np.abs(u).max():.1f} m/s", flush=True)


if __name__ == "__main__":
    main()
