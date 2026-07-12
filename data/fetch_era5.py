"""Fetch ERA5 10m winds for the Western North Pacific typhoon season 2020
from the public ARCO-ERA5 archive (Google Cloud, anonymous access) and cache a
compact regional slice for the real-cyclone recovery experiment.

Store: gs://gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr
  * 6-hourly, 0.703° equiangular grid (512 x 256), 10m u/v wind components
  * chunked (8, 512, 256): a two-month season is ~25 chunks (~100 MB)

Output: data/era5_wnp_2020.npz with
  u, v        (T, n_lat, n_lon) float32   10m wind components [m/s]
  lat, lon    (n_lat,), (n_lon,) float64  grid coordinates [deg]
  time        (T,) '<M8[ns]'              snapshot timestamps (6-hourly)

Region: 0..45N, 100..180E (Western North Pacific typhoon basin)
Season: 2020-08-01 .. 2020-09-30 (Bavi, Maysak, Haishen, ... in IBTrACS)

Run:  python data/fetch_era5.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

STORE = "gs://gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr"
OUT = Path(__file__).parent / "era5_wnp_2020.npz"

TIME_SLICE = slice("2020-08-01", "2020-09-30")
LAT_SLICE = (0.0, 45.0)
LON_SLICE = (100.0, 180.0)


def main() -> None:
    print(f"opening {STORE} ...")
    ds = xr.open_zarr(STORE, storage_options={"token": "anon"}, consolidated=True)
    print(f"store dims: {dict(ds.sizes)}")

    u = ds["10m_u_component_of_wind"].sel(time=TIME_SLICE)
    v = ds["10m_v_component_of_wind"].sel(time=TIME_SLICE)

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    lat_mask = (lat >= LAT_SLICE[0]) & (lat <= LAT_SLICE[1])
    lon_mask = (lon >= LON_SLICE[0]) & (lon <= LON_SLICE[1])
    u = u.isel(latitude=lat_mask.nonzero()[0], longitude=lon_mask.nonzero()[0])
    v = v.isel(latitude=lat_mask.nonzero()[0], longitude=lon_mask.nonzero()[0])

    print(f"slice: time={u.sizes['time']} lat={u.sizes['latitude']} lon={u.sizes['longitude']}")
    print("downloading u ...")
    u_np = u.values.astype(np.float32)
    print("downloading v ...")
    v_np = v.values.astype(np.float32)

    np.savez_compressed(
        OUT,
        u=u_np,
        v=v_np,
        lat=u["latitude"].values.astype(np.float64),
        lon=u["longitude"].values.astype(np.float64),
        time=u["time"].values,
    )
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")
    print(f"sanity: |u| mean {np.abs(u_np).mean():.2f} m/s, max {np.abs(u_np).max():.1f} m/s")


if __name__ == "__main__":
    main()
