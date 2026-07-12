"""Fetch the IBTrACS v04r01 Western-Pacific best-track catalog (NOAA NCEI) and
cache the 2020-season subset used as EXTERNAL ground truth by the real-cyclone
vortex-localization experiment.

IBTrACS (International Best Track Archive for Climate Stewardship) is the
authoritative, agency-merged tropical-cyclone position/intensity record — an
entirely independent source from the ERA5 reanalysis winds, which is what makes
it a genuine external validation target.

Output: data/ibtracs_wp_2020.csv with columns
  sid, name, iso_time, lat, lon, wind_kt  (all storms with any fix in
  2020-08-01..2020-09-30, all agencies' positions merged by IBTrACS)

Run:  python data/fetch_ibtracs.py
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import requests

URL = ("https://www.ncei.noaa.gov/data/"
       "international-best-track-archive-for-climate-stewardship-ibtracs/"
       "v04r01/access/csv/ibtracs.WP.list.v04r01.csv")
OUT = Path(__file__).parent / "ibtracs_wp_2020.csv"

SEASON_PREFIXES = ("2020-08", "2020-09")


def fetch_with_retry(url: str, tries: int = 5, timeout: int = 300) -> bytes:
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            print(f"GET {url} (attempt {attempt}) ...")
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            chunks = []
            n = 0
            for chunk in r.iter_content(chunk_size=1 << 20):
                chunks.append(chunk)
                n += len(chunk)
                if n % (20 << 20) < (1 << 20):
                    print(f"  ... {n/1e6:.0f} MB")
            return b"".join(chunks)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  failed: {type(e).__name__}: {e}; retrying in 10s")
            time.sleep(10)
    raise RuntimeError(f"could not fetch {url}: {last}")


def main() -> None:
    raw = fetch_with_retry(URL)
    print(f"downloaded {len(raw)/1e6:.1f} MB")

    text = io.StringIO(raw.decode("utf-8", errors="replace"))
    reader = csv.reader(text)
    header = next(reader)
    next(reader)  # units row
    col = {name: i for i, name in enumerate(header)}
    keep = ["SID", "NAME", "ISO_TIME", "LAT", "LON", "USA_WIND"]

    rows = []
    for row in reader:
        iso = row[col["ISO_TIME"]]
        if iso[:7] in SEASON_PREFIXES:
            rows.append([row[col[k]] for k in keep])

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sid", "name", "iso_time", "lat", "lon", "wind_kt"])
        w.writerows(rows)
    storms = {r[0] for r in rows}
    print(f"saved {OUT}: {len(rows)} fixes, {len(storms)} storms in Aug-Sep 2020")
    print("storms:", sorted({r[1] for r in rows}))


if __name__ == "__main__":
    main()
