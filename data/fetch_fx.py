"""Refetch data/fx_rates.json from api.frankfurter.app (free, no API key).

The repository ships with the cached 2024 file so all experiments run offline;
run this script only if you want to regenerate or extend the dataset.
Frankfurter serves the ECB daily reference rates (no weekends/holidays).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

URL = (
    "https://api.frankfurter.app/2024-01-01..2024-12-31"
    "?base=USD&symbols=EUR,GBP,JPY,CHF,CAD,AUD,SEK,NOK"
)

if __name__ == "__main__":
    out = Path(__file__).with_name("fx_rates.json")
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = json.load(r)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"saved {out}  base={data['base']}  days={len(data['rates'])}")
