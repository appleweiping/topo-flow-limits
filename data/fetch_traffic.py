"""Refetch the vendored TNTP road-network files in data/traffic/.

Source: the Transportation Networks for Research repository
(github.com/bstabler/TransportationNetworks), the community-standard test
networks for traffic assignment. The files are small (~100 KB total) and are
vendored in this repo so all experiments run offline; run this script only to
regenerate them. Eastern-Massachusetts publishes no flow solution file.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master"
FILES = [
    ("SiouxFalls/SiouxFalls_net.tntp", "SiouxFalls_net.tntp"),
    ("SiouxFalls/SiouxFalls_flow.tntp", "SiouxFalls_flow.tntp"),
    ("Eastern-Massachusetts/EMA_net.tntp", "EMA_net.tntp"),
    ("Anaheim/Anaheim_net.tntp", "Anaheim_net.tntp"),
    ("Anaheim/Anaheim_flow.tntp", "Anaheim_flow.tntp"),
]

if __name__ == "__main__":
    out_dir = Path(__file__).with_name("traffic")
    out_dir.mkdir(exist_ok=True)
    for remote, local in FILES:
        target = out_dir / local
        with urllib.request.urlopen(f"{BASE}/{remote}", timeout=60) as r:
            target.write_bytes(r.read())
        print(f"saved {target} ({target.stat().st_size} bytes)")
