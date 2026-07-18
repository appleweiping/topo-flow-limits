"""Geophysical wind fields as edge flows on a triangulated regional mesh.

This is the bridge from a gridded vector field (ERA5 10m winds) to the
simplicial edge-flow observation model: mesh nodes subsample the grid, each
grid cell splits into two triangles, and the flow on an edge is the line
integral of the wind along it. By Stokes' theorem the curl of that edge flow
over a triangle approximates the CIRCULATION of the wind around it — i.e.
area-integrated relative vorticity. Detecting high-curl triangles from edge
flows is therefore detecting real atmospheric vortices (tropical cyclones),
validated against two references:

* INTERNAL: a SAME-FIELD consistency reference — relative vorticity computed
  from the full-resolution field by finite differences (a different
  functional of the same winds, at finer resolution than anything the mesh
  detector sees; NOT an independent measurement);
* EXTERNAL: the IBTrACS best-track archive — an external, separately-curated
  record of agency best-track cyclone positions (not derived from the
  reanalysis winds).

Geometry note: edges are measured in a local equirectangular metric
(dx = R cos(lat) dlon, dy = R dlat) — standard for regional meshes at these
latitudes; distances in km.

A triangulated simply-connected planar mesh has #triangles = |E| - |V| + 1
(Euler), which equals dim ker(B1) = rank(B2): B2 has FULL COLUMN RANK, the
curl degrees-of-freedom ratio is exactly 1, and no equal-image confusers
exist. Real recovery on such a mesh is the achievability side of the
excitation trichotomy — class-(a) favorable geometry with no equal-image
confusers (limits.py). The vortex-localization experiment exercises exactly
this regime.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tfl.hodge import Complex

EARTH_RADIUS_KM = 6371.0


@dataclass
class RegionalMesh:
    """A triangulated mesh over a lat-lon region.

    * ``cx``          : the simplicial 2-complex (all mesh triangles are candidates)
    * ``node_lat/lon``: node coordinates, degrees, shape ``(n_nodes,)``
    * ``lat_idx/lon_idx``: node positions in the SOURCE grid (for sampling fields)
    * ``edge_vec_km`` : per-edge displacement (dx_km, dy_km), orientation i -> j
    * ``tri_orientation``: ``+1`` if the sorted-vertex traversal ``i->j->k->i``
      is counter-clockwise in the (lon, lat) plane, else ``-1``. The curl
      statistic ``B2.T f`` follows the sorted-vertex traversal, so the
      GEOMETRIC circulation (positive = cyclonic in the northern hemisphere)
      is ``tri_orientation * (B2.T f)``. Energy detectors square the statistic
      and are orientation-free; the factor matters only for signed
      interpretation.
    """

    cx: Complex
    node_lat: np.ndarray
    node_lon: np.ndarray
    lat_idx: np.ndarray
    lon_idx: np.ndarray
    edge_vec_km: np.ndarray  # (n_edges, 2)
    tri_orientation: np.ndarray  # (n_triangles,) in {+1, -1}


def triangular_mesh(lat: np.ndarray, lon: np.ndarray, step: int) -> RegionalMesh:
    """Build a right-triangulated mesh from every ``step``-th grid point.

    Each grid cell [r, c] splits along its main diagonal into triangles
    ``(a, b, d)`` and ``(a, c', d)`` where ``a=(r,c), b=(r,c+1), c'=(r+1,c),
    d=(r+1,c+1)`` in row-major node numbering (so all vertex triples are
    already sorted, matching the hodge.py orientation conventions).
    """
    li = np.arange(0, len(lat), step)
    lj = np.arange(0, len(lon), step)
    H, W = len(li), len(lj)
    if H < 2 or W < 2:
        raise ValueError("mesh needs at least 2x2 nodes")

    node_lat = np.repeat(lat[li], W)
    node_lon = np.tile(lon[lj], H)
    lat_idx = np.repeat(li, W)
    lon_idx = np.tile(lj, H)

    def nid(r: int, c: int) -> int:
        return r * W + c

    edges: set[tuple[int, int]] = set()
    triangles: list[tuple[int, int, int]] = []
    for r in range(H - 1):
        for c in range(W - 1):
            a, b = nid(r, c), nid(r, c + 1)
            cc, d = nid(r + 1, c), nid(r + 1, c + 1)
            edges |= {(a, b), (a, cc), (a, d), (b, d), (cc, d)}
            triangles += [(a, b, d), (a, cc, d)]
    edges_sorted = sorted(edges)

    cx = Complex(n_nodes=H * W, edges=edges_sorted, triangles=triangles)

    ev = np.zeros((len(edges_sorted), 2))
    for e, (i, j) in enumerate(edges_sorted):
        mid_lat = 0.5 * (node_lat[i] + node_lat[j])
        dx = np.deg2rad(node_lon[j] - node_lon[i]) * EARTH_RADIUS_KM * np.cos(np.deg2rad(mid_lat))
        dy = np.deg2rad(node_lat[j] - node_lat[i]) * EARTH_RADIUS_KM
        ev[e] = (dx, dy)

    orient = np.empty(len(triangles))
    for t, (i, j, k) in enumerate(triangles):
        # z-component of (pj - pi) x (pk - pi) in the (lon, lat) plane
        ax, ay = node_lon[j] - node_lon[i], node_lat[j] - node_lat[i]
        bx, by = node_lon[k] - node_lon[i], node_lat[k] - node_lat[i]
        orient[t] = 1.0 if (ax * by - ay * bx) > 0 else -1.0
    return RegionalMesh(cx=cx, node_lat=node_lat, node_lon=node_lon,
                        lat_idx=lat_idx, lon_idx=lon_idx, edge_vec_km=ev,
                        tri_orientation=orient)


def triangle_areas_km2(mesh: RegionalMesh) -> np.ndarray:
    """Planar (equirectangular-metric) area of each mesh triangle in km^2.

    Needed to convert a triangle's CIRCULATION (what the curl statistic
    measures, units m/s * km) into a mean-VORTICITY scale (circulation/area):
    on a mesh spanning many latitudes the cell areas vary systematically with
    cos(lat), so ranking triangles by raw circulation energy conflates vortex
    strength with cell size.
    """
    areas = np.empty(len(mesh.cx.triangles))
    for t, (i, j, k) in enumerate(mesh.cx.triangles):
        lat0 = (mesh.node_lat[i] + mesh.node_lat[j] + mesh.node_lat[k]) / 3.0
        sc = np.cos(np.deg2rad(lat0)) * EARTH_RADIUS_KM * np.pi / 180.0
        sy = EARTH_RADIUS_KM * np.pi / 180.0
        ax = (mesh.node_lon[j] - mesh.node_lon[i]) * sc
        ay = (mesh.node_lat[j] - mesh.node_lat[i]) * sy
        bx = (mesh.node_lon[k] - mesh.node_lon[i]) * sc
        by = (mesh.node_lat[k] - mesh.node_lat[i]) * sy
        areas[t] = 0.5 * abs(ax * by - ay * bx)
    return areas


def wind_edge_flows(mesh: RegionalMesh, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Edge-flow snapshots from wind fields.

    ``u, v`` have shape ``(T, n_lat, n_lon)`` on the SOURCE grid; the flow on
    edge ``(i, j)`` at time ``t`` is the trapezoidal line integral
    ``0.5 (V_i + V_j) . (dx, dy)`` [m/s * km], oriented ``i -> j``.
    Returns ``F`` of shape ``(n_edges, T)``.
    """
    ui = u[:, mesh.lat_idx, mesh.lon_idx]  # (T, n_nodes)
    vi = v[:, mesh.lat_idx, mesh.lon_idx]
    E = len(mesh.cx.edges)
    T = u.shape[0]
    F = np.zeros((E, T))
    src = np.array([i for i, _ in mesh.cx.edges])
    dst = np.array([j for _, j in mesh.cx.edges])
    mean_u = 0.5 * (ui[:, src] + ui[:, dst])  # (T, E)
    mean_v = 0.5 * (vi[:, src] + vi[:, dst])
    F = (mean_u * mesh.edge_vec_km[:, 0] + mean_v * mesh.edge_vec_km[:, 1]).T
    return F


def grid_vorticity(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Relative vorticity ``zeta = dv/dx - du/dy`` [1/s] on the full grid by
    centered finite differences in the local equirectangular metric.
    ``u, v``: (T, n_lat, n_lon). Handles ascending or descending latitude."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dy = np.gradient(lat_r) * EARTH_RADIUS_KM * 1e3          # meters, per lat row
    dx = (np.gradient(lon_r)[None, :] * EARTH_RADIUS_KM * 1e3
          * np.cos(lat_r)[:, None])                           # meters, per point
    dv_dx = np.gradient(v, axis=2) / dx[None, :, :]
    du_dy = np.gradient(u, axis=1) / dy[None, :, None]
    return dv_dx - du_dy


def triangle_grid_points(mesh: RegionalMesh, lat: np.ndarray, lon: np.ndarray
                         ) -> list[np.ndarray]:
    """For each mesh triangle, the (flattened) indices of full-grid points
    inside it (barycentric test in the lon-lat plane). Used to average the
    full-resolution vorticity into per-triangle internal ground truth."""
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.stack([LON.ravel(), LAT.ravel()], axis=1)
    out: list[np.ndarray] = []
    for (i, j, k) in mesh.cx.triangles:
        p0 = np.array([mesh.node_lon[i], mesh.node_lat[i]])
        p1 = np.array([mesh.node_lon[j], mesh.node_lat[j]])
        p2 = np.array([mesh.node_lon[k], mesh.node_lat[k]])
        d = pts - p0
        e1, e2 = p1 - p0, p2 - p0
        den = e1[0] * e2[1] - e1[1] * e2[0]
        s = (d[:, 0] * e2[1] - d[:, 1] * e2[0]) / den
        t = (e1[0] * d[:, 1] - e1[1] * d[:, 0]) / den
        inside = (s >= -1e-9) & (t >= -1e-9) & (s + t <= 1 + 1e-9)
        out.append(np.nonzero(inside)[0])
    return out


def triangle_mean_abs_vorticity(zeta: np.ndarray, tri_pts: list[np.ndarray]) -> np.ndarray:
    """Per-triangle mean |vorticity| over a time block: internal ground truth.
    ``zeta``: (T, n_lat, n_lon). Returns shape ``(n_triangles,)``."""
    zt = np.abs(zeta).mean(axis=0).ravel()
    return np.array([zt[idx].mean() if len(idx) else 0.0 for idx in tri_pts])


# ---------------------------------------------------------------------------
# IBTrACS external ground truth
# ---------------------------------------------------------------------------

@dataclass
class StormFix:
    sid: str
    name: str
    iso_time: str
    lat: float
    lon: float
    wind_kt: float


def load_ibtracs(csv_path: str | Path) -> list[StormFix]:
    fixes: list[StormFix] = []
    with Path(csv_path).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                wind = float(row["wind_kt"]) if row["wind_kt"] not in ("", " ") else np.nan
                fixes.append(StormFix(
                    sid=row["sid"], name=row["name"], iso_time=row["iso_time"],
                    lat=float(row["lat"]), lon=float(row["lon"]), wind_kt=wind,
                ))
            except (KeyError, ValueError):
                continue
    return fixes


def cyclone_triangle_labels(
    mesh: RegionalMesh,
    fixes: list[StormFix],
    times: np.ndarray,
    window: slice,
    min_wind_kt: float = 34.0,
    radius_deg: float = 1.5,
) -> np.ndarray:
    """External per-triangle labels for one time window: True iff an IBTrACS
    fix (at tropical-storm strength or above) within the window's time range
    falls within ``radius_deg`` of the triangle centroid. Times are matched by
    ISO date-hour prefix (IBTrACS is 3/6-hourly, ERA5 6-hourly)."""
    t0 = np.datetime_as_string(times[window][0], unit="h")
    t1 = np.datetime_as_string(times[window][-1], unit="h")
    sel = [fx for fx in fixes
           if t0 <= fx.iso_time.replace(" ", "T")[:13] <= t1
           and (np.isnan(fx.wind_kt) or fx.wind_kt >= min_wind_kt)]

    cen_lat = np.array([np.mean([mesh.node_lat[v] for v in t]) for t in mesh.cx.triangles])
    cen_lon = np.array([np.mean([mesh.node_lon[v] for v in t]) for t in mesh.cx.triangles])
    labels = np.zeros(len(mesh.cx.triangles), dtype=bool)
    for fx in sel:
        d2 = (cen_lat - fx.lat) ** 2 + ((cen_lon - fx.lon) * np.cos(np.deg2rad(fx.lat))) ** 2
        labels |= d2 <= radius_deg**2
    return labels
