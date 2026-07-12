"""Guardrails for the wind-field -> edge-flow bridge (tfl.geo).

The synthetic ground truth is a Rankine-style vortex: the mesh curl statistic
must (1) light up exactly on the triangles containing the vortex core,
(2) agree in sign/location with the full-grid finite-difference vorticity, and
(3) the mesh itself must satisfy the Euler full-column-rank property that the
achievability theory relies on.
"""
from __future__ import annotations

import numpy as np
import pytest

from tfl.geo import (
    RegionalMesh,
    grid_vorticity,
    triangle_grid_points,
    triangle_mean_abs_vorticity,
    triangular_mesh,
    wind_edge_flows,
)
from tfl.hodge import build_incidences


def make_grid(n_lat: int = 25, n_lon: int = 41):
    lat = np.linspace(5.0, 29.0, n_lat)      # ascending
    lon = np.linspace(120.0, 160.0, n_lon)
    return lat, lon


def rankine_vortex(lat, lon, lat0: float, lon0: float, r_core_deg: float = 3.0,
                   v_max: float = 25.0):
    """Cyclonic (counter-clockwise, NH) vortex centered at (lat0, lon0)."""
    LON, LAT = np.meshgrid(lon, lat)
    dx = (LON - lon0) * np.cos(np.deg2rad(lat0))
    dy = LAT - lat0
    r = np.sqrt(dx**2 + dy**2) + 1e-9
    speed = np.where(r <= r_core_deg, v_max * r / r_core_deg,
                     v_max * r_core_deg / r)
    u = -speed * dy / r
    v = speed * dx / r
    return u[None], v[None]  # (1, n_lat, n_lon)


def test_mesh_full_column_rank_by_euler():
    lat, lon = make_grid()
    mesh = triangular_mesh(lat, lon, step=4)
    _, B2 = build_incidences(mesh.cx)
    p = B2.shape[1]
    assert p == mesh.cx.n_edges - mesh.cx.n_nodes + 1  # simply connected
    assert np.linalg.matrix_rank(B2) == p              # full column rank


def test_vortex_curl_localizes_at_core():
    lat, lon = make_grid()
    lat0, lon0 = 17.0, 140.0
    u, v = rankine_vortex(lat, lon, lat0, lon0)
    mesh = triangular_mesh(lat, lon, step=4)
    F = wind_edge_flows(mesh, u, v)          # (E, 1)

    _, B2 = build_incidences(mesh.cx)
    # orientation-corrected circulation (positive = cyclonic, NH)
    curl = mesh.tri_orientation * (B2.T @ F[:, 0])

    cen_lat = np.array([np.mean([mesh.node_lat[t] for t in tri])
                        for tri in mesh.cx.triangles])
    cen_lon = np.array([np.mean([mesh.node_lon[t] for t in tri])
                        for tri in mesh.cx.triangles])
    d = np.hypot(cen_lat - lat0, (cen_lon - lon0) * np.cos(np.deg2rad(lat0)))

    # strongest |curl| triangle sits within one mesh cell of the core
    tau_star = int(np.argmax(np.abs(curl)))
    assert d[tau_star] < 5.0
    # cyclonic vortex in the NH: positive circulation at the core
    assert curl[tau_star] > 0
    # circulation decays away from the core
    near = np.abs(curl)[d < 4.0].mean()
    far = np.abs(curl)[d > 12.0].mean()
    assert near > 5 * far


def test_mesh_curl_matches_grid_vorticity_ranking():
    lat, lon = make_grid()
    u, v = rankine_vortex(lat, lon, 17.0, 140.0)
    mesh = triangular_mesh(lat, lon, step=4)
    F = wind_edge_flows(mesh, u, v)
    _, B2 = build_incidences(mesh.cx)
    curl = np.abs(B2.T @ F[:, 0])

    zeta = grid_vorticity(u, v, lat, lon)
    tri_pts = triangle_grid_points(mesh, lat, lon)
    gt = triangle_mean_abs_vorticity(zeta, tri_pts)

    # detector statistic and full-grid vorticity agree on the ranking
    from scipy.stats import spearmanr
    rho, _ = spearmanr(curl, gt)
    assert rho > 0.6

    # and the vorticity magnitude at the core is physically sensible:
    # v_max/r_core = 25 m/s over ~3 deg (~333 km) -> zeta ~ 1.5e-4 1/s
    assert 1e-5 < np.abs(zeta).max() < 1e-2


def test_descending_latitude_grid_gives_same_vorticity_sign():
    lat, lon = make_grid()
    u, v = rankine_vortex(lat, lon, 17.0, 140.0)
    zeta = grid_vorticity(u, v, lat, lon)

    lat_desc = lat[::-1]
    u_desc, v_desc = u[:, ::-1, :], v[:, ::-1, :]
    zeta_desc = grid_vorticity(u_desc, v_desc, lat_desc, lon)
    assert np.allclose(zeta_desc[:, ::-1, :], zeta, atol=1e-12)
