"""CURL-BASED VORTEX LOCALIZATION on real data: tropical cyclones from ERA5
edge flows.

Honest framing (this is NOT "filled-triangle topology recovery"): the mesh is
full column rank with no equal-image confusers, and real weather has no
latent boolean triangle support — what this experiment demonstrates is that
the paper's curl statistic LOCALIZES genuine, unplanted rotational structure
(cyclone circulation) on a real vector field, validated against two
references:

  INTERNAL  relative vorticity by finite differences on the FULL SOURCE GRID
            (0.7 deg; the mesh detector only sees 2.1-deg edge flows). Honest
            scope: this is a SAME-FIELD consistency reference computed from
            the same u,v arrays by a different functional at 3x finer
            resolution -- not an independent measurement;
  EXTERNAL  IBTrACS best-track cyclone positions from an external,
            separately-curated archive (agency best-tracks, not derived from
            the reanalysis winds) -> ROC / precision.

BASELINE  a classical non-simplicial comparator at the same information
          budget: pointwise finite-difference vorticity computed from the
          winds subsampled AT THE MESH NODES (2.1 deg), averaged per
          triangle. The edge-flow detector should not lose to it; the
          contribution here is the limits framework, not a new TC detector.

Pipeline per 4-day window (16 six-hourly snapshots):
  wind field -> mesh edge flows F -> temporally centered curl-ENERGY scores
  (the paper's primary energy statistic, cf. main Prop. 2), area^2-normalized so scores rank
  triangles on a common mean-VORTICITY scale across latitudes
  -> compare to both references. No oracle parameters anywhere.

Statistic choice, stated honestly: the GLS/BLUE-decorrelated score (main Prop. 1)
targets exact-support recovery under edge-sharing leakage at a known SNR; on
this nearly-regular full-rank mesh its G^{-1} noise amplification hurts pure
DETECTION ranking (AUC 0.77 vs 0.91 internal). Both scores are computed and
reported in the JSON; the energy score is the headline panel.

A third panel degrades the snapshot budget N inside each window (mean +/- sd
over 12 subsample draws; uniformly centered statistic, N >= 3). No theory
floor is overlaid (its units are incommensurate with AUC); evaluation is
threshold-free ROC ranking throughout.

Outputs: results/real_cyclone.json + results/figures/real_cyclone.png
Run:     python experiments/run_real_cyclone.py            (~2-4 min CPU)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _util import save_json, savefig  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from tfl.estimators import whitened_curl_scores  # noqa: E402
from tfl.generative import FlowDataset, FlowParams  # noqa: E402
from tfl.geo import (  # noqa: E402
    cyclone_triangle_labels,
    grid_vorticity,
    load_ibtracs,
    triangle_areas_km2,
    triangle_grid_points,
    triangle_mean_abs_vorticity,
    triangular_mesh,
    wind_edge_flows,
)
from tfl.hodge import build_incidences  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
MESH_STEP = 3          # 3 x 0.703 deg ~ 2.1 deg mesh (~230 km edges)
WINDOW_LEN = 16        # 16 x 6h = 4 days
VORT_THRESH = 3e-5     # 1/s; typhoon-scale area-mean |vorticity|
VORT_MIN_WIND_KT = 34.0  # tropical-storm strength cutoff for external labels
# Headline label policy. STRICT (finite wind >= 34 kt only) is the conservative
# default; INCLUSIVE (missing-wind fixes also count) is reported alongside. The
# ~1/3 of 2020 WNP fixes with no reported intensity are genuine best-track
# positions, so neither policy is "correct" -- both are released.
PRIMARY_POLICY = "strict"


def dataset_from_flows(F: np.ndarray, B1, B2, cx) -> FlowDataset:
    """Wrap raw edge-flow snapshots in the FlowDataset container (the real
    data has no known generative parameters; placeholders are unused by the
    plug-in detector)."""
    return FlowDataset(
        F=F, B1=B1, B2_all=B2, active=np.zeros(B2.shape[1], bool),
        params=FlowParams(sigma_curl=0.0), candidate_triangles=list(cx.triangles),
    )


def roc_curve(scores: np.ndarray, labels: np.ndarray, n_thresh: int = 200):
    """Pooled ROC + AUC. Vectorized (cumsum over score-sorted labels); returns
    exactly the same step-function AUC as the per-point loop, ~100x faster --
    which is what makes the moving-block / storm bootstraps affordable."""
    order = np.argsort(-scores, kind="mergesort")   # stable: deterministic ties
    l = labels[order].astype(float)
    P = float(l.sum())
    Nn = float(len(labels)) - P
    tp = np.concatenate([[0.0], np.cumsum(l)])
    fp = np.concatenate([[0.0], np.cumsum(1.0 - l)])
    tpr = tp / max(P, 1.0)
    fpr = fp / max(Nn, 1.0)
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (step-wise PR-AUC): mean over positives, in score
    order, of precision-at-that-rank. Vectorized; identical to the loop form."""
    order = np.argsort(-scores, kind="mergesort")
    l = labels[order].astype(float)
    P = float(l.sum())
    if P == 0:
        return 0.0
    ranks = np.arange(1, len(l) + 1, dtype=float)
    prec = np.cumsum(l) / ranks
    return float(np.sum(prec * l) / P)


def cluster_bootstrap_ci(
    per_window_scores: list[np.ndarray],
    per_window_labels: list[np.ndarray],
    metric,
    n_boot: int = 1000,
    seed: int = 7,
    block_len: int = 3,
) -> tuple[float, float]:
    """95% percentile CI for a pooled ranking metric via a MOVING-BLOCK
    bootstrap over the time-ordered windows. A storm persists across several
    consecutive 4-day windows, so windows are serially dependent, not
    exchangeable; resampling contiguous blocks of ``block_len`` windows
    (circular, default 3 ~ storm residence time) preserves that within-storm
    correlation. (A plain window bootstrap, which assumes exchangeable
    windows, would understate the variance; we make no conservatism claim.)"""
    rng = np.random.default_rng(seed)
    W = len(per_window_scores)
    n_blocks = int(np.ceil(W / block_len))
    vals = []
    for _ in range(n_boot):
        starts = rng.integers(0, W, size=n_blocks)
        idx = np.concatenate(
            [(np.arange(s, s + block_len) % W) for s in starts])[:W]
        s = np.concatenate([per_window_scores[i] for i in idx])
        l = np.concatenate([per_window_labels[i] for i in idx])
        if l.sum() == 0 or l.sum() == len(l):
            continue
        vals.append(metric(s, l))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def storm_coverage(mesh, fixes, times, windows,
                   min_wind_kt: float = 34.0, radius_deg: float = 1.5,
                   include_missing_wind: bool = True) -> dict:
    """Flattened (window, triangle) positive mask for each storm, keyed by
    IBTrACS ``sid``. A storm covers a cell iff one of ITS OWN qualifying fixes
    (within the window, under the same wind policy) lies within ``radius_deg``
    of the centroid. Only storms that cover at least one cell are returned."""
    cover: dict[str, np.ndarray] = {}
    for sid in sorted({fx.sid for fx in fixes}):
        fx_s = [fx for fx in fixes if fx.sid == sid]
        m = np.concatenate([
            cyclone_triangle_labels(mesh, fx_s, times, w, min_wind_kt,
                                    radius_deg, include_missing_wind)
            for w in windows])
        if m.any():
            cover[sid] = m
    return cover


def storm_cluster_bootstrap_ci(scores_flat: np.ndarray, cover: dict, metric,
                               n_boot: int = 1000, seed: int = 11):
    """95% percentile CI for a pooled ranking metric under a STORM-cluster
    bootstrap: the resampling unit is the whole storm (its full set of positive
    cells), not the time window. Background negatives (cells no storm covers)
    are held fixed; positives are the union of the resampled storms' cells, and
    cells covered only by non-drawn storms are excluded from the evaluation set
    (so they never contaminate the negatives). This is the storm-level analogue
    of the moving-block CI and is reported as EXPLORATORY (single season, no
    multi-year validation)."""
    rng = np.random.default_rng(seed)
    sids = list(cover.keys())
    S = len(sids)
    any_pos = np.zeros_like(scores_flat, dtype=bool)
    for m in cover.values():
        any_pos |= m
    bg_neg = ~any_pos
    vals = []
    for _ in range(n_boot):
        drawn = rng.integers(0, S, size=S)
        pos = np.zeros_like(any_pos)
        for j in np.unique(drawn):
            pos |= cover[sids[j]]
        sel = pos | bg_neg
        lab = pos[sel].astype(float)
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        vals.append(metric(scores_flat[sel], lab))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    era = np.load(DATA / "era5_wnp_2020.npz")
    u, v = era["u"], era["v"]
    lat, lon, times = era["lat"], era["lon"], era["time"]
    print(f"ERA5 slice: {u.shape} lat[{lat.min():.1f},{lat.max():.1f}] "
          f"lon[{lon.min():.1f},{lon.max():.1f}]")

    mesh = triangular_mesh(lat, lon, MESH_STEP)
    B1, B2 = build_incidences(mesh.cx)
    p = B2.shape[1]
    rank = int(np.linalg.matrix_rank(B2))
    print(f"mesh: {mesh.cx.n_nodes} nodes, {mesh.cx.n_edges} edges, {p} triangles, "
          f"rank(B2)={rank} (full={rank == p})")

    fixes = load_ibtracs(DATA / "ibtracs_wp_2020.csv")
    print(f"IBTrACS: {len(fixes)} fixes")

    tri_pts = triangle_grid_points(mesh, lat, lon)
    F_all = wind_edge_flows(mesh, u, v)
    # normalize flows to unit RMS so plug-in scales are well-conditioned
    F_all = F_all / np.sqrt(np.mean(F_all**2))
    # circulation -> mean-vorticity scale: the curl statistic measures
    # circulation ~ zeta * area, and cell areas vary with cos(lat) across the
    # 0-45N mesh, so scores are normalized by area^2 (scores are energies)
    # to rank triangles on a common vorticity scale.
    areas = triangle_areas_km2(mesh)
    area_norm = (areas / areas.mean()) ** 2

    T = u.shape[0]
    windows = [slice(s, s + WINDOW_LEN) for s in range(0, T - WINDOW_LEN + 1, WINDOW_LEN)]
    print(f"{len(windows)} windows of {WINDOW_LEN} snapshots")

    all_scores, all_int_gt = [], []
    all_ext = {"strict": [], "inclusive": []}
    per_window = []
    # classical baseline inputs: winds subsampled at the mesh nodes (2.1 deg)
    lat_c = lat[np.unique(mesh.lat_idx)]
    lon_c = lon[np.unique(mesh.lon_idx)]
    u_c = u[:, np.unique(mesh.lat_idx)][:, :, np.unique(mesh.lon_idx)]
    v_c = v[:, np.unique(mesh.lat_idx)][:, :, np.unique(mesh.lon_idx)]
    coarse_mesh = triangular_mesh(lat_c, lon_c, 1)
    assert len(coarse_mesh.cx.triangles) == len(mesh.cx.triangles)
    coarse_pts = triangle_grid_points(coarse_mesh, lat_c, lon_c)

    all_scores_wh, all_scores_base = [], []
    for w in windows:
        F = F_all[:, w]
        ds = dataset_from_flows(F, B1, B2, mesh.cx)
        C = B2.T @ F
        C = C - C.mean(axis=1, keepdims=True)      # centered curl statistic
        scores = np.mean(C**2, axis=1) / area_norm  # energy on vorticity scale
        scores_wh = whitened_curl_scores(ds, center=True) / area_norm
        # baseline: coarse pointwise FD vorticity from mesh-node winds
        zeta_c = grid_vorticity(u_c[w], v_c[w], lat_c, lon_c)
        scores_base = triangle_mean_abs_vorticity(zeta_c, coarse_pts)

        zeta = grid_vorticity(u[w], v[w], lat, lon)
        gt_vort = triangle_mean_abs_vorticity(zeta, tri_pts)
        ext_incl = cyclone_triangle_labels(mesh, fixes, times, w,
                                           include_missing_wind=True)
        ext_strict = cyclone_triangle_labels(mesh, fixes, times, w,
                                             include_missing_wind=False)

        all_scores.append(scores)
        all_scores_wh.append(scores_wh)
        all_scores_base.append(scores_base)
        all_int_gt.append(gt_vort)
        all_ext["inclusive"].append(ext_incl)
        all_ext["strict"].append(ext_strict)
        per_window.append({
            "t0": np.datetime_as_string(times[w][0], unit="h"),
            "n_ext_pos_strict": int(ext_strict.sum()),
            "n_ext_pos_inclusive": int(ext_incl.sum()),
        })

    scores_flat = np.concatenate(all_scores)
    scores_wh_flat = np.concatenate(all_scores_wh)
    scores_base_flat = np.concatenate(all_scores_base)
    int_gt_flat = np.concatenate(all_int_gt)

    # ---- internal validation: ROC against full-res vorticity (policy-free) ----
    int_labels = int_gt_flat > VORT_THRESH
    fpr_i, tpr_i, auc_i = roc_curve(scores_flat, int_labels)
    print(f"INTERNAL: {int(int_labels.sum())}/{len(int_labels)} vorticity-positive "
          f"triangle-windows; AUC = {auc_i:.3f}")
    int_labels_pw = [g > VORT_THRESH for g in all_int_gt]
    boot_int = cluster_bootstrap_ci(all_scores, int_labels_pw,
                                    lambda s, l: roc_curve(s, l)[2])
    from scipy.stats import spearmanr
    rho_s, _ = spearmanr(scores_flat, int_gt_flat)
    _, _, auc_i_wh = roc_curve(scores_wh_flat, int_labels)
    rho_s_wh, _ = spearmanr(scores_wh_flat, int_gt_flat)
    fpr_b, tpr_b, auc_i_base = roc_curve(scores_base_flat, int_labels)
    thresh_sweep = {}
    for th in (1e-5, 2e-5, 3e-5, 4e-5, 5e-5):
        _, _, a = roc_curve(scores_flat, int_gt_flat > th)
        thresh_sweep[f"{th:.0e}"] = float(a)

    # ---- wind-label bookkeeping (raw / finite / missing / >=34 / eligible) ----
    finite = [f for f in fixes if not np.isnan(f.wind_kt)]
    missing = [f for f in fixes if np.isnan(f.wind_kt)]
    ge34_finite = [f for f in finite if f.wind_kt >= VORT_MIN_WIND_KT]
    def _sids(fs):
        return len({f.sid for f in fs})
    wind_counts = {
        "raw_fixes": len(fixes), "raw_storms": _sids(fixes),
        "finite_wind_fixes": len(finite), "finite_wind_storms": _sids(finite),
        "missing_wind_fixes": len(missing), "missing_wind_storms": _sids(missing),
        "finite_ge34_fixes": len(ge34_finite), "finite_ge34_storms": _sids(ge34_finite),
        "inclusive_qualifying_fixes": len(ge34_finite) + len(missing),
        "inclusive_qualifying_storms": _sids(ge34_finite + missing),
        "note": "STRICT labels use only finite wind >=34 kt; INCLUSIVE also "
                "counts missing-wind fixes (genuine best-track positions with no "
                "logged intensity, ~1/3 of the 2020 WNP fixes).",
    }
    print(f"wind counts: raw {len(fixes)}f/{_sids(fixes)}s | finite {len(finite)} | "
          f"missing {len(missing)} | finite>=34 {len(ge34_finite)} | "
          f"inclusive {len(ge34_finite)+len(missing)}")

    def external_suite(incl: bool) -> dict:
        """Full external-metric suite under one wind policy (incl=inclusive)."""
        ext_pw = all_ext["inclusive" if incl else "strict"]
        ext_flat = np.concatenate(ext_pw)
        _, _, auc_e = roc_curve(scores_flat, ext_flat)
        ap_e = pr_auc(scores_flat, ext_flat)
        k = int(ext_flat.sum())
        precision = int(ext_flat[np.argsort(-scores_flat)[:k]].sum()) / max(k, 1)
        _, _, auc_e_wh = roc_curve(scores_wh_flat, ext_flat)
        _, _, auc_e_base = roc_curve(scores_base_flat, ext_flat)
        ap_e_base = pr_auc(scores_base_flat, ext_flat)
        prec_base = int(ext_flat[np.argsort(-scores_base_flat)[:k]].sum()) / max(k, 1)
        b_auc = cluster_bootstrap_ci(all_scores, ext_pw, lambda s, l: roc_curve(s, l)[2])
        b_pr = cluster_bootstrap_ci(all_scores, ext_pw, pr_auc)
        b_base = cluster_bootstrap_ci(all_scores_base, ext_pw,
                                      lambda s, l: roc_curve(s, l)[2])

        def pooled(min_wind_kt, radius_deg):
            lab = np.concatenate([
                cyclone_triangle_labels(mesh, fixes, times, w, min_wind_kt,
                                        radius_deg, incl) for w in windows])
            _, _, a = roc_curve(scores_flat, lab)
            return {"auc_ext": float(a), "pr_ext": float(pr_auc(scores_flat, lab)),
                    "n_pos": int(lab.sum())}
        radius_sweep = {f"{r:.1f}deg": pooled(VORT_MIN_WIND_KT, r)
                        for r in (1.0, 1.5, 2.0, 2.5)}
        wind_sweep = {f"{int(wk)}kt": pooled(wk, 1.5)
                      for wk in (0.0, 34.0, 50.0, 64.0)}
        window_len_sweep = {}
        for wl in (8, 16, 24):
            wins = [slice(s, s + wl) for s in range(0, T - wl + 1, wl)]
            sc, ex = [], []
            for w_ in wins:
                Cc = B2.T @ F_all[:, w_]; Cc = Cc - Cc.mean(axis=1, keepdims=True)
                sc.append(np.mean(Cc**2, axis=1) / area_norm)
                ex.append(cyclone_triangle_labels(mesh, fixes, times, w_,
                                                  VORT_MIN_WIND_KT, 1.5, incl))
            sc_f, ex_f = np.concatenate(sc), np.concatenate(ex)
            _, _, a = roc_curve(sc_f, ex_f)
            window_len_sweep[f"{wl}x6h"] = {"auc_ext": float(a),
                                            "pr_ext": float(pr_auc(sc_f, ex_f)),
                                            "n_windows": len(wins)}
        block_len_sweep = {}
        for bl in (1, 2, 3, 5):
            block_len_sweep[f"block{bl}"] = list(cluster_bootstrap_ci(
                all_scores, ext_pw, lambda s, l: roc_curve(s, l)[2], block_len=bl))
        cover = storm_coverage(mesh, fixes, times, windows,
                               include_missing_wind=incl)
        storm_boot = {
            "auc_external_ci95": list(storm_cluster_bootstrap_ci(
                scores_flat, cover, lambda s, l: roc_curve(s, l)[2])),
            "pr_auc_external_ci95": list(storm_cluster_bootstrap_ci(
                scores_flat, cover, pr_auc)),
            "n_storms_resampled": len(cover),
        }
        # monotonicity is a numerical property -- test it, don't assert it
        def _mono(seq):
            d = np.diff(seq)
            return bool(np.all(d <= 1e-9) or np.all(d >= -1e-9))
        rad_vals = [radius_sweep[k]["auc_ext"] for k in radius_sweep]
        win_vals = [window_len_sweep[k]["auc_ext"] for k in window_len_sweep]
        wind_vals = [wind_sweep[k]["auc_ext"] for k in wind_sweep]
        return {
            "auc": float(auc_e), "auc_ci95_block_bootstrap": list(b_auc),
            "pr_auc": float(ap_e), "pr_auc_ci95_block_bootstrap": list(b_pr),
            "n_cyclone_triangle_windows": k, "precision_at_k": float(precision),
            "prevalence": k / int(len(ext_flat)),
            "decorrelated_auc_external": float(auc_e_wh),
            "baseline_auc_external": float(auc_e_base),
            "baseline_pr_auc_external": float(ap_e_base),
            "baseline_precision_at_k": float(prec_base),
            "baseline_auc_ci95_block_bootstrap": list(b_base),
            "sensitivity": {
                "external_vs_localization_radius": radius_sweep,
                "external_vs_wind_cutoff_kt": wind_sweep,
                "external_vs_window_length": window_len_sweep,
                "auc_external_ci_vs_bootstrap_block_len": block_len_sweep,
                "radius_auc_monotone_in_radius": _mono(rad_vals),
                "window_len_auc_monotone_in_length": _mono(win_vals),
                "wind_cutoff_auc_monotone": _mono(wind_vals),
                "auc_range_over_all_sweeps": [
                    float(min(rad_vals + win_vals + wind_vals)),
                    float(max(rad_vals + win_vals + wind_vals))],
            },
            "storm_cluster_bootstrap_exploratory": storm_boot,
        }

    external = {"strict": external_suite(False),
                "inclusive": external_suite(True)}
    for pol in ("strict", "inclusive"):
        e = external[pol]
        print(f"EXTERNAL[{pol}]: {e['n_cyclone_triangle_windows']} pos; "
              f"AUC={e['auc']:.3f} {e['auc_ci95_block_bootstrap']}; "
              f"PR={e['pr_auc']:.3f}; P@k={e['precision_at_k']:.3f}; "
              f"baseline AUC={e['baseline_auc_external']:.3f}")
    prim = external[PRIMARY_POLICY]
    all_ext_gt = all_ext[PRIMARY_POLICY]     # primary policy drives the figure
    ext_gt_flat = np.concatenate(all_ext_gt)
    fpr_e, tpr_e, _ = roc_curve(scores_flat, ext_gt_flat)
    fpr_be, tpr_be, _ = roc_curve(scores_base_flat, ext_gt_flat)
    auc_e, ap_e = prim["auc"], prim["pr_auc"]
    auc_e_base, ap_e_base = prim["baseline_auc_external"], prim["baseline_pr_auc_external"]

    # ---- degradation: snapshot budget vs detection quality (ALL windows) ----
    # For each (noise, N): subsample N snapshots in EVERY window, score vs that
    # window's internal GT, average over reps; the plotted band is the mean and
    # sd ACROSS windows (window-to-window spread), not a single hand-picked
    # window. (wi below is used only for the illustrative Panel-A score map.)
    wi = int(np.argmax([pw[f"n_ext_pos_{PRIMARY_POLICY}"] for pw in per_window]))
    win_gt = [g > VORT_THRESH for g in all_int_gt]
    rng = np.random.default_rng(0)
    noise_levels = [0.0, 0.5, 1.0, 2.0]
    Ns = [3, 4, 6, 8, 12, 16]  # N >= 3 so the CENTERED statistic is used
                               # uniformly (mixing statistics across N made
                               # the curve non-monotone and uninterpretable)
    reps = 6
    degradation = {}
    deg_std = {}
    for nl in noise_levels:
        means, sds = [], []
        for N in Ns:
            per_win = []
            for wi_, w_ in enumerate(windows):
                gtw = win_gt[wi_]
                if gtw.sum() == 0 or gtw.sum() == len(gtw):
                    continue  # no internal positives in this window -> no ROC
                F_w = F_all[:, w_]
                frms = float(np.sqrt(np.mean(F_w**2)))
                vv = []
                for _ in range(reps):
                    idx = rng.choice(WINDOW_LEN, size=N, replace=False)
                    F = F_w[:, idx] + nl * frms * rng.standard_normal(
                        (F_w.shape[0], N))
                    C = B2.T @ F
                    C = C - C.mean(axis=1, keepdims=True)
                    s = np.mean(C**2, axis=1) / area_norm
                    vv.append(roc_curve(s, gtw)[2])
                per_win.append(float(np.mean(vv)))
            means.append(float(np.mean(per_win)))
            sds.append(float(np.std(per_win)))
        degradation[str(nl)] = means
        deg_std[str(nl)] = sds

    # ---- figure ----
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 12,
                         "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig = plt.figure(figsize=(13.5, 4.0))

    ax = fig.add_subplot(1, 3, 1)
    wi_show = wi
    sc = all_scores[wi_show]
    cen_lat = np.array([np.mean([mesh.node_lat[t] for t in tri]) for tri in mesh.cx.triangles])
    cen_lon = np.array([np.mean([mesh.node_lon[t] for t in tri]) for tri in mesh.cx.triangles])
    im = ax.scatter(cen_lon, cen_lat, c=np.log10(sc + 1e-12), s=12, cmap="viridis")
    fx_w = all_ext_gt[wi_show]
    # red circles are TRIANGLE CENTROIDS labeled positive by IBTrACS (a fix
    # within 1.5 deg), NOT the raw IBTrACS fix lat/lon.
    ax.scatter(cen_lon[fx_w], cen_lat[fx_w], facecolors="none", edgecolors="red",
               s=60, linewidths=1.2, label="IBTrACS-positive triangle centroids")
    # DISCLOSED: this panel shows the single most active window
    # (argmax of n_ext_pos) for illustration; ALL metrics pool every window.
    ax.set_title(f"(A) curl-energy score, {per_window[wi_show]['t0']}\n"
                 f"(most-active window, illustrative)")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10 score")

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(fpr_i, tpr_i,
            label=f"ours vs internal labels, AUC={auc_i:.3f}")
    ax.plot(fpr_e, tpr_e,
            label=f"ours vs external (IBTrACS), AUC={auc_e:.3f}, PR={ap_e:.3f}")
    ax.plot(fpr_be, tpr_be, ":", color="gray",
            label=f"baseline vs external, AUC={auc_e_base:.3f}, PR={ap_e_base:.3f}")
    ax.plot(fpr_b, tpr_b, ":", color="lightgray",
            label=f"baseline vs internal, AUC={auc_i_base:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("(B) ROC vs both references")
    ax.legend(loc="lower right", fontsize=7)

    ax = fig.add_subplot(1, 3, 3)
    for nl in noise_levels:
        y = np.array(degradation[str(nl)])
        e = np.array(deg_std[str(nl)])
        ax.errorbar(Ns, y, yerr=e, fmt="o-", ms=3, capsize=2,
                    label=f"noise x{nl}")
    ax.set_xlabel("snapshots N in window")
    ax.set_ylabel("AUC vs internal GT")
    ax.set_title(f"(C) budget degradation\n(mean ±sd across all {len(windows)} windows)")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    savefig(fig, "real_cyclone.png")

    save_json("real_cyclone.json", {
        "mesh": {"nodes": mesh.cx.n_nodes, "edges": mesh.cx.n_edges,
                 "triangles": p, "rank_B2": rank, "full_column_rank": rank == p,
                 "mesh_step_deg": float(MESH_STEP * abs(lat[1] - lat[0]))},
        "season": {"t0": np.datetime_as_string(times[0], unit="h"),
                   "t1": np.datetime_as_string(times[-1], unit="h"),
                   "n_snapshots": int(T), "n_windows": len(windows),
                   "n_snapshots_used": int(len(windows) * WINDOW_LEN),
                   "n_snapshots_dropped": int(T - len(windows) * WINDOW_LEN),
                   "window_len": WINDOW_LEN,
                   "note": f"{T} six-hourly snapshots; non-overlapping "
                           f"{WINDOW_LEN}-snapshot windows use the first "
                           f"{len(windows) * WINDOW_LEN} and drop the trailing "
                           f"{T - len(windows) * WINDOW_LEN} (a partial window). "
                           f"All pooled metrics are over these {len(windows)} "
                           f"complete windows."},
        "task": "curl-based vortex localization (NOT filled-triangle "
                "topology recovery: the mesh has no equal-image confusers "
                "and real weather has no latent boolean support)",
        "statistic": "temporally centered curl energy, area^2-normalized "
                     "(vorticity scale); decorrelated (GLS, main Prop.1) variant reported "
                     "for comparison",
        "internal_validation": {"auc": auc_i,
                                "auc_ci95_block_bootstrap": list(boot_int),
                                "vort_thresh_per_s": VORT_THRESH,
                                "n_positive": int(int_labels.sum()),
                                "n_total": int(len(int_labels)),
                                "spearman_score_vs_vorticity": float(rho_s),
                                "decorrelated_auc_internal": float(auc_i_wh),
                                "decorrelated_spearman": float(rho_s_wh),
                                "baseline_auc_internal": float(auc_i_base)},
        "wind_label_counts": wind_counts,
        "primary_label_policy": PRIMARY_POLICY,
        "external_validation_primary": external[PRIMARY_POLICY],
        "external_by_wind_policy": external,
        "moving_block_is_primary_ci": True,
        "ci_status": "All real-data confidence intervals here are EXPLORATORY: "
                     "one season (WNP 2020), no multi-year or out-of-sample "
                     "validation; they quantify within-season resampling spread "
                     "only. The moving-block bootstrap is the PRIMARY CI; the "
                     "storm-cluster bootstrap is corroboration only (it collapses "
                     "storm multiplicity and is mildly anti-conservative).",
        "decorrelated_note": "G^{-1} noise amplification hurts pure detection "
                             "ranking on this nearly-regular full-rank mesh",
        "baseline_note": "classical pointwise FD vorticity from winds subsampled "
                         "at the mesh nodes (2.1 deg) - same information budget; "
                         "the edge-flow statistic should match it (the "
                         "contribution is the limits framework, not a new TC "
                         "detector)",
        "internal_auc_vs_vorticity_threshold": thresh_sweep,
        "degradation": {"Ns": Ns, "noise_levels": noise_levels,
                        "auc_by_noise": degradation,
                        "auc_sd_by_noise": deg_std,
                        "reps_per_window": reps, "n_windows_aggregated": len(windows),
                        "note": "uniform centered statistic, N>=3; each (noise,N) "
                                "is the mean over ALL windows (sd = window-to-"
                                "window spread); no theory-floor overlay "
                                "(units incommensurate)"},
        "per_window": per_window,
    })
    print("done.")


if __name__ == "__main__":
    main()
