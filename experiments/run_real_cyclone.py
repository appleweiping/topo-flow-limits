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
  EXTERNAL  IBTrACS best-track cyclone positions (agency-verified, genuinely
            independent of the reanalysis) -> ROC / precision.

BASELINE  a classical non-simplicial comparator at the same information
          budget: pointwise finite-difference vorticity computed from the
          winds subsampled AT THE MESH NODES (2.1 deg), averaged per
          triangle. The edge-flow detector should not lose to it; the
          contribution here is the limits framework, not a new TC detector.

Pipeline per 4-day window (16 six-hourly snapshots):
  wind field -> mesh edge flows F -> temporally centered curl-ENERGY scores
  (the paper's primary statistic, Prop. 1), area^2-normalized so scores rank
  triangles on a common mean-VORTICITY scale across latitudes
  -> compare to both references. No oracle parameters anywhere.

Statistic choice, stated honestly: the GLS/BLUE-decorrelated score (Thm. 2)
targets exact-support recovery under edge-sharing leakage at a known SNR; on
this nearly-regular full-rank mesh its G^{-1} noise amplification hurts pure
DETECTION ranking (AUC 0.77 vs 0.91 internal). Both scores are computed and
reported in the JSON; the energy score is the headline panel.

A third panel degrades the snapshot budget N inside each window and shows the
empirical detection quality alongside the theoretical invisibility floor
rho*(N) (arbitrary units — a shape reference for the N-scaling, not a fitted
prediction; evaluation is threshold-free ROC ranking throughout).

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


def dataset_from_flows(F: np.ndarray, B1, B2, cx) -> FlowDataset:
    """Wrap raw edge-flow snapshots in the FlowDataset container (the real
    data has no known generative parameters; placeholders are unused by the
    plug-in detector)."""
    return FlowDataset(
        F=F, B1=B1, B2_all=B2, active=np.zeros(B2.shape[1], bool),
        params=FlowParams(sigma_curl=0.0), candidate_triangles=list(cx.triangles),
    )


def roc_curve(scores: np.ndarray, labels: np.ndarray, n_thresh: int = 200):
    order = np.argsort(-scores)
    s_sorted = scores[order]
    l_sorted = labels[order].astype(float)
    P = l_sorted.sum()
    Nn = len(labels) - P
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0.0
    for v in l_sorted:
        tp += v
        fp += 1 - v
        tpr.append(tp / max(P, 1))
        fpr.append(fp / max(Nn, 1))
    auc = float(np.trapezoid(tpr, fpr))
    return np.array(fpr), np.array(tpr), auc


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision (step-wise PR-AUC): sum over positives, in score
    order, of precision-at-that-rank weighted by recall increments."""
    order = np.argsort(-scores)
    l_sorted = labels[order].astype(float)
    P = l_sorted.sum()
    if P == 0:
        return 0.0
    tp = 0.0
    ap = 0.0
    for i, v in enumerate(l_sorted, start=1):
        if v:
            tp += 1
            ap += (tp / i) / P
    return float(ap)


def cluster_bootstrap_ci(
    per_window_scores: list[np.ndarray],
    per_window_labels: list[np.ndarray],
    metric,
    n_boot: int = 1000,
    seed: int = 7,
) -> tuple[float, float]:
    """95% percentile CI for a pooled ranking metric, resampling WHOLE
    WINDOWS with replacement (windows are the exchangeable clusters here;
    storms span windows, so window-level resampling is the conservative
    unit)."""
    rng = np.random.default_rng(seed)
    W = len(per_window_scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, W, size=W)
        s = np.concatenate([per_window_scores[i] for i in idx])
        l = np.concatenate([per_window_labels[i] for i in idx])
        if l.sum() == 0 or l.sum() == len(l):
            continue
        vals.append(metric(s, l))
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

    all_scores, all_int_gt, all_ext_gt = [], [], []
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
        ext = cyclone_triangle_labels(mesh, fixes, times, w)

        all_scores.append(scores)
        all_scores_wh.append(scores_wh)
        all_scores_base.append(scores_base)
        all_int_gt.append(gt_vort)
        all_ext_gt.append(ext)
        per_window.append({
            "t0": np.datetime_as_string(times[w][0], unit="h"),
            "n_ext_pos": int(ext.sum()),
        })

    scores_flat = np.concatenate(all_scores)
    scores_wh_flat = np.concatenate(all_scores_wh)
    scores_base_flat = np.concatenate(all_scores_base)
    int_gt_flat = np.concatenate(all_int_gt)
    ext_gt_flat = np.concatenate(all_ext_gt)

    # ---- internal validation: ROC against full-res vorticity ----
    int_labels = int_gt_flat > VORT_THRESH
    fpr_i, tpr_i, auc_i = roc_curve(scores_flat, int_labels)
    print(f"INTERNAL: {int(int_labels.sum())}/{len(int_labels)} vorticity-positive "
          f"triangle-windows; AUC = {auc_i:.3f}")

    # ---- external validation: ROC + PR + P@k against IBTrACS ----
    fpr_e, tpr_e, auc_e = roc_curve(scores_flat, ext_gt_flat)
    ap_e = pr_auc(scores_flat, ext_gt_flat)
    # operating point: top-k where k = number of external positives
    k = int(ext_gt_flat.sum())
    top = np.argsort(-scores_flat)[:k]
    hits = int(ext_gt_flat[top].sum())
    precision = hits / max(k, 1)
    print(f"EXTERNAL: {k} cyclone triangle-windows; AUC = {auc_e:.3f}; "
          f"PR-AUC = {ap_e:.3f}; P@k = {precision:.3f}")

    # cluster (per-window) bootstrap CIs for the headline pooled metrics
    int_labels_pw = [g > VORT_THRESH for g in all_int_gt]
    boot = {
        "auc_internal": cluster_bootstrap_ci(all_scores, int_labels_pw,
                                             lambda s, l: roc_curve(s, l)[2]),
        "auc_external": cluster_bootstrap_ci(all_scores, all_ext_gt,
                                             lambda s, l: roc_curve(s, l)[2]),
        "pr_auc_external": cluster_bootstrap_ci(all_scores, all_ext_gt, pr_auc),
    }
    print(f"window-bootstrap 95% CI: AUC_int={boot['auc_internal']}, "
          f"AUC_ext={boot['auc_external']}, PR_ext={boot['pr_auc_external']}")

    # rank correlation between detector score and internal GT
    from scipy.stats import spearmanr
    rho_s, _ = spearmanr(scores_flat, int_gt_flat)
    print(f"Spearman(score, |vorticity|) = {rho_s:.3f}")
    # honesty comparison: the decorrelated (Thm.2) score on the same task
    _, _, auc_i_wh = roc_curve(scores_wh_flat, int_labels)
    _, _, auc_e_wh = roc_curve(scores_wh_flat, ext_gt_flat)
    rho_s_wh, _ = spearmanr(scores_wh_flat, int_gt_flat)
    print(f"decorrelated variant: AUC_int={auc_i_wh:.3f} AUC_ext={auc_e_wh:.3f} "
          f"spearman={rho_s_wh:.3f}")
    # classical baseline at the same information budget (full metric set)
    fpr_b, tpr_b, auc_i_base = roc_curve(scores_base_flat, int_labels)
    _, _, auc_e_base = roc_curve(scores_base_flat, ext_gt_flat)
    ap_e_base = pr_auc(scores_base_flat, ext_gt_flat)
    top_b = np.argsort(-scores_base_flat)[:k]
    precision_base = int(ext_gt_flat[top_b].sum()) / max(k, 1)
    boot["auc_external_baseline"] = cluster_bootstrap_ci(
        all_scores_base, all_ext_gt, lambda s, l: roc_curve(s, l)[2])
    print(f"coarse-vorticity baseline: AUC_int={auc_i_base:.3f} "
          f"AUC_ext={auc_e_base:.3f} PR_ext={ap_e_base:.3f} "
          f"P@k={precision_base:.3f}")
    # sensitivity of the internal AUC to the vorticity threshold
    thresh_sweep = {}
    for th in (1e-5, 2e-5, 3e-5, 4e-5, 5e-5):
        _, _, a = roc_curve(scores_flat, int_gt_flat > th)
        thresh_sweep[f"{th:.0e}"] = float(a)
    print("threshold sweep (internal AUC):", thresh_sweep)

    # ---- degradation: snapshot budget vs detection quality ----
    # pick the most active window (most external positives) and degrade N
    wi = int(np.argmax([pw["n_ext_pos"] for pw in per_window]))
    w = windows[wi]
    zeta = grid_vorticity(u[w], v[w], lat, lon)
    gt_labels_w = triangle_mean_abs_vorticity(zeta, tri_pts) > VORT_THRESH
    rng = np.random.default_rng(0)
    noise_levels = [0.0, 0.5, 1.0, 2.0]
    Ns = [3, 4, 6, 8, 12, 16]  # N >= 3 so the CENTERED statistic is used
                               # uniformly (mixing statistics across N made
                               # the curve non-monotone and uninterpretable)
    degradation = {}
    deg_std = {}
    F_w = F_all[:, w]
    flow_rms = float(np.sqrt(np.mean(F_w**2)))
    for nl in noise_levels:
        aucs = []
        for N in Ns:
            vals = []
            for rep in range(12):
                idx = rng.choice(WINDOW_LEN, size=N, replace=False)
                F = F_w[:, idx] + nl * flow_rms * rng.standard_normal((F_w.shape[0], N))
                C = B2.T @ F
                C = C - C.mean(axis=1, keepdims=True)
                s = np.mean(C**2, axis=1) / area_norm
                _, _, a = roc_curve(s, gt_labels_w)
                vals.append(a)
            aucs.append(float(np.mean(vals)))
            deg_std.setdefault(str(nl), []).append(float(np.std(vals)))
        degradation[str(nl)] = aucs

    # ---- figure ----
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12,
                         "axes.labelsize": 12})
    fig = plt.figure(figsize=(12.5, 3.6))

    ax = fig.add_subplot(1, 3, 1)
    wi_show = wi
    sc = all_scores[wi_show]
    cen_lat = np.array([np.mean([mesh.node_lat[t] for t in tri]) for tri in mesh.cx.triangles])
    cen_lon = np.array([np.mean([mesh.node_lon[t] for t in tri]) for tri in mesh.cx.triangles])
    im = ax.scatter(cen_lon, cen_lat, c=np.log10(sc + 1e-12), s=12, cmap="viridis")
    fx_w = all_ext_gt[wi_show]
    ax.scatter(cen_lon[fx_w], cen_lat[fx_w], facecolors="none", edgecolors="red",
               s=60, linewidths=1.2, label="IBTrACS cyclone")
    ax.set_title(f"(A) curl-energy score (vorticity scale), window {per_window[wi_show]['t0']}")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10 score")

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(fpr_i, tpr_i,
            label=f"internal (vorticity), AUC={auc_i:.3f}")
    ax.plot(fpr_e, tpr_e,
            label=f"external (IBTrACS), AUC={auc_e:.3f}, PR={ap_e:.3f}")
    ax.plot(fpr_b, tpr_b, ":", color="gray",
            label=f"baseline (coarse vort.), AUC={auc_i_base:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("(B) vortex localization: ROC vs both references")
    ax.legend(loc="lower right", fontsize=8)

    ax = fig.add_subplot(1, 3, 3)
    for nl in noise_levels:
        y = np.array(degradation[str(nl)])
        e = np.array(deg_std[str(nl)])
        ax.errorbar(Ns, y, yerr=e, fmt="o-", ms=3, capsize=2,
                    label=f"noise x{nl}")
    ax.set_xlabel("snapshots N in window")
    ax.set_ylabel("AUC vs internal GT")
    ax.set_title("(C) budget degradation (mean ± sd over 12 draws)")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    savefig(fig, "real_cyclone.png")

    save_json("real_cyclone.json", {
        "mesh": {"nodes": mesh.cx.n_nodes, "edges": mesh.cx.n_edges,
                 "triangles": p, "rank_B2": rank, "full_column_rank": rank == p,
                 "mesh_step_deg": float(MESH_STEP * abs(lat[1] - lat[0]))},
        "season": {"t0": np.datetime_as_string(times[0], unit="h"),
                   "t1": np.datetime_as_string(times[-1], unit="h"),
                   "n_snapshots": int(T), "n_windows": len(windows)},
        "task": "curl-based vortex localization (NOT filled-triangle "
                "topology recovery: the mesh has no equal-image confusers "
                "and real weather has no latent boolean support)",
        "statistic": "temporally centered curl energy, area^2-normalized "
                     "(vorticity scale); decorrelated (Thm.2) variant reported "
                     "for comparison",
        "internal_validation": {"auc": auc_i,
                                "auc_ci95_window_bootstrap": list(boot["auc_internal"]),
                                "vort_thresh_per_s": VORT_THRESH,
                                "n_positive": int(int_labels.sum()),
                                "n_total": int(len(int_labels)),
                                "spearman_score_vs_vorticity": float(rho_s)},
        "external_validation": {"auc": auc_e,
                                "auc_ci95_window_bootstrap": list(boot["auc_external"]),
                                "pr_auc": ap_e,
                                "pr_auc_ci95_window_bootstrap": list(boot["pr_auc_external"]),
                                "n_cyclone_triangle_windows": k,
                                "precision_at_k": precision,
                                "prevalence": k / int(len(ext_gt_flat)),
                                "n_ibtracs_fixes": len(fixes)},
        "decorrelated_variant": {"auc_internal": float(auc_i_wh),
                                 "auc_external": float(auc_e_wh),
                                 "spearman": float(rho_s_wh),
                                 "note": "G^{-1} noise amplification hurts pure "
                                         "detection ranking on this nearly-regular "
                                         "full-rank mesh"},
        "baseline_coarse_fd_vorticity": {
            "auc_internal": float(auc_i_base), "auc_external": float(auc_e_base),
            "auc_external_ci95_window_bootstrap": list(boot["auc_external_baseline"]),
            "pr_auc_external": float(ap_e_base),
            "precision_at_k_external": float(precision_base),
            "note": "classical pointwise FD vorticity from winds subsampled at "
                    "the mesh nodes (2.1 deg) — same information budget; the "
                    "edge-flow statistic should match it (the contribution is "
                    "the limits framework, not a new TC detector)"},
        "internal_auc_vs_vorticity_threshold": thresh_sweep,
        "degradation": {"Ns": Ns, "noise_levels": noise_levels,
                        "auc_by_noise": degradation,
                        "auc_sd_by_noise": deg_std,
                        "note": "uniform centered statistic, N>=3; no "
                                "theory-floor overlay (units incommensurate)"},
        "per_window": per_window,
    })
    print("done.")


if __name__ == "__main__":
    main()
