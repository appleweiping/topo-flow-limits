# AUTHOR REWRITE CHECKLIST

This repository's manuscript (`paper/main.tex`, `paper/supplement.tex`) and
`README.md` are an **AI-assisted draft**. Substantial portions of the prose,
code, and figures were produced with an AI system under the author's direction.
IEEE/SPS policy does **not** permit AI-generated manuscript *text* in a
submission. Before any submission the author (Weiping Yan) must:

1. **Independently rewrite the manuscript text** in their own words (the math,
   experiments, and numbers below are the substance to re-express — do not copy
   the AI-drafted sentences).
2. **Write the AI-use disclosure to match the final, author-written text** —
   describe only the uses that actually remain (e.g. code, figures, editing,
   review feedback). Do **not** minimise or misstate AI involvement, and do
   **not** rewrite git history to hide it (the full history is intentionally
   preserved).
3. Verify every number against `results/*.json` (see `scripts/`/tests) and every
   citation against `paper/refs.bib`.

Nothing here should be read as "ready to submit"; it is a draft plus an audit
trail.

---

## Part A — Per-section: what the author must independently rewrite

| Section | Must be re-written by the author | Substance to preserve (verify, don't copy) |
|---|---|---|
| Abstract | Entire paragraph | Three excitation regimes; regime (b) determines only `C_S={M⪰0:im M⊆im U_S}`; strict-positive-diagonal K4 witness; projector = Hodge-smoothness prior; NNLS + O(1/N) bound; cyclone vortex localization |
| §1 Introduction / related work | Entire section | Gaps in Barbarossa–Sardellitti §VII, Marinucci et al. 2025 (SGM), Gurugubelli–Chepuri, Liu et al.; the bounded novelty claim ("to our knowledge none characterizes, as a function of the excitation class, when …") |
| §2 Model + Definition 1 | Prose around the equations | Candidate-orthogonal harmonic `h∈ker B₂ᵀ`; population-covariance framing; realized range `R=im M` |
| §3 Theorem 1 (three regimes) | Statement prose + proof sketch | (a) diagonal ⇒ weighted support; (b) `C_S` set identity, support only up to `{S:im M⊆im U_S}`, strict-positive-diagonal witness; (c) projector indistinguishability. **The math is verified; re-express it.** |
| §3 Theorem 2 (separations/exponents) | Statement + proof sketch | `‖ΔD‖²≥9` (subset) / `≥16` (swap); `C_G=ρ₂²‖ΔD‖²/16=‖ΔM‖²/(16σ_n⁴)`; "rank deficiency free" **scoped to fixed pair + isotropic + leading low-SNR** |
| §4 Theorem 3 (NNLS) + experiments | Prose | Consistency + explicit `16[(trΣ)²+‖Σ‖²]/(Nσ_min²w_min²)` bound; worst-case-over-supports per cell; non-vacuous at 7/260 cells; largest-gap is **worse** (0.63 vs 0.69); "three evaluated implementations, not tuned literature methods"; projector "chance = 1/m" |
| §5 Cyclone | Prose | Vortex localization (not topology recovery); external (separately-curated) IBTrACS; raw 726 fixes/13 storms vs eligible 414/23310 triangle-windows; most-active-window panel A is illustrative; moving-block bootstrap |
| §6 Conclusion + open problems | Entire section | Regime summary (no "image-only"); open problems |
| Page-5 compliance | **Author rewrites the AI-use disclosure to match the final text** | Funding/COI; data sources; the disclosure must be truthful about the final manuscript |
| Supplement S1–S5 | Prose | Fano converses; partial sampling; plug-in (now an **empirical remark**, not a proven bound); S4 proofs; S5 cyclone methods |

---

## Part B — Open problems / claims deliberately NOT proven (do not overstate)

1. **Plug-in perturbation bound (supplement §S3).** Demoted from a proposition
   to an *empirical observation*: there is **no proven uniform explicit-constant**
   bound on the adaptive detector's excess error. A non-asymptotic joint
   selection-plus-estimation bound with explicit constants is open.
2. **Sub-Gaussian converse with matching constants** (only achievability given).
3. **Sharp first-order sparse-support thresholds over `ker B₂`.**
4. **Temporal dependence** in the cyclone study: `N_eff` not characterized.

---

## Part C — Experiments: status (round-8 update)

Items 1–3 below were **completed in round 8** on a reachable GPU server (see
supplement §S6 and `results/{scaling,gpu_mc,selection}.json`); items 4–6 and
the GNN remain **not done and not claimed as done**.

1. **Server / scale experiments (`p = 10², 10³, 10⁴`).** ✅ **DONE.** A rented
   AutoDL GPU instance (live endpoint/IP redacted from the public repo) — an
   RTX 4080 SUPER (32 GB) + 128-core Xeon — was used; each result JSON now
   carries a same-process `_provenance` block (git SHA, host, command,
   timestamp, env, hardware, wall time, peak RSS/VRAM) as the authoritative
   machine record (`src/tfl/provenance.py`), not a stapled-on probe. A matrix-free
   lifted operator (`src/tfl/estimators_mf.py`, verified equal to the dense
   `scipy.nnls` support with 0 mismatches) recovers at `p=1000` in ~0.49 s/solve
   on CPU (dense operator would be 2.0 GB), and a GPU-batched Monte-Carlo
   (`src/tfl/gpu.py`, genuine CUDA batched GEMMs) gives 0.967/1.00/1.00 at
   `p=1000` and a tiled `p≈10⁴` feasibility point — all logged with runtime and
   peak GPU memory (§S6, `run_scaling.py` / `run_gpu_mc.py`).
2. **Non-oracle threshold estimator — with a RETRACTION (round-9).** A
   BIC-over-support-path rule (`src/tfl/selection.py`) uses no `w_min` and an
   estimated `σ_n`. The earlier claim of *unconditional* "classical BIC selection
   consistency" was **wrong and is retracted**: `run_bic_boundary.py` /
   `tests/test_bic_boundary.py` show it FAILS on dense supports even at `N=10⁵`
   (K6, `k≥12` → 0 recovery) because the median-eigenvalue `σ_n` is a noise
   eigenvalue only while the active curl-covariance rank `< r/2`
   (noise-identifiability). What holds: GIVEN an identifiable `σ_n`, classical
   Gaussian-BIC consistency under fixed `p`, lifted-atom injectivity, β-min, and
   a unique minimizer (the oracle-`σ_n` path then recovers even `k=p`). The
   median-`σ_n` rule is therefore an **empirical selector** valid in the
   noise-identifiable (sufficiently sparse) regime the paper's `k=3` experiments
   use; it matches — even beats at small `N` — the oracle there (§S6). A
   sample-split variant is the assumption-light fallback.
3. **Faithful literature baseline.** ✅ **DONE (convex-sparse).** A non-negative
   lifted-LASSO (soft-threshold FISTA, `λ` by BIC) in the sparse-covariance /
   sparse-PCA lineage is implemented and evaluated. *Not done:* a tuned
   Gurugubelli–Chepuri MAP / Barbarossa PCA-BFMTV port (those target different
   objectives on our model). **Because those direct literature baselines are not
   ported, the paper makes NO generalized superiority claim: every comparison is
   scoped to the three implementations actually evaluated** (NNLS, subspace,
   greedy) plus the convex-sparse LASSO, on our generative model.
   **Also not done:** a GPU-trained GNN detector baseline (`src/tfl/neural.py`
   was scoped but not implemented). This is a genuine gap in the baseline suite;
   it is **not** excused by "the GPU is already exercised elsewhere" (that
   conflates a scaling demo with a missing comparator).
4. **Cyclone reprocessing.** ✅ **DONE** (`run_real_cyclone.py`,
   `results/real_cyclone.json`, supplement cyclone §). All 244 snapshots are
   loaded; the 15 non-overlapping 16-snapshot windows use 240 and drop the
   trailing 4 (disclosed in `season`). Panel C (budget degradation) is now the
   mean ± sd across **all 15 windows**, not the most-active one. Raw-vs-eligible
   denominators are reported. **Round-9: split wind labels.** ~1/3 of fixes
   (229/726, 11 storms) have no reported wind; the old labeling silently counted
   them as ≥34 kt. Now BOTH **strict** (finite ≥34 kt; 312 positive) and
   **inclusive** (missing counted; 414 positive) are computed and reported;
   neither is privileged. Strict AUC 0.950 [0.915, 0.975], inclusive 0.920
   [0.879, 0.951]; the detector beats the baseline on all three external metrics
   under both. Sensitivity sweeps (radius, wind cutoff, window length, block
   length) run for both; **monotonicity is a per-sweep computed flag, not an
   assertion** — the wind-cutoff sweep is NOT monotone under inclusive labeling.
   The **moving-block bootstrap is the primary CI**; the storm-cluster bootstrap
   is corroboration only (mildly anti-conservative). All real-data CIs are
   **exploratory** (one 2020 season, no multi-year validation).
5. **CI smoke tests.** ✅ **DONE** (`.github/workflows/ci.yml`). CPU-only,
   torch-free matrix (Python 3.11/3.12): asserts the core imports without torch,
   compile-checks every `src/`/`experiments/` file, runs `pytest -q`.
6. **Clean-environment rerun.** ✅ **DONE (server).** A fresh `venv` on the GPU
   server with the exact pinned deps (`numpy==2.4.6`, `scipy==1.17.1`,
   `matplotlib==3.10.9`, `pytest==9.0.3`) runs the full suite green
   (**58 passed**), confirming `requirements.lock` is sufficient. The GPU extra
   (`torch==2.8.0+cu128`) is recorded separately. Bit-identical manifest hashes
   across machines are **not** claimed: RNG/BLAS differences make experiment
   outputs vary at the last digits (the round-8 server JSONs are the committed
   authoritative copies), so the manifest binds the *released* artifacts, not a
   cross-machine hash match.

---

## Part D — Reproduction pointers

- Environment: `requirements.lock`; provenance: `results/manifest.json`
  (regenerate with `python experiments/make_manifest.py`).
- Tests: `pytest -q` (63 tests). These are **guardrails, not proofs** — the
  identifiability claims are proven in `paper/supplement.tex` §S4 and
  independently re-derived in the commit history.
- Every quoted number lives in `results/*.json`.
