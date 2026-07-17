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

## Part C — Experiments NOT run in this revision (honest deferrals)

These were requested but are **not done and not claimed as done**; they require
resources unavailable in the drafting environment.

1. **Server / scale experiments (`p = 10², 10³, 10⁴`).** The provided server
   `connect.weste.seetacloud.com` (and the earlier `…westb…`) did **not resolve
   via DNS** from the drafting environment, so no server run was possible. No
   GPU or full-scale result is claimed. `results/manifest.json` records the
   **local** hardware only. TODO: on a reachable machine, run a sparse/implicit
   lifted-operator version of `run_second_order.py` at `p≈10²–10⁴`, logging
   rank, N, runtime, and peak RSS; report GPU only if genuine CUDA/batched ops
   are used.
2. **Non-oracle threshold estimator.** The paper's `w_min/2` rule and known
   `σ_n` are an **oracle theorem setting** (now stated as such). A reproducible
   non-oracle rule (sample splitting / calibrated residual / BIC) with a proven
   or calibrated error is **not yet implemented**; the released oracle-free
   largest-gap rule is reported honestly as *worse*.
3. **Faithful literature baseline.** The grid compares three *implementations*
   (NNLS, oracle-aided subspace, generic greedy), not tuned literature methods.
   A faithful Gurugubelli–Chepuri MAP / Barbarossa–Sardellitti MTV baseline is
   **not implemented**; superiority claims are scoped to "the three evaluated
   implementations".
4. **Cyclone reprocessing.** Still TODO on the cached data (local, feasible):
   process all 244 snapshots (or state the exact dropped dates), aggregate
   panel C across all windows/storms, add sensitivity sweeps (localisation
   radius, 34-kt threshold, window length, bootstrap block length), and a
   storm-cluster bootstrap; label all real-data CIs **exploratory** (no
   multi-year validation).
5. **CI smoke tests.** A GitHub Actions workflow running `pytest -q` and a
   figure-free smoke of the fast experiments is **not yet added**.
6. **Clean-environment full rerun.** All experiments should be regenerated from
   a fresh environment against `requirements.lock`, then `make_manifest.py`
   re-run so the manifest hashes match; **not done here**.

---

## Part D — Reproduction pointers

- Environment: `requirements.lock`; provenance: `results/manifest.json`
  (regenerate with `python experiments/make_manifest.py`).
- Tests: `pytest -q` (51 tests). These are **guardrails, not proofs** — the
  identifiability claims are proven in `paper/supplement.tex` §S4 and
  independently re-derived in the commit history.
- Every quoted number lives in `results/*.json`.
