# topo-flow-limits

**When Are Triangles Invisible? Fundamental Limits of Higher-Order Structure
Identifiability from Edge Flows.**

A signal-processing theory project (targeting IEEE ICASSP, Signal Processing
Theory & Methods / MLSP). It asks a question the existing topological-signal-
processing literature answers only algorithmically: *given edge-flow observations
on a graph, when is it even possible to identify which triangles of the underlying
simplicial complex are "filled"?*

The answer is governed entirely by the **curl** of the flows, and it has two failure
modes — one driven by signal-to-noise, one by geometry — that we make precise and
validate against simulation and real data.

## The idea in one paragraph

Model the observed edge flow as
`f_t = B1ᵀ a_t + B2_S y_t + h_t + n_t` (gradient nuisance + triangle-signal curl +
harmonic nuisance + noise), where the latent set `S` of filled triangles is what we
want. Taking the curl `c_t = B2ᵀ f_t` **annihilates the gradient and harmonic
nuisances** (because `B1 B2 = 0` and the harmonic space is curl-free), leaving only
the active-triangle signal plus projected noise. Per-triangle detection then reduces
to a **two-variance Gaussian test** whose optimal error exponent is a Gaussian
Chernoff information — and that exponent vanishes as the **curl-SNR** `ρ → 0`, giving
a sharp **curl-invisibility** phase transition `ρ*(T) ~ 1/√T`. A **whitened** version
of the statistic, `ŷ = G⁺ c` with `G = B2ᵀB2`, decorrelates edge-sharing (confusable)
triangles and achieves a **geometry-aware** effective SNR
`ρ_eff = σ_c² / (σ_n² (G⁺)_ττ)`. Finally, a **rank obstruction** means that on dense
graphs the candidate triangles vastly outnumber the curl dimensions: on `K_n` only a
`3/n` fraction is identifiable at *any* SNR.

## Key results (all Monte-Carlo validated)

| Result | Statement | Evidence |
|---|---|---|
| Curl annihilation (Lemma 1) | `B2ᵀf` sees only the curl; grad+harmonic vanish | `test_hodge.py` |
| Two-variance test (Prop. 2) | isolated-triangle detection, `v0=3σ_n²`, `v1=9σ_c²+3σ_n²`, error `~exp(-T·C)` | `test_theory_vs_sim.py` |
| Curl-invisibility (Thm 3) | `C(ρ)~ρ²/16`, floor `ρ*(T)~1/√T` | phase-transition fig |
| Geometry-aware whitening (Thm 4) | `ŷ=G⁺c`, `ρ_eff=σ_c²/(σ_n²(G⁺)_ττ)`; beats naive under confusability | confusability fig |
| Rank obstruction (Thm 5) | on `K_n`, identifiable fraction `= 3/n` | real-FX fig, `test_hodge.py` |

The empirical 50% exact-recovery contour matches the finite-sample theory to a
**median ratio of 1.01**.

## Figures

- `results/figures/phase_transition.png` — curl-invisibility phase transition;
  empirical recovery vs. theory contour.
- `results/figures/confusability.png` — naive curl-energy detector fails under
  edge-sharing; whitened detector recovers and matches geometry-aware theory.
- `results/figures/real_fx.png` — real FX flows: (A) arbitrage-free market is
  curl-free to machine precision; (B) `K_9` exposes only `28/84 = 3/9` of its
  triangles.

## Layout

```
paper/            main.tex (ICASSP draft), refs.bib, main.pdf, figures/
src/tfl/          hodge.py     — incidences, Hodge 1-Laplacian, grad/curl/harmonic split
                  generative.py— edge-flow model, planted complexes, curl-SNR
                  estimators.py— curl statistics, energy & whitened detectors, baselines
                  limits.py    — Chernoff info, Bayes error, invisibility floor,
                                 exact/heterogeneous recovery probabilities
experiments/      run_phase_transition.py, run_confusability.py, run_real_fx.py, run_all.py
tests/            test_hodge.py, test_theory_vs_sim.py   (16 tests)
data/             fx_rates.json — 257 days ECB reference rates (cached from frankfurter.app)
results/          figures/ + *.json metrics
```

## Reproduce

CPU-only; no GPU, no `torch`. Uses [`uv`](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe numpy scipy networkx cvxpy matplotlib pandas pytest

# validate the theory against simulation (16 tests, ~25 s)
.venv/Scripts/python.exe -m pytest -q

# regenerate every figure and metric (~2 min)
.venv/Scripts/python.exe experiments/run_all.py

# build the paper (requires a LaTeX toolchain, e.g. MiKTeX/TeX Live)
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data provenance & honesty notes

- `data/fx_rates.json` is real ECB daily reference-rate data for 2024, fetched once
  from `api.frankfurter.app` and cached so the repo is fully offline-reproducible.
- The FX study demonstrates the two **invisibility** mechanisms on real data; it does
  *not* claim a real recovered structure (an efficient market is curl-free — there is
  nothing to recover, which is exactly the point).
- The recovery *achievability* results are validated on controlled complexes where
  the higher-order structure is identifiable (the rank obstruction shows why the
  complete-graph FX case is not). Extending the converse to sub-Gaussian signals and
  to partial edge sampling is stated as future work in the paper.

## License

Research code released for reproducibility. MIT.
