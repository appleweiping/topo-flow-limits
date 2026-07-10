# topo-flow-limits

**When Are Triangles Invisible? Fundamental Limits of Higher-Order Structure
Identifiability from Edge Flows.**

A complete, reproducible signal-processing-theory research project (targeting the
IEEE ICASSP *Signal Processing Theory & Methods* track). Pure CPU — no GPU, no
`torch`. Every theorem in the paper is cross-checked against Monte-Carlo simulation
by the test suite before it is allowed into the manuscript.

> **TL;DR** — Topological signal processing needs to know *which triangles of a
> network are "filled"* (carry higher-order interactions). Existing papers give
> **algorithms** that estimate this from edge-flow data. This project answers the
> question those papers skip: **when is that estimation possible at all?** The
> answer: the hidden triangles are visible *only* through the **curl** of the flows,
> and they become provably invisible in two independent ways — (1) when the curl
> signal-to-noise ratio drops below a sharp threshold `ρ*(T) ~ 1/√T`
> (**curl-invisibility phase transition**), and (2) when the graph's geometry makes
> triangle signatures linearly dependent (**rank obstruction**: on a complete graph
> only a `3/n` fraction of triangles is identifiable at *any* SNR). Both effects
> show up, exactly as predicted, in real foreign-exchange data.

---

## Table of contents

1. [The problem in plain language](#1-the-problem-in-plain-language)
2. [Mathematical setting (5-minute Hodge primer)](#2-mathematical-setting-5-minute-hodge-primer)
3. [The observation model](#3-the-observation-model)
4. [Main results](#4-main-results)
5. [The three figures, explained](#5-the-three-figures-explained)
6. [Repository layout — every file explained](#6-repository-layout--every-file-explained)
7. [Installation and full reproduction](#7-installation-and-full-reproduction)
8. [What each test proves](#8-what-each-test-proves)
9. [Real data: provenance and format](#9-real-data-provenance-and-format)
10. [Relation to prior work](#10-relation-to-prior-work)
11. [Limitations and honesty notes](#11-limitations-and-honesty-notes)
12. [Roadmap](#12-roadmap)
13. [中文速览](#13-中文速览)
14. [Citation & license](#14-citation--license)

---

## 1. The problem in plain language

Many networked measurements are **flows on edges**: money flowing between
currencies, cars between intersections, packets between routers, rank preferences
between items. Graph signal processing treats these edge values pairwise. But real
systems also have **higher-order** structure: *triangles* of the graph where three
edges interact as a unit (a triangular arbitrage loop, a three-way traffic
circulation, a 3-clique of correlated brain regions).

Topological signal processing (TSP) models this by "filling in" triangles of a
**simplicial 2-complex** and placing signals on them. The catch: in practice nobody
tells you *which* triangles are filled. A growing literature infers the filled set
from edge-flow data — with greedy searches, MAP estimators, sparse recovery. All of
these are **algorithms**. None of them answers the prior question:

> Given `T` snapshots of edge flows with a given noise level, is the filled-triangle
> set *identifiable at all*? How many snapshots do you need? Does the answer depend
> on the shape of the graph?

This project answers those questions with sharp, closed-form limits — and provides
the estimator that achieves them.

---

## 2. Mathematical setting (5-minute Hodge primer)

A **simplicial 2-complex** consists of nodes, oriented edges, and oriented
triangles. We fix the conventions used throughout `src/tfl/hodge.py`:

- An edge is an ordered pair $(i,j)$ with $i<j$, oriented $i \to j$.
- The **node–edge incidence** $B_1 \in \mathbb{R}^{|V|\times|E|}$ has
  $B_1[i,e]=-1,\ B_1[j,e]=+1$ for edge $e=(i,j)$.
- A triangle is an ordered triple $(i,j,k)$, $i<j<k$. The **edge–triangle
  incidence** $B_2 \in \mathbb{R}^{|E|\times|\mathcal{T}|}$ has, for triangle
  $t=(i,j,k)$: $+1$ on edge $(i,j)$, $-1$ on edge $(i,k)$, $+1$ on edge $(j,k)$.

These signs implement the boundary operators of algebraic topology, and give the
fundamental identity ("the boundary of a boundary is zero"):

$$B_1 B_2 = 0 .$$

Consequently the space of edge flows splits **orthogonally** (the *Hodge
decomposition*) into three interpretable parts:

```math
\mathbb{R}^{|E|} \;=\; \underbrace{\mathrm{im}\, B_1^{\top}}_{\text{gradient}}
\;\oplus\; \underbrace{\mathrm{im}\, B_2}_{\text{curl}}
\;\oplus\; \underbrace{\ker L_1}_{\text{harmonic}},
\qquad L_1 = B_1^{\top}B_1 + B_2 B_2^{\top}.
```

- **Gradient** flows are induced by node potentials (e.g. log prices): $f = B_1^\top a$.
- **Curl** flows circulate around *filled triangles*: $f = B_2 y$ for a triangle signal $y$.
- **Harmonic** flows circulate around *holes* (cycles not filled by triangles);
  $\dim(\ker L_1)$ equals the first Betti number $b_1$ of the complex.

The **curl operator** is $\mathrm{curl}(f) = B_2^{\top} f$; the divergence is $B_1 f$.
Everything above is unit-tested in `tests/test_hodge.py` (orthogonality,
reconstruction, $B_1B_2=0$, Betti dimension counts).

---

## 3. The observation model

The 1-skeleton (nodes and edges) is known. The **latent** object is the set
$S \subseteq \mathcal{T}$ of filled triangles among a candidate set (e.g. all
3-cliques). We observe $T$ i.i.d. edge-flow snapshots

```math
f_t \;=\; \underbrace{B_1^{\top} a_t}_{\text{gradient nuisance}}
\;+\; \underbrace{B_{2,S}\, y_t}_{\text{signal (carries } S\text{)}}
\;+\; \underbrace{h_t}_{\text{harmonic nuisance}}
\;+\; \underbrace{n_t}_{\text{noise}}, \qquad t = 1,\dots,T,
```

with $a_t \sim \mathcal N(0,\sigma_g^2 I)$, $y_t \sim \mathcal N(0,\sigma_c^2 I)$
(one coordinate per *active* triangle), harmonic $h_t$, and
$n_t \sim \mathcal N(0,\sigma_n^2 I)$. Only the snapshots $f_t$ are observed; the
task is to recover $S$. This is the same generative family used by the algorithmic literature
(e.g. Gurugubelli & Chepuri, EUSIPCO 2024) — we characterize when *their* task is
solvable. Implementation: `src/tfl/generative.py` (`sample_flows`, `FlowParams`,
planted-complex builders).

The single most important structural fact (Lemma 1 in the paper):

> **Curl annihilation.** $B_2^{\top} B_1^{\top} = 0$ and $B_2^{\top} h = 0$ for
> harmonic $h$. So the curl statistic
> $c_t = B_2^{\top} f_t = (B_2^{\top} B_{2,S}) y_t + B_2^{\top} n_t$
> is **completely immune to the gradient and harmonic nuisances, no matter how
> large they are.** All information about $S$ — and only that information —
> survives the curl map.

The test `test_curl_detection_is_immune_to_gradient_and_harmonic_nuisance` verifies
this end-to-end with nuisance 25× stronger than the signal.

---

## 4. Main results

Everything below is implemented in `src/tfl/limits.py` + `src/tfl/estimators.py`
and validated in `tests/test_theory_vs_sim.py`.

### R1 — Per-triangle detection is a two-variance Gaussian test

For an *isolated* candidate triangle (no shared edges), the curl scalar
$c_{\tau,t}$ is zero-mean Gaussian with variance

```math
v_0 = 3\sigma_n^2 \quad (\tau \notin S), \qquad
v_1 = 9\sigma_c^2 + 3\sigma_n^2 = v_0(1+\rho) \quad (\tau \in S),
\qquad \rho \;\triangleq\; \frac{3\sigma_c^2}{\sigma_n^2}.
```

(the 3 and 9 come from each $B_2$ column having exactly three $\pm 1$ entries).
The energy $E=\sum_t c_{\tau,t}^2$ is sufficient, $E/v_i \sim \chi^2_T$, and the
minimum Bayes error has an exact closed form at every finite $T$ (implemented in
`two_variance_bayes_error`) and decays like $e^{-T C(v_0,v_1)}$ where
$C$ is the **Gaussian Chernoff information** (`gaussian_chernoff_information`).

### R2 — The curl-invisibility phase transition

As $\rho \to 0$, $C(\rho) = \rho^2/16 + o(\rho^2)$ (validated to 5% accuracy at
$\rho \le 0.02$). Hence for a target error $\delta$ and budget $T$, detection
requires

```math
\rho \;\ge\; \rho^\star(T) \;=\; \Theta\!\Big(\sqrt{\tfrac{\log(1/\delta)}{T}}\Big),
```

and **below $\rho^\star$ no estimator whatsoever can recover the structure** — the
triangles are *information-theoretically invisible*. Numerically
(`invisibility_curl_snr_floor`, $\delta=0.05$): $T{=}25 \to \rho^\star{=}3.07$,
$T{=}100 \to 1.00$, $T{=}400 \to 0.41$, $T{=}1600 \to 0.19$ — the $1/\sqrt{T}$ law.

### R3 — Exact finite-sample recovery law (edge-disjoint case)

With independent triangles and the Bayes-optimal threshold, exact support recovery
has probability $\prod_{\tau \in S}(1-P_{\rm miss}) \prod_{\tau \notin S}(1-P_{\rm fa})$
with $\chi^2_T$ tail expressions (`exact_recovery_probability`). Its 50% contour is
the phase boundary in Figure 1, and the **empirical contour matches it to a median
ratio of 1.01** across the whole grid.

### R4 — Geometry-aware whitening beats naive detection under confusability

Edge-sharing triangles have correlated curl signatures: the triangle Gram matrix
$G = B_2^{\top} B_2$ has $G_{\sigma\tau} = \pm 1$ when $\sigma,\tau$ share an edge.
An active triangle then *leaks* energy onto its inactive neighbours, so naive
curl-energy thresholding fails — **and gets worse as SNR grows** (more signal, more
leakage). The fix: whiten with the pseudoinverse,

```math
\hat y_t = G^{+} c_t \;\Rightarrow\;
\hat y_{\tau,t} = y_{\tau,t}\,\mathbf{1}\{\tau\in S\} + e_{\tau,t},\qquad
\mathrm{Cov}(e_t) = \sigma_n^2 G^{+},
```

turning each triangle back into a two-variance test with **its own effective SNR**

```math
\rho^{\mathrm{eff}}_\tau = \frac{\sigma_c^2}{\sigma_n^2 (G^{+})_{\tau\tau}}.
```

$(G^{+})_{\tau\tau}$ is the *price of geometry*: $1/3$ for an isolated triangle
(recovering R1 exactly), larger as signatures overlap. On the edge-sharing strip
benchmark at $\rho{=}8,\ T{=}200$: naive Hamming error **2.59** (never recovers),
greedy baseline 0.27, **whitened 0.00**, and the whitened empirical recovery matches
the heterogeneous theory (`heterogeneous_exact_recovery_probability`) within 0.025.

### R5 — The rank obstruction: geometry-side invisibility

Whitening cannot beat linear dependence. Two triangle sets with
$\mathrm{im}~B_{2,S} = \mathrm{im}~B_{2,S'}$ are indistinguishable **at any SNR
and any $T$**. On the complete graph $K_n$ the curl subspace has dimension
$\mathrm{rank}(B_2) = \binom{n-1}{2}$ while there are $\binom{n}{3}$ candidate
triangles — the identifiable fraction is **exactly $3/n$** (proved via
$\mathrm{im}~B_2 = \ker B_1$ for the simply-connected 2-skeleton; verified
numerically for $K_5,\dots,K_{12}$ in `test_curl_dimension_ratio_on_complete_graph`).

---

## 5. The three figures, explained

All are regenerated by `experiments/run_all.py` (~2–3 min, CPU) into
`results/figures/`; the copies in `paper/figures/` used by the manuscript are a
manual copy of these outputs (kept in sync whenever figures change).

### Figure 1 — `phase_transition.png` (headline result)

Heatmap of the empirical probability of *exact* filled-triangle recovery on an
edge-disjoint planted complex (8 candidates, 4 active, gradient+harmonic nuisance
present), sweeping snapshots $T \in [5, 90]$ vs curl-SNR $\rho \in [0.3, 30]$,
200 trials per cell. **Solid white line** = exact finite-sample theory contour (R3);
it lies right on the empirical 50% boundary (median ratio 1.01, all within ±25%).
**Dotted white line** = asymptotic Chernoff+Bonferroni floor (R2), visibly the
$1/\sqrt T$ law and conservative, as an asymptotic bound should be.
Produced by `experiments/run_phase_transition.py`, metrics in
`results/phase_transition.json`.

### Figure 2 — `confusability.png` (why geometry matters)

Left: mean Hamming error vs $T$ on a 9-triangle *edge-sharing strip* with
alternating active triangles at $\rho = 8$. The naive curl-energy detector
plateaus at ≈2.1–2.6 wrong triangles and **worsens with more data** (leakage is
systematic, not noise); greedy stalls ≈0.3; the **whitened detector goes to 0**.
Right: whitened detector's empirical exact-recovery probability vs its
geometry-aware theoretical law — max deviation 0.025.
Produced by `experiments/run_confusability.py`, metrics in
`results/confusability.json`.

### Figure 3 — `real_fx.png` (both invisibilities in the wild)

Real data: 257 trading days (Dec 29, 2023 – Dec 31, 2024) of ECB daily reference
rates, 9 currencies (USD, AUD, CAD, CHF, EUR, GBP, JPY, NOK, SEK) → complete graph
$K_9$, 36 edges, 84 candidate triangles. Edge flow on day $t$ = log-price difference (a real,
heavy-tailed, temporally-correlated **gradient**).
**(A)** Hodge energy split: curl/gradient ≈ $1.8\times10^{-31}$ — an arbitrage-free
market is a *pure gradient*, i.e. genuinely curl-invisible: no estimator, however
clever, can find higher-order structure because there is none in the observable
component. **(B)** Identifiable fraction $\mathrm{rank}(B_2)/\binom{n}{3}$
for $K_5,\dots,K_{12}$: the numeric rank sits exactly on the $3/n$ curve; $K_9$
gives $28/84 = 1/3$. Produced by `experiments/run_real_fx.py`, metrics in
`results/real_fx.json`.

---

## 6. Repository layout — every file explained

```
topo-flow-limits/
├── pyproject.toml            project metadata; deps; pytest config (pythonpath=src)
├── LICENSE                   MIT
├── README.md                 this file
├── .gitignore                venv, caches, LaTeX aux, raw-data bulk
│
├── src/tfl/                  the library (numpy/scipy/cvxpy/networkx only)
│   ├── __init__.py           public API re-exports
│   ├── hodge.py              Complex dataclass (validated); build_incidences (B1, B2);
│   │                         hodge_1_laplacian; hodge_decomposition (SVD-based
│   │                         projections); curl / divergence operators
│   ├── generative.py         the observation model: FlowParams, sample_flows,
│   │                         curl_snr; planted-complex builders
│   │                         (disjoint_triangle_complex = clean regime,
│   │                         triangle_strip_complex = confusable regime,
│   │                         all_triangles = 3-clique candidate enumeration);
│   │                         harmonic_basis (eigendecomposition of L1)
│   ├── estimators.py         curl_statistics (sufficient statistics);
│   │                         energy_detector_bayes_support (naive, Chernoff-optimal
│   │                         threshold); whitened_curl_scores +
│   │                         whitened_curl_detector_support (proposed, geometry-aware,
│   │                         'bayes'/'fwer' threshold modes); effective_curl_snr;
│   │                         sparse_curl_covariance_support (cvxpy lifted lasso);
│   │                         greedy_support (baseline); hamming_error/exact_recovery
│   └── limits.py             the theory in closed form: curl_variances;
│                             gaussian_chernoff_information; two_variance_bayes_error
│                             (+ optimal threshold); invisibility_curl_snr_floor
│                             (bisection on the Chernoff rate);
│                             exact_recovery_probability & recovery_contour_rho
│                             (finite-sample, edge-disjoint); whitened_variances &
│                             heterogeneous_exact_recovery_probability (geometry-aware);
│                             per_triangle_threshold (bayes | fwer/Bonferroni)
│
├── experiments/
│   ├── _util.py              results paths; Agg (headless) matplotlib; JSON/figure
│   │                         savers; FastFlowSampler (pre-factorized fast sampling)
│   ├── run_phase_transition.py   Figure 1 + phase_transition.json
│   ├── run_confusability.py      Figure 2 + confusability.json
│   ├── run_real_fx.py            Figure 3 + real_fx.json
│   └── run_all.py                one-click: all of the above from fixed seeds
│
├── tests/                    16 tests — the theorems' guardrails (see §8)
│   ├── test_hodge.py
│   └── test_theory_vs_sim.py
│
├── data/
│   ├── fx_rates.json         cached real ECB rates (2024, 257 days) — offline repro
│   └── fetch_fx.py           refetch/extend the dataset (no API key needed)
│
├── results/                  regenerated by run_all.py
│   ├── figures/              phase_transition.png, confusability.png, real_fx.png
│   └── *.json                all numbers quoted in the paper
│
└── paper/
    ├── main.tex              the manuscript (IEEEtran; swap to spconf for camera-ready)
    ├── refs.bib              bibliography
    ├── main.pdf              compiled draft (3 pages, zero overfull boxes)
    └── figures/              figure copies used by the manuscript
```

---

## 7. Installation and full reproduction

Requirements: Python ≥ 3.11 and [`uv`](https://github.com/astral-sh/uv) (or plain
`pip`). Everything is CPU-only.

**Windows (PowerShell):**

```powershell
git clone https://github.com/appleweiping/topo-flow-limits.git
cd topo-flow-limits
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe numpy scipy networkx cvxpy matplotlib pandas pytest

# 1) validate all theory against Monte-Carlo (16 tests; ~25 s idle, up to ~90 s under load)
.\.venv\Scripts\python.exe -m pytest -q

# 2) regenerate every figure and metric (~2-3 min)
.\.venv\Scripts\python.exe experiments\run_all.py
```

**Linux / macOS:**

```bash
git clone https://github.com/appleweiping/topo-flow-limits.git
cd topo-flow-limits
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy networkx cvxpy matplotlib pandas pytest
.venv/bin/python -m pytest -q
.venv/bin/python experiments/run_all.py
```

**Build the paper** (needs a LaTeX toolchain — MiKTeX or TeX Live):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Refetch the FX data** (optional; the repo ships the cached file):

```bash
python data/fetch_fx.py
```

Expected `run_all.py` output (seeds are fixed; numbers reproduce exactly):

```
[1/3] phase transition ...  p=8 active=4; exact-theory rho* at T=90 -> 0.513
[2/3] confusability ...     Hamming@T=200: naive=2.59 whitened=0.00 greedy=0.27
[3/3] real FX ...           curl/grad=1.8e-31 ; K9: 28 curl dims vs 84 triangles
```

---

## 8. What each test proves

| Test | Guards |
|---|---|
| `test_boundary_of_boundary_is_zero` | $B_1B_2=0$ — the identity every result rests on |
| `test_hodge_decomposition_reconstructs_and_is_orthogonal` | grad ⊕ curl ⊕ harmonic is exact and mutually orthogonal |
| `test_gradient_is_curl_free_and_curl_is_divergence_free` | operator sanity ($\mathrm{curl}\circ\mathrm{grad}=0$, $\mathrm{div}\circ\mathrm{curl}=0$) |
| `test_curl_flow_lives_entirely_in_curl_component` | a $B_2 y$ flow has zero gradient/harmonic part |
| `test_harmonic_space_dimension_matches_betti_1` | $\dim\ker L_1 = b_1$ (filled tetrahedron 0; hollow $K_4$ 3) |
| `test_curl_dimension_ratio_on_complete_graph` | R5: $\mathrm{rank}(B_2)=\binom{n-1}{2}$ on $K_n$, fraction $3/n$ |
| `test_analytic_bayes_error_matches_gaussian_montecarlo` | R1 closed form vs direct Gaussian MC (±0.02) |
| `test_curl_detection_is_immune_to_gradient_and_harmonic_nuisance` | Lemma 1 end-to-end with 25× nuisance (±0.03) |
| `test_chernoff_is_error_exponent` | $-\log P_{\rm err}/T \downarrow C$ (within 10% at $T{=}640$) |
| `test_small_rho_chernoff_scaling` | R2: $C(\rho)/(\rho^2/16) \to 1$ (±5% at $\rho\le0.02$) |
| `test_exact_recovery_probability_matches_simulation` | R3 finite-sample law vs full pipeline (±0.06) |
| `test_whitened_detector_beats_naive_under_confusability` | R4: whitened ≪ naive, matches theory (±0.05) |
| `test_invisibility_floor_decreases_with_budget` | R2: $\rho^\star(4T)/\rho^\star(T) \approx 1/2$ in the small $\rho$ regime |

Philosophy: **a limits paper dies if a constant is wrong**, so every closed form
must beat a Monte-Carlo cross-examination before it is cited in the manuscript.

---

## 9. Real data: provenance and format

- Source: [frankfurter.app](https://www.frankfurter.app/) — a free, keyless mirror
  of the **ECB daily reference rates** (the same data the ECB publishes at 16:00 CET;
  no weekends/holidays, hence 257 samples spanning Dec 29, 2023 – Dec 31, 2024:
  the API backs the requested 2024-01-01 start off to the previous business day).
- File: `data/fx_rates.json` — essentially
  `{"base": "USD", "rates": {"2023-12-29": {"EUR": ..., ...}, ...}}` plus
  `amount`/`start_date`/`end_date` bookkeeping keys. The shipped file starts with a
  UTF-8 BOM (it was cached via PowerShell), which is why the loaders read it with
  the `utf-8-sig` codec — plain `json.load(open(...))` would choke on the BOM.
- Construction: currency $i$ gets log price $p_i = -\log(\text{rate}_i)$ in USD
  (base gets $p=0$); the edge flow on $(i,j)$ is $p_j - p_i$ via $B_1^\top p$
  — i.e., flows are *exactly* the market's log exchange rates re-expressed on the
  complete graph.
- Refetch/extend anytime with `python data/fetch_fx.py` (edit the date range/symbols
  in the URL).

Why FX is the *right* real dataset for a limits paper: covered-interest/triangular
**arbitrage-freeness means the true flow is a pure gradient** (rates derive from one
price vector), so theory predicts machine-zero curl — and that is precisely what we
measure ($10^{-31}$, i.e. floating-point residue). Real data here *instantiates the
converse*, not the achievability.

---

## 10. Relation to prior work

| Work | What it does | What it does *not* do |
|---|---|---|
| Barbarossa & Sardellitti 2020; Yang et al. 2022 (TSP filters); Schaub et al. 2020 | Builds the Hodge-Laplacian signal-processing toolbox | Assumes the filled-triangle set is **known** |
| Gurugubelli & Chepuri, EUSIPCO 2024 (sparse clique sampling, MAP); greedy topology learning (arXiv 2502.20159); sparse cell complexes (arXiv 2309.01632) | **Algorithms** that estimate the filled set from flows | No identifiability conditions, no sample-complexity or SNR thresholds, no converse |
| Marinucci et al. 2025 (topological adaptive LMS) | Online estimation over simplicial complexes, edge-sampling design | Complex structure assumed known; lists structure-discovery as future work |
| Hypergraph/simplicial SBM detectability (e.g. arXiv 2312.00708, 2108.06547) | Phase transitions for community detection when the **structure itself is observed** | Different observation model: our structure is *latent* and observed only through flow curls |
| **This project** | **Fundamental limits** for the latent-structure-from-flows problem: converse (R2), matching estimator (R4), geometry obstruction (R5) | — |

In short: prior TSP work asks *how* to infer the complex; this project proves *when*
it can and cannot be done, and hands back a threshold + estimator that the
algorithmic line can calibrate against.

---

## 11. Limitations and honesty notes

Stated plainly, because reviewers (and users) deserve to know:

1. **Gaussianity.** The converse and the exact finite-sample laws are proved under
   Gaussian signals/noise. The mechanism (curl annihilation + variance testing) is
   distribution-agnostic, but extending the constants to sub-Gaussian families is
   future work (stated in the paper).
2. **Known second-order parameters.** Detectors take $(\sigma_c, \sigma_n)$ as
   inputs. Plug-in estimation of these from data is straightforward but its effect
   on the thresholds is not yet quantified here.
3. **The real-data study demonstrates the converse, not recovery.** An efficient FX
   market is curl-free — there is genuinely nothing to recover, which is exactly the
   theory's point. Recovery/achievability is validated on controlled planted
   complexes (Figures 1–2), where ground truth exists.
4. **`sparse_curl_covariance_support`** (the lifted cvxpy estimator) is provided as
   a baseline but is dominated by the whitened detector in our benchmarks; it is not
   used for any headline claim.
5. **i.i.d. snapshots.** Temporal dependence (e.g. AR flows) shrinks the effective
   $T$; the theory applies with $T \to T_{\rm eff}$ but we do not characterize
   $T_{\rm eff}$ here.

---

## 12. Roadmap

- [ ] Sub-Gaussian converse (Fano with sub-Gaussian KL bounds).
- [ ] Partial edge observation: identifiability vs edge-sampling rate (connects to
      the simplicial-sampling literature).
- [ ] Plug-in variance estimation + adaptive thresholds.
- [ ] A recoverable real dataset (traffic flows on a planar road network — sparse
      triangles, favorable geometry) as an achievability companion to the FX converse.
- [ ] Camera-ready port from IEEEtran to ICASSP `spconf.sty`, 4+1 page format.

---

## 13. 中文速览

**问题**：拓扑信号处理需要知道网络中哪些三角形被"填充"（承载高阶相互作用），
现有文献只给出**估计算法**（贪婪、MAP、稀疏恢复），没人回答更根本的问题——
**给定 T 个边流快照和噪声水平，这个结构到底可不可辨识？**

**核心机制（旋度湮灭）**：对边流取旋度 $c_t=B_2^\top f_t$ 会把梯度分量和谐和分量
**精确消掉**（因为 $B_1B_2=0$ ，且谐和空间无旋），只剩下活跃三角形的信号加投影噪声。
所以隐结构只透过"旋度"这一扇窗可见。

**四个主要结果**：
1. **两方差检验**：孤立三角形的检测化为方差 $3\sigma_n^2$ vs $9\sigma_c^2+3\sigma_n^2$
   的高斯检验，最优误差指数是 Gaussian Chernoff 信息；
2. **curl-invisibility 相变**：curl 信噪比 $\rho<\rho^\star(T)\sim 1/\sqrt T$ 时
   **任何估计器都不可能恢复结构**（小 $\rho$ 极限下 $C\sim\rho^2/16$ ）；
3. **几何感知白化**：共边三角形互相"泄漏"能量，朴素能量检测会失败且 SNR 越高越糟；
   用 $\hat y=G^+c$ 白化后每个三角形获得有效信噪比
   $\rho^{\mathrm{eff}}_\tau=\sigma_c^2/(\sigma_n^2(G^+)_{\tau\tau})$ ，
   实测完胜朴素法并与理论精确吻合（偏差 ≤0.025）；
4. **秩障碍**：完全图 $K_n$ 上旋度子空间只有 $\binom{n-1}{2}$ 维，而候选三角形有
   $\binom n3$ 个——**无论信噪比多高**，可辨识比例恰好是 $3/n$ 。

**真实数据**：257 个交易日（2023-12-29 至 2024-12-31）的欧洲央行汇率
（9 种货币，完全图 $K_9$ ）。无套利市场的边流是纯梯度 → 旋度能量只有梯度的
$10^{-31}$ （机器零），且 $K_9$ 的可辨识比例 $28/84=1/3$ 正好落在 $3/n$ 曲线上——
理论预言的两种"不可见"在真实市场同时成立。

**复现**：`pytest -q`（16 个测试，空闲约 25 秒、高负载下最多约 90 秒，
每条定理都对照蒙特卡洛验证）；`python experiments/run_all.py`
（约 2–3 分钟重生成全部图表）；论文在 `paper/main.pdf`。全程只需 CPU。

---

## 14. Citation & license

If you use this code or the results, please cite the repository (paper citation to
follow upon publication):

```bibtex
@misc{topoflowlimits2026,
  title  = {When Are Triangles Invisible? Fundamental Limits of Higher-Order
            Structure Identifiability from Edge Flows},
  author = {appleweiping},
  year   = {2026},
  howpublished = {\url{https://github.com/appleweiping/topo-flow-limits}}
}
```

**License:** MIT (see `LICENSE`).
