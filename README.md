# topo-flow-limits

**Excitation-Dependent Identifiability of Latent Higher-Order Structure from
Edge Flows.**

A complete, reproducible signal-processing-theory research project (targeting the
IEEE ICASSP *Signal Processing Theory & Methods* track). Pure CPU — no GPU, no
`torch`. Every theorem in the paper is cross-checked against Monte-Carlo simulation
by the test suite before it is allowed into the manuscript.

> **TL;DR** — Topological signal processing needs to know *which triangles of
> a network are "filled"* (carry higher-order interactions). Existing papers
> give **algorithms** that estimate this from edge flows, or **detectors**
> that test known Hodge subspaces. This project answers the question both
> skip: **when is the latent structure identifiable at all — and what part
> of it?** The answer is *excitation-dependent*. Model the triangle signals
> as `y_S ~ N(0, Γ_S)`; then (main theorem — **three excitation regimes**,
> overlapping constraints on `Γ_S`, not a partition):
> **(a)** for positive-**diagonal** `Γ_S` (known or unknown) the weighted
> support is identifiable at **any** rank deficiency of `B₂` — because the
> lifted signatures `{b_τ b_τᵀ}` are *always* linearly independent (spark
> lemma: two edges lie in at most one triangle);
> **(b)** for **arbitrary PSD** `Γ_S`, the achievable set is
> `C_S = {M⪰0 : im M ⊆ im U_S}`, so the **population** covariance determines
> `M = Σ_z − σ_n²I` in full but pins the support only to
> `{S : im M ⊆ im U_S}`: the realized range `R = im M ⊆ im U_S` is always
> readable, yet the candidate image `im B₂,S` is **not** — even a
> **strictly-positive-diagonal but correlated** `Γ_S` defeats it (rank-one K4
> witness `Γ'=vvᵀ`, all `diag>0`, giving `u₀u₀ᵀ`); regime (a) additionally
> needs `Γ_S` *diagonal*;
> **(c)** at the **projector excitation** `Γ_S = σ_c²(B₂,Sᵀ B₂,S)⁺` the
> covariance equals `σ_n²I + σ_c²P_im` **exactly**, so equal-image supports
> are indistinguishable at every SNR and N.
> In regime (a): global analytic separations (9 unrestricted / 16
> equal-cardinality, proved for *every* clique complex via a Johnson-graph
> eigenvalue argument), worst-case exponent `(9/16)ρ₂²` = **exactly one
> isolated-triangle detection**, and a **lifted-covariance NNLS estimator**
> that is consistent with a fully explicit `O(1/N)` failure bound —
> validated on K4–K8 with random supports, Wilson CIs, and an
> α-interpolation experiment where equal-image distinguishability provably
> vanishes. On real data the same curl statistic **localizes genuine,
> unplanted vortices** (ERA5 tropical cyclones; positioned as *vortex
> localization*, not topology recovery), beating a classical baseline on all
> three independent external metrics, with moving-block bootstrap CIs.

---

## Table of contents

1. [The problem in plain language](#1-the-problem-in-plain-language)
2. [Mathematical setting (5-minute Hodge primer)](#2-mathematical-setting-5-minute-hodge-primer)
3. [The observation model](#3-the-observation-model)
4. [Main results](#4-main-results)
5. [The figures, explained](#5-the-figures-explained)
6. [Repository layout — every file explained](#6-repository-layout--every-file-explained)
7. [Installation and full reproduction](#7-installation-and-full-reproduction)
8. [What each test proves](#8-what-each-test-proves)
9. [Real data: provenance and format](#9-real-data-provenance-and-format)
10. [Relation to prior work](#10-relation-to-prior-work)
11. [Limitations and honesty notes](#11-limitations-and-honesty-notes)
12. [Supplement: Fano converses, partial sampling, plug-in estimation](#12-supplement-fano-converses-partial-sampling-plug-in-estimation)
13. [Roadmap](#13-roadmap)
14. [中文速览](#14-中文速览)
15. [Citation & license](#15-citation--license)

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

> Given `N` snapshots of edge flows with a given noise level, is the filled-triangle
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
3-cliques). We observe $N$ i.i.d. edge-flow snapshots

```math
f_t \;=\; \underbrace{B_1^{\top} a_t}_{\text{gradient nuisance}}
\;+\; \underbrace{B_{2,S}\, y_t}_{\text{signal (carries } S\text{)}}
\;+\; \underbrace{h_t}_{\text{harmonic nuisance}}
\;+\; \underbrace{n_t}_{\text{noise}}, \qquad t = 1,\dots,N,
```

(in the code the snapshot-count argument is named `T`; the paper and this README
use $N$ to avoid clashing with the candidate set $\mathcal{T}$)

with $a_t \sim \mathcal N(0,\sigma_g^2 I)$, triangle excitation
$y_t \sim \mathcal N(0,\Gamma_S)$ (one coordinate per *active* triangle;
the isotropic case $\Gamma_S=\sigma_c^2 I$ is regime (a)),
harmonic $h_t$, and $n_t \sim \mathcal N(0,\sigma_n^2 I)$. Only the
snapshots $f_t$ are observed; the task is to recover $S$. Prior generative
models are special cases of this general `Γ_S`: e.g. Gurugubelli & Chepuri
(EUSIPCO 2024) draw the triangle signal *smooth w.r.t. the Hodge Laplacian*,
`y ~ N(0, L₂⁺)` with `L₂ = B₂ᵀB₂` — which is exactly the **projector
excitation** (regime (c)), the second-order-**unidentifiable** case;
their MAP works only through its sparsity prior. The three regimes say when the
latent structure is recoverable at all. The harmonic nuisance `h_t` is taken
**candidate-orthogonal**, `h_t ∈ ker B₂ᵀ` (so it is annihilated by the curl
projection; the gradient part is annihilated unconditionally since
`B₂ᵀB₁ᵀ = 0`). Implementation: `src/tfl/generative.py` (`sample_flows`,
`FlowParams`, planted-complex builders).

The single most important structural fact (the curl-annihilation identity,
main paper §2):

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
The energy $E=\sum_t c_{\tau,t}^2$ is sufficient, $E/v_i \sim \chi^2_N$, and the
minimum Bayes error has an exact closed form at every finite $N$ (implemented in
`two_variance_bayes_error`) and decays like $e^{-N C(v_0,v_1)}$ where
$C$ is the **Gaussian Chernoff information** (`gaussian_chernoff_information`).

### R2 — The curl-invisibility sample-complexity threshold

As $\rho \to 0$, $C(\rho) = \rho^2/16 + o(\rho^2)$ (validated to 5% accuracy at
$\rho \le 0.02$). Hence for a target error $\delta$ and budget $N$, detection
requires

```math
\rho \;\ge\; \rho^\star(N) \;=\; \Theta\!\Big(\sqrt{\tfrac{\log(1/\delta)}{N}}\Big),
```

and **below that scale no estimator attains the target error exponent** — the
triangles become *information-theoretically invisible* at the exponent level.
Calibration note (important): `invisibility_curl_snr_floor` solves the exponent
equation $e^{-NC(\rho^\star)}=\delta$, which is the Chernoff (achievability-side)
calibration of the boundary; the exact boundary at finite $N$ sits an order-one
factor below it (the exact Bayes error can meet $\delta$ at $\rho$ roughly
40–50% smaller at moderate $N$), while both share the $1/\sqrt N$ law.
Numerically ($\delta=0.05$): $N{=}25 \to \rho^\star{=}3.07$,
$N{=}100 \to 1.00$, $N{=}400 \to 0.41$, $N{=}1600 \to 0.19$.

### R3 — Exact finite-sample recovery law (edge-disjoint case)

For edge-disjoint candidates the curl statistics are genuinely **independent**
across triangles (disjoint column supports, white noise), so with the Bayes-optimal
threshold, exact support recovery has probability — exactly, not approximately —
$\prod_{\tau \in S}(1-P_{\rm miss}) \prod_{\tau \notin S}(1-P_{\rm fa})$
with $\chi^2_N$ tail expressions (`exact_recovery_probability`). Its 50% contour is
the phase boundary in Figure 1, and the **empirical contour matches it to a median
ratio of 1.01** across the whole grid.

### R4 — Geometry-aware decorrelation (GLS/BLUE) beats naive detection under confusability

Edge-sharing triangles have correlated curl signatures: the triangle Gram matrix
$G = B_2^{\top} B_2$ has $G_{\sigma\tau} = \pm 1$ when $\sigma,\tau$ share an edge.
An active triangle then *leaks* energy onto its inactive neighbours, so naive
curl-energy thresholding fails — **and gets worse as SNR grows** (more signal, more
leakage). The fix: when $B_2$ has full column rank ($G$ invertible), whiten:

```math
\hat y_t = G^{-1} c_t \;\Rightarrow\;
\hat y_{\tau,t} = y_{\tau,t}\,\mathbf{1}\{\tau\in S\} + e_{\tau,t},\qquad
\mathrm{Cov}(e_t) = \sigma_n^2 G^{-1},
```

which holds **exactly**, so the *marginal* law of every coordinate is a
two-variance test with **its own effective SNR**

```math
\rho^{\mathrm{eff}}_\tau = \frac{\sigma_c^2}{\sigma_n^2 (G^{-1})_{\tau\tau}}.
```

$`(G^{-1})_{\tau\tau}`$ is the *price of geometry*: it equals $`1/3`$ for an
isolated triangle (recovering R1 exactly) and grows as signatures overlap. Two
careful points about the **joint** recovery probability: the whitened noise
coordinates remain correlated (off-diagonal entries of $`G^{-1}`$), so the joint
law does **not** factorize in general. What is rigorous under arbitrary
correlations is the **union bound** `heterogeneous_recovery_union_bound`:
$`P(\hat S = S) \ge 1 - \sum_\tau P_{\mathrm{err},\tau}`$ over the exact marginal
error probabilities. The product of marginals
(`heterogeneous_exact_recovery_probability`) is an **independence approximation**
— exact in the edge-disjoint case; on the diagonally-dominant strip benchmark it
tracks Monte-Carlo within 0.03 (≈1.5 MC standard errors at 600 trials) across
the whole transition, while the union bound is a strict lower bound that is
visibly conservative in the transition region, as a bound should be. On the
edge-sharing strip at $\rho{=}8,\ N{=}200$: naive Hamming error **2.60**
(never recovers), greedy baseline 0.28, **whitened 0.00**.

### R5 — Excitation-dependent identifiability: three excitation regimes (main theorem)

**The 2026-07 major revision replaced the unscoped "random signals remove the
rank obstruction" claim with three excitation regimes** (overlapping
constraints on `Γ_S`, not a partition; `y_S ~ N(0, Γ_S)`, verified in
`tests/test_excitation.py`). All statements are about the **population**
covariance `Σ_z` (the `N→∞` limit; finite-`N` estimation is R4):

| Excitation regime | What the population covariance identifies | Mechanism |
|---|---|---|
| `Γ_S` positive **diagonal** (known or unknown) | the **weighted support** `(S, γ)` — at any rank deficiency | lifted spark: atoms `{b_τ b_τᵀ}` always LI; weights read off the covariance. On `K_n` with σ_n *unknown*: exactly one ambiguous direction, since `Σ_τ u_τu_τᵀ = n·I_r` |
| `Γ_S` **arbitrary PSD** | the achievable set `C_S = {M⪰0 : im M ⊆ im U_S}` ⇒ the support **only** up to `{S : im M ⊆ im U_S}` | `M = Σ_z − σ_n²I` is fully determined; its range `R = im M ⊆ im U_S` is always readable, but the candidate image `im B₂,S` is **not**: a larger-image `S'` with a suitable `Γ` gives the same `M`. Even a **strictly-positive-diagonal but correlated** `Γ'=vvᵀ` (K4, `v=e₀−¼c`, `c∈ker U_S`, all `diag>0`) yields `U_S Γ' U_Sᵀ = u₀u₀ᵀ` — identical to `{τ₀}` though images are 3 vs 1. Positive variance isn't enough; regime (a) needs `Γ` **diagonal**. (`Γ_S ≻ 0` does recover `R = im U_S`.) |
| `Γ_S = σ_c²(B₂,Sᵀ B₂,S)⁺` (**projector**) | equal-image supports **indistinguishable** at every SNR/N | `B_S G_S⁺ B_Sᵀ = P_im` exactly ⇒ identical covariances. **This is exactly the Hodge-smoothness prior `Γ=L₂⁺` used by Gurugubelli–Chepuri 2024** — second-order-unidentifiable; works only via their sparsity prior |

The α-interpolation `Γ_α = (1−α)I + α(B_SᵀB_S)⁺` connects (a) to (c)
continuously: the equal-image covariance gap shrinks to zero as α→1, and
thresholded NNLS recovery of the specific `S` (declared `ρ₂/2` rule — the
diagonal model is deliberately misspecified for α>0) collapses **before**
the gap does (`run_second_order.py`, panel C: 1.00 → 0.82 → 0.00 by
α=0.75); only the α=1 endpoint is threshold-free — identical covariances,
no estimator beats chance `1/m` (m equal-image supports, identical distributions under a uniform prior; the plotted 0 is recovery of the *specific* S, itself ≤ 1/m).

**Achievability (class (a)): the lifted-covariance NNLS estimator**
(`nnls_lifted_support`): `ŵ = argmin_{w≥0} ‖Σ̂_z − σ_n²I − Σ w_τ u_τu_τᵀ‖_F²`,
threshold `w_min/2`. Consistent, with the fully explicit failure bound
`P(Ŝ≠S) ≤ 16[(trΣ)² + ‖Σ‖_F²]/(N σ_min(A)² w_min²)` — every constant
derived (cone-LS perturbation + exact Wishart moment + Markov) and
numerically checked; conservative by design (Markov), the empirical
transition is ~an order of magnitude earlier. The per-cell bound is the
**worst case over the trial supports** (not one representative), so it upper-bounds
every trial. Validated on K4–K8, random supports, full
(N, ρ₂) grid, 200 trials/cell, 95% Wilson CIs, vs an *oracle-aided*
matched-subspace baseline (population scores tie on dependent candidates —
its finite-N tie-breaking rides on eigen-anisotropy and dies under projector
excitation) and a *generic* greedy atom-fitting heuristic (not a tuned
literature baseline; stalls where atoms overlap: 0.43 on
K5 at N=400 vs NNLS 1.00). See `results/second_order.json`.

#### The price within class (a) — corrected constants (formerly "R5")

What happens when curl signatures are linearly dependent (singular Gram $G$ —
e.g. all $\binom n3$ triangles of $K_n$)? The folklore answer — "supports are
identifiable only modulo $\ker B_2$, at any SNR and any $N$" — turns out to be
**true at first order and false at second order**, and an earlier revision of
this very project stated it in the false general form ([honesty note 7](#11-limitations-and-honesty-notes)).
The correct statement is excitation-dependent --- the deterministic-vs-random
contrast below, subsumed by the paper's Theorem 1 (three excitation regimes) as
the boundary-of-(b) and regime-(a) cases:

- **First order (impossibility).** Under *unknown deterministic* triangle
  signals, supports with $\mathrm{im}~B_{2,S} = \mathrm{im}~B_{2,S'}$ induce
  identical families of flow distributions — indistinguishable at any SNR/$N$.
  Such confusers exist whenever the candidate set contains all four faces of a
  **tetrahedron** ($b_{012}-b_{013}+b_{023}-b_{123}=0$). On $K_n$:
  $\mathrm{rank}(B_2)=\binom{n-1}{2}$ against $\binom n3$ candidates — the
  **$3/n$ DoF ratio** (verified for $K_5,\dots,K_{12}$).
- **Second order (identifiability).** Under the random-signal model the flow
  covariance carries $\sigma_c^2\sum_{\tau\in S} b_\tau b_\tau^\top$ — strictly
  finer than the column space. The **lifted spark lemma** (`lifted_atoms_linearly_independent`):
  a pair of distinct edges lies in at most one common triangle, so the atoms
  $\{b_\tau b_\tau^\top\}$ are *always* linearly independent — the covariance
  map $S \mapsto \Sigma(S)$ is **injective**, and every support is identifiable
  from i.i.d. snapshots **at any rank deficiency** (exhaustively verified over
  all $2^4$ supports of $K_4$; the achiever is the lifted-covariance
  **NNLS** estimator `nnls_lifted_support`, which recovers exact supports on
  rank-deficient $K_5$–$K_8$).
- **The price (sample complexity).** Telling a confuser pair apart costs
  $N^\star \sim \log(1/\delta)/C_G$ snapshots, $C_G$ = the Gaussian Chernoff
  information between the two curl-domain covariances. The tetrahedron hosts
  two kinds of equal-image confusers, with exact separations (both verified
  **exhaustively over all $2^{10}$ supports of $K_5$**, 45 000+ equal-image
  pairs — `test_exhaustive_k5_confuser_separations`):
  the face **swap** with $\|u_a u_a^\top - u_b u_b^\top\|_F^2 = 9+9-2 = 16$
  (the minimum among supports of **equal cardinality**; independent of $n$,
  $|S|$, and the hosting tetrahedron), and the **subset** confuser $S$ vs
  $S\cup\{\text{4th face}\}$ with $\|u_4 u_4^\top\|_F^2 = 9$ — the
  **unrestricted minimum** (an earlier revision wrongly claimed 16 here; see
  honesty note 7). With the **dimensionless** signature Gram
  $D_S=\sum_{\tau\in S} u_\tau u_\tau^\top$ (so $M_S=\sigma_c^2 D_S$), the
  Bhattacharyya expansion
  $C_G = (\rho_2^2/16)\|\Delta D\|_F^2\,(1+o(1)) = \|\Delta M\|_F^2/(16\sigma_n^4)\,(1+o(1))$
  (the two coincide; do **not** write $\rho_2^2\|\Delta M\|^2$, which
  double-counts $\sigma_c^4$) gives exponents $\rho_2^2$
  (swap; numerically checked within $6\times10^{-4}$ of 1 at $\rho_2=10^{-4}$)
  and $(9/16)\rho_2^2$ (subset) — and
  $(9/16)\rho_2^2$ **equals the isolated-triangle detection exponent**
  $C(\rho)=\rho^2/16$ at $\rho=3\rho_2$ (numerically checked to $3\times10^{-5}$
  relative at $\rho_2=10^{-5}$ —
  `test_subset_confuser_exponent_equals_isolated_triangle_exponent`).
  So the hardest equal-image decision costs, at the exponent level, exactly
  one isolated-triangle detection: **rank deficiency is free at the exponent
  level**; only the Fano log-multiplicity over the $4\binom n4$ tetrahedral
  hypotheses ($\asymp\log n$) reflects the geometry.

Implementation: `curl_domain_signatures`, `second_order_covariance`,
`matrix_gaussian_chernoff`, `candidate_tetrahedra`,
`equal_image_single_swap_pairs`, `confuser_pair_chernoff`,
`second_order_min_snapshots`, `confuser_family_fano_min_snapshots`,
plus the excitation-regimes/NNLS section (`excitation_covariance`,
`projector_excitation_gamma`, `interpolated_excitation_gamma`,
`lifted_atom_matrix`, `nnls_recovery_bound`, `share_edge_adjacency`) in
`src/tfl/limits.py`; all constants Monte-Carlo-validated in
`tests/test_second_order.py`. Sharp *first-order* sparse-support thresholds
over $\ker B_2$ remain open (paper §6, open problems).

### R6 — Curl-based vortex localization on real data (cyclones), plus real-geometry checks

**The flagship real-data result (Figure 2): curl-based vortex localization** ---
a genuine, unplanted higher-order signal localized in nature (a
localization/ranking task, deliberately *not* claimed as filled-triangle
topology recovery). ERA5 10m winds over the Western North Pacific
(Aug–Sep 2020, 13 IBTrACS storms incl. Bavi, Maysak, Haishen) become edge
flows on a triangulated mesh; by Stokes' theorem a triangle's curl is its
circulation (area-integrated vorticity), so high-curl triangles are real
atmospheric vortices. The centered curl-energy detector — no oracle
parameters, nothing planted — is validated against **two references**: a
*same-field consistency reference* — full-resolution finite-difference
vorticity, a different functional of the same winds (internal, quantitative:
**AUC 0.914**, Spearman 0.77) — and the **genuinely independent** IBTrACS
best-track cyclone archive (external, agency-verified:
**AUC 0.920**, PR-AUC 0.494 at 1.8 % prevalence, precision@k 0.48, with
moving-block bootstrap 95 % CIs in the figure and JSON). The mesh is simply connected, so $B_2$ has
full column rank by Euler's formula — the class-(a) favorable-geometry
favorable-geometry regime (a) of the excitation hierarchy, with no confusers. Detection degrades as the snapshot budget
shrinks (uniformly centered statistic, $N \ge 3$); the budget panel carries no
theory-floor overlay --- an arbitrary-units $\rho^\star(N)$ curve was removed
in the 2026-07 revision, and we make no quantitative $1/\sqrt N$-tracking
claim on real weather. Against a classical baseline
(pointwise vorticity from the same 2.1° mesh-node winds), the edge-flow
statistic wins on the independent external reference (0.920 vs 0.898) and
is slightly below on the internal one (0.914 vs 0.948), which shares its
functional with the baseline.

**Real-geometry checks (repo figure `real_traffic.png`):** on three standard TNTP traffic networks
the street geometry contains few 3-cliques and $B_2$ has **full column rank**
in all three — curl DoF ratio exactly **1**; real user-equilibrium flows carry
genuine curl energy (2.4% on Sioux Falls, 4.6% on Anaheim), unlike FX's
machine zero. Planted-recovery panels: Anaheim ($G=3I$ — an exact-product-law
check, honestly labeled as synthetic-in-substance since centering removes the
real background exactly) and EMA (non-diagonal $G$ — the panel that exercises
the geometry-aware marginal laws of R4 on real geometry).

---

## 5. The figures, explained

All are regenerated by `experiments/run_all.py` (~30–45 min for all 9
figures, CPU) into `results/figures/`; the copies in `paper/figures/` used by
the manuscript are a manual copy of these outputs (kept in sync whenever
figures change). **Paper figures** (2, both full-width): Fig. 1 = `second_order.png`,
Fig. 2 = `real_cyclone.png`. The phase-transition, confusability, traffic, and
FX figures are repo-only (the supplement's own figures are the Fano,
partial-sampling, and plug-in panels); the old phase-transition main-text
branch was cut in the 2026-07 method revision to make room for rigorous
definitions.

### Paper Figure 1 — `second_order.png` (achievability + α-interpolation)

**(A)** Exact-recovery vs $N$ on $K_4$–$K_8$ with per-trial *random*
supports at $\rho_2=1$: the lifted-covariance **NNLS** estimator (solid,
95% Wilson bands) reaches probability 1 by $N=400$ at *every* rank
deficiency (incl. $21/56$ on $K_8$) — the only one of the **three evaluated implementations** to do so (not tuned literature methods): the
*oracle-aided* matched-subspace baseline (dashed) can lead at small $N$
but collapses on dependent candidates ($0.00$ at $K_8$), and the greedy
atom fitter (dotted) stalls at 0.43 on $K_5$. **(B)** NNLS vs $\rho_2$ at $N=200$ across rank
deficiency — the empirical face of regime (a). **(C)** Under
$\Gamma_\alpha=(1-\alpha)I+\alpha(B_S^\top B_S)^+$ on the $K_5$ tetrad, the
analytic equal-image covariance gap closes only at $\alpha=1$, while
thresholded NNLS recovery collapses *earlier* (1.00 → 0.82 → 0.00, zero
from $\alpha=0.75$) — regime (c), and the price of thresholding,
made visible. Produced by
`experiments/run_second_order.py`, all cells (with the derived bound) in
`results/second_order.json`.

### Repo figure — `phase_transition.png` (recovery threshold; not in the 4-page paper)

Heatmap of the empirical probability of *exact* filled-triangle recovery on an
edge-disjoint planted complex (8 candidates, 4 active, gradient+harmonic nuisance
present), sweeping snapshots $N \in [5, 90]$ vs curl-SNR $\rho \in [0.3, 30]$,
200 trials per cell. **Solid white line** = exact finite-sample theory contour (R3);
it lies right on the empirical 50% boundary (median ratio 1.01, all within ±25%).
**Dotted white line** = asymptotic Chernoff+Bonferroni floor (R2), visibly the
$1/\sqrt N$ law and conservative, as an asymptotic bound should be.
Produced by `experiments/run_phase_transition.py`, metrics in
`results/phase_transition.json`.

### Repo figure — `confusability.png` (why geometry matters)

Left: mean Hamming error vs $N$ on a 9-triangle *edge-sharing strip* with
alternating active triangles at $\rho = 8$ (here $B_2$ has full column rank, so
R4 applies with $G^{-1}$). The naive curl-energy detector plateaus at ≈2.1–2.6
wrong triangles and **worsens with more data** (leakage is systematic, not noise);
greedy stalls ≈0.3; the **whitened detector goes to 0**.
Right: whitened detector's empirical exact-recovery probability across the full
S-curve (N from 3 to 200, 600 trials) vs the two theoretical curves — the
independence approximation (dashed) tracks it within 0.03, and the rigorous
union bound (dotted) lower-bounds it, conservatively in the transition region.
Produced by `experiments/run_confusability.py`, metrics in
`results/confusability.json`.

### Paper Figure 2 — `real_cyclone.png` (curl-based VORTEX LOCALIZATION)

**Task framing (2026-07 revision): this is curl-based *vortex
localization*, not filled-triangle topology recovery** — the mesh has no
equal-image confusers and real weather carries no latent boolean support;
the experiment grounds the paper's sufficient statistic on genuine,
unplanted rotational structure. Nothing planted, no oracle parameters
(temporally centered curl-energy scores; area²-normalization puts triangles
across 0–45°N on a common mean-vorticity scale). **(A)** Scores on the
triangulated Western-North-Pacific mesh (836/2389/1554, full column rank by
Euler) for one 4-day window, with independent IBTrACS fixes circled in red.
**(B)** ROC pooled over all 15 windows against two references — internal
*same-field consistency* (finite-difference vorticity, a different
functional of the same winds at 3× finer resolution): **AUC 0.914**
[moving-block bootstrap 95% CI 0.894–0.936]; external, genuinely independent
IBTrACS: **AUC 0.920** [0.879–0.951], **PR-AUC 0.494** [0.273–0.666] at
prevalence 1.8%, P@k 0.483, Spearman(score, |ζ|) = 0.77. Baseline (coarse
pointwise vorticity, same information budget): external AUC 0.898, PR-AUC
0.360, P@k 0.389 — **ours wins on all three independent external metrics**;
internal 0.948 vs ours 0.914 (it shares the reference's functional).
**(C)** Detection quality vs snapshot budget $N$ (mean ± sd over 12
subsample draws; uniformly centered, $N\ge3$; the arbitrary-units theory
floor overlay was removed in the 2026-07 revision). Internal-threshold
sensitivity: AUC 0.876→0.959 across $10^{-5}$–$5\times10^{-5}$ s⁻¹ (JSON).
The GLS-decorrelated score ranks worse here (AUC 0.77 int / 0.81 ext;
$G^{-1}$ amplifies noise on a near-regular mesh — in the JSON). Produced by
`experiments/run_real_cyclone.py`, metrics in `results/real_cyclone.json`.

### Repo figure — `real_traffic.png` (recovery laws on real road geometry)

**(A)** Curl DoF ratio for three real TNTP road networks — Sioux Falls
(24 nodes / 38 edges / 2 triangles), Eastern-Massachusetts (74/129/33), Anaheim
(416/634/54) — all have **full-column-rank $B_2$**, i.e. ratio 1 (green), against
$K_9$'s $28/84$ (red). Sioux Falls and Anaheim triangles are pairwise
edge-disjoint ($G=3I$); EMA has 10 edge-sharing pairs yet stays full rank.
**(B)** Planted recovery on Anaheim. Honest framing (this changed after
adversarial self-review): because $G=3I$, this panel is a check of the
edge-disjoint product law (R3) on real-derived geometry — the real
constant-background equilibrium flow is removed *exactly* by centering
(df $N-1$) and the gradient/harmonic nuisances are annihilated by the curl
map, so the surviving statistical problem is synthetic. The genuine real-data
recovery evidence is Figure 2.
**(C)** Planted recovery on EMA — the panel that actually exercises R4's
geometry-aware machinery on real geometry: 10 edge-sharing pairs make $G$
non-diagonal and the per-triangle laws heterogeneous
($(G^{-1})_{\tau\tau} \in [0.333, 0.436]$); empirical recovery tracks the
independence approximation and respects the rigorous union bound.
Produced by `experiments/run_real_traffic.py`, metrics in
`results/real_traffic.json`.

### Repo figure — `real_fx.png` (FX: machine-precision consistency check)

257 trading days (Dec 29, 2023 – Dec 31, 2024) of ECB daily reference rates,
9 currencies → $K_9$, 36 edges, 84 candidate triangles. **Honest framing
(changed after adversarial self-review):** the FX edge flow is *constructed*
as $f = B_1^\top p$ from a single daily log-price vector, so
$B_2^\top f = (B_1 B_2)^\top p = 0$ **identically, by construction** — the
measured curl/gradient ratio of $1.8\times10^{-31}$ is floating-point residue
confirming the curl-annihilation arithmetic on real market data, *not* an
empirical discovery that markets are arbitrage-free (a genuine test of that
would need independent per-pair cross-rate quotes, whose triangular
inconsistencies would carry real curl). **(B)** Curl DoF ratio
$\mathrm{rank}(B_2)/\binom{n}{3}$ for $K_5,\dots,K_{12}$: numeric rank exactly
on the $3/n$ curve; $K_9$ gives $28/84 = 1/3$. Produced by
`experiments/run_real_fx.py`, metrics in `results/real_fx.json`.

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
│   ├── tntp.py               loader for TNTP road networks (net + equilibrium
│   │                         flow files) -> TrafficNetwork (Complex + real flow)
│   ├── geo.py                gridded wind fields -> mesh edge flows:
│   │                         triangular_mesh (regional right-triangulated mesh,
│   │                         full-column-rank B2 by Euler), wind_edge_flows
│   │                         (trapezoidal line integrals), grid_vorticity
│   │                         (full-res internal ground truth), IBTrACS loader +
│   │                         cyclone_triangle_labels (external ground truth)
│   ├── estimators.py         curl_statistics (sufficient statistics);
│   │                         energy_detector_bayes_support (naive, Chernoff-optimal
│   │                         threshold); whitened_curl_scores +
│   │                         whitened_curl_detector_support (proposed, geometry-aware,
│   │                         'bayes'/'fwer' threshold modes; center=True removes any
│   │                         constant background flow, thresholds at df=N-1);
│   │                         effective_curl_snr;
│   │                         estimate_noise_sigma / estimate_curl_sigma /
│   │                         adaptive_whitened_detector_support (supplement S3
│   │                         plug-in pipeline, with one refinement pass);
│   │                         sparse_curl_covariance_support (cvxpy lifted lasso);
│   │                         greedy_support (baseline); hamming_error/exact_recovery
│   └── limits.py             the theory in closed form: curl_variances;
│                             gaussian_chernoff_information; two_variance_bayes_error
│                             (+ optimal threshold); invisibility_curl_snr_floor
│                             (bisection on the Chernoff rate);
│                             exact_recovery_probability & recovery_contour_rho
│                             (finite-sample, edge-disjoint); whitened_variances,
│                             heterogeneous_exact_recovery_probability (independence
│                             approximation) & heterogeneous_recovery_union_bound
│                             (rigorous bound, correlation-robust);
│                             fano_min_snapshots / signal_agnostic_fano_min_snapshots
│                             / fano_rho_floor (supplement S1 joint-recovery
│                             converses); median_sigma_envelope (supplement S3);
│                             per_triangle_threshold (bayes | fwer/Bonferroni);
│                             EXCITATION TRICHOTOMY section (main thm):
│                             curl_domain_signatures, second_order_covariance,
│                             lifted_atoms_linearly_independent (spark lemma),
│                             matrix_gaussian_chernoff, candidate_tetrahedra,
│                             equal_image_single_swap_pairs,
│                             confuser_pair_chernoff, second_order_min_snapshots,
│                             confuser_family_fano_min_snapshots;
│                             excitation_covariance, projector_excitation_gamma,
│                             interpolated_excitation_gamma, lifted_atom_matrix,
│                             nnls_recovery_bound, share_edge_adjacency
│
├── experiments/
│   ├── _util.py              results paths; Agg (headless) matplotlib; JSON/figure
│   │                         savers; FastFlowSampler (pre-factorized fast sampling)
│   ├── run_second_order.py       PAPER FIG 1: NNLS achievability grid
│   │                         (K4-K8, random supports, Wilson CIs) + alpha
│   │                         sweep + second_order.json (~15-25 min)
│   ├── run_phase_transition.py   repo figure + phase_transition.json (out of
│   │                             the 4-page paper; supplement S1 references it)
│   ├── run_confusability.py      repo figure + confusability.json (confuser
│   │                             exponents; quoted as text in the paper)
│   ├── run_real_cyclone.py       PAPER FIG 3: curl-based vortex localization
│   │                             + real_cyclone.json
│   ├── run_real_traffic.py       repo figure + real_traffic.json (out of the
│   │                             4-page paper; A geometry,
│   │                             B Anaheim G=3I, C EMA non-diagonal G)
│   ├── run_real_fx.py            repo figure + real_fx.json (consistency check;
│   │                             one sentence in the paper; figure repo-only)
│   ├── run_fano.py               supplement S1 figure + fano.json
│   ├── run_partial_sampling.py   supplement S2 figure + partial_sampling.json
│   ├── run_plugin.py             supplement S3 figure + plugin.json
│   └── run_all.py                one-click: all of the above from fixed seeds
│
├── tests/                    51 tests — the theorems' guardrails (see §8)
│   ├── test_hodge.py
│   ├── test_theory_vs_sim.py
│   ├── test_excitation.py    the three excitation regimes + NNLS guardrails (15 tests:
│   │                         diagonal/PSD/projector regimes, Kn ambiguity,
│   │                         separation identity + Johnson bound, NNLS
│   │                         consistency + failure-bound validity (incl.
│   │                         non-vacuous cells), subspace
│   │                         tie/collapse, alpha-interpolation)
│   ├── test_geo.py           the wind-to-edge-flow bridge (Euler full rank;
│   │                         synthetic Rankine vortex localizes with correct
│   │                         sign; matches finite-difference vorticity ranking)
│   └── test_second_order.py  second-order guardrails, regime (a)
│                             (spark lemma,
│                             exhaustive K4 injectivity, deterministic confuser
│                             replication, C_G constants vs MC, rank-deficient
│                             recovery, Fano consistency)
│
├── data/
│   ├── fx_rates.json         cached real ECB rates (257 days) — offline repro
│   ├── fetch_fx.py           refetch/extend the FX dataset (no API key needed)
│   ├── traffic/*.tntp        vendored TNTP road networks + equilibrium flows
│   ├── fetch_traffic.py      refetch the TNTP files
│   ├── era5_wnp_2020.npz     cached ERA5 10m winds, W. North Pacific Aug-Sep
│   │                         2020, 0.7 deg / 6-hourly (from ARCO-ERA5 on GCS)
│   ├── fetch_era5.py         refetch via xarray/zarr (sequential)
│   ├── fetch_era5_fast.py    refetch via concurrent chunk reads (fast on
│   │                         high-latency links); identical output
│   ├── probe_era5.py         inspect the ARCO-ERA5 store layout first
│   ├── ibtracs_wp_2020.csv   cached IBTrACS WP best tracks, Aug-Sep 2020
│   └── fetch_ibtracs.py      refetch from NOAA NCEI (v04r01)
│
├── results/                  regenerated by run_all.py
│   ├── figures/              9 PNGs: 2 main-paper (second_order,
│   │                         real_cyclone) + 3 supplement (fano_bounds,
│   │                         partial_sampling, plugin) + 4 repo-only
│   │                         (phase_transition, confusability, real_traffic, real_fx)
│   └── *.json                all numbers quoted in the paper + supplement
│
└── paper/
    ├── main.tex              the manuscript (official ICASSP spconf format)
    ├── spconf.sty, IEEEbib.bst   official ICASSP paper-kit style files
    ├── refs.bib              bibliography (28 entries, 22 cited in the draft)
    ├── main.pdf              compiled draft (4 pages technical content +
    │                         page 5 references & compliance statements)
    ├── supplement.tex/.pdf   repository supplement (§12): Fano converses, excitation-regimes/separation/NNLS proofs (S4), vortex-localization methods (S5),
    │                         partial edge sampling, plug-in estimation
    └── figures/              figure copies used by the manuscript + supplement
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

# 1) validate all theory against Monte-Carlo (51 tests; ~3-5 min)
.\.venv\Scripts\python.exe -m pytest -q

# 2) regenerate every figure and metric (~30-45 min, 8 experiments)
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

**Refetch the real datasets** (optional; the repo ships the cached files):

```bash
python data/fetch_fx.py        # ECB FX rates
python data/fetch_traffic.py   # TNTP road networks + equilibrium flows

# the ERA5/IBTrACS refetch additionally needs:
#   uv pip install gcsfs numcodecs requests   (xarray+zarr only for fetch_era5.py)
python data/fetch_era5_fast.py # ERA5 winds (concurrent chunk reads from GCS)
python data/fetch_ibtracs.py   # IBTrACS WP best tracks (NOAA NCEI)
```

Expected `run_all.py` output (seeds are fixed; numbers reproduce exactly):

```
[1/8] phase transition ...  p=8 active=4; exact-theory rho* at N=90 -> 0.513
[2/8] confusability ...     Hamming@N=200: naive=2.60 whitened=0.00 greedy=0.28
[3/8] real FX ...           curl/grad=1.8e-31 ; K9: 28 curl dims vs 84 triangle coefficients
[4/8] real traffic ...      Anaheim DoF 54/54=1.0 ; recovery@N=95: emp=1.00 theory=1.00
[5/8] Fano curves ...       floors at N=2000: chernoff=0.167 fano(p=1e4)=0.089
[6/8] sampling + plug-in .. P(exact) emp=0.53 theory=0.56 at q=0.99 ; plugin max gap=0.057
[7/8] second-order  ...     K8 (rank 21/56) NNLS=1.00 @ N=400; alpha sweep 1.00 -> 0.00
[8/8] vortex localization . INTERNAL AUC = 0.914 ; EXTERNAL AUC = 0.920, PR = 0.494, P@k = 0.483
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
| `test_curl_dimension_ratio_on_complete_graph` | R5: $\mathrm{rank}(B_2)=\binom{n-1}{2}$ on $K_n$ ($K_5,\dots,K_{12}$), DoF ratio $3/n$ |
| `test_analytic_bayes_error_matches_gaussian_montecarlo` | R1 closed form vs direct Gaussian MC (±0.02) |
| `test_curl_detection_is_immune_to_gradient_and_harmonic_nuisance` | curl annihilation (main §2) end-to-end with 25× nuisance (±0.03) |
| `test_chernoff_is_error_exponent` | $-\log P_{\rm err}/N \downarrow C$ (within 10% at $N{=}640$) |
| `test_small_rho_chernoff_scaling` | R2: $C(\rho)/(\rho^2/16) \to 1$ (±5% at $\rho\le0.02$) |
| `test_exact_recovery_probability_matches_simulation` | R3 finite-sample law vs full pipeline (±0.06) |
| `test_whitened_detector_beats_naive_under_confusability` | R4: whitened ≪ naive, matches marginal theory (±0.05) |
| `test_union_bound_is_rigorous_lower_bound` | R4: union bound ≤ independence approx algebraically, and ≤ empirical rate under correlated noise |
| `test_road_network_geometry_and_centered_recovery` | R6: TNTP loader integrity; full-rank $B_2$ (DoF ratio 1) on all three road networks; $G=3I$ where claimed; centered detector matches the df $N-1$ law under a strong constant background |
| `test_fano_bounds_are_valid_converses` | S1: both Fano bounds lower-bound the empirical Fig.-1 recovery budget; monotone in $\rho$; sparse/dense regime ordering |
| `test_partial_sampling_closed_form_matches_simulation` | S2: the $q^{3k}$ closed form vs Monte-Carlo (±0.07) |
| `test_median_sigma_envelope_and_plugin_consistency` | S3: median estimate inside the DKW / $\chi^2$ envelope (≥93% at 95% nominal); adaptive ≈ known detector (±0.08) |
| `test_invisibility_floor_decreases_with_budget` | R2: $\rho^\star(4N)/\rho^\star(N) \approx 1/2$ in the small $\rho$ regime |
| `test_lifted_atoms_independent_on_standard_complexes` | R5 spark lemma: atoms $\{b_\tau b_\tau^\top\}$ linearly independent on $K_4,K_6,K_9$, strips, disjoint unions, random clique complexes |
| `test_second_order_covariance_map_is_injective_on_k4_exhaustively` | R5: all $2^4$ supports of $K_4$ have pairwise-distinct covariances despite rank$(B_2)=3<p$ |
| `test_tetrahedral_confusers_have_equal_image_and_separation_16` | R5: swap pairs have equal image and separation exactly 16 (all $\binom{6}{4}$ tetrahedra of $K_6$) |
| `test_first_order_indistinguishability_deterministic_signals` | R5(i): any deterministic signal under $S$ is replicated exactly under $S'$ (residual $<10^{-9}$) |
| `test_chernoff_small_snr_constant_is_rho2_squared` | R5(iii): swap $C_G/\rho_2^2 \to 1$ (within $6\times10^{-4}$ at $\rho_2=10^{-4}$) |
| `test_exhaustive_k5_confuser_separations` | R5(iii): exhaustive over all $2^{10}$ supports of $K_5$ (45 000+ equal-image pairs): unrestricted min separation = 9, equal-cardinality min = 16 |
| `test_subset_confuser_exponent_equals_isolated_triangle_exponent` | R5(iii): subset $C_G = (9/16)\rho_2^2$ = isolated-triangle exponent (to $3\times10^{-5}$ at $\rho_2=10^{-5}$) |
| `test_binary_confuser_test_error_matches_chernoff_exponent` | R5(iii): MC error of the optimal test $\le e^{-NC_G}$; empirical exponent decreases onto $C_G$ |
| `test_second_order_recovery_succeeds_in_rank_deficient_regime` | R5(ii): greedy covariance-atom recovery exact on rank-deficient $K_5$ with gradient nuisance |
| `test_sample_complexity_and_fano_are_consistent` | R5(iii): $N^\star = \log(1/\delta)/C_G$; Fano positive/finite/monotone; vacuous without tetrahedra |
| `test_rho2_definition_links_first_and_second_order_snr` | $\rho_2 = \sigma_c^2/\sigma_n^2$ bookkeeping ($\rho = 3\rho_2$) |
| `test_diagonal_excitation_weighted_support_identifiable` | Regime (a): weights read off the covariance exactly (atoms full rank) |
| `test_kn_unknown_noise_ambiguity_direction` | Regime (a) caveat: $\sum_\tau u_\tau u_\tau^\top = n I_r$ on $K_5$–$K_7$ |
| `test_arbitrary_psd_equal_image_families_overlap` | Regime (b): equal-image $S'$ matches any $S$-covariance with a PSD $\Gamma'$ (family overlap) |
| `test_strict_positive_diagonal_witness_defeats_identifiability` | Regime (b) **strong** witness: strictly-positive-diagonal but correlated $\Gamma'=vvᵀ$ on K4 reproduces $u_0u_0^\top$; images 3 vs 1 |
| `test_singular_gamma_degenerate_example` | Regime (b) degenerate example: $\Gamma'=\mathrm{diag}(1,0)$ (inactive triangle); kept only as the easy case |
| `test_feasible_support_set_is_not_a_singleton` | Regime (b) structural: feasible supports $\{S:\mathrm{im}\,M\subseteq\mathrm{im}\,U_S\}$ has $>1$ element ⇒ population cov doesn't identify the support |
| `test_full_rank_gamma_identifies_image` | Regime (b): $\Gamma \succ 0 \Rightarrow$ realized range $=$ full curl image (on a full-column-rank support) |
| `test_singular_gamma_can_still_give_full_image_when_support_rank_deficient` | Regime (b): the converse $R=\mathrm{im}\,U_S\Rightarrow\Gamma\succ0$ fails on rank-deficient supports |
| `test_projector_excitation_equal_image_indistinguishable` | Regime (c): $B_S G_S^+ B_S^\top = P_{\rm im}$ exactly; identical covariances |
| `test_separation_identity_and_johnson_eigenvalue_bound` | sep $= c^\top(9I{+}A)c$; $\lambda_{\min}(A) \ge -3$ on $K_4$–$K_7$ + random complexes |
| `test_nnls_consistency_and_recovery_bound_components` | NNLS theorem steps: exact Wishart moment (±6%), cone-LS bound (0 violations in 220 trials), contraction |
| `test_nnls_recovery_bound_upper_bounds_empirical_failure` | the explicit $O(1/N)$ failure bound holds across an $(N,\rho_2)$ grid |
| `test_nnls_recovery_bound_nonvacuous_cells` | the bound is informative ($\approx0.73$–$0.75<1$) at dedicated cells and still upper-bounds the observed failures (zero) |
| `test_subspace_baseline_population_tie_and_projector_collapse` | subspace scores tie at 1 in population; specific-S recovery → 0 under projector excitation |
| `test_projector_chance_is_one_over_m_under_uniform_prior` | minimax: m equal-image supports identical ⇒ no estimator > 1/m under a uniform prior (m=4, chance=1/4) |
| `test_alpha_interpolation_kills_equal_image_distinguishability` | equal-image gap monotone → exactly 0 at $\alpha=1$ |
| `test_geo.py` (4 tests) | wind→edge-flow bridge: Euler full rank; Rankine vortex localizes with correct sign; vorticity ranking |

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
**ERA5 winds + IBTrACS cyclones (Figure 2 — the vortex-localization study):**

- **Winds**: ERA5 reanalysis (Hersbach et al. 2020), 10m u/v components, from
  the public [ARCO-ERA5](https://github.com/google-research/arco-era5) archive
  on Google Cloud Storage (`gs://gcp-public-data-arco-era5`, anonymous access):
  store `ar/1959-2022-6h-512x256_equiangular_conservative.zarr` (6-hourly,
  0.703° equiangular grid). Cached slice: Western North Pacific
  (0–45°N, 100–180°E), 2020-08-01 – 2020-09-30, 244 snapshots →
  `data/era5_wnp_2020.npz`. Refetch with `python data/fetch_era5_fast.py`
  (concurrent chunk reads; the sequential `fetch_era5.py` does the same via
  plain xarray, and `assemble_era5.py` builds the cache from chunks downloaded
  by any external HTTP tool on very slow links).
- **Cyclone ground truth**: IBTrACS v04r01 (Knapp et al. 2010), the
  agency-merged best-track archive from NOAA NCEI — **fully independent of the
  reanalysis**. Cached subset: all Western-Pacific fixes in Aug–Sep 2020
  (726 fixes, 13 storms, incl. Bavi, Maysak, Haishen) →
  `data/ibtracs_wp_2020.csv`; refetch with `python data/fetch_ibtracs.py`.
- Why this is the *right* achievability dataset: the latent structure
  (cyclone circulation) is genuinely present and **nobody planted it**; the
  mesh curl is physically the circulation (Stokes), so "filled triangles" have
  unambiguous meaning; and there are two complementary references — a
  same-field consistency reference (full-resolution vorticity of the same
  winds: internal, quantitative) and the genuinely independent IBTrACS
  positions (external, semantic).

**Road networks (repo figure `real_traffic.png`; outside the 4-page paper):**

- Source: the [Transportation Networks for Research](https://github.com/bstabler/TransportationNetworks)
  repository — the community-standard test networks for traffic assignment.
  Vendored under `data/traffic/` (~100 KB total): `SiouxFalls`, `EMA`
  (Eastern-Massachusetts; no published flow solution), and `Anaheim`, each as
  `*_net.tntp` (directed link list → simple undirected graph) and, where
  available, `*_flow.tntp` (user-equilibrium link flows → undirected *net* flow,
  sign convention $i \to j$ positive for $i<j$). Refetch with
  `python data/fetch_traffic.py`.

**FX construction:**

- Construction: currency $i$ gets log price $p_i = -\log(\text{rate}_i)$ in USD
  (base gets $p=0$); the edge flow on $(i,j)$ is $p_j - p_i$ via $B_1^\top p$
  — i.e., flows are *exactly* the market's log exchange rates re-expressed on the
  complete graph.
- Refetch/extend anytime with `python data/fetch_fx.py` (edit the date range/symbols
  in the URL).

Why FX is the *right* real dataset for a limits paper: covered-interest/triangular
**arbitrage-freeness means the true flow is a pure gradient** (rates derive from one
price vector), so theory predicts machine-zero curl — and that is precisely what we
measure ($10^{-31}$, i.e. floating-point residue). FX *instantiates the
converse*; the road networks (repo figure) supply the achievability side —
full-rank geometry, genuine curl energy in the real flows, and recovery at the
predicted budget.

---

## 10. Relation to prior work

| Work | What it does | What it does *not* do |
|---|---|---|
| Barbarossa & Sardellitti 2020 (TSP), §VII PCA-BFMTV | **Covariance/PCA precedent**: infers filled triangles from the *sample covariance* of edge flows (eigendecomposition + total-variation) | An *algorithm*, no identifiability conditions — no statement of *when* the covariance determines the support, no sample-complexity/SNR threshold, no converse |
| Yang et al. 2022 (TSP filters); Schaub et al. 2020 | Builds the Hodge-Laplacian signal-processing toolbox | Assumes the filled-triangle set is **known** |
| Gurugubelli & Chepuri, EUSIPCO 2024 (sparse clique sampling, MAP); greedy topology learning (arXiv 2502.20159); sparse cell complexes (arXiv 2309.01632) | **Algorithms** that estimate the filled set from flows | No identifiability conditions, no sample-complexity/SNR thresholds, no converse. G–C's generative model `y~N(0,L₂⁺)` is precisely the **projector excitation** our regime (c) proves second-order-**unidentifiable** — recoverable only via their sparsity prior |
| Marinucci et al. 2025 (topological adaptive LMS) | Online estimation over simplicial complexes, edge-sampling design | Complex structure assumed known; lists structure-discovery as future work |
| Marinucci, D'Acunto, Di Lorenzo & Barbarossa 2025, *Simplicial Gaussian Models* (arXiv:2510.12983) | **Closest second-order model**: a Hodge-structured Gaussian over nodes/edges/triangles, fit by **maximum likelihood** to recover parameters + conditional-dependence structure (edge-marginal, node/triangle latent) | Estimates continuous parameters of an assumed model; to our knowledge gives **no excitation-class-dependent identifiability characterization** — i.e. no statement of *when* the filled-triangle structure is recoverable from the edge covariance vs. only up to `{S : im M ⊆ im U_S}` (our regime (b)) |
| Hypergraph/simplicial SBM detectability (e.g. arXiv 2312.00708, 2108.06547) | Phase transitions for community detection when the **structure itself is observed** | Different observation model: our structure is *latent* and observed only through flow curls |
| High-dimensional support-recovery / sparse-covariance detection limits (Wainwright 2009; Amini & Wainwright 2009; Berthet & Rigollet 2013; Cai-Zhang-Zhou 2010) | The generic information-theoretic machinery (Fano log-factors, variance-detection scalings) we **specialize** — cited as such in the paper | No simplicial/curl geometry: nothing about $B_2$, Gram leakage, $\ker B_2$ confusers, or the lifted-atom spark that drives our regime-(a) identifiability |
| Liu, Tenorio, Marques & Isufi 2025 (matched topological subspace detectors, arXiv:2504.05892) | **Detection** of signals lying in a *known* Hodge/topological subspace | The subspace is an input, not an estimand; population statistics see only $\mathrm{im}\,B_{2,S}$ — our regime (b) delimits exactly what such image-level methods can resolve |
| **This project** | **Fundamental limits** for the latent-structure-from-flows problem: converse scalings (R2), the three excitation regimes with global analytic separation constants (R5 — main theorem), a consistent NNLS estimator with an explicit $O(1/N)$ failure bound (R4/R5), and curl-based vortex localization on real data (R6) | — |

In short: prior TSP work asks *how* to infer the complex; this project proves *when*
it can and cannot be done, and hands back a threshold + estimator that the
algorithmic line can calibrate against.

---

## 11. Limitations and honesty notes

Stated plainly, because reviewers (and users) deserve to know:

1. **Gaussianity.** The converse and the exact finite-sample laws are proved under
   Gaussian signals/noise. The supplement extends both directions — a
   signal-agnostic Fano converse (any zero-mean signal law) and sub-Gaussian
   achievability with $K^4$-degraded constants (Prop. S1.5) — but a
   sub-Gaussian *converse with matching constants* remains open.
2. **Known second-order parameters.** The paper's detectors take
   $(\sigma_c, \sigma_n)$ as inputs; supplement §S3 provides consistent plug-in
   estimators and a fully
   adaptive detector. The remaining gap is a non-asymptotic *joint* analysis of
   selection and estimation (the refinement pass is conditionally biased at
   small $N$; see §12/S3).
3. **What is (and is not) real in the real-data studies.** FX demonstrates the
   converse — an efficient market is curl-free, so there is genuinely nothing to
   recover. The road-network study demonstrates achievability with **planted**
   triangle supports: real traffic carries no labelled triangle ground truth, so
   what is real there is the network geometry (full-rank $B_2$) and the
   equilibrium-flow background the detector must be invariant to. Since the
   2026-07 revision both studies live in the repository/supplement only; the
   4-page paper's sole real-data study is the cyclone vortex localization. Fully-real
   ground-truth recovery is validated on controlled planted complexes
   (Figures 1–2).
4. **`sparse_curl_covariance_support`** (the lifted cvxpy estimator) is provided as
   a baseline but is dominated by the whitened detector in our benchmarks; it is not
   used for any headline claim.
5. **i.i.d. snapshots.** Temporal dependence (e.g. AR flows) shrinks the effective
   sample size; the theory applies with $N \to N_{\rm eff}$ but we do not
   characterize $N_{\rm eff}$ here.
6. **Joint vs marginal laws under confusability.** The decorrelated per-triangle
   laws are exact *marginals*; the joint recovery probability is guaranteed only
   via the union bound. The product form is an independence approximation (exact
   for edge-disjoint candidates) that happens to be tight in our benchmarks.
7. **Corrected claims (scientific honesty).** Two errors were caught by
   adversarial multi-agent review of our own drafts and are corrected rather
   than silently fixed. (a) An earlier revision stated the rank obstruction
   in the false general form — "supports with equal curl image induce
   identical flow distributions, indistinguishable at any SNR and any $N$".
   True for *deterministic* signals; false for the random model (the
   covariance is strictly finer than the column image; spark lemma). The
   corrected statement was R5's first/second-order dichotomy --- since
   generalized to the excitation trichotomy (note 10 below). (b) A subsequent revision claimed
   the tetrahedral *swap* separation 16 was "exhaustively minimal over all
   equal-image pairs on $K_5$" — the scan behind that claim only compared
   supports of equal cardinality. The unrestricted minimum is **9** (the
   subset confuser), now verified by a genuinely exhaustive test over all
   $2^{10}$ supports (`test_exhaustive_k5_confuser_separations`) — and the
   corrected constant makes the theorem *stronger* (the $(9/16)\rho_2^2$
   worst case equals the isolated-triangle exponent). A limits paper dies if
   a constant is wrong; we prefer the scar tissue visible.
8. **Naming.** The `whitened_*` estimator functions decorrelate the *mean* of
   the curl statistic (GLS/BLUE inversion); the residual noise covariance
   $\sigma_n^2 G^{-1}$ is **not** white. The paper says "geometry-aware
   decorrelation"; function names keep the historical prefix for API stability.
9. **Cyclone study scope.** The cyclone experiment recovers *unplanted* real
   structure, but the observation model there is an idealization too: 6-hourly
   the latent support is **not** static — cyclones translate 3–6°/day, i.e.
   ~6–11 triangle-widths per 4-day window. Temporal centering removes the
   window-mean flow, so the statistic detects the *time-varying* circulation
   a translating vortex deposits in every triangle it crosses (a perfectly
   static vortex would be removed by the centering); evaluation is by
   window-pooled ROC ranking, not per-window support recovery, and
   the detector's Gaussian null is a working model for weather-scale
   fluctuations; evaluation is threshold-free ROC ranking, so no null
   calibration is required.
10. **The 2026-07 method-level revision.** An earlier revision claimed that
   "random signals" per se make every support identifiable at any rank
   deficiency. That is true only for *structured* excitations: the claim was
   implicitly scoped to isotropic `Γ_S = σ_c²I`. The main theorem is now the
   excitation trichotomy, with the earlier isotropic results retained as the
   diagonal class. The revision retitled the paper, renamed the cyclone study
   to vortex localization, and removed an arbitrary-units theory-floor overlay
   from its budget panel.
11. **The 2026-07 second correction (this revision).** A subsequent hostile
   read caught five further errors, all fixed here — again visible, not
   silently patched:
   - **Case (b) was still wrong.** We had claimed arbitrary PSD `Γ_S`
     identifies "exactly `im B₂,S`". False for **singular** `Γ_S`: the
     population covariance identifies only the *realized range* `R = im M ⊆ im
     U_S` (`M = Σ_z − σ_n²I`); a rank-deficient `Γ_S` hides part of the image
     and even the support. (This note's own first patch, "`R = im U_S` iff
     `Γ_S ≻ 0`", was **itself too strong** and was scoped/superseded in
     note 12 below.)
   - **Prior work mis-cast.** Gurugubelli–Chepuri (EUSIPCO 2024) do **not**
     use isotropic excitation — their generative model is `y ~ N(0, L₂⁺)`,
     `L₂ = B₂ᵀB₂`, i.e. the **projector excitation** (case (c), the
     unidentifiable one). The "prior work is the isotropic special case" claim
     was removed. Barbarossa–Sardellitti 2020 §VII (PCA-BFMTV) is a genuine
     covariance/PCA precedent for filled-triangle inference and is now cited
     as such.
   - **Chernoff dimension.** The exponent was written `C_G =
     (ρ₂²/16)‖ΔM‖_F²`, double-counting `σ_c⁴`. Correct: `C_G =
     (ρ₂²/16)‖ΔD‖_F² = ‖ΔM‖_F²/(16σ_n⁴)` with the dimensionless
     `D_S = Σ u_τu_τᵀ`.
   - **Harmonic assumption.** Curl annihilation kills the harmonic nuisance
     only for `h ∈ ker B₂ᵀ`; the model now states this candidate-orthogonal
     assumption explicitly (the gradient part is killed unconditionally).
   - **largest-gap ≠ better.** The oracle-free largest-gap threshold is
     *worse* than the theorem's `w_min/2` rule (grid-mean 0.63 vs 0.69, ahead
     in only 18/260 cells); an earlier note claiming it beat the oracle was
     wrong and is retracted. The per-cell failure bound is now the
     **worst-case over the trial supports**, not one representative.
   - Also: the sample-complexity gap over the single-triangle budget is a
     **constant** factor (both `Θ(1/ρ₂²)` at low SNR), not `1/ρ₂`; the old
     phase-transition main-text branch was cut; and "test-certified"/
     "certifying every constant" language was replaced with "numerically
     checked".
12. **The 2026-07 third correction (this revision).** A submission-blocking
   review demanded a sharper regime (b), and again caught real errors:
   - **Structural characterization.** The right object is the achievable set
     `C_S = {M⪰0 : im M ⊆ im U_S}`; given the **population** covariance
     (`M = Σ_z − σ_n²I`), the support is fixed only to
     `{S : im M ⊆ im U_S}`. We now use *population covariance* throughout
     (not "one observation") and renamed the theorem **three excitation
     regimes** (overlapping constraints, not a partition). The conclusion's
     "image-only under unrestricted PSD" was deleted.
   - **The note-11 "iff" was too strong.** `R = im U_S ⟺ Γ_S ≻ 0` holds only
     when `B₂,S` has full column rank; on a rank-deficient support a *singular*
     `Γ` can still give the full image (`Γ = I − ccᵀ/‖c‖²` on K4).
   - **Stronger counterexample.** `diag(1,0)` (an inactive triangle) is now
     only a degenerate example; the headline witness is a
     **strictly-positive-diagonal but correlated** `Γ'=vvᵀ` (K4, `v=e₀−¼c`,
     all `diag>0`) with `U_S Γ' U_Sᵀ = u₀u₀ᵀ` — positive per-triangle variance
     is not enough; regime (a) needs `Γ` *diagonal*
     (`test_strict_positive_diagonal_witness_defeats_identifiability`,
     `test_feasible_support_set_is_not_a_singleton`).

---

## 12. Supplement: Fano converses, partial sampling, plug-in estimation

Five sections live in [`paper/supplement.pdf`](paper/supplement.pdf) (LaTeX
source `paper/supplement.tex`; the main paper cites it): the three extensions
below, plus **§S4** — full proofs of the lifted-spark lemma, the three excitation
regimes, the global separations (Johnson-graph interlacing), and the NNLS
consistency/failure bound, with exhaustive numerical verification — and
**§S5** — construction details of the vortex-localization study. Each
extension has code, tests, a figure, and JSON metrics.

### S1 — Fano converses for *joint* support recovery

The paper's Prop. 2 bounds a *single* triangle's detectability (and already
states the joint Fano bound proved here). Recovering
the whole support of size $k$ among $\binom{p}{k}$ candidates is harder:

- **Gaussian Fano converse** (`fano_min_snapshots`): with edge-disjoint
  candidates, $I(S; C^N) \le N k \cdot \mathrm{KL}_1(\rho)$ exactly, with
  $\mathrm{KL}_1 = (\rho - \log(1{+}\rho))/2 \le \rho^2/4$, so
  $\rho^2 N \gtrsim 4\log(p/k)$ — **joint recovery pays a $\log p$ factor**
  where the single-triangle test pays $\log(1/\delta)$.
- **Signal-agnostic Fano converse** (`signal_agnostic_fano_min_snapshots`):
  a max-entropy budget $I \le (p/2)\log(1+\rho k/p)$ valid for *arbitrary*
  zero-mean signal distributions with the given variance (Gaussian noise).
  Weaker by a factor $\rho$ at small SNR, but it **overtakes** the
  Gaussian-KL bound for dense supports at high SNR — a valid converse takes
  the pointwise max (`fano_bounds.png`, panel B).
- **Sub-Gaussian achievability**: a Bernstein/Hanson-Wright argument shows
  the energy detector needs only
  $N \gtrsim K^4 (1{+}\rho)^2 \log(p/\delta)/\min(\rho,\rho^2)$
  under sub-Gaussian signals/noise — the same $\log p/\rho^2$ small-SNR
  scaling, constants degraded by the fourth power of the $\psi_2$-proxy $K$
  (supplement Prop. S1.5).

Figure: `results/figures/fano_bounds.png` (script `experiments/run_fano.py`).
Test: `test_fano_bounds_are_valid_converses` checks both bounds lower-bound
the *empirical* 50%-recovery budget measured in Figure 2 (the phase
transition), are monotone, and
exhibit the documented sparse/dense ordering.

### S2 — Partial edge observation

If each edge is observed independently with probability $q$, a triangle's curl
statistic is computable **iff all three edges are observed** (survival
probability $q^3$). On edge-disjoint geometries this gives an exact law
(supplement Cor. S2.2):

```math
P(\hat S = S \mid q) \;=\; \big[q^3 (1 - P_{\mathrm{miss}})\big]^{k}\,
\big[1 - q^3 P_{\mathrm{fa}}\big]^{p-k}.
```

The $q^{3k}$ factor is brutal — on Anaheim ($k{=}18$), $q=0.9$ already
destroys recovery — and the experiment shows the collapse tracks the closed
form exactly while recovery *restricted to observable candidates* stays at
$\approx 1$: **the bottleneck is coverage, not detection**. Honest scope note:
this is an operational limit for curl-statistic detectors; whether partial
triangles (2 of 3 edges) can be exploited by marginalizing the gradient is
stated as open (supplement Rem. S2.3).

Figure: `results/figures/partial_sampling.png`
(script `experiments/run_partial_sampling.py`).
Test: `test_partial_sampling_closed_form_matches_simulation`.

### S3 — Plug-in variance estimation & the fully adaptive detector

`estimate_noise_sigma` ($\chi^2$-median-calibrated median of normalized
whitened scores — robust while $<50\%$ of candidates are active),
`estimate_curl_sigma` (mean excess energy over a conservative Bonferroni
screen), plus one refinement pass (`adaptive_whitened_detector_support`,
`refine=True`): re-fit $\sigma_n$ on the detected off-support, $\sigma_c$ on
the on-support excess, re-run once. Supplement Lemma S3.1 gives an **explicit,
numerically computable envelope** for the median estimate (DKW + $\chi^2$
quantiles — no hidden constants); the plug-in threshold's effect on
recovery is reported **empirically only** (demoted from a proposition to an
empirical observation — no proven uniform-constant bound). Empirically both estimates contract
at the $1/\sqrt N$ rate and the adaptive detector matches the known-parameter
detector within 0.06 everywhere (below 0.01 for $N \ge 30$, exactly zero for
$N \ge 45$) on the strip benchmark.

Figure: `results/figures/plugin.png` (script `experiments/run_plugin.py`).
Test: `test_median_sigma_envelope_and_plugin_consistency`.

**Build the supplement:**

```bash
cd paper
pdflatex supplement && pdflatex supplement
```

---

## 13. Roadmap

- [x] A recoverable real dataset (traffic flows on planar road networks — sparse
      triangles, favorable geometry) as an achievability companion to the FX
      converse. **Done: `run_real_traffic.py` (TNTP Sioux Falls, EMA, Anaheim);
      the EMA panel exercises the non-diagonal-G geometry-aware theory.**
- [x] Camera-ready port to the official ICASSP `spconf.sty`, 4 pages + references-only
      page 5. **Done.**
- [x] Fano converses for joint recovery + sub-Gaussian achievability.
      **Done: supplement §S1 / `run_fano.py`.**
- [x] Partial edge observation: identifiability vs edge-sampling rate.
      **Done: supplement §S2 / `run_partial_sampling.py` (exact $q^{3k}$ law).**
- [x] Plug-in variance estimation + adaptive thresholds.
      **Done: supplement §S3 / `run_plugin.py` (median envelope + refinement).**
- [x] **Correct the rank obstruction** (found false-as-stated by adversarial
      self-review) and replace it with the first-/second-order identifiability
      dichotomy + universal sample-complexity constants.
      **Done: main theorem / `tests/test_second_order.py` / supplement §S4.**
- [x] **Genuine unplanted real-data recovery** with independent external ground
      truth. **Done: tropical-cyclone circulation from ERA5 edge flows,
      validated against full-resolution vorticity + IBTrACS
      (`run_real_cyclone.py`).**
- [x] **2026-07 method-level revision** (adversarially triggered): the dichotomy
      generalized to the **excitation trichotomy** (diagonal / free-PSD /
      projector), a lifted-covariance **NNLS estimator** with consistency +
      explicit $O(1/N)$ failure bound on a K4–K8 random-support grid, the
      cyclone study repositioned as **vortex localization** (PR-AUC, baselines,
      moving-block bootstrap CIs), and the page-5 compliance statements.
      **Done: main Thm. 1–3 / `tests/test_excitation.py` / `run_second_order.py`.**

Remaining open problems (deliberately left for a journal version): sharp
sparse-support thresholds over $\ker B_2$ (compressed-sensing conditions —
paper §6 open problems), the information-theoretic limit of partial observation beyond
curl detectors (supplement Rem. S2.3), and temporal dependence
($N_{\mathrm{eff}}$ — §11 item 5 above).

---

## 14. 中文速览

**问题**：拓扑信号处理需要知道网络中哪些三角形被"填充"（承载高阶相互作用），
现有文献只给出**估计算法**（贪婪、MAP、稀疏恢复），没人回答更根本的问题——
**给定 N 个边流快照和噪声水平，这个结构到底可不可辨识？**

**核心机制（旋度湮灭）**：对边流取旋度 $c_t=B_2^\top f_t$ 会把梯度分量和谐和分量
**精确消掉**（因为 $B_1B_2=0$ ，且谐和空间无旋），只剩下活跃三角形的信号加投影噪声。
所以隐结构只透过"旋度"这一扇窗可见。

**四个主要结果**：
1. **两方差检验**：孤立三角形的检测化为方差 $3\sigma_n^2$ vs $9\sigma_c^2+3\sigma_n^2$
   的高斯检验，最优误差指数是 Gaussian Chernoff 信息；边不相交时精确恢复概率有
   **严格的**有限样本乘积公式（此时各三角形统计量真独立）；
2. **curl-invisibility 样本复杂度阈值**：curl 信噪比 $\rho<\rho^\star(N)\sim 1/\sqrt N$ 时
   **任何估计器都达不到目标误差指数**（小 $\rho$ 极限下 $C\sim\rho^2/16$ ；
   这是指数层面的标度律，我们不宣称临界点意义上的"相变"）；联合恢复整个大小为
   $k$ 的支持集还要多付 $\log\binom pk$ 因子（Fano converse）；
3. **几何感知去相关（GLS/BLUE）**：共边三角形互相"泄漏"能量，朴素能量检测会失败且
   SNR 越高越糟；当 $B_2$ 满列秩时用 $\hat y=G^{-1}c$（教科书 GLS/BLUE 反演——
   论文如实标注其出处），每个三角形获得**严格的边际**两方差律，有效信噪比
   $`\rho^{\mathrm{eff}}_\tau=\sigma_c^2/(\sigma_n^2(G^{-1})_{\tau\tau})`$
   （经典方差膨胀因子）。注意 $G^{-1}$ 去相关的是**均值**，噪声坐标仍相关
   （所以不叫"白化"），联合恢复概率一般**不**分解——严格保证由 union bound
   给出，乘积式只是独立近似（与蒙特卡洛全程相差 ≤0.03）；
4. **主定理——激励依赖的可辨识性：三个激励区制（overlapping，非划分）**
   （2026-07 方法级修订；修正了早期版本"随机信号自动消除秩障碍"的过强结论，
   见诚实注记 7、10、11、12）。所有结论均针对**总体协方差** $\Sigma_z$。设三角
   激励 $y_S\sim N(0,\Gamma_S)$，$\Gamma_S$ 半正定：
   - **(a) 正对角 $\Gamma_S$（已知或未知）**：**带权支持集在任意秩亏下
     可辨识**。**提升火花引理**：一对不同的边至多属于一个公共三角形，故
     原子 $\{b_\tau b_\tau^\top\}$ **永远线性无关** ⇒ 权重可从协方差直接
     读出（$K_n$ 上若 $\sigma_n$ 未知恰有一维歧义：
     $\sum_\tau u_\tau u_\tau^\top=nI_r$）。配套估计器：**提升协方差
     NNLS**（一致 + 全显式 $O(1/N)$ 失败界，常数全部推导并数值核验）；
   - **(b) 任意 PSD $\Gamma_S$**：可达协方差集
     $C_S=\{M\succeq0:\mathrm{im}\,M\subseteq\mathrm{im}\,U_S\}$，故总体协方差
     完全确定 $M=\Sigma_z-\sigma_n^2 I$，但支持集只能定到
     $\{S:\mathrm{im}\,M\subseteq\mathrm{im}\,U_S\}$：实现秩空间
     $R=\mathrm{im}\,M\subseteq\mathrm{im}\,U_S$ 始终可读，而候选像
     $\mathrm{im}\,B_{2,S}$ **不可辨识**（像更大的 $S'$ 配合适 $\Gamma$ 给出
     同一 $M$）。**即便严格正对角但相关**的 $\Gamma'=vv^\top$（$K_4$，
     $v=e_0-\tfrac14 c$，$c\in\ker U_S$，各 $\mathrm{diag}>0$）也给出
     $U_S\Gamma' U_S^\top=u_0u_0^\top$——与 $\{\tau_0\}$ 相同而像维数 3 vs 1；
     正方差不够，区制 (a) 还需 $\Gamma$ **对角**（$\Gamma_S\succ0$ 时确实恢复
     $R=\mathrm{im}\,U_S$）；
   - **(c) 投影激励 $\Gamma_S=\sigma_c^2(B_{2,S}^\top B_{2,S})^+$**（正是
     Gurugubelli–Chepuri 2024 用的 Hodge 光滑先验 $\Gamma=L_2^+$）：协方差
     恰为 $\sigma_n^2 I+\sigma_c^2 P_{\mathrm{im}}$，等像支持集在任何
     SNR、任何 $N$ 下**完全不可分**。$\alpha$-插值实验
     （$\Gamma_\alpha=(1-\alpha)I+\alpha(B_S^\top B_S)^+$）中解析间隙连续
     缩小，而带阈值（声明的 $\rho_2/2$ 规则）的 NNLS 恢复在间隙归零
     **之前**就已塌到 0（1.00→0.82→0.00，$\alpha=0.75$ 处已为 0）——
     阈值化估计先于可区分性死亡，仅 $\alpha=1$ 端点是无阈值的完全
     不可分。
   一阶/确定性信号是 (b) 的退化边界：等像支持集诱导相同分布族；$K_n$ 上
     自由度比恰为 $3/n$；
   - **(a) 类内的代价（样本复杂度）**：区分一个混淆对需
     $N^\star\sim\log(1/\delta)/C_G$。分离度现在有**全局解析证明**
     （share-edge 图是 Johnson 图 $J(n,3)$ 的诱导子图，交错定理给
     $\lambda_{\min}\ge-3$，对任意 clique complex 成立；并在 $K_5$ 上对
     全部 $2^{10}$ 个支持集、45000+ 等像对穷举验证）：**面交换**分离度恰为
     $9+9-2=16$（**等基数**最小值），**子集混淆对**（$S$ vs $S\cup\{$第四
     面$\}$）分离度恰为 $9$ ——**无限制最小值**（早期版本误称 16，见诚实
     注记 7）。Chernoff 结论严格限定在固定图/固定支持对、$\rho_2\to0$：
     交换 $C_G=\rho_2^2(1+o(1))$，子集
     $C_G=\tfrac{9}{16}\rho_2^2(1+o(1))$ ——后者**恰好等于孤立三角形检测
     指数** $C(\rho)=\rho^2/16|_{\rho=3\rho_2}$：(a) 类内秩亏在指数层面
     免费，几何只通过 Fano 的 $\log$ 多重性因子出现。

**补充材料（`paper/supplement.pdf`，仓库内 11 页；除下述三项外还含 S4
激励三分定理+分离度+NNLS 全部证明与穷举验证、S5 涡旋定位实验方法细节）**：
① *联合恢复的 Fano converse*——恢复整个大小为 $k$ 的支持集要付出 $\log\binom pk$
因子（ $\rho^2 N \gtrsim 4\log(p/k)$ ），并给出对**任意信号分布**成立的
signal-agnostic 变体（稠密支持、高信噪比时反而更紧，取两者逐点最大值）与
次高斯噪声下的 achievability（同阶样本复杂度，常数差 $K^4$ ）；
② *部分边观测*——三角形三边全被观测才可测（存活率 $q^3$ ），边不相交几何上
精确恢复有闭式 $[q^3(1-P_{\rm miss})]^k[1-q^3P_{\rm fa}]^{p-k}$ ，
$q^{3k}$ 因子极其残酷（Anaheim 上 $q=0.9$ 即摧毁恢复），且瓶颈纯粹是覆盖
而非检测；
③ *插入式方差估计*——中位数估 $\sigma_n$ （给出无隐藏常数的 DKW/卡方分位数
包络）、超阈值超额能量估 $\sigma_c$ 、加一次 refinement，全自适应检测器与
已知参数检测器几乎无差（ $N\ge 30$ 时零差距）。

**真实数据**：
① **旗舰结果——基于旋度的真实涡旋定位（台风，未种植）**：ERA5 再分析 10m 风场
（西北太平洋，2020 年 8–9 月，0.7°/6 小时）投影为三角剖分网格上的边流；
由 Stokes 定理，三角形的旋度就是环流（面积分涡度），所以"高旋度三角形"
就是真实的大气涡旋。检测器用时间中心化的旋度能量统计量（面积²归一到
涡度尺度），无 oracle 参数，**什么都没有种植**；用**两个参考**验证
（一个同场一致性参考 + 一个真正独立的 IBTrACS 档案）：
全分辨率有限差分涡度（内部定量：**AUC 0.914**，Spearman 0.77）与 IBTrACS
台风最佳路径档案（外部、机构核验、与再分析完全独立；覆盖 Bavi、Maysak、
Haishen 等 13 个风暴：**AUC 0.920**，PR-AUC 0.494（阳性率 1.8%），
precision@k 0.48，附时序 moving-block bootstrap 95% CI（块长 3 窗口，尊重风暴尺度序列相关）。网格单连通 ⇒
$B_2$ 满列秩（Euler 公式）⇒ 激励三分定理 (a) 类的有利几何侧，无混淆对。降采样快照
预算后检测质量随之下降（统一使用中心化统计量、$N\ge3$；面板不再叠加
任意单位的理论下限曲线——2026-07 修订中已删除，不宣称定量跟踪
$1/\sqrt N$）。对照经典基线（同信息预算的网格点涡度）：
外部独立参考上我们更优（0.920 vs 0.898），内部同场参考上略低
（0.914 vs 0.948——该参考与基线共享泛函）。诚实注记：GLS 去相关
统计量（R4，面向精确支持集恢复）在这种近规则满秩网格上因 $G^{-1}$
噪声放大而检测排序更差（AUC 0.77/0.81，已在 JSON 中报告）。
② *汇率（一致性校验，诚实重标）*：边流按构造是单一价格向量的梯度
$f=B_1^\top p$，故 $B_2^\top f\equiv 0$ **恒等成立**——测得的 $10^{-31}$
是浮点残差，验证的是旋度湮灭的算术而非市场的实证性质；$K_9$ 的自由度比
$28/84=1/3$ 落在 $3/n$ 曲线上（这一半是真实的几何测量）。
③ *路网交通（真实几何上的恢复律校验）*：三个 TNTP 网络 $B_2$ 全部满列秩，
真实均衡流带 2.4%–4.6% 真旋度能量；Anaheim（$G=3I$，乘积律严格——诚实
标注：中心化精确移除真实背景后剩下的统计问题是合成的）与 EMA
（$G$ 非对角、10 对共边三角形——真正让几何感知边际律干活的面板）。
种植实验的真实成分是几何；真正未种植的恢复证据见 ①。

**复现**：`pytest -q`（51 个测试，约 3–5 分钟，每条定理都对照蒙特卡洛验证）；
`python experiments/run_all.py`（约 30–45 分钟重生成全部 9 张图）；
论文在 `paper/main.pdf`（ICASSP 官方 spconf 格式，4 页正文 + 第 5 页参考
文献与合规声明），补充材料在 `paper/supplement.pdf`（11 页）。全程只需 CPU。

---

## 15. Citation & license

If you use this code or the results, please cite the repository (paper citation to
follow upon publication):

```bibtex
@misc{topoflowlimits2026,
  title  = {Excitation-Dependent Identifiability of Latent Higher-Order
            Structure from Edge Flows},
  author = {Yan, Weiping},
  year   = {2026},
  howpublished = {\url{https://github.com/appleweiping/topo-flow-limits}}
}
```

**License:** MIT (see `LICENSE`).
