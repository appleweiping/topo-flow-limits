"""Matrix-free lifted-covariance NNLS — the scalable form of the Theorem-3
estimator.

The lifted design operator ``A`` maps ``w in R^p`` to
``sum_tau w_tau vec(u_tau u_tau^T) in R^{r^2}``.  The dense ``(r^2, p)`` matrix
built by :func:`tfl.limits.lifted_atom_matrix` is fine for ``K4-K8`` but has
``r^2 p`` float64 entries -- with ``r = dof * p`` and ``dof ~ 0.5`` that is
about ``2`` GB at ``p=1e3`` (``r=500``) and grows to the ``~1-2`` TB range at
``p=1e4``, so it does not scale.  Everything here works only through ``U``
(``r x p``) and two factored kernels:

    A w      = U diag(w) U^T              (r x r),   :func:`lifted_apply`
    A^T R    = [u_tau^T R u_tau]_tau      (p,),      :func:`lifted_adjoint`

Two identities the module rests on (both machine-verified in
``tests/test_estimators_mf.py`` / the round-8 rule-1 script):

  * gradient of ``f(w) = 1/2 ||C - A w||_F^2`` is ``A^T A w - A^T C = -A^T R``,
    ``R = C - A w`` — i.e. ``[-u_tau^T R u_tau]_tau``;
  * ``A^T A = G o G`` (Hadamard square of the triangle Gram ``G = U^T U``):
    diagonal ``9``, off-diagonal ``G_{sigma tau}^2 in {0,1}`` — the object
    already used by :func:`tfl.limits.share_edge_adjacency`.  Hence the FISTA
    Lipschitz constant is ``lambda_max(G o G)`` and the estimator-bound
    quantity is ``sigma_min(A) = sqrt(lambda_min(G o G))`` — both computed
    matrix-free.

Per-iteration cost is ``O(r^2 p)`` compute, ``O(r p + r^2)`` memory (there is
no ``O(r p)`` route for a dense ``w`` — every one of the ``r^2`` outputs is a
length-``p`` reduction).  Since ``r = dof * p``, cost ``~ dof^2 p^3`` and
memory ``~ dof^2 p^2``: strong rank deficiency (small dof) is both the
scientifically interesting regime and the affordable one.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Factored operator kernels
# ---------------------------------------------------------------------------
def lifted_apply(U: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``A w = U diag(w) U^T`` (``r x r``).  ``O(r^2 p)``."""
    return (U * w[None, :]) @ U.T


def lifted_adjoint(U: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``A^T R = [u_tau^T R u_tau]_tau`` (``p,``).  ``O(r^2 p)``.

    Uses the two-step ``RU = R @ U`` then ``sum(U * RU, axis=0)`` (pins flops
    and memory; avoids a 3-operand einsum building a bad intermediate)."""
    RU = R @ U                      # (r, p)
    return np.einsum("ap,ap->p", U, RU)


def lifted_ATC_from_Z(U: np.ndarray, Z: np.ndarray, sigma_noise: float,
                      N: int | None = None) -> np.ndarray:
    """``b0 = A^T C`` with ``C = Z Z^T / N - sigma_n^2 I``, formed WITHOUT the
    ``r x r`` matrix: ``b0_tau = (1/N) sum_t (u_tau^T z_t)^2 - 3 sigma_n^2``
    (since ``||u_tau||^2 = 3``).  ``O(p r N)`` compute, ``O(p N)`` memory."""
    if N is None:
        N = Z.shape[1]
    W = U.T @ Z                     # (p, N),  W[tau,t] = u_tau^T z_t
    return np.sum(W * W, axis=1) / N - 3.0 * sigma_noise ** 2


def gram_hadamard(U: np.ndarray) -> np.ndarray:
    """``A^T A = G o G`` densely (``p x p``) — only for small ``p`` / tests.
    Do NOT call at ``p >= 1e4``."""
    G = U.T @ U
    return G * G


# ---------------------------------------------------------------------------
# Spectral constants (matrix-free)
# ---------------------------------------------------------------------------
def lipschitz_ATA(U: np.ndarray, iters: int = 40, seed: int = 0,
                  tol: float = 1e-7) -> float:
    """``lambda_max(A^T A) = lambda_max(G o G)`` via matrix-free power
    iteration (``A^T A v = lifted_adjoint(U, lifted_apply(U, v))``)."""
    p = U.shape[1]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(p)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        Av = lifted_adjoint(U, lifted_apply(U, v))
        nrm = np.linalg.norm(Av)
        if nrm == 0:
            return 0.0
        v = Av / nrm
        lam_new = float(v @ lifted_adjoint(U, lifted_apply(U, v)))
        if abs(lam_new - lam) <= tol * max(1.0, lam_new):
            lam = lam_new
            break
        lam = lam_new
    return lam


def sigma_min_A(U: np.ndarray, dense_max_p: int = 2500) -> float:
    """``sigma_min(A) = sqrt(lambda_min(G o G))`` — the constant in
    :func:`tfl.limits.nnls_recovery_bound`.  Dense eigvalsh for ``p <=
    dense_max_p``; matrix-free ``eigsh(which='SA')`` otherwise."""
    p = U.shape[1]
    if p <= dense_max_p:
        lam_min = float(np.linalg.eigvalsh(gram_hadamard(U))[0])
        return float(np.sqrt(max(lam_min, 0.0)))
    from scipy.sparse.linalg import LinearOperator, eigsh
    op = LinearOperator((p, p), matvec=lambda v: lifted_adjoint(U, lifted_apply(U, v)),
                        dtype=float)
    lam_min = float(eigsh(op, k=1, which="SA", maxiter=2000,
                          return_eigenvectors=False)[0])
    return float(np.sqrt(max(lam_min, 0.0)))


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def nnls_lifted_fista(C: np.ndarray, U: np.ndarray, L: float | None = None,
                      max_iter: int = 2000, tol: float = 1e-10,
                      restart: bool = True) -> np.ndarray:
    """Non-negative least squares ``min_{w>=0} 1/2 ||C - A w||_F^2`` by FISTA
    with adaptive (gradient) restart.  ``C`` is ``r x r`` symmetric
    (``= Sigma_hat - sigma_n^2 I``).  Returns ``w_hat`` (``p,``)."""
    p = U.shape[1]
    if L is None:
        L = lipschitz_ATA(U)
    if L <= 0:
        return np.zeros(p)
    b0 = lifted_adjoint(U, C)               # A^T C
    w = np.zeros(p)
    y = np.zeros(p)
    t = 1.0
    for _ in range(max_iter):
        grad = lifted_adjoint(U, lifted_apply(U, y)) - b0   # A^T A y - b0
        w_new = np.maximum(0.0, y - grad / L)
        if restart and float((y - w_new) @ (w_new - w)) > 0.0:
            t = 1.0                                          # gradient restart
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        step = np.linalg.norm(w_new - w)
        w, t = w_new, t_new
        if step <= tol * max(1.0, np.linalg.norm(w)):
            break
    return w


def lasso_lifted_fista(C: np.ndarray, U: np.ndarray, lam: float,
                       L: float | None = None, max_iter: int = 2000,
                       tol: float = 1e-10, restart: bool = True) -> np.ndarray:
    """Non-negative lifted LASSO ``min_{w>=0} 1/2||C - A w||_F^2 + lam ||w||_1``
    by FISTA (prox = soft-threshold then clamp: ``max(0, y - grad/L - lam/L)``).
    The convex sparse-covariance / sparse-PCA baseline (Berthet-Rigollet,
    Amini-Wainwright lineage) evaluated against the NNLS estimator."""
    p = U.shape[1]
    if L is None:
        L = lipschitz_ATA(U)
    if L <= 0:
        return np.zeros(p)
    b0 = lifted_adjoint(U, C)
    w = np.zeros(p); y = np.zeros(p); t = 1.0
    thr = lam / L
    for _ in range(max_iter):
        grad = lifted_adjoint(U, lifted_apply(U, y)) - b0
        w_new = np.maximum(0.0, y - grad / L - thr)          # nonneg soft-threshold
        if restart and float((y - w_new) @ (w_new - w)) > 0.0:
            t = 1.0
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        step = np.linalg.norm(w_new - w)
        w, t = w_new, t_new
        if step <= tol * max(1.0, np.linalg.norm(w)):
            break
    return w


def lasso_lifted_support(Z: np.ndarray, U: np.ndarray, sigma_noise: float,
                         N: int | None = None, n_lam: int = 12,
                         L: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Faithful convex-sparse baseline: non-negative lifted LASSO with the
    regularization ``lam`` chosen by Gaussian-model BIC along a geometric
    ``lam`` path (no oracle).  Returns ``(support_mask, w_hat)``."""
    from tfl.selection import gaussian_cov_bic
    r, Nz = Z.shape
    if N is None:
        N = Nz
    Sig = (Z @ Z.T) / N
    C = Sig - sigma_noise ** 2 * np.eye(r)
    if L is None:
        L = lipschitz_ATA(U)
    b0 = lifted_adjoint(U, C)
    lam_max = float(np.max(b0)) if np.max(b0) > 0 else 1.0
    lams = lam_max * np.geomspace(1.0, 1e-3, n_lam)
    best_bic, best_w = np.inf, np.zeros(U.shape[1])
    for lam in lams:
        w = lasso_lifted_fista(C, U, lam, L=L)
        supp = w > 0
        Us = U[:, supp]
        Sig_S = sigma_noise ** 2 * np.eye(r) + (Us * w[supp][None, :]) @ Us.T
        bic = gaussian_cov_bic(Sig, Sig_S, N, dof=int(supp.sum()))
        if bic < best_bic:
            best_bic, best_w = bic, w
    return best_w > 0, best_w


def nnls_lifted_active_set(C: np.ndarray, U: np.ndarray, tol: float = 1e-10,
                           max_outer: int | None = None) -> np.ndarray:
    """Exact matrix-free Lawson-Hanson NNLS.  Passive-set normal equations use
    only the small ``|P| x |P|`` gram ``(G o G)[P, P] = (U[:,P]^T U[:,P])^2``.
    Gives exact zeros; ``~|S|`` outer iterations.  For single large instances
    and as the ground-truth cross-check of FISTA."""
    p = U.shape[1]
    if max_outer is None:
        max_outer = 3 * p + 10
    b0 = lifted_adjoint(U, C)               # A^T C  = A^T s
    w = np.zeros(p)
    passive = np.zeros(p, dtype=bool)
    scale = max(1.0, float(np.max(np.abs(b0))))

    def solve_passive(idx):
        Up = U[:, idx]
        GG = (Up.T @ Up) ** 2               # (|P|,|P|)
        # small ridge guards near-singular tetrahedral blocks
        z = np.linalg.solve(GG + 1e-12 * np.eye(len(idx)) * np.trace(GG) / max(1, len(idx)),
                            b0[idx])
        return z

    for _ in range(max_outer):
        grad = b0 - lifted_adjoint(U, lifted_apply(U, w))   # A^T (s - A w)
        cand = np.where(~passive)[0]
        if cand.size == 0 or np.max(grad[cand]) <= tol * scale:
            break
        j = cand[int(np.argmax(grad[cand]))]
        passive[j] = True
        # inner loop
        for _ in range(max_outer):
            idx = np.where(passive)[0]
            z_p = solve_passive(idx)
            if np.all(z_p > tol):
                w = np.zeros(p)
                w[idx] = z_p
                break
            # move as far as possible keeping w >= 0
            neg = z_p <= tol
            wi = w[idx]
            ratios = wi[neg] / (wi[neg] - z_p[neg])
            alpha = float(np.min(ratios))
            w_new = np.zeros(p)
            w_new[idx] = wi + alpha * (z_p - wi)
            w = w_new
            drop = idx[(w[idx] <= tol)]
            passive[drop] = False
            if drop.size == 0:              # numerical safeguard
                passive[idx[neg][0]] = False
    return w


def nnls_lifted_support_mf(Z: np.ndarray, U: np.ndarray, sigma_noise: float,
                           threshold: float, solver: str = "fista",
                           L: float | None = None,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Drop-in matrix-free replacement for
    :func:`tfl.estimators.nnls_lifted_support` (same signature + optional
    ``solver`` / ``L``).  ``solver in {'fista','active_set'}``."""
    r, N = Z.shape
    C = (Z @ Z.T) / N - sigma_noise ** 2 * np.eye(r)
    if solver == "active_set":
        w_hat = nnls_lifted_active_set(C, U)
    else:
        w_hat = nnls_lifted_fista(C, U, L=L)
    return w_hat > threshold, w_hat
