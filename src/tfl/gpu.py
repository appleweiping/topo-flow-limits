"""GPU (torch) batched lifted-covariance NNLS + Monte-Carlo.

Ports the matrix-free FISTA of :mod:`tfl.estimators_mf` to torch and runs ``B``
Monte-Carlo trials as a leading batch dimension: sample ``B`` curl-domain
covariances on-device, solve all ``B`` NNLS problems with batched einsum GEMMs,
count exact recovery.  Only ``U`` (``r x p``) is resident; the ``(B, r, p)``
intermediate is the memory ceiling, so the batch is tiled.

Sampling is done directly in the curl domain: because ``Q^T`` annihilates the
gradient and candidate-orthogonal harmonic components exactly (paper §2), the
population curl covariance is ``sigma_n^2 I_r + U_S Gamma_S U_S^T`` and the
sample covariance ``(1/N) Z Z^T`` with ``Z = U_S (sigma_c Y) + sigma_n E`` is
the exact finite-``N`` estimator input (cross-checked vs
:func:`tfl.limits.second_order_covariance`).

Requires torch; import lazily so the CPU core stays torch-free.
"""
from __future__ import annotations

import numpy as np


def _torch():
    import torch
    return torch


def sample_curl_cov_batched(U, masks, sigma_c, sigma_n, N, generator=None):
    """Batched curl-domain covariance-signal ``C = Sigma_hat - sigma_n^2 I``.

    ``U`` (r,p) tensor; ``masks`` (B,p) bool tensor (true supports).  Returns
    ``C`` (B,r,r).  All on ``U.device`` / ``U.dtype``."""
    torch = _torch()
    r, p = U.shape
    B = masks.shape[0]
    dev, dt = U.device, U.dtype
    Y = torch.randn(B, p, N, device=dev, dtype=dt, generator=generator)
    Ym = Y * masks.unsqueeze(-1).to(dt)                    # (B,p,N)
    sig = sigma_c * torch.matmul(U, Ym)                    # (r,p)@(B,p,N)->(B,r,N)
    Z = sig + sigma_n * torch.randn(B, r, N, device=dev, dtype=dt, generator=generator)
    Sigma = torch.matmul(Z, Z.transpose(1, 2)) / N         # (B,r,r)
    C = Sigma - (sigma_n ** 2) * torch.eye(r, device=dev, dtype=dt).unsqueeze(0)
    return C


def _apply_batched(U, w):
    """A w = U diag(w) U^T, batched.  U (r,p), w (B,p) -> (B,r,r)."""
    torch = _torch()
    Uw = U.unsqueeze(0) * w.unsqueeze(1)                   # (B,r,p)
    return torch.matmul(Uw, U.t())                         # (B,r,r)


def _adjoint_batched(U, R):
    """A^T R = [u_tau^T R u_tau], batched.  U (r,p), R (B,r,r) -> (B,p)."""
    torch = _torch()
    RU = torch.matmul(R, U)                                # (B,r,p)
    return (U.unsqueeze(0) * RU).sum(dim=1)                # (B,p)


def lipschitz_ATA_torch(U, iters=40, seed=0):
    """lambda_max(A^T A) = lambda_max(G o G), matrix-free power iteration."""
    torch = _torch()
    p = U.shape[1]
    g = torch.Generator(device=U.device).manual_seed(seed)
    v = torch.randn(1, p, device=U.device, dtype=U.dtype, generator=g)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        Av = _adjoint_batched(U, _apply_batched(U, v))     # (1,p)
        nrm = Av.norm()
        if nrm == 0:
            return 0.0
        v = Av / nrm
        lam = float((v * _adjoint_batched(U, _apply_batched(U, v))).sum())
    return lam


def nnls_lifted_fista_batched(C, U, L=None, max_iter=2000, tol=1e-9,
                              restart=True):
    """Batched FISTA for ``min_{w>=0} 1/2 ||C_b - A w_b||_F^2``.
    ``C`` (B,r,r), ``U`` (r,p) -> ``w_hat`` (B,p)."""
    torch = _torch()
    B, r, _ = C.shape
    p = U.shape[1]
    if L is None:
        L = lipschitz_ATA_torch(U)
    if L <= 0:
        return torch.zeros(B, p, device=U.device, dtype=U.dtype)
    b0 = _adjoint_batched(U, C)                            # (B,p)
    w = torch.zeros(B, p, device=U.device, dtype=U.dtype)
    y = torch.zeros_like(w)
    t = 1.0
    for _ in range(max_iter):
        grad = _adjoint_batched(U, _apply_batched(U, y)) - b0
        w_new = torch.clamp(y - grad / L, min=0.0)
        if restart and float(((y - w_new) * (w_new - w)).sum()) > 0.0:
            t = 1.0
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = w_new + ((t - 1.0) / t_new) * (w_new - w)
        step = (w_new - w).norm().item()
        w, t = w_new, t_new
        if step <= tol * max(1.0, w.norm().item()):
            break
    return w


def exact_recovery_rate(w_hat, masks, w_min):
    """Fraction of trials with exact support recovery at threshold w_min/2."""
    support = w_hat > (w_min / 2.0)
    hits = (support == masks).all(dim=1)
    return float(hits.float().mean().item()), int(hits.sum().item())


def mc_recovery_grid(U_np, k, rho2, N, B_total, B_tile, device="cuda",
                     dtype="float32", sigma_n=1.0, seed=0, max_iter=1500):
    """Run ``B_total`` GPU Monte-Carlo trials in tiles of ``B_tile``; return
    exact-recovery rate + Wilson CI + peak GPU memory.

    ``U_np`` (r,p) numpy; random size-``k`` supports per trial; isotropic
    excitation ``sigma_c^2 = rho2 sigma_n^2``.  Returns a dict."""
    torch = _torch()
    dt = getattr(torch, dtype)
    dev = torch.device(device)
    U = torch.tensor(U_np, device=dev, dtype=dt)
    r, p = U.shape
    sc = float(np.sqrt(rho2)) * sigma_n
    w_min = rho2 * sigma_n ** 2
    L = lipschitz_ATA_torch(U)
    g = torch.Generator(device=dev).manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    hits = 0
    done = 0
    rng = np.random.default_rng(seed)
    import time as _time
    t0 = _time.perf_counter()
    while done < B_total:
        b = min(B_tile, B_total - done)
        # random supports (cpu) -> mask tensor
        masks = torch.zeros(b, p, dtype=torch.bool, device=dev)
        idx = np.stack([rng.choice(p, k, replace=False) for _ in range(b)])
        rows = torch.arange(b, device=dev).unsqueeze(1).expand(b, k)
        masks[rows.reshape(-1), torch.tensor(idx.reshape(-1), device=dev)] = True
        C = sample_curl_cov_batched(U, masks, sc, sigma_n, N, generator=g)
        w_hat = nnls_lifted_fista_batched(C, U, L=L, max_iter=max_iter)
        _, h = exact_recovery_rate(w_hat, masks, w_min)
        hits += h
        done += b
        del C, w_hat, masks
        if dev.type == "cuda":
            torch.cuda.synchronize()
    wall = _time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated(dev) / 1e6) if dev.type == "cuda" else float("nan")
    # Wilson CI
    z = 1.96
    ph = hits / done
    den = 1 + z**2 / done
    ctr = (ph + z**2 / (2 * done)) / den
    half = z * np.sqrt(ph * (1 - ph) / done + z**2 / (4 * done**2)) / den
    return {
        "p": int(p), "r": int(r), "k": int(k), "rho2": rho2, "N": int(N),
        "B_total": int(done), "B_tile": int(B_tile),
        "recovery": ph, "recovery_ci95": [max(0.0, ctr - half), min(1.0, ctr + half)],
        "lipschitz": float(L), "wall_s": wall, "peak_gpu_mb": peak_mb,
        "device": str(dev), "dtype": dtype,
    }
