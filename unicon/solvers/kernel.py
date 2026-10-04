"""Kernelized UniCon (Section 3.2, Theorem 9, Corollary 10, Algorithm 2).

For a reference batch (X_b, Y_b) the encoders live in the RKHS spanned by the
batch: f_x(x) = A^T kappa_X(x), f_y(y) = B^T kappa_Y(y), A, B in R^{n x r}.
The optimum satisfies

  A B^T = (1/rho) K_X^{-1/2} [K_X^{1/2} S(gamma) K_Y^{1/2}]_r K_Y^{-1/2}.

S(gamma) depends on (A, B), so each batch runs a short fixed-point iteration.
Batch solutions are fused at inference by validation-accuracy weights.
"""

import time
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

from unicon.kernels import get_kernel
from unicon.losses import ContrastiveLoss, clip_loss
from unicon.eval.metrics import matching_accuracy
from unicon.core import compute_S_gamma
from unicon.solvers.spectral import factorize, psd_power


class KernelUniCon:
  def __init__(self, r: int, kernel: Union[str, Callable] = "angular",
               loss: Optional[ContrastiveLoss] = None, rho: float = 1.0,
               n_iters: int = 10, tol: float = 1e-3, ridge: float = 1e-6,
               use_kernel_sqrt: bool = True, init: str = "random", seed: int = 0):
    """
    Args:
      r: shared embedding dimension.
      kernel: kernel name from ``unicon.kernels.KERNELS`` or a callable k(x, y).
      n_iters, tol: per-batch fixed-point iterations and stopping tolerance on ||AB^T||.
      ridge: Tikhonov term for K^{+-1/2} (Section 3.4, numerical stability).
      use_kernel_sqrt: if True, apply the exact Theorem 9 update (SVD of
        K_X^{1/2} S K_Y^{1/2}); if False, take the SVD of S(gamma) directly.
      init: initial similarity for S(gamma): "random" (Gaussian) or "zeros".
    """
    self.r = r
    self.kernel = get_kernel(kernel) if isinstance(kernel, str) else kernel
    self.loss = loss or clip_loss()
    self.rho = rho
    self.n_iters = n_iters
    self.tol = tol
    self.ridge = ridge
    self.use_kernel_sqrt = use_kernel_sqrt
    self.init = init
    self.seed = seed
    self.batches = []  # list of (X_b, Y_b, A, B)
    self.weights = None
    self.S_history = []

  def _initial_similarity(self, n, device, dtype, generator):
    if self.init == "zeros":
      return torch.zeros(n, n, device=device, dtype=dtype)
    return torch.randn(n, n, generator=generator, dtype=dtype).to(device)

  @torch.no_grad()
  def fit_batch(self, xb, yb, generator=None, record_S=False):
    """Fixed-point iteration of Algorithm 2 on one reference batch. Returns (A, B, n_steps)."""
    n = xb.shape[0]
    Kx, Ky = self.kernel(xb, xb), self.kernel(yb, yb)
    if self.use_kernel_sqrt:
      Kx_half, Ky_half = psd_power(Kx, 0.5, self.ridge), psd_power(Ky, 0.5, self.ridge)
      Kx_inv_half, Ky_inv_half = psd_power(Kx, -0.5, self.ridge), psd_power(Ky, -0.5, self.ridge)

    s = self._initial_similarity(n, xb.device, xb.dtype, generator)
    AB_prev = None
    S_trace = []
    for step in range(1, self.n_iters + 1):
      S = compute_S_gamma(s, **self.loss.s_gamma_kwargs) / self.rho
      if record_S:
        S_trace.append(S.cpu())
      if self.use_kernel_sqrt:
        U, V = factorize(Kx_half @ S @ Ky_half, self.r)
        A, B = Kx_inv_half @ U, Ky_inv_half @ V
      else:
        A, B = factorize(S, self.r)
      fx = F.normalize(A.T @ Kx, dim=0)
      fy = F.normalize(B.T @ Ky, dim=0)
      s = fx.T @ fy
      AB = A @ B.T
      if AB_prev is not None and torch.norm(AB - AB_prev) < self.tol:
        break
      AB_prev = AB
    if record_S:
      self.S_history.append(S_trace)
    return A, B, step

  @torch.no_grad()
  def fit(self, x, y, batch_size: int, x_val=None, y_val=None,
          val_fn: Optional[Callable] = None, record_S: bool = False, verbose: bool = False):
    """Fit one kernel solution per batch and weight them by validation accuracy.

    ``val_fn(sim)`` scores an (n_val, n_val) similarity matrix; defaults to top-1 matching
    accuracy. Without validation data all batches get equal weight.
    """
    g = torch.Generator().manual_seed(self.seed)
    val_fn = val_fn or matching_accuracy
    self.batches, scores = [], []
    start = time.time()
    for b in range(0, x.shape[0] - batch_size + 1, batch_size):
      xb, yb = x[b:b + batch_size], y[b:b + batch_size]
      A, B, steps = self.fit_batch(xb, yb, generator=g, record_S=record_S)
      self.batches.append((xb, yb, A, B))
      if x_val is not None:
        score = val_fn(self._batch_similarity(len(self.batches) - 1, x_val, y_val))
      else:
        score = 1.0
      scores.append(score)
      if verbose:
        print(f"batch {len(self.batches) - 1}: steps={steps} val={score:.4f}")
    w = torch.tensor(scores, dtype=x.dtype, device=x.device).clamp(min=0)
    self.weights = w / w.sum() if w.sum() > 0 else torch.full_like(w, 1 / len(w))
    self.train_time = time.time() - start
    return self

  def _batch_embed(self, i, x=None, y=None):
    xb, yb, A, B = self.batches[i]
    zx = None if x is None else F.normalize(A.T @ self.kernel(xb, x), dim=0).T
    zy = None if y is None else F.normalize(B.T @ self.kernel(yb, y), dim=0).T
    return zx, zy

  def _batch_similarity(self, i, x, y):
    zx, zy = self._batch_embed(i, x, y)
    return zx @ zy.T

  @torch.no_grad()
  def similarity(self, x, y):
    """Accuracy-weighted fusion of the per-batch similarities (Corollary 10)."""
    return sum(w * self._batch_similarity(i, x, y) for i, w in enumerate(self.weights))

  @torch.no_grad()
  def embed(self, x=None, y=None):
    """Weighted average of the per-batch embeddings (used for visualization)."""
    zx = zy = 0
    for i, w in enumerate(self.weights):
      ex, ey = self._batch_embed(i, x, y)
      zx = zx + (0 if ex is None else w * ex)
      zy = zy + (0 if ey is None else w * ey)
    return zx, zy
