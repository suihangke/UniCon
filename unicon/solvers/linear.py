"""UniCon with linear encoders (Section 3.1, Algorithm 1).

Encoders f_x(x) = F1 x and f_y(y) = F2 y. The spectral update sets
F1^T F2 from the top-r SVD of C(gamma) = X^T S(gamma) Y (Theorem 8). Because
S(gamma) depends on the current similarities, the update is iterated. For large
datasets each mini-batch yields its own closed-form solution G_b = F1^T F2; the
solutions are averaged with validation-score weights and factorized once more.
"""

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from unicon.losses import ContrastiveLoss, clip_loss
from unicon.eval.metrics import matching_accuracy, retrieval_metrics
from unicon.core import compute_S_gamma, compute_S_gamma_generalized
from unicon.solvers.spectral import top_r_svd


class LinearUniCon:
  def __init__(self, r: int, loss: Optional[ContrastiveLoss] = None, rho: float = 1.0,
               factorization: str = "sqrt", ridge: float = 0.0, svd_method: str = "full",
               scale_inputs: bool = False, seed: int = 0):
    """
    Args:
      r: shared embedding dimension.
      loss: contrastive loss defining S(gamma); CLIP loss with tau=1 by default.
      rho: strength of R(F1, F2) = rho/2 ||F1^T F2||_F^2; only rescales the solution.
      factorization: how F1^T F2 = U_r Sigma_r V_r^T is split into encoders.
        "sqrt":        F1 = Sigma_r^{1/2} U_r^T, F2 = Sigma_r^{1/2} V_r^T (Theorem 8).
        "orthonormal": F1 = U_r^T, F2 = V_r^T, i.e. project onto the top-r singular
                       subspaces with equal weight (used for all real-data results).
      ridge: Tikhonov term lambda added to the leading diagonal of the aggregated solution
        G = F1^T F2 before its final SVD (Appendix C.4). Under the "orthonormal"
        factorization G has unit singular values, so a large ridge keeps the solution
        close to the identity map (useful when both inputs already share a space, e.g. CLIP).
      svd_method: "full" or "randomized" (Appendix C.4).
      scale_inputs: build C(gamma) from x_i / ||F1 x_i|| and y_j / ||F2 y_j||, i.e. the
        hyperspherical similarity's linearization at the current encoders.
    """
    if factorization not in ("sqrt", "orthonormal"):
      raise ValueError(f"Unknown factorization '{factorization}'.")
    self.r = r
    self.loss = loss or clip_loss()
    self.rho = rho
    self.factorization = factorization
    self.ridge = ridge
    self.svd_method = svd_method
    self.scale_inputs = scale_inputs
    self.seed = seed
    self.F1 = None
    self.F2 = None
    self.history = []

  def init_encoders(self, d1: int, d2: int, device=None, dtype=torch.float32):
    """Same initialization as torch.nn.Linear (Kaiming-uniform)."""
    g = torch.Generator().manual_seed(self.seed)
    self.F1 = ((torch.rand(self.r, d1, generator=g, dtype=dtype) * 2 - 1) / d1 ** 0.5).to(device)
    self.F2 = ((torch.rand(self.r, d2, generator=g, dtype=dtype) * 2 - 1) / d2 ** 0.5).to(device)

  def encode_x(self, x):
    return F.normalize(x @ self.F1.T, dim=1)

  def encode_y(self, y):
    return F.normalize(y @ self.F2.T, dim=1)

  @torch.no_grad()
  def similarity(self, x, y, chunk_size: int = 8192):
    zy = self.encode_y(y)
    return torch.cat([self.encode_x(x[i:i + chunk_size]) @ zy.T
                      for i in range(0, x.shape[0], chunk_size)])

  @torch.no_grad()
  def batch_C_gamma(self, xb, yb, pos_mask=None):
    """C(gamma) = X^T S(gamma) Y / rho for one batch at the current encoders."""
    gx, gy = xb @ self.F1.T, yb @ self.F2.T
    s = F.normalize(gx, dim=1) @ F.normalize(gy, dim=1).T
    if pos_mask is None:
      S = compute_S_gamma(s, **self.loss.s_gamma_kwargs)
    else:
      S = compute_S_gamma_generalized(s, pos_mask, **self.loss.s_gamma_kwargs)
    if self.scale_inputs:
      xb = xb / gx.norm(dim=1, keepdim=True).clamp(min=1e-12)
      yb = yb / gy.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return xb.T @ S @ yb / self.rho

  @torch.no_grad()
  def set_from_G(self, G, ridge: float = 0.0):
    """Closed-form update from a target F1^T F2 = G (rank-r SVD, Theorem 8)."""
    if ridge > 0:
      G = G.clone()
      k = min(G.shape)
      G[range(k), range(k)] += ridge
    U, S, V = top_r_svd(G, self.r, method=self.svd_method)
    if self.factorization == "sqrt":
      sqrt_s = S.clamp(min=0).sqrt()
      self.F1, self.F2 = (U * sqrt_s).T.contiguous(), (V * sqrt_s).T.contiguous()
    else:
      self.F1, self.F2 = U.T.contiguous(), V.T.contiguous()

  @property
  def G(self):
    return self.F1.T @ self.F2

  @torch.no_grad()
  def fit(self, x, y, batch_size: int, n_iters: int = 2, n_batches: Optional[int] = None,
          val_fn: Optional[Callable[["LinearUniCon"], float]] = None,
          pos_mask_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
          verbose: bool = True):
    """Algorithm 1.

    Args:
      x, y: (N, d1), (N, d2) paired training features (row i of x pairs with row i of y).
      batch_size: rows per mini-batch; a trailing partial batch is dropped.
      n_iters: outer fixed-point iterations (S(gamma) is recomputed at every iteration).
      n_batches: use only the first ``n_batches`` batches (data-efficiency setting).
      val_fn: scores a candidate model (higher is better). Each batch solution is weighted
        by its validation score; without ``val_fn`` the weights are uniform.
      pos_mask_fn: maps batch row indices to an (n, n) positive mask (many-to-many data).
    """
    if self.F1 is None:
      self.init_encoders(x.shape[1], y.shape[1], device=x.device, dtype=x.dtype)
    total = x.shape[0] // batch_size
    n_batches = total if n_batches is None else min(n_batches, total)
    if n_batches == 0:
      raise ValueError(f"batch_size={batch_size} exceeds the {x.shape[0]} training pairs.")
    _sync(x)
    start = time.time()

    for it in range(n_iters):
      F1, F2 = self.F1, self.F2
      G_sum, weight_sum = 0.0, 0.0
      for b in range(n_batches):
        idx = torch.arange(b * batch_size, (b + 1) * batch_size, device=x.device)
        self.F1, self.F2 = F1, F2
        C_b = self.batch_C_gamma(x[idx], y[idx], None if pos_mask_fn is None else pos_mask_fn(idx))
        self.set_from_G(C_b)
        w = max(float(val_fn(self)), 0.0) if val_fn is not None and n_batches > 1 else 1.0
        G_sum = G_sum + w * self.G
        weight_sum += w
      self.set_from_G(G_sum / max(weight_sum, 1e-12), ridge=self.ridge)

      _sync(x)
      record = {"iter": it + 1, "time": time.time() - start}
      if val_fn is not None:
        record["val"] = float(val_fn(self))
      self.history.append(record)
      if verbose:
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in record.items()))
    self.train_time = self.history[-1]["time"]
    return self


def _sync(t):
  if t.is_cuda:
    torch.cuda.synchronize(t.device)


def val_matching_accuracy(x_val, y_val):
  """Validation scorer: one-to-one top-1 matching accuracy."""
  return lambda model: matching_accuracy(model.similarity(x_val, y_val))


def val_mean_recall_at_1(x_val, y_val):
  """Validation scorer: mean of image-to-text and text-to-image Recall@1."""
  return lambda model: retrieval_metrics(model.similarity(x_val, y_val), ks=(1,))["avg_R@1"]
