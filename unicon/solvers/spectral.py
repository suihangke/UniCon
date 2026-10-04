"""Rank-r spectral factorization used by the closed-form UniCon update (Theorems 8 and 9)."""

import torch


def top_r_svd(matrix: torch.Tensor, r: int, method: str = "full", oversample: int = 10,
              n_power_iter: int = 4):
  """Return (U_r, sigma_r, V_r) with matrix ~= U_r diag(sigma_r) V_r^T.

  ``method='randomized'`` uses randomized SVD with power iterations (Halko et al.,
  2011), which is the stabilized variant of Appendix C.4.
  """
  if method == "full":
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U[:, :r], S[:r], Vh[:r].T
  if method == "randomized":
    q = min(r + oversample, *matrix.shape)
    U, S, V = torch.svd_lowrank(matrix, q=q, niter=n_power_iter)
    return U[:, :r], S[:r], V[:, :r]
  raise ValueError(f"Unknown SVD method '{method}'.")


def factorize(matrix: torch.Tensor, r: int, **svd_kwargs):
  """Split the best rank-r approximation into two factors: matrix_r = L @ R^T.

  L = U_r sqrt(Sigma_r), R = V_r sqrt(Sigma_r).
  """
  U, S, V = top_r_svd(matrix, r, **svd_kwargs)
  sqrt_s = S.clamp(min=0).sqrt()
  return U * sqrt_s, V * sqrt_s


def psd_power(K: torch.Tensor, power: float, ridge: float = 0.0, eps: float = 1e-10):
  """K^power for a symmetric PSD matrix via eigendecomposition, with Tikhonov ridge K + ridge*I.

  Negative powers use the Moore-Penrose convention: eigenvalues below ``eps`` map to 0.
  """
  K = (K + K.T) / 2
  if ridge > 0:
    K = K + ridge * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
  evals, evecs = torch.linalg.eigh(K)
  evals = evals.clamp(min=0)
  if power < 0:
    keep = evals > eps
    powered = torch.where(keep, evals.clamp(min=eps) ** power, torch.zeros_like(evals))
  else:
    powered = evals ** power
  return (evecs * powered) @ evecs.T
