"""Synthetic latent-factor models (Section 4.1, Appendix C.2)."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import ortho_group


@dataclass
class PairedSplit:
  x: torch.Tensor
  y: torch.Tensor
  labels: torch.Tensor = None


def _to_tensor(a):
  return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)


def linear_latent_factor_data(d1=40, d2=30, r=10, n_clusters=3, n_train=600, n_val=100,
                              n_test=100, snr=0.1, cluster_spread=0.6, center_range=2.0,
                              seed=42):
  """Linear model x = U1 z + xi1, y = U2 z + xi2 with z drawn around K cluster centers.

  U1, U2 are the first r columns of Haar-random orthogonal matrices; xi ~ N(0, snr^2).
  Every split draws its own cluster centers (as in the original experiment), so the
  test set does not share clusters with the training set.
  """
  rng = np.random.default_rng(seed)
  U1 = ortho_group.rvs(d1, random_state=rng)[:, :r]
  U2 = ortho_group.rvs(d2, random_state=rng)[:, :r]

  def split(n):
    centers = rng.uniform(-center_range, center_range, size=(n_clusters, r))
    labels = np.repeat(np.arange(n_clusters), int(np.ceil(n / n_clusters)))[:n]
    z = centers[labels] + rng.normal(0, cluster_spread, size=(n, r))
    x = z @ U1.T + rng.normal(0, snr, size=(n, d1))
    y = z @ U2.T + rng.normal(0, snr, size=(n, d2))
    return PairedSplit(_to_tensor(x), _to_tensor(y), torch.tensor(labels))

  return split(n_train), split(n_val), split(n_test)


def nonlinear_latent_factor_data(d1=40, d2=39, r=10, n_train=1000, n_val=100, n_test=100,
                                 snr=0.3, seed=0):
  """Nonlinear model x = tanh(U3 tanh(U1 z)) + xi1, y = tanh(U4 tanh(U2 z)) + xi2.

  U1, U2: first r columns of Haar-random orthogonal matrices; U3, U4: full orthogonal
  matrices; z ~ N(0, I_r); xi ~ N(0, snr^2).
  """
  rng = np.random.default_rng(seed)
  U1 = ortho_group.rvs(d1, random_state=rng)[:, :r]
  U2 = ortho_group.rvs(d2, random_state=rng)[:, :r]
  U3 = ortho_group.rvs(d1, random_state=rng)
  U4 = ortho_group.rvs(d2, random_state=rng)

  def split(n):
    z = rng.normal(0, 1, size=(n, r))
    x = np.tanh(np.tanh(z @ U1.T) @ U3.T) + rng.normal(0, snr, size=(n, d1))
    y = np.tanh(np.tanh(z @ U2.T) @ U4.T) + rng.normal(0, snr, size=(n, d2))
    return PairedSplit(_to_tensor(x), _to_tensor(y))

  return split(n_train), split(n_val), split(n_test)
