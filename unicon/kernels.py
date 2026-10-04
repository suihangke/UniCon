"""Kernels k(x, y) used by kernelized UniCon. All return an (n_x, n_y) Gram matrix."""

import math

import torch
import torch.nn.functional as F

_EPS = 1e-5


def _cosine(x, y):
  return (F.normalize(x, dim=1) @ F.normalize(y, dim=1).T).clamp(-1 + _EPS, 1 - _EPS)


def _sq_dist(x, y):
  d = (x ** 2).sum(1, keepdim=True) - 2 * x @ y.T + (y ** 2).sum(1, keepdim=True).T
  return d.clamp(min=0.0)


def linear_kernel(x, y):
  return x @ y.T


def angular_kernel(x, y):
  """k(x, y) = 1 - theta / pi, theta = angle(x, y). Default kernel of UniCon."""
  return 1 - torch.arccos(_cosine(x, y)) / math.pi


def arccos_kernel(x, y):
  """First-order arc-cosine kernel on normalized inputs: (sin theta + (pi - theta) cos theta) / pi."""
  cos = _cosine(x, y)
  theta = torch.arccos(cos)
  return (torch.sin(theta) + (math.pi - theta) * cos) / math.pi


def cosine_kernel(x, y):
  return _cosine(x, y)


def exp_cosine_kernel(x, y, beta=1.0):
  return torch.exp(beta * _cosine(x, y))


def rbf_kernel(x, y, gamma=1.0):
  return torch.exp(-gamma * _sq_dist(x, y))


def matern_kernel(x, y, lengthscale=1.0):
  """Matern kernel with nu = 3/2."""
  r = torch.sqrt(_sq_dist(x, y) + 1e-12) / lengthscale
  return (1 + math.sqrt(3) * r) * torch.exp(-math.sqrt(3) * r)


KERNELS = {
  "linear": linear_kernel,
  "angular": angular_kernel,
  "arccos": arccos_kernel,
  "cosine": cosine_kernel,
  "exp_cosine": exp_cosine_kernel,
  "rbf": rbf_kernel,
  "matern": matern_kernel,
}


def get_kernel(name):
  if name not in KERNELS:
    raise ValueError(f"Unknown kernel '{name}'. Choose from {sorted(KERNELS)}.")
  return KERNELS[name]
