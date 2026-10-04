"""UniCon: Unified Framework for Efficient Contrastive Alignment via Kernels (NeurIPS 2026).

unicon.core      S(gamma), the core computation (compute_S_gamma, compute_S_gamma_generalized)
unicon.solvers   closed-form spectral updates: LinearUniCon (Alg. 1), KernelUniCon (Alg. 2)
unicon.losses    members of the contrastive loss family (CLIP, triplet, sigmoid)
"""

from unicon.core import compute_S_gamma, compute_S_gamma_generalized
from unicon.losses import ContrastiveLoss, clip_loss, sigmoid_pairwise_loss, triplet_loss
from unicon.solvers import KernelUniCon, LinearUniCon

__all__ = [
  "compute_S_gamma",
  "compute_S_gamma_generalized",
  "ContrastiveLoss",
  "clip_loss",
  "sigmoid_pairwise_loss",
  "triplet_loss",
  "KernelUniCon",
  "LinearUniCon",
]
