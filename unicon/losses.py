"""Members of the generalized contrastive loss family (paper Eq. 2 / Eq. 23, Appendix A).

A loss is specified by (psi, phi, nu, eps). UniCon never evaluates the loss itself;
``ContrastiveLoss.s_gamma_kwargs`` holds the arguments of ``unicon.core.compute_S_gamma``.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

Fn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class ContrastiveLoss:
  psi: Fn
  diff_psi: Fn
  phi: Fn
  diff_phi: Fn
  nu: float = 1.0
  epsilon_ii: float = 1.0
  epsilon_ij: float = 1.0

  @property
  def s_gamma_kwargs(self):
    return dict(nu=self.nu, psi=self.psi, diff_psi=self.diff_psi, diff_phi=self.diff_phi,
                epsilon_ij=self.epsilon_ij, epsilon_ii=self.epsilon_ii)


def clip_loss(tau: float = 1.0, nu: float = 1.0, eps: float = 1e-8) -> ContrastiveLoss:
  """CLIP / symmetric InfoNCE: psi(x) = exp(x / tau), phi(x) = tau log x.

  nu = 1 recovers the CLIP loss exactly; nu > 1 up-weights the positive pair.
  """
  return ContrastiveLoss(
    psi=lambda x: torch.exp(x / tau),
    diff_psi=lambda x: torch.exp(x / tau) / tau,
    phi=lambda x: tau * torch.log(x + eps),
    diff_phi=lambda x: tau / (x + eps),
    nu=nu,
  )


def triplet_loss(margin: float = 0.5, tau: float = 1.0) -> ContrastiveLoss:
  """Smoothed triplet loss of Appendix C.7, with phi(x) = x and nu = 1.

  The margin term [margin - x]_+ (Appendix A) is smoothed with a softplus of temperature
  tau, and S(gamma) uses psi'(x) = tau * sigmoid((margin - x) / tau) with x = s_ij - s_ii,
  which pulls positive pairs together and pushes negatives apart. Because phi is the
  identity, only psi' enters S(gamma); psi below is the antiderivative of that psi'.
  """
  return ContrastiveLoss(
    psi=lambda x: -tau ** 2 * F.softplus((margin - x) / tau),
    diff_psi=lambda x: tau * torch.sigmoid((margin - x) / tau),
    phi=lambda x: x,
    diff_phi=lambda x: torch.ones_like(x),
  )


def sigmoid_pairwise_loss(tau: float = 1.0, nu: float = 1.5) -> ContrastiveLoss:
  """SigLIP-style member of the family (Appendix C.7): psi(x) = softplus(x / tau), phi = identity."""
  return ContrastiveLoss(psi=lambda x: F.softplus(x / tau),
                         diff_psi=lambda x: torch.sigmoid(x / tau) / tau,
                         phi=lambda x: x, diff_phi=lambda x: torch.ones_like(x), nu=nu)


def clip_loss_value(logits: torch.Tensor) -> torch.Tensor:
  """Symmetric cross-entropy on an (n, n) logit matrix whose diagonal holds positives."""
  labels = torch.arange(logits.shape[0], device=logits.device)
  return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2