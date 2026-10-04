"""The core computation of UniCon: the contrastive similarity weight matrix S(gamma).

Paper: Definition 3 (Eq. 3-6), Appendix B (Eq. 35-39) and Appendix C.1 (Listings 1-2).

Minimizing the generalized contrastive loss

  L = 1/(2n) sum_i 1/|P_x(i)| sum_{k in P_x(i)} phi( sum_{j not in P_x(i)} eps_ij psi(s_ij - nu s_ik)
                                                     + eps_ik psi(s_ik - nu s_ik) ) + (same for y)

is equivalent to maximizing tr(F_x S(gamma) F_y^T) (Lemma 4), so every UniCon
update is a spectral decomposition built from S(gamma):

  linear encoders   C(gamma) = X^T S(gamma) Y             -> top-r SVD (Theorem 8)
  kernel encoders   M        = K_x^1/2 S(gamma) K_y^1/2   -> top-r SVD (Theorem 9)

``s[i, j]`` is the hyperspherical similarity between x_i and y_j. Only psi, psi'
and phi' enter S(gamma); the defaults (psi = exp, phi = log, nu = 1) give the
CLIP / InfoNCE loss with temperature 1 (see ``unicon.losses`` for other losses).
"""

import torch


def _diff_log(x, eps=1e-8):
  return 1.0 / (x + eps)


def compute_S_gamma(s, nu=1.0, psi=torch.exp, diff_psi=torch.exp, diff_phi=_diff_log,
                    epsilon_ij=1.0, epsilon_ii=1.0):
  """S(gamma) for one-to-one pairs (x_i, y_i), i = 1..n (Listing 1).

  Args:
    s: (n, n) similarity matrix, s[i, j] = <f(x_i), f(y_j)> on the unit sphere.
    nu: weight of the positive pair inside the loss.
    psi, diff_psi: psi and its derivative.
    diff_phi: derivative of phi.
    epsilon_ij, epsilon_ii: weights of negative / positive pairs inside phi.

  Returns:
    (n, n) matrix with S_ii = beta_i / n and S_ij = -beta_ij / n, i != j.
  """
  n = s.size(0)

  # Build epsilon mask for weighting
  epsilon = torch.full_like(s, epsilon_ij)
  epsilon.fill_diagonal_(epsilon_ii)

  # Row-wise similarity terms: s_ij - nu * s_ii
  s_diag_row = torch.diag(s).unsqueeze(1).expand(-1, n)
  s_nu_row = s - nu * s_diag_row
  psi_terms = psi(s_nu_row)
  sum_psi_terms = torch.sum(epsilon * psi_terms, dim=1, keepdim=True)
  diff_phi_terms = diff_phi(sum_psi_terms)
  diff_psi_terms = diff_psi(s_nu_row)
  alpha = epsilon * diff_phi_terms * diff_psi_terms
  del s_nu_row, psi_terms, diff_psi_terms  # keeps peak memory low for large batches

  # Column-wise similarity terms: s_ji - nu * s_ii
  s_diag_col = torch.diag(s).expand(n, n)
  s_nu_col = s - nu * s_diag_col
  psi_terms_bar = psi(s_nu_col)
  sum_psi_terms_bar = torch.sum(epsilon * psi_terms_bar.T, dim=1, keepdim=True)
  diff_phi_terms_bar = diff_phi(sum_psi_terms_bar)
  diff_psi_terms_bar = diff_psi(s_nu_col.T)
  alpha_bar = epsilon * diff_phi_terms_bar * diff_psi_terms_bar
  del s_nu_col, psi_terms_bar, diff_psi_terms_bar, epsilon

  # Compute S_gamma weights
  S_ij = (alpha + alpha_bar.T) / 2
  S_i = nu * torch.sum((alpha + alpha_bar) / 2, dim=1) - torch.diag(alpha + alpha_bar) / 2

  S_gamma = -S_ij / n
  S_gamma[range(n), range(n)] = S_i / n
  return S_gamma


def compute_S_gamma_generalized(s, pos_mask, nu=1.0, psi=torch.exp, diff_psi=torch.exp,
                                diff_phi=_diff_log, epsilon_ij=1.0, epsilon_ii=1.0,
                                chunk_size=128):
  """S(gamma) for many-to-many pairs (Eq. 35-39, Listing 2).

  Args:
    s: (n, n) similarity matrix.
    pos_mask: (n, n) bool, pos_mask[i, j] = True iff (x_i, y_j) is a positive pair,
      i.e. j in P_x(i) and i in P_y(j). With pos_mask = I this equals compute_S_gamma.
    chunk_size: anchors processed together; memory is O(chunk_size * n^2).

  Returns:
    (n, n) matrix S_ij = -(gamma_ij / |P_x(i)| + gamma_bar_ji / |P_y(j)|) / (2n).
  """
  n = s.shape[0]
  pos_mask = pos_mask.bool()

  # Create epsilon matrix
  epsilon = torch.full_like(s, epsilon_ij)
  epsilon.fill_diagonal_(epsilon_ii)

  # gamma (anchors x_i, rows of s) and gamma_bar (anchors y_i, columns of s)
  terms = dict(nu=nu, psi=psi, diff_psi=diff_psi, diff_phi=diff_phi, chunk_size=chunk_size)
  gamma = _gamma(s, pos_mask, epsilon, **terms)
  gamma_bar = _gamma(s.T, pos_mask.T, epsilon.T, **terms)

  # Combine, normalizing by the number of positives |P_x(i)| and |P_y(j)|
  pos_mask_row_sum = pos_mask.sum(dim=1, keepdim=True).to(s.dtype)
  pos_mask_col_sum = pos_mask.sum(dim=0, keepdim=True).to(s.dtype)
  S_gamma = -(gamma / pos_mask_row_sum + gamma_bar.T / pos_mask_col_sum) / 2

  C_n = n  # Normalization constant
  return S_gamma / C_n


def _gamma(s, pos_mask, epsilon, nu, psi, diff_psi, diff_phi, chunk_size):
  """gamma_ij of Eq. 36 for the anchors on the rows of ``s``.

  Tensors indexed [i, j, m] hold, for anchor i, candidate j and summation index m.
  """
  n = s.shape[0]
  neg_mask = (~pos_mask).to(s.dtype)
  gamma = torch.empty_like(s)

  for start in range(0, n, chunk_size):
    i = slice(start, min(start + chunk_size, n))
    s_i_m = s[i].unsqueeze(1)  # s[i, m]
    s_i_j = s[i].unsqueeze(2)  # s[i, j]
    epsilon_i_m = (epsilon[i] * neg_mask[i]).unsqueeze(1)  # eps[i, m], m not in P_x(i)

    # sum_{m not in P(i)} eps_im psi(s_im - nu s_ij) and the same with psi'
    sum_psi = (epsilon_i_m * psi(s_i_m - nu * s_i_j)).sum(dim=2)
    sum_diff_psi = (epsilon_i_m * diff_psi(s_i_m - nu * s_i_j)).sum(dim=2)

    # phi'_ij (Eq. 38)
    diff_phi_ij = diff_phi(epsilon[i] * psi((1 - nu) * s[i]) + sum_psi)

    # gamma for positive samples, j in P(i)
    gamma_pos = diff_phi_ij * (
      epsilon[i] * (1 - nu) * diff_psi((1 - nu) * s[i]) - nu * sum_diff_psi
    )

    # gamma for negative samples: sum_{k in P(i)} phi'_ik eps_ij psi'(s_ij - nu s_ik)
    diff_phi_i_k = (diff_phi_ij * pos_mask[i]).unsqueeze(1)  # nonzero only for k in P(i)
    gamma_neg = epsilon[i] * (diff_phi_i_k * diff_psi(s_i_j - nu * s_i_m)).sum(dim=2)

    gamma[i] = torch.where(pos_mask[i], gamma_pos, gamma_neg)
  return gamma
