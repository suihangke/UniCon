"""Retrieval / matching metrics."""

import torch


def matching_accuracy(similarity: torch.Tensor) -> float:
  """Fraction of rows whose argmax is the diagonal (one-to-one top-1 matching)."""
  idx = similarity.argmax(dim=1)
  return (idx == torch.arange(len(idx), device=idx.device)).float().mean().item()


def recall_at_k(similarity: torch.Tensor, ks=(1, 5, 10), gt_mask: torch.Tensor = None) -> dict:
  """Recall@k for queries on the rows of ``similarity``.

  ``gt_mask[i, j]`` marks every correct item j for query i (defaults to the
  identity, i.e. one-to-one pairs). A query counts as a hit if any correct
  item ranks in the top-k.
  """
  if gt_mask is None:
    gt_mask = torch.eye(similarity.shape[0], similarity.shape[1], dtype=torch.bool,
                        device=similarity.device)
  topk = similarity.topk(max(ks), dim=1).indices
  hits = gt_mask.gather(1, topk)
  return {k: hits[:, :k].any(dim=1).float().mean().item() for k in ks}


def retrieval_metrics(sim_i2t: torch.Tensor, ks=(1, 5, 10), gt_mask: torch.Tensor = None) -> dict:
  """Image-to-text and text-to-image Recall@k from an (n_img, n_txt) similarity matrix."""
  gt_t = None if gt_mask is None else gt_mask.T
  i2t = recall_at_k(sim_i2t, ks, gt_mask)
  t2i = recall_at_k(sim_i2t.T, ks, gt_t)
  out = {}
  for k in ks:
    out[f"i2t_R@{k}"] = i2t[k]
    out[f"t2i_R@{k}"] = t2i[k]
    out[f"avg_R@{k}"] = (i2t[k] + t2i[k]) / 2
  return out


def format_metrics(metrics: dict) -> str:
  return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
