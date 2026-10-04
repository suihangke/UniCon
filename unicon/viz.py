"""Plotting helpers for the paper figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402


def _np(a):
  return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def plot_alignment_tsne(z1, z2, similarity, labels=None, title="", path=None,
                        perplexity=30, seed=42):
  """t-SNE of both modalities in the shared space (Figure 2).

  Crosses: modality 1, circles: modality 2, colors: ground-truth clusters,
  lines connect each x_i to its retrieved y (solid = correct, dashed = wrong).
  """
  z1, z2, similarity = _np(z1), _np(z2), _np(similarity)
  emb = TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(
    np.vstack([z1, z2]))
  e1, e2 = emb[:len(z1)], emb[len(z1):]
  match = similarity.argmax(axis=1)
  correct = match == np.arange(len(match))

  labels = np.zeros(len(z1), dtype=int) if labels is None else _np(labels)
  colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
  fig, ax = plt.subplots(figsize=(7, 6))
  for i, j in enumerate(match):
    ax.plot([e1[i, 0], e2[j, 0]], [e1[i, 1], e2[j, 1]], "g-" if correct[i] else "r--",
            alpha=0.4, linewidth=1)
  for c, lab in enumerate(np.unique(labels)):
    m = labels == lab
    ax.scatter(e1[m, 0], e1[m, 1], color=colors[c], marker="x", s=60, label=f"mod 1, cluster {lab}")
    ax.scatter(e2[m, 0], e2[m, 1], color=colors[c], marker="o", s=60, alpha=0.6,
               label=f"mod 2, cluster {lab}")
  ax.set_title(f"{title} (matching acc {correct.mean():.2f})")
  ax.set_xticks([])
  ax.set_yticks([])
  ax.legend(fontsize=8, loc="best")
  fig.tight_layout()
  if path:
    fig.savefig(path, dpi=150)
  plt.close(fig)
  return fig


def plot_S_gamma_heatmaps(S_list, titles, path=None):
  """Evolution of S(gamma) across fixed-point iterations (Figure 3)."""
  fig, axes = plt.subplots(1, len(S_list), figsize=(4 * len(S_list), 3.6))
  axes = np.atleast_1d(axes)
  for ax, S, title in zip(axes, S_list, titles):
    im = ax.imshow(_np(S), cmap="viridis")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
  fig.tight_layout()
  if path:
    fig.savefig(path, dpi=150)
  plt.close(fig)
  return fig


def plot_curve(xs, ys, xlabel, ylabel, title="", path=None):
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.plot(xs, ys, marker="o", markersize=2)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  if path:
    fig.savefig(path, dpi=150)
  plt.close(fig)
  return fig
