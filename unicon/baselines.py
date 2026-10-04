"""SGD-CLIP baseline: projection heads trained with the CLIP loss by mini-batch gradient descent."""

import copy
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from unicon.losses import clip_loss_value


class DualEncoder(nn.Module):
  """Two projection heads mapping each modality into a shared r-dim hypersphere."""

  def __init__(self, d1: int, d2: int, r: int, hidden: Optional[int] = None,
               activation: str = "none"):
    super().__init__()
    self.encoder1 = self._head(d1, r, hidden, activation)
    self.encoder2 = self._head(d2, r, hidden, activation)

  @staticmethod
  def _head(d_in, r, hidden, activation):
    act = {"none": nn.Identity, "tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    if hidden is None:
      return nn.Sequential(nn.Linear(d_in, r, bias=False), act())
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, r), act())

  def encode_x(self, x):
    return F.normalize(self.encoder1(x), dim=1)

  def encode_y(self, y):
    return F.normalize(self.encoder2(y), dim=1)

  def similarity(self, x, y):
    return self.encode_x(x) @ self.encode_y(y).T


def train_sgd_clip(model: DualEncoder, x, y, epochs: int, batch_size: int, lr: float = 2e-3,
                   tau: float = 1.0, optimizer: str = "adamw", weight_decay: float = 1e-2,
                   eval_fn: Optional[Callable[[DualEncoder], float]] = None,
                   eval_every: int = 1, keep_best: bool = False, seed: int = 0,
                   verbose: bool = True):
  """Train ``model`` on paired features with the symmetric CLIP loss.

  Args:
    optimizer: "adam", "adamw" or "sgd" (momentum 0.9).
    eval_fn: scores the model (higher is better) every ``eval_every`` epochs.
    keep_best: restore the parameters with the best ``eval_fn`` score at the end
      (model selection on a validation split).

  Returns a list of {epoch, loss, eval, time} records; ``time`` is cumulative training
  wall-clock time excluding evaluation.
  """
  params = model.parameters()
  if optimizer == "adam":
    opt = torch.optim.Adam(params, lr=lr)
  elif optimizer == "adamw":
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
  elif optimizer == "sgd":
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
  else:
    raise ValueError(f"Unknown optimizer '{optimizer}'.")

  g = torch.Generator().manual_seed(seed)
  n = x.shape[0]
  history, best = [], (-float("inf"), None)
  train_time = 0.0
  for epoch in range(1, epochs + 1):
    model.train()
    _sync(x)
    t0 = time.time()
    perm = torch.randperm(n, generator=g).to(x.device)
    total, count = 0.0, 0
    for b in range(0, n, batch_size):
      idx = perm[b:b + batch_size]
      if len(idx) < 2:
        continue
      loss = clip_loss_value(model.similarity(x[idx], y[idx]) / tau)
      opt.zero_grad()
      loss.backward()
      opt.step()
      total += loss.item()
      count += 1
    _sync(x)
    train_time += time.time() - t0

    if eval_fn is not None and (epoch % eval_every == 0 or epoch == epochs):
      model.eval()
      with torch.no_grad():
        score = float(eval_fn(model))
      history.append({"epoch": epoch, "loss": total / max(count, 1), "eval": score,
                      "time": train_time})
      if keep_best and score > best[0]:
        best = (score, copy.deepcopy(model.state_dict()))
      if verbose:
        print(f"epoch {epoch:4d}  loss={total / max(count, 1):.4f}  eval={score:.4f}  "
              f"time={train_time:.2f}s")
  if keep_best and best[1] is not None:
    model.load_state_dict(best[1])
  return history


def _sync(t):
  if t.is_cuda:
    torch.cuda.synchronize(t.device)
