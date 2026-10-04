"""Shared driver for the image-text retrieval experiments (Flickr30K, MSCOCO)."""

import torch

from unicon.baselines import DualEncoder, train_sgd_clip
from unicon.data.features import FeatureSplit
from unicon.solvers.linear import LinearUniCon, val_mean_recall_at_1
from unicon.losses import clip_loss
from unicon.eval.metrics import retrieval_metrics


def evaluate(model, split: FeatureSplit, ks=(1, 5, 10)):
  with torch.no_grad():
    if model is None:
      sim = split.image @ split.text.T
    else:
      sim = model.similarity(split.image, split.text)
  return retrieval_metrics(sim, ks)


def run_unicon(train: FeatureSplit, val: FeatureSplit, cfg: dict, verbose=True):
  """Fit linear UniCon with the hyperparameters in ``cfg`` (see the per-dataset presets).

  ``cfg["loss"]`` selects another member of the loss family; the default is the CLIP loss.
  """
  loss = cfg.get("loss") or clip_loss(tau=cfg["tau"], nu=cfg["nu"])
  model = LinearUniCon(r=cfg["r"], loss=loss, rho=cfg.get("rho", 1.0),
                       factorization=cfg.get("factorization", "orthonormal"),
                       ridge=cfg.get("ridge", 0.0), svd_method=cfg.get("svd_method", "full"),
                       scale_inputs=cfg.get("scale_inputs", False), seed=cfg.get("seed", 0))
  pos_mask_fn = None
  if cfg.get("many_to_many") and train.image_ids is not None:
    ids = train.image_ids
    pos_mask_fn = lambda idx: ids[idx][:, None] == ids[idx][None, :]
  model.fit(train.image, train.text, batch_size=cfg["batch_size"], n_iters=cfg["iters"],
            n_batches=cfg.get("n_batches"), val_fn=val_mean_recall_at_1(val.image, val.text),
            pos_mask_fn=pos_mask_fn, verbose=verbose)
  return model


def run_sgd_clip(train: FeatureSplit, val: FeatureSplit, cfg: dict, verbose=True):
  """Linear heads trained with the CLIP loss; the epoch with the best validation R@1 is kept."""
  torch.manual_seed(cfg.get("seed", 0))
  model = DualEncoder(train.image.shape[1], train.text.shape[1], cfg["r"]).to(train.image.device)
  val_fn = lambda m: evaluate(m, val, ks=(1,))["avg_R@1"]
  history = train_sgd_clip(model, train.image, train.text, epochs=cfg["epochs"],
                           batch_size=cfg["batch_size"], lr=cfg["lr"], tau=cfg.get("tau", 1.0),
                           optimizer=cfg.get("optimizer", "adam"),
                           weight_decay=cfg.get("weight_decay", 0.0), eval_fn=val_fn,
                           keep_best=True, seed=cfg.get("seed", 0), verbose=verbose)
  model.train_time = history[-1]["time"]
  model.history = history
  return model


def format_row(name, metrics, train_time=None, ks=(1, 10)):
  cells = [f"{name:<12s}"]
  cells.append(f"{train_time:9.2f}s" if train_time is not None else f"{'-':>10s}")
  for d in ("i2t", "t2i", "avg"):
    cells.append(" ".join(f"{metrics[f'{d}_R@{k}']:.3f}" for k in ks))
  return " | ".join(cells)


def header(ks=(1, 10)):
  rk = " ".join(f"R@{k:<3d}" for k in ks)
  return f"{'method':<12s} | {'train time':>10s} | I2T {rk} | T2I {rk} | Avg {rk}"
