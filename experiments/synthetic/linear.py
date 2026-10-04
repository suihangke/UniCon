"""Section 4.1 / Figure 2: linear latent-factor model, UniCon vs. SGD-CLIP.

python experiments/synthetic/linear.py --out outputs/synthetic_linear
"""

import argparse
import os
import time

import torch

from unicon.baselines import DualEncoder, train_sgd_clip
from unicon.data.synthetic import linear_latent_factor_data
from unicon.solvers.linear import LinearUniCon
from unicon.losses import clip_loss
from unicon.eval.metrics import matching_accuracy
from unicon.utils import save_json, set_seed
from unicon.viz import plot_alignment_tsne, plot_curve


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--d1", type=int, default=40)
  p.add_argument("--d2", type=int, default=30)
  p.add_argument("--r", type=int, default=10)
  p.add_argument("--n-clusters", type=int, default=3)
  p.add_argument("--n-train", type=int, default=600)
  p.add_argument("--n-test", type=int, default=100)
  p.add_argument("--snr", type=float, default=0.1, help="std of the observation noise")
  p.add_argument("--batch-size", type=int, default=30)
  p.add_argument("--tau", type=float, default=1.0)
  p.add_argument("--unicon-iters", type=int, default=1)
  p.add_argument("--sgd-epochs", type=int, default=400)
  p.add_argument("--sgd-lr", type=float, default=2e-3)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--out", default="outputs/synthetic_linear")
  p.add_argument("--no-plots", action="store_true")
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  set_seed(args.seed)

  train, val, test = linear_latent_factor_data(
    d1=args.d1, d2=args.d2, r=args.r, n_clusters=args.n_clusters, n_train=args.n_train,
    n_test=args.n_test, snr=args.snr, seed=args.seed)

  # UniCon: closed-form spectral update on C(gamma), aggregated over mini-batches
  unicon = LinearUniCon(r=args.r, loss=clip_loss(args.tau), seed=args.seed)
  t0 = time.time()
  unicon.fit(train.x, train.y, batch_size=args.batch_size, n_iters=args.unicon_iters,
             verbose=False)
  unicon_time = time.time() - t0
  unicon_sim = unicon.similarity(test.x, test.y)
  unicon_acc = matching_accuracy(unicon_sim)
  print(f"UniCon    test matching acc = {unicon_acc:.4f}  train time = {unicon_time:.3f}s")

  # SGD-CLIP baseline: linear heads, CLIP loss, AdamW
  torch.manual_seed(args.seed)
  clip = DualEncoder(args.d1, args.d2, args.r)
  history = train_sgd_clip(
    clip, train.x, train.y, epochs=args.sgd_epochs, batch_size=args.batch_size, lr=args.sgd_lr,
    tau=args.tau, eval_fn=lambda m: matching_accuracy(m.similarity(test.x, test.y)),
    seed=args.seed, verbose=False)
  final = history[-1]
  reach = next((h for h in history if h["eval"] >= unicon_acc), None)
  print(f"SGD-CLIP  test matching acc = {final['eval']:.4f}  train time = {final['time']:.3f}s "
        f"({args.sgd_epochs} epochs)")
  if reach:
    print(f"SGD-CLIP  first reaches UniCon accuracy at epoch {reach['epoch']} ({reach['time']:.3f}s)")

  results = {
    "args": vars(args),
    "unicon": {"acc": unicon_acc, "time": unicon_time},
    "sgd_clip": {"acc": final["eval"], "time": final["time"],
                 "epochs_to_match_unicon": reach["epoch"] if reach else None},
  }
  save_json(results, os.path.join(args.out, "results.json"))

  if not args.no_plots:
    with torch.no_grad():
      plot_alignment_tsne(unicon.encode_x(test.x), unicon.encode_y(test.y), unicon_sim,
                          test.labels, "UniCon", os.path.join(args.out, "tsne_unicon.png"))
      plot_alignment_tsne(clip.encode_x(test.x), clip.encode_y(test.y),
                          clip.similarity(test.x, test.y), test.labels, "SGD-CLIP",
                          os.path.join(args.out, "tsne_sgd_clip.png"))
    plot_curve([h["epoch"] for h in history], [h["eval"] for h in history], "epoch",
               "test matching accuracy", "SGD-CLIP (Figure C1a)",
               os.path.join(args.out, "sgd_clip_curve.png"))


if __name__ == "__main__":
  main()
