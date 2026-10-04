"""Section 4.1 / Figure 3: nonlinear latent-factor model, kernel UniCon vs. SGD-CLIP (MLP).

python experiments/synthetic/nonlinear.py --out outputs/synthetic_nonlinear
python experiments/synthetic/nonlinear.py --seeds 0 1 2 3 4 5 6 7 8 9 --no-sgd
"""

import argparse
import os

import numpy as np
import torch

from unicon.baselines import DualEncoder, train_sgd_clip
from unicon.data.synthetic import nonlinear_latent_factor_data
from unicon.solvers.kernel import KernelUniCon
from unicon.losses import clip_loss
from unicon.eval.metrics import matching_accuracy
from unicon.utils import save_json, set_seed
from unicon.viz import plot_alignment_tsne, plot_curve, plot_S_gamma_heatmaps


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--d1", type=int, default=40)
  p.add_argument("--d2", type=int, default=39)
  p.add_argument("--r", type=int, default=10)
  p.add_argument("--n-train", type=int, default=1000)
  p.add_argument("--n-val", type=int, default=100)
  p.add_argument("--n-test", type=int, default=100)
  p.add_argument("--snr", type=float, default=0.3)
  p.add_argument("--batch-size", type=int, default=30)
  p.add_argument("--tau", type=float, default=1.0)
  p.add_argument("--kernel", default="angular")
  p.add_argument("--unicon-iters", type=int, default=10, help="max fixed-point steps per batch")
  p.add_argument("--init", default="zeros", choices=["zeros", "random"])
  p.add_argument("--no-kernel-sqrt", action="store_true",
                 help="take the SVD of S(gamma) instead of K_x^1/2 S K_y^1/2 (ablation)")
  p.add_argument("--sgd-epochs", type=int, default=500)
  p.add_argument("--sgd-lr", type=float, default=2e-3)
  p.add_argument("--no-sgd", action="store_true")
  p.add_argument("--seeds", nargs="+", type=int, default=[0])
  p.add_argument("--out", default="outputs/synthetic_nonlinear")
  p.add_argument("--no-plots", action="store_true")
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)

  runs = []
  for seed in args.seeds:
    set_seed(seed)
    train, val, test = nonlinear_latent_factor_data(
      d1=args.d1, d2=args.d2, r=args.r, n_train=args.n_train, n_val=args.n_val,
      n_test=args.n_test, snr=args.snr, seed=seed)

    unicon = KernelUniCon(r=args.r, kernel=args.kernel, loss=clip_loss(args.tau),
                          n_iters=args.unicon_iters, init=args.init,
                          use_kernel_sqrt=not args.no_kernel_sqrt, seed=seed)
    unicon.fit(train.x, train.y, batch_size=args.batch_size, x_val=val.x, y_val=val.y,
               record_S=True)
    unicon_sim = unicon.similarity(test.x, test.y)
    run = {"seed": seed, "unicon_acc": matching_accuracy(unicon_sim),
           "unicon_time": unicon.train_time}
    line = f"seed {seed}: UniCon acc={run['unicon_acc']:.3f} ({run['unicon_time']:.2f}s)"

    if not args.no_sgd:
      torch.manual_seed(seed)
      clip = DualEncoder(args.d1, args.d2, args.r, activation="tanh")
      history = train_sgd_clip(
        clip, train.x, train.y, epochs=args.sgd_epochs, batch_size=args.batch_size,
        lr=args.sgd_lr, tau=args.tau,
        eval_fn=lambda m: matching_accuracy(m.similarity(test.x, test.y)), seed=seed,
        verbose=False)
      best = max(history, key=lambda h: h["eval"])
      run.update(sgd_final_acc=history[-1]["eval"], sgd_best_acc=best["eval"],
                 sgd_best_epoch=best["epoch"], sgd_time=history[-1]["time"])
      line += (f" | SGD-CLIP final acc={run['sgd_final_acc']:.3f}, best={run['sgd_best_acc']:.3f} "
               f"@ epoch {run['sgd_best_epoch']} ({run['sgd_time']:.2f}s)")
    print(line)
    runs.append(run)

    if not args.no_plots and seed == args.seeds[0]:
      trace = unicon.S_history[0]
      picks = sorted({0, 1, len(trace) - 1} & set(range(len(trace))))
      plot_S_gamma_heatmaps([trace[i] for i in picks],
                            ["initial S(γ)" if i == 0 else f"after step {i}" for i in picks],
                            os.path.join(args.out, "S_gamma_evolution.png"))
      z1, z2 = unicon.embed(test.x, test.y)
      plot_alignment_tsne(z1, z2, unicon_sim, None, "UniCon (kernel)",
                          os.path.join(args.out, "tsne_unicon.png"))
      if not args.no_sgd:
        plot_curve([h["epoch"] for h in history], [h["eval"] for h in history], "epoch",
                   "test matching accuracy", "SGD-CLIP (Figure C1b)",
                   os.path.join(args.out, "sgd_clip_curve.png"))

  summary = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if k != "seed"}
  print("mean over seeds: " + "  ".join(f"{k}={v:.3f}" for k, v in summary.items()))
  save_json({"args": vars(args), "runs": runs, "mean": summary},
            os.path.join(args.out, "results.json"))


if __name__ == "__main__":
  main()
