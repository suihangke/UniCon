"""Design choices of the closed-form update.

1. Kernel UniCon on synthetic nonlinear data: the exact Theorem 9 update
   (SVD of K_x^{1/2} S K_y^{1/2}) vs. taking the SVD of S(gamma) directly, and the
   initial similarity used to build the first S(gamma).
2. Linear UniCon on Flickr30K: splitting F1^T F2 = U Sigma V^T as
   F1 = Sigma^{1/2} U^T, F2 = Sigma^{1/2} V^T ("sqrt", Theorem 8) or F1 = U^T, F2 = V^T
   ("orthonormal", used for all real-data results), and the number of iterations.

python ablations/spectral_update.py
"""

import argparse
import os

import numpy as np

from unicon.data.features import load_split
from unicon.data.synthetic import nonlinear_latent_factor_data
from unicon.solvers.kernel import KernelUniCon
from unicon.eval.metrics import matching_accuracy
from unicon.eval.retrieval import evaluate, run_unicon
from unicon.utils import get_device


def kernel_update(seeds):
  data = {s: nonlinear_latent_factor_data(seed=s) for s in seeds}
  print("Kernel UniCon, synthetic nonlinear (test matching accuracy)")
  for use_sqrt in (True, False):
    for init in ("zeros", "random"):
      accs = []
      for seed, (train, val, test) in data.items():
        model = KernelUniCon(r=10, use_kernel_sqrt=use_sqrt, init=init, seed=seed)
        model.fit(train.x, train.y, 30, val.x, val.y)
        accs.append(matching_accuracy(model.similarity(test.x, test.y)))
      name = "K^1/2 S K^1/2" if use_sqrt else "S only"
      print(f"  update={name:<14s} init={init:<7s} mean={np.mean(accs):.3f}  "
            + " ".join(f"{a:.2f}" for a in accs))


def linear_factorization(data_root, device):
  cache = os.path.join(data_root, "flickr30k", "clip_vit_b32")
  if not os.path.exists(os.path.join(cache, "train_visual_features.pt")):
    print(f"\nSkipping Flickr30K part: no cache at {cache}")
    return
  train, val, test = (load_split(cache, s).to(device) for s in ("train", "val", "test"))
  base = dict(r=128, tau=1.0, nu=2.0, batch_size=23420, n_batches=1)
  print("\nLinear UniCon, Flickr30K CLIP ViT-B/32 (test Recall@1 / Recall@10, avg of both directions)")
  for fact in ("orthonormal", "sqrt"):
    for iters in (1, 2, 5):
      model = run_unicon(train, val, dict(base, factorization=fact, iters=iters), verbose=False)
      m = evaluate(model, test)
      print(f"  factorization={fact:<12s} iters={iters}  R@1={m['avg_R@1']:.3f}  "
            f"R@10={m['avg_R@10']:.3f}")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
  p.add_argument("--data-root", default="data")
  p.add_argument("--device", default="auto")
  args = p.parse_args()
  kernel_update(args.seeds)
  linear_factorization(args.data_root, get_device(args.device))


if __name__ == "__main__":
  main()
