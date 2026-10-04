"""Appendix C.7: UniCon under different members of the contrastive family (Eq. 2).

Kernel UniCon on the nonlinear latent-factor model (same setup as Section 4.1) with
the CLIP loss, the smoothed triplet loss and a sigmoid (SigLIP-style) pairwise loss.

python ablations/losses.py
"""

import argparse

import numpy as np

from unicon.data.synthetic import nonlinear_latent_factor_data
from unicon.eval.metrics import matching_accuracy
from unicon.losses import clip_loss, sigmoid_pairwise_loss, triplet_loss
from unicon.solvers.kernel import KernelUniCon

LOSSES = {
  "clip": lambda: clip_loss(tau=1.0),
  "triplet": lambda: triplet_loss(margin=0.5, tau=1.0),
  "sigmoid": sigmoid_pairwise_loss,
}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--losses", nargs="+", default=list(LOSSES), choices=list(LOSSES))
  p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
  args = p.parse_args()

  data = {s: nonlinear_latent_factor_data(seed=s) for s in args.seeds}
  print(f"{'loss':<10s} {'mean':>6s} {'max':>6s}  per-seed test matching accuracy")
  for name in args.losses:
    accs = []
    for seed, (train, val, test) in data.items():
      model = KernelUniCon(r=10, loss=LOSSES[name](), init="zeros", seed=seed)
      model.fit(train.x, train.y, 30, val.x, val.y)
      accs.append(matching_accuracy(model.similarity(test.x, test.y)))
    print(f"{name:<10s} {np.mean(accs):6.3f} {max(accs):6.2f}  " + " ".join(f"{a:.2f}" for a in accs))


if __name__ == "__main__":
  main()
