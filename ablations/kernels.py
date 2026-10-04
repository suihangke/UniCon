"""Appendix C.6 / Table 7 (synthetic column): kernel choice for kernelized UniCon.

python ablations/kernels.py --seeds 0 1 2 3 4
"""

import argparse

import numpy as np

from unicon.data.synthetic import nonlinear_latent_factor_data
from unicon.solvers.kernel import KernelUniCon
from unicon.eval.metrics import matching_accuracy

KERNELS = ["rbf", "matern", "cosine", "exp_cosine", "arccos", "angular"]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--kernels", nargs="+", default=KERNELS)
  p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
  p.add_argument("--batch-size", type=int, default=30)
  args = p.parse_args()

  data = {s: nonlinear_latent_factor_data(seed=s) for s in args.seeds}
  print(f"{'kernel':<12s} {'mean acc':>9s}  per-seed")
  for kernel in args.kernels:
    accs = []
    for seed, (train, val, test) in data.items():
      model = KernelUniCon(r=10, kernel=kernel, init="zeros", seed=seed)
      model.fit(train.x, train.y, args.batch_size, val.x, val.y)
      accs.append(matching_accuracy(model.similarity(test.x, test.y)))
    print(f"{kernel:<12s} {np.mean(accs):9.3f}  " + " ".join(f"{a:.2f}" for a in accs))


if __name__ == "__main__":
  main()
