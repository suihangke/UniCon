"""Appendix C.7 / Table 8: UniCon under the sigmoid (SigLIP) loss on MSCOCO, CLIP ViT-B/32.

Same setting as the CLIP row of Table 2 with psi = softplus(x / tau) and phi = identity.

python ablations/siglip.py
"""

import argparse
import os

from unicon.data.features import load_split, require_cache
from unicon.eval.retrieval import evaluate, run_unicon
from unicon.losses import sigmoid_pairwise_loss
from unicon.utils import get_device, set_seed

UNICON = dict(r=512, ridge=10.0, scale_inputs=True, many_to_many=True,
              factorization="orthonormal", batch_size=1000, n_batches=1, iters=2)
PAPER = [.3340, .2862, .5816, .5334, .6852, .6394]
KEYS = ["i2t_R@1", "t2i_R@1", "i2t_R@5", "t2i_R@5", "i2t_R@10", "t2i_R@10"]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--cache", default="data/mscoco/clip_vit_b32_5cap")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="auto")
  args = p.parse_args()
  set_seed(args.seed)
  require_cache(args.cache, "Run experiments/mscoco/extract_features.py --backbone clip_vit_b32 --train-captions 5.")
  device = get_device(args.device)
  train, val, test = (load_split(args.cache, s, normalize=True).to(device)
                      for s in ("train", "val", "test"))

  cfg = dict(UNICON, loss=sigmoid_pairwise_loss(tau=0.07, nu=1.5), seed=args.seed)
  rows = {
    "UniCon (SigLIP)": [evaluate(run_unicon(train, val, cfg, verbose=False), test)[k] for k in KEYS],
    "  paper": PAPER,
    "no alignment": [evaluate(None, test)[k] for k in KEYS],
  }
  print(f"{'':<18s}" + "".join(f"{k:>10s}" for k in KEYS))
  for name, values in rows.items():
    print(f"{name:<18s}" + "".join(f"{v:10.4f}" for v in values))


if __name__ == "__main__":
  main()
