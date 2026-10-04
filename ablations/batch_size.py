"""Appendix C.3: robustness of UniCon to the batch size used to build S(gamma).

Batch solutions are aggregated with validation-R@1 weights (Algorithm 1).

python ablations/batch_size.py --backbone clip_vit_b32
"""

import argparse
import os

from unicon.data.features import load_split, require_cache
from unicon.eval.retrieval import evaluate, run_unicon
from unicon.utils import get_device

NORMALIZE = {"clip_vit_b32": True, "resnet18_minilm": False, "resnet50_minilm": False}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", default="clip_vit_b32", choices=sorted(NORMALIZE))
  p.add_argument("--batch-sizes", nargs="+", type=int, default=[100, 500, 1000, 5000, 10000, 20000])
  p.add_argument("--data-root", default="data/flickr30k")
  p.add_argument("--device", default="auto")
  args = p.parse_args()

  cache = os.path.join(args.data_root, args.backbone)
  require_cache(cache, "Run experiments/flickr30k/extract_features.py first.")
  device = get_device(args.device)
  train, val, test = (load_split(cache, s, NORMALIZE[args.backbone]).to(device)
                      for s in ("train", "val", "test"))
  print(f"Flickr30K / {args.backbone}: UniCon test recall vs. batch size")
  print(f"{'batch':>7s} {'#batches':>9s} {'time':>8s} {'avg R@1':>8s} {'avg R@10':>9s}")
  for bs in args.batch_sizes:
    cfg = dict(r=128, tau=1.0, nu=2.0, factorization="orthonormal", batch_size=bs, iters=2)
    model = run_unicon(train, val, cfg, verbose=False)
    m = evaluate(model, test)
    print(f"{bs:7d} {len(train) // bs:9d} {model.train_time:7.2f}s {m['avg_R@1']:8.3f} "
          f"{m['avg_R@10']:9.3f}")


if __name__ == "__main__":
  main()
