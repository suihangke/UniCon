"""Appendix C.4 / Table 5: numerical stabilization of the spectral step on MSCOCO (CLIP ViT-B/32).

Plain truncated SVD vs. adding Tikhonov regularization, randomized SVD and
unit-sphere normalization. Trained on all MSCOCO training images (one caption each),
three batches aggregated with validation weights.

python ablations/stabilized_svd.py
"""

import argparse

from unicon.data.features import load_split, require_cache
from unicon.eval.retrieval import evaluate, run_unicon
from unicon.utils import get_device

BASE = dict(r=196, tau=0.07, nu=1.5, factorization="orthonormal", batch_size=27593, iters=2)
VARIANTS = {
  "standard SVD": dict(normalize=False),
  "+ Tikhonov": dict(normalize=False, ridge=10.0),
  "+ Tikhonov + unit sphere": dict(normalize=True, ridge=10.0, scale_inputs=True),
  "+ Tikhonov + unit sphere + randomized": dict(normalize=True, ridge=10.0, scale_inputs=True,
                                                svd_method="randomized"),
}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--cache", default="data/mscoco/clip_vit_b32")
  p.add_argument("--device", default="auto")
  args = p.parse_args()
  require_cache(args.cache, "Run experiments/mscoco/extract_features.py --backbone clip_vit_b32.")
  device = get_device(args.device)

  print(f"{'variant':<40s} {'time':>7s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s}  (avg of I2T and T2I)")
  for normalize in (False, True):
    train, val, test = (load_split(args.cache, s, normalize).to(device)
                        for s in ("train", "val", "test"))
    for name, variant in VARIANTS.items():
      variant = dict(variant)
      if variant.pop("normalize") != normalize:
        continue
      model = run_unicon(train, val, dict(BASE, **variant), verbose=False)
      m = evaluate(model, test)
      print(f"{name:<40s} {model.train_time:6.2f}s {m['avg_R@1']:7.4f} {m['avg_R@5']:7.4f} "
            f"{m['avg_R@10']:7.4f}")
    if normalize:
      m = evaluate(None, test)
      print(f"{'no alignment (raw CLIP features)':<40s} {'-':>7s} {m['avg_R@1']:7.4f} "
            f"{m['avg_R@5']:7.4f} {m['avg_R@10']:7.4f}")


if __name__ == "__main__":
  main()
