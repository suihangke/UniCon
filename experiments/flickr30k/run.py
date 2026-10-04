"""Table 1: image-text retrieval on Flickr30K with frozen backbones.

python experiments/flickr30k/run.py --backbone clip_vit_b32
python experiments/flickr30k/run.py --backbone resnet50_minilm --methods unicon
python experiments/flickr30k/run.py --backbone all
"""

import argparse
import os

import torch

from unicon.data.extract import BACKBONES
from unicon.data.features import load_split, require_cache
from unicon.eval.retrieval import evaluate, format_row, header, run_sgd_clip, run_unicon
from unicon.utils import get_device, save_json, set_seed

# Hyperparameters used for Table 1.
UNICON = dict(r=128, tau=1.0, nu=2.0, factorization="orthonormal", batch_size=23420,
              n_batches=1, iters=2)
SGD_CLIP = dict(r=128, tau=1.0, optimizer="adam", lr=1e-4, batch_size=128, epochs=50)
# The cached CLIP features are L2-normalized, the ResNet / Sentence-BERT ones are raw.
NORMALIZE = {"clip_vit_b32": True, "resnet18_minilm": False, "resnet50_minilm": False}


def run_backbone(backbone, args, device):
  cache = os.path.join(args.data_root, backbone)
  require_cache(cache, f"Run: python experiments/flickr30k/extract_features.py --backbone {backbone}")
  train, val, test = (load_split(cache, s, NORMALIZE[backbone]).to(device)
                      for s in ("train", "val", "test"))
  print(f"\n== Flickr30K / {backbone}: train {len(train)}, val {len(val)}, test {len(test)} ==")

  results = {}
  if "unicon" in args.methods:
    model = run_unicon(train, val, dict(UNICON, seed=args.seed), verbose=False)
    results["UniCon"] = (evaluate(model, test), model.train_time)
  if "sgd_clip" in args.methods:
    cfg = dict(SGD_CLIP, seed=args.seed, epochs=args.sgd_epochs)
    model = run_sgd_clip(train, val, cfg, verbose=args.verbose)
    results["SGD-CLIP"] = (evaluate(model, test), model.train_time)
  if "none" in args.methods and train.image.shape[1] == train.text.shape[1]:
    results["no-align"] = (evaluate(None, test), None)

  print(header())
  for name, (metrics, t) in results.items():
    print(format_row(name, metrics, t))
  return {name: {"metrics": m, "train_time": t} for name, (m, t) in results.items()}


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", default="all", choices=BACKBONES + ("all",))
  p.add_argument("--methods", nargs="+", default=["unicon", "sgd_clip", "none"],
                 choices=["unicon", "sgd_clip", "none"])
  p.add_argument("--data-root", default="data/flickr30k")
  p.add_argument("--sgd-epochs", type=int, default=SGD_CLIP["epochs"])
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="auto")
  p.add_argument("--out", default="outputs/flickr30k")
  p.add_argument("--verbose", action="store_true")
  args = p.parse_args()
  set_seed(args.seed)
  device = get_device(args.device)
  backbones = BACKBONES if args.backbone == "all" else (args.backbone,)
  results = {b: run_backbone(b, args, device) for b in backbones}
  save_json(results, os.path.join(args.out, "results.json"))


if __name__ == "__main__":
  torch.backends.cuda.matmul.allow_tf32 = False
  main()
