"""Table 2: retrieval on the MSCOCO 5K test split and zero-shot transfer to Flickr30K.

python experiments/mscoco/run.py --backbone clip_vit_b32
python experiments/mscoco/run.py --backbone resnet50_minilm --methods unicon
python experiments/mscoco/run.py --backbone resnet50_minilm --sgd-epochs 1000   # full baseline

All models are trained on MSCOCO; the Flickr30K numbers use its test split without
any adaptation. UniCon trains in seconds; the SGD-CLIP baseline defaults to a short
schedule (the paper trained it for 1000 epochs, see Appendix C.2).
"""

import argparse
import os

import torch

from unicon.data.features import load_split, require_cache
from unicon.eval.retrieval import evaluate, format_row, header, run_sgd_clip, run_unicon
from unicon.utils import get_device, save_json, set_seed

PRESETS = {
  # Many-to-many UniCon on the first 200 training images x 5 captions (Section 5, Data
  # efficiency). r = d keeps the full space; the large ridge keeps the closed-form
  # rotation close to the identity because CLIP features already share one space.
  "clip_vit_b32": dict(
    train_cache="clip_vit_b32_5cap", eval_cache="clip_vit_b32_5cap",
    flickr_cache="clip_vit_b32",
    unicon=dict(r=512, tau=0.07, nu=1.5, ridge=10.0, scale_inputs=True, many_to_many=True,
                factorization="orthonormal", batch_size=1000, n_batches=1, iters=2),
    sgd=dict(train_cache="clip_vit_b32", r=196, tau=1.0, optimizer="adamw", lr=1e-3,
             weight_decay=1e-4, batch_size=1024, epochs=100),
  ),
  "resnet50_minilm": dict(
    train_cache="resnet50_minilm", eval_cache="resnet50_minilm",
    flickr_cache="resnet50_minilm",
    unicon=dict(r=256, tau=0.07, nu=1.5, rho=0.1, ridge=1e-4, scale_inputs=True,
                factorization="orthonormal", batch_size=27593, n_batches=1, iters=2),
    sgd=dict(train_cache="resnet50_minilm", r=196, tau=1.0, optimizer="adam", lr=1e-4,
             batch_size=27593, epochs=100),
  ),
}


def load(root, cache, split, device):
  path = os.path.join(root, cache)
  require_cache(path, "See experiments/mscoco/extract_features.py / experiments/flickr30k/extract_features.py.")
  return load_split(path, split, normalize=True).to(device)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", default="clip_vit_b32", choices=sorted(PRESETS))
  p.add_argument("--methods", nargs="+", default=["unicon", "sgd_clip", "none"],
                 choices=["unicon", "sgd_clip", "none"])
  p.add_argument("--data-root", default="data")
  p.add_argument("--n-batches", type=int, default=None,
                 help="UniCon: number of training batches (default: preset)")
  p.add_argument("--sgd-epochs", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="auto")
  p.add_argument("--out", default="outputs/mscoco")
  p.add_argument("--verbose", action="store_true")
  args = p.parse_args()
  set_seed(args.seed)
  device = get_device(args.device)
  preset = PRESETS[args.backbone]
  coco_root = os.path.join(args.data_root, "mscoco")

  val = load(coco_root, preset["eval_cache"], "val", device)
  test = load(coco_root, preset["eval_cache"], "test", device)
  flickr_dir = os.path.join(args.data_root, "flickr30k", preset["flickr_cache"])
  flickr = (load_split(flickr_dir, "test", normalize=True).to(device)
            if os.path.exists(os.path.join(flickr_dir, "test_visual_features.pt")) else None)

  results = {}
  if "unicon" in args.methods:
    cfg = dict(preset["unicon"], seed=args.seed)
    if args.n_batches is not None:
      cfg["n_batches"] = args.n_batches
    train = load(coco_root, preset["train_cache"], "train", device)
    model = run_unicon(train, val, cfg, verbose=args.verbose)
    results["UniCon"] = (model, model.train_time)
    del train
  if "sgd_clip" in args.methods:
    cfg = dict(preset["sgd"], seed=args.seed)
    if args.sgd_epochs is not None:
      cfg["epochs"] = args.sgd_epochs
    train = load(coco_root, cfg.pop("train_cache"), "train", device)
    model = run_sgd_clip(train, val, cfg, verbose=args.verbose)
    results["SGD-CLIP"] = (model, model.train_time)
    del train
  if "none" in args.methods and test.image.shape[1] == test.text.shape[1]:
    results["no-align"] = (None, None)

  summary = {}
  print(f"\n== MSCOCO 5K test / {args.backbone} ==")
  print(header())
  for name, (model, t) in results.items():
    m = evaluate(model, test)
    summary[name] = {"mscoco": m, "train_time": t}
    print(format_row(name, m, t))
  if flickr is not None:
    print(f"\n== zero-shot Flickr30K test ({len(flickr)} pairs) / {args.backbone} ==")
    print(header(ks=(5, 10)))
    for name, (model, t) in results.items():
      m = evaluate(model, flickr)
      summary[name]["flickr30k_zero_shot"] = m
      print(format_row(name, m, None, ks=(5, 10)))
  save_json(summary, os.path.join(args.out, f"{args.backbone}.json"))


if __name__ == "__main__":
  torch.backends.cuda.matmul.allow_tf32 = False
  main()
