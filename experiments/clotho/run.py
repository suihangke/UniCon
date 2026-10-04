"""Appendix C.5 / Table 6: audio-text retrieval on Clotho with frozen encoders.

python experiments/clotho/run.py --backbone wav2clip_clip

Work in progress: see experiments/clotho/README.md. The hyperparameters marked
TODO(coauthor) still have to be set to the values behind the paper's numbers.
"""

import argparse
import os

import torch

from unicon.data.features import FeatureSplit
from unicon.eval.retrieval import evaluate, format_row, header, run_sgd_clip, run_unicon
from unicon.utils import get_device, save_json, set_seed

# TODO(coauthor): confirm against the paper run (Table 6).
UNICON = dict(r=256, tau=0.07, nu=1.5, rho=0.1, ridge=1e-4, scale_inputs=True,
              factorization="orthonormal", batch_size=4096, iters=2)
SGD_CLIP = dict(r=256, tau=1.0, optimizer="adam", lr=1e-4, batch_size=128, epochs=100)


def load(cache, split, device):
  def get(name):
    feats = torch.load(os.path.join(cache, f"{split}_{name}_features.pt"), map_location="cpu")
    return torch.nn.functional.normalize(feats.float(), dim=1).to(device)
  # Audio plays the role of the first modality ("image") in the retrieval helpers.
  return FeatureSplit(image=get("audio"), text=get("text"))


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", default="wav2clip_clip")
  p.add_argument("--methods", nargs="+", default=["unicon", "sgd_clip", "none"],
                 choices=["unicon", "sgd_clip", "none"])
  p.add_argument("--data-root", default="data/clotho")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="auto")
  p.add_argument("--out", default="outputs/clotho")
  args = p.parse_args()
  set_seed(args.seed)
  device = get_device(args.device)
  cache = os.path.join(args.data_root, args.backbone)
  if not os.path.exists(os.path.join(cache, "train_audio_features.pt")):
    raise FileNotFoundError(f"No feature cache in '{cache}'. See experiments/clotho/README.md.")
  train, val, test = (load(cache, s, device) for s in ("train", "val", "test"))

  results = {}
  if "unicon" in args.methods:
    model = run_unicon(train, val, dict(UNICON, seed=args.seed), verbose=False)
    results["UniCon"] = (evaluate(model, test), model.train_time)
  if "sgd_clip" in args.methods:
    model = run_sgd_clip(train, val, dict(SGD_CLIP, seed=args.seed), verbose=False)
    results["SGD-CLIP"] = (evaluate(model, test), model.train_time)
  if "none" in args.methods and train.image.shape[1] == train.text.shape[1]:
    results["no-align"] = (evaluate(None, test), None)

  print("(I = audio, T = text)")
  print(header(ks=(1, 5, 10)))
  for name, (metrics, t) in results.items():
    print(format_row(name, metrics, t, ks=(1, 5, 10)))
  save_json({k: {"metrics": m, "train_time": t} for k, (m, t) in results.items()},
            os.path.join(args.out, f"{args.backbone}.json"))


if __name__ == "__main__":
  main()
