"""Section 4.2: CIFAR-10 image-image alignment, kernel UniCon + linear probe.

python experiments/cifar10/run.py --backbone resnet18

Work in progress: see experiments/cifar10/README.md. The hyperparameters marked
TODO(coauthor) still have to be set to the values behind the paper's numbers.
"""

import argparse
import os
import time

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from unicon.solvers.kernel import KernelUniCon
from unicon.utils import get_device, save_json, set_seed

# TODO(coauthor): confirm against the paper run (61.82% in 23.38 s).
UNICON = dict(r=128, kernel="angular", batch_size=300, n_iters=2, init="zeros")
PROBE = dict(max_iter=1000)


def load(cache, split, device):
  def get(name):
    return torch.load(os.path.join(cache, f"{split}_{name}.pt"), map_location="cpu")
  return get("view1_features").float().to(device), get("view2_features").float().to(device), get("labels")


def run_unicon(train, test, seed):
  v1, v2, y_train = train
  model = KernelUniCon(r=UNICON["r"], kernel=UNICON["kernel"], n_iters=UNICON["n_iters"],
                       init=UNICON["init"], seed=seed)
  start = time.time()
  model.fit(v1, v2, batch_size=UNICON["batch_size"])
  z_train, _ = model.embed(x=v1)
  z_test, _ = model.embed(x=test[0])
  probe = LogisticRegression(**PROBE).fit(z_train.cpu().numpy(), y_train.numpy())
  train_time = time.time() - start
  pred = probe.predict(z_test.cpu().numpy())
  return {"accuracy": float((pred == test[2].numpy()).mean()), "train_time": train_time,
          "confusion_matrix": confusion_matrix(test[2].numpy(), pred).tolist()}


def run_sgd_baseline(train, test, seed):
  """TODO(coauthor): two-layer MLP head, bidirectional InfoNCE, SGD for 300 epochs, then the
  same linear probe as ``run_unicon``. Return the same dictionary."""
  raise NotImplementedError("The CIFAR-10 SGD baseline is not implemented yet.")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", default="resnet18")
  p.add_argument("--methods", nargs="+", default=["unicon"], choices=["unicon", "sgd"])
  p.add_argument("--data-root", default="data/cifar10")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="auto")
  p.add_argument("--out", default="outputs/cifar10")
  args = p.parse_args()
  set_seed(args.seed)
  device = get_device(args.device)
  cache = os.path.join(args.data_root, args.backbone)
  if not os.path.exists(os.path.join(cache, "train_view1_features.pt")):
    raise FileNotFoundError(f"No feature cache in '{cache}'. See experiments/cifar10/README.md.")
  train, test = load(cache, "train", device), load(cache, "test", device)

  runners = {"unicon": run_unicon, "sgd": run_sgd_baseline}
  results = {}
  for method in args.methods:
    results[method] = runners[method](train, test, args.seed)
    print(f"{method:8s} accuracy={results[method]['accuracy']:.4f} "
          f"time={results[method]['train_time']:.2f}s")
  save_json(results, os.path.join(args.out, f"{args.backbone}.json"))


if __name__ == "__main__":
  main()
