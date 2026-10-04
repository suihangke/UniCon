"""Two-view frozen ResNet-18 features for CIFAR-10 (Section 4.2).

python experiments/cifar10/extract_features.py --backbone resnet18

Writes, for split in {train, test}, to data/cifar10/<backbone>/:
  {split}_view1_features.pt  FloatTensor (N, d)
  {split}_view2_features.pt  FloatTensor (N, d)
  {split}_labels.pt          LongTensor (N,)
"""

import argparse


def extract(backbone: str, split: str, out_dir: str, batch_size: int, device: str):
  """TODO(coauthor): load CIFAR-10 ``split``, draw two augmentations per image, encode both
  with the frozen ``backbone`` and save the three tensors listed in the module docstring."""
  raise NotImplementedError("CIFAR-10 feature extraction is not implemented yet.")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", default="resnet18")
  p.add_argument("--out", default=None, help="defaults to data/cifar10/<backbone>")
  p.add_argument("--batch-size", type=int, default=256)
  p.add_argument("--device", default="auto")
  args = p.parse_args()
  out = args.out or f"data/cifar10/{args.backbone}"
  for split in ("train", "test"):
    extract(args.backbone, split, out, args.batch_size, args.device)


if __name__ == "__main__":
  main()
