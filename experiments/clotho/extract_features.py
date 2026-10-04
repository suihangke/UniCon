"""Frozen Wav2CLIP (audio) and CLIP (text) features for Clotho (Appendix C.5).

python experiments/clotho/extract_features.py --backbone wav2clip_clip

Writes, for split in {train, val, test}, to data/clotho/<backbone>/:
  {split}_audio_features.pt  FloatTensor (N, d_audio)
  {split}_text_features.pt   FloatTensor (N, d_text)
  {split}_captions.pt        list[str]
"""

import argparse


def extract(backbone: str, split: str, out_dir: str, batch_size: int, device: str):
  """TODO(coauthor): load the Clotho ``split``, encode audio clips and captions with the
  frozen encoders and save the tensors listed in the module docstring (row i = positive pair)."""
  raise NotImplementedError("Clotho feature extraction is not implemented yet.")


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", default="wav2clip_clip")
  p.add_argument("--out", default=None, help="defaults to data/clotho/<backbone>")
  p.add_argument("--batch-size", type=int, default=64)
  p.add_argument("--device", default="auto")
  args = p.parse_args()
  out = args.out or f"data/clotho/{args.backbone}"
  for split in ("train", "val", "test"):
    extract(args.backbone, split, out, args.batch_size, args.device)


if __name__ == "__main__":
  main()
