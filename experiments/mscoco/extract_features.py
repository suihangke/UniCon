"""Extract frozen-backbone features for MSCOCO 2014 (train / val) and the 5K test split.

python experiments/mscoco/extract_features.py --backbone clip_vit_b32 --train-captions 5
python experiments/mscoco/extract_features.py --backbone resnet50_minilm

``--train-captions 5`` stores every (image, caption) pair of the training images plus a
``train_image_to_caption_mapping.pt`` used to build the many-to-many positive mask.
Validation and test always use the first caption of every image.
"""

import argparse

from unicon.data.datasets import mscoco_splits
from unicon.data.extract import BACKBONES, FeatureExtractor, extract_pairs, save_split
from unicon.utils import get_device

DEFAULT_TEXT_MODEL = "all-MiniLM-L6-v2"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", required=True, choices=BACKBONES)
  p.add_argument("--text-model", default=DEFAULT_TEXT_MODEL,
                 help="Sentence-BERT model for the ResNet backbones")
  p.add_argument("--train-captions", type=int, default=1, choices=[1, 5])
  p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
  p.add_argument("--out", default=None)
  p.add_argument("--batch-size", type=int, default=256)
  p.add_argument("--device", default="auto")
  args = p.parse_args()

  suffix = "_5cap" if args.train_captions == 5 else ""
  out = args.out or f"data/mscoco/{args.backbone}{suffix}"
  extractor = FeatureExtractor(args.backbone, args.text_model, get_device(args.device))
  splits = mscoco_splits(train_captions_per_image=args.train_captions)
  for split in args.splits:
    save_split(out, split, *extract_pairs(extractor, splits[split], args.batch_size, desc=split))


if __name__ == "__main__":
  main()
