"""Extract frozen-backbone features for Flickr30K (first caption per image, 80/10/10 split).

python experiments/flickr30k/extract_features.py --backbone clip_vit_b32
python experiments/flickr30k/extract_features.py --backbone resnet18_minilm
python experiments/flickr30k/extract_features.py --backbone resnet50_minilm

To evaluate MSCOCO-trained RN-50 models zero-shot on Flickr30K, extract with the same
text model used for MSCOCO, e.g. ``--text-model all-MiniLM-L6-v2 --out data/flickr30k/resnet50_allminilm``.
"""

import argparse

from unicon.data.datasets import flickr30k_splits
from unicon.data.extract import BACKBONES, FeatureExtractor, extract_pairs, save_split
from unicon.utils import get_device

# Sentence-BERT model behind the cached Flickr30K features used in the paper.
DEFAULT_TEXT_MODEL = "paraphrase-MiniLM-L6-v2"


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--backbone", required=True, choices=BACKBONES)
  p.add_argument("--text-model", default=DEFAULT_TEXT_MODEL,
                 help="Sentence-BERT model for the ResNet backbones")
  p.add_argument("--out", default=None, help="defaults to data/flickr30k/<backbone>")
  p.add_argument("--batch-size", type=int, default=256)
  p.add_argument("--device", default="auto")
  args = p.parse_args()

  out = args.out or f"data/flickr30k/{args.backbone}"
  extractor = FeatureExtractor(args.backbone, args.text_model, get_device(args.device))
  for split, dataset in flickr30k_splits(captions_per_image=1).items():
    save_split(out, split, *extract_pairs(extractor, dataset, args.batch_size, desc=split))


if __name__ == "__main__":
  main()
