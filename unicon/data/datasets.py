"""Image-caption datasets from the Hugging Face hub, returning (PIL image, [captions])."""

import torch
from torch.utils.data import Dataset

FLICKR30K_HF = "lmms-lab/flickr30k"
MSCOCO_HF = "shunk031/MSCOCO"
MSCOCO_5K_TEST_HF = "nlphuji/mscoco_2014_5k_test_image_text_retrieval"


def _as_list(caption):
  return list(caption) if isinstance(caption, (list, tuple)) else [caption]


class CaptionDataset(Dataset):
  def __init__(self, hf_split, caption_fn, indices=None, captions_per_image=1):
    self.data = hf_split
    self.caption_fn = caption_fn
    self.indices = list(range(len(hf_split))) if indices is None else list(indices)
    self.captions_per_image = captions_per_image

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, i):
    item = self.data[self.indices[i]]
    captions = self.caption_fn(item)
    if self.captions_per_image is not None:
      captions = captions[:self.captions_per_image]
    return item["image"], captions


def flickr30k_splits(captions_per_image=1, seed=42):
  """Flickr30K (31,783 images) shuffled into 80% / 10% / 10% train / val / test.

  Gives 25,426 / 3,178 / 3,179 images (Appendix C.2). The first caption of every image is used.
  """
  from datasets import load_dataset
  data = load_dataset(FLICKR30K_HF)["test"]
  n = len(data)
  n_train, n_val = int(0.8 * n), int(0.1 * n)
  perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
  idx = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val],
         "test": perm[n_train + n_val:]}
  fn = lambda item: _as_list(item["caption"])
  return {k: CaptionDataset(data, fn, v, captions_per_image) for k, v in idx.items()}


def mscoco_splits(train_captions_per_image=1):
  """MSCOCO 2014: train (82,783 images), val (40,504 images, one caption) and the 5K test split."""
  from datasets import load_dataset
  coco = load_dataset(MSCOCO_HF, year=2014, coco_task="captions", trust_remote_code=True)
  test = load_dataset(MSCOCO_5K_TEST_HF)["test"]
  coco_fn = lambda item: _as_list(item["annotations"]["caption"])
  test_fn = lambda item: _as_list(item["caption"])
  return {
    "train": CaptionDataset(coco["train"], coco_fn, captions_per_image=train_captions_per_image),
    "val": CaptionDataset(coco["validation"], coco_fn, captions_per_image=1),
    "test": CaptionDataset(test, test_fn, captions_per_image=1),
  }
