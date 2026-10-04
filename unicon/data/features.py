"""Loading of pre-extracted (frozen-backbone) feature caches.

A cache directory contains, for each split in {train, val, test}:
  {split}_visual_features.pt  FloatTensor (N, d_img)
  {split}_text_features.pt    FloatTensor (N, d_txt)
  {split}_captions.pt         list[str] of length N (optional)
and, for many-to-many training data, {split}_image_to_caption_mapping.pt:
  list[int] of length N giving the image id of every (image, caption) row.
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class FeatureSplit:
  image: torch.Tensor
  text: torch.Tensor
  image_ids: Optional[torch.Tensor] = None

  def __len__(self):
    return self.image.shape[0]

  def to(self, device):
    ids = None if self.image_ids is None else self.image_ids.to(device)
    return FeatureSplit(self.image.to(device), self.text.to(device), ids)


def load_split(cache_dir: str, split: str, normalize: bool = False) -> FeatureSplit:
  def load(name):
    return torch.load(os.path.join(cache_dir, f"{split}_{name}.pt"), map_location="cpu")

  image = load("visual_features").float()
  text = load("text_features").float()
  mapping = os.path.join(cache_dir, f"{split}_image_to_caption_mapping.pt")
  image_ids = torch.as_tensor(torch.load(mapping)) if os.path.exists(mapping) else None
  if normalize:
    image, text = F.normalize(image, dim=1), F.normalize(text, dim=1)
  return FeatureSplit(image, text, image_ids)


def require_cache(cache_dir: str, hint: str):
  if not os.path.exists(os.path.join(cache_dir, "train_visual_features.pt")):
    raise FileNotFoundError(f"No feature cache in '{cache_dir}'. {hint}")
