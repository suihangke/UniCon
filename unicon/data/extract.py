"""Frozen-backbone feature extraction for Flickr30K and MSCOCO.

Backbones (Section 4.3):
  clip_vit_b32     openai/clip-vit-base-patch32 image + text towers (512 / 512)
  resnet18_minilm  torchvision ResNet-18 (ImageNet V1) + Sentence-BERT  (512 / 384)
  resnet50_minilm  torchvision ResNet-50 (ImageNet V2) + Sentence-BERT  (2048 / 384)

Image features are the pooled outputs (ResNet: global average pool before the
classifier; CLIP: ``get_image_features``). Features are stored unnormalized; the
experiment scripts decide whether to L2-normalize.
"""

import os
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

CLIP_MODEL = "openai/clip-vit-base-patch32"
BACKBONES = ("clip_vit_b32", "resnet18_minilm", "resnet50_minilm")


class FeatureExtractor:
  def __init__(self, backbone: str, text_model: str, device: torch.device):
    if backbone not in BACKBONES:
      raise ValueError(f"Unknown backbone '{backbone}'. Choose from {BACKBONES}.")
    self.backbone = backbone
    self.device = device
    if backbone == "clip_vit_b32":
      from transformers import CLIPModel, CLIPProcessor
      self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
      self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    else:
      import torch.nn as nn
      from sentence_transformers import SentenceTransformer
      from torchvision import models, transforms
      if backbone == "resnet18_minilm":
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
      else:
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
      self.model = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
      self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      self.sbert = SentenceTransformer(text_model, device=str(device))

  @torch.no_grad()
  def images(self, images: Sequence) -> torch.Tensor:
    images = [im.convert("RGB") for im in images]
    if self.backbone == "clip_vit_b32":
      inputs = self.processor(images=images, return_tensors="pt").to(self.device)
      return self.model.get_image_features(**inputs).float().cpu()
    batch = torch.stack([self.transform(im) for im in images]).to(self.device)
    return self.model(batch).flatten(1).float().cpu()

  @torch.no_grad()
  def texts(self, texts: List[str]) -> torch.Tensor:
    if self.backbone == "clip_vit_b32":
      inputs = self.processor(text=list(texts), return_tensors="pt", padding=True,
                              truncation=True).to(self.device)
      return self.model.get_text_features(**inputs).float().cpu()
    return self.sbert.encode(list(texts), convert_to_tensor=True,
                             show_progress_bar=False).float().cpu()


def extract_pairs(extractor: FeatureExtractor, dataset, batch_size: int = 256,
                  num_workers: int = 4, desc: str = ""):
  """``dataset[i]`` returns (PIL image, list of captions). Every caption becomes one row.

  Returns (image_features, text_features, captions, image_ids) where image_ids[k] is the
  dataset index of row k; one-caption datasets therefore give image_ids = arange(N).
  """
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      collate_fn=lambda batch: batch)
  img_feats, txt_feats, captions, image_ids = [], [], [], []
  offset = 0
  for batch in tqdm(loader, desc=desc):
    images = [item[0] for item in batch]
    caps = [item[1] for item in batch]
    img = extractor.images(images)
    flat = [c for cs in caps for c in cs]
    repeats = torch.tensor([len(cs) for cs in caps])
    img_feats.append(img.repeat_interleave(repeats, dim=0))
    txt_feats.append(extractor.texts(flat))
    captions.extend(flat)
    image_ids.append(torch.arange(offset, offset + len(batch)).repeat_interleave(repeats))
    offset += len(batch)
  return torch.cat(img_feats), torch.cat(txt_feats), captions, torch.cat(image_ids)


def save_split(out_dir: str, split: str, image, text, captions, image_ids=None):
  os.makedirs(out_dir, exist_ok=True)
  torch.save(image, os.path.join(out_dir, f"{split}_visual_features.pt"))
  torch.save(text, os.path.join(out_dir, f"{split}_text_features.pt"))
  torch.save(captions, os.path.join(out_dir, f"{split}_captions.pt"))
  if image_ids is not None and len(image_ids) != len(torch.unique(image_ids)):
    torch.save(image_ids.tolist(), os.path.join(out_dir, f"{split}_image_to_caption_mapping.pt"))
  print(f"saved {split}: image {tuple(image.shape)}, text {tuple(text.shape)} -> {out_dir}")
