import json
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto"):
  if name == "auto":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(name)


def save_json(obj, path):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w") as f:
    json.dump(obj, f, indent=2)
