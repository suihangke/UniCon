#!/usr/bin/env bash
# Reproduce the main-text experiments and the appendix ablations.
# Requires the feature caches from experiments/{flickr30k,mscoco}/extract_features.py.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### Section 4.1: synthetic"
python experiments/synthetic/linear.py
python experiments/synthetic/nonlinear.py --seeds 0 1 2 3 4 5 6 7 8 9 --no-sgd

echo "### Table 1: Flickr30K"
python experiments/flickr30k/run.py --backbone all

echo "### Table 2: MSCOCO + zero-shot Flickr30K"
python experiments/mscoco/run.py --backbone clip_vit_b32
python experiments/mscoco/run.py --backbone resnet50_minilm

# In progress (interfaces only): experiments/cifar10/run.py, experiments/clotho/run.py

echo "### Appendix ablations"
python ablations/kernels.py
python ablations/losses.py
python ablations/siglip.py
python ablations/spectral_update.py
python ablations/batch_size.py
python ablations/stabilized_svd.py
