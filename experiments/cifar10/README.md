# CIFAR-10 image–image alignment (Section 4.2) — work in progress

Status: interface only; the numbers below are not reproduced by this folder yet.

## Protocol (paper)

* Positive pair = two SimCLR-style augmentations of the same image; frozen ResNet-18 features.
* **UniCon**: angular-kernel UniCon on the two views, batch-wise solutions averaged
  (batch sizes 200 / 300 / 400 are reported to behave the same), 128-d embeddings, then a
  linear probe. Paper: **61.82%** accuracy in **23.38 s**.
* **SGD baseline**: two-layer MLP projection head, bidirectional InfoNCE, SGD for 300 epochs,
  same linear probe. Paper: **62.21%** in **41.98 s**.
* Kernel ablation (Table 7, CIFAR-10 column): RBF .11, Matérn .44, Cosine .63,
  Exp-cosine .63, Arc-cosine .63, Angular .63.

## What to implement

| File | Status |
|---|---|
| `extract_features.py` | `TODO(coauthor)`: write the feature cache described below |
| `run.py` | UniCon + linear probe are wired up; set `UNICON` / `PROBE` to the paper's values and implement `run_sgd_baseline` |

Feature cache, `data/cifar10/<backbone>/`, for `split` in {train, test}:

```
{split}_view1_features.pt   FloatTensor (N, d)   features of augmentation 1
{split}_view2_features.pt   FloatTensor (N, d)   features of augmentation 2
{split}_labels.pt           LongTensor  (N,)     class labels
```

Then `python experiments/cifar10/run.py --backbone resnet18`.
