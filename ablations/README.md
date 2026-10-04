# Ablations

| Script | Paper | What it varies |
|---|---|---|
| `kernels.py` | Appendix C.6, Table 7 (synthetic column) | kernel of kernelized UniCon |
| `losses.py` | Appendix C.7 | CLIP vs. triplet vs. sigmoid instance of Eq. 2 |
| `siglip.py` | Appendix C.7, Table 8 | sigmoid (SigLIP) loss on MSCOCO |
| `batch_size.py` | Appendix C.3 | batch size used to build `S(γ)` (Flickr30K) |
| `stabilized_svd.py` | Appendix C.4, Table 5 | Tikhonov / randomized SVD / unit-sphere normalization (MSCOCO) |
| `spectral_update.py` | Sections 3.1-3.2 | Theorem 9 update vs. SVD of `S(γ)`; `sqrt` vs. `orthonormal` factorization |

Run from the repository root, e.g. `python ablations/kernels.py`. Retrieval ablations need
the feature caches produced by `experiments/*/extract_features.py`.

Reference output (synthetic kernels: mean over seeds 0-4; Flickr30K: CLIP ViT-B/32):

| Kernel | RBF | Matérn | Cosine | Exp-cosine | Arc-cosine | Angular |
|---|---|---|---|---|---|---|
| Table 7 | .56 | .73 | .81 | .73 | .85 | .86 |
| `kernels.py` | .68 | .76 | .86 | .86 | .86 | .85 |

| Loss (seeds 0-9) | CLIP | Triplet (m = 0.5) | Sigmoid |
|---|---|---|---|
| Paper (C.7) | .86 | .90 | — |
| `losses.py`, mean | .852 | .858 | .876 |
| `losses.py`, best seed | .90 | .90 | .93 |

| Table 8, UniCon (SigLIP) | I→T R@1 | T→I R@1 | I→T R@5 | T→I R@5 | I→T R@10 | T→I R@10 |
|---|---|---|---|---|---|---|
| Paper | .3340 | .2862 | .5816 | .5334 | .6852 | .6394 |
| `siglip.py` | .3376 | .2918 | .5882 | .5422 | .6892 | .6474 |

| Batch size | 100 | 500 | 1000 | 5000 | 10000 | 20000 |
|---|---|---|---|---|---|---|
| avg R@1 | .368 | .371 | .368 | .358 | .348 | .349 |

| Table 5 (avg R@1 / R@5 / R@10) | Paper | `stabilized_svd.py` |
|---|---|---|
| Standard truncated SVD | .2235 / .4486 / .5649 | .2257 / .4568 / .5705 |
| Stabilized | .2601 / .4990 / .6149 | .2625 / .5023 / .6147 (Tikhonov + unit sphere) |

The gain comes from the Tikhonov term; randomized SVD with r = 196 lowers recall
(.2332 / .4667 / .5869). `stabilized_svd.py` and `siglip.py` also print retrieval with the
raw CLIP features (Table 5 setting: .3161 / .5633 / .6695).
