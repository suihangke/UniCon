# Reproducibility

All numbers below were produced with `bash scripts/reproduce.sh` on one NVIDIA L40S
(seed 0 unless noted). Training times depend on hardware.

## Section 4.1 — synthetic data

| Setting | Method | Paper | This code |
|---|---|---|---|
| Linear (K = 3 clusters) | UniCon | 100%, 0.02 s | 100%, 0.02 s |
| | SGD-CLIP (AdamW) | 100% | 100% (from epoch 29) |
| Nonlinear | UniCon (angular kernel) | 86% | 85.2% (mean over seeds 0-9) |
| | SGD-CLIP (tanh heads) | 84% | 82.7% at the best epoch, 70.7% after 500 epochs (mean over seeds 0-2) |

## Table 1 — Flickr30K (Recall@1 / Recall@10)

| Backbone | Method | I→T paper | I→T ours | T→I paper | T→I ours |
|---|---|---|---|---|---|
| RN-18 + SBERT | SGD-CLIP | .043 / .221 | .038 / .215 | .041 / .217 | .039 / .207 |
| | UniCon | .020 / .145 | .020 / .145 | .087 / .361 | .087 / .362 |
| RN-50 + SBERT | SGD-CLIP | .043 / .221 | .090 / .364 | .041 / .217 | .094 / .383 |
| | UniCon | .134 / .464 | .134 / .464 | .188 / .567 | .187 / .567 |
| CLIP ViT-B/32 | SGD-CLIP | .231 / .595 | .229 / .591 | .241 / .600 | .243 / .606 |
| | UniCon | .284 / .636 | .284 / .636 | .421 / .777 | .421 / .777 |

## Table 2 — MSCOCO 5K test and zero-shot Flickr30K

| Backbone | Method | I→T R@1 / R@10 | T→I R@1 / R@10 | Flickr30K R@5 I→T / T→I |
|---|---|---|---|---|
| RN-50 + SBERT | UniCon, paper | .105 / .388 | .129 / .439 | .171 / .249 |
| | UniCon, ours | .102 / .383 | .125 / .431 | .160 / .227 |
| CLIP ViT-B/32 | UniCon, paper | .329 / .685 | .292 / .644 | .808 / .766 |
| | UniCon, ours | .338 / .686 | .294 / .647 | .810 / .769 |
| | raw CLIP features (no alignment) | .338 / .687 | .294 / .652 | .816 / .771 |

The SGD-CLIP baseline defaults to 100 epochs (`--sgd-epochs 1000` for the paper's schedule).

## Settings used for the reported numbers

* **Factorization of `F1ᵀF2`.** Theorem 8 fixes only the product `F1ᵀF2 = U_r Σ_r V_rᵀ`.
  Synthetic experiments use `F1 = Σ_r^{1/2} U_rᵀ` (`factorization="sqrt"`); real-data experiments
  use `F1 = U_rᵀ, F2 = V_rᵀ` (`factorization="orthonormal"`), see `ablations/spectral_update.py`.
* **Kernel update.** `KernelUniCon` implements Theorem 9 exactly and starts from `s = 0`.
* **Angular kernel.** `k(u, v) = 1 − θ/π`; the first-order arc-cosine kernel is `"arccos"`.
* **Flickr30K.** τ = 1, ν = 2, r = 128, one batch of 23,420 pairs, 2 iterations.
* **MSCOCO, CLIP ViT-B/32.** Many-to-many `S(γ)` on the first 200 training images × 5 captions,
  τ = 0.07, ν = 1.5, r = 512, Tikhonov λ = 10 on the aggregated `F1ᵀF2` (Appendix C.4).
  With orthonormal factors this keeps the map close to the identity, hence the comparison
  with raw CLIP features above.
* **MSCOCO, RN-50 + SBERT.** τ = 0.07, ν = 1.5, r = 256, ρ = 0.1, one batch of 27,593 pairs.
* **Synthetic linear data.** Observation noise std 0.1.
* **Text encoders.** Sentence-BERT `paraphrase-MiniLM-L6-v2` (Flickr30K) and `all-MiniLM-L6-v2`
  (MSCOCO), both 384-d; set with `--text-model` in the extraction scripts.
* **In progress.** CIFAR-10 (Section 4.2) and Clotho audio-text (Appendix C.5): interfaces in
  `experiments/cifar10/` and `experiments/clotho/`.

## Appendix C.7 — triplet loss

`python ablations/losses.py` (kernel UniCon, nonlinear model, seeds 0-9):

| Loss | Paper | Mean | Best seed |
|---|---|---|---|
| CLIP | .86 | .852 | .90 |
| Triplet (m = 0.5) | .90 | .858 | .90 (seed 8) |
