# Clotho audio–text retrieval (Appendix C.5) — work in progress

Status: interface only; the numbers below are not reproduced by this folder yet.

## Protocol (paper)

Pretrained Wav2CLIP (audio) and CLIP (text) encoders, frozen, followed by a linear
projection for cross-modal alignment (Table 6):

| Method | R@1 a2t | R@1 t2a | R@5 a2t | R@5 t2a | R@10 a2t | R@10 t2a | Time |
|---|---|---|---|---|---|---|---|
| No alignment | .0010 | .0000 | .0048 | .0029 | .0096 | .0096 | – |
| UniCon | .0335 | .0249 | .1311 | .1110 | .1943 | .1789 | 13.45 s |
| SGD-CLIP | .0373 | .0278 | .1244 | .1139 | .1923 | .2077 | 347.48 s |

## What to implement

| File | Status |
|---|---|
| `extract_features.py` | `TODO(coauthor)`: write the feature cache described below |
| `run.py` | linear UniCon and SGD-CLIP are wired up through `unicon.eval.retrieval`; set `UNICON` / `SGD_CLIP` to the paper's values |

Feature cache, `data/clotho/<backbone>/`, for `split` in {train, val, test}:

```
{split}_audio_features.pt   FloatTensor (N, d_audio)
{split}_text_features.pt    FloatTensor (N, d_text)
{split}_captions.pt         list[str] of length N (optional)
```

Row *i* of the audio and text files must be a positive pair. Then
`python experiments/clotho/run.py --backbone wav2clip_clip`.
