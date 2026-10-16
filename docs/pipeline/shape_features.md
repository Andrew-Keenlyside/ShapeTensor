# 1. Getting the shape features

## What the stage is for

Every later stage reads *gradients*: the direction a quantity changes in, at every voxel. The structure tensor
reads the gradients of the image intensity. A shape feature is any other per-voxel quantity whose gradients might
carry the fibre direction more cleanly — smoother than the raw intensity, less sensitive to the stain's noise, or
sensitive to texture rather than to edges. The stage computes a stack of such channels, `F (C, Z, Y, X)`, from
the image; everything downstream treats intensity and shape features alike, as `C` channels whose gradients are
summed or kept channel-resolved.

## What is built

One interface, three engines (`shapetensor.features.make_engine`):

| engine | channels | what it is | why it is there |
|---|---|---|---|
| `intensity` | 1 | the image itself | the control arm; what the structure tensor uses |
| `jet` | 27 → PCA | a locally normalised Gaussian derivative jet (orders 0–2 at three scales) | a hand-made shape space with the same orientation content as the image: separates "learned" from "multi-channel" |
| `anatomix16` | 16 | the anatomix network's output features | the pre-trained shape space as published |
| `anatomix48` | 48 → PCA (16 or 8) | anatomix layer 58, the last decoder features before the output head | the richer representation one layer up; the engine the suite mostly reports |

```python
from shapetensor.features import make_engine
eng = make_engine("anatomix48", nyquist="none", n_pcs=8)     # tiles of 256 + 64-voxel margins, bf16, in-place phases
F = eng(volume)                                              # (8, Z, Y, X) float16, principal components in order
```

The anatomix engine runs the network in overlapping tiles (256 voxels with a 64-voxel context margin; a 384³ pass
needs about 38 GB, which is why it never runs beside another GPU job), accumulates the phases in place, and fits a
per-volume PCA on a sample of tiles so that the 48 channels become 16 or 8 components ordered by variance. `nyquist`
optionally band-limits the features before differentiation; "none" is the setting the suite used.

## What the features look like

![feature RGB and the shape tensor on the real crop](../img/real_S2146_dec_axial.png)

The first three principal components as RGB on the HiP-CT crop: structure at the texture scale, with bundles and
vessels visible as streaks — not uniform regions per population. This is the picture to keep in mind for the next
stage.

## What the features encode — and where

The suite asked directly which layer of the network carries orientation (results log §7). With two bundles
crossing at 60°, the per-channel selectivity between them:

| layer | channels with selectivity > 0.2 | maximum |
|---|---|---|
| encoder block 1 (layer 8) | 4 / 16 | 0.42 |
| **encoder block 2 (layer 15)** | **18 / 32** | **0.83** |
| layer 58 (the `anatomix48` engine) | 5 / 48 | 0.46 |
| output (the `anatomix16` engine) | 0 / 16 | 0.06 |

Orientation is in the early convolutional features, which behave like Gabor filters with clean tuning curves
(`encoder_tuning.png`), and is progressively removed by later layers — anatomix is trained for correspondence
across subjects and modalities, which rewards invariance to orientation. The engines the pipeline exposes (layer 58,
the output) are therefore nearly orientation-blind as *feature vectors*; their *gradients* still carry orientation,
which is why they work as a gradient source and not as a population label.

## Validation

- Engine numerics (`tests/test_anatomix.py`): tiled output equals a single pass; phases accumulate without seams;
  the PCA projection is chunk-invariant.
- Read-out level: the same phantom scored through each engine (results page, "The interface" table). At the
  interface no shape engine beats the intensity tensor; in clear tissue at full texture `anatomix48` TCA is at
  4.7 / 3.1 / 2.8° against 4.6 / 2.5 / 2.0° (ρ 8 / 16 / 32); the jet is flat at 8–10°.
- Speed (A6000): the anatomix48 features cost about 1 s per 256³ pass on top of the tensor; the shape tensor plus
  eigen is 3.5× the intensity tensor.

## Adding an engine

Subclass `features.base.Engine`, return `(C, Z, Y, X)` from `__call__`, register the name in `features.__init__`.
Anything that is a per-voxel map of the image — another network's features, a filter bank, a second stain — plugs
in here; several stains are stacked at the API (`st.shapetensor([a, b], ...)`).
