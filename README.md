<div align="center">

<img src="images/shapetensor_logo.png" alt="ShapeTensor" width="55%" />

</div>

# ShapeTensor

Fibre orientation fields for high-resolution white-matter volumes (HiP-CT and similar), on the GPU. One package gives
you the classic structure tensor, a boundary-robust version of it, several fibre directions per site, and the same
quantities built from the gradients of learned "shape space" features instead of the intensity — plus the phantom
suite that says which of these to trust where.

## Install

```bash
git clone https://github.com/Andrew-Keenlyside/ShapeTensor.git
cd ShapeTensor
pip install -e .                 # numpy, scipy, torch (CUDA), zarr >= 3, nibabel, matplotlib
pip install -e ".[anatomix]"     # + the anatomix feature engines (weights come from the HF hub on first use)
```

A CUDA GPU is assumed (everything is torch on `device="cuda"`); `envs/shapetensor-gpu.def` builds the container and
`scripts/cluster/shapetensor_sge.sh` runs a volume on an H100 node, one process per GPU.

## Use

```python
from shapetensor import np as st          # or: from shapetensor import cp as st   (cupy out, no copy)

vec  = st.shapetensor(image, sigma=1.0, rho=16.0)                      # (3, Z, Y, X) fibre direction per voxel, (x, y, z); boundary-robust
vec  = st.shapetensor(image, rho=16.0, excise=False)                   # the plain structure tensor's direction
vec  = st.shapetensor(image, rho=16.0, engine="anatomix48")            # from shape-space features instead of the intensity
dirs = st.shapetensor(image, rho=16.0, fixels=True)                    # (3, 3, Z, Y, X): up to three directions per voxel
```

`image` is a `(Z, Y, X)` numpy, cupy or torch array of grey levels; `sigma` and `rho` are the derivative and
integration scales in voxels (at 20 µm, `rho=16` is 0.32 mm). Everything runs on the GPU. Engines: `intensity`
(default), `jet` (a locally normalised derivative jet, no network), `anatomix16`, `anatomix48` (the block needs 64
voxels of context per face). The pieces behind the call — `structure_tensor_3d` / `eig_special_3d` in the layout of
the `structure-tensor` package, `excised_structure_tensor_3d`, `shape_structure_tensor_3d`, `fixels_3d` — are in
`shapetensor.st`; the feature engines, the channel-resolved tensor and the TCA / CSD read-outs are in `features/`,
`tensor.py` and `readout/`; `field.py` writes the orientation-field store (direction, boundary, quality, fixels on a
coarse grid) a tractography pipeline samples.

A whole zarr volume, chunked with halos, one process per GPU: `shapetensor volume in.ome.zarr --level 0 out.zarr
--rho 16 --rank 0 --world 4`. Phantoms and the validation suite: `shapetensor phantoms | run | report`. Tests:
`python tests/test_core.py`, `tests/test_st.py`, `tests/test_anatomix.py`.

## How it fits together

<div align="center">

<img src="images/pipeline_overview.png" alt="ShapeTensor pipeline overview" width="100%" />

</div>

A block with context margins goes through a feature engine; the gradients of every channel, integrated over a window
ρ, form the channel-resolved tensor `K = G_ρ ∗ (g gᵀ)` (the structure tensor when there is one channel) and the
gradient-direction ODF; read-outs turn them into directions (the summed tensor's eigenvector, TCA's pieces paired into
populations, CSD peaks, the excised tensor); the field carries the single direction, the boundary, a quality and up to
three fixels per site. `figures/tca_field/graphical_abstract.png` shows the validation design beside it.

## What the validation found (20 µm phantoms calibrated to a real crop)

- The structure tensor's boundary failure is a step-to-texture-ratio effect: at the real crop's texture level it is
  41 / 13 / 7° (ρ 8 / 16 / 32 voxels) next to a 40-level step, against 3–6° in clear tissue.
- Shape-space TCA does not rescue it at this resolution; **gradient excision does** (5.4 / 2.9 / 2.1° at the step,
  5.2 / 2.5 / 1.9° at the gyrus with the `cl` gate, 1.3° at a vessel), with the coarse gradient magnitude as the
  boundary detector and nothing lost in clear tissue; the gyrus needs the gate (or a white-matter reference median),
  because the ungated threshold is too low when most of the window is smooth grey matter.
- For several populations, **CSD of the gradient-direction ODF** is the rank-free estimator; the anatomix layer-58
  features sharpen it on tight crossings (60° at 100% both within 15°).
- Orientation is encoded in anatomix's *encoder* (block 2: selectivity up to 0.83), not its output; a small
  phantom-trained head reads it per voxel at ±15–20°.

Details and every number: `docs/TCA_FIELD_REVIEW.md` (§11 for the verdict), `docs/TCA_FIELD_RESULTS.md`; the
explainer figures `figures/tca_field/tca_multi_explainer.png`, `shape_space_clusters.png`, `encoder_tuning.png`.

## Package layout

```
src/shapetensor/
  st.py             the volume API above (structure_tensor_3d, eig_special_3d, excised / shape variants, fixels_3d, run_volume)
  features/         engines: intensity | jet | anatomix16 | anatomix48      (make_engine(name))
  filters.py        scipy-identical Gaussian / derivative filters on the GPU
  tensor.py         K and the ODF on a site grid or a block; excision weights
  odf.py            spherical harmonics, Funk-Radon, CSD, peaks
  readout/          single (eigen, TCA, detectors), multi (CP, pairing, N selection), hybrid
  field.py          the orientation-field store
  phantoms/         P1-P9 phantom families with twins and truth; stats.py measures real crops to set them
  validate/         the suite, scores and report
  cli.py            shapetensor stats | phantoms | run | report | volume
```

Everything here is a **prototype**. The pre-package study scripts, their docs and data paths are described in
`docs/PROTOTYPES.md`.
