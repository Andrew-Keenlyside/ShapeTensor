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

## Tutorial

**1. The structure tensor of a volume** — a drop-in for the `structure-tensor` package (same layout, same eigen
conventions), with numpy, cupy or torch in and the same kind out:

```python
import numpy as np
from shapetensor import st

vol = np.load("block.npy").astype(np.float32)        # (Z, Y, X) grey levels, e.g. a 20 µm HiP-CT block

S = st.structure_tensor_3d(vol, sigma=1.0, rho=16.0)  # (6, Z, Y, X) = [Sxx, Syy, Szz, Sxy, Sxz, Syz], x = last axis
val, vec = st.eig_special_3d(S)                       # val (3, ...) descending; vec (3, ...) = fibre direction (x, y, z)
```

`sigma` is the derivative scale and `rho` the integration window, both in voxels: at 20 µm, `rho=16` is 0.32 mm.

**2. The boundary-robust tensor** — the same, with the gradients of tissue interfaces excised before integration.
This is what removes the "boundary shell" (directions dragged into the grey/white boundary plane) at no cost
elsewhere:

```python
Sx = st.excised_structure_tensor_3d(vol, sigma=1.0, rho=16.0)    # factor=3: drop gradients > 3x the median coarse gradient
_, fibre = st.eig_special_3d(Sx)
```

**3. Several fibre populations per site** — constrained spherical deconvolution of the gradient-direction
distribution; the number of populations comes from the data, not from a fixed k:

```python
dirs, fracs, npop = st.fixels_3d(vol, sigma=1.0, rho=16.0)       # dirs (3, 3, Z, Y, X): up to three (x, y, z) directions
                                                                 # fracs (3, Z, Y, X), npop (Z, Y, X) in {0, 1, 2, 3}
```

**4. Shape features instead of the intensity** — any engine's channels can feed the same tensor:

```python
Ss = st.shape_structure_tensor_3d(vol, sigma=1.0, rho=16.0, engine="anatomix48")   # the block needs 64 voxels of context per face
```

or, lower level, the engine itself and the channel-resolved tensor the TCA read-outs work on:

```python
import torch
from shapetensor.features import make_engine
from shapetensor.tensor import gradients, k_dense
from shapetensor.readout import common, single

v = torch.as_tensor(vol, device="cuda")
F = make_engine("anatomix48")(v)                                 # (16, Z, Y, X): layer-58 channels, PCA to 16
G = gradients(F, sigma=1.0, truncate=4.0)                        # (48, Z, Y, X): every channel's gradient
K, sites = k_dense(G, rho=16.0, truncate=4.0, reduce=4, core=((64, 192), (64, 192), (64, 192)))   # (nz, ny, nx, 1176)
tca = single.tca(common.unpack(K.reshape(-1, K.shape[-1]), 48))  # TCA, rank 3: tca["dir"] (N, 3) in (z, y, x)
```

Engines: `intensity`, `jet` (a locally normalised derivative jet, no network), `anatomix16`, `anatomix48`.

**5. A whole volume on one or more GPUs** — chunked with halos, outputs written to a zarr group:

```bash
shapetensor volume brain_20um.ome.zarr --level 0 out.zarr --rho 16 --outputs S eig excised fixels \
    --chunk 256 256 256 --rank 0 --world 4          # one process per GPU, rank 0..3
```

**6. The orientation field a tractography consumer samples** — plain and excised directions, boundary strength and
normal, quality, fixels, on a coarse site grid:

```python
from shapetensor.field import compute_field, write_store
field = compute_field(v, rho=16.0, reduce=4)                     # dir, dir_st, gm, cl, n_cl, gap, dirs, fracs, npop
write_store("block_field.zarr", field, voxel_mm=0.02)
```

**7. Phantoms and validation**

```bash
shapetensor phantoms --pids P3-40-t0.5 P8 --seeds 0 1         # build phantoms (with twins and truth)
shapetensor run --spec results/spec_full.json                  # caches, read-outs, scores
shapetensor report --tag main                                  # REPORT.md + figures
```

Tests: `python tests/test_core.py` (filters against scipy, K against the structure tensor, SH, CP),
`python tests/test_st.py` (the volume API), `python tests/test_anatomix.py` (the engine on the GPU).

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
- Shape-space TCA does not rescue it at this resolution; **gradient excision does** (7.9 / 2.7 / 1.8° where the plain
  tensor is 60 / 50 / 37°), with the coarse gradient magnitude as the boundary detector.
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
