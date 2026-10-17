# Quick start

## One call

```python
import numpy as np
import shapetensor as st

image = np.load("block.npy").astype(np.float32)                   # (Z, Y, X) grey levels, e.g. a 20 µm HiP-CT block

vec = st.shapetensor(image, rho=10.0)                              # (3, Z, Y, X): one unit direction per voxel, (x, y, z)
vecs, w = st.shapetensor(image, rho=10.0, multi=True, weights=True)   # (N, 3, Z, Y, X) and (N, Z, Y, X): every population, by activation
vec = st.structure_tensor(image, rho=10.0)                         # the excised structure tensor's direction: the baseline
```

`image` may be a numpy, cupy or torch array; the result comes back as the same kind (`shapetensor.np` and
`shapetensor.cp` force numpy or cupy). Vectors are axial: a direction and its negative are the same fibre.

| keyword | meaning | default |
|---|---|---|
| `rho` | the integration window in voxels, used alike by the teacher, every population's tensor and the baseline (at 20 µm, 10 = 0.2 mm) | 10 |
| `sigma` | the derivative scale in voxels | 1 |
| `multi` | every population's direction at every voxel, ordered by activation, instead of one merged direction | `False` |
| `weights` | also return the activation weights: where a population is present | `False` |
| `tissue` | a bool mask `(Z, Y, X)` of the tissue to read (calibration samples and the window's population density) | all |
| `order` | component order, `"xyz"` (as the `structure-tensor` package) or `"zyx"` | `"xyz"` |
| `engine` | the feature engine | `"anatomix_enc2"` |

The block is calibrated on itself: the readers of the shape space are fitted on its own confident structure-tensor
sites. Give it real context on every face (the filters reach about 4 (σ + ρ) voxels, the network 24 more); a block
with few orientations (a single straight bundle) calibrates poorly — calibrate on a larger or more varied block and
apply (below).

## Everything the pipeline produces

```python
out = st.population_field(image, rho=10.0)
out["modes"]        # (N, 3) the populations' directions (z, y, x)
out["activation"]   # (N, Z, Y, X) one layer per population
out["dirs"], out["gap"]          # each layer's own direction and how unique it is
out["vec"], out["vec_gap"], out["weight"]      # the single field: merged population tensors, total activation
out["vecs"], out["weights"]      # the multi-vector field
out["tissue_mask"], out["masks"], out["strongest"], out["tau"]   # where the activation exceeds the null (not applied)
```

## Calibrate once, apply to many blocks

```python
from shapetensor.pipeline import calibrate, apply
cal = calibrate(calibration_block, rho=10.0)            # or a list of blocks
out = apply(cal, block)                                   # any block of the same data
```

## A whole volume

```bash
shapetensor run brain_20um.ome.zarr out.zarr --level 0 --rho 10 --chunk 192 192 192 --tissue-threshold 142
```

calibrates once on a central 256³ block, then runs every chunk with halos and writes `vec (3, Z, Y, X)`, `gap`,
`weight`, and `vecs (3, 3, Z, Y, X)` with `weights`. One process per GPU: `--rank r --world n`.

```bash
shapetensor structure-tensor brain_20um.ome.zarr st.zarr --level 0 --rho 10        # the baseline, same layout
shapetensor block block.tif block_field.zarr --rho 10                               # one block (.npy, .tif, .nii, zarr), with its layers
shapetensor demo                                                                     # a crossing phantom through the pipeline: no data needed
shapetensor bench --size 192 256                                                     # the structure tensor and the pipeline timed on this GPU
```

On a 48 GB Ampere GPU a 192³ block takes about 2.7 s for stages 2–4 after a 10 s calibration on the 384³ around
it, half of which is the network; the whole-volume driver runs a 256³ chunk in about 20 s.

```bash
shapetensor validate --out validation                                                # the three phantoms
```

Anisotropic voxels: `--spacing dz dy dx` (or `spacing=` in the API), for example `2.5 1 1` for 50 × 20 × 20 µm.
