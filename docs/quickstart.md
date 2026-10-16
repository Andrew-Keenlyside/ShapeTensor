# Quick start

## One call

```python
import numpy as np
from shapetensor import np as st          # or: from shapetensor import cp as st   (cupy out, no copy)

image = np.load("block.npy").astype(np.float32)                # (Z, Y, X) grey levels, e.g. a 20 µm HiP-CT block

vec  = st.shapetensor(image, sigma=1.0, rho=16.0)              # (3, Z, Y, X): unit fibre direction (x, y, z); boundary-robust
vec  = st.shapetensor(image, rho=16.0, excise=False)           # the plain structure tensor's direction
vec  = st.shapetensor(image, rho=16.0, engine="anatomix48")    # from shape-space features instead of the intensity
dirs = st.shapetensor(image, rho=16.0, fixels=True)            # (3, 3, Z, Y, X): up to three directions per voxel
```

| keyword | meaning | default |
|---|---|---|
| `sigma` | derivative scale, voxels | 1.0 |
| `rho` | integration window, voxels (at 20 µm, 16 = 0.32 mm) | 16.0 |
| `engine` | `"intensity"`, `"jet"`, `"anatomix16"`, `"anatomix48"` | `"intensity"` |
| `excise` | drop boundary gradients before integrating (intensity engine) | `True` |
| `fixels` | return up to three directions per voxel from CSD of the gradient ODF | `False` |
| `order` | component order of the vectors, `"xyz"` or `"zyx"` | `"xyz"` |
| `truncate` | filter extent in standard deviations | 4.0 |
| `device` | torch device | `"cuda"` |

`image` may be a numpy, cupy or torch array. `shapetensor.np` returns numpy, `shapetensor.cp` returns cupy through
DLPack (no copy), and `shapetensor.st.shapetensor` returns the caller's kind. Vectors are axial (a direction and its
negative are the same fibre); the sign convention makes the largest-magnitude component positive.

!!! tip "Context"
    Give the block real context on every face: the filters reach `truncate * (sigma + rho)` voxels, and the anatomix
    engines a further 64. Directions computed within that distance of a block face are unreliable. The chunked
    driver below handles this for you.

## A whole volume

```bash
shapetensor volume brain_20um.ome.zarr --level 0 out.zarr --rho 16 --outputs S eig excised fixels \
    --chunk 256 256 256 --rank 0 --world 1
```

reads the volume chunk by chunk with halos and writes a zarr group holding `S` and `S_excised` (the tensors,
`(6, Z, Y, X)` in the `structure-tensor` package's layout), `val` / `vec` / `vec_excised` (eigenvalues and the fibre
direction) and `dirs` / `fracs` / `npop` (the fixels). Start one process per GPU with `--rank r --world n`; every
chunk's output equals a one-pass computation, so the split is exact.

## The pieces

The one-call API is a thin layer over `shapetensor.st`:

```python
from shapetensor import st
S = st.structure_tensor_3d(image, 1.0, 16.0)        # (6, Z, Y, X) = [Sxx, Syy, Szz, Sxy, Sxz, Syz], x = last axis
val, vec = st.eig_special_3d(S)                     # descending values; the smallest eigenvalue's vector, (x, y, z)
Sx = st.excised_structure_tensor_3d(image, 1.0, 16.0, factor=3.0)
Ss = st.shape_structure_tensor_3d(image, 1.0, 16.0, engine="anatomix48")
dirs, fracs, npop = st.fixels_3d(image, 1.0, 16.0)
```

These mirror the `structure-tensor` package's functions and conventions, so their outputs substitute for its outputs
wherever those are consumed.

## The orientation field for a tractography pipeline

```python
import torch
from shapetensor.field import compute_field, write_store
field = compute_field(torch.as_tensor(image, device="cuda"), rho=16.0, reduce=4)
write_store("block_field.zarr", field, voxel_mm=0.02)
```

On a coarse site grid (`reduce` voxels per site): the excised and plain directions, the boundary strength (coarse
gradient magnitude) and normal, the sheet dominance `cl`, the orientation gap as a quality, and the fixels.

## Phantoms and the validation suite

```bash
shapetensor phantoms --pids P3-40-t0.5 P8 --seeds 0 1       # build phantoms with twins and truth
shapetensor run --spec results/spec_full.json                # caches, read-outs, scores
shapetensor report --tag main                                # REPORT.md and figures
```

See [Phantoms and validation](phantoms.md).
