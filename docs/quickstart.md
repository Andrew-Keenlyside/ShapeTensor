# Quick start

## One call

```python
import numpy as np
import shapetensor as st

image = np.load("block.npy").astype(np.float32)                                 # (Z, Y, X), any scalar array with oriented texture

vec = st.shapetensor(image, scale=10.0)                                          # (3, Z, Y, X): the strongest direction at every voxel, (x, y, z)
vecs, w, labels = st.shapetensor(image, scale=10.0, multi=True, weights=True, labels=True)
vec_st = st.structure_tensor(image, rho=10.0)                                    # the excised structure tensor: the baseline
```

`image` may be a numpy, cupy or torch array; the result comes back as the same kind (`shapetensor.np` and
`shapetensor.cp` force numpy or cupy). Vectors are axial: a direction and its negative are the same structure.
`vecs[k]` is zero and `w[k]` is 0 where a k-th population is absent; `labels[k]` is its identity (0 = none).

| keyword | meaning | default |
|---|---|---|
| `scale` | the window in voxels; every other length is tied to it (at 20 µm, 10 = 0.2 mm) | 10 |
| `multi` | the co-located vectors `(nmax, 3, Z, Y, X)` instead of the single field | `False` |
| `weights` | also the weights: the presence weight `(Z, Y, X)`, or per vector `(nmax, Z, Y, X)` | `False` |
| `labels` | (with `multi`) also the population identities | `False` |
| `order` | component order, `"xyz"` (as the `structure-tensor` package) or `"zyx"` | `"xyz"` |
| `spacing` | voxel size per axis `(dz, dy, dx)` for anisotropic data | isotropic |
| `nmax`, `min_sep_deg` | the most populations per voxel and their least separation | 2, 30° |

Give the block context on every face: the field is unreliable within `0.45 scale + 2 scale` voxels of a face (25 at
the default), where the weight is zero.

## A whole volume

```bash
shapetensor run brain.ome.zarr field.zarr --level 0 --scale 10 --chunk 384 384 384 --tissue-threshold 80
shapetensor structure-tensor brain.ome.zarr st.zarr --level 0 --rho 10 --arrays vec gap --float16
```

Chunks are computed with a halo and share one presence floor and one response, so the field has no seams. Several
GPUs: one process per GPU with `--rank i --world n`, then `python -c "from shapetensor.volume import stitch_labels;
stitch_labels('field.zarr')"`.

## Everything the chain produces

```python
out = st.orientations(image, scale=10.0)      # vectors in (z, y, x)
out["vecs"], out["weights"], out["labels"]    # the co-located field
out["vec"], out["weight"], out["N"], out["sheet"]
out["fodf"], out["dirs_cells"], out["mass"], out["presence"], out["labels_cells"], out["table"]   # the lattice stages
out["floor"], out["response"]                 # what the block measured about itself
```

## The install check

```bash
shapetensor demo        # a 90° band crossing phantom through the chain, tracked against the structure tensor
shapetensor bench       # the structure tensor and the chain timed on your GPU
```
