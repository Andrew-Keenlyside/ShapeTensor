# API and command line

## The one call

```python
import shapetensor as st
vec        = st.shapetensor(image, scale=10.0)                                            # (3, Z, Y, X), (x, y, z)
vecs, w, l = st.shapetensor(image, scale=10.0, multi=True, weights=True, labels=True)     # (nmax, 3, Z, Y, X), (nmax, Z, Y, X), (nmax, Z, Y, X)
vec        = st.structure_tensor(image, rho=10.0, excise=True)                            # the baseline
out        = st.orientations(image, scale=10.0)                                           # everything (vectors in (z, y, x))
```

`image` is a `(Z, Y, X)` numpy, cupy or torch array (or `(K, Z, Y, X)` channels); the result is the same kind.
`shapetensor.np` and `shapetensor.cp` force numpy or cupy. Keywords: `scale`, `multi`, `weights`, `labels`, `order`
(`"xyz"` / `"zyx"`), `spacing`, `device`, and every `Orientations` parameter ([parameters](parameters.md)).

## The chain: `shapetensor.field`

| | |
|---|---|
| `Orientations(scale=10.0, **params)` | the parameters, all tied to `scale` |
| `.lift(v)` | the lifted field `(n_dirs, z, y, x)` on the lattice |
| `.null_floor(v, mask=None)` | the presence floor from the block's isotropic surrogate |
| `.populations(v, floor="null", response=None, labels=True)` | stages 1–2 on the lattice: `N`, `mass`, `share`, `dirs_cells`, `presence`, `weight`, `fodf`, `sheet`, `labels_cells`, `table`, `floor`, `response`, `lift` |
| `.field(pop, shape, labels=True, tensor=False)` | the field at every voxel: `vecs`, `weights`, `labels`, `vec`, `weight`, `N`, `sheet` (and `tensor`) |
| `.run(v, floor="null", response=None)` | both (the lattice `N` / `sheet` become `N_cells` / `sheet_cells`) |
| `orientations(image, spacing=None, **params)` | `run` on any array kind, with anisotropic voxels handled |

## The stages

| module | contents |
|---|---|
| `shapetensor.lift` | `hemisphere`, `shell_lags`, `direction_weights`, `lift`, `iso_surrogate`, `presence`, `response_profile`, `self_response`, `response_matrix`, `deconvolve`, `angular_smooth` |
| `shapetensor.populations` | `momentum`, `peaks`, `ring_fraction`, `lobe_ring_ratio`, `lobe_shape`, `label_populations` |
| `shapetensor.volume` | `run_volume(src, dst, level, chunk, tissue_threshold, calibration_size, calibration_origin, floor, response, rank, world, **params)`, `stitch_labels(dst)` |
| `shapetensor.st` | `structure_tensor_3d`, `excised_structure_tensor_3d`, `excision_weights`, `eig_special_3d`, `structure_tensor`, `run_volume`, `spacing_ratio` |
| `shapetensor.tracking` | `track(dirs, mask, seeds, step, max_angle_deg, weights, min_weight, headings)`, `density`, `endpoint_divergence` |
| `shapetensor.phantoms` | `build(name, seed, grid, tex, texture, **kw)`: `bundle`, `crossing`, `interface`, `bend`, `band_crossing` (`t_junction`, `only_second`), `texture`; `Grid`, `TextureParams`, `truth_field` |
| `shapetensor.validate` | `run(out_dir, seeds, cases, scale)`: the acceptance suite; `SETTINGS`, `tex_for`, `lattice_scores`, `identity_scores`, `tracking_scores` |
| `shapetensor.viz` | `dec_rgb`, `identity_palette`, `identity_rgb`, … |
| `shapetensor.pipeline`, `.separate`, `.tensors`, `.merge`, `.features`, `.dev` | the 0.2 pipeline (anatomix shape features, populations unmixed per block), unchanged and importable |

## Command line

```bash
shapetensor run <in.zarr> <out.zarr> [--level 0] [--scale 10] [--chunk 384 384 384] [--tissue-threshold T] [--rank 0 --world 1]
shapetensor block <in.npy|zarr|tif|nii> <out.zarr> [--scale 10] [--spacing dz dy dx]
shapetensor structure-tensor <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--arrays vec gap] [--float16]
shapetensor validate [--out DIR] [--seeds 1] [--cases band,bend] [--scale 10]
shapetensor phantom <bundle|crossing|interface|bend|band_crossing|texture> <out.npz> [--seed 0] [--cross-deg 60] [--t-junction]
shapetensor demo                      # a 90° band crossing through the chain, tracked against the structure tensor
shapetensor bench [--size 256]        # the structure tensor and the chain timed on this GPU
shapetensor stats <in.ome.zarr> -o stats.json
```
