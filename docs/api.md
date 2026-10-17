# API and command line

## The one call: `shapetensor`

```python
import shapetensor as st
vec = st.shapetensor(image, rho=10.0, sigma=1.0)                          # (3, Z, Y, X), (x, y, z)
vecs, w = st.shapetensor(image, rho=10.0, multi=True, weights=True)       # (N, 3, Z, Y, X), (N, Z, Y, X)
vec = st.structure_tensor(image, rho=10.0, excise=True)                   # the baseline
out = st.population_field(image, rho=10.0)                                # everything (below)
```

`image` is a `(Z, Y, X)` numpy, cupy or torch array; the result is the same kind. `shapetensor.np` and
`shapetensor.cp` force numpy or cupy. Keywords: `rho`, `sigma`, `multi`, `weights`, `tissue`, `order` (`"xyz"` /
`"zyx"`), `engine`, `nmax`, `smooth`, `device`.

## The pipeline: `shapetensor.pipeline`

| function | what |
|---|---|
| `calibrate(volumes, tissue=None, *, engine, sigma, rho, smooth, margin, n_sample)` | fit the stage-2 readers on a block or a list of blocks → `Calibration` |
| `apply(cal, volume, tissue=None, *, nmax, min_sep_deg, min_mask_share, iterations, spacing)` | stages 2–4 on a block → dict |
| `population_field(volume, tissue=None, **calibration)` | both on one block |
| `run_volume(src, dst, *, level, chunk, nmax, tissue_threshold, calibration_size, multi, rank, world, **calibration)` | a whole zarr volume, chunked with halos |

`apply` returns, with vectors in (z, y, x): `modes (N, 3)`, `activation (N, Z, Y, X)`, `dirs (N, 3, Z, Y, X)` and
`gap (N, Z, Y, X)` per population; `vec (3, Z, Y, X)`, `vec_gap`, `weight` (the single field); `vecs (N, 3, Z, Y, X)`,
`weights` (the multi-vector field); `masks`, `tissue_mask`, `strongest`, `residual`, `tau`, `n_candidates`.
`merge.export(path, out, voxel_mm)` writes it as a zarr group.

## The stages

| module | contents |
|---|---|
| `shapetensor.features` | `make_engine(name)`: `anatomix_enc2` (default), `anatomix48`, `anatomix16`, `jet`, `intensity`; `cache.cached_features` |
| `shapetensor.separate` | `SignatureDictionary`, `OrientationReadout`, `modes_from_density`, `unmix`, `nnls_batched`, `direction_gains`, `activation_null`, `population_masks`, `hemisphere` |
| `shapetensor.tensors` | `population_tensor(image, activation, sigma, rho, excision)`, `layer_tensor`, `direction_and_gap` |
| `shapetensor.merge` | `single_field(S, A)`, `multi_field(dirs, A)`, `export` |
| `shapetensor.st` | `structure_tensor_3d`, `excised_structure_tensor_3d`, `excision_weights`, `eig_special_3d`, `structure_tensor`, `run_volume`, `spacing_ratio` |
| `shapetensor.phantoms` | `build(name, seed, grid, tex, texture, **kw)` for `bundle`, `crossing`, `interface`; `Grid`, `TextureParams`, `truth_field`, `save`, `load` |
| `shapetensor.validate` | `run(out_dir, seeds, sigma, rho, settings)` |
| `shapetensor.tracking` | `track(dirs, mask, seeds, step, max_angle_deg, weights, min_weight)`, `density`, `endpoint_divergence`, `dice` |
| `shapetensor.viz` | `dec_rgb`, `dec_activation_overlay`, `sticks`, `streamline_segments` |
| `shapetensor.stats` | `real_crop_profile(path)`: the image statistics the phantom texture is matched to |
| `shapetensor.archive` | the ODF / fixel and TCA routes, kept importable, not maintained |

## Command line

```
shapetensor run <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--chunk 192 192 192] [--nmax 3]
                [--tissue-threshold T] [--calibration-size 256] [--calibration-origin z y x] [--no-multi] [--rank r --world n]
shapetensor block <in.npy|in.zarr> <out.zarr> [--rho 10] [--sigma 1] [--voxel-mm 0.02]
shapetensor structure-tensor <in.zarr> <out.zarr> [--level 0] [--rho 10] [--sigma 1] [--no-excise] [--chunk 256 256 256]
shapetensor validate [--out validation] [--seeds 0 1 2] [--rho 10] [--settings coherent calibrated]
shapetensor phantom <bundle|crossing|interface> <out.npz> [--seed 0] [--angle 60] [--cross-deg 60] [--step 40]
shapetensor demo [--out field.zarr] [--cross-deg 90]
shapetensor bench [--size 192 256]
shapetensor stats <in.ome.zarr> [--level 0] [--threshold T] -o stats.json
```

`run` and `block` take `--spacing dz dy dx`; `block` reads `.npy`, `.tif` / `.tiff` (tifffile), `.nii` / `.nii.gz`
(nibabel) and zarr.

`run` writes `vec (3, Z, Y, X)` in (x, y, z) as the `structure-tensor` package, `gap`, `weight`, and `vecs`,
`weights`; `structure-tensor` writes `S`, `val`, `vec` in that package's layout.
