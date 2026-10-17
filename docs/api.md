# API

## `shapetensor.np` / `shapetensor.cp` / `shapetensor.st` — the one call

```python
shapetensor(image, *, sigma=1.0, rho=16.0, truncate=4.0, engine="intensity", excise=True, excise_mask=None,
            fixels=False, order="xyz", n_pcs=16, engine_kwargs=None, spacing=None, device="cuda", **fixel_kwargs)
```

The fibre direction field of `image (Z, Y, X)`: `(3, Z, Y, X)` unit axial vectors, or with `fixels=True`
`(3, 3, Z, Y, X)` — up to three directions per voxel, strongest first, zero where absent. `np` returns numpy, `cp`
cupy (DLPack, no copy), `st` the caller's kind. `engine` selects the feature engine; `excise` the boundary-robust
integration (intensity engine); `order` the component order.

## `shapetensor.st` — volumes in the `structure-tensor` layout

| function | returns |
|---|---|
| `structure_tensor_3d(volume, sigma, rho, truncate=4.0, *, weights=None, spacing=None, device, dtype)` | `S (6, Z, Y, X) = [Sxx, Syy, Szz, Sxy, Sxz, Syz]`, x the last axis |
| `excised_structure_tensor_3d(volume, sigma, rho, truncate=4.0, *, factor=3.0, sigma_coarse=3.0, gate=None, mask=None, spacing=None, ...)` | `S` with boundary gradients excised and the integration renormalised |
| `shape_structure_tensor_3d(volume, sigma, rho, truncate=4.0, *, engine, n_pcs=16, engine_kwargs, ...)` | `S` summed over a feature engine's channel gradients |
| `eig_special_3d(S, full=False, eigenvalue_order="desc", *, device, chunk)` | `val (3, ...)`, `vec (3, ...)` the smallest eigenvalue's vector in (x, y, z); `(3, 3, ...)` with `full=True` |
| `fixels_3d(volume, sigma, rho, truncate=4.0, *, lmax=8, reduce=None, csd_width_deg=20, peak_rel=0.5, min_sep_deg=25, nmax=3, interface_gate=3.0, sigma_coarse=3.0, upsample=True, spacing=None, device)` | `dirs (nmax, 3, ...)`, `fracs (nmax, ...)`, `npop (...)` |
| `run_volume(src, dst, *, sigma, rho, truncate, chunk, outputs, excise_factor, fixel_kwargs, level, rank, world, spacing, device, log)` | a chunked driver over a zarr volume with halos; writes `S`, `S_excised`, `val`, `vec`, `vec_excised`, `dirs`, `fracs`, `npop` |

Inputs may be numpy, cupy or torch; outputs come back as the input's kind.

## `shapetensor.features` — feature engines

`make_engine(name, **kwargs)` → an engine: `intensity` (`IntensityEngine(log=False)`), `jet`
(`JetEngine(scales=(1, 2, 4), orders=(1, 2), norm_rho=8.0, l2=True)`), `anatomix16` / `anatomix48`
(`AnatomixEngine(layer, phases=2, nyquist="phase"|"filter"|"none", tile=256, margin=64, n_pcs, l2=True, amp=True,
device)`). An engine maps a `(Z, Y, X)` torch block to `(C, Z, Y, X)` features on the same grid; `engine.margin`
says how far from a block face its output is unreliable. `shapetensor.pca.PCA.fit(F, n)` / `pca(F)` compress channels.

## `shapetensor.tensor` — the channel-resolved tensor and the ODF

- `gradients(F, sigma, truncate)` → `G (3C, Z, Y, X)`, row `c*3 + i`.
- `k_sites(G, rho, truncate, reduce, y_rows, core_zx, weights=None)` → `K (ny, nz, nx, P)` packed upper triangle on
  the sites of the given rows; `k_dense(G, rho, truncate, reduce, core, weights=None)` → `K (nz, ny, nx, P)` on a block.
- `odf_sites(...)`, `odf_dense(...)` → the gradient-direction ODF in even real spherical harmonics on the same sites.
- `excision_weights(vol, sigma_coarse=3.0, factor=3.0, truncate, mask=None, gate=None, spacing=None)` → weights in [0, 1].
- `st.spacing_ratio(spacing)`, `st.to_isotropic(v, spacing)`, `st.from_isotropic(t, shape)`: the anisotropic-voxel helpers (ratios relative to the finest axis; trilinear resampling).

## `shapetensor.readout`

- `common`: `unpack`, `channel_sum`, `cholesky_factor` (K → the per-site 3-way array), `null_dir`, `eigh_desc`,
  `canonical_sign`, `westin`, `angle_deg`.
- `single.eigen(K)` → the structure-tensor read-out; `single.tca(K, rank=3, starts, iters)` → the TCA direction, the
  pieces (`D`, `B`, `share`), the boundary candidate; `single.detectors(...)`.
- `multi.cp_als(Y, R, init, starts, iters)`, `pair_geometric(D, W)`, `n_by_peaks(odf, lmax, mode="frt"|"csd",
  kernel, rel, ...)`, `refine_fixed_n`, `n_by_bic`, `overcomplete_group`, `stability`, `postfilter`.
- `hybrid.blend(T_int, G_shape, s, s0)`, `hybrid.switch(...)`, `hybrid.calibrate_s0(s_null, q=0.99)`.

## `shapetensor.odf`

`real_sh(dirs, lmax)`, `frt(coef, lmax)`, `ring_profile(width)`, `zonal_kernel(lmax, profile)`,
`fit_ring_width(coef, fibres, lmax)`, `csd(coef, lmax, kernel, lam, tau, iters)`, `peaks(coef, lmax, nmax, rel,
min_sep_deg, ...)` — all batched on the device.

## `shapetensor.field`

`compute_field(vol, rho, reduce, sigma, truncate, margin, excise, excise_factor, multi, lmax, csd_width_deg,
peak_rel, min_sep_deg, shape_engine, ...)` → a dict of site-grid arrays; `write_store(path, field, voxel_mm,
block_origin_vox)` → a zarr group.

## `shapetensor.phantoms` and `shapetensor.validate`

`phantoms.make.build(pid, seed, tex, grid)`, `save`, `load`, `PHANTOMS`, `TextureParams`, `Grid`, `default_grid`;
`phantoms.stats.profile(vol, mask)`, `real_crop_profile(path, level, threshold)`.
`validate.suite.Spec`, `build_caches`, `run_single`, `run_multi`, `run_hybrid`, `run_excised`, `run_all`;
`validate.score` (band summaries, parallel share, detector PR, multi scores); `validate.report.write(tag)`.

## Command line

```
shapetensor stats <omezarr> [--level 0] [--threshold T] -o stats.json
shapetensor phantoms --pids ... --seeds ... [--tex tex.json] [--rho-max 32] [--vox 0.02]
shapetensor run --spec spec.json [--only caches|single|multi|hybrid|all]
shapetensor report --tag main [--out REPORT.md]
shapetensor volume SRC DST [--level L] [--sigma 1] [--rho 16] [--truncate 4] [--chunk 256 256 256]
                   [--outputs S eig excised fixels] [--excise-factor 3] [--rank r] [--world n] [--device cuda]
```

## Stage 2: dividing the shape space (`shapetensor.readout.orientation`, `readout.populations`)

- `OrientationReadout(smooth=3, ridge=1.0).fit(blocks)` — a ridge map from shape features to the voxel's orientation
  tensor, fitted on `(features, teacher directions, weights)` blocks (the teacher: the excised tensor at its
  confident single-population sites; no labels); `.predict(F)` → per-voxel orientation `(3, Z, Y, X)` and confidence.
- `window_modes(o, w, kappa_deg=15, rel=0.5, min_sep_deg=25)` — the modes of one window's orientations (populations);
  `assign(o, modes, n)` → memberships; `modes_dense(o, conf, rho, truncate, reduce, core)` — the same on a whole site
  grid at once (spherical-harmonic density, order 8).
- `populations.separate_sites(F, G, centers, rho)` — the principal-signature division of a window's voxel cloud with a
  membership-weighted tensor per signature (the first form; superseded by the readout).
- `features.cache.cached_features(engine, vol, tag)` — engine outputs cached on disk by engine configuration and volume hash.

## Stages 2–4 in one call: `shapetensor.populations.population_field`

```python
out = shapetensor.populations.population_field(image, rho=10.0, sigma=1.0, engine="anatomix_enc2", nmax=3, smooth=3, mask_threshold=None)
```

Self-calibrated on the block (the teacher is the excised tensor at `rho_teacher` on confident sites; no labels).
Returns a dict: `modes (N, 3)` (z, y, x), `activation (N, Z, Y, X)` (one layer per population), `dirs (N, 3, Z, Y, X)`
(each layer's own tensor direction) and `gap`, `masks (N, Z, Y, X)`, `union_mask`, `strongest` (int8, 1 + index; 0
outside), `vec_combined (3, Z, Y, X)` and `weight_combined` (the single field), `vec_multi (N, 3, Z, Y, X)`,
`weight_multi` and `n_vectors` (both where the masks intersect, else the strongest), `residual`, `tau`,
`n_candidates`. Pass `timings={}` to receive the seconds per stage. `readout.layers.export_population_field` writes
the same arrays to a zarr group. The pieces are `readout.layers.SignatureDictionary`, `unmix`, `activation_null`,
`population_masks`, `layer_tensor_direction`, `combined_field`, `multi_vector_field`, `compress_activation`.

