# 3. The field at every voxel

The lifted field lives on a lattice of `step` voxels; the output lives on the input's grid. `Orientations.field`
interpolates the distribution `g` to every voxel (trilinear, in slabs), takes its lobes there with the same rules, and
reads:

| output | shape | meaning |
|---|---|---|
| `vecs` | `(nmax, 3, Z, Y, X)` | the co-located vectors, largest first, unit axial, zero where absent |
| `weights` | `(nmax, Z, Y, X)` | each vector's share × the presence weight; 0 below the floor or on a sheet |
| `labels` | `(nmax, Z, Y, X)` | each vector's population identity (the nearest cell's best-aligned fixel), 0 = none |
| `vec`, `weight` | `(3, Z, Y, X)`, `(Z, Y, X)` | the single field: the strongest direction everywhere, with the presence weight — never cut |
| `N`, `sheet` | `(Z, Y, X)` | populations per voxel; the sheet flag |

Vectors are in `(z, y, x)` component order inside the package and `(x, y, z)` from `shapetensor()`, the
`structure-tensor` package's layout, so the single field substitutes wherever that package's output is consumed.

## Two fields, one rule

No mask is applied anywhere. The single field has a direction at every voxel and a weight that says how much to
trust it; a consumer weights or thresholds as it needs. The co-located vectors are the one thresholded output: a
vector appears only where its population is present above the floor, because a tracker must not be offered a
direction that is not there. Where one population is present the second vector is zero; where two cross both are
there with their weights and their identities, and a tracker that follows the vector most aligned with its heading
passes through on its own bundle.

## Whole volumes

Every stage is local — the lift reaches `r_max + 3 scale` voxels, the carry-forward `reach` cells more — so a chunk
computed with a halo of that size gives, on its core, exactly what one pass would, provided the presence floor and the
response are shared. `volume.run_volume` measures both once on a calibration region and applies them to every chunk;
no blending, no seams (`tests/test_chain.py` checks the chunked field against one pass). Chunks without tissue are
skipped; a failed chunk is logged and left zero. Labels are unique per chunk and `stitch_labels` joins them.

```bash
shapetensor run brain.ome.zarr field.zarr --level 0 --scale 10 --chunk 384 384 384 --tissue-threshold 80
```

The output group holds `vecs`, `weights`, `labels`, `vec`, `weight`, `N`, `sheet` (float16 / int32 / uint8) and the
parameters, floor and response in its attributes. `--rank i --world n` splits the chunks between `n` processes (one
per GPU); stitch the labels once at the end (`shapetensor run` does when `world == 1`).

## Following the field

`shapetensor.tracking.track` is a deterministic streamline tracker over a `(K, 3, Z, Y, X)` field: at each step the
vector of the nearest voxel most aligned with the heading, within a turning limit; with `weights` and `min_weight` a
vector is followed only where its population is present. The same tracker scores every field in the
[validation](../validation.md), so fields are compared by their streamlines, not only by their vectors.
