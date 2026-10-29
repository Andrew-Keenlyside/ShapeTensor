# The pipeline

Five stages on a lattice, then the field at every voxel. One scale: `scale` is the window σ in voxels, and the lags,
the lattice step, the high-pass, the carry-forward reach and the trusted margin are all fractions or multiples of it.

| stage | what it does | module |
|---|---|---|
| [1. The lift](lift.md) | the array as a field over positions × directions: its local lagged autocorrelation on a shell; the null; the response; the deconvolution | `shapetensor.lift` |
| [2. Populations](populations.md) | lobes = populations (N from the data); sheets; carry-forward along each direction; identities | `shapetensor.populations` |
| [3. The field](field.md) | the lifted field read at every voxel: single field, co-located vectors, weights, labels; whole volumes in chunks | `shapetensor.field`, `shapetensor.volume` |

`shapetensor.field.Orientations` runs stages 1–2 on a block (`populations`) and reads the field (`field`); `run` does
both. `shapetensor.volume.run_volume` runs a whole zarr array chunk by chunk; `shapetensor.shapetensor()` is the one
call.

## Why a lift

Every reading of local orientation is a reading of the local autocorrelation `C(l) = ⟨f(x) f(x + l)⟩` over a window.
The structure tensor is `−½ ∇²C(0)`, its curvature at zero lag: a quadratic form, one axis of least change. Two
populations crossing give a quadratic form whose least axis is their bisector; nothing in the tensor can say there
were two. Read at a finite lag instead, as a function of the lag's direction, the autocorrelation keeps every
population: structure along `d` is correlated along `d` at any lag and decorrelated across it, so each population is a
ridge, and ridges add. That reading is the lift, and it is the whole reason the rest of the chain can be simple.

## What is and is not a parameter

- The lattice step is `scale / 2.5` voxels (4 at the default): the window makes the lifted field smooth at that scale,
  and the field is interpolated back to every voxel. `step=1` reads at every voxel at 64× the cost and no information.
- The presence floor and the response are measured, not set: the floor on an isotropic surrogate of the block (its
  own spectrum, random phases), the response on the block's own single-lobe positions. A whole volume shares one of
  each, so chunks agree exactly.
- The number of populations is capped by `nmax` (2) and their separation bounded below by `min_sep_deg` (30°); within
  that, N is the data's. A further lobe must hold `rel` (0.3) of the strongest and `min_share` (0.15) of the position's
  distribution.
- The carry-forward reach is `2.4 scale` voxels (0.5 mm at 20 µm) and its Gaussian `1.2 scale`; momentum can be
  switched off.
- Excision leaves voxels whose coarse gradient exceeds 3× the median out of the correlation (the structure tensor's
  boundary rule, carried over).

See [parameters](../parameters.md) for the full list and what each does on the phantoms.
