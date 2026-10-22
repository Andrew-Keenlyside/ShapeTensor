# 1. The lift

The array `f` (grey levels, or any scalar field; channels add) becomes a field over positions and directions:

$$\rho(x, u) = \frac{\langle f_c(y)\, f_c(y + r u)\rangle_{y \sim x,\; r \in [r_\min, r_\max]}}{\langle f_c(y)^2 \rangle_{y \sim x}}$$

the normalised local autocorrelation at lags on a shell, `f_c` the array with its local mean removed (a Gaussian of
`0.3 scale`), the window a Gaussian of `scale` voxels. A population of structures along `d` is nearly invariant along
`d`, so its contribution is a ridge along `d`; independent populations add (`C = C_1 + C_2`), so an intermixed crossing
lifts to two lobes, and a window over interleaved fascicles sums their parts. The structure tensor is this field's
second moment at an infinitesimal lag: it is to the lift what the diffusion tensor is to Q-ball imaging.

## Exact on the lattice

The lags are the integer vectors of the shell `0.25 scale ≤ |l| ≤ 0.65 scale` (554 of them at the default scale,
one of each ± pair), so no value is interpolated and no direction is blurred more than another. Each lag's products
are pooled onto cells of `step` voxels, summed into the directions of a hemisphere grid (160 Fibonacci directions)
with an angular kernel of 8°, and averaged over the window on the lattice. The cost is one shifted product per lag
at full resolution; everything else is on the lattice. `lift.lift` does this; `Orientations.lift` adds the excision.

## Excision

Voxels whose coarse (σ 3) gradient magnitude exceeds 3× the median — a tissue step, a vessel wall — are left out of
the correlation, each pair counting with `w(x) w(x + l)` and every direction normalised by the weight of the pairs it
kept. It is the excised structure tensor's rule carried over, so that a boundary's step does not lift to a ring.

## Presence and the null

`presence(ρ)` is the mean positive excess of the field over its median direction, times 4: the mass in the lobes,
whatever their number (two populations sharing the mass measure the same as one). Its floor is measured on the
block's **isotropic surrogate** — a random field with the block's radially averaged power spectrum and random phases,
the same scales and contrast and no anisotropy — as the 99th percentile of the surrogate's presence. The sensitivity
weight rises from 0 at the floor to 1 at twice it. Nothing is cut by it: the single field carries the weight; only
the co-located vectors are thresholded (`min_weight`, 0.2).

### What the surrogate cannot see

An isotropic surrogate has no anisotropy, so any anisotropy of the *acquisition* — a point-spread function longer
along the rotation axis, ring arcs that are constant across slices, slice-wise structure — counts as presence and can
lift to a population of its own. On the HiP-CT volume it does: see [real data](../data.md). `null="spectrum"` keeps
the block's full power spectrum in the surrogate and subtracts its mean directional lift; it removes the acquisition's
contribution, but also a region's dominant real orientation, so it is an option for a calibration region with many
orientations, not the default.

## The response and the deconvolution

A single population with the data's dispersion lifts to a lobe of some width (about 33° at half height on the
HiP-CT texture). That **response** is measured on the block itself: the positions in the top 5% of presence whose
second lobe is below a fifth of the first, aligned on their peaks and averaged as a profile over the angle; when fewer
than 200 positions qualify the selection is loosened, and failing that a Gaussian profile of 33° half-width (the
phantoms' value) stands in. (On data with known directions `response_profile` measures it directly; the two agree.) The lifted distribution is then
written as a non-negative combination of the response centred on every grid direction plus an isotropic term
(FISTA, 300 iterations, every position at once), which sharpens the lobes so that crossings separate, and smoothed
over 6° of angle.

```python
from shapetensor.lift import hemisphere, lift, presence, self_response, response_matrix, deconvolve
grid = hemisphere(160, "cuda")
O = lift(v, grid, r_min=2.5, r_max=6.5, sigma_w=10.0, sigma_m=3.0, step=4, origin=2)       # (160, z, y, x)
R, n = self_response(O, grid)                                                              # the block's own response
f = deconvolve(O, response_matrix(R, grid))                                                # the sharpened distribution
```

## Anisotropic voxels, channels, 2-D

`orientations(image, spacing=(dz, dy, dx))` resamples the array to its finest axis, lifts there and resamples the
field back. A `(K, Z, Y, X)` array is read as channels whose products add (several stains, several derived fields).
Two-dimensional arrays are read as one-slice volumes with the lags confined to the plane (`dims=2`; in preparation).
