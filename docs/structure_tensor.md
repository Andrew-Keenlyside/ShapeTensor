# The structure tensor, the baseline

The structure tensor of a volume is the window average of the outer product of its gradient:

$$S = G_\rho * \left(\nabla I_\sigma\, \nabla I_\sigma^{\mathsf T}\right)$$

with the gradient at scale σ and the average over a window ρ. The fibre direction is the eigenvector of the
smallest eigenvalue — the direction along which the intensity changes least — and the orientation *gap*
(λ₂ − λ₃) / λ₂ is how unique that direction is. `shapetensor.st` computes it on the GPU in the layout of the
`structure-tensor` package, so its outputs substitute wherever that package's are consumed.

```python
from shapetensor import st
S = st.structure_tensor_3d(image, 1.0, 10.0)        # (6, Z, Y, X) = [Sxx, Syy, Szz, Sxy, Sxz, Syz]
val, vec = st.eig_special_3d(S)                     # eigenvalues (descending); the smallest's eigenvector, (x, y, z)
vec = st.structure_tensor(image, rho=10.0)          # the same in one call, excised (below)
```

## Where it fails

**At a tissue interface** — the grey/white boundary, the edge of a tract, a vessel wall — the gradients of the
step overwhelm those of the fibre texture inside the window, and the direction is dragged into the plane of the
boundary: streamlines wrap along it. The size of the effect is the ratio of the step to the texture; at the real
tissue's texture level, next to a step of 40 grey levels, it is tens of degrees within 0.1 mm of the boundary.

**Where bundles cross**, a single tensor reports the bisector, whatever the window.

## The excised structure tensor

The interface failure has a direct remedy inside the tensor: leave the boundary's gradients out. Gradients whose
coarse magnitude (σ = 3 voxels) exceeds three times the median are given zero weight and the window average is
renormalised by the surviving weight. Nothing changes in clear tissue (the test suite checks that a block with no
step is untouched), and at a step the error drops to the clear-tissue level. The *excised* structure tensor is the
package's baseline, its default `structure_tensor`, the teacher of the pipeline's stage 2, and the excision is
built into every population tensor of stage 3.

```python
Sx = st.excised_structure_tensor_3d(image, 1.0, 10.0, factor=3.0, mask=tissue)
```

`mask` is the tissue over which the reference median is taken (without it, the whole block; too low a reference
where much of the block is smooth grey matter or background).

## A whole volume

```bash
shapetensor structure-tensor brain.ome.zarr st.zarr --level 0 --rho 10 --chunk 256 256 256
```

writes `S`, `val` and `vec` chunk by chunk with halos; every chunk equals a one-pass computation (for the excised
tensor the reference median is per chunk).

## Anisotropic voxels

`spacing=(dz, dy, dx)` declares the voxel size per axis (for example `(2.5, 1, 1)` for 50 × 20 × 20 µm). The
filters are scaled per axis and the gradients divided by the spacing, so the tensor and its directions are in
physical coordinates. This is correct only while the coarse axis still resolves the fibre texture; where it does
not (fibres running perpendicular to a 50 µm axis at 20 µm fibre spacing), no filtering recovers what the sampling
destroyed, and isotropic sampling at the texture scale is the remedy.
