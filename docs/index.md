# ShapeTensor

![ShapeTensor](img/shapetensor_logo.png){ width="45%" }

**Fibre orientation fields for high-resolution white-matter volumes, on the GPU.**

ShapeTensor computes, from a 3-D image of white matter (hierarchical phase-contrast X-ray tomography — HiP-CT — at
micrometre resolution, or any comparable volume), the direction of the fibres at every voxel: the classic structure
tensor, a boundary-robust version of it, several fibre directions per site where bundles cross, and the same
quantities built from the gradients of learned "shape space" features instead of the intensity. It comes with the
phantom suite that was used to decide which of these to trust where, and a chunked driver that runs a whole volume
on one or more GPUs.

```python
from shapetensor import np as st          # or: from shapetensor import cp as st   (cupy out, no copy)

vec  = st.shapetensor(image, sigma=1.0, rho=16.0)             # (3, Z, Y, X) fibre direction per voxel; boundary-robust
dirs = st.shapetensor(image, rho=16.0, fixels=True)           # (3, 3, Z, Y, X): up to three directions per voxel
```

## Why

Tractography on microscopy-scale volumes starts from a local orientation estimate, almost always the **structure
tensor**: the eigenvector of least gradient energy in a window. It is accurate and cheap, and it has two systematic
failures that propagate straight into streamlines. At a **tissue interface** — the grey/white boundary, the edge of
a tract, a vessel wall — the gradients of the step overwhelm those of the fibre texture and the estimated direction
is dragged into the plane of the boundary, so streamlines wrap along it. Where **bundles cross**, a single tensor can
only report the bisector. ShapeTensor was built to test one proposed remedy — tensors and tensor decompositions of
the gradients of a pretrained shape-feature space, where a boundary is "one small change among many texture
channels" — and, having tested it, to ship what the evidence supports.

## What the evidence supports

The phantom validation (20 µm voxels, statistics matched to a real HiP-CT crop) found that:

- the boundary failure is a **step-to-texture-ratio** effect; at the real tissue's texture level the structure
  tensor is 41 / 13 / 7° off (windows of 8 / 16 / 32 voxels) next to a strong step, against 3–6° in clear tissue;
- **shape-space tensors do not rescue it** at this resolution, but **gradient excision** does — dropping the
  boundary's gradients before integrating brings the error to 7.9 / 2.7 / 1.8° where the plain tensor is
  60 / 50 / 37°, at no cost elsewhere;
- for **several populations**, constrained spherical deconvolution of the gradient-direction distribution is the
  rank-free estimator, and the shape features earn their place there, sharpening it on tight crossings;
- orientation *is* encoded in the shape network — in its encoder, not its output — and can be read per voxel by a
  small phantom-trained head.

The default `shapetensor()` call is therefore the excised intensity tensor; the shape engines, the tensor component
analysis and the fixels are available by keyword. Read [The structure tensor and where it fails](structure_tensor.md),
[Theory of the shape tensor](theory.md), [Phantoms and validation](phantoms.md) and [Results](results.md) for the
reasoning and the numbers.

## How it fits together

![pipeline overview](img/pipeline_overview.png)

A block with context margins goes through a feature engine (the intensity itself, a derivative jet, or a pretrained
3-D UNet's channels); the gradients of every channel, integrated over a window, form a channel-resolved tensor and a
gradient-direction distribution; read-outs turn them into directions; the field carries the direction, the boundary,
a quality and the fixels at every site.

!!! note "Status"
    ShapeTensor is a research prototype (version 0.1). Every claim in these pages is backed by a phantom experiment
    whose numbers are in the [results log](research/TCA_FIELD_RESULTS.md); real-tissue results are consistency
    demonstrations, not accuracy claims.
