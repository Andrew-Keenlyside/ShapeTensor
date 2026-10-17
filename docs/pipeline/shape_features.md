# 1. Shape features

The block goes through **anatomix**, a contrast-agnostic 3-D UNet pretrained for registration across modalities,
and the pipeline reads its **encoder block 2**: 32 channels at half resolution (upsampled to the grid), a
receptive field of 16 voxels (0.32 mm at 20 µm). This is the layer where orientation is encoded as a linear code:
the mean feature pattern of an oriented voxel is a linear image of its orientation dyad d dᵀ to 98% on phantoms
and 79% on a real block (`notes/results_log.md` §26). Deeper layers (the decoder's layer 58, the output) lose the
per-fascicle division: their receptive fields average crossing populations into the bisector before anything
can be read.

```python
from shapetensor.features import make_engine
F = make_engine("anatomix_enc2")(block)        # (32, Z, Y, X)
```

The engine runs the network on tiles of 256³ with 64 voxels of real context on every face and keeps only each
tile's core, with reflect padding of the block (the network's own zero padding would put a step at the faces).
Its output is unreliable within 24 voxels of a block face; the pipeline's calibration never samples there.

Two other engines exist for comparison: `jet`, a multiscale Gaussian derivative jet without any network (the
control arm), and the deeper anatomix layers `anatomix48` and `anatomix16`.

## What the features are, and what they are not

A channel of encoder block 2 is a rectified (ReLU) filter response. It is zero over much of the texture and
responds where its filter lands on a fibre edge *in phase*, so a per-voxel read of these features oscillates with
the fibre texture (on a fibre / between fibres) the way the raw gradient energy does; the structure tensor is
smooth because it squares the gradient and integrates over ρ. The pipeline therefore averages the features over
3³ before any read-out (`smooth`), and every tensor it builds integrates over ρ as the baseline does. No root or
log of the features or of the activations can replace that integration: a pointwise transform moves no power
between spatial frequencies (`notes/results_log.md` §25).

The orientation content of a voxel's features is of second order in the direction — the same content as a
structure tensor at the receptive-field scale, with a learned amplitude. That sets what stage 2 can and cannot
separate: two populations, yes; a 50/50 mixture at Δθ looks like a weaker single population at the bisector; a
third population is linearly dependent on two.
