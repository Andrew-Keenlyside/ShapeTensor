# ShapeTensor

![ShapeTensor](img/shapetensor_logo.png){ width="40%" }

**Orientation fields of any array — one or several directions per voxel, chosen by the data — on the GPU.**

The structure tensor is the standard local orientation estimate for tractography on microscopy-scale volumes
(HiP-CT, serial histology, light-sheet), for fibre analysis in materials, for any array with oriented texture. It
holds one axis per voxel: where structures cross it reports their bisector, at a tissue step it is dragged into
the plane of the step, and it has no memory from one voxel to the next. A tracker following it deflects or stops
at every crossing.

ShapeTensor is built so that a tracker arriving at a crossing leaves along the population it came in on. Five
stages, all classical, nothing trained on the analysed data:

1. **Lift** — the array becomes a field over positions × directions: its local autocorrelation read on a shell of
   lags. A population of structures along `d` is invariant along `d` and lifts to a lobe at `d`; independent
   populations add. The structure tensor is this field's zero-lag second moment, which can hold one direction; the
   lifted field holds any number.
2. **Separate** — the lobes of each position's distribution are its populations, N found by the data (0, 1, 2, …),
   after the single-population response (measured on the block itself) is deconvolved. Presence is judged against
   the block's own null — a surrogate with its power spectrum and no anisotropy — and sheets (boundaries, vessel
   walls) are recognised by their ring and never read as crossings.
3. **Carry forward** — each direction's share of the field is smoothed *along that direction* and bridged where it
   holds on both sides, never past a free end: a population continues through a crossing or a gap on the evidence
   of its own arms, and a bundle that ends is not extended.
4. **Identify** — lobes of neighbouring positions that are each other's best-aligned are joined, best first, under
   the rule that one population never holds two lobes of one position: identities 1, 2, 3, … that follow a bundle
   through its bends and through the crossings it passes.
5. **Field** — the lifted field is read at every input voxel: a weighted single field everywhere (never cut) and
   co-located vectors with weights and identities where populations coexist.

```python
import shapetensor as st
vec        = st.shapetensor(image, scale=10.0)                                             # (3, Z, Y, X): the strongest direction, (x, y, z)
vecs, w, l = st.shapetensor(image, scale=10.0, multi=True, weights=True, labels=True)      # co-located vectors, weights, identities
vec        = st.structure_tensor(image, rho=10.0)                                          # the excised structure tensor, the baseline
```

One parameter matters: `scale`, the window in voxels; every other length is tied to it. The excised structure tensor
at the same window is the baseline in every comparison.

![pipeline overview](img/pipeline_overview.png)

Read [the pipeline](pipeline/index.md) for each stage, [parameters](parameters.md) for what can be set,
[validation](validation.md) for what the phantoms and the tracking tests show, and [real data](data.md) for the
HiP-CT volume.

!!! note "Status"
    Version 0.3. The research notes behind every design decision, with all the numbers, are in the repository's
    `notes/` and `TCA_FIELD_RESULTS.md`; the documentation keeps to what the package does and what has been verified.
    The 0.2 pipeline (anatomix shape features, populations unmixed per block) remains importable as
    `shapetensor.pipeline`; its pages are under `docs/archive/`.
