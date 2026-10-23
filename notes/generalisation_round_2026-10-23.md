# Generalisation round (23 Oct 2026, afternoon): the 25-item list, implemented, and a local pilot

Request: "implement all changes and run a local pilot training to test feasibility and theoretical correctness".

## What was built (by item)

Generator (`learn/scene.py`, `learn/physics.py`, `learn/generator.py`):

1. **Crossing geometry** -- `Scene.interleave`: two resolved bundles crossing at an angle regroup into fascicles
   (slabs across both directions) that alternate, a fibre diameter apart, fading in along each fibre; the bundles
   thicken where they cross (the density change). Kissing: the newer bundle yields to the other's centreline
   (`deflect(extra=...)`). Splitting tracts: `bundle(mode="split")`, the halves diverge past a point (two
   populations). Crossing points are chosen inside the canvas. Test: a controlled 12-voxel crossing goes from
   hundreds of shared voxels to < 10 %.
2. **Physics image formation** -- `physics.form_image`: per-population materials; X-ray (attenuation + Paganin
   phase + propagation edge), histology (Beer--Lambert, two stains in RGB, a camera's channel mix), MRI (spoiled
   gradient echo, compartments' complex sum); additive as a special case. Overlaps are not additive (test).
3. **Orientation-dependent contrast** -- dark-field (scatter across a fibre's axis), PLI (retardance vectors in the
   2 phi plane: 90-degree crossings cancel, through-plane fibres vanish -- tests), MRI (R2* ~ sin^2/sin^4 of the angle
   to B0, frequency shift ~ 3cos^2 - 1 -- test). Any tensor, via u^T T u.
4. **Paired renders** -- `draw_pair(kind="zoom")`: the crop and the region twice its size at half the resolution
   (lattice-aligned); `kind="modality"`: the same crop under another acquisition. The background is now drawn per
   scene on the canvas, so crops at any zoom share it.
5. **Native anisotropic acquisition** -- `_slice_profile`: Gaussian or box slices of thickness t every sp voxels
   (gaps), random phase, linear or nearest resampling (MRI, histology).
6. **Multi-scale crops** -- `draw_scene(zoom=k)`: every field block-averaged (partial volume is linear).
7. **Families** -- plywood lamellae (orthogonal / twisted), fascicle hierarchies (fibres -> fascicles in sheaths,
   crimp), trabeculae (plates and rods from two level sets), tubules (tortuous, mutually deflected), epithelia
   (columnar cells on a folded basement membrane), glands (narrow tubes ending in acini), dendrites (Rall's 3/2 law,
   soma), crystals rewritten as point-rendered solids (second-moment tensors, no device syncs).
8. **Branch points as populations** -- trees are two populations by branch side (the smaller child changes side);
   capillary networks one population per dominant edge axis.
9. **Warps move identities** -- `nuisance` returns its warp grid; ids, fractions, points and descriptors follow
   (sizes scaled by the local stretch). Warps are on again in scene draws (p 0.5).
10. **Domain gap** -- `scripts/learn/domain_gap.py`: label-free image statistics, Frechet distance real vs synthetic,
    the generator settings (physics x strength) closest to a real dataset.
11. **Confidence targets** -- `visibility(per_layer_hp=True, local_corr=True)`: each layer's evidence high-passed at
    its own scale (grouped), the noise correlation volume per window.
12. **Throughput** -- `background(nprng=...)` draws its scalars on the CPU; crystals no longer use the old renderer;
    the foam and deflection searches are memory-bounded (local blocks; adaptive chunks).

Model and training (`learn/lobes.py`, `learn/lobe_train.py`, `learn/model.py`, `learn/train.py --head lobes`):

13. **Two (or more) directions per cell** -- the lobe-mixture head: J Watson lobes per cell (axis, mass, signed
    concentration, five descriptors), rendered to harmonics in closed form (Funk--Hecke; a polynomial SH basis with a
    smooth gradient), compared with the exact density. No matching; a crossing is two lobes.
14. **Concentration and NLL** -- each lobe's kappa; the direction cross-entropy (target vs predicted distribution over
    directions) is the density's NLL.
15. **Descriptors per direction; no kind/class heads; presence by mass** -- per-lobe descriptors (log size, relative
    wall, lumen fraction, flattening, angularity) rendered as descriptor densities; no kind, class or presence heads.
16. **Identity and branch / endpoint heads** -- a stride-2 embedding with a discriminative loss on object ids at the
    resolvable level (an object smaller than 1.5 voxels identified by its parent); branch-point and endpoint maps.
17. **Voxel size and the query decoder** -- FiLM by log2 voxel size (dropped 30 % of the time: works without it); the
    query decoder outputs lobes at any point and scale.
18. **More context** -- a stride-8 trunk level upsampled back: receptive field 54 -> 96 voxels.
19. **Gate and difficulty widening** -- the gate on the order-2 error; `--difficulty auto` raises a level (families,
    populations per scene, physics share) whenever the order-2 error falls below 0.6 of its energy.
20. **Loss mix** -- density per-order MSE over running energies, direction cross-entropy, descriptors, fraction,
    points, identity, query, zoom / modality consistency (stop-gradient on the fine / other side).
21. **Strength and modality coverage** -- every modality and physics in the mix; strength ramps to `--strength1`.
22. **Scale-up** -- `scripts/cluster/myriad_train_lobes.sh` (resumable; NOT submitted -- needs a go-ahead).
23. **Self-supervised adaptation** -- `--real PATH@LEVEL=UM`: rotation equivariance and intensity invariance on
    unlabelled real crops (HiP-CT S-21-46 cortex, 20 and 40 um).

Evaluation:

24. **Capability banks** -- `scripts/learn/capability_bank.py`: crossing, collapse, branching, planar, blobs,
    unresolved, snr_low / snr_high, and the same scenes under each physics; scores from any model's density.
25. **Real validation** -- `scripts/learn/real_validation.py`: HiP-CT 20 um deep brain vs SynthSeg (white vs grey matter
    anisotropy AUC, orientation coherence) on five specimens; BigBrain cortex vs its layer labels (radial / tangential
    share per layer); a structure tensor as the baseline.

## Tests

`tests/test_scene.py` (19), `tests/test_lobes.py` (8); full suite green before the pilot (see the log below).

## Pilot

(results below)

### Theory check: direct lobe fits (no network)

Free lobes fitted to the exact density of two crossing lines from 16 random starts (both axes within 10 deg):

| configuration | 90 deg, equal | 60 deg, 0.3 | 45 deg, equal | 30 deg, equal |
|---|---|---|---|---|
| kappa in [-15, 40], any loss | 3-4 | 3 | 4 | -- |
| kappa in [-10, 20], per-order MSE | 13 | 4 | 7 | 11 |
| kappa in [-10, 20], per-order MSE + direction CE, J = 4 | 12 | 11 | 14 | 12 |
| kappa in [-10, 20], per-order MSE + direction CE, J = 6 | 16 | 13 | 15 | 14 |

The failure mode was a **girdle trap**: girdles sharper than any target sheet intersecting into a fake line (three
girdles at kappa -15 whose normals are all perpendicular to the first axis), plus lobes sharper than any straight line.
The target is built from per-voxel lobes of exactly +20 (lines) and -10 (sheets) and windowing only broadens them, so
one lobe of the target lies in [-10, 20]: bounding kappa there removes the trap's ingredients. Anchored (spread)
initial axes did not help (1/8 vs 3/8): the trap is in the landscape, not the start. The direction cross-entropy
matters most for unequal crossings (60 deg / 0.3: 4 -> 11 of 16). J = 6 by default.

### Baselines on the scene banks (out of distribution for them)

G1 (slots, 1200 steps, old generator) and F1 (free SH head on the exact density, 900 steps): single-population axis
error 23-34 deg; unexplained density energy > 1 on most banks (worse than predicting nothing); G1 never predicts a
second population (0 % detected, 0 % false); F1 a second peak in ~50 % of crossings and ~50 % of single cells (its
peaks are noise). `pilot_lobes/banks_baselines.json`.

### Domain gap to real HiP-CT (S-21-46 cortex, 20 um; 32 crops of 64^3)

Frechet distance of label-free image statistics, standardised by the real set (real vs real: 4.4): best settings MRI
strength 0.6 (60.5), X-ray 0.6 (65.8), X-ray 1.0 (78.0), additive 0.6 (79.5); histology 136-286, PLI 235-2896. The
statistics that differ most: the low quantile of local coherence at a 3-voxel window (real HiP-CT is more coherent
everywhere -- fine oriented texture the generator lacks) and mid-frequency power (bands 2-4). `pilot_lobes/domain_gap_hipct20.json`.

### Contrast routes (added after the user's question: "is there additional augmentation to make it fully contrast
agnostic even within modality ... rather than categories of synthetic image data?")

`learn/augment.py` (batch time, a fresh route every time a buffered pair is used): random transfer functions
(monotone either way or non-monotone, guarded against flattening; spatially varying), gamma, unsharp / blur,
Laplacian edges of either sign, relief (directional derivative, DIC-like), smooth gain / offset fields, noise,
**missing lines** and **small missing blocks** (targets kept: small gaps are carried across from context). Draw time
(`draw_scene`): a blend with a second physics of either polarity (`p_blend`), an orientation-dependent term under any
physics (`p_aniso`), per-population and per-region intensity scrambling (`p_scramble`). Training: a second route of
the same pair and a **contrast-consistency** loss; route strength rises with the difficulty level. Bank `routes`:
16 scenes x 4 routes, scored for accuracy and for how far the top axis moves between routes. `domain_gap.py` gained
a coverage measure (real crops inside the synthetic distribution) and the full mixture with / without routes.
