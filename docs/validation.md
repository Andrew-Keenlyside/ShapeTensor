# Validation on phantoms

Three phantoms with truth, at 20 µm voxels, 192³ blocks with 32 voxels of context, three seeds each; every field
at σ 1, ρ 10 voxels. The structure tensor (plain and excised) is scored against the shape tensor's single field
and, at crossings, its multi-vector field. Two texture settings: **coherent** bundles (8° dispersion of the fibres
within a population, the full fibre darkening) and the **calibrated** texture (20° dispersion and half the
darkening, where the real HiP-CT crop lies).

```bash
shapetensor validate --out validation        # validation.json, validation_coherent.png, validation_calibrated.png
```

The shape pipeline is calibrated once on twelve single bundles at 15° steps of orientation (a phantom block has
one or two orientations, too few to calibrate on itself; a real block has them all).

## The three phantoms

- **bundle** — one population at 60°: the clear-tissue floor.
- **crossing** — two populations at 90° or 60°, equal density, interdigitated at the fascicle scale.
- **interface** — one population crossing a tissue step of 40 grey levels, at 90°, 60° and 30° to it in three bands.

## What the pictures show

![coherent](img/validation_coherent.png)

With coherent bundles, the structure tensor is the bisector at a crossing (the red–blue patches at 90° are its
flips between the two directions), while the shape tensor's multi-vector field carries both populations: at 60°,
where the single tensor is 26° from either population, the pair is within 10° of both at 96% of voxels. On the
single bundle and across the interface bands the shape tensor reproduces the structure tensor.

![calibrated](img/validation_calibrated.png)

At the calibrated texture the step drags the plain structure tensor by 35° within 0.1 mm of the boundary; the
excision removes this (6°), and the shape tensor inherits the excision. The crossings, however, are no longer
separable per voxel: the fibres within a population spread over 20°, the per-voxel orientation code of the features
is of second order, and the window's orientation density no longer shows two populations. The shape tensor is then
no better than the structure tensor at crossings, and on the three-band interface phantom, where the block-level
detection finds one population for three bands, it is noisier than the excised tensor (15° against 6°).

## The numbers

Median axial error in degrees (share within 15° in brackets), mean over seeds.

| case | plain structure tensor | excised structure tensor | shape tensor, single | shape tensor, multi (pair) |
|---|---|---|---|---|
| coherent: bundle | 1.4 (1.00) | 1.4 (1.00) | 1.3 (1.00) | — |
| coherent: crossing 90° | 10.4 to the nearest (0.65) | 10.4 (0.65) | 12.4 (0.60) | 16.8 (0.65); both within 15° at 55% |
| coherent: crossing 60° | 26.2 (0.00) | 26.2 (0.00) | 17.1 (0.40) | 10.4 (1.00); both within 15° at 96% |
| coherent: interface, within 0.1 mm of the step | 5.2 (0.93) | 4.4 (0.94) | 6.1 (0.80) | — |
| calibrated: bundle | 3.9 (1.00) | 3.9 (1.00) | 2.5 (1.00) | — |
| calibrated: crossing 90° | 20.2 (0.37) | 20.2 (0.37) | 33.6 (0.08) | 31.5 (0.00) |
| calibrated: crossing 60° | 23.8 (0.08) | 23.8 (0.08) | 22.6 (0.01) | not found |
| calibrated: interface, within 0.1 mm of the step | 35.4 (0.13) | 6.3 (0.93) | 14.9 (0.47) | — |
| calibrated: interface, beyond 0.5 mm | 5.5 (0.94) | 5.5 (0.94) | 14.5 (0.49) | — |

## What to take from it

- The excision is the remedy for interfaces, at any texture; it is in the baseline and inside every population
  tensor.
- Separating populations in the shape space works where the fibres within a population are coherent and the
  crossing is not too tight; it is the one thing the structure tensor cannot do at all, and the multi-vector field
  is where to look for it. Where the data's dispersion is that of the HiP-CT crop, the pipeline finds one
  population and its single field is the (excised) structure tensor with the population's weighting.
- The remaining weakness is spatial: populations are found per block, so a block holding regions of different
  orientation is read as one population and the other regions get a weakly activated tensor. Detection per
  sub-window is the next step (`notes/release_plan.md`).
