# Generator: poor resolution, wider windows, meandering bundles (27 Oct 2026)

What the scene generator (the full training generator: `train.py`, `gen={"scene": ...}`) provides, checked against the
code, and the gaps fixed. Not retrained. **The overnight pilot (`pilot3x.pt`) did not train on it**: `phantom_ladder.py`
trained it on the phantom families (`phantom2`: crossings 30-90 deg, fibre radius 0.35-1.4 voxels, uniform slabs,
targets at conc 0.94 / kappa ~17, windows sigma 1-6, no voxel size or acquisition); the `gen` entry in its checkpoint's
arguments is an unused default. These fixes are for the next model, trained on the scene generator.

## What was already there

- **Voxel size per scene**: half log-uniform 1-60 um, half 2 nm-2 mm (`scene.py` `VOXEL_CORE` / `VOXEL_WIDE`); fibre
  radius 0.25-12 um, so fibres from far below a voxel to ~10 voxels (capped at a sixth of the canvas).
- **Acquisition** (`generator.nuisance`, strength ramped 0.15 -> 1 over the first half of training): PSF sigma 0.3-2
  voxels per axis, x1.5-4 along one axis when anisotropic; resolution loss (down-up by 1.2-2.5); thick slices x2-6;
  noise and artefacts by modality.
- **Targets that respect resolution**: objects closer than the blurred resolution merge into their parent
  (`entities.py`); each population's tensor fades toward isotropic where its orientation does not show (a per-voxel
  visibility from regressing the image on the population's occupancy in a sigma-5 window).
- **Crossings** in 30 % of scenes (`cross_p`): angled 5-90 deg, kissing 5-20, fanning, splitting; two or three bundles.
- **Sharp targets**: per-voxel tensors are exact tangent dyads, so a straight bundle's density target is capped only by
  the direction basis (`DENS_KAPPA_MAX` 40, half height at ~8 deg) -- crossings down to ~15 deg are two lobes in the
  target. (The evaluation phantoms' conc 0.94, kappa ~17, is theirs only.)

(Earlier the same day I described these scene-generator ranges as the overnight pilot's; they are not. For the pilot
the phantom ranges stand: no crossings below 30 deg, fibres to 1.4 voxels -- DiSCo at 0.25-0.5 um was outside them --
targets at kappa ~17, which it draws at ~7.5.)

## Gaps fixed

1. **Unresolved bundles had no texture.** Fibres below 0.45 voxel were drawn as a uniform tube (fill 0.3-0.8): the image
   showed a bundle's outline and nothing inside it, unlike DiSCo at 2-4 um or HiP-CT, where sub-voxel fibres leave
   streaky partial-volume texture along their direction. Now (`P_UNRESOLVED_TEXTURE` 0.85) a share `beta` (0.1-0.7) of
   the bundle's material is in **strands** -- groups of sub-voxel fibres, radius 0.6-3 voxels, packed over the section
   (`fibres.random_section`), each with its own density (log-normal, sd 0.15-0.5), radius varying along it, wandering
   (rms turn 2-20 deg over 6-30 voxels); the rest stays a smooth tube. Same population and object (the fibres stay
   unresolved), true tangents in the tensors, mean fill kept (within ~0.1).
   - Inside a bundle (r < R/2, rho 2) the structure tensor of a smooth tube points anywhere (median 43-73 deg off its
     axis); of a textured one along it (7-15 deg; 26-27 on fanning bundles, whose directions spread).
   - Where two cross, both directions show in the overlap's texture: with both at beta 0.4, 48 % and 7 % of its
     voxels point along one or the other by the voxel's structure tensor (3 seeds; the denser bundle dominates),
     against ~1 % in a smooth overlap.
   - The targets barely change: the visibility already credited thin tubes' interiors from their outlines (bundle
     layers' orientation visibility, median 0.98 smooth / 0.97 textured, 24 scenes at 10-40 um). What changes is the
     evidence the model is given -- the interior and the crossing now show in the image. (`train.py`'s
     `orient_from_texture` would make the targets ask for orientation only where texture shows it -- an option, not
     switched on.)
2. **Windows stopped at sigma 6.** The default query range is now sigma 1-12 voxels (`train.ENTITY["sigma"]`); targets
   for a 64^3 crop take 0.2 s (median) with it.
3. **Bundle paths only bent gently** (curvature 0.001-0.02 / voxel over 15-80 voxels), so a window's content rarely
   changed with its size. A quarter of bundles (`P_TORTUOUS`) now meander: curvature 0.03-0.08 / voxel over 5-15
   voxels, capped at 0.4 / the bundle's radius (no folding): ~15-20 deg per 15 voxels, as DiSCo's strands turn.
4. **Blur stopped at ~2 voxels.** A heavy PSF (`heavy_psf`, drawn at the cross-mix rate, 8 % x strength) widens it
   1.5-3.5x, to at most 6 voxels; the targets' resolution rule (`acquisition_blur`) follows it.

Cost: a 64^3 training crop at the pilot's settings 1.7 s to draw (scene included), 0.2 s for its targets. Random draws
moved, so scenes differ from before for the same seeds.

Tests: `tests/test_generator_resolution.py` (texture shows the direction inside and keeps the fill; both directions in a
textured crossing, neither in a smooth one; meanders turn more and do not fold; the heavy PSF widens the blur and the
resolution rule). Scene, fibre and acquisition suites: 73 passed.

## Figure

`scripts/learn/figure_generator.py` (the training pipeline itself: `draw_scene`, `entity_queries`): one scene (seed 11,
4 um voxels: two bundles of 0.65-0.7 um fibres -- below the voxel, textured -- crossing at 58 deg, the second
meandering) on the plane of its crossing. `.../overnight/figures/generator_s11.{pdf,png}` (183 mm, Nimbus Sans 7 pt).

Draft legend: **How the generator makes a training example.** **a**, The scene's populations on the plane of their
crossing: two bundles of fibres finer than the 4-um voxel (0.65-0.7 um radius), drawn as partial volume with the
texture of their packing; bundle 2 meanders. **b**, The true fibre direction (colour: direction in the plane, key;
blue: through it), weighted by fibre content. **c**, The rendered image before acquisition, with two query windows
(circles: 2 sigma, sigma = 16 um). **d**, The training target at the two windows: the population density over
directions in the plane, measured from bundle 1's local direction (ticks: the bundles' directions at the crossing) --
one lobe on bundle 1 alone (1), two at the crossing (2). **e-h**, The same crop through four acquisitions drawn by the
generator: X-ray phase contrast (e); a wide point-spread function with resolution loss (f); MRI-like contrast (g);
voxels twice as large (h). Scale bars, 100 um.

## Not done (training, not the generator)

- A sharpness input (kappa per query) and true-tangent density targets at that kappa; consistency losses across window
  size and sharpness (they need no labels: real HiP-CT too).
- More directions per query for the density loss (32 now), sampled near lobes and between neighbouring ones.
- A retrain on this generator, then the crossing-angle tuning set, the selection criteria and DiSCo.

## The rest of the generator list (27 Oct, afternoon; `notes/shortfalls_solutions_2026-10-27.md`)

**Packed tissue (`Scene.tissue`, `P_TISSUE` 0.4 of scenes where the voxel size suits; shortfall 1).** The canvas filled
wall to wall by domains -- a warped Voronoi partition, each domain stretched 2-6x along its own fibre direction as
fascicles are, abutting, with thin extracellular bands in 30 % of scenes:
- tract domains: one population on a smooth direction field, packing 0.3-0.85, contrast drawn apart from it, often
  3-25 deg from its neighbour (fascicles of one tract);
- crossing zones (30 % of domains, at least one in half the scenes): 2-3 populations interleaved **fibre by fibre**,
  **by fascicles** (the first in tubes along itself, the others between) or **in layers** (slabs along the planes'
  common normal); angles 8-40 deg in half the draws; minority shares 0.1-0.5; contrasts related in 40 %; in 15 %,
  3-4 line populations in one plane (what a sheet must be told from);
- packed cell domains (8 %); rows of cells along the fibres (half the scenes); vessels through the tissue (40 %);
- the grey-matter variant (25 % of tissue scenes): sparse fibres turning in every direction among dense cells and
  neuropil.
Elements fill each domain by a coverage grid (half resolution; seeded where free, grown in one batch along per-voxel
field grids until they leave their domain): fibres >= 0.45 voxel one by one, each an object (<= 3000); finer ones as
strands (<= 4000) **drawn as a field** -- their centrelines' material and direction splatted and blurred by their
radius -- over a smooth part: the swept renderer took 2 minutes a scene on a CPU, the field ~1 s. A tissue scene on
the CPU (the training's workers): 4-9 s to build, 1-4 s to render (ordinary scenes 3-20 s), shared by 8 crops.
Crops centre on crossing zones (`centre_pt`); the targets' orientation evidence comes from texture in tissue scenes
(`nuisance(orient_from_texture=True)`: packed tissue has no edges to credit).

**Crop strata** (`draw_scene(fill_quota=...)`, `FILL_QUOTA` (0.3, 0.3, 0.2, 0.2) packed / partial / sparse / none by
oriented fill; the training passes it): of 8 random crops the first in the drawn stratum, else the nearest. Hit 31 %
(a stratum the scene cannot supply is missed: packed crops come from packed scenes).

**Crossings and contrasts.** Bundle crossings: 8-40 deg in half the draws (`P_NARROW`); related contrasts in 40 %
(`P_RELATED_CONTRAST`); fans 25 % (`P_FAN`, was 15 %) and a fan beside 40 % of crossings; `cross_p` 0.5 in training
(was 0.3); 20 % of training queries at crossing cores (`ENTITY["p_cross"]`, now found at window scale -- interleaved
fibres of two populations rarely share a voxel). Direction-free scenes 15 % (`ISO_SCENE_P`, was 8 %); sheets weighted
0.12 (was 0.09).

**Phantom families version 2** (`phantom2.VERSION = 2`; version 1, the default, reproduces the selection and final
sets exactly -- tested): crossings 10-90 deg with 8-40 in half; a third population in 40 %; packed (slab half-width
16-30 of 64; fibre fill 0.3-0.6 split with a minority share 0.1-0.5); contrasts related in 40 %; **exact tangents for
every family** -- one target convention with the scene generator.

**Outcome** (48 training crops each, the training's settings, the same measures; `generator_v2_before_after.png`):

| | before | after |
|---|---|---|
| oriented fill of a crop, median | 7 % | 41 % |
| crops < 10 % oriented | 52 % | 33 % |
| crops > 70 % oriented | 8 % | 21 % |
| crossings below 40 deg | 30 % (of 20) | 58 % (of 38) |
| queries at crossing cores | 3.8 % | 6.2 % |
| populations per query | 2.30 | 2.67 |
| largest training window | sigma 6 | sigma 12 |
| draw time per crop (GPU, median) | 0.65 s | 0.95 s |

Tests: `tests/test_generator_resolution.py` (16: tissue fills the canvas and keeps its packing, tract tensors follow
their fields, a crossing zone at its recorded angle, grey tissue mostly direction-free, packed strata from packed
scenes, phantom version 1 unchanged / version 2 narrow and packed, plus the earlier texture, meander and heavy-PSF
tests). Scene, acquisition, fibre, dataflow, entity, direct-path, augment and readout suites: 178 passed.

**Found on the way, not fixed: the distortion mirrors the crop's edge.** The elastic warp (up to half the crops) and the
affine (30 %) resample the crop alone; what they pull from beyond it is filled by reflection (affine) or edge
replication (warp): mirrored V-shaped "crossings" and kinked fibres, or streaks across the face, up to ~20 voxels in.
The labels follow the image, but the entity targets clamp pulled points to the face -- image and targets disagree
there. Tapering the displacement to zero at the faces removed the mirrors but bent everything near them (an affine of
15-20 % undone within a band); a margin crop would cost ~5x the voxels (the affine reaches 15-25 voxels at a 64^3
crop's corners). **The fix: apply the affine and warp when the crop is cut from the canvas** (which holds the tissue
beyond the crop): resample the canvas fields through the displacement (tensors by the local Jacobian, identities by
nearest), and let the targets take entries from beyond the crop face -- a change to `draw_scene`, the crop helpers
and `entity_targets` together. Until then `p_affine=0` avoids the mirrors (the figures here use it).

**Not done.**
- Texture spectra and intensity histograms calibrated on real HiP-CT, and label-free losses on real crops: the
  project's rule is "real data stays checks only" (`plan_full_generator.md`), and the only local HiP-CT is the test
  crops. A decision for you: a held-out specimen set for calibration would make it legitimate.
- Training-side items (augmentation by the cube's symmetries, test-time rotations, losses, the sharpness input).
- A fresh test set: phantom version 2 and tissue crops from unused seeds (4.1M, 4.2M), to be generated once and kept
  unseen until a candidate is scored.
