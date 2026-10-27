# Generator: poor resolution, wider windows, meandering bundles (27 Oct 2026)

What the overnight pilot trained on (the scene generator, `gen={"scene": ...}`), checked against the code, and the gaps
fixed. Not retrained.

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

Corrections to what I said earlier the same day: the training windows were sigma 1-6 (not 1-4); crossings go to 5 deg
(30-90 is the evaluation phantoms'); DiSCo's 1.7-3.4-voxel fibres at 0.5 / 0.25 um are inside the training range; and
the narrow-crossing target is not single-lobed down to ~15 deg. The model's broad lobes (kappa ~7.5) are its own.

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
