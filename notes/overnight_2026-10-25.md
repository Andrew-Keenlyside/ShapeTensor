# Overnight 2026-10-25 (ET): mass right, a combinatorial generator, local retrain

Directive (verbatim in rules.md): a purely synthetic multistage generator -- composition, then modality, then
augmentation / warps / noise, then loss and artefacts -- combinatorial, for diversity of objects and mixing; let the
running pilots test the mass; then the generator; retrain locally; test every output and stage; then speed; phantoms
and validation.

## 1. Why the entity pilots (EP9 / EP10) never learned density, mass or region

| symptom | cause | fix |
|---|---|---|
| `val_density` 1.0254 from step 0 to 2000 (f flat: max/mean 1.009) | the decoder saw the direction only by concatenation `[p, Y(u)] -> MLP`; with f flat the shape loss (scale-free) has zero net gradient on the point code (it cancels over the sphere): a saddle | `dens_sh`: log-density `y0 + Linear(p) . Y(u)` (zero init) -- a direct path. Fixed batch, 60 steps: density loss 1.46 -> 0.86, density-target correlation 0.68 -> 0.82 |
| mass no better than a constant (held-out: model 1.34, batch median 0.78) | the mass target *is* the region target windowed (corr 0.995) -- learnable -- but the PL4 trunk does not encode presence (a refit region head on frozen features: held-out corr 0.18), and ~42-50 % of training crops were unreadable (<1 % material 20-30 %, visibility < 0.2 30-35 %) | readability cap: crops with < 1 % visible material redrawn, 25 % kept (`READ_KEEP`) |
| region map a flat haze | as above, plus: **bug** -- `roles_out` (leave interfaces out) had been put on `layer_density`, not `voxel_density`: the overall-ODF lattice target still held the interfaces (edges) | fixed (`39bd776`), regression test |

New validation skills that a constant cannot fake: `val_region_corr`, `val_mass_skill` (1 - error / error of the batch
median), `val_dens_corr`.

## 2. The compositional generator: elements

`learn/compose.py` -- a population = direction field x growth x modulation x packing x topology x relation x **element**:

* lines (fibres, bundles, tubes along grown paths) -- as before;
* **sheets**: level sets of a phase the field defines (laminar -> planes, cylinder -> concentric shells, surface ->
  folded offsets, boundary -> star-shaped shells, source -> spheres, smooth -> warped lamellae containing the mean
  direction or plates across it); finite or whole, perforated, undulating; each sheet an object;
* **blobs**: ellipsoids / superquadrics / cells, the third semi-axis along the field (spindles e > 1, discs e < 1);
  scattered, packed, beaded along grown paths (chains in the tree), on a surface, inside an earlier tube.

Relations compose across elements: a sheet population wrapping an earlier bundle is a sheath; fibres ending on a sheet;
blobs embedded in a tube. Tests: per operator (`tests/test_compose.py`, 18); entity targets hold the total on every
element (median relative error 3.1 % lines, 3.4 % sheets, 2.4 % blobs). Figures: `generator_stages_sheets.png`,
`generator_stages_blobs.png`, `elements_check.png`, `training_crops_ep12_lv06.png`.

## 3. Speed

* targets: the single-object pass's unique over (group, leaf) rows -> one 1-D key: pair 6.8 -> 5.1 s (CPU, 3 threads)
* training step: the in-process producer thread cost 37 % (1.48 -> 1.08 s/step, -7 GB): pilots now `--no-producer`
  with CPU workers
* the buffer starts at 16 pairs and fills from the workers (start-up 13 min -> 36 s)
* compositional lines capped at ~35 % of the canvas (overpacked draws were the slow tail)

## 4. Pilots (64^3, PL4 init, 8000 steps, A6000 shared)

* EP11c: all fixes, the families only (compose off) -- the fixes' effect
* EP12: all fixes, compositional populations on (about half of all populations) -- the new generator

## 5. Phantom suite

`scripts/learn/phantom_suite_entities.py`: planes, shells, a two-layer helical wall, laminar rotation, fanning,
aligned spindles, fibres through a sheet, a bundle through a step; region / mass / axis against ground truth (axis
truth checked against the analytic field: 0.7 deg). EP10 (before the fixes, step 2000): vote axis 6.5 deg on planes,
18 deg shells, but 49-56 deg on dense 1-voxel fibres (laminar rotation, fanning, fibres through a sheet, the step) --
the density axis is meaningless there (f flat).
