# Overnight 2026-10-25 (ET): mass right, a combinatorial generator, local retrain

Directive (verbatim in rules.md): a purely synthetic multistage generator -- composition, then modality, then
augmentation / warps / noise, then loss and artefacts -- combinatorial, for diversity of objects and mixing; let the
running pilots test the mass; then the generator; retrain locally; test every output and stage; then speed; phantoms
and validation.

## Summary (for the morning)

* **Why mass/density/region did not learn**: three causes, all fixed -- a saddle in the density head (direction only
  concatenated), mass and region hedging on a stream that was ~40-60 % unreadable crops, and a bug that left interfaces
  (edges) in the overall-ODF target. Mass is now the region map's window mean (consistent by construction).
* **Generator**: the compositional generator now composes the *element* too (lines, sheets as level sets of a field's
  phase, blobs aligned to the field) with relations and appearance operators; every operator tested; entity targets
  hold the total on every element; 0 failures in a 300-scene sweep after one singular-frame fix.
* **Speed**: data pipeline ~2x fresh pairs per step (workers drew a new scene per crop), steps +37 % without the
  producer thread, start-up 13 min -> 36 s; trainer stalls traced to per-candidate GPU syncs in blob placement (fixed).
* **Validation built**: phantom suite (11 phantoms x 3 imagings, axis truth checked to 0.7 deg), held-out skill at three
  levels, crossing benchmark, separated layers (tree or vote grouping), presence calibration, label-free real-data check.
* **State of the model** (see sections 7-8 and the final table): the density now has shape and the vote is ~14-18 deg;
  the region is learning slowly (it is a hard target: a hand-feature probe reaches only 0.03-0.27); step edges still
  light up as material; **the tree embedding is not learning** (populations are not separated in crossings) -- the main
  open problem; EP15 tests giving the embedding the vote's axes.

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
* **CPU workers drew strided seeds**: each worker rendered a new scene for almost every crop (8 crops share a scene):
  0.5 -> 1.0 new pairs per step after drawing whole seed blocks per worker; redraws reuse the cached scene once
* channels-last 3-D layout: no measurable effect (1.25 vs 1.17-1.26 s/step, A6000 shared) -- not used
* GPU at 100 % with the pilots: the step is now compute-bound

## 4. Pilots (64^3, PL4 init, 8000 steps, A6000 shared)

* EP11c: all fixes, the families only (compose off) -- the fixes' effect (stopped at ~1000 for EP13)
* EP12: all fixes, compositional populations on (about half of all populations) -- the new generator (to 2075)
* EP13: + density harmonics through an MLP, mass = the region map's window mean
* EP14b: EP13 + region decoder + interface hard negatives (`iface 2`)
* EP16: from EP13 (step 1250) + `--z-vote` + `tree_pop 2` (the tree embedding)
* X1 / X2: region-only diagnostics (with / without contrast routes)
* note: an evaluation process on the shared GPU (~47 of 48 GB in use) stalled the trainers for tens of minutes;
  so did per-candidate GPU syncs in the trainers' own in-process draws -- both fixed / avoided

## 5. Phantom suite

`scripts/learn/phantom_suite_entities.py`: planes, shells, a two-layer helical wall, laminar rotation, fanning,
aligned spindles, fibres through a sheet, a bundle through a step; region / mass / axis against ground truth (axis
truth checked against the analytic field: 0.7 deg). EP10 (before the fixes, step 2000): vote axis 6.5 deg on planes,
18 deg shells, but 49-56 deg on dense 1-voxel fibres (laminar rotation, fanning, fibres through a sheet, the step) --
the density axis is meaningless there (f flat).

## 6. Is the region learnable? (01:00-02:00)

The region map stayed a near-uniform haze for ~1000 steps in every run. Checks, in order:

* **not interference between losses**: X1 -- the region loss alone (every other weight 0), PL4 init -- region corr
  0.04 at 300 steps, BCE at the constant-predictor level, no better than the multi-task runs;
* **not the batch-time contrast routes**: X2 -- X1 without routes -- the same (0.03-0.06);
* **not a misalignment**: image / target correlations are unchanged by the cube augmentation;
* **the target is hard**: a linear probe on hand features (intensity, deviation, local variances, band-pass) predicts
  the region at held-out per-crop corr 0.27 (level 0.4, strength 0.3) and **0.03** at the validation set's level 1 /
  strength 1 (pooled across crops 0.25-0.29). The networks' pooled validation corr (EP12 0.19 at step 1250) is at
  that level; on the phantom suite (simple imaging) EP12 reaches 0.81 on planes. The val set (hardest level) cannot
  show early progress: skill_eval at levels 0.3 / 0.6 / 1.0 and the phantoms are the measures to watch.
* fixed batches fit (region 0.92) -- capacity is there; generalisation is slow on a hard, partly ambiguous target
  (resolvable background texture vs population blobs; very large blobs looked like background regions -> blobs now
  capped at a sixth of the canvas).

Options left for the region: a small decoder (`--region-dec`, EP14), a where-first curriculum (`--where-first`,
implemented, no evidence yet that it helps: X1 learned no faster alone).

## 7. EP12 at step 2000 (compose generator, all fixes): what the outputs look like

`pilot_entities/phantom_layers_EP12_s2000.png` (crossing phantoms, separated layers):
* a single resolved bundle: the mass now **fills the bundle** (no tram-tracking either side), axis 14 deg;
* uniform bands of sub-voxel fibres: the mass still runs along the band **edges** (interiors under-filled);
* a bundle across a grey / white-like step: the **step edge** lights up as material (a false positive -- interfaces
  are no longer in any target, so this is the trunk's inheritance from the PL4 init, which was trained with them);
* crossings: one model layer only (the populations are not yet separated in the tree), the second bundle's axis
  poor (27-88 deg); phantom suite crossings: two density peaks found but 62-69 deg off.

The phantom suite (step ~1300): density axis 11-37 deg on every phantom (EP10: 30-89 deg, flat density), region
0.81 on planes (0.29). Inverted planes give -0.91: a periodic stack's sheet vs gap is ambiguous under inversion.

Two pre-existing test failures (they fail identically at 411e99e, before tonight): `test_learn::test_no_falloff_at_the_faces`
(the old voxel generator's fibres, face / centre occupancy 0.75) and `test_model::test_lattice_cells_centred_on_the_target_cells`
(0.33 > 0.25). Everything touched tonight passes (346 passed).

## 8. The tree embedding is not learning yet (main open problem)

EP12 at step 2000 (and EP10): the Poincaré distance between two owned directions' embeddings does not depend on their
target tree distance -- median 0.49-0.53 whether the same fibre (t = 0), siblings (0.6-1.2) or different populations
(2.4-3.0); corr 0.085. `val_tree` 0.68-0.73 from step 750 to 2000 -- about what a collapsed embedding scores. On a
fixed batch it fits (0.65 -> 0.32 in 100 steps): no bug, slow generalisation of instance embeddings from the stream.
Consequence: crossings come out as one model layer (populations not separated).

Tried tonight (from EP13's step-1250 weights, level from 0.45): **EP15** `--z-vote` -- the embedding gets the vote's
axes (detached, zero-initialised linear term); its training tree loss at step 150 was 0.49 (EP13 0.60-0.67), stopped
at 161 for **EP16** = `--z-vote` + `tree_pop 2` (pairs from two populations weighted, not discounted by their distance).

Options (for discussion): weight the population level first (pairs from different populations vs the same population,
before instance pairs); an explicit spatial-embedding form (each direction's offset to its entity's centreline, plus
the population's orientation / appearance code) instead of a free embedding; much longer / larger-batch training
(the cluster run).

## 9. Robustness

* a cell blob's frame was singular where the field has a singular point (a source, an axis): `linalg.inv` raised in a
  CPU worker -- and a worker error stopped the whole run. Fixed (random direction there), and workers now report and
  skip a failed draw (stopping only after five in a row).
* sweep after the fix: 300 compositional scenes at 128^3 (1-5 populations, every 7th dense) + 60 training draws at random
  levels and strengths: 0 errors.
* a watchdog resumed any pilot that died (none did after the fix).

## 10. Readability by element (compositional populations only, level 0.6, before the cap's redraws)

| element | visible material (median) | crops < 1 % |
|---|---|---|
| lines | 0.030 | 38 % |
| sheets | 0.060 | 25 % |
| blobs | 0.009 | 54 % |

Blob scenes are often near-empty after acquisition: the cap redraws those crops, at a generation cost. More blobs per
population (log-uniform 20-400 instead of 6-300) barely changed it (median 0.009 -> 0.012, 54 -> 50 %): small blobs are
lost to the acquisition (visibility), not missing -- left as it was.

