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
* **State of the model** (section 11): the density now has shape (held-out corr 0.27-0.63, EP10 ~0; phantom axes
  5-20 deg) and mass fills resolved bundles; region / mass skill on held-out crops is still low (a hard target: a
  hand-feature probe reaches 0.03-0.27); EP14b (region decoder + interface hard negatives) no longer marks a grey /
  white step as material and is the best model; **the tree embedding does not separate populations** (crossings: two
  density peaks, one entity) -- the main open problem; z-vote + population weighting did not fix it in 900 steps.
* **Running**: EP14b, resumed at 05:45. Pre-existing test failures (not tonight's): 2, listed in section 7.

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
* **trainer stalls** (EP12 from step 2000: 24 s/step; EP13 twice for 15-20 min): (1) an evaluation process on the
  shared GPU (~47 of 48 GB in use); (2) the trainers' own in-process draws (overlapping-window pairs, every 4 steps,
  on the GPU) -- many small kernels and syncs starve behind the other pilots' work (a blob-spacing loop with one sync
  per candidate made it worse). Fixed: the spacing on the CPU; the pairs drawn on the CPU, refreshed every 16 steps;
  no large evaluation beside three pilots. Also: `close()` now waits for and kills the CPU workers (one run's workers
  outlived terminate).
* the trainer's own CPU threads (its pair draws on the CPU with torch's default count): three pilots on 64 cores reached
  load 80 and 3.5-9 s/step -- now capped at 6 (commit 7a1159c, for runs started after it). Still in-process on the
  GPU: the val set (minutes at start-up on a shared GPU) -- cache it or draw it on the CPU.

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

Result at 05:00 (CPU check, 6 crops at level 0.5, owned-direction pairs): Poincaré distance by target tree distance --
EP12 final 0.47 (t = 0) ... 0.45 (t = 3), corr 0.125; EP13 step 2000 0.33 ... 0.30, corr 0.07; **EP16 step 500
1.39 ... 1.36, corr 0.16** -- the population weight spread every embedding further apart (same entity included)
without structure. Neither change separates populations within 500 steps. The embedding also varies with direction
for one entity (same-entity pairs are not close), which a per-direction output invites.

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

## 11. Final evaluation (05:35, pilots stopped; `pilot_entities/eval_FINAL/`)

Checkpoints: EP10 (step 2000, before the fixes), EP12 (2075), EP13 (2433), EP14b (2193), EP16 (897, from EP13's 1250).

### Held-out skill (12 crops per cell; region corr / mass skill / density corr / mode axis deg)

| checkpoint | compose L0.3 | compose L0.6 | compose L1.0 | families L0.3 | families L0.6 | families L1.0 |
|---|---|---|---|---|---|---|
| EP10_noif_s2000 | 0.27 / -0.69 / 0.03 / 15 | 0.25 / -0.62 / 0.12 / 12 | 0.16 / -0.18 / -0.04 / 20 | 0.35 / -0.32 / 0.01 / 15 | 0.40 / -0.03 / 0.04 / 13 | 0.20 / -0.22 / 0.06 / 13 |
| EP12_final | 0.21 / -0.04 / 0.61 / 15 | 0.16 / -0.12 / 0.40 / 14 | 0.29 / -0.27 / 0.27 / 21 | 0.24 / -0.02 / 0.30 / 15 | 0.38 / 0.07 / 0.40 / 13 | 0.26 / -0.10 / 0.46 / 16 |
| EP13_massregion | 0.38 / -0.04 / 0.59 / 16 | 0.25 / -0.04 / 0.52 / 15 | 0.23 / -0.22 / 0.40 / 18 | 0.27 / -0.02 / 0.41 / 14 | 0.31 / -0.01 / 0.38 / 15 | 0.28 / -0.08 / 0.55 / 15 |
| EP14b_dec_iface | 0.38 / 0.00 / 0.61 / 16 | 0.29 / -0.03 / 0.56 / 13 | 0.15 / -0.21 / 0.40 / 19 | 0.25 / -0.03 / 0.49 / 14 | 0.32 / 0.05 / 0.51 / 14 | 0.19 / -0.06 / 0.56 / 12 |
| EP16_zvote_treepop_from13 | 0.26 / -0.04 / 0.63 / 17 | 0.19 / -0.18 / 0.55 / 18 | 0.17 / -0.22 / 0.51 / 20 | 0.28 / -0.07 / 0.34 / 15 | 0.32 / 0.06 / 0.43 / 19 | 0.18 / -0.10 / 0.49 / 18 |

### Phantom suite, easy imaging (region corr / density axis / vote axis deg; crossings: two-peak share, error)

| phantom | EP10_noif_s2000 | EP12_final | EP13_massregion | EP14b_dec_iface | EP16_zvote_treepop_from13 |
|---|---|---|---|---|---|
| planes | 0.29 / 30.3 / 6.5 | 0.405 / 12.7 / 11.7 | -0.071 / 6.5 / 5.4 | 0.785 / 12.4 / 2.4 | 0.863 / 11.9 / 26.6 |
| shells | 0.153 / 37.8 / 18.2 | 0.008 / 20.7 / 33.5 | 0.227 / 15.2 / 13.9 | 0.138 / 17.8 / 23.3 | 0.24 / 13.3 / 15.9 |
| helical_wall | 0.366 / 74.1 / 33.3 | 0.381 / 32.7 / 45.3 | 0.283 / 12.9 / 21.8 | 0.383 / 11.0 / 23.1 | 0.341 / 18.0 / 26.8 |
| laminar_rotation | 0.195 / 29.3 / 56.2 | 0.211 / 19.5 / 31.1 | 0.294 / 15.0 / 21.3 | 0.165 / 7.8 / 15.5 | 0.275 / 9.1 / 20.7 |
| fanning | 0.378 / 77.1 / 55.7 | 0.355 / 20.9 / 15.8 | 0.409 / 19.5 / 25.2 | 0.401 / 20.2 / 18.7 | 0.408 / 24.3 / 9.3 |
| spindles | 0.541 / 51.1 / 39.0 | 0.59 / 6.1 / 18.7 | 0.75 / 18.8 / 20.6 | 0.648 / 8.5 / 10.7 | 0.74 / 14.1 / 14.0 |
| fibres_through_sheet | 0.604 / 89.4 / 52.6 | 0.684 / 26.2 / 42.1 | 0.673 / 11.9 / 14.9 | 0.623 / 8.6 / 17.5 | 0.588 / 23.7 / 37.3 |
| bundle_through_step | 0.597 / 80.6 / 49.3 | 0.537 / 21.9 / 19.3 | 0.381 / 20.3 / 19.2 | 0.503 / 23.0 / 25.8 | 0.335 / 26.7 / 30.7 |
| crossing_90 | 0.319 / 46.1 / 43.0; 0.528, 72.3 | 0.353 / 8.1 / 25.3; 0.826, 57.4 | 0.338 / 11.5 / 16.4; 0.988, 74.5 | 0.382 / 4.6 / 15.7; 0.947, 61.5 | 0.25 / 10.5 / 40.9; 0.851, 57.4 |
| crossing_45 | 0.392 / 25.4 / 62.9; 0.68, 78.0 | 0.403 / 28.8 / 43.9; 0.848, 62.4 | 0.361 / 20.3 / 20.9; 0.833, 45.4 | 0.433 / 10.1 / 9.7; 0.99, 32.4 | 0.246 / 18.4 / 34.1; 0.963, 46.0 |
| tubes | 0.521 / 36.5 / 58.3 | 0.513 / 15.5 / 32.0 | 0.584 / 11.5 / 8.4 | 0.499 / 7.4 / 9.5 | 0.566 / 10.8 / 12.8 |

### Phantom suite, hard imaging (region corr / density axis / vote axis deg; crossings: two-peak share, error)

| phantom | EP10_noif_s2000 | EP12_final | EP13_massregion | EP14b_dec_iface | EP16_zvote_treepop_from13 |
|---|---|---|---|---|---|
| planes | 0.237 / 30.6 / 6.0 | 0.225 / 10.7 / 17.9 | 0.158 / 9.5 / 56.4 | 0.672 / 12.5 / 2.4 | 0.63 / 17.7 / 33.7 |
| shells | 0.143 / 39.4 / 22.5 | 0.034 / 18.4 / 36.2 | 0.144 / 15.9 / 11.4 | 0.101 / 15.7 / 28.8 | 0.131 / 14.3 / 17.1 |
| helical_wall | 0.35 / 74.8 / 29.4 | 0.365 / 42.0 / 59.2 | 0.282 / 18.7 / 24.2 | 0.355 / 16.7 / 28.3 | 0.298 / 20.3 / 25.5 |
| laminar_rotation | 0.192 / 31.3 / 71.2 | 0.23 / 26.9 / 41.6 | 0.318 / 37.2 / 29.1 | 0.22 / 8.5 / 13.1 | 0.344 / 12.8 / 30.6 |
| fanning | 0.397 / 78.6 / 64.4 | 0.391 / 21.0 / 18.3 | 0.438 / 35.5 / 33.7 | 0.445 / 22.3 / 18.8 | 0.477 / 29.5 / 18.0 |
| spindles | 0.446 / 53.6 / 45.7 | 0.534 / 11.6 / 26.3 | 0.744 / 37.1 / 34.2 | 0.68 / 12.9 / 9.9 | 0.706 / 16.7 / 17.3 |
| fibres_through_sheet | 0.553 / 88.0 / 21.2 | 0.623 / 26.2 / 41.7 | 0.413 / 19.9 / 14.1 | 0.557 / 6.4 / 11.8 | 0.048 / 15.9 / 9.4 |
| bundle_through_step | 0.484 / 80.6 / 56.3 | 0.284 / 18.0 / 18.1 | 0.09 / 26.7 / 25.2 | 0.215 / 42.0 / 41.3 | 0.037 / 35.5 / 41.0 |
| crossing_90 | 0.368 / 48.8 / 44.7; 0.505, 58.5 | 0.353 / 16.6 / 43.4; 0.826, 56.4 | 0.331 / 26.7 / 24.1; 0.99, 74.5 | 0.366 / 8.8 / 14.6; 0.992, 64.1 | 0.285 / 19.4 / 66.0; 0.9, 57.4 |
| crossing_45 | 0.435 / 26.5 / 67.1; 0.765, 78.0 | 0.412 / 41.3 / 63.2; 0.818, 67.0 | 0.363 / 31.1 / 39.8; 0.951, 53.0 | 0.409 / 12.1 / 12.5; 0.995, 35.7 | 0.268 / 25.1 / 67.0; 0.928, 49.6 |
| tubes | 0.482 / 38.8 / 64.9 | 0.528 / 20.7 / 38.3 | 0.554 / 28.0 / 23.1 | 0.529 / 8.7 / 9.5 | 0.561 / 13.9 / 6.7 |

### Phantom suite, inverted imaging (region corr / density axis / vote axis deg; crossings: two-peak share, error)

| phantom | EP10_noif_s2000 | EP12_final | EP13_massregion | EP14b_dec_iface | EP16_zvote_treepop_from13 |
|---|---|---|---|---|---|
| planes | -0.301 / 30.0 / 6.2 | -0.325 / 12.6 / 12.9 | 0.249 / 6.8 / 5.3 | -0.699 / 12.6 / 2.4 | -0.723 / 13.7 / 27.8 |
| shells | 0.101 / 37.6 / 18.2 | 0.306 / 20.6 / 34.0 | -0.022 / 15.2 / 13.8 | 0.109 / 17.7 / 23.6 | 0.027 / 13.4 / 15.8 |
| helical_wall | 0.358 / 74.0 / 37.6 | 0.401 / 33.3 / 45.3 | 0.333 / 13.0 / 21.5 | 0.39 / 10.7 / 23.6 | 0.327 / 18.5 / 27.1 |
| laminar_rotation | 0.127 / 29.0 / 60.2 | 0.318 / 19.7 / 32.3 | 0.198 / 15.6 / 21.0 | 0.119 / 8.7 / 14.8 | 0.123 / 10.5 / 20.1 |
| fanning | 0.334 / 77.8 / 55.9 | 0.461 / 20.1 / 16.5 | 0.337 / 18.3 / 27.3 | 0.365 / 20.8 / 19.1 | 0.353 / 25.0 / 9.7 |
| spindles | 0.483 / 51.6 / 38.6 | 0.683 / 9.2 / 21.3 | 0.75 / 18.6 / 22.6 | 0.685 / 15.0 / 11.9 | 0.687 / 13.7 / 14.6 |
| fibres_through_sheet | 0.595 / 89.5 / 48.6 | 0.553 / 25.9 / 39.3 | 0.529 / 12.6 / 14.6 | 0.585 / 8.3 / 17.9 | 0.596 / 21.4 / 35.3 |
| bundle_through_step | 0.591 / 80.5 / 48.2 | 0.509 / 22.8 / 20.1 | 0.475 / 20.3 / 20.1 | 0.429 / 24.2 / 27.1 | 0.455 / 26.3 / 33.5 |
| crossing_90 | 0.305 / 45.4 / 43.7; 0.53, 72.3 | 0.373 / 7.3 / 24.8; 0.826, 56.4 | 0.42 / 10.2 / 16.9; 0.963, 74.5 | 0.367 / 5.2 / 14.4; 0.951, 61.5 | 0.184 / 9.8 / 34.6; 0.873, 57.4 |
| crossing_45 | 0.355 / 24.7 / 64.0; 0.632, 78.0 | 0.426 / 27.1 / 42.3; 0.802, 60.6 | 0.452 / 18.4 / 20.5; 0.833, 41.5 | 0.42 / 9.3 / 9.2; 0.991, 31.5 | 0.214 / 17.4 / 26.5; 0.974, 46.0 |
| tubes | 0.492 / 36.1 / 59.5 | 0.578 / 16.6 / 32.9 | 0.55 / 11.9 / 7.9 | 0.512 / 5.8 / 10.2 | 0.526 / 10.8 / 13.1 |

### Crossing benchmark (eval_entities, 40 crops): second-entity recall (axis error) per angle; false second; tree Spearman

| checkpoint | 0-10 | 10-20 | 20-30 | 30-45 | 45-60 | 60-91 | false 2nd | tree rho |
|---|---|---|---|---|---|---|---|---|
| EP12_final.json | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.28 | 0.09 |
| EP13_massregion.json | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.01 (14) | 0.43 | 0.08 |
| EP14b_dec_iface.json | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.02 | 0.04 |
| EP16_zvote_treepop_from13.json | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 (None) | 0.0 | 0.08 |


**Reading.**
* **Density (which way)**: learned in every fixed model -- held-out density corr 0.27-0.63 (EP10 ~0); on the phantoms
  the density axis is 5-20 deg where EP10's was 30-89 deg. EP14b is the most accurate (density axis 4.6-12 deg on most
  phantoms, vote axis 2.4-25 deg).
* **Region / mass (where)**: no clear held-out gain over EP10 yet (region corr 0.15-0.40 for all); mass now ~ a constant
  (skill ~0) instead of worse than one (EP10 -0.3 to -0.7). On phantoms the mass fills resolved bundles (no tram-
  tracking); sub-voxel bands are still edge-dominated.
* **Boundaries**: EP14b (interface hard negatives + region decoder) no longer marks the grey / white step as material
  (`layersvote_EP14b...png`: the step layer nearly gone; EP12 showed it strongly).
* **Crossings / populations**: the density shows two peaks at crossings (EP14b: 0.95-0.99 of shared windows, 45 deg
  crossing error 32 deg -- the best), but clustering finds no second entity (benchmark recall ~0) and the tree embedding
  does not separate populations (Spearman 0.04-0.09) -- grouping by the votes' axes separates crossing bundles outside
  the crossing core. EP16 (z-vote + population weight) did not fix the embedding in 900 steps.
* **Real data** (`realcheck.png`): the fixed models give density-peak axes (EP10 none); region maps are still hazy.

**Recommendation for the next run**: EP14b's configuration (all fixes + `--region-dec` + `{"iface": 2}`), longer; the
tree embedding needs a different formulation (see section 8) before the cluster run. EP14b was resumed at 05:45 and keeps
training (`pilot_entities/EP14b_64_dec_iface_pl4`); EP13 / EP16 resume with the same command line (`--resume auto`).

## 12. Entities from the density's lobes (06:00)

`entity_infer.cluster_lobes`: entities from the density alone (steepest ascent to its maxima on the direction grid,
maxima within 25 deg merged) -- for when the votes do not switch between a crossing's lobes. Exact on a synthetic
two-lobe density (test). On EP14b (crossing benchmark, 40 crops, `--lobes`): second-entity recall 0.01-0.05 (votes: 0)
but a false second entity in 87 % of single-entity windows (votes: 2 %) -- the learned density has extra maxima and its
second crossing peak is off-axis (the suite's 32-74 deg). Kept as an option for when the density sharpens; the votes'
clustering stays the default.

