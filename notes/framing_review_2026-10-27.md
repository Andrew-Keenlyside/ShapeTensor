# Framing review: are we posing the problem the model can solve? (2026-10-27)

Asked: "take a look fundamentally at the overall structure ... are we framing this problem correctly for the model to
solve? how are we handling an ill posed model, what are we assuming default morphology is?" and "is this still the same
single foundation model ... can you test this new version as long as it remains the same style of foundation model".

## 1. The problem as posed

From one scalar volume of any contrast, at any point x and scale σ, predict for every direction u:
- the material density;
- the owning entity's tree embedding and Bingham orientation fit (its "vote");
- descriptors.

Every target is one fixed answer: the true geometry, edited by hand-written visibility rules.

## 2. Findings

1. **One fixed answer under a per-direction median loss: flat is optimal.** *(Not supported by the fast tests, §5.)*
   - `density_shape_loss` is an L1 in log space per sampled direction, so its optimum is the per-direction median over
     the worlds consistent with the input.
   - When the uncertainty about a lobe's position exceeds its width (~14° at κ ~ 17), every direction is a lobe in fewer
     than half of those worlds, so the median is the floor everywhere.
   - This explains the round plateau, the near-uniform crossing cores at every orientation (X0), famN's flat crossing
     loss from step 0, and the slow fit of even 8 fixed crossings.
   - Early in training every model is uncertain, so this is also a saddle where the answer *is* identifiable.
2. **Ill-posedness handled by editing labels.**
   - The hand-written rules: presence faded over t = 3-8, orientation over t = 6-18 (towards isotropic), a fixed 0.94
     dispersion floor, radius rules for entities.
   - The model cannot compute these statistics exactly, so the same image comes with different targets: label noise and
     more hedging.
   - The rules were wrong twice: κ 600 where nothing was visible (27 Oct); two directions asked where the lattice image
     showed three (X0).
3. **The default morphology is the generator's hidden prior.**
   - The defaults:
     - no visible orientation: isotropic mass;
     - no material: background;
     - unresolved: the parent entity;
     - crossing vs fan vs kiss unclear: the generator's convention;
     - otherwise, the family frequencies. In the default mix the shape mass is 59 % sheets, 22 % blobs and 17-19 % lines,
       plus 8 % isotropic-only scenes.
   - Under ambiguity, a white-matter HiP-CT region falls back towards "sheet-ish / isotropic", not "fibres".
4. **Identifiable and unidentifiable targets share one loss.** The total orientation density at a coarse σ is identifiable
   from the image. Ownership between identical touching fibres, and tree levels beyond the receptive field, often are not.
5. **Orientation is learned per orientation.** The trunk has no rotation awareness; the fit screen (26-27 Oct) shows no
   variant fitting or transferring 64 crossings, though the image carries both directions.

## 3. Reframing (same single foundation model)

Every item keeps one model and one forward pass, with the downstream uses unchanged:

1. **Tiered outputs**, all heads of the same model:
   - T1, the image's own orientation structure: identifiable, checkable against a classical oriented-filter score,
     possibly self-supervised;
   - T2, the material populations and their densities given a stated prior;
   - T3, the entities and the hierarchy, as an interpretation with its own uncertainty.
2. **Losses that move lobes instead of erasing them.**
   - The density target smoothed on the sphere, annealed from broad to sharp (`entity_train.DENS_KAPPA_SMOOTH`).
   - Later, a spherical transport distance or multiple hypotheses where ambiguity is real.
3. **Uncertainty instead of label editing:** train against the true geometry, with a confidence output whose calibration
   is checked. The visibility rules become learned, not hand-set.
4. **An explicit, choosable prior:** an optional conditioning input ("generic" by default, which is today's behaviour;
   e.g. "white matter"). A domain mix of the generator is a mode, never a replacement of the broad mix. Outputs record
   the prior they assumed.
5. **Orientation-aware trunk features:** X1 (a classical filter bank as input) or X3 (equivariant layers).

Scope: none of this narrows the model. Tractography, fixel / ODF export, registration features and post-hoc selection
by predicates work from the same outputs, with uncertainty added.

## 4. Tests running (fit screen, Monte Carlo crossings, from famN; `pilot_entities/x12/reframe.*`)

| variant | question |
|---|---|
| base, 1000 steps | the reference |
| `anneal=2` | does the broad-to-sharp density target let the model fit and transfer crossings? (finding 1) |
| `oracle=24` | given the true directions as input, can the decoder and loss fit 64 crossings? (if not: the loss / decoder) |
| `oracle=24,anneal=2` | both |
| `oriented=24,stem=rand,anneal=2` | the classical filter bank, its stem weights not zero-started, with the annealed loss |

The long screen was replaced by fast tests (19:10 ET), at the user's request: smaller tests before hours of work.

## 5. Fast tests (19:10-19:20 ET; `scripts/learn/loss_toy.py`, `x12/loss_toy.*`)

**Finding 1 does not hold for this loss.**

- **A. The loss's optimum under positional uncertainty** (a free density fitted to 64 worlds whose two crossing axes are
  jittered by 0-30°). The current loss keeps lobes at the true axes. Its share within 12° stays at or above the truth's
  0.20 at every noise level: 0.20 / 0.20 / 0.20 / 0.20 / 0.26 / 0.32 at 0 / 5 / 10 / 15 / 20 / 30°. It does not go
  flat. The background mass (target floor ≈ 0.25 of the mean) and the mean normalisation keep the per-direction optimum
  off the floor near the axes; the median argument was wrong.
- **B. A decoder-sized MLP given the true axes** (outer products, random order; order-8 harmonics, softplus; batch 128;
  random crossings at 30-90° in random 3-D orientations):

  | loss | 300 | 600 | 900 | 1200 | 1500 steps |
  |---|---|---|---|---|---|
  | current | 0.066 | 0.077 | 0.108 | 0.153 | 0.174 |
  | annealed 2 → 200 | 0.046 | 0.091 | 0.134 | 0.168 | 0.183 |
  | current, axes 10° noisy | 0.058 | 0.068 | 0.079 | 0.113 | 0.150 |
  | annealed, axes 10° noisy | 0.045 | 0.063 | 0.103 | 0.140 | 0.162 |

  Target 0.197; uniform 0.022. With the information present, the loss and a decoder-sized head learn two lobes at any
  orientation. Annealing helps only a little.

**Consequence:** the crossing failure is upstream of the loss. The model's trunk does not deliver direction-resolved
features to the decoder (findings 5 and 2), or they do not get through the point encoding. The loss reframing
(`DENS_KAPPA_SMOOTH`) stays available but is not the fix. Findings 2-4 (label editing, hidden prior, mixed
identifiability) are unaffected by these tests.

Next fast test, C (the real model, 64 Monte Carlo crossings, 400 steps; `x12/fastC.*`): base vs `oracle=24` (the true
directions as input channels) vs `oriented=24,stem=rand` (the classical filter bank). It asks whether the network fits
and transfers when the true directions are in its input.

## 6. Where the information is lost, and the fix (19:20-20:20 ET)

Fast tests on the real model (famN start, 64 fixed Monte Carlo crossings, 400 steps, 8 held-out crossings; core share
within 12°, target 0.122, uniform 0.022; `x12/fastC.*`, `fastD.*`, `fastD2.*`):

| variant | train | held-out | density loss (first → last 50 steps) |
|---|---|---|---|
| base | 0.028 | 0.023 | 0.49 → 0.48 |
| true directions as input channels (`oracle=24`) | 0.026 | 0.022 | 0.49 → 0.48 |
| filter bank as input channels (`oriented=24,stem=rand`) | 0.029 | 0.022 | 0.49 → 0.47 |
| **direct oriented path into the decoder, true energies** (`doracle=48`) | **0.072** | **0.068** | 0.49 → **0.23** |
| direct path, image energies, first bank (48 dirs, 3 along, normalised per voxel) | 0.033 | 0.029 | 0.49 → 0.41 |
| **direct path, image energies, new bank** (256 dirs, 8 along, pooled then normalised; `odirect=256`) | **0.079** | **0.074** | 0.49 → **0.24** |
| **the same, lr 3e-4** | **0.112** | **0.112** | 0.48 → **0.18** |

Findings:
- **Through the trunk, directions do not arrive.** Information at the input, even the truth, does not reach the decoder
  within this budget.
- **The direct path works.** Per query direction, the decoder reads the line energy along it at the point; it is the
  same input at every orientation, so the density generalises across orientations by construction. With the truth it
  fits and transfers.
- **The bank's quality was the gap.**
  - `bank_check.py` measures, at crossing cores, how strongly the true axes stand out over the median direction: the
    first bank 1.1×, finding both axes in 0-2/8; the new bank 1.9×, 7/8; the true energies 2.4×, 7/8.
  - With the new bank the image version matches the oracle.

The model is unchanged in kind: the same single foundation model and outputs, with one added input path to the decoder
(`PopulationNet(odirect=...)`; new layers zero-initialised, so the start reproduces the checkpoint). The pilot runs from
20:18 ET: `famO`, phantom families with Monte Carlo crossings, famN + `--odirect 256`, 3000 steps, batch 2.

## 7. The direct-path pilot (phantom families) and the voxel-first readout test (20:18-23:45 ET)

### Run names

| file name | what it is |
|---|---|
| `famN` | **baseline phantom model**: narrow trunk (32/64/128), from scratch, 8000 steps on all phantom families (old lattice crossings); no direct path |
| `famW` | **wide-trunk phantom model**: as famN, trunk 64/128/128 |
| `fam_C_final` | **earlier best phantom model** (5/28 on the final set before today) |
| `famO` | **direct-path pilot**: famN + the direct oriented path (`odirect=256`), 3000 steps, batch 2, lr 3e-4, all families with Monte Carlo crossings |
| `Z4_all_fixes` | **generator model**: scene-generator crops, 6000 steps (0/6 on held-out generator crops) |
| `E1_control`, `E1_mc` | **generator fibre-relaxation test**: Z4 continued without / with Monte Carlo fibres; paused at ~step 775 of 2000 (relaunch with `--resume auto`) |

### Direct-path pilot

Held-out density loss over training (2 phantoms per family):
- crossings 0.49 → 0.25;
- fine crossings 0.44 → 0.21;
- sheets 0.13 → 0.05;
- curves 0.15 → 0.09;
- mixes 0.28 → 0.15;
- fans 0.15 → 0.13;
- blobs: transient rise to 0.055, back to 0.002 by step 2000.

Final set (4 per family, 32 phantoms), phantoms passing each criterion:

| | baseline | wide trunk | direct-path pilot |
|---|---|---|---|
| all criteria | 4 (empties) | 4 (empties) | 4 (empties) |
| C6 direction | 11 | 12 | **24** |
| C9 found | 8 | 14 | **19** |
| C4 leakage | 4 | 7 | 8 |
| C5 edges | 9 | 12 | 12 |
| C3 count | 17 | 14 | **10** |
| C8 ODF shape | 11 | 12 | 11 |
| C1 mass level | 4 | 4 | 4 |

The glyph figure on held-out crossings (`x12/glyphs_famO.png`) shows both bundles traced at the right angles along their
length. In the overlap the second direction is weaker; the crossing-only fast-test model drew both cleanly.

### Voxel-first readout test (no training; `scripts/learn/voxel_first_test.py`, `voxel_glyphs.py`)

The pilot was read at sigma 1 per voxel: up to two lobes, grouped by continuity, then windowed like the targets.

On the final set:
- ODF shape 11 → 24, direction 24 → 29;
- single populations no longer split into fragments ("split" leakage 0.2-0.95 → 0);
- count 10 → 8; mass level unchanged (4). The voxel masses were 1.1-6x too high (the coarse mass output used per voxel).

**But the per-voxel orientations themselves are poor** (`x12/voxel_glyphs_final.png`). In crossing cores one bundle
dominates, and the fan gets spurious second directions. The model was trained for windows of 0.5-6 voxels and the
direct path pools over 2-6, so it cannot answer per voxel. The user's verdict: "these orientations shown for per voxel
are terrible - the previous structure was far better". **Decision: voxel-first dropped; the previous structure (the
density per direction at the query scale, with the direct path) is kept.** Nothing was rebuilt; the test was a readout.

### Remaining problems in the kept structure

1. **Density to entities.** The density's directions are right, but the readout from the per-direction ownership and
   vote heads fragments and miscounts. The next quick test: rescore the pilot with a lobe readout
   (`entity_infer.cluster_lobes`) to tell whether the ownership heads or the density cause the bad counts. Objects stay
   the separable output (24 Oct instruction); a lobe readout is a diagnostic, not the goal.
2. **Mass calibration (C1)**, failing for every model so far.
3. **Cost of the direct path** (measured, while another job shared the GPU): training step 1.0 → 1.9 s and memory
   3.1 → 6.4 GB (batch 2, 512 queries × 32 directions); inference on a 64³ crop with 4096 points × 300 directions
   0.65 → 1.9 s. Likely reducible (a 128-direction bank, coarse levels at half resolution, the bank once per tile);
   untested.
