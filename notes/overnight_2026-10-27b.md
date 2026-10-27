# Overnight 27-28 Oct: readout, mass by maps, direct-path speed, short tests, pilot

Asked (26 Oct, 23:10 ET): "make all fixes, test as you go and retrain to a pilot model overnight locally ... run an
adversarial review as independent agent after each change an implement any recommended improvements. aim for correct
mass, population sepearation and orientations in the simple, then hard phantoms ... really push the correctness at low
training ... then extend to longer training runs ... run a seperate review into the generator and if it is creating
data to answer the correct formulation of the question."

Only one A6000 works (the second still needs a reset), so GPU jobs share it. Times are ET.

## 1. What the direct-path pilot's numbers said (no training)

The pilot's per-phantom scores, re-read with a new mass budget (`phantom_ladder.py`'s `mass_budget`: matched,
off-axis "other", isotropic, total, as fractions of the target):

- **The total mass is roughly right; it lands in the wrong entities.** Curves and fans: 0.48-0.82 on the right
  population and 0.21-0.51 in entities pointing more than 30° off, summing to about 1.
- **The density is not flat.** With the background counted, the model's density matches the target's profile (crossing
  peaks 6.8 vs 6.4 times the sphere mean; single bundles 2.6-3.4 vs 4.3). But in the phantoms a population fills only
  10-25 % of a sigma-2.5 window (fibre volume fraction 0.2-0.35), so a 4 % ripple in the 75-89 % background floor is a
  quarter of a population's mass. The readout turned those ripples into spurious peaked entities 50-75° off axis.
- **The ODF shape failures (C8) came from the readout**: averaging the votes' second moments over an entity's directions
  broadens the fitted Bingham (kappa ~ (11, 7.5) for a (17, 13) target).

## 2. Readout changes (`entity_infer.entities_from_query`, defaults changed)

Measured with `scripts/learn/readout_lab.py` (decoder outputs cached once; each variant scored in seconds), selection set,
direct-path pilot:

| readout | pass | C1 mass | C3 count | C8 ODF shape | C9 found |
|---|---|---|---|---|---|
| votes (before) | 2/16 | 3 | 7 | 4 | 12 |
| density's lobes | 2/16 | 3 | 10 | 4 | 10 |
| lobes + votes' own kappa (new default) | 2/16 | 3 | 10 | **16** | 12 |

- Entities now form from the density's own lobes (basins of its peaks on the sphere), not from vote/embedding agreement.
- The reported concentrations are the density-weighted mean of the directions' own kappa votes; the mass split still
  uses the moment fit (a lobe as broad as the density's), so masses don't move into the floor.
- Tried and not kept: frames from the density's moment (sheets lost: a great-circle density has many peaks), pruning
  weak lobes (no effect once the above was in). A saddle-height merge of lobes (join basins across a saddle >= 0.8 of the
  lower peak) keeps sheets whole and made density frames usable (direction 12 -> 16/16 on the selection set) but split
  fans' mass to "other" (fans matched 0.82 -> 0.50): available (`saddle=0.8`), off by default.

**The same direct-path pilot on the final set (32 phantoms), old against new readout:**

| | all | C1 mass | C2 flat | C3 count | C4 leakage | C5 edges | C6 direction | C7 background | C8 ODF shape | C9 found |
|---|---|---|---|---|---|---|---|---|---|---|
| old readout (votes) | 4 | 4 | 20 | 10 | 8 | 12 | 24 | 25 | 11 | 19 |
| new readout (lobes, votes' kappa) | 4 | 5 | 20 | **16** | 9 | 11 | **27** | 25 | **30** | 17 |

The readout alone fixes the ODF shapes and most counts; mass (C1), flatness across overlaps (C2) and leakage (C4) need
the model -- the crossing cores and the split of mass among lobes.

## 3. Speed (`scripts/learn/direct_path_speed.py`)

- The bank's energies are sampled once per point, not once per (point, direction): the direct path's code for a query
  of 1024 points x 200 directions 54-57 ms -> 7.5-9.5 ms (x0.14-0.18), outputs equal (max 0.4 % of the mean density:
  half vs float storage).
- The bank is built 32 directions at a time, and its Gaussian pooling (3 levels x 256 directions) is done as matrix
  products (`filters.gaussian_mm`, equal to `gaussian` to 1e-7): the bank 192 -> 115 ms per 64³ phantom.
- A training step (2 phantoms, 512 points x 32 directions): **902 -> 636 ms, peak memory 7.4 -> 5.0 GB**.
- Decimated levels: off below sigma 8 ((1, 2, 2) changed a few crossing points' density by up to 19 % of the mean); the
  sigma-8 context level is kept at every 2nd voxel (log energies move <= 0.015).

**The whole training step** (`scripts/learn/step_profile.py`, batch 2, all families, one A6000):

| | before | after |
|---|---|---|
| step | 2087 ms | **901 ms** |
| targets (`window_mean` of every population's occupancy and tensor) | 962 | 68 |
| the direct path's bank | 256 | 255 |
| phantom generation | 221 | 220 |
| backward | 285 | 191 |
| loss | 202 | 32 |

`window_mean` (the Gaussian window mean at the queries, used for every mass target and every map loss) blurred each map
with single-channel 3-D convolutions at ~9 scales; it is now banded-matrix products with the same normalised truncated
kernel and zero padding (64³ x 15 maps: 238 -> 11 ms; 128³: 1730 -> 68 ms; max difference 3e-4, the old convolutions'
TF32 rounding). Making the next step's phantoms in a thread did not help (68 vs 67 s per 60 steps): generation and
training serialise on the GPU. Running four jobs at once on the one GPU halved the total throughput, so the short tests
were cut to two at a time.

## 4. Mass by maps, crossing cores and the context level

**Split head.** A head predicting the objects' occupancy by kind at every voxel (oriented material, isotropic objects;
densities, crossings count both; window means at the queries, log-L1 like the objects head). The readout's `mass="maps"`
then takes the oriented material's mass from its map and only *shares it out* among the oriented entities by their lobes;
the isotropic objects' mass is their map's. Before, the isotropic objects were the density's floor less the background
head -- a small difference of two large numbers.

**Crossing cores: the brighter bundle takes the density.** At the cores of the final set's crossings, the pilot's density
along one axis was 0.8-1.5 (target 4.7-6.5) and along the other 5.8-11.2 (target 5.4-6.4). In every case the winner was
the population with the larger |contrast| (e.g. contrasts 0.32 / 0.71: density 1.4 / 11.2). Line energy ~ contrast^2 x
occupancy; the target is occupancy. Per-population contrasts in the phantoms are drawn independently (0.3-0.8).

**Context level.** If E_s(u) is the line energy along u pooled at sigma s, divided by its mean over directions, then for
two populations (E_2(A) / E_8(A)) / (E_2(B) / E_8(B)) ~ (O_A,2 / O_A,8) / (O_B,2 / O_B,8): the contrasts cancel. A
classical check on 16 held-out crossings (`ctx_check`): the log mass ratio A/B at the cores from the sigma-2 energies has
median error 0.45 (up to 1.5 where contrasts differ 2x); with the sigma-8 ratio 0.25 (the large errors gone). The direct
path now takes its pooling levels from `od_sigmas` (default 2, 4, 6 as before; `--od-sigmas 2 4 6 8` adds the context
level).

**Per-population lobe-mass loss** (option `--lobe-mass W`): each oriented population's share of the density (directions
weighted by the target's ownership) against the target's, log-L1. Review 2 measured it as weak and entangled with lobe
shape (a 2x mass error moves it by 0.25-0.34 in log; a 2x kappa error by 0.11-0.29), so it is off in the pilot.

**Mass votes** (`--mass-vote`; the reviewer's suggestion): each direction an oriented population owns also votes that
population's window mass (a small MLP beside the per-direction one; log-L1 on owned directions). The readout's
`mass="votes"` takes each entity's density-weighted mean vote; `"maps_votes"` takes the votes' shares of the oriented
map's total. Masses then no longer hang on the density's shape (lobe widths, floor ripples, which lobe stands out at a
crossing core). The pilot trains it; which readout to use is chosen afterwards on the selection set.

**Maps readout hardening (review 2):** the maps branch no longer runs after the objects branch (which could zero every
lobe); the scale-up is at most 2x the density split; a partner lobe typed isotropic keeps its share; oriented mass with
no lobe to carry it is reported as `ori_unres`, never as isotropic objects.

## 5. Short training tests

- 23:45: four fine-tunes/short runs from the direct-path pilot (control; + split head; + split head + lobe weighting; from
  scratch on the simple families). On one GPU they ran ~4x slower each (half the total throughput), and review 1 then
  found the bank ran in bf16 under autocast in all of them -- stopped at ~400 steps (00:35) without results; their
  question is answered by the runs below on fixed code.
- 00:43: **crossing context A/B** -- from scratch, crossings only (cross, fine), 1200 steps, scored on 8 held-out
  crossings: without (`x_base`) and with (`x_ctx`) the sigma-8 context level. (results below)
- The pilot itself is scored on the selection set every 2000 steps (`--score-every`), so correctness is followed from low
  training upward on every family.

## 6. Reviews

### 6.1 Code review 1: readout and speed (independent agent, CPU checks)

Verified by running: the (1, 1, 1) bank equals the old build; the per-point sampling matches the per-pair code (2.5e-4 on a
code scale of 3, from the old half storage); plain-tensor `feats.oe` callers, `query(t_rel)` indexing, `with_scal` /
`with_oe` and the frame conventions (`BG.fit` columns a, b, m against `_types` / `match_p2`) are all right; peak memory of
the bank build 4.15 -> 1.80 GB (CPU).

Found and fixed:
- **Major (inherited): the bank ran in bf16 under the training's autocast** (the high-pass conv and the pooling), but in
  fp32 at inference: log energies differed by mean 0.12, p99 1.0 at the sigma-2 level, the direct path's code by up to
  1.4 on a scale of 3.9 -- the direct path trained on rounding noise in low contrast and was scored on clean inputs. Now
  `direct_levels`, `scalar_fields` and `od_code`'s interpolation run with autocast off.
- The bank's `1e-3 * mean` floor was a mean over the whole batch (an item's levels depended on its batch-mates by up to
  12 %): now per item.
- The reported vote kappa could flip an entity's type (peaked / girdle) away from the frame fitted for the split: kept
  only where the type agrees.
- `query(t_rel)`: `od_net` now runs only on the evaluated (point, direction) pairs.
- `mass_budget` now uses `evaluate_p2`'s normaliser and prints "iso spurious"; option strings are validated; `--ckpt`
  without `--plot-only` is an error.

Not taken: `entities="lobes"` with `axes="density"` breaks sheets (a girdle splits into ~4 basins whose moments fit as
peaked); the default keeps the votes' frames. A saddle-height merge of basins (join when the boundary is >= 0.8 of the
lower peak) would allow density frames later.

### 6.1b Code review 2: split head, context level, losses (independent agent)

Verified sound: old checkpoints load and behave identically (bit-identical bank); the split targets double-count
crossings exactly as `e_mass` does; the sampler's masks and fallbacks; the autocast wrappers.

Found and fixed:
- **Warm-starting the context level from a 3-level checkpoint put the old mean-feature weights on the new level**
  (`load_partial` grew the first layer column-wise): the start's density moved by up to 2x the mean -- remapped.
- **The maps readout could inflate a lone ripple lobe to all of the oriented mass (12x)**, give a crossing partner's
  share to the remaining lobe when the partner was typed isotropic, and report unexplained oriented mass as isotropic
  objects -- now the scale-up is capped at 2x the density split, isotropic-typed lobes keep their share, and the
  remainder is `ori_unres`. With a perfect split head on the direct-path pilot's density, maps fixed sheets, blobs and
  the mix's blob (C1 1.00) but lowered fans, curves and crossings (their off-axis lobes) -- the lobe split is the limit.
- The objects branch could zero every lobe before the maps branch ran (now exclusive); maps without a background head
  double-counted blobs; a loss loop variable shadowed the lattice-crop helper (latent crash); split targets on
  supersampled phantoms came from the coarse maps (now from the populations' fine window means); the fresh split head
  started at a loss of ~6 (bias now -4); the context level is stored at stride 2.
- The lobe-mass loss is a consistent but weak mass signal (a 2x mass error moves it 0.25-0.34 in log; shape errors move
  it as much): off. The reviewer's alternative, **mass votes**, was implemented.

### 6.1c Code review 3: mass votes, window_mean, trainer (independent agent)

No blockers. Fixed:
- Vote masses were not conserved: two pieces of one population each kept the full vote (scored 2x), and `votes` mode
  had no bound -- pieces of one type within 30 deg now share one vote; votes weighted by the floor-subtracted density;
  supervised over lobe flanks and overlaps too (14 % of a lobe's directions had none); the head starts near the masses.
- `window_mean` ran its matrix products and sampling coordinates in bf16 under autocast (0.06-0.28 voxel off): now fp32;
  its matrices are built on the device (each step's new sigma levels cost a host copy) and without transposes.
- Speed: the four full-resolution map heads now apply their 1x1 convolution before upsampling (exact, ~2.4x faster);
  the bank uses the real half spectrum with Nyquist-symmetrised, cached filters (exact to 4e-7); cudnn benchmark on.
- Trainer: optimiser state in checkpoints and `--resume`; progress lines.

### 6.1d Code review 4: the pre-norm, the divergence guard (independent agent)

- **Why the trunk blows up** (verified on the diverged checkpoint): the filter norms grew only 1.6-3.2x over their
  initialisation; shuffling each filter's 27 taps leaves a block's gain unchanged, shuffling across channels cuts it
  6-400x -- the tap-summed channel-mixing matrices lined up with the smooth part of the features. Zero-mean filters
  (weight standardisation) cut it only ~13x, so normalising activations is the right fix. At initialisation every family
  prefers smaller features; in a trained model the saturating heads (objects softplus, background, votes) pull toward
  larger ones -- a weak, consistent pull that Adam's per-coordinate steps turn into full-size updates.
- **The pre-norm stops the compounding but does not bound what the heads read** (diverged weights with the pre-norm:
  |h| 18 / 320 / 5e3 instead of 1.6 / 1e5 / 4e15) -- now `--block-norm 2` also normalises each block's update (each block
  then adds ~g per voxel; |h| 0.2-2 in a test run). The running pilot (pre-norm only) stayed healthy (|h| ~0.4 / 3-4.6 /
  1.2-1.6 to step 1200), so it was not restarted.
- **Guard bugs, fixed:** the optimiser snapshot was aliased by `load_state_dict` (a second rollback restored diverged
  moments); snapshots were of weights no forward had checked, with none in the first 250 steps; a non-finite gradient
  with a finite loss went through; only three levels were watched (l3 had the largest gains); diverged weights could be
  saved. Now: snapshots of checked weights (from the start), a copy restored, non-finite gradients skipped, all levels
  and the mixed features watched with a relative test (10x their running size), saves only when healthy. Tested: a trip
  rolled back to the start snapshot at half rate and training recovered.
- Also: `--resume` takes the run's own args; CPU workers exit when the trainer is killed; the bank's filter cache has a
  byte budget (a whole-volume bank would have cached GBs).

### 6.2 Generator review (independent agent)

**Verdict: the phantom families pose the question correctly; the scene generator does not yet.** A perfect model output
built from the targets passes C1-C9 on 16/16 selection phantoms through the real readout, so for a perfect density the
readout's masses equal the targets.

Findings (ranked by the reviewer; [run] = checked by code):
1. **Blocker (scene): targets edited by visibility rules measured on an image the network never sees** (computed before
   the batch-time contrast routes, ~95 % of images get a destructive route). Presence visibility removes mass without
   giving it to the background; orientation visibility blends tensors toward isotropic and changes the *type*: 26-32 %
   of line populations' visible mass gets girdle targets, 10-16 % isotropic; sheets 9-18 % isotropic [run].
2. **Major: the readout's mass and type depended on vote-kappa thresholds** -- density lobes 3x broader than the votes
   read crossings at 0.26-0.41x and sheets at 0.51x [run]. Addressed tonight by the mass maps (split head) and the lobe
   readout; typing by the density's own anisotropy is still open.
3. **Major: at Monte Carlo crossing cores the image locally shows mostly the population with the larger |contrast|**
   (per-population contrast U(0.3, 0.8); 81 % of crossings differ by > 0.05) [run] -- the same finding as §4 below.
4. **Major (scene): mass not conserved and window operators disagree**: populations + background > 1.1 of the window in
   6.5-10 % of queries; background replicate-padded at faces but objects zero-padded (+10 % near faces); the Gaussian
   truncated at 3 sigma not renormalised (0.97x) [run]. Phantoms sum to 1 except near faces.
5. **Major: phantom and scene conventions disagree**: sheet dispersion floor (phantom kappa1 8.3, scene 16.7: a
   scene-trained model fails C8 on phantom sheets by construction); scenes split a population into resolved objects,
   phantoms never do; scene blobs 99 % peaked/girdle-typed, phantom blobs isotropic; 47 % of scene crossings under 30°,
   phantoms only 30-90°.
6. The mix family's bundle x interleaved-fan still uses the lattice (crossing-plane normal is the strongest direction at
   its core).
7. Scenes have no near-isotropic cut-off (cell layers typed girdle 31-74 %).
8. **Direction sampling drew isotropic entities' directions around the frame's m-axis** (y/z axes at a third of the
   rate) -- fixed; and the background (61-79 % of the mass) took ~2/3 of the entity-directed samples: new option
   `SAMPLE_ORIENTED` draws them from the oriented entities only.
9. `fine` was excluded from training by default but is 4/32 of the final set (the direct-path pilot never trained on
   it): the overnight runs train on it.

The scene generator's actual mix (Z4/E1 configuration) [run]: lines 64-68 % of object mass, sheets 20-25 %, blobs
11-12 %; by target type peaked 36-44 %, girdle 40-48 %, isotropic 17 %; background 61-68 % of all target mass (the
earlier 59 % sheets estimate does not describe it).

**Generator workplan from the review (not done tonight -- the pilot trains on phantoms):**
1. Scene targets from the geometry (true occupancy shared as the image shares it, the true ODF with one floor);
   visibility / evidence a separate learned output computed after every degradation, never changing the type.
2. One convention across phantoms and scenes: window operator (normalised, same padding), dispersion floor, population
   rule, near-isotropic rule; add equal-contrast, low-angle and kissing crossings; replace the mix family's lattice.
3. Types and masses from the ODF's shape and per-kind mass maps; crossing cores whose second population is not locally
   visible labelled (and scored) as context-only.

## 7. Overnight pilot

**First launch (00:43) diverged.** From scratch at lr 3e-4 with every fix, scored every 2000 steps. By step 1000 the
density losses were where they started and the split / vote heads output zero. In the checkpoint the trunk's features
had blown up: mean |h| 8.5, 6e5 and 1e16 at its three levels. Every head saturated, gradients ~0, but no NaN. The
from-scratch simple-family run of 23:45 (older code) showed the same signature at step 500, and the crossing A/B runs
used the same configuration, so all three were stopped (01:35). A probe (`scratch_stability.py`, 400 steps) of a
famN-like configuration against the pilot's showed both stable to step 400 (|h| ~0.1 / 2-3.7 / 0.5-1, growing), so
the blow-up is intermittent and later. The cause is not found; the earlier from-scratch model (famN) trained at
batch 1 without the direct path.

**Cause found (01:56).** Tracing the diverged checkpoint layer by layer:

| layer | mean \|h\| in -> out |
|---|---|
| l0 | 0.32 -> 12 |
| l1 (2 blocks) | 6.4 -> 2.3e3 -> 8.8e5 |
| l2 | 1.1e6 -> 1.9e9 |
| l3 (2 blocks) | 4.9e9 -> 4.6e12 -> 1.0e16 |

Each residual block amplified 300-400x although its weights stayed small (mean |w| 0.02, max 0.12; gains g ~0.3). The
3^3 convolutions had grown *coherent* (summing their neighbourhood), and nothing normalises the trunk, so the gain
compounds over its eight blocks. Every head then saturates (softplus/sigmoid), gradients vanish, and the run is dead
without a NaN. **Fix: a per-voxel RMS pre-norm in each block** (`--block-norm`, `model.ChannelRMSNorm`: each voxel's
features over their channels' RMS, a learned scale -- local, so the same at any crop or tile size, unlike a group
norm). Growth through a block is then additive, not multiplicative. Old checkpoints load unchanged (off by default).

**Second launch (01:55, stopped after 2 minutes for the third):**
- same model, at lr 2e-4 with a 300-step warm-up and gradient clip 2;
- **a divergence guard**: if the trunk's mean |feature| exceeds 50 or the loss is non-finite, it rolls back to the last
  healthy snapshot (every 250 steps) and halves the rate;
- 10000 steps, sized to finish with time to analyse.

The CPU phantom workers did not keep up (fine-crossing phantoms take 7-30 s each on the CPU), so phantoms stay on
the GPU. A step is ~0.8 s.

**Third launch (01:57): with the block pre-norm**, lr 3e-4 (300-step warm-up, clip 2), the guard kept as a safety net;
10000 steps, scored on the selection set every 2000.

Healthy: |h| ~0.3-0.9 / 3-4.6 / 1-1.6 throughout, no guard trips, 0.7-0.85 s/step alone. Held-out losses at step 1000:
density cross 0.34 (start 0.69), fine 0.34, curve 0.15, sheet 0.25, blob 0.02; oriented-mass map 0.13-0.35; mass votes
0.19-0.34. At step 2000: density cross 0.29, curve 0.135, sheet 0.24; oriented-mass map 0.085-0.23.

**Low training (step 2000), selection set -- where it stands:** 3/16 (blobs and empties). **The maps learn first, the
orientations later.** The oriented-material map already holds the populations' mass (training loss 0.09-0.23), but the
density is still nearly flat: read by the density alone, 83-121 % of every oriented phantom's mass is "isotropic".
Read by the maps, the mass is there but in entities whose axes do not yet match ("other": curves 0.58, mixes 0.40).
C9 found 4/16, C6 direction passes only vacuously where nothing matched. The same flat start was seen in the framing
review (the density's per-direction loss is flat-friendly early on).

**But the density already carries the orientations at step 2000 -- it is the vote heads that lag.** Reading each
entity's frame and type from the density's own moment (with lobes joined across high saddles so a sheet's ridge stays
one entity) instead of from the votes, on the same step-2000 outputs:

| step 2000, selection set | frames from the votes | frames from the density |
|---|---|---|
| crossings: matched mass | 0.20 | **0.80** |
| curves: matched mass | 0.01 | **0.83** |
| sheets: matched mass | 0.00 | **1.06** |
| C3 count | 5 | **13** |
| C6 direction | 11 | **15** |
| C9 found | 4 | **8** |
| C1 mass | 3 | **5** |

So at low training the structure learns where (maps) and which way (density) first; the per-direction votes (frames,
concentrations) come last. On the long-trained direct-path pilot the density frames were neutral to better (count
10 -> 12, direction 12 -> 16 of 16), so they are now the readout's default (`axes="density"`, `saddle=0.8`); the votes
still supply the reported concentrations (ODF shape). Fans lose some mass to "other" with density frames (their spread
lobe fits a broader, sometimes girdle-typed Bingham).

**A readout bug this showed:** the maps' 2x cap (review 2) starved weak lobes -- oriented totals 0.03-0.12 of the target
at step 2000 -- now off (`maps_cap`, default None): 0.13-0.70. The pilot's own mid-run scorings use the capped
readout; its checkpoints are kept (`pilot3_step*.pt`) and rescored with the final readout below.

(results below)
