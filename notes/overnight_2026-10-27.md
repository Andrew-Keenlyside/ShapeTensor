# Overnight 2026-10-26/27 — separation of every layer: scoping review (running)

Directive (2026-10-26 ~22:45 ET): iterate on the design until all layers can be separated and oriented correctly, with
object- and population-specific ODFs, in the most complex inputs; adversarial review after each result; expand the
validation phantoms (fibres, blobs, planes, fanning, crossing; with and without orientation); test how inference
depends on output resolution; investigate fibre super-resolution. To 07:00 ET.

Results directories: `/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/pilot_entities/simple/` (phantoms),
`.../pilot_entities/gensep/` (generator crossings), `.../pilot_entities/Z1_fixes`, `.../Z2_from_families` (pilots).

## 1. Status against the goal (honest)

Not reached. Every criterion is defined and fixed in advance (`src/shapetensor/learn/crossing_acceptance.py`); a
result passes only when all criteria hold on every held-out case, and every measurement is first checked with the
ground truth fed in as the model (cases the truth fails are dropped and listed).

| test | best so far | notes |
|---|---|---|
| simple crossing, axis-aligned, angles 30-90 | 12/12 PASS | fits that family only: 0/4 on rotated crossings (review) |
| simple crossing, random 3-D orientation | 0/4 | runs superseded by the families (which include orientation) |
| phantom families (cross, fan, curve, sheet, blob, mix, empty) | 2/14 (fam_C, fam_D) -- only the two empty volumes | fam_A 1/14, fam_E 0/14; probe (fam_D) no measurable change |
| generator crossings (difficulty 0.6), 9 scoreable of 12 | 0/9 (EP14b; Z1 at 500/1000/2000; Z2 at 500/1000/2000) | |

No phantom with structure passes in any model: every pass is an empty volume (where "other directions" sits near
the 5 % limit). All family scores are on the selection set (reused for every choice); the final set is scored for
fam_C below. Lower density loss did not track acceptance (fam_E: lowest loss, 0/14; fine-tuning raised C3 failures from
2/12 to 7/12 over fam_A and added spurious isotropic mass). Per family (fam_C): sheets pass direction and ODF shape where
found (C6, C8) and are found (C9); blobs are found as isotropic objects but miss level / edges; curves and fans are
found with the right direction but miss level, counts and ODF shape; crossings fail almost everything. C1 (level)
fails on 10-12 of the 12 structured phantoms and C4 (leakage) on 8-12, in every model: mass calibration and leakage
are failures as broad as the crossings (review 3).

## 2. What was built tonight
- `learn/phantom2.py` — composable phantoms: straight / curved / fanning bundles of fibres (per-voxel tangents), sheet
  stacks (girdle ODFs), blobs (non-fibrous, isotropic), empty volumes, mixtures; random 3-D orientation and look
  (bright / dark, background, noise, blur). Targets: per population the window mass and the Bingham of the
  window-averaged moment (fans / curves broaden with the window: population-specific ODFs); the background.
- Background head (`PopulationNet(bg_head=True)`, `bg_t`): the density's isotropic floor is background **plus**
  isotropic objects; the head tells them apart by place (supervised on window means at the queries).
- Criteria C1-C9 for any population mix (`evaluate_p2`), the generator version (`evaluate_generic`), ground-truth checks.
- `scripts/learn/phantom_ladder.py`, `generator_separation.py`, `resolution_study.py`.

## 3. Adversarial reviews and what changed
Review 1 (crossing ladder, generator criteria): the angle-rung pass is fit-to-family (held-out phantoms near
duplicates; test set reused for selection) — reported as such, a separate final set added; the generator criteria
failed on the ground truth (C3 counted entities; generator populations are 2-4 resolved entities) — now populations,
plus the ground-truth filter; one isotropy threshold (readout = training); whole-plane level in C1.
Review 2 (families): a perfect model failed blobs/mixes — absent populations typed isotropic, blob tails charged as
leakage, empty phantoms normalised differently, C8 asked for an unobservable dispersion (fixed now), splits/misses
counted three times (C6/C8 on matched points + C9 match rate), background head biased high (now window-mean loss),
colliding crossing layers. After the fixes the ground truth passes 42/42.

## 4. Findings so far
- The model learns clean structures fast: families' held-out density loss fell to 0.20 (cross) / 0.13 (fan) /
  0.07 (curve) / 0.04 (sheet) / 0.01 (blob) / 0.002 (empty) in 6000 steps; blobs, sheets and curves are found.
- Crossings in arbitrary orientation still lose the second population in the overlap (the angle rung's failure mode
  in 3-D); the oriented probe is being tested (fam_D vs fam_C).
- The generator remains 0/9: the density shape barely learns there (Z1 val density 0.459 -> 0.436 in 500 steps).
- Isotropic / weakly oriented objects are 1-14 % of generator object mass (difficulty 0.3-1.0): the floor readout
  would call them background; the background head is the fix (blob phantoms now found).

### Output resolution and super-resolution (fam_A, `resolution/fam_A.png`)
- Output scale sigma 1 / 1.5 / 2.5 / 4 / 6 (fam_A was trained on sigma 1-6; 2.5 is the scale the thresholds were set
  at): the direction is right where a population is found (C6 12/14 at every scale; but C9, finding it, 7/14 at every
  scale); the population count (C3) is best at sigma 2.5 (12/14; 6/14 at sigma 1, 8/14 at sigma 6).
- Sub-voxel output: not a real super-resolution test yet -- the phantom's occupancy and targets live on the 64^3 grid
  (no sub-voxel truth), the test fibres were resolved (spacing 3.9), and the crossing used is one the model merges
  (pop 2's map edges are the merge). The model's half-voxel maps are smooth where the sigma-1 target shows fibres.
- Unresolved fibres (spacing 1.6-2.4, radius 0.4-0.7): fam_A 0/6, but fam_A never saw that family and the images are
  point-sampled near the 2-voxel Nyquist limit (they may alias); a classical structure-tensor baseline is needed to
  say whether the orientation is in the image. fam_E/F train on the family; their scores on it are not yet measured.
- A proper super-resolution test needs phantoms rendered at a finer grid and averaged down (true partial volume),
  with targets from the fine grid.

### Crossings in arbitrary orientation: the dominant failure
At crossing centres (random 3-D orientation) the above-floor density within 12 deg of A / B is 0.04-0.15 for the
models against 0.24-0.28 in the targets at every angle -- at 90 deg only 0.06 / 0.04 (about 3x uniform; the target
~10x): the density at the core is broad. The readout then merges (42 deg row: one entity on the bisector carries 94 %
of the target mass) or fragments (85 deg row: ~4 entities, the heaviest 39 %, both populations at 0.3-0.4x) (review 3). The oriented probe (fam_D) and crossings x3 + sub-voxel windows
(fam_E) move this by <= 0.01. With the earlier result -- a single rotated crossing is fitted in ~100 steps -- the
architecture can represent it; what is missing is generalisation across orientations (data / duration / rotation-aware
features), not the readout. fam_F (longer, batch 4, crossings x3) tests duration.

### Generator
Z1 (all fixes, from EP14b) and Z2 (from the phantom-family model) both 0/9 at 500 / 1000 / 2000 steps; training
density loss ~0.55-0.69 on generator crops; model mass along A / B ~2-6 % of the targets even on single arms, the rest
isotropic floor: a detection failure, not (yet) a separation failure -- the density stays nearly flat on generator
images. The phantom-family initialisation does not carry over (Z2 starts worse and converges to the same).

## 5. Open structural questions
1. Crossings across orientations: the model resolves one rotated crossing quickly but not crossings in general. Options:
   (a) much longer training / larger batches (fam_F is the first test); (b) a wider full-resolution trunk (32 / 64
   channels at strides 1 / 2 carry the fibre-scale structure; crossings need many oriented filters); (c) rotation-aware
   features beyond the probe (steerable / equivariant layers at strides 1-2); (d) test-time rotation averaging.
2. The generator: the density does not form lobes there at all (unlike phantoms). Candidates: appearance variety
   (contrast routes, textures) needing far more training (cluster scale); targets whose orientation is not visible in
   the image despite the visibility weighting (a per-case check that the target ODF is inferable would separate the
   two); capacity.
3. Non-fibrous objects: the background head separates blobs from the background by place; level and edges are not yet
   right (C1, C5), and in the generator cells / foams own no directions under OWN_OBJECTS (no tree identity).
4. Fibre super-resolution: not learned (smooth sub-voxel maps; unresolved fibres 0/6); needs unresolved-fibre training
   data and sub-voxel supervision (fam_E/F add both).
5. Measurement: every criterion now has a ground-truth check; the remaining weakness is the small held-out sets
   (2 per family; a 4-per-family final set exists and has not been used for selection).
