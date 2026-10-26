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
| phantom families, final set (4 per family, never used for selection) | fam_C 5/28: 4 empty + 1 blob | first structured pass (a non-fibrous blob phantom); 0/24 fibre / sheet / mix |
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
features), not the readout. fam_F (longer, batch 4, crossings x3) tests duration: cross density 0.172 -> 0.167 -> 0.168 at 500 / 1000 steps --
flat: more of the same training does not resolve crossings in arbitrary orientation; a structural change (wider
stride-1/2 trunk, rotation-aware features) is the next thing to try there.

### Generator
Z1 (all fixes, from EP14b) and Z2 (from the phantom-family model) both 0/9 at 500 / 1000 / 2000 steps; training
density loss ~0.55-0.69 on generator crops; model mass along A / B ~2-6 % of the targets even on single arms, the rest
isotropic floor: a detection failure, not (yet) a separation failure -- the density stays nearly flat on generator
images. The phantom-family initialisation does not carry over (Z2 starts worse and converges to the same).

Is the orientation in the generator images at all? A classical structure tensor (gradients, sigma 2.5) on single-
population arms of the held-out generator crossings finds the target axis on only 3 of 10 arms (1, 10, 13 deg; the
others 30-71 deg; median 47 deg, random would be ~60) -- on the phantom crossings it finds it at 3 deg median. So in
most generator populations the orientation the targets demand is not visible as local structure in the image (the
targets' visibility weighting measures contrast, not orientation); the model's flat densities may be the right hedge.
Caveat: a local tensor can miss orientation carried only by larger-scale shape.

And the targets on those same arms demand near-perfect orientation: mass-weighted median target kappa1 = 600 (the cap)
on 6 of 10 arms and 40-221 on 2 more -- including arms where the structure tensor is 60-71 deg off (seeds 6100013 A,
6100043 B, 6100063 A). The orientation-visibility blending toward isotropic (entities.entity_targets, ``ovis``) is not
firing there: in `generator.py` (`nuisance`, ~line 1320) `ovis` is the same contrast t-statistic as the presence
visibility (`visibility(img, regs, ...)`: the high-passed image regressed on the layers' occupancy), only with a higher
threshold range (`ORIENTATION_T = (6, 18)`). It measures how strongly a population's *region* explains the image, not
whether its *oriented internal structure* does -- a high-contrast region filled with orientation-free texture gets
ovis = 1 and a full-kappa target. Fix: the t-statistic of the image on each layer's fibre-scale texture (occupancy
minus its own low-pass: the part that carries orientation), regressed jointly with the regions. Also the entity
tensors carry no dispersion (straight objects give kappa -> cap). The generator's targets ask for delta-like
lobes the image does not show -- ill-posed, and consistent with the model's flat, hedged densities. This is the first
thing to fix for complex inputs: (1) orientation visibility from oriented-structure evidence (e.g. a structure-tensor
or directional-energy coherence of the rendered image per population), (2) realistic dispersion in the entity tensors
(kappa within what the image and the density basis can express), (3) the structure-tensor baseline as a standing
check that each generated target orientation is inferable -- before more generator training.

## 5. Open structural questions
1. Crossings across orientations: the model resolves one rotated crossing quickly but not crossings in general. Options:
   (a) much longer training / larger batches (fam_F is the first test); (b) a wider full-resolution trunk (32 / 64
   channels at strides 1 / 2 carry the fibre-scale structure; crossings need many oriented filters); (c) rotation-aware
   features beyond the probe (steerable / equivariant layers at strides 1-2); (d) test-time rotation averaging.
2. The generator: the density does not form lobes there at all (unlike phantoms). The targets are the first suspect:
   in 7 of 10 crossing-population arms the target orientation is not visible locally, yet the targets ask for kappa up
   to 600 there. Fix the targets (orientation-evidence visibility, realistic dispersion) and keep the structure-tensor
   inferability check, then reassess capacity and duration.
3. Non-fibrous objects: the background head separates blobs from the background by place; level and edges are not yet
   right (C1, C5), and in the generator cells / foams own no directions under OWN_OBJECTS (no tree identity).
4. Fibre super-resolution: not learned (smooth sub-voxel maps; unresolved fibres 0/6); needs unresolved-fibre training
   data and sub-voxel supervision (fam_E/F add both).
5. Measurement: every criterion now has a ground-truth check; the remaining weakness is the small held-out sets
   (2 per family; a 4-per-family final set exists and has not been used for selection).

### First attempt at the target fix (`generator.ORIENT_SINGLE_SCALE`, off by default)
`ORIENTATION_T` was calibrated on single-scale evidence; the high-pass scale ladder added for crossing cores inflated
the shared t, so `ovis` saturated. With orientation visibility from the single scale (window ladder kept), 2 of the 5
checked crossings now get near-isotropic targets where the structure tensor fails (seed 6100063: kappa 600 / 6.9 ->
0.8 / 0.8; seed 6100074: 4.5 / 40 -> 0.7 / 3.1), but 3 arms still demand kappa 144-600 where the tensor is 60-71 deg
off (seeds 6100013 A/B, 6100043 B). Not paired exactly: the extra visibility call shifts the generator's random draws
downstream (a few structure-tensor errors moved, e.g. 51 -> 29 deg). So: a step, not the fix -- orientation evidence
from the populations' oriented texture is still needed (section 4).

### The target fix: orientation visibility from oriented texture (`generator.ORIENT_FROM_TEXTURE`, train.py `--orient-from-texture`)
Each shape layer's fibre-scale texture (occupancy less its blur at 2x its radius) regressed on the image at the single
scale (window ladder kept) gives the orientation visibility. On the same 5 crossings the targets now follow the
image's orientation evidence: arms where the structure tensor finds the axis (1, 10, 13 deg) keep sharp targets
(kappa 600 / 600 / 82); arms where it does not become weak or isotropic (71 deg: 600 -> 5.8; 72 / 59 deg: isotropic;
29 / 31 deg: 0.7 / 2.1); two arms remain moderately oriented (60 deg: 16, 70 deg: 23; were 144-600). Z3 (from Z2's
step-2750 weights, all fixes, bg head, --orient-from-texture) started 06:05 ET; its generator scores at 500 / 1000 /
2000 steps land after 07:00 in `.../pilot_entities/gensep/Z3_texture_targets_s*_L06_acceptance.png`. Caveat: the
generator criteria assume oriented A / B populations; with orientation-faithful targets some crossings may become
unscoreable by the ground-truth check (correctly: their orientation is not in the image).

Tests after tonight's changes: entity readout 18/18, dataflow 8/8, scene generator 43/43 (the new generator flags
default off).

### Z3 after 07:00 (addendum, 08:15 ET)
Scored against the old targets (the scoring script's default generator settings) Z3 was 0/9 at 500 / 1000 / 2000 --
an inconsistent test; `generator_separation.py --orient-from-texture` now scores against the targets it was trained on.
At step ~2000 on those: 0/7 scoreable (5 of 12 unscoreable: with orientation-faithful targets more crossings have
populations that are not oriented in the image, which these criteria assume), but the first generator pilot to find
oriented mass: seed 6100064 A 127 of 175, seed 6100093 A 131 of 663 (Z1 / Z2: 0-37 on the same arms); B mostly missed;
total mass ~1.03 per point, most read as isotropic. Training density loss 0.32 vs 0.56-0.69 for Z1 / Z2: with
inferable targets the density shape is learned. Direction confirmed; not yet a pass.

## 6. What is still running at 07:00
- Z3 (generator pilot on orientation-faithful targets, from Z2 step 2750) to step 4000, scored at 500 / 1000 / 2000.
  Z2 was stopped at step 2750 (0/9 at 500 / 1000 / 2000; trained on the ill-posed targets).
- fam_F (families, crossings x3, batch 4) to step 2000: `.../simple/fam_F.png` + table when done.

## 7. Suggested order of work
1. Generator targets: orientation from oriented texture (done, flag) -- evaluate Z3; add dispersion to the entity tensors;
   keep the structure-tensor inferability check per case.
2. Crossings across orientations on phantoms: longer / larger-batch training did not move them (fam_F flat) -> a
   wider stride-1/2 trunk and rotation-aware layers; a proper rotation-generalisation test (held-out orientations,
   the final set).
3. Mass calibration (C1) and leakage (C4) on phantoms: they fail broadly even where direction is right.
4. Super-resolution: phantoms rendered on a finer grid and averaged down, with fine-grid targets.
