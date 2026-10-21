
## 27. The release review, pass 1: what the restructured pipeline taught (2026-10-17)

Measured on the three phantoms (`shapetensor.phantoms`), 192³ blocks, calibration on twelve bundles at 15° steps,
σ 1, ρ 10; the debug scripts lived in the session scratchpad and their numbers are quoted here.

- **A joint unmixing of many signatures is degenerate.** With six candidate signatures the NNLS coefficients rose
  to 30–50 (against 1–7 with four) and every layer's axis collapsed onto the bisector (49° / 48° from the two
  populations of a 90° crossing). The orientation code of the features has rank five (§26), so four signatures
  plus the background are the limit. Hence the two-pass confirmation: each candidate unmixed alone (always well
  conditioned) → distinct axes; then the kept axes unmixed together (twice) → distinct axes.
- **The readers respond unequally to orientation.** On a crossing of x against z fibres the readout density of
  the x population was 1.0 and of the z population 0.54, their single-signature activation masses 0.74 and 0.29;
  the density peaks' heights ranked three x shoulders above the z population. Per-direction gains measured on the
  calibration block (density over the teacher's kernel histogram: 0.95–1.56 at 8° dispersion, 0.94–1.67 at 20°;
  activation on the own signature: 0.82–1.20 and 0.76–1.23) normalise the candidates and the weights.
- **The per-voxel layers keep an orientation dependence the gains do not reach.** With the same code the
  multi-vector field on a crossing symmetric about the diagonal (135° / 45°) reached a pair error of 9–10° with
  both populations within 15° at 87–96% of voxels (90° and 60°), while on the x–z crossing it reached 16° and
  20%. Averaging the layers over the grid symmetries of the block (z–x swap and x flip, or the six axis orderings)
  did not help (90° at the diagonal: 87% → 18–37%), because the transformed frames are not better estimates and
  the average only blurs; the six orderings also take the phantom's fibres out of the plane the dictionary was
  calibrated on. Dropped.
- **The activation floor.** None in the confirmation; 1% of the null in every population tensor (a population
  absent from a voxel falls back to the excised tensor instead of 0 / 0); 10% of the null in the merge (a voxel
  with every activation at zero gave an all-zero tensor whose eigenvector is the x axis — large uniform red
  regions in an early real-block DEC). Larger floors in the population tensor (5–30% of the null) do not improve
  the single field on the three-band interface (7.2° → 6.8° far from the step) and cost the 60° crossing
  (both within 15°: 67% → 46%), so the single-population noise there is the one population found for three
  bands, not the floor.
- **Sub-window candidates** (64³) with the two-pass confirmation find a second band on the interface (N 2, far
  error 7.2° → 5.4°) and merge a dispersed bundle's spurious second population (20°: N 2 → 1), at the cost of one
  60° crossing at one orientation. Kept.
- **The NNLS warm start** by `torch.linalg.lstsq` on the GPU took 55 s per 192³ block; the normal-equation
  solution is identical to 1e-6 and takes 0.04 s.
- **On the real block** (S-21-46, 192³, calibrated on 384³) the pipeline finds two populations and its single
  field agrees with the excised structure tensor to 8° over the block and 6° where the tensor is confident
  (95% within 15°).

## 28. The figures and the parameters, after the review (2026-10-17)

- **One figure layout everywhere**: image | excised structure tensor | shape space (the voxels' standardised
  features on two principal axes, coloured by the population whose activation is largest above the null) | one
  activation layer per population | the shape tensor's single field | the first vector of the multi-vector field
  (`viz.pipeline_row`). The validation, the HiP-CT block and BigBrain use it; the illustration's partner on the
  HiP-CT block (`figures.py overview`) mirrors the illustration's six columns instead.
- **The coherent texture setting is no longer the default validation.** It was introduced when detection failed
  at the calibrated texture, to show stage 2 working on tight bundles (8° dispersion); it is not the regime of the
  HiP-CT crop and its results (per-voxel separation on diagonal crossings, see §27) overstated what the data
  allows. The validation is now the calibrated texture only, on bundle 60°, crossing 60° and interface 40 (the
  90° crossing dropped); `--settings coherent` keeps the setting runnable.
- **The first vector of the multi-vector field is the useful crossing output at the calibrated texture**: 11.5°
  to the nearest population (58% within 15°) against 23° for the single field and 24° for the structure tensor;
  the pair is still unresolved (both within 15° at 0%).
- **Parameter sweep on the HiP-CT block** (`scripts/parameters.py`, `results/parameters.json`): the field's
  direction is set by the scales and by which populations are found; `tensor_floor` (0.3° between 0 and 0.1),
  `tissue` (2.7°) and `window` 64 vs 96 (0°) leave it; `nmax` 2, `min_sep_deg` 15 and `window` 0 all find the
  same alternative pair, 8.1° from the default's; `iterations` 4 and `min_sep_deg` 40 merge to one population;
  `smooth` 1 loses the second population (the null drops with the spiky activations); `sigma` 0.5 changes the
  teacher and the field (23°); `rho` 20 gives one population and a null four times larger.
- **Run-to-run variation**: the same block, calibration and settings gave 8.3° / 6.3° (§27) and 11.8° / 7.2°
  (block / where the tensor is confident) agreement with the excised tensor on two days; the readout's fit on the
  GPU is not bit-reproducible and the population pair found can differ. Worth a seeded, deterministic fit before
  release.

## 29. HiP-CT across voxel sizes, four regions, a whole brain (2026-10-17)

- **Six voxel sizes, one 192³ block each, σ 1 / ρ 10 voxels throughout** (`scripts/scales.py`, `results/scales.json`):
  I74 at 2.2 µm (VOI 2) and 10 µm (chunk), S-21-46 deep brain at 20 µm (the 20 µm scan) and 50 / 100 / 200 µm
  (levels 0–2 of the 50 µm scan, the same thalamic site). Agreement with the excised tensor, median over the
  tissue / where the tensor is confident: 2.2 µm 23° / 20° (3 populations, one of them a vessel); 10 µm 25° / 16°
  (3); 20 µm 6.2° / 4.8° (2); 50 µm 5.9° / 3.2° (2; the layers follow the visible fascicles); 100 µm 11.8° / 4.6°
  (2); 200 µm 9.9° / 5.7° (3 populations that are tissue classes: a white-matter band, the ventricular edge, the
  rest). The pipeline reads shape at its receptive field (16 voxels): at sub-cellular voxel sizes that is cells,
  vessels and single axons, and the voxel size decides what a "fibre" is.
- **Four regions at 20 µm (S-21-46)**, each calibrated on its own 384³: deep brain 2 populations, 6.2° from the
  tensor; gyral white matter 3, 12.7°; cerebellar white matter 2, 25.6° (tensor gap 0.08: little oriented
  texture); **brainstem 1 population, 2.4° — but the self-calibration is degenerate**: one fibre direction, 7 of 24
  teacher bins, null 2.8 (seven times the deep brain's), 6% of tissue above it. Self-calibration needs orientation
  diversity in the calibration block.
- **A calibration does not travel** (the regions rerun with the deep-brain block's calibration applied to the other
  three): cortex 13.3° → 33.1° from the tensor (tissue above the null 84% → 44%), brainstem 2.5° → 14.6° (33% → 14%),
  cerebellum 25.5° → 22.6° (81% → 33%). The readers are specific to the local feature statistics and signatures. So
  the whole-volume rule is per-chunk self-calibration with the volume calibration as fallback below 12 of 24
  teacher bins (`run_volume(calibrate_chunks=True, min_bins=12)`); the single-calibration whole-brain run (one
  256³ at the centre) is superseded.
- **Whole brain at 100 µm, single calibration** (for the record): 121 tissue chunks of 256³, 26 s each after the
  speed work (130 s before), 51 min for the shape pass and 5 min for the structure tensor; agreement with the
  excised tensor on the three mid planes 8.7–9.8° (4.1–4.8° where the tensor is confident). The unmixing of a
  masked volume's air gave large "activations" (a constant block standardised is an outlier vector): the driver
  now zeroes the weights where the volume has no data.

## 30. Speed review (2026-10-17)

Profiled on the HiP-CT block (192³ stages 2–4; 384³ calibration), synchronised timers on every hot function:

- Before: 12.8 s per block, of which **7.8 s in cusolver's batched `eigh`** (14 per-voxel eigensolutions of 0.57 s),
  2.5 s in the NNLS (108 small slabs, 60 projected-gradient steps each), 1 s in all Gaussian filtering, 0.6 s in
  the network. Calibration 17 s: features 5 s, the readout's 2000 Adam steps 3.5 s, three averaged-feature passes
  and CPU multinomial sampling of 56M weights 2 s, the teacher 1.5 s.
- **Closed-form symmetric 3 × 3 eigensolver** (`st._eig3_sym`, float64 internally): within 0.02° of `eigh` on random
  SPD and tensor-like matrices with a 1e6 dynamic range, 0.05 s on 7M voxels (11×), 0.46 s on 60M.
- **NNLS by cyclic coordinate descent** from the normal-equation warm start: the old 60 projected-gradient steps
  were *not converged* on correlated 5-column problems (max error 17% of the mean activation, 99.9th percentile 4%);
  25 coordinate-descent sweeps reach 1e-3 in a third of the time, 12 sweeps for N ≤ 3 in a tenth.
- One averaged-feature pass per block for every reader (candidates, confirmation, final unmixing); the block's
  gradient products once for every population tensor; the sub-window densities' sum as the block's density; GPU
  sampling (Gumbel-top-k) of the calibration voxels; slabs sized by voxels, not planes.
- The driver: the network sees the full halo (reach + engine margin), stages 2–4 only the core + the filters' reach
  (344³ instead of 392³ for a 256³ core).
- After: **2.7 s per 192³ block** (4.7×), calibration 10 s (features 4.9, readout fit 2.8, teacher 1.5). Validation
  unchanged within noise (crossing 2/2/2 populations, first vector 10.5° / 58%, interface 16.8° at the step).
  The remaining cost is the Gaussian integration of 12–13 population tensors (0.9 s) and the network.

## 31. Seams, and plain DEC (2026-10-17)

- The 100 µm whole brain showed chunk seams in the single field: each chunk's population set (and, with per-chunk
  calibration, its readers) differs from its neighbour's, so the merged tensor jumps at the face although every
  voxel's tensors are exact. Fix in `run_volume`: chunks overlap by `blend` (32) voxels per face, each chunk's
  unit-trace merged tensor and its weight are accumulated with linear ramps summing to 1 across the seam
  (`merge.merged_tensor` is linear in the activations), and the field is the eigensolution of the blended tensor;
  the network sees overlap + reach + margin and stages 2–4 core + overlap + reach, so the blended tensors are exact.
  Cost: 408³ instead of 344³ per 256³ core (1.65× the stage 2–4 work) and a read-add-write of the accumulators.
  Test: a bundle in eight chunks, the blended field within 5° of one pass and no step at the faces.
- The multi-vector field stays per chunk (no blending of population sets); documented.
- Display: every direction field as plain DEC at every voxel, the shape tensor's exactly as the structure tensor's;
  the activation shown only as its own layers (the renderings page keeps the alternatives).
- **Whole brain at 100 µm, blended and per-chunk calibrated** (`/hdd/andrew/shapetensor_runs/S-21-46_100um_chunkcal`):
  121 tissue chunks, 114 own-calibrated, 7 on the volume calibration (4 of them because their own fit failed on
  too little oriented tissue — now caught and logged), 0 failed; median 62 s per chunk (26 s of stages 2–4 on the
  408³ block + calibration + accumulator I/O), 2 h 3 min for the shape pass, 6 min for the blended write, 4.5 min
  for the structure tensor. Agreement with the excised tensor on the mid planes 7.9 / 8.5 / 8.2° (4.3 / 4.5 / 3.5°
  where confident). No seams visible. The single-calibration run (9.4 + 7.8 GB) deleted.

## 32. The dev pipeline, validated and debugged (2026-10-21)

Branch `dev-pipeline`. The dev pipeline as formalised (anatomix-dev-vit + sliceTCA, fibre sets, directions from the
features' gradients) on the validation phantoms (calibrated texture, seed 0) and a real slab, then debugged stage by
stage. Results under `/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/`.

**Stage 2 found nothing.** Calibrated on the straight bundles (the prototypes' protocol), the fibre sets found 0 of
441 bend windows: the fibre-free texture reference's window prominence (24.0) was above every fibre window's (median
11.9). `diag_presence.py`: with dev-vit the readout gives texture confident directions, 83% of texture windows peak
within 10° of a coordinate axis (chance 4%; the 8-voxel patch grid is 17% of the texture features' variance, 0.1% for
encoder 2; the half-patch shift average leaves its period-4 and period-2 parts), and a texture window's prominence
(median 12.9) exceeds a bundle's (8.7). sliceTCA removes most of the grid (2%) but not the confidence. Encoder 2's
weak readout (fit 0.40) is diffuse on texture, which is why prominence worked there. The released per-window mask
share does not separate either (texture median 0.81 with dev-vit).
- The features do carry it: a linear probe on dev-vit features, bundles against texture, has voxel AUC 0.999 (bend
  fascicle vs rest 0.92; encoder 2 0.72-0.79) (`diag_probe.py`).
- Tried in turn: a *no orientation* readout output (takes crossings for texture: 26/169 windows of a full crossing);
  the linear probe as presence (`PresenceProbe`; crossings 12-28% of windows); a Gaussian model of fibre-free texture,
  presence = the window's mean Mahalanobis distance above the largest of the held-out texture's windows
  (`TextureModel`, `presence="distance"`): crossings 59-63% of windows, bend 280/441, band crossings 341-386/441.
  The texture's padded faces must be left out of the model (with them, every distance shrank and crossings fell
  below the null). Default: distance.

**Stage 3 from the features' gradients follows the patch grid.** With the true fascicles as the layers
(`diag_stage3.py`, bend): image structure tensor 3.0-4.4° (p90 5-9°); shape tensor on encoder 2 4.7-7.1°; on dev-vit
9-32° median with p90 ~89° (in-plane flips; not the slab-in-y ambiguity: |y| 0.02). New default `direction="readout"`:
the readout's distributions as dyads `sum_u P_u u u^T`, integrated with the layer (gradient-free). Bundle-calibrated:
encoder 2 2.5-3.7° but dev-vit 14-46°; **self-calibrated on the bend: dev-vit 2.2-3.7° (p90 5.5-11°)**, encoder 2
6.3-8.6°. dev-vit's readers do not carry over from straight bundles to other geometry and tissue (readout fit 0.97 on
the bundles notwithstanding); they must be calibrated in-domain, as `run_volume` calibrates chunks. A case with one
orientation (the straight control) is a degenerate self-calibration, so the validation now has `CALIBRATION=pool`:
one calibration on the 12 bundles + all six cases (label-free, every orientation, in-domain).

**Validation, pooled calibration** (`validate_dev.py`; errors in degrees; bend: way in / turn / way out over the
three fascicles, populations carrying them; bands: each band's largest share, error outside the overlap):

| method | bend | straight | band 90 | band 60 | crossing 90 / 60 merged |
| --- | --- | --- | --- | --- | --- |
| released | 61-73 / 30-36 / 1-2, 1 pop | 7.1-7.6, 2 pops | 0.76/0.74, 15/29 | 0.52/0.73, 12/42 | 29.2 / 27.5 |
| fibre sets, enc2 | 2.4-3.7 / 1.7-3.6 / 1.1-1.5, 1 | 1.2-2.4, 1 | 0.88/0.84, 6/7 | 0.89/0.97, 5/9 | 18.0 / 25.1 |
| dev-vit, probe, readout | 0.9-4.1 / 2.8-5.5 / 1.8-3.4, 1 | 0.5-1.6, 1 | 0.86/0.90, 3/7 | 0.96/0.87, 3/6 | 14.3 / 26.3 |
| dev-vit, distance, readout | 1.3-4.0 / 2.9-5.1 / 1.8-2.1, 1 | 0.4-2.1, 1 | 0.89/0.85, 3/9 | 0.92/0.84, 4/6 | 12.7 / 26.1 |
| dev-vit + sliceTCA, probe, readout | 0.9-3.2 / 2.2-4.0 / 1.5-2.5, 1 | 0.7-0.9, 1 | 0.91/0.89, 3/7 | 0.90/0.93, 3/3 | 11.5 / 25.8 |

(excised structure tensor on the crossings: 18.5 / 23.9.) One population per fascicle through the bends, the two bands
whole and apart, the lowest errors outside crossings. sliceTCA helps a little at 5.5x the calibration time (700 vs
126 s). Not solved by any method: inside an overlap or a crossing the two directions are not resolved (overlap pair
error 23-36°, both within 15° ~0); the windows rarely yield a second population (dev-vit's second-share null is 0:
single bundles never show one).

**Artefacts** (`shapetensor.artefacts`, `scripts/artefacts.py`, `artefacts_real.py`): streaks (lines in z slices,
1 voxel thick, ~0.5 per slice, bright or dark) along x (in the fibre plane), along y, at any direction; rings about an
axis half a block beyond x = 0 (sheets along z). Strength in grey levels; the fibre darkening is 50. Bundle at 60°,
merged median error at strengths 0 / 25 / 50 / 100 / 200, readers calibrated clean:

| | streaks along x | streaks along y | streaks, any | rings |
| --- | --- | --- | --- | --- |
| structure tensor | 4.0 / 4.0 / 6.7 / 28.7 / 47.6 | 4.0 / 4.1 / 4.9 / 10.7 / 86.5 | 4.0 / 4.0 / 4.7 / 12.8 / 37.6 | 4.0 / 8.2 / 13.8 / 17.1 / 13.5 |
| released | 2.4 / 2.5 / 2.6 / 27.4 / 55.3 | 2.4 / 2.5 / 2.7 / 4.4 / 20.4 | 2.4 / 2.4 / 2.9 / 15.9 / 57.3 | 2.4 / 8.2 / 18.5 / 21.3 / 19.7 |
| dev | 4.0 / 6.9 / 28.8 / 28.5 / 27.0 | 4.0 / 8.3 / 9.3 / 9.0 / 17.1 | 4.0 / 3.2 / 8.1 / 13.4 / 25.2 | 4.0 / 7.2 / 17.9 / 27.8 / 28.8 |

- dev reacts earliest to streaks in the fibre plane (at 50 it makes them a set of their own, which takes the merged
  field; the true set survives: best vector 16.9°) but saturates lowest at 200 (27° against 55° and 48°).
- Readers calibrated on the artefacted data (a chunk calibrating on itself) help the released pipeline against rings
  (8.5-11.4°) and dev against weak streaks (10.1° at 50), but at 100-200 both learn the streaks (dev 46-59° along x,
  both ~89° along y).
- Rings pull every method towards z (the rotation axis); on the 90° crossing they flip the released field to the z
  population (change 74-86°).
- Real slab (S-21-46 20 µm cortex, 128 x 32 x 128 core, change against the method's own unaltered field): with readers
  calibrated on the unaltered slab, **dev is the most stable**: streaks along x 5.5° / 7.6° at 50 / 200, rings 6.9° /
  6.9° (released 9.7 / 14.3 and 72.9 / 74.4; structure tensor 10.6 / 18.0 and 49.8 / 24.6). Calibrated on the altered
  slab, dev loses it (rings 59 / 44°; released 24 / 42°): calibrate on clean tissue (the volume calibration), not on an
  artefact-heavy chunk.

**QC figures** (`scripts/figures_dev.py`, `results/dev_pipeline/figures/`): the six phantoms, two artefact examples
and the real slab, mid-y plane: image, truth, structure tensor, released, dev, presence, dev layers, error maps.

**Open:**
- Crossings: found (presence) but not resolved; dense crossings partly look like texture to dev-vit.
- The presence null is synthetic texture: on the real slab 146/169 windows count as fibres (real tissue is far from
  synthetic texture whatever it holds); a real-data null is needed (e.g. the block's least oriented tissue).
- Streaks in the fibre plane become a population; a z-slab-confinement test (tomographic geometry) would reject them.
- Blocky population layers (32³ window blending); stage 2 costs 2-4 min per 128 x 32 x 128 block (per-window stage 2).

## 33. Population completion across crossings, and dev-vit's tile seams (2026-10-21)

**Completion** (`shapetensor.completion`, `scripts/test_completion.py`, pooled calibration, stage 2 shared by the four
stage 3s): each set carried across a crossing along its own direction from its arms, only where it holds on *both*
sides within reach (a free end is never extended). Three findings in turn:
- Carried along its first-pass direction it changed nothing (band 90 overlap pair 32.9 -> 29-36°): inside an overlap a
  set's own field is read from crossing-confused dyads. `diag_completion.py`: the band-2 set's direction there was 71°
  off (pulled onto band 1).
- Steering by the set's direction extrapolated from its *pure* voxels (alone and strong; normalised convolution, σ 8)
  fixes the direction (band-2 set 71 -> 4-7.5° in the overlap) but the activation stayed low (bridge = mean along 16
  voxels, mostly inside the 20-voxel overlap: band-1 set 0.16 -> 0.22 vs 1.6 on its arms).
- Bridge = each side's strongest activation within 24 voxels, the smaller side:

| overlap pair error, both within 15° | none | image | image + continuation | shape |
| --- | --- | --- | --- | --- |
| band crossing 90 | 27.1, 0 | 43.7, 0 | 12.0, 0.20 | 14.3, 0.27 |
| band crossing 60 | 24.8, 0 | 26.5, 0 | 15.7, 0 | 25.6, 0 |
| T-junction 90 (false extension) | 42.7 (0.029) | 43.1 (0.027) | 27.0 (0.027) | 38.9 (0.027) |
| T-junction 60 (false extension) | 21.9 (0.0) | 22.6 (0.0) | 18.1 (0.0) | 22.2 (0.0) |

  No false extension past a T. The shape-space variant (features averaged along the set's direction on each side,
  confirmed by the readout, directions read from them) keeps the bands outside the overlap as they were and is best at
  90° (the first overlap voxels with both directions within 15°), but does not help at 60° (24 voxels along a direction
  60° from the other band do not average it out); image + continuation reaches further (60°: 15.7°) but blurs the
  second band's arm by 2-5°. Both cost the tightest bend's turn 2-3°. Whole-block crossings: no change (stage 2 forms
  one or two mixed sets there; there is nothing to carry). Opt-in (`DevConfig.completion`), default off.
- Run-to-run: the pooled calibration and stage 2 are not deterministic on the GPU (band 90 baseline 27-33°, 2 or 3 sets
  between runs); compare modes within a run (they share stage 2).

**dev-vit's tile seams** (`scripts/figure_models.py`, the first three principal components of each model's features on
an axial slice of the S-21-46 20 µm VOI): the UNet's 16-channel output is smooth and anatomical, its encoder block 2
(the released pipeline's) fine texture, dev-vit sharp on vessels and the grey-white boundary but **offset tile by tile
at its 64-voxel cores** (the transformer attends over the whole 128³ tile, so a voxel's features depend on its tile)
with the 8-voxel patch grid faintly over it. Every window statistic of the dev pipeline straddles these seams; tiles
overlapping with blended cores (stride 32, a 2³ = 8x cost, or 48) would remove them.

## 34. Lag-TCA, and the lag ODF: a classical flexible-N pathway (2026-10-21)

Design decisions of the day (the user's): flexible N per position with population-specific momentum is a semi-hard
requirement; lag-TCA (a spatial-lag mode) is the primary separation and the populations model (section 33's kNN
distribution regression) the comparison; label-free; weighted rather than thresholded; native resolution; N <= 2 and
>= 30° by default, tunable; momentum ~0.5 mm following bends; quantitative tests on phantoms, visual QC on real data.

**Lag-TCA on dev-vit's features fails** (`shapetensor.lagtca`, `tests/test_lagtca.py`, `scripts/test_lagtca_phantoms.py`).
- A plain CP of the positions x channels x lags tensor does not separate populations (a field is not low rank over
  positions x lags). The low-rank tensor is the channels x channels x lags covariance, `C_l = A diag(g(l)) A^T`
  (INDSCAL), solved as SOBI (whitening + Jacobi joint diagonalisation); each source's lag profile gives a direction
  (least axis of a quadratic fitted to its squared differences) and an anisotropy.
- On synthetic multichannel fields it works: one population, two mixed in the same channels, isotropic texture with
  none, interleaved halves kept apart (5 tests pass).
- On dev-vit's 32 output channels it does not: fibre-free texture gives 4-5 "anisotropic" sources per window. Single
  channels look anisotropic in random directions; the features are smooth (a 32³ window holds ~64 independent
  samples) and sources in bundles miss the truth by 27-33° (median). The lag profile of dev-vit's features is also
  broad on the sphere (half height ~45-50° from the fibre direction), too broad to resolve crossings even after
  angular deconvolution (crossings: both within 15° in 0-9% of cells).

**The lag ODF** (`shapetensor.lagodf`, `tests/test_lagodf.py`, `scripts/test_lagodf.py`). A population invariant along
`d` adds a ridge along `d` to the image's local autocorrelation, and independent populations add. So the normalised
local autocorrelation of the *intensity* at lags `± r u` (r = 3, 6 voxels; 160 directions; Gaussian window σ 10
voxels; local mean removed at σ 8) peaks at every population's direction: Q-ball to the structure tensor's DTI. Then:
presence = peak minus median direction (weight 0 at 0.06, 1 from 0.12); angular deconvolution by NNLS with a
single-population response (synthetic bundles, or the block's own top-5% single-lobe cells); momentum per direction
(`populations.momentum`, σ 3 cells along, reach 6 cells ≈ 0.5 mm, bridging both sides, no free-end extension); lobes
at least 30° apart with share >= 0.15 and >= 0.3 of the strongest, refined off the grid; N = 0 below weight 0.2 and
on sheets (below). Held-out phantoms (seed 1; cells of the core; N share, top-lobe error, pair error and both within
15°), self response with momentum vs the populations model:

| case | lag ODF (self response, momentum) | populations model (no momentum) |
| --- | --- | --- |
| bundle 60 | N1 0.95, 2.4° | N1 0.86, 12.4° |
| bend, fascicles (GM around) | N1 0.63, 4.6° (GM N0 0.76) | N1 0.05 (GM N0 1.00) |
| straight fascicles | N1 0.94, 4.7° (GM N0 0.71) | N1 0.42 (GM N0 0.99) |
| band crossing 90 (interleaved) | N2 0.62, 5.1°, 0.62 | N2 0.00 |
| band crossing 60 | N2 1.00, 7.0°, 1.00 | N2 0.00 |
| T-junction 90 | false extension 0.00 | 0.00 |
| crossing 90 (intermixed) | N2 0.70, 9.6°, 0.45 | N2 0.91, 12.5°, 0.49 |
| crossing 60 | N2 0.95, 7.6°, 0.65 | N2 0.97, 10.9°, 0.78 |
| crossing 75 (new angle) | N2 0.78, 9.6°, 0.47 | N2 0.99, 7.7°, 0.98 |
| crossing 50 (new angle) | N2 0.74, 9.0°, 0.53 | N2 0.96, 6.2°, 0.83 |
| texture | N0 1.00 | N0 1.00 |

- Calibration-free in the separation; the synthetic and self responses perform alike (self better on thin fascicles,
  synthetic slightly better on 90/75° crossings). Momentum helps interleaved crossings most (band 90: both within 15°
  0.38 -> 0.62; band 60: 0.72 -> 1.00).
- N must count lobe *shares*, not absolute mass: a crossing splits its weight between two lobes, and an absolute floor
  dropped 90° crossings to N2 0.25. The weight floor must come after momentum (bridging otherwise spreads small weights
  into texture: N0 1.00 -> 0.72).
- A window elongated along each direction (σ 4 or 6 across, 12 or 16 along) instead of σ 10 everywhere was not better:
  fewer samples, noisier presence (GM N0 0.73 -> 0.44 at 4/12) and single-bundle error 3.6 -> 9.7°.
- Weak points: intermixed crossings at 90/75° (N2 0.70-0.78 vs 0.91-0.99) and the pair accuracy (both within 15° about
  half; the populations model is calibrated on crossings, the lag ODF is not); leakage of ~one window (200 µm) into
  grey matter beside fascicles (GM N0 0.71-0.83).

**Sheets.** On the S-21-46 20 µm cortex VOI (`scripts/figure_lagodf_real.py`, a 96-slice slab) the first version
reported two populations along every tissue boundary, sulcus and vessel wall: a sheet correlates along a whole plane,
so its lag ODF is a ring and peak-picking returns two of its points. `lagodf.ring_fraction`: the share of the
distribution's in-plane mass away from the kept peaks (~0.6 for a ring, ~0.15 for crossings) and the strongest value
over that ring's level (prominence). Sheet = ring > 0.35 and prominence < 5: on the phantoms the bend's fascicles
(planar edges, as the phantom is extruded along y) have prominence q10 5.3 and stay; on the real slab 3.6% of tissue
cells are sheets (boundaries, vessels). Real-data numbers: presence median 0.071 in the darker tissue, 0.099 in the
brighter, 0.18 outside the tissue (edges); tissue cells N0/N1/N2 0.45/0.18/0.37; where both give a direction the first
population is a median 12° from the standard structure tensor. Visual QC pending.

**Population identities** (2026-10-22, `lagodf.label_populations`, `viz.identity_rgb`, `scripts/test_population_labels.py`).
Instead of per-position ranks (strongest / second), each fixel (a position's lobe) gets a population number shared
across space. Candidate links: a fixel and a neighbour's fixel (26-neighbourhood) that are each other's best-aligned and
within 20°. Plain connected components merged everything into one population on every phantom: chains of in-between
directions (unresolved crossing cells, grey-matter leakage) bridge crossing bundles, and the bridging fixels look like
good ones (same weight, share and lobe sharpness). The rule that separates them: one population cannot hold two
fixels of one position. Links are taken best-aligned first (Kruskal) and two populations merge only if the merged one
would hold two fixels of a position at no more than max(2, 2% of the smaller's fixels) positions (the tolerance lets a
few noisy fixels not keep one population in pieces, and a tight bend's corner, where the window sees both arms, not
split it); where a population still holds two fixels of a position the weaker loses the label. A face margin (max
radius + 2σw voxels) removes a frame of spurious labels. Phantoms (dominant label per truth population, coverage of its
cells): band crossings 90/60 and the T-junction keep two populations, each one label through the overlap (coverage
1.00/0.90, 1.00/0.92, 0.99/0.84); the bend is one population through its turn; intermixed crossings 60/90 two
populations (coverage 0.88/0.89, 0.56/0.65, the latter limited by N = 2 detection). The bend's and straight control's
three fascicles (0.2 mm apart) share one label: the 10-voxel window's leakage joins them, below this resolution they
are one orientation population. Real slab: 137 populations of >= 20 fixels; the largest runs along x in the white
matter under the cortex with a second, interleaved one in the same cells.
