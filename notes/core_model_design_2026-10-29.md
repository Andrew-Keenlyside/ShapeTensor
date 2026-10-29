# Core model design (29 Oct 2026)

**Scope.** ShapeTensor is a general computer-vision model. It maps any array of intensities to a separable latent from
which object-specific orientations, and groupings of objects at any size, can be read. It is not a biology model. The lab
applies it to welds, materials and non-biological fibres as well as tissue. Biological images, HiP-CT above all, are the
primary *test*, not the scope. Training data is built "like the world": natural image histograms, smooth flowing curves,
objects and their groupings. The loss is only ever computed on synthetic images. Real images are used only for validation.

The agreed core, after a day of density pilots that failed for structural reasons (see
`full_review_2026-10-28.md`, the structural-review section). The guiding rules: **established methods only**
(SynthSeg / anatomix-style domain randomisation, DETR-style set prediction, proper scoring rules); **targets from
construction only**; **the structure tensor is a benchmark, never part of the model or its training**; prefer removing
machinery to adding it.

## What the model is

A contrast-, resolution- and artefact-agnostic **shape latent**, computed once per volume by a 3D CNN trunk. Its job is
to identify objects and how they group. A light, knob-conditioned query decoder reads any number of outputs from it.
Populations are **hierarchical**, like clusters at different cut levels, not fixed semantic layers. A voxel may hold
contributions from a sheet (for example a grey-white boundary), a vessel, several fibre populations, background and
minor noise.

### Readout knobs (one latent pass, many outputs)

| knob | meaning | how it is trained |
|---|---|---|
| **x** | any point, so any output voxel size | continuous queries |
| **sigma** | the spatial window: how much tissue is described | random per query |
| **g** | the grouping scale: the size of structure a population is (axons, then fascicles or bundles, then tracts) | the construction tree is cut at radius g, using `Tree.resolve` |
| **tau** | the angular blur: sharp changes between neighbouring populations, or a smooth field between their ODFs | random per query; exact on log-SH by Funk-Hecke scaling per order, analytic on Bingham |
| **L** | the ODF order per population | random truncation during training (nested SH), so any order up to L_max reads out well |

The **g x L trade-off** is the user's choice at readout time:
- one population of high order: g coarse, L = 8, which is a classical fibre ODF;
- many populations of order 2, each just a direction and its anisotropy: g fine, L = 2;
- anything in between, for example a fibre population of order 2 beside other populations.

### Per query output

A set of populations at the requested grouping scale, plus the remainder (background and noise). For each population:

| output | form |
|---|---|
| existence | probability |
| mass | its share of the window's material |
| ODF | log-SH coefficients up to L, smoothed by tau (normalised and positive) |
| descriptors | learned object attributes, supervised from construction: kind (line, sheet, blob), radius, hollow lumen, relative brightness, packing |
| embedding | a population code, for grouping the same population across queries |

"Fibre only" is a filter on the descriptors. Lumen, brightness and the rest are learned evidence that separates vessels
from fibres. The fibre ODF of a voxel is the mass-weighted sum over the fibre-like populations. Context from the wider
region informs which objects are present within the window.

## Targets (from construction only)

For each query window and grouping scale g: the construction tree's nodes resolved at radius g (`Tree.resolve`), whose
material falls in the Gaussian window of width sigma. Each population gets its node id, descriptors from construction,
mass, and its ODF: the mass-weighted mixture of its member objects' orientation distributions inside the window, as
log-SH up to L. The existing `entity_targets` already gives per-entity tensors, Bingham fits and descriptors per node. The targets carry **no visibility, presence or
classical-reader terms**. Whatever the image doesn't show, the model must express as uncertainty: broad lobes, a low
existence probability or a high isotropic fraction. Targets are smoothed with the same tau and truncated to the same L as the query.

## Losses (proper and standard)

- **Set matching:** DETR-style Hungarian matching of the K slots (K = 4-6) to the target instances. The matching cost is
  the cross-entropy of the target instance's ODF under the slot's ODF, plus mass and class costs.
- **Matched slots:** cross-entropy of the target instance's ODF on a fixed direction grid, L1 on the volume fraction, and
  class cross-entropy. **All slots:** existence BCE.
- **Total ODF:** cross-entropy of the window's full non-background ODF under the sum of the slots. It is an auxiliary
  term and is proper, so it trains masses and shapes together.
- **Instance embedding:** a discriminative pull/push loss across the queries of a crop (De Brabandere et al. 2017).
- No gates, no readout tricks, no curricula beyond a standard warm-up.

## Data (SynthSeg-style, deliberately broad)

The generator must keep widening, in geometry and in corruption. The current modality and artefact list (X-ray / CT,
EM, MRI, light microscopy, ...) is a proof of concept for teams in those fields, not an exhaustive list.

- **Geometry already in place:** bundles, trees, networks, helices, sheets, foams, porous media, cells, crystals, radial
  forms, plywood layering, lattices, honeycombs, jigsaws, and compositional populations (about half of all draws).
- **Geometry to add, materials:**
  - weld microstructure: columnar and dendritic grains growing from a fusion boundary, and orientation fields across a
    heat-affected zone;
  - fibre-reinforced composites: unidirectional and woven tows, laminates with ply angles, short-fibre composites with
    random orientation distributions;
  - polycrystalline grain maps with texture;
  - cracks, voids and wood.
- **Corruptions to add:** speckle (ultrasound, OCT), beam hardening, scatter and metal streaks (CT), photon-starvation
  streaks, dead and hot pixels, compression artefacts, motion blur, stitching seams, resampling artefacts, quantisation,
  saturation and clipping, and non-stationary noise.

- **Geometry:** the scene generator (bundles, packed fibres, vessels, sheets, blobs, crossings, fanning), plus fibres
  from synthspline (MIT licence) through an adapter. Every object carries an instance id and a class. Orientations are
  drawn isotropically, and random rotations and flips are applied with the targets rotated to match.
- **Contrast (main path):** random intensities per label or instance (Gaussian mixtures, random textures), then random
  resolution (anisotropic downsampling by 1-4x and back), blur, bias field, gamma, white and correlated noise, and
  artefacts (rings, stripes, missing lines and sections). Full randomisation from the first step, as in SynthSeg.
- **Contrast (extra):** the physics renderers (X-ray phase contrast, EM, light sheet and others, including the HiP-CT
  profile) are used as random alternative renderings in some of the crops.

## Evaluation (what adoption will be judged on)

1. **Phantoms with known truth where the structure tensor fails and the model succeeds:**
   - crossings at 30-90 deg;
   - parallel structures closer than the tensor's window;
   - curving and fanning;
   - low-contrast texture;
   - sheets against lines;
   - fine fibres inside bundles, read at several grouping scales.
2. **HiP-CT against paired diffusion MRI** of the same specimens, at matched scale (validation outside training).

Also:

- **Synthetic hold-outs** from our generator and from synthspline. Report per class and per instance:
  - the ODF cross-entropy gain over uniform;
  - the axis error of matched instances;
  - crossing separation: both bundles recovered within 15 deg, for crossings of 45 deg or more;
  - fraction error;
  - calibration.
- **Benchmark row:** the fitted structure tensor, compared with the fibre-class ODF.
- **Real data, evaluation only:** comparison with co-registered diffusion MRI of the same specimens, at matched scale,
  through the SH export.

## Removed from the current code

The structure-tensor paths (`--st-path`, `st6` inputs, the gate's tensor cues), visibility targets (`e_ovis`,
`ORIENT_VIS` and presence discounts in the targets), the gates and readouts (`--dens-gate/readout/share`,
`gate_share`), the log-L1 density loss, the per-direction vote/owner machinery (replaced by set prediction) and the
strength curriculum.

## Kept

The trunk and its voxel conditioning, query encoding at (x, sigma), per-entity Bingham targets, the cross-entropy grid
code, the scene geometry, today's target bug fixes, the evaluation script structure and the unit tests.

## Stages, each with a test that must pass before the next

1. **Targets and data.**
   - Per-query instance lists from construction.
   - The SynthSeg contrast module.
   - Tests: targets are rotation-equivariant; crossing cores give two instances; contrast draws are diverse.
2. **Decoder and loss.**
   - The set-prediction head, Hungarian loss, analytic tau and embeddings.
   - Tests: slot permutation invariance; tau smoothing exact against numerical convolution; normalisation.
3. **Overfit 8 clear crops.** Pass: at least 80 % of the oracle cross-entropy gain, and crossings separated.
4. **Local pilot, about 2 h.**
   - The fibre ODF must beat the fitted structure tensor at every query kind, including crossing cores.
   - Matched-instance axis error must improve with training.
   - Then the cluster.

## Open questions (29 Oct 2026)

**For the user and collaborators (context only you have):**
1. **Physical scale.** Which voxel sizes and windows matter? Is sigma in µm, which needs the voxel size at inference, or
   in voxels? Which sigma matches a dMRI voxel for the HiP-CT validation?
2. **Hierarchy semantics.** Should the grouping levels be set by physical size (radius), by construction groups, or by
   both? Where does a strand stop being an instance and become texture inside a population?
3. **Descriptors.** Which object attributes do downstream users need (lumen, brightness, radius, shape kind, tortuosity,
   packing), for fibre/vessel filtering, vascular work and registration?
4. **Outputs.** SH convention (MRtrix / DIPY), normalisation (amplitude = volume fraction?), fixels, file formats
   (OME-Zarr, NIfTI), and inference throughput for whole-organ volumes.
5. **Validation data.** Which paired HiP-CT / dMRI specimens, their registration quality and resolution; any manual
   annotations; which phantom metrics reviewers will expect.
6. **Baselines beyond the structure tensor:** multi-scale tensor, Fourier / OrientationJ-style methods, optimally
   oriented flux, Frangi, existing deep-learning tools.
7. **Non-biological domains to add first:** composites, foams, wood, fibre-packing simulators the lab already has.
8. **Compute.** Cluster budget and limits (pryor's seymour2 jobs get pre-empted), and model-size limits.
9. **Real-data boundaries.** Is real data off-limits for hyperparameter choice and early stopping too? The phantoms'
   texture parameters and a generator grain comment cite a real HiP-CT crop: keep or remove?
10. **Heads to keep or retire.** Which current heads (tree, votes, regions, contrast) does anything downstream depend
    on?

**Technical, to settle by short tests:**
1. **The texture case.** Does the density learn from the oriented-texture family, and stop claiming orientation in
   isotropic texture?
2. **Set prediction.** The number of slots K, the matching cost, stability, and how the grouping knob g interacts with
   the set.
3. **Angular smoothing tau is not exact on log-SH or Bingham.** Convolution acts on the linear-domain ODF: per-order
   Funk-Hecke scaling is exact only for linear SH, and Bingham convolved with Watson is only approximately Bingham.
   Simplest: condition the head on tau and train against tau-smoothed targets (learned, not analytic).
4. **Resolution agnosticism.** Voxel-size conditioning against multi-resolution training; sigma and g in physical
   units.
5. **The trunk.** Keep the current trunk, or use a standard 3D U-Net as in anatomix, which is easier to defend.
6. **Context size.** 64³ crops against 128³ for tract-scale grouping (the receptive field is 182 voxels).
7. **Loss balance** across heads, and the held-out sets: an independent generator (synthspline) and the phantoms.

## Answers from the user (28 Oct 2026, 22:00 ET)

1. **Windows are in voxels,** like the structure tensor's rho, and arbitrary during training. They are converted from
   the known voxel size at use.
2. **"Angular smoothing" is spatial pooling, not blurring an ODF's shape.** The ODF at each sample point describes the
   orientations present in a larger region. Small values give sharp spatial changes between neighbouring populations, as
   a small-rho structure tensor does; large values give gradual transitions. There is still one ODF per sample point.
   *Consequence:* this is the window (rho) knob. The ODF target is the mass-weighted orientation distribution of
   material within a Gaussian window of that width, which is what the window knob already does. **The separate
   shape-blur knob tau is dropped.**
3. **Grouping levels follow structure.**
   - An object is a single fibre, or the finest unit the resolution allows.
   - Coarser levels group objects into sub-populations (2-3 fibres), then larger parts of a tract, then the whole tract.
   - Objects may cross and overlap in space, and groups may overlap in space.
   - At a given level, **each object belongs to exactly one group:** groups partition the objects, and no two groups
     claim the same object. This is a nested partition, which is the construction tree. To decide or test.
4. **Outputs are plain arrays:** array in, array out.
   - A multi-channel SH array per voxel, with another axis per population or object for per-population readouts.
   - Chunked or OME-Zarr scaling comes later, in bridge, as anatomix plus chunkreg does today.
   - Fixels are post-processing after the ODFs.
   - The SH convention is ours to choose and document; MRtrix3's real even basis is a sensible default, with a
     converter.
5. **Validation:**
   - paired 20 µm HiP-CT and 600 µm ex vivo dMRI of the same hemisphere, already registered (post-validation only);
   - phantoms for known cases: crossing and fanning fibres.
   - Baselines: dMRI and the structure tensor only.
6. **Non-biological domains:** dense, varied texture first. Foams and other domains come after an initial model works
   well on biological data.
7. **Compute:**
   - a shared 8 x H100 cluster; single- or multi-day runs on at most 3 GPUs (up to 4-6 at most);
   - UCL and MIT compute only for well-proven final models, for example if 10-50 M parameters prove necessary;
   - the current model has 3.9 M.
8. **Heads:** nothing relies on the current heads, so they can change freely.
9. **Technical questions are answered by tests on pilot models** once those exist.

Still open: whether real data may be used for hyperparameter choice or early stopping; the real-crop provenance of the
phantoms' texture parameters and the generator's grain comment.

## Implementation plan: per-object second-order ODFs (28 Oct 2026, 23:15 ET)

**Framing.**
- **Per voxel, per object:** the second moment of the object's local material directions, with its own spread drawn at
  construction. Solids are isotropic.
- **Per query, per object (at the grouping level):** one second-order Bingham lobe (axis + two spreads, so fanning) and a
  mass.
- **Scene target:** the mass-weighted sum of the objects' lobes (sum of densities, never of tensors).

### Issues to settle first
1. **Post-render displacement** (drift, misalignment, slice motion, ghosts, folds, held sections, augment `sections`,
   `resample_artefacts`) moves the image but not the targets. Disable now; route through the targets' `grid` later.
2. **Oriented nuisances** (`oriented_noise`, oriented speckle) look like texture populations without targets. Make them
   isotropic.
3. **Solids** carry their whole shape moment at every voxel. Make them I/3 unless the family is declared fibrous.
4. **Fibres and strands** are pure tangent dyads. Give each its own spread at construction, then remove the global
   `ENTITY_DISPERSION` floor.
5. **Vessel walls** (user's choice): the local material (planar wall), or the vessel axis (line).
6. **Sharpness against the direction grid.** Without the floor, straight fibres have kappa in the hundreds; the 256-point
   grid (about 9 deg) cannot represent that. Keep an explicit *resolution* cap per lobe, tied to the grid, or use a
   denser grid. This is a representational limit, not a semantic floor.
7. **Texture-family visibility.** Most draws visible, a minority faint.
8. **Mass dropped** by `m_min` / `share_keep` / `emax`: measure and report; keep it negligible.

### Steps, each with its tests (no training until A-C pass)
- **A. Construction semantics** (scene, generator, compose).
  - Tests on single-object scenes:
    - a straight fibre gives axis = its direction and kappa = its drawn spread;
    - a sphere gives isotropic;
    - a sheet gives a girdle;
    - a fan gives kappa1 >> kappa2 with the spread in the fan plane;
    - a solid ellipsoid gives isotropic.
- **B. Targets** (entities.py, train.py, entity_train.py).
  - Per-object window moments become one Bingham each, with the resolution cap and no floor.
  - The CE target is their mass-weighted sum.
  - Tests:
    - the scene target equals the sum of the object lobes;
    - the dropped mass is reported;
    - the construction figure is re-rendered.
- **C. Acquisition consistency.**
  - Displacement nuisances off; oriented nuisances isotropic.
  - Tests: a rod scene's image axis matches its target axis within 5 deg under every corruption, and the records contain
    no displacement nuisance.
- **D. Short training tests** (at most 30 min each):
  - the capacity overfit on 8 clear crops (at least 80 % of the oracle);
  - from scratch against a warm start on a balanced strength mix;
  - figures: synthetic truth against model, phantoms with a fanning case added, and the construction figure.
- **E. Per-object set head**, once D passes.

## Fixes for the remaining issues (28 Oct 2026, 23:45 ET)

| # | issue | cause (diagnosed) | fix |
|---|---|---|---|
| 1 | a rendered structure without targets | region-interface layers are rendered but excluded from targets (`NOT_ENTITY`) | each interface becomes a boundary-sheet entity: planar ODF (I - nn^T)/2 with n = the region's gradient direction, mass = the boundary's sharpness (gradient magnitude). Interiors stay plain tissue. `INTERFACE_SHEETS` in train.py; NOT_ENTITY elsewhere unchanged |
| 2 | "fascicles" all isotropic | the family is skipped when the scene's voxel size is outside its scale range, leaving only background; the figure did not check | the figures confirm the family is in the scene record, and draw at a voxel size inside its range |
| 3 | unresolved bundles are one object | strands share the bundle's object id | each strand becomes its own object (leaf under the bundle). Texture-field strands are grouped into local clusters (objects under the field) to keep the entry count bounded |
| 4 | the acquisition seed changes the scene | sparse labelling, the specimen edge and cut-outs draw from the acquisition stream | draw them from a geometry stream tied to the scene seed and crop; acquisition keeps only appearance draws |
| 5 | full corruption can erase a scene | no control on surviving signal | measure the corrupted image's correlation with the clean render (after a small blur). Below a floor, redraw the acquisition at half strength with probability 0.8, so about a fifth of hopeless crops remain for robustness |
| 6 | the density head uses a direction grid and a kappa cap | a cross-entropy grid head | a set head: K = 4 population slots per query, each with existence, mass share (softmax with an "other" slot) and a trace-1 ellipse (Cholesky, positive by construction). Targets are the top-4 entities by mass, the rest pooled. Matching by brute-force minimum over the 24 slot permutations on the GPU (exact for K = 4). Loss: existence BCE, mass L1, and mass-weighted squared Frobenius distance on the ellipses (minimised by the conditional mean, so proper). No grid, no cap |

**Review.**
- **1:** a boundary is a real, rendered sheet, consistent with the vessel-wall rule. Its mass is a surface density, not a volume. Acceptable for a target share, documented.
- **3:** per-strand objects can be thousands for texture fields, so they are clustered there. Unresolved bundles have about 200-400 strands, which is fine.
- **5:** the threshold is a heuristic. It is logged per crop so its rate can be tuned. It never touches targets.
- **6:** brute-force permutations are exact and cheap at K = 4. More than 4 populations in one window are rare (under 1 % measured). Their mass goes to "other" and is reported. A summed-tensor total is deliberately not trained, since it hides crossings.
- **Order:** 4 and 1 first (targets), then 3, 5, 6; figures and tests after each.

## Overnight plan and acceptance criteria (fixed before the pilot, 29 Oct 2026, 00:10 ET)

### Remaining work before the pilot
1. **Fix 3:** per-strand objects in unresolved bundles, and clustered objects in texture fields (in progress).
2. **`scripts/eval_populations.py`:** held-out synthetic sets plus phantoms, scored on the populations head against the
   fitted structure tensor and trivial baselines, with the metrics below.
3. **`scripts/figure_populations.py`:** truth against model as stacked ellipses, on synthetic crops and phantoms.
4. **Unit tests,** then commit.
5. **The pilot,** from scratch (no warm start), at most 1 h:
   - universal generator with every fix;
   - strength and difficulty mixed per step;
   - populations head (K = 4);
   - old density loss off;
   - construction-only targets.

### Pilot acceptance criteria (gate to the long run)
Measured on 32 held-out crops (universal generator at strength 0.3 and at 1.0) and on the phantoms. Windows of 3
voxels. Targets are each query's populations from construction.

| # | what | criterion |
|---|---|---|
| A | orientation | median axis error of the matched population at single-line queries (one line population with share >= 0.6), strength 0.3: <= the fitted structure tensor's |
| B | crossings | queries with two line populations (each share >= 0.2, >= 45 deg apart): both axes found within 15 deg by slots with existence > 0.5 in >= 25 % (strength 0.3); the structure tensor: 0 % |
| C | populations | existence F1 for populations holding >= 10 % of the window >= the trivial "every slot present" F1 + 0.05; share error <= 0.8 x that of the trivial "one population takes everything" answer |
| D | morphology | shape class of matched populations (line / sheet / isotropic, from eigenvalues) correct >= 60 % (chance ~33 %); fibre-free texture phantom: the dominant population's line-ness < 0.4 at >= 60 % of points |
| E | learning trend | validation population loss at the end <= 0.8 x its value after warm-up; A-D better at the end than at the halfway checkpoint |
| F | no collapse | predicted existence rate within a factor of 2 of the target's; line-ness of predicted ellipses varies (sd > 0.1) |

**Outcomes.**
- **All pass:** the longest local run that ends before 09:00 ET, checkpointed, scored on the same metrics at intervals.
- **Any fail:** diagnose with the figures and fix the *structure or the data* (not the hyperparameters), then another
  pilot of at most 1 h.

**Stretch targets for the long run:**
- A at <= 0.7 x the structure tensor;
- B >= 40 % (and >= 40 % at the 90 deg phantom crossing);
- C F1 >= 0.75;
- D >= 75 %.

*Amendment before the pilot ran (00:35 ET):* C's first form was met by an untrained model (F1 0.89 from predicting every
slot), because most windows hold 3-4 populations including small fragments. It now scores populations with >= 10 % of
the window, against the every-slot-present answer.

## Overnight log (night of 28-29 Oct 2026, times ET)

- **23:35** Pilot P1 (from scratch, populations head, generator before the rendering fixes) started as a baseline: about
  2 s/step, 1500 steps.
- **23:50** Generator phantom check (`scripts/check_generator_orientation.py`): the **targets follow the true
  centrelines** (bundle 0 deg, fascicles 4.5 deg, network 5.5 deg median). **The images often don't show them:**
  - flat polygonal compartments dominate crops;
  - crops often miss the family;
  - unresolved bundles and fascicles render as smooth fills;
  - the clean image's structure tensor is 73 deg from a bundle's true axis.

  Rendering fixes are in progress (strand texture visible, crops centred on material, compartments a minority). The
  pass condition: the image structure tensor within 20 deg of the true axis (median) per line family at strength 0.
- **Figures:** near-isotropic populations (mostly background) are drawn as small grey spheres, so oriented populations
  stand out.
- **00:49** **P1 failed 5 of 6 criteria** (A 55.1 deg vs the structure tensor's 45.3; B 0 %; C passed; D 58 %; E and F
  failed). Every predicted ellipse was isotropic. The ellipse loss sat at its starting value (0.165) for all 1500 steps,
  and the head's ellipse weights barely moved (mean |w| 0.003), while the vote head reached 6 deg on the same trunk.
- **01:00** **Diagnosis**, each step a short test:
  - *Overfit, 8 fixed clean crops, populations loss only:* the ellipse loss stayed at 0.40, so the head could not fit
    even its own training data.
  - *Targets checked:* every target has trace 1, there is no half-precision underflow, and 32-55 % of populations are
    line-like.
  - *A fresh head on the trained net's frozen features:* 0.39 -> 0.09, so the loss and the head's maths are sound.
  - *The trainer's step replicated and ablated:*

    | variant | ellipse loss after 150-200 steps |
    |---|---|
    | full | 0.40 |
    | no augmentation | 0.06 |
    | no cube transform (contrast routes on) | 0.08 |
    | no contrast routes (cube transform on) | 0.41 |

    The cube transform rotates targets and query points consistently with the image (checked in 48 of 48 draws). So
    the head simply could not read orientation from the point code once crops were turned. This is the same failure
    noted on 27 Oct for the density path, which is why that path reads the image along each direction.
  - **Fix (`--pop-dir 96`):** each slot's ellipse is the second moment of learned weights over 96 hemisphere directions
    (randomly rotated in training). Each direction is scored from the point code and the direct path's code along it,
    so the ellipse turns with the image by construction (unit test). Under the cube transform: 0.39 -> 0.26 in 150
    steps.
  - **Loss:** every real population's ellipse is now weighted equally (no longer by share).
- **01:10** **Generator rendering fixes done** (separate agent). Image structure tensor vs true axis, single-line
  queries, strength 0:

  | family | before | after |
  |---|---|---|
  | bundle | 3.3 deg | 3.0 deg |
  | unresolved bundle | 25.8 deg | 4.5 deg |
  | fascicles | 16.5 deg | 3.2 deg |
  | network | 37.2 deg | 9.3 deg |
  | texture | 29.5 deg | 16.0 deg |

  Hollow trees give too few single-line queries to score: their walls are sheets. Committed with the head fix.
- **01:21** **Pilot P2** from scratch: fixed generator, `--pop-dir 96`, populations loss only, 1500 steps (about 1.6
  s/step).
- **02:22** **P2 passed 3 of 6 criteria** (P1: 1 of 6). Evaluated at strength 0.3 unless noted:

  | criterion | result | verdict |
  |---|---|---|
  | A: single-line axis | 50.0 deg vs structure tensor 45.3 (at strength 1.0: 41.2 vs 44.5) | FAIL |
  | B: crossings, both found | 0 % | FAIL |
  | C: existence F1 | 0.841 vs 0.760 | PASS |
  | D: shape class / texture phantom isotropic | 60 % / 100 % | PASS |
  | E: val_pop over training | flat (strength 1.0 held-out set) | FAIL |
  | F: line-ness sd | 0.157 | PASS |

  Phantom bundle axis: 8.9 deg (structure tensor 13.7).
- **02:20-02:45** **Further structural fixes**, each tested on 8 cube-transformed clean crops (300 steps):
  - **Line targets got averaged ellipses.** Predicted line-ness was 0.17 and the axis error 28 deg. The matching
    weighted ellipse cost by share, while the loss weights every population equally. As a result, line and isotropic
    targets swapped slots between like queries.
    - Fix: the matching now uses the loss's weights. Line targets 0.42 -> 0.35, isotropic 0.056 -> 0.012.
    - A learned per-slot sharpness was added: axis error 25 -> 23 deg.
  - **A learned-field probe** (trunk channels sampled along each direction) brought no gain at this budget. It is off
    by default.
  - **The bank is not the limit.** Its strongest direction lies 15-31 deg from line targets (median, by pooling
    level).
- **02:34** **Long run L1** started from scratch with the P2 configuration plus these fixes: 11000 steps, about 2.3
  s/step on the shared card. Gated at step 1500 by `acceptance.py` (750 vs 1500); the run stops if the gate fails.
- **02:45** **Crossings (B = 0 %) look structural.** Each slot scores each direction independently from the same
  features, so two similar peaks go to every slot alike and a slot averages them into a disc.
  - Fix (`--pop-seq`): slots are filled in turn, and each direction's score also sees how much of it the earlier slots
    hold (explaining away).
  - Unit-tested for rotation and batching; being tested on crossings now.
- **03:01-03:08** **L1 stopped** (it would fail B by construction). **Long run L2** from scratch with `--pop-seq`:
  11000 steps. It was restarted once at step 114 to lengthen it, resuming from its checkpoint.
- **03:20** **Figure fix.** `scripts/figure_generator_inplane.py` shows each family in the plane of its fibres.
  - **The targets follow the fibres:** the bundle's ellipses are lines along it, eigenvalues [0, 0, 1], 20-35 % of the
    window.
  - **The old figures misled.** The isotropic background, scaled by its larger share, was drawn as a sphere that hid
    the line inside it. Isotropic glyphs are now small.
- **04:21** **L2 failed the gate at step 1500** (A, B, E; C, D, F passed).

  | measure | L2 at step 1500 | P2 |
  |---|---|---|
  | A: single-line axis | 59.8 deg (structure tensor 43.8) | — |
  | D: shape class | 73 % | 60 % |
  | phantom crossings, both found | 6-30 % | ~0 % |
  | phantom bundle axis | 42 deg (structure tensor 13.6) | 8.9 deg |

  The bundle regression is not the un-annealed learning rate: P2 at step 831 already had 10.7 deg, while L2 had 31.7
  deg at step 912. So one of L2's structural changes (matching weights, sharpness, sequential slots) costs
  single-line orientation. The run was stopped by the gate.
- **04:25** **Readability scan** (`readability.py`, no model; the structure tensor only as an outside reader).
  Structure tensor vs the true axis at the evaluation's single-line queries:

  | strength | median error | by modality |
  |---|---|---|
  | 0 | 2.1 deg | — |
  | 0.1 | 12.6 deg | — |
  | 0.3 | 32 deg | histology 16, CT 64, cryo-ET 50 |
  | 0.6 | 44 deg | — |
  | 1.0 | 40 deg | MRI 36, histology 60 |

  Much of the moderate-to-full-strength data shows no orientation at a 3-voxel window. Two consequences:
  - The proper score teaches hedging there.
  - Criterion E, measured on the trainer's full-strength validation set, can barely move.

  One full-strength CT crop posterised to 3 grey levels and still passed the signal floor (correlation is
  contrast-blind).
- **04:32** **Pilot P3** (1500 steps, from scratch): L2's structure with the share-weighted matching restored
  (`--pop-match-share`), to isolate the cause of the regression.

## Day log (29 Oct 2026, times ET)

- **~05:00** The overnight session ended. P3 died at step ~450 with no long run. The machine rebooted at 09:37.
  - CUDA now starts only when limited to the card at 41:00, by UUID; the card at 2B:00 errors.
  - Jobs are now started with `setsid` and drivers live in the repo (`scripts/pilot.sh`), so a session ending can't
    kill them.
- **10:20** **HiP-CT deep brain** (S-21-46, P2 model; `scripts/figure_hipct_dec.py`). In fibre crops the model's
  predominant direction is a near-uniform yellow-green (mostly y, some x).
  - **10:58** **Check of the line bank's strongest direction** (an image-only reader):
    - in the two fibre crops it is mostly y, |x| 0.4-0.5 (|y| > 0.8 at 54-94 % of points, against 20 % for random);
    - in a grey-matter crop it has no preference (7-17 %).
  - **So the y is anatomical, not an acquisition artefact,** and the model's colour there is coarsely right. The
    remaining fault is coarseness: no spatial variation, and duplicated slots.
- **10:30-11:00** **Fixes implemented** (unit-tested, CPU smoke-tested together):
  - `--pop-seq 2`: claims from smoothed direction weights (not cos^2 lobes); the background as its own share; shares
    by stick breaking with sequential existence;
  - `--od-widths 0.6 1.5 3`: bank cross-sections, each high-passed at 3 x its width;
  - `--probe-scale`: the probe's offsets follow the window;
  - `--e-line`: a line-stratified share of queries;
  - evaluation at extra windows (`WINDOWS`) and duplicate-slot pairs.
- **11:10-12:10** **Sequential-slot variants**, 8-crop test (`scripts/replicate_pops.py`, 300 steps, cube-transformed):

  | variant | line axis | crossings found | duplicate pairs / query | existence |
  |---|---|---|---|---|
  | mode 1 (claims u^T T u) | 18.3 deg | 10 % | 0.07 | learns |
  | mode 2, free matching | 14.7 deg | 23 % | 0.01 | stuck at 0.69 (pilot P4a) |
  | mode 2, rank order (tie 0.1) | 23.6 deg | 3 % | 1.04 | learns |
  | mode 2, rank order (tie 0.3) | 19.5 deg | 14 % | 0.16 | learns |
  | mode 2, soft (share x 4) | diverged | | | |
  | **mode 3, free matching** | **15.9 deg** | **19 %** | **0.02** | from share |

  - **Mode 2 with free matching, existence stuck.** The early stick-breaking shares are alike, so matching is by
    ellipse alone. Which slot gets a target is then unpredictable, so per-slot existence can only learn the mean, and
    the background took 0.78 of the window (true 0.64).
  - **Rank order makes slots hedge.** With near-equal top shares, the best squared-loss answer is their average in
    both slots, which is duplication.
  - **Mode 3** removes the existence output: a slot exists when its share passes 3 %.
- **11:00-11:50** **Tests: the 16 failures triaged.**
  - **7 new, from the generator changes.** Six were tests written before the per-object semantics or the
    visible-strand knob, and were updated. One was a real bug: `curves_like` without a generator.
  - **9 older (failing before 28 Oct).**
    - Updated to the approved semantics: blobs (shape, not isotropic material); vessel walls (sheets); entity totals
      (without interface sheets).
    - Expected failures, with reasons: the legacy lattice generator's saturation and face fall-off; the lattice
      cells' 0.55-voxel offset; tortuosity capped by the no-folding limit.
- **12:09** Queue (`scripts/pilot_queue.sh`):
  - **P4:** mode 3, free matching.
  - **P5:** P4 + bank widths 0.6/1.5/3, learned probe fields scaled by the window, 30 % line-stratified queries.
  - Two pilots cannot share the 48 GB card: validation draws fell back to the CPU.
