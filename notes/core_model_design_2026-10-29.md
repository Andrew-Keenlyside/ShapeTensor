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

## Answers from the user (29 Oct 2026, 22:00 ET)

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

## Implementation plan: per-object second-order ODFs (29 Oct 2026, 23:15 ET)

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
