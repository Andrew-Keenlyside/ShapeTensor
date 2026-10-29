# Core model design (29 Oct 2026)

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

## Data (SynthSeg-style)

- **Geometry:** the scene generator (bundles, packed fibres, vessels, sheets, blobs, crossings, fanning), plus fibres
  from synthspline (MIT licence) through an adapter. Every object carries an instance id and a class. Orientations are
  drawn isotropically, and random rotations and flips are applied with the targets rotated to match.
- **Contrast (main path):** random intensities per label or instance (Gaussian mixtures, random textures), then random
  resolution (anisotropic downsampling by 1-4x and back), blur, bias field, gamma, white and correlated noise, and
  artefacts (rings, stripes, missing lines and sections). Full randomisation from the first step, as in SynthSeg.
- **Contrast (extra):** the physics renderers (X-ray phase contrast, EM, light sheet and others, including the HiP-CT
  profile) are used as random alternative renderings in some of the crops.

## Evaluation

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
