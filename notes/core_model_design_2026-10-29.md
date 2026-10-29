# Core model design (29 Oct 2026)

The agreed core, after a day of density pilots that failed for structural reasons (see
`full_review_2026-10-28.md`, the structural-review section). The guiding rules: **established methods only**
(SynthSeg / anatomix-style domain randomisation, DETR-style set prediction, proper scoring rules); **targets from
construction only**; **the structure tensor is a benchmark, never part of the model or its training**; prefer removing
machinery to adding it.

## What the model is

A contrast-, resolution- and artefact-agnostic **shape latent**, computed once per volume by a 3D CNN trunk. Any number
of outputs can be read from it with a light query decoder at:

- **x**: any point. The output voxel size is free, because the decoder is queried at continuous coordinates.
- **sigma**: the spatial integration scale. This is the window whose contents are described, the equivalent of the
  structure tensor's rho.
- **tau**: the angular smoothing. It is an explicit knob, so ODF sharpness can be tuned independently of sigma.

Each query returns a **set of instances** present in its window: the bundles and objects whose material falls inside it.
For each instance:

| output | form |
|---|---|
| existence | probability |
| class | fibre, vessel, boundary / sheet, blob / cell, artefact / crystal |
| volume fraction | its share of the window's material |
| ODF | a Bingham distribution (frame R, concentrations kappa1 >= kappa2). Angular smoothing is analytic: kappa' = kappa tau / (kappa + tau) |
| embedding | an instance code, so the same bundle can be grouped across queries for tracking and separation |

It also returns the isotropic remainder's fraction, the unstructured background. A class's ODF is the sum over that
class's instances of fraction x Bingham, so the fibre-only component of a voxel is a direct output. Context from the
wider region informs which objects are present in the window.

**Why Bingham per instance:** a single population within a window is close to unimodal. Bingham is exact for axial,
anisotropically dispersed lobes, is closed under the angular-smoothing knob, and is compact (5 numbers). Multi-lobed
structure comes from several instances, which is how crossings are represented and separated. For export, the class ODFs
are projected to log-SH (order 6-8 keeps 100 % of the readout; see `full_review_2026-10-28.md`) and to standard SH for
dMRI tools.

## Targets (from construction only)

For each query window: the instances whose material falls in the Gaussian window of width sigma. Each gets its instance
id, class, mass, and a Bingham fitted to the tangent or orientation distribution of its geometry within the window. The
existing `entity_targets` already computes per-entity Bingham fits. The targets carry **no visibility, presence or
classical-reader terms**. Whatever the image doesn't show, the model must express as uncertainty: broad lobes, a low
existence probability or a high isotropic fraction. Targets are smoothed with the same tau as the query.

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
