# The entity-field model: objects, their populations, and ODFs at any point and scale

Branch `population-model`, planned 2026-10-24 (replaces the lobe head of `plan_population_model.md`). Prototype until the
pilot and the acceptance tests below pass.

## 1. Goal

From one inference, at **any point x** of a volume and **any scale σ**, the model returns the **entities** passing there,
each with its own orientation distribution, and how they group:

| level | what it is | example |
|---|---|---|
| object | one drawn object that the resolution separates | an axon, a vessel segment's tree, a cell's membrane, a sheet |
| sub-population … population | the object's ancestors, **any depth** | fascicle → nerve; subtree → vessel tree; lamella group → plywood |
| overall | everything at (x, σ) | the voxel's total ODF |

Each entity carries a **Bingham ODF** (rendered to spherical harmonics at any order on demand), a **mass** (its visible
share), **shape descriptors** (size, wall, lumen, flattening, angularity) and its **place in the tree**. The overall ODF is
the sum of the entity ODFs. Everything ODFs offer for tuning stays post hoc: aggregation level, scale σ, mass threshold,
sharpening, SH order — all from the same forward pass. Objects can be **followed across context windows** (§6).

Why: the gap probe (2026-10-24) showed crossings cannot be recovered reliably by finding peaks in one ODF per cell —
8th-order targets merge crossings below ~30°, the calibrated peak gate drops more than half of the 60–90° crossings the
targets contain, and the lobes were never told which population is which. Separating **entities** first turns crossing
separation into "two entities share this voxel", exact at any angle the image supports.

## 2. Representation

**Entity ODF — Bingham.** `p(u) ∝ exp(uᵀ R diag(−κ₁, −κ₂, 0) Rᵀ u)`, `κ₁ ≥ κ₂ ≥ 0`, mode axis `R[:, 2]`. A line
population is a peaked Bingham (κ₁ ≈ κ₂ large), a fan an elongated one (κ₁ ≫ κ₂), a sheet a girdle (κ₂ ≈ 0, κ₁ large),
isotropic content flat (κ ≈ 0). The **target** for an entity is moment-matched: its pooled orientation second moment
`T = Σ w·t tᵀ / Σ w` (the shape tensors the generator already draws) has eigenvectors `R` and eigenvalues mapped to (κ₁,
κ₂) through a precomputed table — the maximum-entropy distribution with that second moment. SH rendering by quadrature
on a fixed spherical grid, cached by (κ₁, κ₂).

**Entities are resolution-aware.** An object is its own entity at (x, σ) when its radius is at least `max(r_min, c·σ)`;
otherwise it merges into its parent, repeatedly (the identity target's rule, extended with σ). At coarse σ objects become
sub-populations, then populations; at fine σ in resolved data they stay single objects.

**The tree.** Every drawn object has a parent chain ending at its population node, then the scene root — any depth,
defined for every family (fibre → fascicle → bundle; segment → tree; wall → cell → tissue; lamella → stack …). Tree
distance between entities is the supervision for the hierarchy.

## 3. The model

```
image (1, Z, Y, X) ─► CNN trunk (as now: stride-2 level, FiLM by voxel size, context level)
                        │
                        ├─► total-ODF head: SH order 8 per lattice cell (fast overall ODF, consistency)
                        │
                        └─► entity decoder D(features(x), log σ, Y(u)) at any point x, scale σ, direction u:
                               f   ≥ 0      density of material along u at (x, σ)
                               z   ∈ 𝔻ᴱ     hyperbolic (Poincaré) embedding of the entity owning u — its place in the tree
                               b            Bingham vote of that entity (6-D rotation → R, log κ₁, log κ₂)
                               d            its shape descriptors
```

`features(x)` is trilinear from the trunk's levels; `Y(u)` the even SH basis to order 8 (45 numbers) — directions enter
smoothly and rotate covariantly. **No fixed number of entities**: entities are clusters of (u, z) among the directions
with density; crossings separate by direction, parallel neighbours by embedding. The lobe head is removed.

## 4. Training

**Queries.** Per crop, Q points (≈ 512) with σ log-uniform in 1–8 voxels; per point D directions (≈ 32): half near the
entities' mode axes (jittered), half uniform. Targets come from the generator at those points (§5).

**Losses.**
- density: `f` against the target mixture `Σₑ mₑ pₑ(u)` (log-L1 plus the integral over the sphere = total mass);
- tree: hyperbolic distortion loss — `d_𝔻(zᵢ, zⱼ)` against the tree distance of the owning entities (0 within an
  entity), plus a margin term between unrelated entities; ancestors toward the origin;
- Bingham vote: on directions an entity owns (its share of the density > ½): an axis-invariant frame loss and log-κ L1;
- descriptors: on owned directions;
- total ODF: the SH head against the target total at lattice cells (order 8), and the decoder's integral against it;
- **across windows**: overlapping crop pairs of one scene — the same entity's embedding must agree in the overlap;
  along an entity's course, neighbouring queries' embeddings must stay close (continuity);
- everything weighted by visibility (the targets ask only for what shows).

**Curriculum.** Coarse σ (populations) first, then fine σ (objects); the crossing mode's share raised as it learns.

## 5. Generator changes

1. **Per-object orientation per voxel**: the canvas accumulates, per (voxel, object), the mass and the summed shape tensor
   (beside the identity keys, compacted the same way), so entity tensors are exact sums of their objects'.
2. **The tree for every family** (§2), with object radii; population nodes per group; a scene root.
3. **Entity targets at query points**: the window at (x, σ) gathers the (voxel, object) entries, groups them by the
   deepest resolvable ancestor, and returns each entity's mass, tensor (→ Bingham), descriptors and tree node.
   Warps: queries are taken in the output space and pulled through the warp grid, tensors rotated by its Jacobian.
4. **Visibility per entity** from its population's visibility map.
5. **Crossing mode**: crossings centred in the crop; angle uniform 5–90°; 2–3 populations; share 20/80–50/50;
   interdigitated, intermingled at sub-voxel scale, kissing, line across sheet; equal or different contrast; radius
   sub-voxel to resolved; **hard negatives** (bend inside the window, fan, dispersion, branch) labelled as one entity.
6. **Overlapping crop pairs** (shifted, same scene, same acquisition or another) for the cross-window losses.
7. **Speed**: vectorised identity lookup; target gathering on the GPU; then profiling.

## 6. Inference

1. Evaluate the decoder at the chosen grid (any spacing) and σ, on a fixed spherical grid of directions (≈ 300).
2. Per point: drop low density; cluster the remaining (u, z) (direction proximity + hyperbolic distance) → entities;
   each entity's Bingham is the robust mean of its votes, its mass the integral of its density.
3. Hierarchy: agglomerative clustering in hyperbolic space → the tree; entities at a chosen depth = sub/full populations.
4. Large volumes: overlapping windows; entities matched across overlaps (shared voxels, axis, embedding, descriptors) and
   merged by union-find → global object ids. **Seeded following**: from an entity, step along its axis, re-query the
   decoder, keep the continuing entity.
5. Outputs: per-point entity lists; per-entity ODF fields (SH, any order) by object / any tree depth; the total ODF.

## 7. Tests and acceptance

Unit (generator and targets):
- per-object tensors summed over objects reproduce the population tensors exactly; entities sum to the total;
- Bingham moment matching round-trips (sample → moments → fit → moments);
- crossings of two bundles at 5–90°: two entities with axes within 1°, at every angle; bends, fans, branches: one;
- resolution merging: an object below c·σ merges into its parent, repeatedly;
- the tree: distances match the drawn parents; every object reaches a population node;
- warps: entity axes rotate with the Jacobian; overlapping crop pairs share entity ids in the overlap.

Model: shapes; gradients; rotation covariance of `D` (rotate the input by a cube symmetry → outputs rotate); overfit
of a few crops (all losses → near zero).

Benchmarks (fixed seeds, new generator), per angle bin, population count, contrast, SNR, voxel size, modality:
- entity recall/precision (second and third entities), axis error, false splits on bends/fans;
- tree: grouping quality (adjusted Rand) at each depth;
- following: split/merge rates of stitched objects over chained windows;
- total ODF error against the lobe model's (no regression).

Acceptance for the cluster run: second entity found in ≥ 90 % of cells at ≥ 30° and ≥ 70 % at 15–30°, pair axis error
< 10°, false second entity < 10 % on singles/bends, cross-window object identity kept in ≥ 90 % of steps.

Real data — checks only, never training: rotation/contrast invariance of entities on the sense-check set; HiP-CT
crossing regions inspected per entity.

## 8. Reviews and schedule

Independent adversarial reviews (robustness: artefacts, contrast, noise, resolution; generalisability: modalities,
scales, morphologies) after the generator targets, after the first pilot, and before the cluster run.

| step | content | target |
|---|---|---|
| 1 | generator: per-object tensors, tree, entity targets, crossing mode, crop pairs + unit tests | day 0–1 |
| 2 | model: entity decoder, losses, training loop, inference clustering + tests | day 1 |
| 3 | very small pilot (64³ locally), benchmarks, review | day 1–2 |
| 4 | speed: generator and training profiling | after correctness |
| 5 | cluster run at 128³ (H100) — with the user's go-ahead to stage and submit | after acceptance |

## 9. Theoretical bounds and limits (2026-10-24)

Numbers for the current trunk: receptive field 146 voxels (54 without the stride-8 context level; an earlier
measurement on a 96³ input clipped it to 96), sigma trained 1–4 voxels (1–6 with ``--e-sigma``), ``emax`` 8.

**Set by the image (no model removes them).**
- Orientation lives only in spatial pattern: populations mixed below the blur scale carry no orientation in the
  intensity. The targets blend each layer's tensor toward isotropic by its orientation visibility (entities and the
  total alike), so the model is not asked for what does not show.
- Uniform material farther than ~RF/2 (~73 voxels) from any edge or texture has no measurable orientation (the
  effective field, where the weights are, is smaller: to be measured on the pilot).
- Crossing vs fan vs split vs kiss needs both sides of the overlap in view: bundles of radius R overlap over ~2R/sin θ,
  so below θ ≈ asin(2R/RF) (≈ 4° at R = 5, 8° at R = 10, 16° at R = 20 voxels) the entity count is the generator's
  prior (fans one entity, crossings two). Resolved fibres inside the bundles lower this to a few degrees.
- Partial volume: below resolution, volume fraction, size and contrast are confounded (training is contrast-agnostic);
  absolute masses and sizes are priors, shares between entities better determined.
- No polarity: Bingham ODFs are antipodal; branch / end points come from the points head.

**Set by the representation.**
- One entity per direction at a (x, sigma): co-directional entities at one point and scale (intermixed parallel
  populations differing only in contrast or calibre) merge there and separate only at finer sigma or other x.
  (Removable with two votes per direction; not done unless the pilot shows the need.)
- Separability scales with angle over dispersion: votes are trained where one entity owns > 1/2 of the density.
- An entity's Bingham is its second moment: an arc (a bend in the window) or two arms (a branch) are smeared into one
  lobe. The density f(u) is free; at inference the entity ODF is also given non-parametrically (f over the entity's
  directions), the Bingham as its summary.
- Translation equivariance (no coordinates, no norms: no block artefacts) gives identical-looking objects in a regular
  arrangement the same embedding: z encodes structure (population, shared ancestors); single look-alike objects are
  told apart by position (spatial clustering, seeded following).
- z is defined per window up to an isometry; the overlap and continuity losses align neighbouring windows; whole-volume
  identities come from linking, whose errors compound (90 % per step -> ~35 % intact over 10 windows): linking must be
  conservative (union-find merges are irreversible): links are mutual-best, never join objects seen apart at one point
  (cannot-link), never join different tree depths (a parent between two parallel children cannot chain them).
- Tree levels defined beyond the receptive field (which tract) are not locally inferable; two tracts identical in
  appearance and orientation share a parent in z. Tree edges have hyperbolic length 0.6 and the ball's cap is 1e-4:
  ~16 levels fit (the deep-tree generator reaches 10). The embedding's distance from the centre is trained to equal
  the entity's depth (root at the centre), so depth is readable at inference.
- The resolution rule was radius-only (radius >= max(1.5, 0.5 sigma)): isolated thin objects (sparse labelling,
  light-sheet axons) merged into their population although traceable. Now (2026-10-24): an entity holding a single
  object in its window is that object; objects merge only where they share the window with siblings.
- At most ``emax`` = 8 entities per query in training (1 % of queries reach it); sharpest lobe ~1.6° (kappa <= 600);
  sigma beyond the trained range is aggregated post hoc, not predicted.

**Set by the architecture.**
- Directions enter as order-8 SH features: switches between entities closer than ~10–15° are soft (spectral bias);
  measured per angle bin in the pilot; Fourier direction features if needed.
- No rotation equivariance (augmentation only): residual lattice bias, larger with anisotropic voxels.
- Stride-8 levels: exact shift equivariance only for shifts by multiples of 8 (no seams). The downsampling is already
  max-blur pooling (binomial anti-aliasing), so the remaining period-8 dependence is small; measured by the sub-stride
  shift benchmark.
- Exact tiling needs a halo of RF/2 + a lattice corner = 80 voxels and tile origins on multiples of 8 (inputs padded
  to multiples of 8): inference tiles of 256³ cores (416³ inputs) or more; a measured effective halo can trade a
  bounded error for speed.
- Training crops (64³ locally, 128³ planned) are smaller than the receptive field: no training query is free of the
  crop's padding, while interior points of a large volume are. To measure on the pilot (the gradient's support beyond
  32 / 48 / 64 voxels); if it matters, part of the cluster run at 160–192³ with queries placed centrally.

**Inputs.** Training windows each crop's intensities; at inference one window for the whole volume (``normalise``) and
the local-contrast input channel (``--lcn``, translation-equivariant) -- a per-tile window would make blocks. Voxel
size is given in µm (``voxel_um``); anisotropic volumes are resampled to isotropic voxels first.

**Set by synthetic training.** Objects, populations and the fan convention are the generator's; outputs are point
estimates (kappa mixes dispersion and uncertainty); large-scale coherence remains the generator's weakest area.

**Inference cost.** Points x directions: ~5 min per 1000³ x 300 directions on an H100; whole brains on coarser grids or
with directions chosen from the total ODF's peaks. Entity counts come from clustering thresholds (inference-time knobs).
