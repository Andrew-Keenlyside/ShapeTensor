# ShapeTensor as a foundation model for biological orientations

Branch `population-model`, planned 2026-10-22/23. Supersedes `plan_population_model.md` where they differ (that file
records how we got here: the population model, two adversarial reviews, stages 1-5).

## 1. Goal

One pretrained model that describes the **populations of shapes** around any point of any 3-D image -- fibres,
sheets, membranes, boundaries, blobs, vessels, at any scale, through crossings -- in one continuous shape space, for
any modality (HiP-CT, lab CT, MRI, histology, and more as the generator grows). Nothing is categorised by the model:
every population is described by continuous quantities, and the user selects what to extract at inference. Every
output is computed from the model's output; the image is the model's input only.

The model is a **queryable field**:

    populations(x, σ_A, B)

the populations within a Gaussian window of size `σ_A` centred at any point `x` (continuous: any output grid,
including sub-voxel spacing), using image context up to `B`. `σ_A` (what is described) and `B` (what is looked at)
are inputs, not constants; beyond the trained ranges the model extrapolates and degrades.

## 2. The description of a population (per slot, up to K per query)

| quantity | meaning | source in the synthesis |
|---|---|---|
| shape tensor `S` (trace 1) | direction and shape: linear / planar / isotropic as continuous Westin shares | the rendered shapes' tensors (`t tᵀ`, `(I - n nᵀ)/2`, blob moments), pooled |
| mass `m` | the population's visible volume fraction in the window (a boundary: its visible step weight) | occupancy × presence visibility, pooled |
| presence | is this slot a population at all | mass above a floor |
| radius `r` | cross-section scale (voxels) | each layer's radius × the warp's local stretch, mass-weighted |
| edge / ridge | a step between regions (boundary) vs a thin structure (fibre, membrane, wall) | interface layers vs shape layers |
| identity embedding | which population this is, across the volume | contrastive: same generating layer = same population |

No kind labels. The cell scalars kept: the shape fraction. Orientation visibility enters through `S` (blended
towards isotropic where orientation is not recoverable: "present, direction unknown").

## 3. Selection and scale at inference

- **Predicates** pick populations: e.g. fibres = linearity >= 0.5 and radius <= 3 voxels; large tubes = linearity
  >= 0.5 and radius > 3; membranes = planarity >= 0.5 and ridge; boundaries = edge. Sub-populations = clusters of the
  slots' descriptors (tensor shape, radius, edge, embedding) found after the fact (mixture / density clustering).
- **Description scale `σ_A`**: a model input. Coarsening an existing output is also exact by **shape-space weighting**:
  `m`, `m·S`, `m·r` are additive, so Gaussian-weighting them per population (matched across cells by axis and
  identity, never mixed) by `sqrt(σ_out² - σ_A²)` equals querying at `σ_out`.
- **Context `B`**: a model input (encoder depth gating, below).
- **Output spacing**: any; default `σ_A / 2` (Nyquist for the scale).
- Then, on the selected populations: carry-forward along each population (walks that follow it cell by cell),
  bridging where the same population arrives from both sides, the N heatmap, identities, the voxel field.

## 4. Architecture

- **Encoder**: fully convolutional feature hierarchy, strides 1, 2, 4, ..., 32 (context to several hundred voxels),
  no normalisation layers (chunk-exact), anti-aliased downsampling (even binomial, cells centred at the targets'
  centres), a top-down path whose links are multiplied by gates `g_l(B)` (1 where level l's view fits within B, 0
  beyond); levels gated off are not computed. Trained with random B so any truncated hierarchy works.
- **Query decoder**: at a continuous point x, the active levels' features interpolated there, a positional encoding of
  x's offset inside the finest voxel (sub-voxel queries), FiLM by `log σ_A` and `log B`, an MLP to K slots (`S` from
  6 numbers as `L Lᵀ / tr`, `√m`, presence logit, `log r`, edge logit, embedding) and the cell scalars.
- Size: 5-30 M parameters (today's fixed-lattice network: 0.37 M).

## 5. Training data and targets

- **Label scale-space**: per sample, each layer's additive quantities (`w`, `w·S`, `w·r`, `w·edge`) blurred at
  log-spaced σ (0.5-16 voxels); targets at random query points and random `σ_A` by trilinear interpolation in space
  and log σ. Slots per query: layers merged where they are the same population *to the image* (orientation and radius
  within resolvable differences), K = 6, sorted by mass.
- **Supersampled rendering** (2-4×) then the acquisition to the input grid: labels exist below the voxel, so
  `σ_A < 1` has a defined target.
- **Visibility measured over the context B** (the local t-statistic of each layer's fit in the image, already built for
  a fixed window): presence (t 3 -> 8) scales mass, orientation (t 6 -> 18, to be re-measured against the trained
  model's own error) blends `S` towards isotropic. Sub-voxel orientation is asked for exactly when the context makes it
  recoverable.
- **Losses**: exact slot matching (all K! permutations, or a Hungarian solver at K > 6) on tensor (per-degree balanced
  across shapes), √mass, presence BCE, log-radius L1, edge BCE; Hellinger on the rendered densities; contrastive
  identity with batch-wide negatives; fraction BCE.
- **Curriculum**: nuisance strength 0.15 -> 1 over the first half; slot terms first, the rest ramped in; random
  `σ_A`, `B`, query positions, cube-group symmetries, intensity-window jitter.

## 6. The generator's breadth (labels come free: they accumulate from what is rendered)

Built: lines, sheets (discs), blobs, hollow tubes, crossing groups (companion or independent fields, shared volume
fraction, interdigitated), arc bends, region / tissue / cell-stain backgrounds with laminae, interfaces; HiP-CT, CT,
MRI, histology acquisition profiles with cross-mixing and a strength knob.

To add, in order:
1. **Multiscale fascicles**: thick bundles made of thin fibres (the radius-selection and sub-population test).
2. **Branching trees** (vessels, dendrites): bifurcations, tapering radii, tortuosity.
3. **Cells with processes** (soma + lines: neurons, glia).
4. **Helices and twisted bundles** (collagen, tendon, muscle) and periodic striations (sarcomeres).
5. **Weaves and meshes**; **foams and curved sheets** (Voronoi cell walls, shells); **lamellae** (stacked,
   concentric).
6. **Oriented textures** (statistical anisotropy without discrete shapes).
7. **Real arrangements**: real segmentations (BigBrain layers, SynthSeg labels) as regions; real orientation fields
   (structure tensors of HiP-CT, dMRI) as generating fields.
8. **Modalities**: EM, light-sheet (depth attenuation, striping, detection PSF), confocal (bleaching), OCT /
   ultrasound (speckle), polarised light.
9. **Physics**: supersampled partial volume, defocus PSFs, Paganin phase retrieval.

Throughput: cache clean samples and redraw nuisances only; generate in a separate process; on the cluster, generate on
one GPU while training on another.

## 7. Blockers (23 Oct)

1. **Training collapses to the dataset mean** on diverse data: the fixed-lattice network fits 8 fixed samples
   (fraction correlation 0.75, shape error 0.74 -> 0.09) but stays on the mean for 64 samples over 1000 steps.
   Must be solved first (T1).
2. Targets exist only for the fixed design (lattice, σ_w = 5, categorical kinds): the label scale-space, random
   queries, supersampling, continuous descriptors and B-limited visibility are to be built.
3. The queryable model (gated encoder, query decoder) is not built.
4. Generator throughput (~1 s per 96³ sample; supersampling multiplies it).
5. No real-data checks set up (HiP-CT nested zooms, a VOI benchmark for the model).
6. H100 training needs explicit permission on the cluster.
7. Open calibrations: the orientation-visibility range; the bridge's 2-4% spurious insertions (against an oracle).

## 8. Feasibility tests (toys: A6000, 64³, 1-3 h each)

| | question | setup | pass |
|---|---|---|---|
| T1 | can a network learn populations at all? | today's model, a ladder: one clean population -> + nuisances -> two crossing -> full generator | axis error near the target's ceiling at each rung, no collapse; find where it breaks |
| T2 | σ_A as an input | random σ_A, clean data | matches the label scale-space at every σ_A; coarsening a fine output = querying coarse |
| T3 | B as an input | gap-bridging and crossing-vs-bend tasks needing context | accuracy rises with B to the task's need, degrades smoothly below |
| T4 | sub-voxel | supersampled thin fibres, σ_A < 1, small vs large B | long straight fibres recovered with large B; texture reported not oriented |
| T5 | continuous queries | random sub-cell offsets | smooth, no lattice-phase artefacts |
| T6 | sub-populations | one orientation field, radii 1 and 4 (then spacing) | two slots with the right radii; embeddings cluster by layer |

## 9. Validation beyond the synthesis

- **HiP-CT nested zooms** (the same tissue at e.g. 20 µm and 2.5-6 µm, registered): sub-voxel orientation from the
  coarse scan against the zoom; also a training signal.
- **Polarised-light imaging** sections: fibre orientation ground truth.
- **Cross-modality pairs** (BigBrain histology <-> MRI; HiP-CT <-> MRI): the same populations.
- The phantom and tracking suite of the classical chain (`shapetensor validate`), and the real VOI against the
  structure tensor at confident voxels.
- **v2 consistency training on unlabelled real data**: rotating the volume rotates the output; coarsening equals
  querying coarser; two acquisitions of one tissue agree.

## 10. Downstream uses

**Tractography.** The populations parameterise it directly:
- sampling masks and densities from selected populations' masses (per sub-population cluster if wanted);
- one node / seed per population (co-located), so tracking passes through crossings;
- direction confidence from the population's coherence and orientation visibility instead of the structure tensor's
  FA (low at every crossing);
- angle and bend limits per node from coherence (tight where straight, looser in bends), step and edge lengths from
  radius and σ_A;
- identity labels/embeddings keep edges within a population and match fragments across chunks;
- the bridge marks where a population continues through a dropout;
- interface slots give cortical entry points and angles (endpoint assignment);
- streamline tools: export as fixels (directions + masses) or as an FOD (the rendered density with absolute masses):
  mass-proportional seeding, SIFT-style weighting, a tissue map (line volume / boundary / isotropic) for anatomical
  constraints, per-population dispersion for probabilistic tracking.
For graph tractography (BRIDGE) the full benefit needs multi-population vertices (today a vertex holds one merged
direction); the masks and per-node parameters fit the current design.

**Registration (anatomix-style feature registration).** Contrast-invariant like anatomix's features, and adding what
they lack -- orientation structure: rotation-invariant channels (fraction, radius, Westin shares, N) as images for
feature-based registration (cc / mi / mse losses); orientation-aware similarity with tensors reoriented by the warp
(`J⁻¹ S J⁻ᵀ`, as the generator does) as in DTI / fixel registration -- aligning tracts and laminae across modalities
whose intensities disagree; anatomix features for the coarse, global alignment, these for the fine refinement.

## 11. Staging

1. **T1** (resolve the collapse) -> the flexible target side -> the gated encoder and query decoder -> **T2-T6**;
   generator breadth phase 1 and throughput in parallel.
2. **v1**: the queryable model on the current synthetic world, validated on HiP-CT nested zooms and the tracking
   suite (serves tractography through crossings).
3. **v2**: consistency training on real unlabelled volumes.
4. **v3**: the broad grammar and more modalities; the latent exposed as orientation-aware features.

## 12. The population density as the core object (overnight 23 Oct)

The slots answer "which populations, with which descriptors", but a set output needs a matching, a cap K, a merge rule
and an order -- each a source of conflicting targets (the full-generator failure: slot ordering and a presence guess
dominated the gradient). The object that needs none of these is the **population density**:

    f(u; x, sigma, B) = sum_L m_L(x, sigma) w(u | S_L(x, sigma))

over the rendered shape layers L, with m_L = G_sigma * (o_L v_L) the windowed visible mass and m_L S_L = G_sigma * (o_L
v_L T_L) the windowed mass x tensor; w the angular density of a tensor (a line a lobe, a sheet a girdle, a solid its
second moment's quadratic form, a blob flat). Properties:

* **any shape**: every layer contributes through its tensor; no kinds. A hollow fibre, a ribbon, a crystal and a sheet
  differ in w, not in a label. The number of populations is unbounded; crossings are peaks.
* **any description scale sigma**: linear in the windowed fields, so a coarser sigma is exactly a Gaussian of the
  finer one, and a query at any sigma has an exact target (`learn/query.py`).
* **any position x**: the windowed fields are continuous; sub-voxel queries are exact samples of them.
* **in even SH** (orders 0..8: 45 numbers): order 0 is the mass, order 2 the mixture tensor (the mean tensor head), 4
  and up separate crossings from sheets (Advani & Tucker's 4th-order orientation tensor). It is what fibre-orientation
  networks regress (FOD-Net) and what tractography consumes.
* **descriptors without slots**: any per-layer descriptor q (thickness, hollowness, angularity, contrast) gives a
  descriptor density f_q(u) = sum_L m_L q_L w_L(u); at a peak direction q(u) = f_q(u) / f(u) is the descriptor of the
  population pointing there. Selection by predicates becomes thresholds on these fields.
* **context B**: the trunk's receptive field. Train with random input crops of side B around the queries (B as a
  decoder input), so a query states how much context it may use; beyond the trained B, expect degradation.
* **input resolution**: sigma and B in physical units (mm), with the voxel size as an input; the generator already
  draws physical scale (`scale`); resample-to-train-grid augmentation (SynthSeg's recipe) covers acquisitions.

Slots remain as a readout (peaks of f, with descriptor ratios at the peaks), trained after the density, not before it.

### 12.1 Built (overnight 23 Oct) and the evidence so far

* `learn/sh.py`: real even SH (orthonormal), the slots' density in SH, peaks (padded neighbour lists, chunked).
* `learn/query.py`:
  * `voxel_density` -- the exact per-voxel density: each voxel's own label tensor's lobe, weighted by its visible
    occupancy, summed over layers (0.28 s per 64³ sample, order 8); a Gaussian window of any sigma on it is the
    window-averaged density of the orientations present. Against the per-layer mean-tensor form (`layer_density`):
    the same mass to 4e-4, top peaks within 6.7 deg, and two peaks at 45% of points vs 29% (bends and branches are no
    longer averaged into one broadened lobe).
  * descriptor densities (`desc_lmax`): f_q = sum o v q w for thickness, hollowness, angularity; at the top peak
    f_q(u) / f(u) recovers a single layer's thickness (test).
  * `QueryDecoder` (LIIF): the density (and descriptor densities) at continuous points and scales, from the three
    feature levels; cube-exact augmentation of query points and SH coefficients (block-diagonal rotations).
* Training options: `--sh-lmax`, `--exact`, `--desc-lmax`, `--query N`, `--order salience`, `--simple/--p-simple`
  (curriculum), `--lcn`, `--presence-band`, `--weights`, `--init`.
* Evaluation under one reference matching rule on rotated held-out draws, single vs shared line cells, against a
  structure tensor; the density's nearest-peak error; descriptors at the target's top peak.
* Evidence (corrected by reviews 2-7): runs trained from scratch on the full generator collapse to round tensors
  (the "round plateau") whatever the width or schedule (R4, W0, N1, D1); easy line-rich draws in the mix break it for
  the wide network (W2: bank single 21.7 / shared 35.0, structure tensor 15.2 / 35.1; P1@2000 15.6 / 32.7). With the
  same curriculum the narrow network did not improve on narrow runs without one (C1@1000 = R4@2000 within the
  intervals): width matters with the curriculum, the narrow net's limit under the joint objective is unresolved (one
  seed). Slots still hedge to round tensors and copy the first axis at crossings; the density head has not yet been
  trained past the plateau. Default width is now 32/64/128.
