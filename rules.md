# ShapeTensor — design rules

The pipeline instructions and preferences Andrew has given (2026-10-15 to 2026-10-22). Three parts:

1. **What you asked for, word for word**: your messages and multiple-choice answers, quoted exactly (typos kept),
   grouped by topic, dated.
2. **My interpretation**: your aims and use cases.
3. **The implementation you want and its design**: each stage, with where the code on this branch stands
   against it.

When an instruction changes, add the new quote to part 1 and update parts 2 and 3. Newer instructions override older
ones where they conflict; the conflicts I know of are listed at the end of part 3.

---

## 1. What you asked for, word for word

### Aim and scope

- 2026-10-16: "complete an in depth review of the concept of using TCA on shape features to extract a multivector
  orientation field which accurately describes axonal bundles while avoiding boundary issues seen in structure tensor
  at the edges of large tracts and grey white boundaries. [...] Consider how we can extract N distinct populations
  without fixed k values, and how boundaries are extracted. come up with a recommended solution that creates the field
  that could be used as an input for BRIDGE to improve streamline outputs in HiP-CT data. Heavy focus on validity,
  proving validity in phantoms and using these to showcase where structure tensor fails. structure how this would be
  run as a package effectively and computationally efficiently (idealy GPU on the cluster)."
- 2026-10-16: "I know you dont think the space and single vector per divided populations cant work, but archive the
  ODF and fixels, and work with me on the core ideas i have proposed. Keep to my pipeline and working towards getting
  this to work, rather than changing the scope and shape of the project in ways i have not asked for"
- 2026-10-21: "Once the current registration and label work is done, begin the testing, validation and debugging of
  the new version of shape tensor 0.2, get to a solid working branch with relevant results. Also test resilience to
  artifacts like streaks"
- 2026-10-21: "Okay, push for a simplest fully correct pathway, as far as possible using classical mathematics rather
  than black box models beyond anatomix."

### The pipeline's stages

- 2026-10-16: "for the pipeline in the docs, add a subpage explaining each stage: getting the shape features,
  sperating in shape space, getting tensors from populations, and merging population vector fields"
- 2026-10-16: "see I thought it clustered fibre populations on the shape space then ran TCA once per population?"
- 2026-10-16: "Modular structure with changeable feature engine (compare options) and single Vs multitensor TCA
  approaches."
- 2026-10-20: "Is it possible to split the shape output, then tca per population? As a single population should have
  a single orientation. The issue of multiple orientations is how to split or particion the shape space into multiple
  spaces (not necessarily images or channels)"
- 2026-10-20: "How would it be most correct to split the space into a tensor per detected population as a formulation
  to pass to TCA or TNMF (not as an image channel, but the partioned raw shape space) based on the newest anatomix
  model"
- 2026-10-21: "Is population detection also local to a region around specific voxels? Im imagining a large processing
  volume wihr overlap, shifting window for anatomix and fibre detection, smaller local kernel for smoothing, then per
  voxel TCA to orientation. How are you suggesting smoothing is done? Does TCA allow this shape directed smoothing
  inheritly in the solving?"

### Populations: flexible N, sets of fibres, identities

- 2026-10-16: "i want to see if there is a more flexible method of separating the space which actually finds N
  populations passing through the voxel, not arbitatry rank R"
- 2026-10-16: "is there a shape derived method currently used in TCA which is flexible up to N reasonable populations
  of different shape?"
- 2026-10-16: "i dont want it to be just masking the image by population, I want to pull the shape information from
  this, I'm thinking this would appear more like gradients per population. it is seperate layers of information not
  dividing the same one by a mask. both should be full image volumes representing their respective populations shape"
- 2026-10-16: "how does this detect N fibre populations? test with 1,2,3 populations without specifying?"
- 2026-10-17: "why three populations here? i am unhappy that this single fibre population is being split, why is this
  and I dont like the arbitary choice of 3 populations, and that an N populations needs to be specified"
- 2026-10-17: "is there a formulation which pulls out a set of fibres from the images rather than portions of the
  image which hold a specific orientation? this is a useful phantom which identifies an issue"
- 2026-10-18: "i thin the gobal reader means that it cant hold a population that changes orientation within the space
  but locally is coherent"
- 2026-10-18: "continue developing ideas untill it can work with both straight and turning fibre popualtions"
- 2026-10-18: "one example shows separation of disconnected populations (90 degree second row), should there be a step
  to concatinate similar populations together? run futher tests and experiments untill working well in all phantoms.
  take as long as you need. once ready, test in real data of 20um cortex"
- 2026-10-21: "I do really need the flexible N population adaptive separation, ideally with some local momentum
  anisotropic smoothing. Make this a semi hard design requirement"
- 2026-10-21: "Is it possible instead of strongest and second population to show as population 1,2,3,... rather than
  per voxel rank, actually separate by fibre population?"

### Smoothing and momentum

- 2026-10-21: "How about anisotropic or shape aware smoothing of each population to extend their population shape
  across the crossing? Can this be done in shape space?"
- 2026-10-21 (multiple choice, momentum): "Tunable, use option 1. Maybe per population calibration (some populations
  are straight, some are bending)" — option 1's text was "0.5 mm, follows bends: Bridges about 0.5 mm along each
  population, with a small angular blur so curving populations are followed."

### TCA

- 2026-10-15: "i was reading about tensor component analysis (TCA) as a way of getting non-orthoginal components. Can
  i use TCA to get a more accurate lowest gradient direction rather than if it were constrained to orthoginal tensor
  axes?"
- 2026-10-15: "i want to run TCA to get orientation from the fibre-only shape tensor"
- 2026-10-16: "okay, keeping this as simple as possible, is it feasible for TCA from the shape to produce N fibres when
  these vary between populations with seperate orientations as defined by shape?"
- 2026-10-18: "can it be done with TCA from the activation populations as you were instructed to do?"
- 2026-10-18: "so what about TCA on the separated shape embedding space rather than on gradients of split images -
  since TCA can be done on non-square matrices? theory only"
- 2026-10-18: "TCA for denoising data - can this property of TCA be applied here?
  https://www.nature.com/articles/s41593-024-01626-2/figures/4"
- 2026-10-18: "okay - build and test a basic version of this with rank and scale parameters."
- 2026-10-20: "Run tests for the first formulation for TCA, avoid having to do any new training"
- 2026-10-21 (multiple choice, primary separation): "Lag-TCA primary (Recommended)" — the option's text was
  "Per-window TCA with a spatial-lag mode: N from rank selection, shape-directed smoothing inside the solve,
  orientation from each component's lag factor. The current orientation-distribution model stays as the comparison."

### Features (anatomix)

- 2026-10-15: "can window size be used comparable to rho? so if I had rho of 6, then a window mirrored to 12? then
  running structure tensor on the rho value on the fibre shape channels?"
- 2026-10-16: "okay, 16-D full model is more valid so we'll work with its output."
- 2026-10-16: "is there something you can do or finetune from earlier in the model which can encode population? is
  there a way of extracting this information from other part of the shape space or anatomix outputs?"
- 2026-10-20: "note that we will use dev-vit"
- 2026-10-21: "Also small QC fig comparing old and new anatomix models (just image, space, RGB channels) in hipct data
  voi"

### Classical mathematics, models and calibration

- 2026-10-21: "Okay, push for a simplest fully correct pathway, as far as possible using classical mathematics rather
  than black box models beyond anatomix."
- 2026-10-18: "Okay, the orientation readout should be pre-trained on some synthetic data like anatomix, not per
  volume."
- 2026-10-21 (multiple choice, synthetic directions as calibration targets): "Im not sure, préférences for self
  supervision, but test and compare both"
- 2026-10-21 (multiple choice, registered anatomical labels for presence): "No, stay label-free"
- 2026-10-15 (standing directions from the handoff): "go step by step, no k-means, ignore segmentation/mask labels, use
  the 20 µm S-21-46 crop, label prototypes as prototypes"
- 2026-10-20: "also can we use TNMF clustering on the shape space?"

### Tiling and artefacts

- 2026-10-15: "make sure you account for padding around the data, make phantom larger then crop orrientations before
  assessment"
- 2026-10-19: "can anatomix be run as a sphere, not a cube to reduce the snapping to axis in some of its outputs?"
- 2026-10-19: "what would be a fully correct solution to remove tiles"
- 2026-10-21: "Look at tiling as a priority, aim to remove edge and block effects, maybe spherical regions test too?"
- 2026-10-21: "Also test resilience to artifacts like streaks"

### Resolution and generality

- 2026-10-21 (multiple choice, output resolution): "Native resolution - this should be more generalised completely to
  any array data input. Given a chunk make one or many vectors per voxel at native input Res. Test at multiple
  resolutions"
- 2026-10-21 (multiple choice, N and angle): "Tunable parameters, use option 2 default" — option 2's text was "N ≤ 2,
  ≥ 30°: At most two populations, at least 30° apart: simpler, misses three-way crossings."
- 2026-10-17: "test at different hipct image scales (2,10,20,50,100,200) for 2,10 use crops from i74 VOIs.
  20,50,100,200 use crops of the deep brain from S-21-46."
- 2026-10-16: "okay - so add a docs page on non-isotropic data and a page on bigbrain."

### Weighting, masking and fallback

- 2026-10-16: "also how is it defining where to put vectors (by masking)? optional export of strongest population and
  union of all population masks (could be used instead of intensity threshold for defining node sampling mask in
  BRIDGE down the line - save this idea but only build mask export now)"
- 2026-10-17: "if no masking arg was used, in regions where masks intersect - use both, where one or no masks - use
  strongest activated population."
- 2026-10-21 (multiple choice, voxels matching nothing calibrated): "So I want a weighted, rather than thresholded
  approach.  Sensitivity weighted single vector field and then threshold multi vector (is there a way to include
  intensity correlation here?)"
- 2026-10-21: "It looks  like it is making small regions of background in the white matter mask? Is it pulling out
  multiple populations? Is population specific smoothing/momentum being done?"
- 2026-10-22: "so that missed so much of the white matter coverage"
- 2026-10-22: "what is the point of actually masking out here? is it just not to make background into orinetations?"

### Crossings

- 2026-10-17: "test also in crossing fibres where populations should be separated to check these methods still work"
- 2026-10-21 (multiple choice, real crossings at 20 µm): "Unknown, design for both (Recommended)" — the option's text
  was "Handle spatially interleaved fascicles (bridged by momentum) and voxel-level mixtures (resolved inside the
  solve)."

### Output, API and BRIDGE

- 2026-10-16: "also no mention of BRIDGE in package, also make output to volume similar to
  https://github.com/Skielex/structure-tensor so it could be subbed in where this is used in BRIDGE, then make sure it
  can be run GPU only (as if H100 cluster)"
- 2026-10-16: "why does ti output values? just vec = st.cp.shapetensor(image, kwargs (settings and modules)) make it
  simple and not copying the structure tensor exactly but jsut that it outputs the vector field as a cp or np array.
  just one simple codebox"
- 2026-10-16 (multiple choice, N populations in BRIDGE): "Co-located vertices (Recommended)"; (where the package
  lives): "Separate package + reader in BRIDGE (Recommended)"; (truth on real tissue): "Phantoms only".
- 2026-10-17: "Integrate into selecting verices. allow a kwarg to allow orientation=shape_tensor. Also allow
  mask=fibres (shape tensor mask), but default as intensity. Default structure tensor for now."
- 2026-10-17: "this looks very intersting. use intensity mask."
- 2026-10-21: "Option for full and fibre only or fast solve as pipeline arg"

### Validation and quality control

- 2026-10-16: "Heavy focus on validity, proving validity in phantoms and using these to showcase where structure
  tensor fails."
- 2026-10-21 (multiple choice, reference on real data): "No good metrics, visual QC, quantity in phantoms"
- 2026-10-21: "Once run, run side by side to standard structure tensor only and show in 20um cortex hipct."
- 2026-10-17: "can you make a phantom with a few fibres taking a 90 degree bend, then show how many and what activation
  populations look like"

### Figures

- 2026-10-17: "for all figures, like real s2146 and bigbrain, show image, st, shape, activation images, shape tensor
  single, multitensor"
- 2026-10-17: "Do not use brightness activation, show full shape tensor in comparison to unweighted structure tensor"
- 2026-10-17: "with the shape tensor rendering using opactiy with black baground, can you show some different rendering
  options? can you show with and without activation weighting (just orientation DEC)"
- 2026-10-21: "Can you show the current partitions the approaches draw in the latent space"
- 2026-10-21: "What about region (3D blobs) annotations in the shape space rather than partitioning that runs out the
  sides"
- 2026-10-21: "Show latwnt space umap for the last 3.layers of anatomix do show how populations are more or less
  detectable"

### Compute

- 2026-10-16: "whats making things run so slowly? structure tensor would be done very quickly"
- 2026-10-16: "the main work is anatomix, split, then TCA? none of these should be particularly slow?"
- 2026-10-21 (multiple choice, whole-brain budget): "Up to a day, everywhere"

### Working practice

- 2026-10-15: "Go step by step and show me results before moving on to the open threads at the end of the handoff."
- 2026-10-17: "prototype 1 and 2, compare in the phantom. work in shapetensor. include in docs only after we settle a
  solution"
- 2026-10-16: "name is ShapeTensor not tensors"
- 2026-10-21: "Also you're using the wrong GitHub, it should be Andrew-Keenlyside not Andrew-k-k also remove Claude
  contribution labels"
- 2026-10-21: "Continue tests for shape tensor, make attribution changes for shape tensor only"
- 2026-10-21: "I am okay with you building conda envs as long as they dont break any other envs"
- 2026-10-21: "Can you just install it? What other useful tools have you not installed? I can help"

---

## 2. My interpretation: your aims and use cases

**The aim.** A drop-in replacement for the structure tensor that gives BRIDGE a more faithful orientation field for
HiP-CT brain data. Two structure-tensor failures matter most:

- **Boundaries.** At the edges of large tracts and at the grey/white boundary, the structure tensor is dragged into the
  plane of the boundary. The shape features (anatomix) should tell a fibre population apart from a boundary.
- **Crossings and mixtures.** The structure tensor holds one direction per voxel, so where populations cross it reports
  their bisector. You want **one or many vectors per voxel**, the number found by the data, with no fixed k, no
  arbitrary rank and no k-means.

**What "a population" means to you.** A *set of fibres*: a bundle that keeps one identity along its length, through
bends and through the crossings it passes. It is not "the voxels that share an orientation". So:

- a bundle turning 90° is one population, not three;
- two disconnected pieces of the same bundle should be joined;
- two crossing bundles stay two populations through their overlap;
- populations are labelled 1, 2, 3, ... across space, not "strongest / second" per voxel.

**How you see the method.** Your mental model: anatomix maps the image into a shape space. In that space, fibre
populations are separable *as partitions of the raw shape space*: separate full-volume layers of shape information,
not masks over one image. Each population has a single orientation, which TCA reads out, and the populations' vector
fields are merged. Your expected processing layout:

- large processing volumes with overlap;
- a shifting window for anatomix and fibre detection;
- a smaller local kernel for smoothing;
- per-voxel TCA to orientation.

Each population is smoothed along its own shape ("momentum"), so it carries across a crossing or a gap but does not
run past a free end. Momentum should follow bends, and might be calibrated per population (straight vs bending).

**Classical maths, few black boxes.** anatomix is the one learned component you accept. Everything after it should be
classical and explainable (decompositions, statistics, geometry), not new networks trained per volume. Your wishes
pull two ways:

- per-volume self-calibration is out, and a synthetic pre-trained readout was suggested;
- but you prefer self-supervision and want both compared.

**Use cases.**

1. Whole-brain HiP-CT at 20 µm (S-21-46; the cohort at 50 µm), chunked, on the cluster's GPUs, within a day per brain,
   the full solve everywhere.
2. Any array at its native resolution: the same code at 2-200 µm, isotropic or not, BigBrain too. One or many vectors
   per voxel of the input.
3. BRIDGE input: a vector field with the structure tensor's call shape (`vec = shapetensor(image, **kwargs)`, a NumPy
   or CuPy array). Multiple populations enter BRIDGE as co-located vertices. BRIDGE itself is never mentioned in the
   package. A population mask might later replace the intensity mask for BRIDGE's node sampling (idea saved, not built).
4. Exploration and QC: figures that let you see each stage: image, structure tensor, shape space, activations / per-
   population layers, single and multi-vector fields, the latent space (UMAP, region blobs rather than partitions that
   run out the sides). Real data is judged visually; phantoms carry the numbers.

**What you want from me in the work.**

- Keep to your pipeline and make its stages work.
- When a stage fails, change it from inside; don't swap in a different estimator or widen the scope.
- Report negative results plainly.
- Test validity first, on phantoms that show where the structure tensor fails.
- Label prototypes as prototypes, and keep the docs for settled solutions.

---

## 3. The implementation you want and its design

### Requirements (semi-hard unless stated)

| # | Requirement | Default / tunable |
| --- | --- | --- |
| R1 | Flexible, adaptive N per voxel, chosen by the data (no fixed k, no arbitrary rank, no k-means) | N ≤ 2, ≥ 30° apart; tunable |
| R2 | Population-specific momentum: anisotropic smoothing along each population's own shape, bridging where it holds on both sides, never extending free ends, following bends | 0.5 mm; tunable; maybe per-population calibration |
| R3 | Populations are sets of fibres with identities across space (one label through bends and crossings) | — |
| R4 | Label-free on the analysed data: no anatomical labels or masks as method input | registered labels not used for presence |
| R5 | Weighted, not thresholded: a sensitivity-weighted single vector field everywhere in tissue; thresholds only for the multi-vector field; an intensity-correlation weight | — |
| R6 | Native resolution, general to any array chunk: one or many vectors per voxel at the input's resolution | tested at several resolutions |
| R7 | Classical maths after anatomix; anatomix-dev-vit is the feature model | — |
| R8 | No tile, block, edge or axis artefacts (overlap, spherical windows considered) | — |
| R9 | Robust to HiP-CT artefacts (streaks, rings) | — |
| R10 | Crossings: both spatially interleaved fascicles and voxel-level mixtures | — |
| R11 | A solve argument: full (everywhere) / fibre (population solve only where fibre-like, weighted single field everywhere) / fast (coarse) | full |
| R12 | GPU-only runnable; whole brain within a day; the structure tensor's call shape; no BRIDGE in the package | — |

### The pipeline as you designed it

1. **Shape features.** anatomix-dev-vit over overlapping windows (128³ tiles), blended so that no tile seams or patch
   grid remain. This is the "large processing volume with overlap, shifting window".
2. **Separation in shape space.** Detect the fibre populations present around each voxel and partition the *raw shape
   space* into one part per population. These are full-volume layers of shape information, not masks over the image.
   N is found per region and is not specified. Lag-TCA (TCA with a spatial-lag mode) is the chosen primary method:
   - rank selection gives N;
   - the lag factor gives each population's orientation and its own smoothing kernel;
   - the orientation-distribution model is the comparison.
3. **Tensor / orientation per population.** One orientation per population, by TCA on that population's part of the
   shape space (not on gradients of masked images), per voxel.
4. **Momentum.** Smooth each population along its own shape with a small local kernel. Bridge across crossings and
   gaps where the population holds on both sides; follow bends; never extend a free end.
5. **Merge.** Two fields from the populations:
   - a sensitivity-weighted single vector field everywhere;
   - a multi-vector field, thresholded, with population identities 1, 2, 3, ...
   - Where populations intersect, use all of them; where there is one or none, use the strongest.
   - The weights include intensity correlation.
6. **Outputs.** `vec = shapetensor(image, **kwargs)` as a NumPy/CuPy array. Optional:
   - the multi-vector field with per-population weights and labels;
   - the strongest-population map;
   - the union of population masks.
   An OME-Zarr writer for whole volumes; a solve argument full / fibre / fast.

### Where this branch stands (2026-10-22, after the overnight build)

| Stage | Built | Against your design |
| --- | --- | --- |
| Lift (shape space) | `lift.py`: the array lifted to positions × directions by its local lagged autocorrelation on an integer-lag shell, on a lattice; null from the block's isotropic surrogate; response from the block itself; excision inside the correlation | The lift is the classical "shape space": it holds any number of populations and the structure tensor is its zero-lag moment. anatomix is not in the chain (optional engine for later presence weighting); R7 read as "classical after the embedding, embedding optional" |
| Separation | `populations.peaks` after deconvolution: N per position from the data, N ≤ 2 and ≥ 30° by default (R1) | Meets R1; intermixed 90° crossings lose a third of cells to low presence; 45° unresolved |
| Momentum | `populations.momentum`: along each direction, bridge where both sides hold, never extend a free end; 0.5 mm at 20 µm (R2) | Meets R2; per-population straight/bending calibration not built; the tightest bend (0.3 mm radius) is partly lost |
| Identities | `populations.label_populations` + `volume.stitch_labels` across chunks (R3) | Meets R3 (one label through a bend, two through a crossing; leakage 4–15%) |
| Weighting | presence weight (N-independent excess over the median) from the surrogate floor; single field never cut; multi-vector field thresholded (R5) | Meets R5 except the intensity-correlation weight (not built) |
| Native resolution, generality | `field.py`: the field at every input voxel; `orientations(spacing=)` for anisotropic voxels; channels add; tested in unit tests (R6) | Meets R6 for 3-D; 2-D arrays not yet; multi-resolution phantom scores not yet |
| Volume driver | `volume.run_volume`: chunks with a halo that reproduce a one-pass result exactly (shared floor and response), multi-process, label stitching (R12) | Meets R12; a solve argument full/fibre/fast (R11) not built |
| Tracking (the aim) | `validate.py`: through-crossing on its own band 0.64/0.81/0.68 at 90/60/45° (structure tensor 0.07/0.42/0.40); T and free end never extended | Below the 0.9 target but the structure tensor's switching is removed; see docs/validation.md |
| Artefacts | not re-tested for the new chain (R9 open) | — |

### Conflicts between instructions (for you to settle)

1. **ODF / fixels archived (2026-10-16) vs the classical pathway (2026-10-21).** Resolved for now as: the lift is the
   shape space the design asked for (positions × directions, holding N populations), the archived route was the
   gradient-direction ODF with fixels as the output primitive; here the primitive is a population with an identity and a
   carry-forward. The 0.3 README and docs describe it so. Your confirmation is still wanted.
2. **anatomix as the feature space vs the chain reading intensity.** anatomix plays no part in the 0.3 chain; the
   candidate role that remains is a presence / class weight (fibre vs vessel vs boundary) from synthetic references,
   pluggable as an engine. Not built overnight.
3. **Pre-trained readout (2026-10-18) vs preference for self-supervision (2026-10-21).** Read as: test and compare
   both; no per-volume training.
4. **Step by step (2026-10-15) vs "run futher tests and experiments untill working well in all phantoms. take as long
   as you need" (2026-10-18).** Read as: run complete test suites autonomously, show results at decision points.

### Standing practice

- Commit as Andrew-Keenlyside (noreply); no Claude attribution lines; history rewrites only where you authorise them
  (done for ShapeTensor only).
- Install into the shapetensors env, or build new conda envs, only without breaking other envs; dry-run first.
- Prototypes are labelled as prototypes; the docs carry settled solutions only; figures and results go in the results
  log and changelog.
- Real data is judged by your visual QC beside the standard structure tensor (S-21-46 20 µm cortex VOI); numbers come
  from phantoms. Each phantom has margins and the core is cropped before scoring.
