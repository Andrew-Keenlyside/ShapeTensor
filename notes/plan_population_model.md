# The population model: a shape → populations network for ShapeTensor

Branch `population-model`, planned 2026-10-22. A network trained on synthesised fibre volumes to output, at every lattice
position, the populations passing through it — their directions, fractions, coherence — and a tissue class. It replaces
one stage of the 0.3 chain, the lift; everything after it (lobes with `nmax` / `min_sep_deg`, carry-forward, identities,
the field at every voxel, the tracking suite) stays as it is and is its acceptance test.

Why a network, given the night's evidence: the lift resolves crossings but reads the acquisition's own anisotropy on real
HiP-CT data (a z population in most white matter) and is limited to a 33°-wide response; anatomix answers "what tissue,
in what context", not "which populations, which way" (a crossing is one class to it; its voxel features carry tile
context). A model supervised on population content, with the acquisition's signature randomised in training, is the one
route that addresses both at the source.

---

## 1. Principles

1. **Synthesise everything, label it exactly.** No real data in training. Every voxel of a training volume has known
   populations (tangent, fraction, coherence), a class, and the nuisances applied to it.
2. **The output is a population descriptor, not an embedding.** A distribution over directions whose lobes are the
   populations; N is read from it, never predicted as a number.
3. **The context is the window.** Receptive field 15–25 voxels (0.3–0.5 mm at 20 µm): enough to see several fibres of
   each population, too little to classify the object. Output at the lattice stride (4), as the lift.
4. **Invariances are chosen.** Contrast, polarity, gamma, bias field, noise, PSF (isotropic and anisotropic), slice-wise
   structure, rings, streaks, patch/tile seams — all randomised with the labels held fixed.
5. **No axis.** Directions uniform on the sphere; rotations applied to rendered volumes; the 24 cube symmetries plus
   continuous rotations as augmentation; SH outputs (rotation-covariant), lobes read off-grid.
6. **Classical after the model.** Lobes, carry-forward, identities, the field, the volume driver: unchanged.

## 2. Inputs

### 2.1 The generator (`shapetensor/learn/generator.py`)

Built from `phantoms.py` (the splatter, textures) and `artefacts.py`, ported to torch (`scatter_add`) so a 128³ sample
renders in ~0.1 s on the GPU, and extended:

| element | what is randomised | label it yields |
|---|---|---|
| fibre populations, 0–3 per region | direction uniform on S²; density; width mixture; dispersion 0–30° and pitch; volume fraction; contrast (dark or bright fibres, any grey level) | per voxel: population id, local **tangent**, fraction |
| geometry | straight; **bends** (radius 0.2–3 mm), fans, spirals; bands and sheets of fibres; crossings at any angle, intermixed or interleaved; T-junctions, free ends, kissing bundles; partial volume at edges | tangent per voxel (so a bend is one turning population) |
| sub-resolution mixtures (experiment) | two populations inside the voxel/patch scale, dispersions wider than the window resolves | fractions (to test whether they are recoverable) |
| non-fibre structure | SynthSeg-style random label maps for background tissue (cortex-like layers, blobs, cells); vessels as tubes (any orientation); tissue steps and sheets | class: fibre / vessel / boundary / background |
| acquisition nuisances | isotropic and **anisotropic PSF** (elongated along a random axis, usually z), slice-wise intensity jitter, **rings** about an axis, **streaks**, noise level and correlation, bias field, gamma, 8-voxel patch and 64-voxel tile seams (as in the fixed models), dropout of slices | none (held fixed) |
| scale | voxel size 10–100 µm equivalents (fibre width and spacing in voxels vary accordingly) | the physical scale, for a scale-conditioned model or a per-scale `scale` input |

Rendering: 2× supersampled then downsampled (no lattice aliasing), the random rotation applied to the rendered volume
and the labels together. Each sample: a 128³ uint8 volume, a class map, and the **target** below. Pre-generate 2–5 k
samples (≈ 60 GB with lattice targets in float16) and augment on the fly (symmetries, contrast, noise, PSF); regenerate
as the generator improves. A fixed held-out set of 200 samples with all nuisances for validation.

### 2.2 Real-data inputs at inference

Any array `(Z, Y, X)` (grey levels or a scalar field), `spacing` for anisotropic voxels, a `scale` in voxels tying the
model's window to the data as today. Whole volumes chunked with a halo of the receptive field plus the carry-forward's
reach, exactly as `volume.run_volume` does now; the model's output per chunk is the lift's shape `(n, z, y, x)`.

## 3. The model (`shapetensor/learn/model.py`)

- **Architecture**: a three-level 3-D UNet, 2–5 M parameters, GroupNorm (not BatchNorm: the texture statistics of a
  chunk must not matter), output taken at stride 4 (the lattice). Receptive field ≈ 20 voxels. Optional warm start of the
  first two encoder blocks from the anatomix UNet (run against a from-scratch control; the inherited x-axis bias is a
  measured risk).
- **Heads** (all at the lattice):
  1. **the orientation distribution**: SH coefficients to l = 8 (45 real coefficients, even orders only — axial), from
     which `f(x, u)` on the 160-direction hemisphere grid is evaluated for the chain; holds up to 3–4 lobes;
  2. **class**: fibre / vessel / boundary / background (softmax);
  3. **scalars**: fibre fraction of the window; per-lobe coherence is read from the lobe's width, not predicted.
- **Scale**: either train across voxel sizes with the physical scale as a conditioning input, or one model per scale band
  with the data resampled to it (the chain already resamples anisotropic voxels). Start with the second (simpler).

## 4. Training (`shapetensor/learn/train.py`)

- **Targets**: the distribution target at a lattice position is the window average (Gaussian σ = scale) of the
  per-voxel kernels `Σ_k a_k K(u; t_k, κ_k)` — one von Mises–Fisher kernel per population at the voxel's tangent `t_k`,
  width from the population's dispersion, mass `a_k` the fraction — projected to SH. A bend is then one broadened lobe
  by construction; a crossing two lobes; texture nothing. Class: the majority class of the window.
- **Losses**: MSE on the SH coefficients (equivalently on the sphere), cross-entropy for the class, L1 for the fraction,
  plus a **consistency term**: two nuisance draws of the same sample must give the same distribution (anatomix's
  correspondence idea applied to our outputs).
- **Schedule**: 128³ crops, batch 2–4, bf16, AdamW, 100–200 k steps; ~0.1 s per step on an H100 → 3–6 h per run.
  Data loading from the pre-generated set with on-the-fly augmentation on the GPU.
- **Runs**: (1) MVP: straight bundles, crossings, bends, texture, basic nuisances — the question is whether the window-
  level distribution is learnable at 20 µm; (2) + full nuisances incl. anisotropic PSF and rings — the question is the
  VOI's z population; (3) + sub-resolution mixtures — the question is whether they are recoverable; (4) warm start vs
  scratch. Each run judged by the suite below before the next.
- **Compute**: generator and prototypes locally (A6000, GPU 0); training on an H100 (seymour4 had ~7 free on 22 Oct;
  one is enough); data and checkpoints under the project storage. Inference back on any GPU: a UNet pass per chunk,
  seconds for 384³, hours for a 20 µm brain.

## 5. Outputs

Everything the 0.3 chain outputs, with the lift replaced — and the scalars the structure tensor cannot give:

| output | shape | from |
|---|---|---|
| `fodf` | `(160, z, y, x)` on the lattice | the SH head evaluated on the grid (the lift's slot) |
| `vecs`, `weights`, `labels` | `(nmax, 3, Z, Y, X)`, `(nmax, Z, Y, X)` | lobes (`nmax`, `min_sep_deg`), carry-forward, identities, interpolation — unchanged |
| `vec`, `weight` | the single field, never cut | unchanged |
| `N` | per voxel, and the **population-count heatmap** with the lobes' masses | lobes of the distribution; cortex many small, white matter few large |
| `masks` | per population above a mass threshold; their union (the fibre mask) | lobe mass × class |
| `coherence_k` | per population, [0, 1] | the lobe's angular width (a Watson concentration) |
| `fibre_fraction` | per voxel | the scalar head |
| `gfa` | per voxel | std/rms of the distribution over the sphere (the dMRI-comparable number) |
| `class` | fibre / vessel / boundary / background | the class head |

Written by `volume.run_volume` with `engine="population_model"`; the one-call API gains `model=` (a checkpoint path).

## 6. Validation and acceptance

The suite as it stands (`shapetensor validate`: phantoms with truth, lattice / identity / single-field scores, tracking
against the excised structure tensor), run on the model's output, plus:

- **axis test**: a bundle at 24 directions over the sphere, error by angle to the nearest axis (flat); on fibre-free
  texture the mean predicted distribution over the sphere is flat;
- **bends**: radii 0.2–3 mm, N stays 1 along the turn, through-the-turn tracking ≥ the structure tensor's;
- **crossings**: both within 15° at 90 / 60 / 45 / 30°, intermixed and interleaved, dominant and non-dominant (density
  ratio 1 : 0.5); the current chain's numbers are the bar (band 90/60/45 through 0.89 / 0.86 / 0.89; intermixed 60° both
  within 15° 0.46);
- **real data**: the VOI against the structure tensor at its confident voxels (≤ 15° median, the chain's level) and the
  share of first lobes near z (must fall from 0.64 toward the structure tensor's 0.05); the N heatmap and the fibre mask
  judged visually against the image, cortex vs white matter;
- **held-out syntheses** with every nuisance: the distribution error, the class accuracy, the fraction error, and the
  sub-resolution test (predicted vs true fractions of unresolved mixtures).

## 7. Timeline (one person, one H100)

| days | milestone |
|---|---|
| 1–2 | generator in torch with dense labels and the nuisance pipeline; 2 k samples; the intensity-patch probe in the overlap (decides the membership question) |
| 3 | model, targets, training loop, local smoke run on the A6000 |
| 4 | MVP run on the H100 (3–6 h); suite; axis test |
| 5–7 | full-nuisance run; the VOI; bends and crossings; warm start vs scratch |
| 8–10 | sub-resolution run; scale handling; `engine="population_model"` in the chain and the driver; docs |
| ~14 | a validated model: suite, VOI, tracking |

## 8. Layout on this branch

```
src/shapetensor/learn/
  generator.py   fibre populations + background + nuisances -> (volume, labels, class) in torch
  targets.py     per-voxel kernels -> lattice distribution (SH l=8), class, fraction
  model.py       the three-level UNet with the three heads; SH -> hemisphere grid
  train.py       data, augmentation, losses, schedule, checkpoints
  engine.py      inference as a feature-engine-like object: block -> fodf on the lattice (the lift's interface)
scripts/learn/   generate.py, train.sh (qsub for the H100), evaluate.py (suite + axis test + VOI)
notes/plan_population_model.md   this plan
```

## 9. Decisions needed

1. Scale: one model per scale band (resample the data) or scale-conditioned — start with the former.
2. The SH order (8) and `nmax` (3) for the released outputs.
3. Whether the class head's "boundary" and "vessel" need the SynthSeg-style background in the MVP or only in run 2.
4. Where training data and checkpoints live on the cluster storage (~60 GB for the samples).
5. The warm start: run it as a comparison, or skip it.

## 10. Risks

- **Domain gap**: the model has seen only syntheses; the VOI comparison against the structure tensor's confident voxels
  and the z share are the guard; the nuisance pipeline is the remedy.
- **The window-level ceiling**: if the intensity probe shows no voxel-level membership, the model's honest output is the
  window distribution; crossings below the kernel's angular width stay unresolved (the generator's dispersion sets it).
- **Learned axis bias**: the axis test after every run.
- **Over-smoothing by the UNet**: GroupNorm, a small receptive field, and the lattice output keep it a window statistic.

## 11. Amendments (22 Oct, evening): full-volume targets, modality profiles, scale

- **Every voxel has a state.** `sample()` appends two background states to the layers: the *interfaces* of the
  background's regions (a sheet whose normal is the boundary normal, present under any additive layer that crosses it,
  removed only where an occluding layer hides it; a step of ≥ 25 grey levels is a full sheet, fainter ones fade) and the
  *isotropic remainder* (κ = 0). `target()` gives, at every lattice cell: the `odf` of the **shapes present** (lines,
  sheets, blobs, interfaces; normalised by their occupancy, flat where nothing is), the shape `fraction` of the window,
  the `classes` shares (line / sheet / blob / interface / isotropic), `size` (occupancy-weighted radius) and `kappa`.
  The isotropic remainder is deliberately *not* mass in the ODF: it would dilute a faint line's lobe by its fraction;
  the fraction and classes carry that.
- **Scale is drawn** per sample (log-uniform 0.5–2.5 × every length) and the target can be given several windows
  (`sigma_w=(5, 10, 20)` → `(W, n, z, y, x)`): the latent must hold the state at each, and a reader selects the window
  (population scale) and reads `size` (shape scale).
- **Modality profiles** (`MODALITIES`): hipct / ct / mri / histology / mixed, each a probability per nuisance, with every
  other nuisance still drawn at `CROSS_MIX` = 0.08. Added nuisances: phase-contrast fringes, beam-hardening cupping,
  stitching seams, zingers (CT); Rician noise, Gibbs ringing, ghosting, thick slices (MRI); section misalignment,
  missing sections, per-section stain variation, stripes, dust, mosaic vignetting (histology / BigBrain).
- **Background styles**: `regions` (anatomix), `tissue` (two or three broad regions, grey/white-like), `cells` (a blob
  population on a flat background).
- **The boundary case** (`boundary_sample`): a step with faint fibres crossing it at 30° to the normal. The target has
  the line's lobe and the step's ring at the step and the lobe alone away from it — the acceptance test for the model:
  recovery of the lobe at the boundary as a function of fibre contrast / step / noise.
- Tests: `tests/test_learn.py` (8, CPU or GPU, 3 s).

## 12. The rebuild after the adversarial review (22 Oct, night) -- supersedes §2.1, §3, §4, §5 and §11 where they differ

Two reviews (data; loss / training / outputs) found the generator's labels did not match its images, the window
target turned bends into crossings, a density-space loss learned only lines, and the plan's network and engine broke
chunking and re-read the image. Rebuilt in four stages (commits 47ce84b, 27f3ecd, 22dac4d, 074ca33):

1. **Generator** (`learn/generator.py`): stationary random fields (padded beyond the blur, exact kernel variance);
   labels are shape tensors accumulated from what was rendered (t tᵀ, (I - n nᵀ)/2, a blob's normalised second
   moment); occupancy a volume fraction normalised analytically; volume-fraction compositing (occluders hide the labels
   under them); crossing groups at a drawn angle sharing one volume fraction, later fibres yielding (interdigitated);
   arc fields for bends of drawn radius; warps that never fold (det J > 0.2), tensors moved as J⁻¹ S J⁻ᵀ; a drawn
   acquisition axis for every slice-wise nuisance, rings and cupping about a random rotation axis, streaks at a random
   angle, artefacts in float and one display window; per-layer visibility (contrast to noise after the chain, lower
   confidence bound) scaling label mass; interface weight from the region means (orientation- and scale-free);
   modality content priors.
2. **Target** (`learn/targets.py`): each layer pooled on its own (σ_w = 5, stride 4, cells at 4i + 1.5): K = 4 slots
   per cell (trace-1 tensor, mass = the population's own visible volume, kind, layer id), merged within 20° for the same
   kind; fraction, class shares, size. A bend is one population with lower linearity; populations come only from
   distinct layers; faint populations keep their own mass.
3. **Model, loss, training** (`learn/model.py`, `losses.py`, `train.py`): a 365 k-parameter local network, no
   normalisation layers, even-binomial anti-aliased downsampling (cells on the target's centres; flips exact),
   receptive field 48 voxels; exact slot matching (24 permutations) with per-degree-balanced tensor costs, √mass,
   kind, classes, fraction, size, Hellinger on rendered densities, contrastive identity; on-the-fly generation into a
   replay buffer with exact cube-group augmentation.
4. **Engine** (`learn/engine.py`): every output from the network's output; per-population carry-forward in the slots'
   space plus the bridge-only density step; sheets reported as their own normal field; chunked volumes exact with a
   halo of half the receptive field plus the reach.

Open: the trained model's numbers (smoke run at 96³ locally; the H100 run needs permission on pryor); identity
linking still by direction (embeddings are output, not yet used to link); training crops at one lattice phase only.

## 13. Stage 5: the re-review (22 Oct, late) -- commits 51b505d, 9102e61

The re-review verified 12 of 13 generator fixes and 7 of 8 model fixes and found, beyond them:

- **Training collapsed**: both smoke runs (96³, batch 4, 2000-3000 steps) gave constant outputs (fraction, classes,
  tensors, masses independent of the input; axis error at the best-of-K chance level ~43°). Contributing causes found:
  closed residual branches (zero-initialised gains), labels of structures the images did not show (the old global
  visibility was ~1 almost always), nuisance chains that often erased orientation, per-batch mean chasing at batch 4.
  Remedies in place: open gains, local scale-free visibility (presence scales mass, orientation blends the tensor to
  isotropic), a strength curriculum, orientation-first loss weights, effective batch 16, lr 3e-4, collapse detectors
  in validation. Whether these fix it is the next experiment (an overfit-64 check, then a streaming smoke run).
- **Data**: saturated line layers (now a drawn volume fraction), interface erased by additive layers (now a boundary
  outside the volume budget), face fall-off (padded seeding, deposition confined to the region), K = 4 overflow (K = 6),
  blob merging axis-dependent, constant-angle crossings (40% independent fields), visibility global/scale-blind.
- **Engine**: empty slots turned into populations (presence head + absolute-mass gate), peak-picking on the grid lost
  30° crossings (populations read from slots), carry-forward mixing narrow crossings (walks that follow each
  population), halo too small for carry + bridge, exact fp32 10x slower than bf16 (bf16 default).

Calibrations to revisit with a trained model: the orientation-visibility range (t 6-18, set below a local structure
tensor's 8-25), the presence range (t 3-8), the bridge's spurious insertions (2-4% against an oracle).
