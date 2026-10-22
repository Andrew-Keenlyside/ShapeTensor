<div align="center">

<img src="images/shapetensor_logo.png" alt="ShapeTensor" width="45%" />

</div>

# ShapeTensor

**A model of the shape populations in any 3-D image.** Ask it about any point, at any scale, with as much context as
you like, and it describes the populations of shapes there — fibres, membranes, boundaries, vessels, cells — each with
its direction, its shape, its size and how much of it there is, through crossings and bends, whatever the modality.
What to extract (fibres only, larger tubes, sheets, a sub-population) is your choice at inference, not the model's.

> **Status (branch `population-model`, October 2026): in development.** The synthetic world, the training targets, a
> fixed-scale network and an inference engine are built and tested (72 tests); training does not yet generalise
> (it collapses to the dataset mean beyond a handful of samples — the first problem being worked on), and the
> flexible-scale model below is designed, not built. The **classical chain** (0.3, below) works today and is the
> baseline the model has to beat. The plan, blockers and feasibility tests: `notes/plan_foundation_model.md`.

## Why

The structure tensor — the usual way to read orientation from an image — holds one axis per voxel. At a crossing it
reports the bisector; at a boundary it is dragged into the plane of the step (a faint fibre crossing a grey/white
boundary disappears); past the end of a bundle it keeps going on texture; it has no notion of scale beyond its
window. A tracker following it switches bundle at every crossing. Learned feature models (anatomix and its kind) answer
"what tissue, in what context", not "which populations pass here, which way, how big".

## The model: a field you query

```
populations(x, σ_A, B)
```

the populations within a Gaussian window of size `σ_A` centred at any point `x`, using image context up to `B`.

- **`x` is continuous**: the output grid is yours — a lattice at `σ_A / 2`, a sub-voxel grid, points along a tract.
- **`σ_A`, the description scale**, and **`B`, the context**, are inputs, not constants: describe fine structure with
  a large context (to follow fibres through crossings, or to attempt sub-voxel orientations), or coarse structure with
  a small one. Outside the trained ranges the model extrapolates and degrades.
- **Every population is described continuously** — no categories:

  | | |
  |---|---|
  | shape tensor (trace 1) | direction and shape: linear, planar, isotropic as continuous shares |
  | mass | how much of it is in the window (visible volume fraction; a boundary: its step weight) |
  | presence | whether it is there at all |
  | radius | its cross-section scale |
  | edge / ridge | a step between regions vs a thin structure |
  | identity | which population it is, across the volume |

- **You select at inference** with predicates — fibres = linear and radius ≤ 3 voxels; larger tubes = linear and
  radius > 3; membranes = planar ridges; boundaries = edges — or find **sub-populations** as clusters of the
  descriptors (thin among thick fibres, different cell types). Carry-forward along each population, bridging across
  gaps, identities, the population-count (N) map and the field at every voxel then run on what you selected.
- **Coarsening is exact in the shape space**: mass, mass × tensor and mass × radius add, so an output can be
  re-described at any coarser scale by weighting each population over its neighbourhood — never mixing populations.
- **Every output comes from the model's output**; the image is its input only.

## How it is built

**A synthetic world with exact labels** (`shapetensor/learn/generator.py`). Layered random shapes on anatomix-style
backgrounds: lines, sheets, blobs, hollow tubes, crossing groups (interdigitated, at any angle, independent or
bending together, dominant and faint populations), bends of chosen radius, cell stains and laminae, region boundaries.
Labels are accumulated from what is rendered (each shape's tensor, volume, radius), so they match the image by
construction. Each sample is then acquired as HiP-CT, lab CT, MRI or histology — elastic warps that move the labels,
PSFs and resolution loss along a random acquisition axis, Poisson / Gaussian / Rician noise, rings, streaks, cupping,
phase-contrast fringes, Gibbs ringing, ghosting, thick slices, section misalignment, stain variation, mosaic tiles,
dust — mixed across modalities, with a strength knob for a curriculum. Each population's **visibility** is measured
in the final image (the local t-statistic of its fit): it scales the population's mass (presence) and blends its
tensor towards isotropic where the orientation cannot be recovered — the labels never ask for what the image does not
show. To come: multiscale fascicles, branching trees, cells with processes, helices, weaves, foams, lamellae, oriented
textures, real arrangements, more modalities (EM, light-sheet, OCT, polarised light), supersampled sub-voxel detail.

**Targets** (`learn/targets.py`): each population pooled on its own (a bend broadens one population and never adds
one), up to K = 6 per cell, merged only where the image cannot tell them apart.

**The network** (`learn/model.py`; today fixed-scale: σ = 5 voxels, stride 4, 48-voxel context): strictly local,
no normalisation layers (a volume in chunks equals one pass), anti-aliased, output slots of tensor, mass, presence and
identity. **To build**: a multiscale encoder whose depth is gated by `B`, and a query decoder conditioned on `σ_A`.

**Training** (`learn/train.py`, `learn/losses.py`): exact matching of predicted to target populations, per-degree
balanced orientation costs, data generated on the fly with exact rotation/flip augmentation, a curriculum.

**Inference** (`learn/engine.py`): populations read straight from the slots, carry-forward and bridging by walks that
follow each population (round bends, never onto a crossing partner), the voxel field, whole volumes in chunks with
exact halos.

## Uses

- **Tractography**: sampling and seeding from the selected populations' masses; one seed or node per population so
  tracking passes through crossings; direction confidence from each population's coherence instead of the structure
  tensor's FA (low at every crossing); angle limits from coherence, step lengths from radius and scale; identities to
  keep streamlines on their population; boundary entry points and angles for endpoints; export as fixels or an FOD for
  streamline tools.
- **Registration**: contrast-invariant channels (fraction, radius, shape shares, N) and orientation-aware similarity
  (tensors reoriented by the warp) — aligning tracts and laminae across modalities whose intensities disagree;
  complementary to anatomix features for the coarse alignment.
- **Maps**: N heatmaps (cortex: many weak populations; white matter: few strong), population masks, coherence (an
  FA that is right at crossings), sub-population maps.

## The classical chain (0.3): what works today

Five classical stages, nothing trained, one scale parameter: the image's local autocorrelation read on a shell of lags
(a field over positions × directions, where populations add), deconvolution and a presence floor from the block's own
null, carry-forward of each direction along itself, identities, and the field at every voxel.

```bash
git clone https://github.com/Andrew-Keenlyside/ShapeTensor.git && cd ShapeTensor
pip install -e .                   # torch (CUDA), numpy, scipy, zarr >= 3, numba, matplotlib
```

```python
import shapetensor as st
vec        = st.shapetensor(image, scale=10.0)                                             # (3, Z, Y, X): the strongest direction
vecs, w, l = st.shapetensor(image, scale=10.0, multi=True, weights=True, labels=True)      # co-located vectors, weights, identities
vec        = st.structure_tensor(image, rho=10.0)                                          # the excised structure tensor: the baseline
```

```bash
shapetensor run brain.ome.zarr field.zarr --level 0 --scale 10 --chunk 384 384 384
shapetensor validate --out validation      # the phantoms and the tracking tests
```

On phantoms with texture matched to 20 µm HiP-CT (same tracker, seeds and limits on both fields; `docs/validation.md`):

| phantom | measure | classical chain | structure tensor |
|---|---|---|---|
| band crossing 90° / 60° / 45° | streamlines through on their own band | 0.89 / 0.86 / 0.89 | 0.06 / 0.42 / 0.40 |
| a bundle that ends in texture | streamlines continuing past the end | 0.02 | 0.40 |
| intermixed crossing 90° | within 15° of their population | 0.99 | 0.10–0.22 |
| bend of radius 0.3 mm | through the turn | 0.39 | 0.92 |

Its limits are what the model is for: on real HiP-CT it also reads the acquisition's own anisotropy (a population
along the rotation axis in white matter, `docs/data.md`), tight bends are lost at the window scale, and its single
scale parameter fixes the window. `docs/` documents the chain in full.

## The model today (experimental API)

```python
from shapetensor.learn.generator import draw
from shapetensor.learn.targets import target
from shapetensor.learn.engine import PopulationEngine

s, d = draw((96, 96, 96), seed=0, modality="hipct")      # a sample: clean layers, its acquisition, visibility maps
t = target(s, d)                                          # its populations per lattice cell
eng = PopulationEngine("model.pt")                        # a trained checkpoint (none yet generalises)
out = eng.run(volume)                                     # N, directions, masses, sheets, the voxel field
```

```bash
python -m shapetensor.learn.train --out run --steps 20000 --size 96      # training on the fly (A6000 / H100)
python scripts/learn/examples.py                                         # the synthesis, labels and targets, by modality
```

## Layout

```
src/shapetensor/
  learn/          the model: generator.py (the synthetic world), targets.py, model.py, losses.py, train.py, engine.py
  lift.py, populations.py, field.py, volume.py      the classical chain (0.3)
  validate.py, phantoms.py, tracking.py             phantoms with truth, the tracking suite
  st.py           the structure tensor (plain, excised), whole volumes
  artefacts.py, viz.py, stats.py, filters.py, cli.py
  archive/, features/, dev.py, ...                  earlier routes (0.2, anatomix features)
tests/            pytest: test_learn, test_targets, test_model, test_engine (the model); test_chain (the chain)
docs/             the classical chain
notes/            plan_foundation_model.md (this direction), plan_population_model.md (how we got here), research log
```
