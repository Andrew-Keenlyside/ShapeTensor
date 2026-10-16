# Shape space and the grey/white boundary shell: status and handoff

Status as of 2026-10-15. This is written for a session on a more powerful machine, picking the work up cold.

The underlying problem is in `BOUNDARY_PROBLEM.md` (same folder). BRIDGE's structure tensor follows the
*surfaces* of tissue (the grey/white boundary, and interfaces between tracts) instead of the fibres inside them.
Walks then wrap along the boundary instead of crossing it.

This document covers what has been tried since, what is running, and where the user is steering.

## Where we are, in one paragraph

- **Phantoms.** Gradient excision (fix A) fixes the tensor on the phantoms.
- **Real 100 µm brain.** Excision ends walks at every surface, so it is not the default. The larger tensor window
  (rho 0.6 mm) is the lever that helps deep WM.
- **Shape space.** The user then turned to anatomix "shape space" (a pretrained, contrast-agnostic 3D UNet,
  16 channels per voxel) as a possibly intensity-agnostic basis for orientation:
  - taking orientation from shape-space gradients failed;
  - separating shape space linearly into boundary and fibre components failed.

  Both failed on the phantoms and on a real VOI, for the reasons below.
- **The 20 µm crop.** The user has moved to the **20 µm S-21-46 calibration crop**: the internal capsule between
  thalamus and pallidum. They are working **step by step**:
  - step 1, the clean shape space, is done;
  - step 2, what the axes and channels mean, plus a continuous read-out, is done;
  - the next step is the user's choice.

  A sweep of anatomix's window (context) size on the phantoms was started at handoff.

## The user's standing directions (carry these over)

**For the shape-space work, right now**
- Go slowly, one step at a time. Show the result and wait.
- No k-means clustering, and no segmentation or mask labels in the VOI ("ignore segmentation or mask labels in
  the voi for now").
- Use the 20 µm calibration block for subcortical work (`/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr`), not
  the 100 µm "deep" VOI. That VOI turned out to be occipital periventricular WM; see below.
- Clear any chunking or axis artifacts in anatomix outputs. This is done; see "Running anatomix cleanly".

**Project rules (from memory; still apply)**
- Solutions must be graph-native. The momentum walk was rejected as "standard tractography".
- Tensor changes at the vertex stage are fine.
- Do not introduce new walk paradigms without asking.
- Label prototypes as prototypes.
- Figures go in the repo under `scratch/<topic>/`. The user reads them over SSH.
- Run targeted tests only. Never commit unless asked, and add no Claude attribution to commits.
- `/hdd` here is NTFS-over-FUSE: run heavy jobs serially, ≤ 4 workers, no `/dev/shm`. This is local-only.

## Results so far

### 1. Fix A, the edge-aware tensor (gradient excision)

Method: drop gradients whose coarse (sigma 3 voxel) magnitude means a step, and normalise the integration.

**Phantoms**
- A (gyrus with a fan crossing the boundary), angle to truth within 0–0.5 mm of the boundary: 71° → 16°;
  4° with rho 6.
- D15 (a tract interface crossed at 90°): 88° → 7°.
- U-fibres are kept, and the no-step controls are unchanged.

**Real 100 µm S2146**
- Whole-brain builds: excision drops wrapping 26.7% → 9.9%, but the median arc goes 115 → 43 mm, the callosum
  3.0 → 0.4% and > 80 mm pairs 8.5 → 2.3%. Walks end at every surface, including the fibre-parallel internal ones.
- rho 6 voxels (0.6 mm), without excision:
  - deep WM to HCP: 31° → 23°;
  - > 80 mm pairs: 8.5 → 15.9%; callosum: 3.0 → 6.1%;
  - wrapping unchanged.
- Wrapping is a mask/walk problem. Most of the cortex is below the threshold, so a walk at the boundary can only
  turn or stop; that is why fix D (the grey-matter stop) worked best.

**Options offered, user undecided**
- rho 6 as the 100 µm default;
- rho 6 plus fix D;
- excision only where texture is resolved.

Details: memory `edge-aware-tensor-fix-a`, scripts `edge_aware_*.py` and `tensor_variants.py`.

### 2. anatomix shape space: first looks

- Phantoms A and D15, and two 100 µm blocks, at 0.1 / 0.2 / 0.4 mm. Figures in
  `scratch/.../anatomix/*_resolution.png`.
- anatomix is a region and boundary encoder:
  - it separates tissue classes and WM compartments;
  - it gives boundaries and interfaces a state of their own (a band about 1 mm thick at the D15 interface);
  - it carries no usable fibre orientation.
- Coarser voxels give cleaner clusters.

### 3. "Shape tensor": orientation from shape-space gradients (phantoms D0/D8/D15/D30)

Variants:
- T0 = Σ_c ∇f_c ∇f_cᵀ over the 16 channels;
- integration over shape clusters (hard or soft memberships, the "shape scale") instead of a spatial rho;
- contrast removed by subtracting each shape cluster's mean intensity.

Median angle to the true fibres within 0.5 mm of the interface:

| tensor | D0 (no step) | D15 | D30 | D30, 3–8 mm away |
|---|---|---|---|---|
| BRIDGE now (intensity, rho 3) | 6° | 59° | 62° | 6° |
| intensity, rho 6 | 2° | 52° | 60° | 2° |
| shape tensor, rho 6 | 11° | 76° | 64° | **41°** |
| intensity integrated within 8 shape clusters | 3° | 53° | 60° | 2° |
| contrast removed by shape clusters | 3° | 28° | 58° | 6° |
| contrast removed, oracle split | 2° | 8° | 41° | 2° |

- anatomix is **not contrast-agnostic to a regional step**: a step between two regions is a "shape" to it.
- The feature change from D0 reaches about ±2 mm at step 15 and about ±7 mm at step 30.
- Shape clusters help only as an integration domain.

### 4. Fibre-selective tensors (same phantoms)

Median angle within 0.5 mm:

| tensor | D0 | D8 | D15 | D30 |
|---|---|---|---|---|
| median residual I − median_7³(I), rho 6 | 2° | 6° | 16° | 47° |
| Hessian line vote | 4° | 18° | 48° | 57° |
| **gradient excision gm3** | 6° | 8° | 9° | 10° |

- The median residual fails at a large step: the median is biased by about one noise width either side, which
  leaves a sharp step in the residual.
- Gradient excision is still the only method that is robust at every step size.

### 5. Linear boundary/fibre separation of shape space (paired D15 − D0)

Figure: `shape_tensor/shape_space_separation.png`.

- The step's change is concentrated: 4 directions carry 95% of it at D15, and 1 direction carries 87% at D30.
- But it lies on the **same principal axes as D0's own fibre variance**: removing those 4 directions keeps only
  8% of D0's variance.
- Removing 99% of the step change still leaves the interface at 40–61°. Orientation information lives in the
  low-variance directions, where the residual step still dominates.
- Conclusion: boundary and fibre are entangled by the network's nonlinearity. In the *image* they add up, which is
  why a linear derivative-jet space was proposed as the place to try this separation (not yet tried).

### 6. The old "subcortical" VOI (S2146 100 µm, voxel 675, 577, 775): NOT subcortical

- `shape_tensor/voi/voi_location.png`: it is 96% left cerebral WM, occipital periventricular, beside the
  occipital horn (sagittal stratum / optic radiation). It was "deep" only because it sits 5 mm inside WM.
- Results there:
  - Shape and fibre tensors are further from HCP than the standard tensor: 15° near interfaces and 35° elsewhere,
    against 8° and 17° for standard rho 0.6.
  - Removing the boundary directions changed nothing.

### 7. The 20 µm S-21-46 calibration crop (current focus)

**The data**
- Crop: `/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr`, 768³ uint8, 375 MB. It is the deep left hemisphere:
  thalamus 41%, pallidum 14%, the internal capsule between them.
- Source TIFFs and SynthSeg: `/hdd/data/brain_atlas_20um_test_crops/S-21-46/` (the QC png shows the location).
  Origin: `andrew@128.40.89.5:/hdd/andrew/brain_atlas_20um_test_crops/`; the full brain is
  `.../brain_atlas_20um_masked/S-21-46_brain_20um.ome.zarr`.
- Array axes: a0 inferior → superior, a1 anterior → posterior, a2 patient right → left.
- Whole-brain threshold: 142.5. At 20 µm it sits inside the thalamus peak and shatters into speckle, so do not use
  it to define regions. (The user has said to ignore masks for now anyway.)
- VOI: the central 256³ (5.12 mm), read with 32 voxels of real context per side.

**Step 1: shape space** (`shape_20um_space.py`)
- Figures: `voi20/voi20_step1_shape_space.png` and `voi20_step1_channels.png`.
- Features saved to `bridge_wmcov/shape_tensor/voi20_features.npy` (16×256³, float16), with the PCA basis in
  `voi20_features.json`.

**Step 2: what it is made of** (`shape_20um_explain.py`)
- `voi20_step2a_axes.png`: the axes.
  - PC1 (43%) = smooth internal capsule vs vessel-rich textured tissue.
  - PC2 (21%) = the fibre-bundle band between thalamus and internal capsule.
  - PC3 (11%) = thalamic granular texture.
  - The 16 channels span about 8 dimensions (97% of the variance) and are strongly correlated. Individual channels
    have no fixed meaning: any rotation of the 16 is equivalent under training.
- `voi20_step2b_channels.png`: more vs fewer channels.
  - The wider layers are internal and coarser: layer 50 is 64 channels at 1/4 resolution, layer 57 is 32 channels
    at 1/2 resolution.
  - Cutting the output to 3 or 8 PCs gives the same similarity maps as all 16.
- `voi20_step2c_continuous.png`: channel-free read-outs. These are cosine similarity to a seed voxel (continuous
  0–1) and the shape-space gradient |df/dx| (continuous boundary strength: rings round vessels, streaks along
  fascicles).

**Superseded files (do not use)**
- `voi20/voi20_shapes.*` and `voi20/voi20_shape_space.png`: the k-means version, stopped at the user's request,
  and computed with seamed features.
- `shape_20um.py` (k-means, label-free) and `shape_20um_tensors.py` (used SynthSeg/mask regions).

**The user's last open question** was which of these to do next:
- a continuous manifold embedding of the feature vectors;
- more seeds placed on specific structures;
- the fibre and boundary content of the continuous maps.

### 8. Window sweep (32 vox to whole block: finished; 8 and 16 vox: requested, NOT yet run)

Script: `shape_window.py`. It reruns the D15 − D0 separation at anatomix windows 32, 64, 96, 128 and the whole
block (3.2–23 mm of context). The user asked for "the separated RGB for the boundary and fibre shapes against
sliding window size, with some reference to the boundary fibre difference".

Per window it records:
- the boundary-component RGB (top 3 step directions);
- the fibre-component RGB (top 4 removed);
- the overlap numbers (step change vs D0 fibre variance in the top 4 directions);
- how far the step reaches into shape space;
- the fibre-tensor angle at the interface.

The hypothesis to test is that less context localises the step's influence and so makes it more separable.

Outputs:
- `bridge_wmcov/shape_tensor/shape_window.json`;
- figures `shape_tensor/shape_window_rgb.png` and `shape_window_metrics.png`;
- features cached in `bridge_wmcov/shape_tensor/win/`.

Rerun: `python shape_window.py run && python shape_window.py fig`. That takes about 25 minutes on an RTX A2000,
mostly the 32-voxel window × 8 grid phases.

**Result: the window (context) does not separate boundary from fibre.** The sweep finished. Figures:
`shape_tensor/shape_window_rgb.png` and `shape_window_metrics.png`.

| window | step change in top 4 | D0 fibre variance in same 4 | D30 change in D15's 4 | reach D15 / D30 (mm) | fibre tensor within 0.5 mm, D0 / D15 / D30 |
|---|---|---|---|---|---|
| 32 (3.2 mm) | 98% | 98% | 86% | 5.1 / 2.7 | 10° / 82° / 68° |
| 64 (6.4 mm) | 94% | 95% | 47% | 3.3 / 4.3 | 9° / 77° / 65° |
| 96 (9.6 mm) | 93% | 92% | 41% | 3.2 / 5.6 | 9° / 78° / 66° |
| 128 (12.8 mm) | 94% | 90% | 32% | 3.1 / 6.7 | 8° / 64° / 66° |
| whole (23 mm) | 96% | 92% | 32% | 2.5 / 6.3 | 7° / 50° / 66° |

- **Overlap stays at 90–98% at every window.** Boundary and fibre share the same shape-space directions whatever
  the context.
- **D15 improves with more context** (82° → 50° at the interface) but stays far from excision's 9°. D30 does not
  improve (65–68°).
- **The step's directions do not transfer between step sizes,** except with little context. With more context,
  D30's change moves away from D15's directions (86% → 32%), so the effect is nonlinear in the step size.
- **Caveat on the RGB maps:** the boundary-component maps show banding. That comes from the float16 feature cache
  (`win/*.npy`), because the boundary directions have a small range. The metrics in the table were computed from
  the float32 features before caching. Cache float32 if minor directions are to be viewed. The same applies to
  `voi20_features.npy` (float16) for anything beyond the top few PCs.

**In progress at the move to czi001: the user asked "Can you try an 8 and 16 vox window too?"** (0.8 and 1.6 mm
of context.) Where it stands:
- **The network cannot take an 8³ or 16³ input.** The UNet max-pools 4 times; at 16³ the bottleneck is 1³, and the
  network's own reflect-padded 3³ convolutions fail there ("Padding size should be less than the corresponding input
  dimension"). The smallest input it runs on is 32³.
- **Code written but untested:** `anatomix_shape.py` now has `NET_MIN = 32` and `_PadTile`. When `window < NET_MIN`,
  `anatomix()` wraps the UNet so every sliding-window tile is mirror-tiled up to 32³ (repeated reflection, since
  torch reflects at most n − 1 per call), run, and the centre cropped back. The network then sees only the tile's
  own content, mirrored, so the window still sets the context. That is the honest way to restrict context, but say
  in the figure that the 8 and 16 windows see mirror images of themselves beyond the tile.
- **Not done:**
  - the timing test (scratch `time_small_win.py`, one grid phase on a 96×64×96 sub-block of D0; rewrite it);
  - adding 8 and 16 to `WIN` in `shape_window.py`, plus its docstring;
  - the run, which only computes the new windows (the others load from `win/`);
  - the figure (7 rows).
- **Cost guess:** at 0.75 overlap the voxel-passes per window are a fixed 64× the volume, but every 8³ tile is padded
  to 32³, so w8 costs about 64× more GPU work than its voxel count suggests (w16 about 8×). If w8 is too slow at
  0.75 overlap, use 0.5 for it and say so; the 8-phase average still removes the checkerboard. Raise `sw_batch` for
  small tiles (e.g. 512; `_window_batch` caps it by the 2³¹ element limit).
- Then: the same table and figures as above, with the 8 and 16 rows, and a reply to the user on whether *less*
  context than 32 voxels starts to separate boundary from fibre (the hypothesis this sweep tests).

## Running anatomix cleanly (important)

Two artifacts come from *how anatomix is run*, not from the image. Fixed by `anatomix_shape.anatomix_clean`:

1. **Sliding windows leave their stride.**
   - chunkreg's `AnatomixFeatures` falls back to 0.25-overlap windows when a block doesn't fit the GPU. That makes
     visible seams, which read as straight "boundaries" in the shape gradient.
   - At 0.75 overlap a faint 40-voxel period remains.
   - Fix: one pass, no windows. On the CPU this is `device="cpu", amp=False`; a 320³ block took 22 s and 28 GB.
     On the GPU it needs the memory.
2. **The network's own period-2 checkerboard.**
   - A 1.5–3% even/odd plane alternation along every axis, strongest along a2, present even in one pass.
   - Fix: average the features over the 8 one-voxel grid phases (roll → run → roll back), then L2-normalise.
     Even/odd goes 1.028 → 1.000 and Nyquist power 5.5× → 0.2×.
3. **Zero padding.** The extractor zero-pads up to a multiple of 32, which puts a hard step at the block faces.
   `anatomix_shape.anatomix` reflect-pads by 16 first. Better still, read real context and crop it off.

What remains in clean features at 42–128 voxel periods is anatomy; it is present with no windows at all.

**Next to check:** whether "tile with discarded margins" (one pass per large tile, keep only each tile's interior)
is exact when the margin exceeds the receptive field. That would be the scalable alternative to Gaussian blending
for whole crops.

## Environments and code

**chunkreg conda env** (anatomix work)
- torch 2.5.1+cu124, anatomix 0.1.0, monai 1.5.2, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.1, zarr 3.4.0,
  nibabel, tifffile, matplotlib.
- Install anatomix with `pip install git+https://github.com/neel-dey/anatomix.git`. Weights come from the HF hub as
  `neeldey/anatomix` via `anatomix.model.load_from_hf`.
- `AnatomixFeatures` / `load_backbone` come from chunkreg: `/home/andrew/scripts/scalable-reg` (a separate repo, at
  1fe84b5). `anatomix_shape.py` puts it on `sys.path`.
- **No numba** here, so `anat_lib.py` (HCP, streamline replays) runs in **bridge-gpu-zv3** instead.
- UNet layer ids for internal activations: `model(x, layers=[50, 57])`. Final layers: 59 Conv 48 → 16,
  62 Conv 16 → 16, 65 Conv 16 → 16.

**Scripts** (all untracked, in `scripts/studies/wm_coverage/bending_wrapping/`)
- The files to copy or commit:
  - `anatomix_shape.py`
  - `shape_20um_space.py`
  - `shape_20um_explain.py`
  - `shape_window.py`
  - `shape_tensor.py` / `shape_tensor_fig.py`
  - `shape_separate.py` / `shape_separate_fig.py`
  - `fibre_select.py`
  - `shape_real.py`
  - `voi_location.py`
  - `surface_phantoms.py`
  - `phantom_shell.py`
  - `tensor_variants.py`
  - `edge_aware_*.py`
- Paths are hard-coded to this machine (`/home/andrew/bridge_wmcov/...`, `/hdd/...`); edit `ROOT` / `OUT` / `CROP`
  at the top of each.

**Data a new machine needs**
- **The 20 µm crop** (375 MB): the path above, or rebuild it from the TIFFs with
  `scripts/studies/atlas20/atlas20_calibrate.py prep`.
- **The phantoms** (about 50 MB): deterministic, so regenerate them with
  - `python phantom_shell.py make` (A, B → `bridge_wmcov/phantom_shell/`);
  - `python surface_phantoms.py make` (C, C0, D0/8/15/30 → `bridge_wmcov/edge_aware/phantoms/`).
- **Only for 100 µm whole-brain work:**
  - the S2146 100 µm OME-Zarr `bridge_wmcov/hundred_um/new/sub-S2146/micr/...`;
  - the study snapshot `bridge_wmcov/edge_aware/bridge_snap` (the patched `_vertex_fields._structure_tensor`; env
    vars `BRIDGE_EDGE_F/GMED/FMED/DETECT_RHO`);
  - HCP-1065 at `/hdd/data/hcp_tracts`.

**Local costs measured** (RTX A2000 12 GB, 251 GB RAM)
- anatomix on a 200×120×200 phantom: 1.3 s, GPU, one pass.
- 320³ doesn't fit 12 GB in one pass; on the CPU it is 22 s and 28 GB.
- A 32-voxel window at 0.75 overlap: 28 s per phantom.

## What a bigger machine makes possible

- **One-pass GPU anatomix on large blocks.** The 20 µm whole 768³ crop is about 14× the 320³ VOI. On a CPU that
  scales to roughly 400 GB, so on an 80 GB GPU it needs either one pass in bf16 (test it) or tiles with discarded
  margins. Then:
  - steps 1–2 on the whole crop instead of a 5 mm VOI;
  - all five atlas brains' crops, for consistency (`/hdd/data/brain_atlas_20um_test_crops/`).
- **More grid-phase averaging, or flip test-time augmentation,** at no practical cost.
- **The window sweep at 20 µm on the real crop,** not only the phantoms.

## Open threads, roughly in the order the user was heading

1. The window sweep results: does less context separate boundary from fibre?
2. The 20 µm VOI, next step (user's choice):
   - a continuous embedding;
   - seeds on chosen structures;
   - the fibre and boundary content of the continuous maps.
3. Boundary/fibre separation in a *linear* feature space (multiscale Gaussian derivatives), where the step and the
   fibres add exactly, as the principled version of excision. Not started.
4. Shape clusters, or a continuous similarity, as the tensor's integration domain instead of a spatial rho
   (useful on phantoms only as a domain).
5. The BRIDGE-level decision still open from fix A:
   - rho 0.6 mm default at 100 µm;
   - with or without fix D;
   - excision at finer scales.

## Traps hit in this work

- `pkill -f <pattern>` matched the invoking shell and killed it (exit 144). Use `pgrep -f "name_2[0]um"`, a
  character class, so the pattern doesn't match itself.
- chunkreg's extractor zero-pads, and falls back to 0.25-overlap windows; see "Running anatomix cleanly".
- A crop's own Otsu threshold misclassifies tissue; use the whole-brain value (`atlas20/thresholds.json`). At
  20 µm even that splits textured nuclei.
- The `deep` label in `real_blocks_abs.json` means "≥ 3 mm inside WM", not subcortical.
