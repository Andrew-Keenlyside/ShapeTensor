# The grey/white boundary problem: a handoff for a fresh session

Written 2026-10-15. This is a self-contained brief for continuing in a new context.

**Do not act on the candidate fixes (section 6) until the user chooses one.**

## 1. The problem

At 100 µm (S2146, the default build: Otsu mask 144.5, no FA gate, pointer walks), streamlines run **along the grey/white boundary**, not the pial surface.

| Measure (SynthSeg surfaces) | BRIDGE | HCP-1065 |
|---|---|---|
| Surface-parallel runs ≥5 mm: depth below the grey/white boundary (median) | 0.6 mm | 0.7 mm |
| Same runs: depth below the pial surface (median) | 4.1 mm | – |
| Same runs: share within 1 mm of the pial surface | 0.1% | – |
| Share of streamline length in such runs | 27% | 8% |
| Share of streamline length inside cortex | 13% | 6% |

- The runs concentrate on **gyral crowns**, at 1.5× the crowns' share of the surface. The worst parcels are pre- and postcentral, superior parietal and superior temporal. HCP's runs sit under the fundi.
- In 3D renders this looks like "wrapping over the cortex", because the grey/white boundary follows every gyrus.

## 2. Mechanism (established)

1. **Ethanol-dehydrated white matter is bright**, so the grey/white boundary is a strong intensity step.
2. **The structure tensor turns that step into a parallel "fibre" orientation.** BRIDGE uses e3, the smallest-eigenvalue direction. At a step, that direction lies in the boundary plane.
   - Within 0.5 mm of the mask edge: edge-ness (L1−L2)/L1 is 0.8–0.94, and e3 lies in the plane in 88–93% of voxels.
   - Vertices: 91–92% are surface-parallel within 1 mm, falling to 62% at 2–4 mm (50% would be random).
   - The shell is about 0.5–1 mm thick and **scales with the tensor window ρ**. At 1–2 mm deep, the parallel share is 57%, 67% and 78% at ρ = 0.15, 0.3 and 0.6 mm.
   - Hypothesis, not yet measured: at gyral crowns the boundary is cylinder-like, which gives a fibre-like tensor along the crest.
3. **The alignment gate (45°) keeps only parallel edges inside the shell.** Inward edges are rejected, so the shell becomes a connected sheet.
4. **The pointer walk has no grey-matter stop.** Walks arriving from the fan turn onto the sheet within the 75° bend limit. Vertices in the outer 1 mm are visited 1.5× as often as deeper ones.
5. **Cortex intensity (vertex median 157) sits at the Otsu threshold (144.5).** Sparse cortex vertices exist (0.2–1.2 per mm³, against 5 in WM), and 87% of them are surface-parallel.
6. `masking.exclude_rind_mm` does NOT remove a shell from the mask; it only affects how the threshold is chosen. Streamlines don't reach the pial surface because the outer cortex is below the threshold.

## 3. Phantom proof

Files: `scripts/studies/wm_coverage/bending_wrapping/phantom_shell.py`; data and builds in `/home/andrew/bridge_wmcov/phantom_shell/`; figure `phantom_boundary_shell.png`.

**Design:**
- Two gyral blades on a white-matter base, 2.5 mm of grey matter, CSF above; 0.1 mm voxels, 20 × 12 × 20 mm, extruded along y.
- True fibres are straight lines fanning up from (x 10, z −15 mm). They cross the boundary and continue faintly into grey matter.
- **A:** WM 190 / GM 150 / CSF 60, threshold 144.5.
- **B:** the same fibre texture with no intensity step, threshold 100.
- Both are built with the real pipeline at the defaults.

**Tensor:** share of voxels parallel to the boundary, with the angle error to the true fibres in brackets.

| Depth | A | B | True fibres |
|---|---|---|---|
| 0–0.5 mm | 100% (16°) | 43–49% (9–12°) | ~51% |
| 0.5–1 mm | 83% (13°) | 51% (7°) | ~52% |
| ≥1 mm | identical | identical | – |

**Streamlines**

| | A | B |
|---|---|---|
| Length within 1 mm of the boundary | 46% | 24% |
| Runs ≥5 mm parallel to the boundary | 14.5% | 7.2% |
| Angle to truth within 1 mm of the boundary | 25° | 10° |
| Angle to truth in deep white matter | 6.7° | 6.5° |
| Streamlines entering grey matter by more than 1 mm | **33%** | **87%** |

**The phantom is the test bed for any fix: success means A behaves like B.**

## 4. Scale facts

- **Tensor parameters are in voxels.** BRIDGE's default (σ 1, ρ 3 voxels) is 0.05 / 0.15 mm at 50 µm but 0.1 / 0.3 mm at 100 µm.
- **At a matched window in mm, 50 µm adds nothing here.** On the same slice, 50 µm at σ 2, ρ 6 voxels has a median angle of 2.6° to 100 µm (97% within 20°). At its voxel default the median is 27°. The 100 µm volume is the 2×2×2 mean of the 50 µm one (r = 0.998).
- **Always compare scales at a fixed window in mm.**
- **Deep white matter at ρ 0.3 mm is at the noise level** in about half of voxels. A larger ρ gives more signal but a thicker shell.

## 5. What was tried (study prototypes; nothing in the pipeline changed)

| Change | Surface runs ≥10 mm | Other effects | Verdict |
|---|---|---|---|
| Stop on entering cortex (a step reaching a SynthSeg cortex vertex ends there) plus a white-matter seed filter (≥2 WM vertices); a function on steps | 21% → 4.2% | Cortex length 13% → 1.1%; long-range unchanged (8.6%); left–right 5.7% → 3.5%; single-parcel share 24% → 35% | Graph-native; the best containment so far |
| Stop at the mask edge (all, outward-only, or parallel-only arrivals) | – | Long-range 8.5% → 2% | Fails: the shell holds real superficial-WM paths too |
| Grey-matter stop from a single vertex's intensity (< 158) | – | Too aggressive: 3–9% of WM vertices are that dark | Fails |
| Smoothing 1–2 mm | Unchanged | Cosmetic | Fails |
| 2 mm minimum-radius cut | Unchanged | Removes hairpins | Fails for wrapping |
| Per-step bend 45° | – | Walks fragment | Fails |
| Momentum walk | – | – | **Rejected by the user as "standard tractography, not the graph approach"**; see memory `graph-native-solutions-only` |

## 6. Candidate fixes, not yet tested (the user decides)

| Fix | What it does |
|---|---|
| A. **Edge-aware tensor** | Suppress gradients that come from the tissue boundary before the ρ integration, as Wanjau 2026 did for vessels (they found that pre-masking intensity creates false gradients). Orientation then comes only from texture inside the WM. Fixes the cause at the vertex stage; the walk is unchanged. Test on phantom A against B, then the 100 µm brain. |
| B. **Per-vertex boundary attribute as a graph rule** | Store each vertex's distance to the mask edge and/or tensor edge-ness. Use it as an edge gate (reject parallel edges within the shell) or as a terminal label. |
| C. **Boundary-run accumulator in the global join** | Stop or cut after more than about 2–3 mm of parallel travel within 1 mm of the edge. This equals cutting, so results stay identical under any chunking. |
| D. **Grey-matter stop from tissue labels** | Already prototyped (section 5); needs a tissue map at build time. |

**Theory on curvature budgets**, from the discussion:
- Accumulate net rotation (it cancels jitter) or turning on links of at least 2 mm, never the sum of unsigned joint angles.
- Use a windowed rate, not a total.
- Use it only to stop or cut, never to steer.
- A curvature budget cannot detect boundary-following, because those paths are smooth.

## 7. Where things are

- **Scripts** (untracked, deliberately left out of v1.0): `scripts/studies/wm_coverage/bending_wrapping/`. Key files:
  - `anat_lib.py`: `extract`, `pointer_walk`, `pointer_walk_terminal`, `Anatomy` (SynthSeg fields), `MaskField`, resampling and measures;
  - `anat_bw.py`: `measure` (bend, wrap, ends, leaving the mask);
  - `variants.py`: walk variants, including `_stopG` and `_stopGw`;
  - `phantom_shell.py`, `st_glyphs.py`, `tensor_profile.py`.
- **Carry-over copy of the session scratchpad:** `/home/andrew/bridge_wmcov/carryover/`, with `anat/` (scripts, `out/*.json`, logs), `hcp_cmp/`, `cosmo_data2/new/meta.json`, `bridge_snap/` (the frozen code used for the builds) and `anat_review/`.
  - `variants.py` has the old scratchpad path in `SCR`; point it at the carry-over folder.
  - New builds can use the main checkout, which now has the `min_fa` default 0 committed (6f37daf).
- **Data and builds:** `/home/andrew/bridge_wmcov/`:
  - `hundred_um/new`: the default 100 µm build;
  - `scale/{s200,c50t,d50t}`: 200 µm whole brain, and the 50 µm cortical and deep blocks at threshold 144.5;
  - `sweep_dm/`: density × reach sweep, plus `d0025_m29_{a35,a55,r6}`;
  - `replays/`: cached graphs and mask distance fields;
  - `phantom_shell/`.
- **Images:** the 100 µm OME-Zarr is at `hundred_um/new/sub-S2146/micr/...XPCT.ome.zarr`; the 50 µm cortical block is `scale/cortical_50um.nii.gz`, corner at (−49.2, −34.2, 25.8) mm.
- **Write-ups and figures:** `scratch/wm_coverage/bending_wrapping/README.md` (the full study) and `literature_review.md`.
- **Memory:** `bending-wrapping-review`, `graph-native-solutions-only`, `bridge-vs-hcp1065`.
- **Still running at handoff:** one queue of the 2 mm-radius parameter search (`radius_constrained.py`, existing pointer walk plus a post-hoc cut). It writes to `/tmp/claude-1001/-home-andrew-scripts-BRIDGE/ffc7874f-afd2-4e0b-84bd-7c0775199f36/scratchpad/anat/out/radius2.json` and `radius2_b.json`. It is not central to the boundary problem.
