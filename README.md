# ShapeTensors

Fibre orientation from anatomix "shape space" rather than from intensity gradients: can a tensor built from
shape-space gradients, or from the fibre component of shape space, ignore grey/white and tract-interface steps that
the intensity structure tensor mistakes for fibres?

The work started inside BRIDGE, as the grey/white boundary-shell problem (`docs/BOUNDARY_PROBLEM.md`). It moved here
on 2026-10-15 to be tested as an approach in its own right. Everything here is a **prototype**.

**For the current recommended pipeline, read `docs/IMAGE_TO_VECTORS.md`** (image to vector field: what to use, the
evidence, the open problems). **For the history, start with `docs/SHAPE_SPACE_HANDOFF.md`.** It holds the status, the results so far, the user's standing
directions, the clean-anatomix recipe and the open threads. The task in progress is at the end of its section 8.

## Layout

- `scripts/`: the study scripts, flat. Each puts its own directory on `sys.path` and imports its siblings.
  - `anatomix_shape.py`: running anatomix cleanly (`anatomix_clean`), PCA/RGB helpers, `_PadTile` for windows under
    32 voxels.
  - `surface_phantoms.py` / `phantom_shell.py`: the phantoms (`python <script> make`).
  - `shape_tensor*.py`, `fibre_select.py`, `shape_separate*.py`, `shape_window.py`: phantom studies.
  - `shape_20um_space.py`, `shape_20um_explain.py`: steps 1 and 2 on the 20 µm S-21-46 crop.
  - `shape_real.py`, `voi_location.py`: the 100 µm S2146 block (needs data not copied here).
  - `tensor_variants.py`, `edge_aware_*.py`, `anat_lib.py`: shared tensor code, phantom figures, and the BRIDGE-side
    helpers (`anat_lib` needs numba; `phantom_shell` and `edge_aware_builds` import `bridge` lazily, only to build
    phantoms with BRIDGE).
- `docs/`: the handoff, the boundary-problem brief, the literature review.
- `figures/`: every figure so far (`shape_tensor/`, `anatomix/`, `edge_aware/`). New figures go here.

## Path mapping from the BRIDGE days

The docs were written inside BRIDGE. Read their paths as:

| in the docs | here |
|---|---|
| `scripts/studies/wm_coverage/bending_wrapping/<x>.py` | `scripts/<x>.py` |
| `scratch/wm_coverage/bending_wrapping/<dir>/` (figures) | `figures/<dir>/` |

Data paths are unchanged and absolute (same user and layout as msm12):
- `/home/andrew/bridge_wmcov/shape_tensor/`: features, scores, the `win/` window cache (float16), JSON results;
- `/home/andrew/bridge_wmcov/edge_aware/phantoms/`, `/home/andrew/bridge_wmcov/phantom_shell/`: the phantoms;
- `/home/andrew/bridge_wmcov/carryover/`: the BRIDGE boundary-problem carry-over (HCP comparison data, old scripts);
- `/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr`: the 20 µm calibration crop; `thresholds.json` beside it;
- `/home/andrew/scripts/scalable-reg`: chunkreg, for `AnatomixFeatures` (put on `sys.path` by `anatomix_shape.py`).

Not here: the 100 µm S2146 OME-Zarr, the BRIDGE study snapshot and HCP-1065 (only `shape_real.py`,
`voi_location.py` and BRIDGE-level work need them), and the SynthSeg label crop path the 20 µm scripts name under
`/hdd/data/...`. On czi001 the source crops, labels included, are at `/hdd/andrew/brain_atlas_20um_test_crops/`;
labels are not in use (standing direction).

## Environment

anatomix work needs torch, anatomix (`pip install git+https://github.com/neel-dey/anatomix.git`), monai, numpy,
scipy, scikit-learn, zarr 3, matplotlib, nibabel, tifffile. On msm12 this was the `chunkreg` conda env (torch 2.5.1,
anatomix 0.1.0, monai 1.5.2; versions in `env_msm12_chunkreg.txt`). Weights come from the HF hub (`neeldey/anatomix`).
czi001 had no env with torch + monai + anatomix at the move.
