# The prototype studies (October 2026, before the package)

The work started as the grey/white boundary-shell problem of a graph-based tractography pipeline
(`BOUNDARY_PROBLEM.md`) and moved here on 2026-10-15 to be tested as an approach in its own right. Everything in
`scripts/*.py` at the top level is a **prototype** study script; the package in `src/shapetensor/` consolidated them.

**For the pre-package pipeline and its evidence, read `IMAGE_TO_VECTORS.md`**; **for the history,
`SHAPE_SPACE_HANDOFF.md`** (status, results, standing directions, the clean-anatomix recipe, open threads).
The post-package review, plan and results are `TCA_FIELD_REVIEW.md`, `TCA_FIELD_PLAN.md`, `TCA_FIELD_RESULTS.md`.

## Study scripts (`scripts/`, flat; each puts its own directory on `sys.path`)

- `anatomix_shape.py`: running anatomix cleanly (`anatomix_clean`), PCA/RGB helpers, `_PadTile` for windows under
  32 voxels.
- `surface_phantoms.py` / `phantom_shell.py`: the 0.1 mm phantoms (`python <script> make`).
- `shape_tensor*.py`, `fibre_select.py`, `shape_separate*.py`, `shape_window.py`: phantom studies.
- `shape_20um_space.py`, `shape_20um_explain.py`: steps 1 and 2 on the 20 µm S-21-46 crop.
- `shape_real.py`, `voi_location.py`: the 100 µm S2146 block (needs data not copied here).
- `tensor_variants.py`, `edge_aware_*.py`, `anat_lib.py`: shared tensor code, phantom figures, and the
  pipeline-side helpers (`anat_lib` needs numba; `phantom_shell` and `edge_aware_builds` import the pipeline lazily,
  only to build phantoms with it).
- `bench/`: the drivers, calibration JSONs and logs of the package's validation run (see its README).
- `figures/`: every figure (`shape_tensor/`, `anatomix/`, `edge_aware/`, `tca_field/`).

## Path mapping from the pipeline days

The older docs were written inside the pipeline repo. Read their paths as:

| in the docs | here |
|---|---|
| `scripts/studies/wm_coverage/bending_wrapping/<x>.py` | `scripts/<x>.py` |
| `scratch/wm_coverage/bending_wrapping/<dir>/` (figures) | `figures/<dir>/` |

Data paths are absolute (same user and layout as msm12):
- `/home/andrew/bridge_wmcov/shape_tensor/`: features, scores, the `win/` window cache (float16), JSON results;
- `/home/andrew/bridge_wmcov/edge_aware/phantoms/`, `/home/andrew/bridge_wmcov/phantom_shell/`: the 0.1 mm phantoms;
- `/home/andrew/bridge_wmcov/shapetensor/`: the package's phantoms, caches, results and figures (`SHAPETENSOR_DATA`);
- `/home/andrew/bridge_wmcov/carryover/`: the boundary-problem carry-over (HCP comparison data, old scripts);
- `/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr`: the 20 µm calibration crop; `thresholds.json` beside it;
- `/home/andrew/scripts/scalable-reg`: chunkreg, for `AnatomixFeatures` (put on `sys.path` by `anatomix_shape.py`).

Not here: the 100 µm S2146 OME-Zarr, the pipeline's study snapshot and HCP-1065 (only `shape_real.py`,
`voi_location.py` need them), and the SynthSeg label crops under `/hdd/data/...`. On czi001 the source crops, labels
included, are at `/hdd/andrew/brain_atlas_20um_test_crops/`; labels are not used (standing direction).

## Environment of the prototypes

anatomix work needs torch, anatomix (`pip install git+https://github.com/neel-dey/anatomix.git`), monai, numpy,
scipy, scikit-learn, zarr 3, matplotlib, nibabel, tifffile. On msm12 this was the `chunkreg` conda env (torch 2.5.1,
anatomix 0.1.0, monai 1.5.2; versions in `env_msm12_chunkreg.txt`); on czi001 it is `shapetensors`. Weights come from
the HF hub (`neeldey/anatomix`).
