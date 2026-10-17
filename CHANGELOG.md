# Changelog

## 0.2.0 — 2026-10

The package as a four-stage pipeline — shape features, populations separated in the shape space, one tensor per
population, the merged field — with the excised structure tensor as the baseline at the same scales.

- `shapetensor()` / `structure_tensor()` / `population_field()`; `pipeline.calibrate` / `apply` / `run_volume`.
- Stage 2: a signature dictionary and an orientation readout fitted on the block itself (the excised structure
  tensor at confident sites as the teacher), per-direction gains, the activation null, candidate populations
  confirmed by their own layers.
- Stage 3: the population tensor = the image's structure tensor weighted by the population's activation and the
  excision.
- Stage 4: the merged population tensors (one vector per voxel) and every population's vector with its
  activation (no masks).
- Three phantoms (bundle, crossing, interface) and a validation at two texture settings; a chunked driver for
  whole zarr volumes; a CLI (`run`, `block`, `structure-tensor`, `validate`, `phantom`, `demo`, `bench`, `stats`).
- The ODF / fixel and TCA routes archived under `shapetensor.archive`; the research record under `notes/`.
- Docs: a parameters page (every parameter varied on the HiP-CT block, `scripts/parameters.py`) and the pipeline
  illustration's partner on real data (`scripts/figures.py overview`).
- Whole-volume drivers: chunks without tissue are skipped and a failed chunk is logged and left zero (`pipeline.run_volume`);
  `st.run_volume` writes the arrays asked for (`vec gap` for a compact field) as float32 or float16 (`--arrays`, `--float16`,
  `--tissue-threshold` on `shapetensor structure-tensor`).
- Docs: scales, regions and a whole brain (`scripts/scales.py`, `scripts/wholebrain.py`).

## 0.1.0 — 2026-10

The research prototype (tag `research-2026-10-17`): the structure tensor and its excision, the channel-resolved
tensor and TCA, the gradient-direction ODF and fixels, the phantom suite and its report.
