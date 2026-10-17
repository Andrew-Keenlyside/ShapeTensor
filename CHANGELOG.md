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
- `run_volume`: every chunk calibrated on itself with the volume calibration as fallback (`calibrate_chunks`, `min_bins`);
  neighbouring chunks' merged tensors ramp-blended over a `blend` overlap so the field has no seams; the weight written
  in multiples of the chunk's null and zero where the volume has no data; `apply` returns the merged `tensor`.
- Figures draw every direction field as plain DEC; the activation appears as its own layers.
- Speed (a 192³ block's stages 2–4 from 12.8 s to 2.7 s, its calibration from 17 s to 10 s): a closed-form symmetric 3 × 3
  eigensolver in `st.eig_special_3d` (float64 internally, within 0.02° of cusolver, 10× faster); the non-negative
  least squares by coordinate descent (exact where 60 projected-gradient steps were 10% off on correlated signatures,
  5× faster); one averaged-feature pass shared by every reader; the gradient products computed once per block;
  GPU sampling of the calibration voxels; the whole-volume driver runs stages 2–4 on the core plus the filters' reach
  rather than the whole network halo.

## 0.1.0 — 2026-10

The research prototype (tag `research-2026-10-17`): the structure tensor and its excision, the channel-resolved
tensor and TCA, the gradient-direction ODF and fixels, the phantom suite and its report.
