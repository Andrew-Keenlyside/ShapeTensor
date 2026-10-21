# Changelog

## Unreleased — the dev pipeline (not in the docs until validated)

- `shapetensor.lagodf` (results log section 34): flexible-N populations from the local lagged autocorrelation of the
  intensity on the sphere (the lag ODF), angular deconvolution by a synthetic or self-estimated response, momentum per
  direction, off-grid lobe peaks, a sensitivity weight from presence and sheet rejection (`ring_fraction`). Classical;
  tests `tests/test_lagodf.py`; phantoms `scripts/test_lagodf.py`; real VOI `scripts/figure_lagodf_real.py`.
- `shapetensor.lagtca`: SOBI (joint diagonalisation of lagged feature covariances) per window and an overlapping-window
  field; works on synthetic multichannel fields, not on dev-vit's features (section 34). Tests `tests/test_lagtca.py`;
  `scripts/test_lagtca_phantoms.py`.
- `shapetensor.dev`: `DevConfig`, `calibrate_dev`, `apply_dev`, `population_field_dev`, `run_volume_dev`; CLI
  `run --pipeline dev [--dev-config cfg.json]`. Stages: anatomix-dev-vit + sliceTCA features, fibre-set populations,
  directions from the shape features, the released merge.
- Engines `anatomix_devvit` (`features/devvit.py`: 128³ tiles with 32 voxels of context, patch-grid shift average) and
  `devvit_slicetca`; `shapetensor.smoothing` (sliceTCA, promoted from `experimental.slice_tca`, which now re-exports it).
- `shapetensor.fibre_sets`: the windowed fibre-set stage 2 promoted from `experimental.fibre_sets` (functions unchanged),
  with `FibreSetConfig`, label-free thresholds from synthetic references (`References`) and `fibre_sets()`.
- `tensors.feature_products` / `tensors.shape_tensor`: a population's tensor from the features' gradients.
- Released pipeline: calibration is seeded (`calibrate(seed=0)`; the gap-quantile draw used the global RNG, so a block
  could calibrate differently between runs); `run_volume` takes `calibrate_fn` / `apply_fn` / `engine_margin` hooks and
  records the pipeline; `apply` returns `n_populations`; the teacher and the tensor helpers keep the block's device.
- Tests `tests/test_dev.py` (CPU); `scripts/smoke_dev.py` (one phantom, end to end) and `scripts/validate_dev.py`
  (released vs dev, one change at a time, on the fibre-set prototypes' phantoms).
- Validation and debugging (2026-10-21, results log section 32):
  - Stage 2 found no fibre windows with dev-vit: its readout gives fibre-free texture confident directions (the patch
    grid puts them on the axes), so texture out-peaks fibres in the prominence test. Presence is now the windows'
    distance from synthetic fibre-free texture (`separate.TextureModel`, `FibreSetConfig.presence="distance"`); the
    linear fibres-vs-texture probe (`separate.PresenceProbe`, `"probe"`) and a readout with a no-orientation output
    (`iso_readout`) are kept as options (both miss dense crossings). `pipeline.calibrate(..., iso=...)` fits them on
    `fibre_sets.iso_textures`.
  - Directions from the features' gradients follow dev-vit's patch grid (p90 ~89° with the true populations given);
    the default is now `direction="readout"`: the readout's distributions as dyads (`OrientationReadout.dyad_products`,
    `tensors.readout_products`), within a degree of the image structure tensor when the readers calibrate in-domain.
  - dev-vit's readers must calibrate in-domain: fitted on straight bundles alone their readout (fit 0.97) misreads a
    bend by 14–46°; `validate_dev.py` gains `CALIBRATION=self|pool`.
- `shapetensor.artefacts`: streaks and rings in z slices (the rotation axis); `scripts/artefacts.py` (phantoms, against
  the truth) and `scripts/artefacts_real.py` (a real slab, against its own artefact-free field).
- `shapetensor.completion` (opt-in, `DevConfig.completion`): each fibre set carried across crossings along its own
  direction, extrapolated from where the set is alone (`steering`), only where the set holds on both sides within
  `completion_half` voxels (no free end is extended). `"image"` extends the layer; `"image_cont"` also takes the arms'
  direction in the bridged voxels; `"shape"` averages the features along the set's direction on each side (a
  population-steered matched filter in shape space), confirms the set with the readout and reads its direction from
  them. `apply_dev(..., stage2=)` reuses a block's stage 2 for another stage 3. `phantoms.band_crossing(t_junction=True)`:
  the second band stops at the first (false-extension control). `scripts/test_completion.py`, `diag_completion.py`.
- `scripts/figure_models.py`: the old (UNet output, encoder block 2) and new (dev-vit) feature spaces on the HiP-CT VOI.

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
