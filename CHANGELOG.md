# Changelog

## 0.3.0 — 2026-10-22

The package rebuilt around one classical chain on any array: lift → populations → carry-forward → identities → field.
Nothing is trained on the analysed data; one scale parameter.

- `shapetensor.lift`: the array lifted to positions × directions by its local lagged autocorrelation on a shell of
  integer lags (exact, no interpolation), pooled onto a lattice and binned into a hemisphere grid; the presence floor
  from the block's isotropic surrogate; the response from the block's own single-lobe positions; non-negative
  deconvolution. Excision (the structure tensor's boundary rule) inside the correlation. 5× faster than the
  interpolated lag ODF and sharper.
- `shapetensor.populations`: carry-forward along each direction (all directions of a batch sampled per lag), lobes
  with off-grid directions and an optional saddle test, the sheet rule, population identities by constrained Kruskal.
- `shapetensor.field`: `Orientations` (every length tied to `scale`), the field read at every input voxel (co-located
  vectors, weights, identities; a single field never cut), `orientations()` for any array kind and anisotropic voxels.
- `shapetensor.volume`: whole arrays chunk by chunk with a halo that makes every chunk's core equal to a one-pass
  result (shared floor and response; no blending), multi-process, label stitching across chunk faces.
- `shapetensor.validate`: the acceptance suite — phantoms with truth (bundle, bend and its control, band crossings at
  90/60/45°, T-junction, free end, intermixed crossings, interface, texture), lattice / identity / single-field scores
  and the tracking scores (through, switched, false continuation) against the excised structure tensor. New phantoms
  `band_crossing(only_second=True)` (a free end) and `texture`; `tracking.track(headings=)`.
- `shapetensor()` runs the chain (`scale`, `multi`, `weights`, `labels`, `spacing`); `shapetensor.orientations()`
  returns everything. CLI: `run`, `block`, `structure-tensor`, `validate`, `phantom`, `demo`, `bench`, `stats`.
- Findings recorded in `TCA_FIELD_RESULTS.md` §35: the high-pass at `0.3 scale` removes most boundary rings; ring atoms
  in the deconvolution dictionary absorb true 90° crossings and were dropped; an N-independent presence; the lattice
  phase must be shared between chunks for exact seams.
- The 0.2 pipeline (anatomix features, populations unmixed per block) stays importable: `shapetensor.pipeline`,
  `.separate`, `.tensors`, `.merge`, `.features`, `.dev`, `.fibre_sets`, `.lagodf`, `.lagtca`, `.regions`,
  `.classical`; its validation and the kNN population model under `shapetensor.archive`; its docs under `docs/archive`.

## 0.2.0 — 2026-10

The package as a four-stage pipeline — shape features, populations separated in the shape space, one tensor per
population, the merged field — with the excised structure tensor as the baseline at the same scales.

- `shapetensor()` / `structure_tensor()` / `population_field()`; `pipeline.calibrate` / `apply` / `run_volume`.
- Stage 2: a signature dictionary and an orientation readout fitted on the block itself (the excised structure
  tensor at confident sites as the teacher), per-direction gains, the activation null, candidate populations
  confirmed by their own layers.
- Stage 3: the population tensor = the image's structure tensor weighted by the population's activation and the
  excision. Stage 4: the merged population tensors and every population's vector with its activation (no masks).
- Three phantoms (bundle, crossing, interface) and a validation at two texture settings; a chunked driver for whole
  zarr volumes with per-chunk calibration and ramp-blended tensors; a CLI; the ODF / fixel and TCA routes archived.
- Dev pipeline (branch `dev-pipeline`): anatomix-dev-vit with blended tiles, sliceTCA, fibre sets, completion across
  crossings, the lag ODF and lag-TCA, population labels, artefact tests (results log §32–34).

## 0.1.0 — 2026-10

The research prototype (tag `research-2026-10-17`): the structure tensor and its excision, the channel-resolved
tensor and TCA, the gradient-direction ODF and fixels, the phantom suite and its report.
