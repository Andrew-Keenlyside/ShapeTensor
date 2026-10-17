# Review and release plan

Three review passes over the restructured package (October 2026), each from six perspectives, each followed by
the work it called for. The validation numbers are in `docs/validation.md` and `results/validation.json`.

## Pass 1

### Correctness

1. **Populations are found per block.** A block holding regions of different orientation (the three-band
   interface phantom) is read as one population; the other regions then carry a weakly activated tensor and the
   single field is noisier there than the excised structure tensor (15° against 6° at the calibrated texture).
   → Candidates from the orientation density of *sub-windows* (64³), merged, then confirmed by their layers at
   block level; `nmax` raised to 4.
2. **The 60° detection sits on the merge threshold.** The two layer axes are pulled towards the bisector (about
   16° off each population, 28° apart) and the distinct-axis rule merges at 25°; one seed in three finds a single
   population. → Iterate the confirmation (re-unmix with the refined axes, recompute) and measure whether the
   axes move apart.
3. **The crossing phantom's populations are symmetric about z**, so their DEC colours coincide and the pictures
   cannot show the separation. → Symmetric about 45° instead (90°: pure blue and pure red).
4. The chunked driver finds populations per chunk, so the *set* of populations may change at a chunk seam; the
   single field is continuous (both sides see the same tensors), the multi-vector field is ordered by activation
   so its meaning is the same on both sides. Documented, not changed.
5. The excision's reference median is per chunk (with halo). Documented.
6. The activation floor: none in the confirmation (the pure activation decides), 1% of the null in every
   population tensor (so a population absent from a voxel falls back to the excised tensor instead of 0 / 0) and
   10% of the null in the merge (so a voxel with every activation at zero is the mean of the population tensors,
   not an undefined eigenvector — the all-red regions of an earlier real-block figure).

### Peer review

7. Every claim in `docs/` is tied to a validation number or to a section of `notes/results_log.md`; the
   separation claim is explicitly conditional on fibre coherence, and the calibrated-texture failure is reported
   in the same table as the coherent success.
8. The teacher is the excised structure tensor; a reviewer will ask whether the pipeline can exceed its teacher.
   It can where the teacher is wrong by construction (a crossing: the teacher is the bisector, the layers are not)
   and it inherits the teacher elsewhere; the bundle and interface rows show exactly that.

### Speed

9. 192³ block: features 0.6 s, calibration 3 s, populations 0.7 s, tensors 1.2 s, merge 0.9 s (RTX A6000); the
   structure tensor 0.6 s. A 768³ crop in 192³ chunks with halos: about 10 min on one GPU. → A `bench` command
   that reports this on the user's GPU.
10. The NNLS warm start was the whole cost until it was replaced by the normal equations (55 s → 0.04 s per
    block). Nothing else is near the budget.

### Clarity

11. The public surface is three functions, `calibrate` / `apply` for the two-step use, five modules for the
    stages, six CLI commands. Kept.
12. `OrientationReadout.predict` and `tensors.layer_tensor` are utilities the pipeline does not call; kept small,
    documented as such.

### Downstream

13. **The tracker follows every vector** of the no-mask multi-vector field, including a population's vector where
    its activation is at the noise level. → `track(..., weights=, min_weight=)`: a vector is followed only where
    its weight exceeds `min_weight` (the null `tau` by default).
14. BRIDGE: `run` writes `vec (3, Z, Y, X)` in (x, y, z) — the `structure-tensor` layout BRIDGE reads — plus `gap`
    and `weight`; `tissue_mask` (the smoothed total activation above the null) is the node-sampling mask. The
    activation weights are in standardised-feature units; `weight / tau` is the dimensionless presence.
15. Anisotropic voxels: `apply` takes `spacing`, `run_volume` did not pass it on. → Passed on, with `--spacing` on
    the CLI.

### Validation and testing

16. `pytest` is not in the environment; the suite was run through a stand-in. 14 tests pass (layout against
    scipy, eigen conventions, excision, spacing, the chunked driver, NNLS against scipy, modes, the merge rules,
    the phantoms, the whole pipeline on a crossing, the engine's tiling). → Tests for the pipeline's chunked driver
    and for the input readers; a CPU-only marker so a CI without a GPU runs what it can.
17. The validation's figure: brightness at the 99th percentile of the activation; the crossing angles of item 3.

### Release readiness for microscopy data

18. Inputs: zarr / OME-zarr and `.npy`. → TIFF stacks (`tifffile`) and NIfTI (`nibabel`) as optional readers for
    `block`; `run` stays on zarr (chunked, any size).
19. A `demo` command that builds a crossing phantom and runs the pipeline end to end, so a user can check the
    install without data.
20. Packaging: classifiers, URLs, `CHANGELOG.md`, a `LICENSE` file; GitHub Actions for the CPU tests and the docs
    build (`mkdocs`).
21. GPU memory: the pipeline on a 256³ block peaks near 22 GB; 24 GB cards should use `--chunk 128 128 128` and
    `--calibration-size 192`. Documented.
22. Reproducibility: seeds are fixed, but the readout's Adam steps on the GPU are not bit-reproducible; borderline
    population decisions (item 2) can differ between runs. Documented; item 2 reduces it.

### Outcome of pass 1

Items 2, 3, 6, 9, 13, 15, 16, 17, 18, 19, 20, 21, 22 done; item 1 tried and dropped (sub-window candidates fed too
many signatures into the joint unmixing, see below); items 4, 5, 8, 11, 12, 14 documented. Two findings changed the
design:

- **The joint unmixing is degenerate beyond three or four signatures.** The orientation code of the features has
  rank five, so unmixing a voxel onto six candidate signatures lets every layer become a mixture and every layer
  axis collapse onto the bisector. The confirmation therefore unmixes each candidate *alone* first (always well
  conditioned), keeps the distinct axes, and only then unmixes the kept axes together (two passes), where the
  shoulders of one population lose their shared voxels and collapse onto it. Rank-safe and robust to the number
  of candidates.
- **The readers respond unequally to orientations.** On a crossing of a population along x with one along z the
  readout's density and the dictionary's activations are about twice as strong for x. A per-direction gain for
  each reader, measured on the calibration block against the teacher (density over the teacher's histogram;
  activation on the own signature), normalises the candidates and the weights. It does not normalise the
  per-voxel layers, and the validation (x against z crossing) shows the second population's vector drawn towards
  the first: pair error 16° at 90° with both populations within 15° at 12% of voxels, against 10° and 96% on a
  crossing symmetric about the diagonal with the same code. This is the first item of pass 2.

## Pass 2

### Correctness

1. **Per-voxel orientation dependence of the layers.** Candidate remedy: symmetry averaging — run stage 2 on the
   block and on its grid symmetries (the axis swap z ↔ x and the x flip give four variants in the fibre plane;
   the 24 rotations of the cube in general), each variant unmixed in the network's own frame, the layers mapped
   back and averaged. Equivariant by construction, no channel mixing, four times the feature and unmixing cost
   (about 3 s on a 192³ block). To validate on the x–z crossing against the diagonal one.
2. **The single field on single-population tissue** is noisier than the excised tensor where the population's
   activation is low (the three-band interface: 6–15° against 2–6°). Candidate remedies: a larger floor on the
   population tensor's weight (0.1 of the null cost 5% of the 60° pair accuracy in an earlier test), or
   populations per sub-window with a rank-safe confirmation.
3. The population count at the calibrated texture (20° dispersion) is right on bundles and wrong on crossings
   (three found at 90°). The multi-vector field should carry a per-block confidence so a consumer can fall back
   to the single field: the readout overlap and the share of tissue above the null are the candidates.

### Peer review, speed, clarity, downstream, validation, release

4. The validation page now states the orientation dependence and the single-population weighting as open
   limitations with their numbers; every claim remains tied to a measured row.
5. Speed is unchanged (a 192³ block in about 6 s); symmetry averaging would make it about 10 s.
6. The notes' results log stops at §26; a §27 should record the pass-1 findings (rank, gains, the two-pass rule)
   with the numbers from the debug runs.
7. Downstream: the tracker follows the multi-vector field only above the null; BRIDGE reads `vec` and
   `tissue_mask`. A per-block confidence (item 3) is the remaining downstream need.
8. Testing: 16 tests through the stand-in; `pytest` still to be installed by the user. A test of the chunked
   pipeline driver exists; a test that the gains flatten the response on a bundle at two orientations would guard
   item 1's fix.

## Pass 3

(after the pass-2 work)

## Pass 3

(after the pass-2 work)
