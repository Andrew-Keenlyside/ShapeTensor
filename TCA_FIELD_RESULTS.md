
## 27. The release review, pass 1: what the restructured pipeline taught (2026-10-17)

Measured on the three phantoms (`shapetensor.phantoms`), 192³ blocks, calibration on twelve bundles at 15° steps,
σ 1, ρ 10; the debug scripts lived in the session scratchpad and their numbers are quoted here.

- **A joint unmixing of many signatures is degenerate.** With six candidate signatures the NNLS coefficients rose
  to 30–50 (against 1–7 with four) and every layer's axis collapsed onto the bisector (49° / 48° from the two
  populations of a 90° crossing). The orientation code of the features has rank five (§26), so four signatures
  plus the background are the limit. Hence the two-pass confirmation: each candidate unmixed alone (always well
  conditioned) → distinct axes; then the kept axes unmixed together (twice) → distinct axes.
- **The readers respond unequally to orientation.** On a crossing of x against z fibres the readout density of
  the x population was 1.0 and of the z population 0.54, their single-signature activation masses 0.74 and 0.29;
  the density peaks' heights ranked three x shoulders above the z population. Per-direction gains measured on the
  calibration block (density over the teacher's kernel histogram: 0.95–1.56 at 8° dispersion, 0.94–1.67 at 20°;
  activation on the own signature: 0.82–1.20 and 0.76–1.23) normalise the candidates and the weights.
- **The per-voxel layers keep an orientation dependence the gains do not reach.** With the same code the
  multi-vector field on a crossing symmetric about the diagonal (135° / 45°) reached a pair error of 9–10° with
  both populations within 15° at 87–96% of voxels (90° and 60°), while on the x–z crossing it reached 16° and
  20%. Averaging the layers over the grid symmetries of the block (z–x swap and x flip, or the six axis orderings)
  did not help (90° at the diagonal: 87% → 18–37%), because the transformed frames are not better estimates and
  the average only blurs; the six orderings also take the phantom's fibres out of the plane the dictionary was
  calibrated on. Dropped.
- **The activation floor.** None in the confirmation; 1% of the null in every population tensor (a population
  absent from a voxel falls back to the excised tensor instead of 0 / 0); 10% of the null in the merge (a voxel
  with every activation at zero gave an all-zero tensor whose eigenvector is the x axis — large uniform red
  regions in an early real-block DEC). Larger floors in the population tensor (5–30% of the null) do not improve
  the single field on the three-band interface (7.2° → 6.8° far from the step) and cost the 60° crossing
  (both within 15°: 67% → 46%), so the single-population noise there is the one population found for three
  bands, not the floor.
- **Sub-window candidates** (64³) with the two-pass confirmation find a second band on the interface (N 2, far
  error 7.2° → 5.4°) and merge a dispersed bundle's spurious second population (20°: N 2 → 1), at the cost of one
  60° crossing at one orientation. Kept.
- **The NNLS warm start** by `torch.linalg.lstsq` on the GPU took 55 s per 192³ block; the normal-equation
  solution is identical to 1e-6 and takes 0.04 s.
- **On the real block** (S-21-46, 192³, calibrated on 384³) the pipeline finds two populations and its single
  field agrees with the excised structure tensor to 8° over the block and 6° where the tensor is confident
  (95% within 15°).

## 28. The figures and the parameters, after the review (2026-10-17)

- **One figure layout everywhere**: image | excised structure tensor | shape space (the voxels' standardised
  features on two principal axes, coloured by the population whose activation is largest above the null) | one
  activation layer per population | the shape tensor's single field | the first vector of the multi-vector field
  (`viz.pipeline_row`). The validation, the HiP-CT block and BigBrain use it; the illustration's partner on the
  HiP-CT block (`figures.py overview`) mirrors the illustration's six columns instead.
- **The coherent texture setting is no longer the default validation.** It was introduced when detection failed
  at the calibrated texture, to show stage 2 working on tight bundles (8° dispersion); it is not the regime of the
  HiP-CT crop and its results (per-voxel separation on diagonal crossings, see §27) overstated what the data
  allows. The validation is now the calibrated texture only, on bundle 60°, crossing 60° and interface 40 (the
  90° crossing dropped); `--settings coherent` keeps the setting runnable.
- **The first vector of the multi-vector field is the useful crossing output at the calibrated texture**: 11.5°
  to the nearest population (58% within 15°) against 23° for the single field and 24° for the structure tensor;
  the pair is still unresolved (both within 15° at 0%).
- **Parameter sweep on the HiP-CT block** (`scripts/parameters.py`, `results/parameters.json`): the field's
  direction is set by the scales and by which populations are found; `tensor_floor` (0.3° between 0 and 0.1),
  `tissue` (2.7°) and `window` 64 vs 96 (0°) leave it; `nmax` 2, `min_sep_deg` 15 and `window` 0 all find the
  same alternative pair, 8.1° from the default's; `iterations` 4 and `min_sep_deg` 40 merge to one population;
  `smooth` 1 loses the second population (the null drops with the spiky activations); `sigma` 0.5 changes the
  teacher and the field (23°); `rho` 20 gives one population and a null four times larger.
- **Run-to-run variation**: the same block, calibration and settings gave 8.3° / 6.3° (§27) and 11.8° / 7.2°
  (block / where the tensor is confident) agreement with the excised tensor on two days; the readout's fit on the
  GPU is not bit-reproducible and the population pair found can differ. Worth a seeded, deterministic fit before
  release.

## 29. HiP-CT across voxel sizes, four regions, a whole brain (2026-10-17)

- **Six voxel sizes, one 192³ block each, σ 1 / ρ 10 voxels throughout** (`scripts/scales.py`, `results/scales.json`):
  I74 at 2.2 µm (VOI 2) and 10 µm (chunk), S-21-46 deep brain at 20 µm (the 20 µm scan) and 50 / 100 / 200 µm
  (levels 0–2 of the 50 µm scan, the same thalamic site). Agreement with the excised tensor, median over the
  tissue / where the tensor is confident: 2.2 µm 23° / 20° (3 populations, one of them a vessel); 10 µm 25° / 16°
  (3); 20 µm 6.2° / 4.8° (2); 50 µm 5.9° / 3.2° (2; the layers follow the visible fascicles); 100 µm 11.8° / 4.6°
  (2); 200 µm 9.9° / 5.7° (3 populations that are tissue classes: a white-matter band, the ventricular edge, the
  rest). The pipeline reads shape at its receptive field (16 voxels): at sub-cellular voxel sizes that is cells,
  vessels and single axons, and the voxel size decides what a "fibre" is.
- **Four regions at 20 µm (S-21-46)**, each calibrated on its own 384³: deep brain 2 populations, 6.2° from the
  tensor; gyral white matter 3, 12.7°; cerebellar white matter 2, 25.6° (tensor gap 0.08: little oriented
  texture); **brainstem 1 population, 2.4° — but the self-calibration is degenerate**: one fibre direction, 7 of 24
  teacher bins, null 2.8 (seven times the deep brain's), 6% of tissue above it. Self-calibration needs orientation
  diversity in the calibration block; the production rule is one calibration per volume on a diverse block
  (`run_volume`), and the regions figure's second row applies the deep-brain calibration to all four.
