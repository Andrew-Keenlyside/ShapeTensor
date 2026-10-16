# Implementation plan: `shapetensor` package and validation run

Approved 2026-10-16 (`TCA_FIELD_REVIEW.md` §10, with the BRIDGE reader deferred). Scope: build the package with a
pluggable feature engine and both read-out families, run the phantom suite, report. No BRIDGE changes.

## Package (`src/shapetensor/`)

| module | role |
|---|---|
| `filters.py` | torch separable Gaussian / derivative filters, scipy-identical kernels (`mode='nearest'`) |
| `features/` | the **feature engine**: `intensity` (1 ch; TCA reduces to the structure tensor), `jet` (multiscale Gaussian derivative jet, locally contrast-normalised, no network), `anatomix` (16-D output or 48-D layer 58; tiles with discarded margins; grid phases or a [1 2 1] Nyquist filter); `pca.py` compresses any engine to `n_pcs` |
| `tensor.py` | the channel-resolved tensor `K = G_rho * (g g^T)` on a site grid (products at full resolution, box-reduced by `b`, Gaussian at `rho/b`), optional gradient excision (fix A) with a spatial gate; the gradient-direction ODF in spherical harmonics on the same sites |
| `readout/single.py` | eigen (the structure tensor), TCA rank 3 (null direction of the pieces' Gram), boundary piece (normal, strength), Westin measures, quality |
| `readout/multi.py` | batched CP-ALS, geometric pairing, N by: BIC (baseline), FRT peaks, CSD peaks + TCA 2N refinement + stability, over-complete CP + grouping |
| `readout/hybrid.py` | H1 Gram blend switched by boundary strength (null-calibrated `s0`); H2 gated excision |
| `odf.py` | real SH basis (torch), Funk–Radon, CSD with a ring response, peak finding |
| `phantoms/make.py` | P1–P9 at any voxel size with twins and truth; `phantoms/stats.py` measures the real 20 µm crop (noise, levels, texture) to set them |
| `validate/` | `suite.py` runs arms × phantoms × seeds (K cached per site row), `score.py` metrics + CIs, `report.py` markdown + figures |
| `cli.py` | `shapetensor stats | phantoms | run | report` |

Data root: `/home/andrew/bridge_wmcov/shapetensor/` (phantoms, K caches, results, figures copied to `figures/tca_field/`).

## Run order

1. Unit tests: filter parity with scipy; `K` for the intensity engine equals BRIDGE's structure tensor; SH basis vs
   scipy; CP recovers a planted decomposition; tiled anatomix equals one pass to 1e-3.
2. Real-crop statistics → phantom parameters (reported).
3. P3 (steps 0/8/15/30) + P8 + P9: ρ sweep {4, 8, 16, 32} vox × engines {intensity, jet, anatomix16, anatomix48} ×
   3 seeds; single read-outs (eigen, TCA R3); boundary detectors (TCA share, Westin linear, coarse gradient); the
   [1 2 1]-vs-phases and n_pcs {8, 16} checks on P3 step 15.
4. Hybrid H1 / H2 on P1 (A/B), P2 (C/C0), P3, calibrated on P8/P9; pick the default.
5. N populations on P4 (lines 90/60/45/30°, fractions 1:1 / 2:1 / 4:1, singles, rods) and P5 (fan × bundle): FRT,
   BIC, CSD+TCA+stability, over-complete grouping.
6. P6 vessels, P7 laminar slab. Seeds to 5 for the chosen configuration.
7. Report: tables against the pre-registered thresholds, the "where ST fails" figures, the engine comparison.
8. Real-crop demonstration (qualitative): glyphs, boundary strength, npop on the S-21-46 20 µm crop.

Deferred: the tiled whole-volume driver + container + SGE script (after the suite); the BRIDGE reader (user).
