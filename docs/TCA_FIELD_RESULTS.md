# TCA field validation: results log

Status 2026-10-16, evening. Everything on phantoms at **20 µm voxels** with statistics matched to the real S-21-46
crop. Single seed unless stated; the full suite (tag `main`, 2 seeds, ρ 8 / 16 / 32 voxels) is running. Numbers are
median angles to the true fibre in degrees; "within 0.1 mm" means the band 0–0.1 mm from the interface.

Data: `/home/andrew/bridge_wmcov/shapetensor/` (`results/<tag>/REPORT.md`, figures in `figures/<tag>/`).

## 1. Calibration of the phantoms to the real crop (label-free)

Measured on the 768³ 20 µm S-21-46 crop (`results/real_stats_S2146.json`, `real_cl_S2146.json`):

| statistic (bright tissue, > 142.5) | real crop | phantom baseline (t = 1) |
|---|---|---|
| noise, DoG bands 1–2 / 2–4 / 4–8 / 8–16 vox (10% smoothest blocks) | 3.9 / 2.1 / 1.1 / 0.8 | 4.6 / 2.9 / 1.2 / 0.4 (white 95.7, blurred σ 1.1) |
| texture, same bands, noise removed | 4.2 / 2.9 / 2.1 / 2.1 | 3.3 / 3.0 / 2.3 / 1.7 |
| intensity modes | 164 (WM-like), 126 | levels 164 / 126 / 60 |
| structure-tensor orientation gap (λ2−λ3)/λ2, median, ρ 8 / 16 / 32 | 0.25 / 0.26 / 0.30 (p10 0.11, p90 0.49) | ≈ 0.64 (dispersion 20° / 10°) |
| Westin linear cl, median at ρ 32; share > 0.3 | 0.115; 5.8% | 0.03; ~0% |

The real noise is **correlated** (Paganin), not white. The real tissue is **less coherent and more interface-rich**
than any homogeneous fibre model: matching the orientation gap needs 30° dispersion at 0.6× amplitude, which halves
the texture bands. So the suite keeps a coherent-bundle baseline (amp 100, dispersion 20°/10°) and sweeps the
**texture amplitude t ∈ {1, 0.5, 0.25, 0.1}**; by the orientation gap the real median sits between t0.25 and t0.5,
its 10th percentile near t0.1. Every claim below is read along that axis.

## 2. Engine numerics (tests, `tests/test_anatomix.py`)

- Tiles with a 64-voxel margin vs one pass (fp32, TF32 off): 0.7% of layer-58 values differ by > 1% of peak (a tile
  edge is a half-space perturbation the UNet answers slowly), but the **orientations move 0.03° (median), 0.07° (p90)**.
  bf16 autocast: the same order. Default: tile 256 / margin 64 / bf16; the output head's ceiling is 352³ (tile 224).
- De-checkerboarding: the 8-phase average changes TCA orientations by 0.03° against none; the [1 2 1] filter by 0.4°.
  Scores on P3-15 (tag `var-nonyq`) agree with the phase-averaged run to 0.1°. The suite runs **without** it (8× cheaper).

## 3. Pilot 1 (tag `pilot`): strong texture, step 15, ρ 4–32

| arm, ρ 16 | P3-15 within 0.1 mm | P3-15 > 2 mm | P3-0 (no step) | P8 homogeneous |
|---|---|---|---|---|
| intensity ST (eigen) | 0.7 | 0.3 | 0.3 | 0.2 |
| anatomix16 TCA R3 | 0.4 | 0.4 | 0.4 | 0.2 |
| anatomix48 TCA R3 | 1.4 | 1.3 | 1.4 | 1.5 (eigen 4.1) |
| jet TCA R3 | 3.4 | 3.0 | 3.1 | 3.0 |

**At the calibrated texture a step of 15 does not break the structure tensor.** The 0.1 mm phantoms' 42–62° came from
sub-voxel, sparse fibres (weak gradients) against a step; at 20 µm with resolved fascicle texture the step is a
minor term. ρ 16 (0.32 mm) is best for every arm; ρ 4 is noise-limited, ρ 32 mixes the angle bands.

## 4. Pilot 2 (tag `pilot2`): the step × texture grid, seed 0

Within 0.1 mm of the interface (ρ 8 / 16 / 32):

| phantom | ST | anatomix16 TCA | anatomix48 TCA | jet TCA | **fix A excision** | excision gated by cl |
|---|---|---|---|---|---|---|
| P3-40 (t1) | 5.0 / 3.0 / 1.8 | 3.1 / 1.1 / 1.1 | 3.5 / 2.3 / 1.6 | 3.1 / 2.5 / 4.0 | 3.4 / 2.0 / 1.4 | 4.9 / 2.8 / 1.7 |
| P3-15-t0.25 | 21.7 / 8.0 / 4.4 | 27.9 / 13.4 / 7.9 | 32 / 32 / 32 | 34 / 32 / 33 | 20.3 / 7.5 / 4.0 | 21.4 / 8.0 / 4.4 |
| **P3-40-t0.25** | **60 / 50 / 37** | 61 / 60 / 57 | 59 / 59 / 33 | 30 / 29 / 31 | **7.9 / 2.7 / 1.8** | 8.4 / 2.8 / 5.7 |
| P3-15-t0.1 | 61 / 52 / 35 | 45 / 52 / 54 | 55 / 46 / 44 | 50 / 39 / 37 | 60 / 48 / 18 | 61 / 51 / 32 |
| P3-40-t0.1 | 75 / 67 / 60 | 62 / 61 / 61 | 61 / 61 / 51 | 44 / 45 / 37 | **44 / 15 / 7.4** | 44 / 15 / 7.4 |

Clear tissue (> 2 mm, or the no-step twin), ρ 8 / 16 / 32:

| texture | ST | anatomix16 TCA | anatomix48 TCA | jet TCA |
|---|---|---|---|---|
| t1 | 0.7 / 0.3 / 1.6 | 0.7 / 0.3 / 1.2 | 1.2 / 1.2 / 1.6 | 2.9 / 2.9 / 5.2 |
| t0.25 | 9.8 / 3.8 / 2.2 | 12.6 / 7.5 / 7.0 | 26 / 30 / 30 | 32 / 31 / 32 |
| t0.1 | 44 / 22 / 8.7 | 60 / 63 / 62 (random) | 43 / 43 / 43 | 50 / 40 / 37 |

Excision costs nothing in clear tissue (identical to ST to 0.1°). Its excised share is 0–1% of voxels (the step).

### What this establishes

1. **The boundary failure is a step-to-texture-ratio effect.** It appears at t ≤ 0.25 with a 40-level step (the real
   IC/thalamus step is ~38) and thickens with ρ (the 0.25–0.5 mm band at P3-40-t0.25: 15 / 37 / 32 for ST). In
   textured WM it is a 1–5° effect at 20 µm.
2. **In the failure regime, shape-space TCA does not rescue the orientation at 20 µm.** The 16-D output is blind to
   weak texture (random in clear tissue at t0.1: it was trained to drop appearance). The 48-D layer has a 26–30°
   floor at t0.25 and still carries the step (59° at the interface). The locally normalised jet is step-insensitive
   (~30° at the interface and in clear tissue alike) but orientation-poor. None approaches the intensity tensor's
   clear-tissue accuracy at weak texture, which with a large window is close to the optimal estimator.
3. **Fix A (gradient excision by the coarse gradient magnitude, ungated, then normalised integration) does rescue
   it**: 7.9 / 2.7 / 1.8° at P3-40-t0.25 against 60 / 50 / 37°, with no clear-tissue cost. At t0.1 it works at ρ 32
   (7.4° vs 60°) and partially at ρ 16; at ρ 8 there is no texture signal to recover (clear tissue itself is at 44°).
   A step of 15 at t0.1 is below the detector's threshold (3× the median coarse gradient) and is not excised.
4. The null-calibrated cl gate reproduces ungated excision except at ρ 32 (5.7 vs 1.8); the TCA-dominance gate does
   not fire. The hybrid blend / switch by any detector is no better than ST (the shape direction it switches to is worse).
5. The 0.1 mm result (48-D TCA 7° at D15) therefore does not transfer to 20 µm with realistic statistics: there the
   network saw strong sub-voxel-fibre texture through a 0.8 mm receptive field; at 20 µm the receptive field is
   0.16 mm and weak texture is below what the features encode.

Open from the pilots: whether the 48-D floor is the PCA to 16 channels (variants `var-pcs48` / `var-pcs8` queued),
the N-population phantoms (P4 / P5), the gyrus (P1 / P2) and the laminar / vessel cases — all in the running suite.

## 5. The full suite (tag `main`): interim, P3 family, 2 seeds, dispersed phantoms

`results/spec_full.json`: 45 phantom instances × 2 seeds × 4 engines × ρ 8 / 16 / 32, with the hybrid and excision
arms, N-population methods (BIC, FRT, CSD + TCA refinement + stability, over-complete grouping) on P4 / P5, and the
null calibration per texture level. Report: `results/main/REPORT.md`. The hybrid / excision arms run last.

Within 0.1 mm of the step (clear tissue in brackets), ρ 8 / 16 / 32, seed-averaged:

| phantom | ST | anatomix16 TCA | anatomix48 TCA | jet TCA |
|---|---|---|---|---|
| step 40, t1 | 7.1 / 3.5 / 2.4 (4.6 / 2.5 / 2.0) | 7.2 / 3.8 / 3.5 (5.4 / 3.4 / 2.8) | 10.8 / 4.6 / 3.3 (4.7 / 3.1 / 2.8) | 10 / 9 / 8 (10.6 / 8.6 / 8.7) |
| step 40, t0.5 | **41 / 13 / 7** (5.9 / 2.8 / 2.1) | 56 / 41 / 12 (7.6 / 5.4 / 4.3) | 53 / 12 / 8 (7.7 / 6.4 / 6.6) | 16 / 14 / 11 (13 / 10 / 10) |
| step 40, t0.25 | 58 / 53 / 42 (14 / 5 / 3) | 60 / 60 / 57 (27 / 25 / 30) | 61 / 60 / 36 (30 / 32 / 32) | 34 / 33 / 33 (35 / 34 / 34) |

With dispersion the shape engines lose their small advantage at t1 (pilot 1) and the picture of pilot 2 holds: at the
realistic texture level (between t0.25 and t0.5) the intensity tensor fails at ρ 8 and is marginal at ρ 16 next to a
40-level step; the shape engines fail there too; the locally normalised jet is **step-insensitive at every level**
(its interface error equals its clear-tissue error) but orientation-poor.

**Boundary detectors**, PR-AUC for "within 0.5 mm of the step" (chance ≈ 0.1), step 40, ρ 8 / 16 / 32:

| detector | t1 | t0.5 | t0.25 |
|---|---|---|---|
| coarse intensity gradient `gm` (fix A's) | 0.50 / 0.85 / 0.90 | 0.77 / 0.96 / 0.95 | 0.88 / 0.95 / 0.95 |
| Westin cl, intensity tensor | 0.09 / 0.07 / 0.07 | 0.21 / 0.25 / 0.06 | 0.57 / 0.88 / 0.74 |
| anatomix16 cl / TCA gap | 0.09 / 0.09 / 0.09 | 0.30 / 0.15 / 0.05 | 0.83 / 0.97 / 0.92 |
| anatomix48 TCA share / dominance | ≤ 0.17 | ≤ 0.35 | ≤ 0.63 |
| jet TCA share | 0.10 / 0.18 / 0.33 | 0.13 / 0.31 / 0.53 | ≤ 0.11 |

The coarse gradient magnitude is the boundary detector; `cl` detects a step only when it dominates the window (which
is exactly when the orientation is at risk, so it is the right *gate*); the TCA pieces do not name the step reliably
at 20 µm. The 16-D output sees the step as a boundary only when the texture is weak.

## 6. N populations (P4 crossings, 2 seeds, `results/csd_calibrated.json`)

CSD of the gradient-direction ODF (SH order 8, ring response fitted on the single-population phantom: width 23–25°
at t1, 29–30° at t0.5), peak threshold **null-calibrated** per engine / ρ / texture level (the smallest relative
threshold with ≤ 2% false second populations on the single-fibre phantom). "found 2 / both < 15°" = share of sites
where two populations are reported / where both are within 15° of the truth; errors are medians.

| crossing | intensity ODF, ρ 16 / 32 | anatomix16, ρ 16 / 32 | anatomix48, ρ 16 / 32 |
|---|---|---|---|
| 90°, t1 | 100 / 68% (8°) · 100 / 97% (5°) | 100 / 82% · 100 / 99% (5°) | 100 / 90% · 100 / 100% (5°) |
| 60°, t1 | 38 / 10% · 21 / 6% | 80 / 61% (7°) · 82 / 80% (5°) | 98 / 93% (5°) · **100 / 100% (4°)** |
| 45°, t1 | 0 · 0 | 25 / 13% · 11 / 5% | 44 / 41% (5°) · **61 / 61% (3°)** |
| 60° at 4:1, t1 | 0 · 0 | 2 / 1% · 0 | 52 / 21% · 68 / 29% (11°) |
| 60°, t0.5 (threshold 0.4–0.5) | 92 / 29% (15°) · 100 / 33% (13°) | 99 / 35% (13°) · **100 / 58% (11°)** | 100 / 0% (18°) · 100 / 0% (16°) |

Other N-selection methods on the same caches (uncalibrated, tag `main`): FRT peaks resolve 90° only; **BIC never
selects two populations**; the over-complete CP grouping reports two populations almost everywhere (false second
population on single-fibre sites: anatomix16 100%, jet 50–82%, anatomix48 33–36%); CSD + TCA refinement *degrades*
the CSD directions (e.g. anatomix48 at 90°, ρ 32: both < 15° 100% → 20%); the residual-bootstrap stability filter
removes most true populations. Fractions, where two populations are found, are within 0.01–0.06 of the truth.

**P5, a fan crossed by a straight bundle** (crossing angle varying continuously; CSD at threshold 0.3, 2 seeds;
found 2 / both < 15°, ρ 16 → 32): at t1 the jet ODF is best (100 / 91% → 100 / 100%), the intensity ODF next
(77 / 73% → 85 / 85% at 50–70°, 92 / 91% → 99 / 99% at 70–90°), anatomix48 71 / 71% → 80 / 80%, anatomix16
47–62%. At t0.5 every engine still *finds* two populations at 65–100% of sites but the directions are at the 15° limit
(both < 15°: intensity 62 / 32%, jet 64 / 34%, anatomix48 65 / 57% at ρ 32). FRT fails on the fan at every ρ.

## 7. Where orientation lives inside anatomix (user question, 2026-10-16 afternoon; CPU experiments)

Shape-space *feature vectors* do not separate fibre populations: on a 60° crossing the two bundles overlap completely
in the layer-58 feature space, k-means clusters follow texture phase and background, and a tensor per cluster gives
one bundle or the bisector (`figures/tca_field/shape_space_clusters.png`). Projecting the features onto a TCA pair's
channel-signature subspace does not help either (signature |cos| 0.6–0.9 between bundles; the projected tensor is the
bisector). But the **encoder** is orientation-selective and the output head removes it:

| layer | channels with \|selectivity\| > 0.2 (bundle 1 vs 2 energy) | max | tensor on bundle-1 / bundle-2 preferring channels (oracle split) |
|---|---|---|---|
| encoder 1 (layer 8, 16 ch) | 4 / 16 | 0.42 | 2° / 62° · 52° / 10° |
| **encoder 2 (layer 15, 32 ch, ½ res)** | **18 / 32** | **0.83** | **8° / 67° · 55° / 7°** |
| layer 58 (48 ch) | 5 / 48 | 0.46 | 2° / 62° · 52° / 10° |
| output (16 ch) | 0 / 16 | 0.06 | — |

Encoder-2 channels have clean orientation tuning curves (depth 0.34–0.98 over 12 in-plane angles;
`figures/tca_field/encoder_tuning.png`). Two label-free ways to use it, trained / calibrated on phantoms only:

- *Window-level unmixing* of channel energies over the tuning matrix (NNLS) fails: the response to a crossing is not
  the sum of the single-population responses, and the dominant peak sits at the bisector for 90°, 60° and 45°.
- *A per-voxel orientation head* (multinomial logistic regression, 32 channels → 12 angle classes) reads a voxel's
  orientation at 73% within ±15° on held-out single-population phantoms (chance 25%). Voting over a window's voxels
  separates the **90° crossing** (peaks 9° and 6° from the two truths; a tensor on each population's voxel set gives
  the same), but at 60° and below the vote histogram is too broad and only one population is found (6–12° from the
  nearer truth). So encoder block 2 does encode population, decodable linearly at ±15–20°; a nonlinear head, 3-D
  angle sampling and sharper classes are the upgrade path, and the CSD-on-ODF route remains the stronger
  N-population estimator today (60° at 100% with the layer-58 features at full texture).

What this establishes: the N-population field should be **CSD on the gradient-direction ODF**; no single engine
dominates across geometries (anatomix48 on tight straight crossings, the jet and the intensity ODF on the fan), so
the engine stays a configuration choice and the intensity ODF is the no-network default. At full texture the
48-channel features earn their place (60° at 100%, 45° at 61%, against 90°-only for the intensity ODF); at the
realistic texture level every engine finds the 60° crossing but with 11–18° errors, and the anatomix48 advantage is
gone (its threshold has to rise to 0.4–0.5 to stay null-safe). SH order 12 is the follow-up for 45° and below.

## 8. The full suite complete (tag `main`, 18:25; `scripts/bench/REPORT_main.md`, `SCORECARD_main.md`)

Every family at every texture level, two seeds, the eight arms of `report.ARMS` including the two excision arms
(`excise_all` ungated, `excise_cl` gated by the null-calibrated `cl`). Median angle within 0.1 mm, ρ 8 / 16 / 32,
seed mean:

| phantom | plain tensor | excision, ungated | excision, `cl`-gated | clear tissue (plain) |
|---|---|---|---|---|
| P3-40-t0.5 | 41 / 13 / 7 | **5.4 / 2.9 / 2.1** | 27 / 9.4 / 6.1 | 5.9 / 2.8 / 2.1 |
| P3-40-t0.25 | 58 / 53 / 42 | **11.9 / 5.5 / 3.1** | 12.1 / 5.5 / 3.5 | 14 / 5.5 / 3.1 |
| P3-40-t0.1 | 74 / 72 / 60 | 49 / 31 / 9.7 | 49 / 31 / 9.7 | 48 / 27 / 9.5 |
| P3-15-t0.25 | 24.5 / 9.4 / 4.0 | 20.6 / 7.6 / 3.7 | 23.7 / 9.2 / 4.0 | 14 / 5.5 / 3.1 |
| P1-t0.5 (gyrus) | 34.5 / 16 / 5.0 | **5.2 / 2.5 / 1.9** | 18.5 / 11.5 / 4.0 | 3.3 / 1.9 / 1.3 |
| P1-t0.25 | 68 / 65 / 13 | **15 / 6.6 / 2.7** | 17.5 / 7.6 / 2.7 | 8.9 / 3.3 / 1.9 |
| P2-t0.5 (U-fibre band 0.25–0.5 mm) | 2.3 / 1.3 / 2.5 | 2.8 / 1.7 / 2.4 | 2.4 / 1.7 / 2.0 | — |
| P6 (vessel, 2 radii) | 10.7 / 4.7 / 1.5 | **7.0 / 3.2 / 1.3** | 10.2 / 4.5 / 1.4 | — |
| P6-t0.5 | 24.7 / 11.8 / 5.3 | **8.2 / 3.3 / 1.3** | 23.9 / 9.7 / 3.3 | — |
| P7-t0.5 (slab face) | 6.5 / 3.2 / 1.2 | 6.8 / 3.5 / 1.3 | 6.5 / 3.3 / 1.2 | — |
| P7-t0.25 | 14.8 / 5.8 / 2.6 | 14.5 / 5.8 / 2.7 | 14.7 / 5.8 / 2.6 | — |

(An earlier draft of this section had the ungated and gated columns swapped at the gyrus and the vessel, read
from a mis-labelled summary; the table above is computed from the per-seed JSON, `results/main/hybrid/`.)

What the interim pass could not show:

- **The ungated excision is right everywhere.** It clears the gyrus (5.2 / 2.5 / 1.9°; 15 / 6.6 / 2.7° at quarter
  texture) and the vessel (8.2 / 3.3 / 1.3°) as it clears the step, at no cost in clear tissue or on the U-fibres,
  and is neutral on the laminar slab, whose face the plain tensor already reads at its clear-tissue level. The
  `cl`-gated form under-excises wherever the step does not yet dominate the window (27° at the step and 24° at the
  vessel at ρ 8, 18° at the gyrus): the gate opens late. The post-hoc variants (§11) confirm that neither a
  white-matter reference median nor a `gm` gate improves on it.
- **P3-40-t0.1** (a tenth of the real texture, ~4× the real step-to-texture ratio) is beyond any window below ρ 32:
  excision brings the interface to the clear-tissue level (49 / 31 / 9.7 vs 48 / 27 / 9.5°), but that level is
  itself poor because the texture is too weak. The step-15 at quarter texture is fixed at ρ ≥ 16 and not at ρ 8
  (20.6° against 14° clear).

The scorecard (`src/shapetensor/validate/scorecard.py`, worst phantom per claim, seed mean of per-seed medians):

| claim (threshold) | pass / arms | best arm |
|---|---|---|
| interface fixed (≤ 10° within 0.1 mm; P3-40-t0.25, P3-40-t0.5, P3-15-t0.25) | 6 / 84 | excise_all ρ 32 **3.68°**, ρ 16 7.59°; excise_cl ρ 32 6.1°, ρ 16 9.4°; jet eigen ρ 32 7.6°, ρ 16 9.2° |
| nothing lost in clear tissue (≤ plain + 1°, > 2 mm; P3-40*, P1*) | 47 / 84 | every excision and hybrid arm (3.1° vs 3.1°) |
| U-fibres kept (≤ 10°, band 0.25–0.5 mm; P2*) | 77 / 84 | 1.3–3° |
| vessels (≤ 15° within 2 radii; P6*) | 52 / 84 | excise_all ρ 32 1.33° |
| crossings 60° (both ≤ 15° at ≥ 90%) | 2 / 60 | anatomix48 csd ρ 32 **1.00**, ρ 16 0.93 |
| crossings 45° | 0 / 60 | anatomix48 csd ρ 32 0.61 |
| no false populations (≤ 2%) | 27 / 60 | null-calibrated CSD |
| fractions 4:1 (≤ 0.1) | met where two are found | 0.04 |
| boundary detector (PR-AUC ≥ 0.9; P3-40*) | 1 / 60 | gm ρ 32 0.90 (ρ 16: 0.85–0.96 by texture) |

The jet eigen read-out passes the interface claim at ρ 32 by being *flat* (7–8° everywhere, interface and clear
tissue alike): it does not see the step, and does not see the fibre well either. The "no shape-space read-out passes"
statement of the review stands for the anatomix engines.

Glyph figures for ten phantoms (truth, five arms, CSD fixels, sticks coloured by error) are in
`figures/tca_field/glyphs/` (`scripts/fig_glyphs.py --tag main --rho 16 --seed 0`); the gyrus one is on the docs
results page.

## 9. CSD at spherical-harmonic order 12 (`scripts/bench/run_lmax12.py`, `results/lmax12_crossings.json`, seed 0)

The ring width refitted at order 12 on the single-population phantom; peak threshold 0.3 (0.2 gave the same
directions and up to 4% false second populations). "N2 / both within 15° / median errors", order 12 against order 8
in brackets:

| crossing | intensity ρ 16 | intensity ρ 32 | anatomix48 ρ 16 | anatomix48 ρ 32 |
|---|---|---|---|---|
| 90° | 100 / 33% (100 / 68%), e 8 / 16° | 100 / 42% (100 / 97%), e 8 / 14° | 100 / 36% (100 / 90%), e 10 / 13° | 100 / 48% (100 / 100%), e 8 / 13° |
| 60° | 99 / 27% (38 / 10%), e 19 / 8° | 100 / 27% (21 / 6%), e 20 / 7° | 100 / 63% (98 / 93%), e 9 / 9° | 100 / 66% (100 / 100%), e 8 / 9° |
| 45° | 45 / 21% (0), e 14 / 10° | 31 / 11% (0) | 100 / **70%** (44 / 41%), e 9 / 10° | 100 / **75%** (61 / 61%), e 6 / 10° |
| 30° | 0 | 0 | 94 / 36% (—), e 7 / 19° | 96 / 38% (—), e 7 / 21° |
| single, error / false 2nd | 2.5°, 0% | 0.6°, 0% | 7.2°, 0% | 7.2°, 0% |

Order 12 *detects* the tighter crossings (45° at 100% N = 2 with the layer-58 features, 30° at 94%) but every
direction it returns is less precise: errors of 8–13° on the 90° crossing that order 8 resolved to 4–5°, and 7.2° on a
single population against < 3°. The share "both within 15°" therefore falls on the crossings order 8 already
resolved, and the net is a trade, not a gain. The cause is the sharper response acting on an ODF sampled at the
same window: the deconvolved peaks are narrower than the sampling of the peak grid and ride on the ringing of the
truncated series. The upgrade path is a finer peak grid with Newton refinement and a Laplace–Beltrami regularised
deconvolution, not a higher order on its own. The anatomix48 ρ 32 cell first ran out of memory (the order-12 basis on
all channels at full resolution, 10.7 GB in one allocation); `odf_sites` now bounds the basis to ~2 GB per pass and
the rerun shows the same trade at ρ 32 (45° at 75% against 61%; 90° and 60° down from 100% to 48% and 66%).

## 10. PCA variants (`scripts/bench/spec_var_pcs*.json`, seed 0, ρ 16, P3-40 family)

anatomix48 with 8 principal components instead of 16: P3-40-t0.25 TCA 59.4° within 0.1 mm (16 PCs: 60°), P3-40
4.5° (16 PCs: 4.6°). The number of components is not what limits the layer-58 read-out at the interface. The
48-component (no PCA) variant ran out of memory on this GPU (48 fp32 channel gradients at 480³) and was not run.

## 11. Excision variants (`scripts/bench/excision_variants.py`, seed 0; `results/excision_variants.json`)

Within 0.1 mm / clear tissue, ρ 8 / 16 / 32, share of voxels excised in brackets — ungated with the block median (UB), ungated with the white-matter median (UW), gated by the null-calibrated `gm` or `cl` with the white-matter median:

| phantom | UB | UW | gated `gm` | gated `cl` |
|---|---|---|---|---|
| P1-t0.5 | 5.4 / 2.5 / 2.1 (3.2%) | 6.2 / 3.1 / 2.7 (2.0%) | 11.6 / 6.7 / 3.9 | 20.3 / 10.4 / 4.1 |
| P1-t0.25 | 16.6 / 7.1 / 3.0 (2.8%) | 16.6 / 7.3 / 3.0 (2.6%) | 17.8 / 8.6 / 5.6 | 18.0 / 7.9 / 3.2 |
| P3-40-t0.5 | 5.9 / 3.1 / 2.4 (0.7%) | 5.9 / 3.1 / 2.4 (0.7%) | 5.9 / 3.3 / 2.7 | 31.0 / 9.4 / 5.8 |
| P3-40-t0.25 | 14.3 / 6.4 / 3.6 (0.9%) | 14.3 / 6.4 / 3.6 (0.9%) | 14.6 / 6.8 / 4.9 | 14.4 / 6.4 / 4.1 |
| P6-t0.5 | 7.2 / 3.3 / 1.4 (1.1%) | 7.1 / 3.3 / 1.4 (1.1%) | 7.5 / 4.6 / 2.1 | 30.6 / 11.7 / 4.1 |

The reference median makes no difference (the block median is already dominated by white matter in every phantom; the white-matter median excises slightly fewer voxels at the gyrus and is slightly worse); any gate is worse than none. Fix A as specified stays the default.

## 12. Speed (`scripts/bench/speed.py`, RTX A6000, ρ 16)

| arm | 128³ (2.1 Mvox) | 256³ (16.8 Mvox) | Mvoxel / s | relative |
|---|---|---|---|---|
| `structure-tensor` package algorithm, CPU (scipy, one process) | 3.9 s | 47.1 s | 0.4–0.5 | 22–33× slower |
| structure tensor + eigen, GPU (`st.structure_tensor_3d`, `eig_special_3d`) | 0.18 s | 1.42 s | 11.8 | 1 |
| excised tensor + eigen, GPU | 0.19 s | 1.51 s | 11.1 | 1.06× |
| fixels: ODF (order 8) + CSD + peaks at every voxel, GPU | 5.1 s | 40.6 s | 0.41 | 29× |

The first run failed inside `torch.linalg.eigh`: cusolver's batched Jacobi solver rejects batches of ~2M matrices and above (`CUSOLVER_STATUS_INVALID_VALUE`), so `eig_special_3d` now chunks at 2²⁰.

## 13. False populations on sheets and boundaries (`scripts/bench/sheet_null.py`, seed 0; `results/sheet_null.json`)

The "no false populations" claim was calibrated and tested on the straight single-population line phantom only. On
phantoms with one population and a sheet-like structure the CSD read-out (order 8, the suite's kernel, the
null-calibrated threshold of that engine / ρ / texture) reports N ≥ 2 at these shares of white-matter sites, by
distance from the structure; after the arrow, the same among the sites the null-calibrated `cl` gate keeps:

| phantom | engine, ρ, threshold | 0–0.1 mm | 0.1–0.25 | 0.25–0.5 | > 2 mm | `cl`-gated out |
|---|---|---|---|---|---|---|
| P7 slab, half texture | intensity 16, 0.4 | 0 → 0% | 4 → 4% | 3 → 3% | — | 0% |
| | intensity 32, 0.3 | 25 → 25% | — | 8 → 8% | — | 0% |
| | anatomix48 16 / 32 | 0% | 0% | 0% | — | 0% |
| P7 slab, quarter | intensity 16 / 32 | **100%** | 100% | 100% | — | 0–1% |
| | anatomix48 16, 0.5 | 35% | 38% | 26% | — | 1% |
| | anatomix48 32, 0.4 | 88% | — | 42% | — | 0% |
| P1 gyrus, half | intensity 16, 0.4 | 69 → 83% | 32 → 40% | 5 → 4% | 0% | 22% |
| | intensity 32, 0.3 | — | — | 37 → 33% | 0% | 33% |
| | anatomix48 16, 0.5 | 99 → 98% | 90 → 89% | 81 → 80% | **11%** | 22% |
| | anatomix48 32, 0.4 | — | — | 100% | **52%** | 33% |
| P3 step 40, half | intensity 16, 0.4 | 60 → 84% | 51 → 65% | 31 → 33% | 7% | 4% |
| | intensity 32, 0.3 | 54% | 46% | 50% | 6 → 5% | 2% |
| | anatomix48 16, 0.5 | 65 → 97% | 62 → 87% | 65 → 69% | **48%** | 4% |
| | anatomix48 32, 0.4 | 62% | 62% | 62% | **51%** | 2% |
| P6 vessel, half | intensity 16, 0.4 | 51 → 37% | 60 → 45% | 40 → 35% | — | 6% |
| | anatomix48 16, 0.5 | 29% | 38 → 33% | 37% | — | 6% |
| | anatomix48 32, 0.4 | — | 81% | 75% | — | 1% |

Three things follow.

- **A sheet makes populations.** The gradient-direction ODF of a sheet is one peak; deconvolving it with the ring
  response gives a ring of fibre directions, and the peak finder reports two or three of them. Near every interface
  — the step, the gyral boundary, the vessel wall, the slab face at quarter texture — the read-out finds N ≥ 2 at
  30–100% of sites. The `cl` gate at its null threshold removes 2–33% of sites and *raises* the false share among
  the rest (it removes the sites where the sheet dominates, not those where it merely competes).
- **The straight-line null does not transfer to dispersed tissue.** Far from any structure (> 2 mm) the intensity
  ODF keeps the false share at 5–7% (the target was 2%) but the anatomix48 ODF reports a second population at
  48–52% of sites on the dispersed single-population texture of P3 and at 11–52% on P1 — its threshold, set on
  straight lines, is far too low for fibres with the real crop's dispersion. This is why the calibrated real-crop
  demo still showed N ≥ 2 at 53% of sites.
- **The slab at half texture is clean** (0–4% at ρ 16, both engines), so a laminar texture as such is not the
  problem; the false populations come from *contrast* interfaces, and from under-calibration.

Consequences for the N-population field: the null calibration must be done on dispersed single-population
texture (the clear tissue of the P3 family, not the straight lines), and fixels need an *interface* gate (the coarse
gradient magnitude, which detects every interface above, not `cl`) before they enter a tractography. Both are
measured in §14 (`sheet_null_gates.py`: thresholds 0.1 and 0.2 above the line-null value; `cl`, `gm` and both as
gates). Until that holds, the "no false populations" row of the scorecard is met on straight lines only.
