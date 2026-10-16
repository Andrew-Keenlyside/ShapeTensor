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

## 14. Thresholds and gates for the false populations (`sheet_null_gates.py`, `dispersed_null.py`, `crossings_rel_sweep`; seed 0)

Share of white-matter sites with N ≥ 2 on single-population phantoms as the CSD threshold rises above the line-null
value, by band (0–0.1 / 0.1–0.25 / 0.25–0.5 mm / > 2 mm from the structure); and what the crossings keep at the
same thresholds.

**Full texture (P3-40, dispersed fibres, a 40-level step):**

| engine, ρ | rel 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| intensity 16 | 0 / 1 / 1 / 0% | 0% | 0% | 0% | 0% |
| intensity 32 | 0% | 0% | 0% | 0% | 0% |
| anatomix48 16 | 29 / 29 / 11 / 1% | 6 / 6 / 2 / 0% | 4 / 1 / 2 / 0% | 0 / 1 / 1 / 0% | 0% |
| anatomix48 32 | 25 / 29 / 17 / 0% | 0% | 0% | 0% | 0% |

Crossings at full texture, "both within 15°" (P4, seed 0), rel 0.3 → 0.7: anatomix48 ρ 32 — 90° 100 → 99%, 60°
**100 → 100%**, 45° 60 → 60%; anatomix48 ρ 16 — 90° 90 → 80%, 60° 94 → 87%, 45° 41 → 35%; intensity ρ 32 — 90°
97 → 97%, 60° 5%. The anatomix48 read-out at **rel 0.5–0.6** is therefore compliant at full texture (≤ 2% false
seconds on dispersed tissue and at the step, 0% far from it) and keeps the 60° crossing at 100% (ρ 32) / 92–94%
(ρ 16). The interface itself does not make populations at this texture level.

**Half texture (the real crop's level; P3-40-t0.5):**

| engine, ρ | rel 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| intensity 16 | 60 / 51 / 31 / 7% | 40 / 40 / 22 / 4% | 31 / 25 / 15 / 3% | 23 / 17 / 11 / 2% | 21 / 9 / 8 / 2% | — |
| intensity 32 | 42 / 33 / 38 / 3% | 25 / 25 / 25 / 2% | 12 / 12 / 12 / 2% | 8 / 4 / 12 / 2% | — | — |
| anatomix48 16 | — | 65 / 62 / 65 / 48% | 56 / 60 / 59 / 38% | 44 / 44 / 55 / 20% | 29 / 32 / 42 / 8% | 10 / 16 / 25 / 2% |
| anatomix48 32 | 62 / 62 / 62 / 51% | 62 / 62 / 62 / 43% | 62 / 62 / 62 / 36% | 58 / 58 / 58 / 27% | 38 / 50 / 38 / 4% | — |

Compliance far from the step needs rel 0.7 (intensity, ρ 16), 0.5 (intensity, ρ 32), 0.9 (anatomix48, ρ 16) or
≈ 0.85 (anatomix48, ρ 32); within 0.5 mm of the step the false share stays at 8–25% (intensity) and 10–50%
(anatomix48) even then. At these thresholds the 60° crossing at half texture is at 25–32% both-within-15° for the
intensity ODF (it was 31–35% at rel 0.3: the directions, not the detection, are the limit at this texture) and at
1–4% for anatomix48 (as before). The gyrus (P1-t0.5) behaves like the step: intensity ρ 16 rel 0.6 → 21 / 4 / 0 / 0%,
anatomix48 ρ 16 rel 0.7 → 71 / 62 / 32 / 0%. The laminar slab at quarter texture is read as two populations
everywhere by the intensity ODF (85–100% at every threshold: the lamination *is* a second gradient family at that
texture ratio), 16–35% by anatomix48. The vessel: intensity ρ 16 rel 0.6 → 30 / 36 / 22%, anatomix48 rel 0.7 →
1 / 5 / 5%.

**Gates.** The `cl` gate (null-calibrated on the no-interface phantoms) removes 0–33% of sites and never lowers the
false share among the rest; the `gm` gate removes *every* site within 0.5 mm of the step and the gyral boundary and
within 0.1 mm of the vessel (the bands disappear from the table), leaving the far-field rate. It is an exclusion
zone, which is the right thing for a tractography input: fixels are not reported where an interface dominates the
window, and the single direction there is the excised tensor's.

**What this settles for the N-population field.** (1) The threshold must be set on dispersed single-population
texture at the data's own texture level, not on straight lines: at full texture that is rel 0.5–0.6 for anatomix48
(crossings intact), at half texture 0.7 / 0.5 for the intensity ODF (ρ 16 / 32) and ≈ 0.9 for anatomix48. (2) The
coarse-gradient interface gate excludes the interface neighbourhood; the `cl` gate is not a substitute. (3) At the
real crop's texture level the multi-population read-out is, with a compliant threshold, a 90°-only detector with
11–18° directions; the 60° result holds at full texture only. The scorecard's "no false populations" row is met by
the intensity ODF at every texture level with these thresholds, and by anatomix48 at full texture; "crossings at
60°" is met by anatomix48 at full texture with a compliant threshold (rel 0.5–0.6, ρ 32: 100%).

**Addendum — no "N = 0" from the fODF.** On isotropic (correlated) noise the CSD read-out reports N ≥ 1 at 100% and
N ≥ 2 at 66–74% of sites at rel 0.5, and the strongest peak sits at 11–16× the fODF's isotropic mean — the same range
as a real fibre (16× at full texture, 7–10× at half). Deconvolution turns noise into peaks, so no amplitude floor on
the fODF can tell "no population" from "one population"; the N = 0 verdict (grey matter, CSF, isotropic tissue) has
to come from the tensor's anisotropy — the orientation gap (λ2−λ3)/λ2 and the planar measure — applied before the
fixels are read. `fixels_3d` now carries the interface exclusion (`interface_gate`, 3× the median coarse gradient,
the same rule as the excision) and the dispersed-tissue threshold (`peak_rel` 0.5) as defaults; the anisotropy floor
is the caller's gate on the field's `gap`.

## 15. The ρ-10 pass (tag `rho10`; intensity + anatomix48, 29 phantoms, 2 seeds; `scripts/bench/REPORT_rho10.md`, `SCORECARD_rho10.md`)

A smaller window than the suite's 8 / 16 / 32, at the user's working value. Median angle within 0.1 mm (U-fibres:
the 0.25–0.5 mm band), seed mean; clear tissue (> 2 mm) in the last column.

| phantom | plain | excised (ungated) | `cl`-gated | anatomix48 TCA | clear (plain / excised) |
|---|---|---|---|---|---|
| P3-40 | 5.6 | 4.6 | 5.5 | 7.6 | 3.7 / 3.7 |
| P3-40-t0.5 | 33.5 | **4.5** | 23.8 | 44.1 | 4.5 / 4.6 |
| P3-40-t0.25 | 56.4 | **9.1** | 9.1 | 60.5 | 10.1 / 10.1 |
| P3-15-t0.25 | 18.7 | 16.2 | 18.3 | 36.5 | 10.5 / 10.5 |
| P1-t0.5 (gyrus) | 28.7 | **4.2** | 15.6 | 26.9 | 2.6 / 2.6 |
| P1-t0.25 | 67.4 | **11.4** | 13.4 | 63.5 | 6.5 / 6.5 |
| P2-t0.5 (U-fibres) | 1.8 | 2.2 | 2.1 | — | — |
| P6 / P6-t0.5 (vessel) | 8.3 / 22.9 | **5.4 / 6.3** | 8.1 / 16.3 | 5.4 / 7.7 | — |
| P7-t0.5 / t0.25 (slab) | 5.1 / 11.0 | 5.4 / 11.1 | 5.1 / 10.9 | 5.9 / 11.3 | — |

- **Excision holds at ρ 10**: the step, the gyrus and the vessel come down to the clear-tissue level (4.5 vs 4.5°;
  9.1 vs 10.1°; 4.2 vs 2.6°; 6.3°), U-fibres and the slab are untouched. The scorecard's "interface fixed" row
  fails at ρ 10 only through P3-15-t0.25 (16.2°), where the *clear tissue itself* is at 10.5° — at a quarter of
  the real texture a 0.2 mm window does not average enough texture for any method; at half texture it does
  (4.5°). So ρ 10 is usable at the real crop's texture level and marginal below it.
- **Crossings lose resolution at ρ 10** (seed 0, CSD at the uncalibrated rel 0.3): anatomix48 60° found at 91% but
  both within 15° at 68% (ρ 16: 94%, ρ 32: 100%); 90° at 70%; 45° at 31%; the intensity ODF 17% / 41% / 3%.
  Multi-population read-outs want ρ ≥ 16.
- **Null calibration at ρ 10** (`results/csd_calibrated_rho10.json`): widths 22.9 / 25.0° (t1) and 28.9 / 30.0°
  (t0.5) for intensity / anatomix48; line-null thresholds 0.3 (t1), 0.7 / 0.5 (t0.5). The dispersed-tissue and
  interface tests (`sheet_null_gates_rho10.py`) repeat §14 at ρ 10: far from the step the intensity ODF is at 7 / 4 /
  2% for rel 0.7 / 0.8 / 0.9 and anatomix48 at 50 / 37 / 24% for 0.5 / 0.6 / 0.7 (compliance needs ≈ 0.9); the `gm`
  gate removes the 0–0.25 mm bands at the step and the gyrus; the laminar slab at quarter texture reads as two
  populations at 69–73% (intensity, rel 0.7) and 22–38% (anatomix48).
- **Boundary detector** at ρ 10: no arm reaches PR-AUC 0.9 (the coarse gradient at ρ 10 integrates too little; it
  is 0.85–0.96 at ρ 16).
- **Real crop at ρ 10** (`demo_real_crop.py --rho 10`, `demo_real_dec.py --rho 10`): excised vs plain direction
  0.35° median over bright tissue (4.6° in the top boundary-strength decile, 2.7% of sites > 30°); anatomix48 TCA
  36° from the tensor with gap 0.84 against 0.23; CSD (29°, 0.4) N = 1 / 2 / 3 at 43 / 39 / 18% of sites.
- **Speed at ρ 10** (A6000, 256³): tensor + eigen 1.42 s (11.8 Mvox/s), excised 1.50 s, the anatomix48 shape tensor
  + eigen 4.9 s (3.5×), K + TCA rank 3 on the ρ/4 site grid 16 s (11.5×), fixels at every voxel 323 s (228×: the
  site grid at ρ 10 is ρ/4 = 2 voxels, 8× more sites than at ρ 16); scipy CPU 41 s.

## 16. Anisotropic voxels: 20 × 20 × 50 µm (`scripts/bench/anisotropy_test.py`, seed 0; `results/anisotropy_test.json`)

Each phantom resampled 2.5× coarser along one axis by area averaging, read back three ways — *naive* (voxel-unit
filters, the anisotropy ignored), *spacing* (per-axis physical filters, `spacing=`), *PSF-matched* (the other two
axes blurred to the same resolution, then `spacing`), *resampled* (interpolated back to isotropic first) — for the
plain and excised tensors and the anatomix48 shape tensor; the isotropic run is the reference. Median angle to the
truth within 0.1 mm (first) and far from the structure (second), ρ 16 (ρ 10 in `anisotropy_test.json` is the same
picture):

| phantom (t0.5) | isotropic plain / excised | coarse axis | naive plain / excised | spacing plain / excised | PSF-matched plain | resampled plain | anatomix48 naive / spacing |
|---|---|---|---|---|---|---|---|
| P3 step 40 | 12.6 / 3.1 · 2.6 | **y** (across the fibres) | 39 / **1.9** · 1.5 | 89 / 88 · 89 | 88 · 87 | 88 · 76 | 41 / 82 · 10 / 76 |
| | | z (along the fibres) | 46 / 34 · 36 | **14 / 20** · 19 | 5.6 · 16 | 3.0 · 16 | 50 / 9.8 · 46 / 16 |
| P1 gyrus | 15.5 / 2.5 · 1.9 | **y** | 37 / **1.7** · 0.9 | 89 / 89 · 87 | 89 · 38 | 89 · 6.0 | 43 / 88 · 3.5 / 79 |
| | | z | 57 / 41 · 31 | **7.2 / 5.4** · 7.3 | 6.2 · 6.5 | 5.2 · 5.2 | 71 / 6.8 · 42 / 5.5 |
| P6 vessel | 14.8 / 3.3 (0.25–0.5 mm: 5.7 / 2.8) | **y** | 8.9 / **2.5** · 1.7 | 68 / 88 · 89 | 55 · 86 | 45 · 57 | 13 / 64 · 12 / 75 |
| | | z | 43 / 40 · 39 | **17 / 17** · 18 | 15 · 16 | 14 · 15 | 47 / 13 · 47 / 14 |

Two regimes, decided by the geometry:

- **Coarse axis across the fibres** (y here; the section-stack case, fibres in the section plane): every physically
  consistent reading fails outright — the direction snaps onto the averaged axis (84–89° for `spacing`, PSF-matched
  and resampled, plain or excised, intensity or anatomix48). The across-fibre texture has a period of ~4 voxels
  (80 µm) at the fine scale, at or below the Nyquist limit of 50 µm sampling along y, so the y-contrast that
  distinguishes "the fibre runs along y" from "y was blurred" is aliased away and y becomes the axis of least
  variation. The naive tensor gives 1.7–2.5° with excision — as good as isotropic — only because its 2.5×
  over-estimated y-derivative (6.25× in energy) happens to cancel the blur-and-aliasing loss at this texture scale
  and ratio; it is a coincidence of the numbers, not an estimator.
- **Coarse axis along the fibres** (z): the physical readings are right (5–20°; the gyrus at 5–7°, the dispersed
  step and the vessel at 14–20°, against 3–5° isotropic) and the naive tensor carries the geometric bias the unit
  test predicted and more (34–57°), as does naive anatomix48 (42–71°).

So the tensors are **not robust to 2.5× anisotropic voxels in general**: a coarse axis destroys the across-fibre
contrast of fibres running perpendicular to it, and no filter choice is right for both geometries. Per-axis
physical filtering is the correct estimator whenever the coarse axis keeps the texture resolved (the unit test's
oblique fibre: 23° → 1.7°), and the wrong one when it does not. The remedy is isotropic sampling at the
fibre-texture scale (≤ 25 µm here); short of that the coarse-axis gradient energy has to be treated as a prior,
not a measurement — an estimator the suite does not yet have. The real-data runs (BigBrain, the HiP-CT crop; results
log §17) are consistency checks against an isotropic reference and carry no truth.

## 17. Anisotropic voxels on real data (`scripts/demo_bigbrain.py --source bigbrain|s2146`; no truth)

The 20 µm block area-resampled 2.5× along y (the section axis), each reading compared with the *isotropic*
structure tensor at the same sites (median angle over tissue; in brackets the share > 30°). Orientation gap of
the isotropic tensor 0.25–0.26 in all three blocks: the per-site anisotropy of real white matter at these windows
is weak, which is what makes the direction easy to tip.

| block, ρ | naive ST | spacing ST | PSF-matched ST | resampled ST | spacing excised | naive / spacing anatomix48 |
|---|---|---|---|---|---|---|
| BigBrain, 10 | **15.9°** (23%) | 59° (74%) | 53° (68%) | 41° (59%) | 64° (83%) | 32° / 65° |
| BigBrain, 16 | **15.3°** (22%) | 62° (76%) | 56° (70%) | 45° (61%) | 67° (87%) | 29° / 67° |
| HiP-CT S-21-46, 10 | **11.7°** (11%) | 68° (90%) | 60° (80%) | 53° (67%) | 71° (94%) | 39° / 75° |

The real blocks behave like the phantoms' *across-the-fibres* regime (§16): every physically consistent reading
snaps onto the averaged axis (the DEC panels turn uniformly green = y, `real_S2146_rho10_aniso_axial.png`,
`bigbrain_rho10_aniso_axial.png`), and only the voxel-unit tensor stays near the isotropic field — 12–16°, by the
same coincidental balance. In the top decile of boundary strength the physical readings agree better with the
reference (6–16° for the resampled / PSF-matched BigBrain tensors), because there the interface's gradient, not
the texture, sets the direction and it survives the averaging. The isotropic references themselves are mottled
fields (the BigBrain one partly section striping), so these are consistency numbers, not accuracy; the accuracy
statement is the phantoms' (§16). Isotropic reference numbers for the record: excised vs plain 1.4 / 2.4° (BigBrain
ρ 10 / 16; 12° in the top boundary decile — the CSF and grey-matter steps are strong in a cell stain), 0.9° (HiP-CT);
the anatomix48 shape tensor 25–29° (BigBrain) and 42° (HiP-CT) from the structure tensor.

## 18. The shape feature space by orientation (`scripts/fig_feature_space.py`; `results/feature_space.json`; CPU)

Single-population phantoms with straight fibres at 12 in-plane angles (two texture realisations), the 45 / 60 / 90°
crossings; per-voxel features averaged over 3³ voxels from four depths of the network and from the jet; per layer
a PCA embedding coloured by angle, the channel tuning, the silhouette over the 12 angle classes, a linear 12-class
probe trained on seed A and tested on seed B (scored within ±15°; chance 25%), and the probe applied per voxel to
the crossings.

| shape space | probe ±15° | silhouette | tuning depth (median; channels > 0.3) | 60° crossing: within 15° of either truth | pop-1 voxels → 30° / pop-2 → 90° | 90° crossing: either / pop-1 / pop-2 |
|---|---|---|---|---|---|---|
| encoder 1 (layer 8, 16 ch) | 51% | −0.05 | 0.33; 10 / 16 | 74% | 60 / 50% | 58% / 47 / 41% |
| **encoder 2 (layer 15, 32 ch)** | **82%** | −0.06 | **0.80; 29 / 32** | **80%** | **63 / 58%** | 72% / 55 / 66% |
| layer 58 (48 ch) | 75% | −0.03 | 0.56; 38 / 48 | 32% | 17 / 30% | 60% / 35 / 47% |
| output (16 ch) | 67% | −0.04 | 0.35; 11 / 16 | 23% | 22 / 12% | 48% / 14 / 58% |
| jet (27 ch, no network) | 64% | −0.02 | 0.09; 0 / 27 | 73% | 65 / 62% | 75% / 54 / 73% |

- **The space is organised by orientation as a continuum, not as clusters.** The deep layers' first two
  principal components run through the angles in order (a hue gradient across the cloud), and a linear readout of
  the angle works in every layer (64–82%); but every silhouette is ≈ 0 — there are no gaps between orientations.
  Dividing the shape space therefore means finding *modes along its orientation axis*, not clusters in C
  dimensions; k-means in the full space found texture phase (§7) for this reason.
- **The division works per fascicle in the shallow, orientation-selective spaces and fails in the deep ones.** On
  the 60° crossing the encoder-2 probe colours each fascicle by its own orientation (two peaks at ~30° and ~75–90°;
  80% of voxels within 15° of a truth; 63% / 58% of each population's voxels assigned to it within ±15°), and the
  jet does nearly as well (73%; 65 / 62%); layer 58 and the output assign nearly every voxel to ~60°, the bisector — their
  receptive fields average the two populations before anything can separate them.
- So the separation stage has to run at the **fascicle scale** (a few voxels) in **encoder block 2** (or the
  jet), before the ρ-window integration; the deep features are the wrong space for it, independently of the
  method. This fixes the layer and the scale of the stage, and leaves its method — label-free mode / signature
  finding — as the open piece (`readout.populations`, `scripts/bench/separate_shape_space.py`).

![embedding](../img/feature_space_embedding.png)
![crossing](../img/feature_space_crossing.png)

## 19. Two stains of one block, read together (`scripts/bench/two_stain_test.py`; `results/two_stain_test.json`)

Phantom pairs with the same geometry and truth under different texture realisations / levels stand in for two
co-registered stains (BigBrain has one stain, so this stays a phantom result): a fibre-rich stain (P3-15 at full
texture) with a cell-like stain (P3-40 at quarter texture: weak fibre texture, strong boundary); two realisations
of P3-40 at half texture (the pure SNR control); the 60° crossing at full + half texture with its single-line null.
Read alone (A, B) and stacked (A+B: the intensity tensors add; the anatomix48 features of both are concatenated,
8 PCs each). Median angle within 0.1 mm · 0.25–0.5 mm · far, ρ 10 / ρ 16:

| pair | A | B | A+B (stacked tensor) | A+B (TCA on the stacked K) |
|---|---|---|---|---|
| fibre-rich + cell-like, intensity | 3.8 · 3.4 · 3.9 / 2.4 · 2.6 · 2.8 | 56 · 26 · 10 / 53 · 42 · 5.5 | 4.5 · 3.4 · 3.7 / 3.6 · 2.8 · 2.6 | 6.5 · 3.5 · 3.9 / 4.0 · 2.8 · 2.8 |
| fibre-rich + cell-like, anatomix48 | 6.0 · 4.6 · 5.3 / 5.1 · 4.5 · 4.6 (TCA 3.8 / 2.8) | 60 · 59 · 55 / 59 · 58 · 55 | 24 · 13 · 13 / 16 · 14 · 12 | 27 · 4.1 · 11 / 7.2 · 3.5 · 11 |
| two realisations, intensity | 29 · 6.1 · 4.4 / 12.5 · 6.7 · 2.6 | 31 · 6.4 · 4.6 / 12.9 · 6.7 · 3.0 | 28 · 4.7 · **3.0** / 12.5 · 6.2 · **1.9** | 38 · 6.1 · 3.6 / 38 · 10 · 2.4 |
| two realisations, anatomix48 | 43 · 16 · 16 / 38 · 21 · 15 | 46 · 23 · 16 / 41 · 30 · 15 | 45 · 19 · 16 / 40 · 27 · 15 | 56 · 32 · 22 / 54 · 48 · 44 |

Crossing at 60° (A full texture, B half), "found 2 / both within 15°", ρ 16: intensity ODF A 20 / 10%, B 78 / 28%,
A+B 29 / 14%; anatomix48 ODF A 98 / 46%, B 100 / 0%, A+B 100 / **51%** (ρ 10: 88 / 36%, 100 / 1%, 100 / 38%). Single-line null: ≤ 3.5% false seconds in
every combination.

Stacking is tensor averaging: two realisations of one stain gain the expected √2 far from structures (4.4 → 3.0°,
2.6 → 1.9°) and nothing at the interface, which both stains share; a fibre-rich stain plus a weak-texture stain
reads like the fibre-rich stain alone, slightly diluted; concatenating the shape features of a weak stain degrades
the joint shape read-out at the interface (6 → 24° at ρ 10) while the crossing keeps its resolution (46 → 51%); TCA on a stacked K is
worse than the stacked tensor (the CP picks the shared boundary piece). The merged field is as good as the best
stain, not better — stain selection, or per-stain weights, belongs in the merge stage.

## 20. Stage 2, round 1: dividing the shape space on the crossings (`separate_shape_space.py`, `orientation_readout.py`; small blocks, seed 0)

Three label-free divisions of a window's voxels, the truth split as the ceiling. Crossings at 90 / 60 / 45° (1:1)
and 60° at 4:1; "found 2 / both directions within 15°" over the windows; the ceiling column is a tensor per
population from the *true* voxel split.

| space, ρ | signatures (principal-signature memberships) | TCA over-complete | TCA BIC | **truth-split ceiling** (both ≤ 15° · error) |
|---|---|---|---|---|
| enc2, 10 — 90 / 60 / 45 / 4:1 | 0 / 0 / 0 / 0% | 100 / 59, 100 / 13, 100 / 11, 74 / 0% | 0 | 100% · 0.8 — 29% · 11.4 — 38% · 11.1 — 0% · 24.5 |
| enc2, 16 | 0% | 100 / 48, 100 / 4, 100 / 0, 76 / 0% | 0 | same |
| layer 58, 10 | 0% | 100 / 53, 100 / 12, 96 / 8, 74 / 3% | 0 | same |
| layer 58, 16 | 0% | 100 / 67, 100 / 25, 99 / 5, 61 / 0% | 0 | same |

- **Signatures fail as predicted by §18**: the principal variance of a window's channel-energy patterns is texture
  phase, so the second signature never clears the single-fibre null (N = 1 everywhere) and the first signature's
  direction is 23–75° off. The split must target the orientation axis, not the principal one.
- **TCA over-complete** finds two populations everywhere but the directions are right in 48–67% of windows at 90°
  and 4–25% at 60°; BIC never selects two (as in §6).
- **The ceiling itself is low**: with the true voxel split, a membership-weighted intensity tensor per population
  gives 11° errors at 60° (both within 15° in only 29–43% of windows) and 24° at 4:1, because the σ = 1 gradients
  of a voxel in one fascicle already mix the neighbouring fascicle of the other population. A tensor per population
  from intensity gradients is bounded by fascicle interdigitation at this density; the per-population direction
  has to come from the shape-space orientations themselves.

**The self-calibrated orientation readout** (ridge map from encoder-2 features to d dᵀ; teacher: the excised tensor
at ρ 8 on confident single-fibre voxels, 2.1° accurate; no labels): fit R² 0.65; per-voxel orientation on held-out
single fibres **10.6° median, 64% within 15°** (the label-trained probe of §18: 82%); on the crossings, voxels within
15° of either truth 50–60% (each population's voxels assigned to it 36–50% / 27–31%); window modes over-split
(N = 3 in 95–100% of windows with a 15° kernel and the single-fibre null), both modes within 15° at 58% (90°) and
0% (60°), mode directions 20–47° off. Layer 58: R² 0.24, 24° median per voxel; the jet: R² 0.47, 15.9°, 48%.

So after round 1: the orientation-carrying space is confirmed (encoder 2 > jet > layer 58), the division has to be
done on per-voxel orientations rather than on principal signatures or on gradient tensors, and the two things to
improve are the per-voxel readout's accuracy (a nonlinear map; the linear one leaves the cos²-type tuning on the
table) and the mode finder's width and null (over-splitting). Round 2 runs both (`OrientationReadoutMLP`, kernel
25°, a 15% mass floor per mode).

## 21. Stage 2, round 2: the per-voxel orientation readout in three forms (`orientation_readout.py --model ridge|mlp|dist`; seed 0, ρ 10, small blocks)

Same self-calibration for all three (teacher: the excised tensor at ρ 8 on confident single-fibre voxels, 2.1°
accurate; no labels). *Ridge*: a linear map to the orientation tensor d dᵀ. *MLP*: the same target through a small
network. *Dist*: a softmax over 160 hemisphere directions trained on a soft target around the teacher direction —
a voxel holding two populations keeps two bumps instead of their mean, and a window's population density is the
sum of its voxels' distributions. (A scoring bug in the first two rounds — the second population's mask rendered
from a fresh random draw, so it marked the wrong voxels — is fixed here; population 1 was unaffected.)

| encoder 2 | single fibres per voxel (median; within 15°) | 90°: voxels near a truth · pop 1 / pop 2 assigned · windows N = 2 · both ≤ 15° · pair error | 60° | 45° | 60° at 4:1 |
|---|---|---|---|---|---|
| ridge | 10.6°; 64% | 50% · 36 / 28 · 100% · 30% · 34° | 54% · 39 / 31 · 100% · 0% · 28° | 60% · 41 / 31 · 98% · 0% | 56% · 50 / 17 · 95% · 1% |
| MLP | 9.2°; 70% | 52% · 42 / 40 · 100% · 41% · 34° | 58% · 46 / 40 · 100% · 0% · 25° | 64% · 49 / 41 · 98% · 1% | 62% · 59 / 26 · 69% · 0% |
| **dist** | 10.0°; 66% | 61% · 51 / 46 · 80% · **75%** · **5.3°** | 59% · 55 / 32 · 95% · 19% · 21° | 67% · 59 / 34 · 41% · 10% · 16° | 70% · 69 / 20 · 15% · 0% |

The jet (no network) with the distribution readout: 90° 84% / 24% / 18°; 60° 24% / 24% / 8.7°; layer 58: N = 1 nearly
everywhere.

- **The form of the readout decides the crossing.** The two tensor-regression forms read every voxel of a crossing
  towards the bisector (the per-voxel histogram is one hump between the truths; the modes they produce are 25–34°
  off and over-split, N = 3 almost everywhere), because a voxel whose features already contain the neighbouring
  fascicle is regressed to the *average* orientation. The distribution form keeps the populations apart: on the
  90° crossing both directions are within 15° in 75% of windows with a 5.3° pair error — the first result of the
  stage that meets its target angle.
- **60° is the current limit**: two populations are found in 95% of windows but the pair is right in 19% (21°
  error); per voxel, 55% / 32% of each population's voxels are read within 15° of their own direction. The
  per-voxel distributions of the two populations overlap at 60° at this texture and smoothing (3³), so the
  summed density's two modes pull together. 45° and 4:1 are not resolved.
- Per-voxel accuracy on single fibres is the same in all three forms (9–11° median, 64–70% within 15°), so the
  gain of the distribution form is entirely in how it composes at a crossing, not in its accuracy.

What to work on next within the stage: the per-voxel distribution's sharpness and the mixing scale — a smaller
feature average (1–2 voxels instead of 3³) and a sharper target kernel (8–10° instead of 12°), the two encoder
blocks together (block 1's finer receptive field), and the readout's resolution of the 60° pair as the
pre-registered test; then the window modes' null and separation (25°) at that sharpness.

## 22. Per-population layers of the shape space (`readout.layers`, `scripts/fig_population_layers.py`; seed 0, ρ 10)

The pipeline's own form of the division: the feature vector at a voxel is a non-negative mixture of the
*signatures* of the populations present, ``f(v) ≈ Σ_r a_r(v) S_r``; unmixing gives one full-volume activation
``a_r(v)`` per population — its own image, read from the shape space, no mask — and the population's structure
tensor is that of its activation (``∇(a_r S_r) = ∇a_r ⊗ S_r``). The dictionary of signatures is the mean
standardised encoder-2 pattern per orientation over the calibration voxels and their teacher direction (the
excised tensor; no labels), plus a *background* signature from the voxels with the lowest teacher gap, which
absorbs what is not a population. The populations present are the modes of the slab's per-voxel distribution (§21).

**The signatures form a double-angle code.** The cosine similarity between the signatures of two orientations is
≈ cos 2Δθ (0.84 at 15°, 0.34 at 30°, ≈ 0 at 45°, −0.5 at 60°, −0.98 at 90°): encoder 2's orientation code is, to a
good approximation, a 2-dimensional (cos 2θ, sin 2θ) subspace — a steerable code. On single-fibre blocks, NNLS onto
a 60°-apart pair picks the own signature in 85–94% of fibre voxels; the orientation subspace holds ~8% of the
feature variance (texture phase holds the rest), and without the background signature the background's projection
on that subspace made one layer of every pair blotchy (its tensor 40–68° off). With it:

| crossing | modes (error to truth) | activation on own voxels vs the other's | layer 1 / layer 2 own-tensor error on own voxels (within 15°) | the image's tensor on the same voxels |
|---|---|---|---|---|
| 90° | 8.6° / 10.4° (+ a third at 45°) | 0.80 / 0.34 · 0.79 / 0.34 | **5.7° (90%) / 9.4°** | 33.5° / 34.1° |
| 60° | 4.5° / 4.9° (+ 26.6°) | 0.78 / 0.41 · 0.88 / 0.45 | **20.2° (29%) / 6.4° (97%)** | 28.3° / 28.1° |
| 45° | 4.9° / 11.2° (+ 18.5°) | 0.99 / 0.74 · 0.79 / 0.41 | **10.2° (90%) / 13.4°** | 20.8° / 23.2° |

Residual energy share 0.72–0.75 (0.92 without the background signature). So on 90° and 45° both populations'
layers carry their own direction to 6–13° where the image's tensor is 21–34° off; at 60° one layer is right (6.4°)
and the other is at 20° — the 90°-population's activation is still contaminated by the 30° fascicles at this
texture (its contrast 0.78 vs 0.41). The spurious third mode is the intersections.

![layers at 60°](../img/population_layers_c60.png)

Next within the stage: unmix in the orientation subspace itself (the dictionary's two principal directions) so
texture-phase variance cannot leak into an activation; sharper features (smaller average); and the layer's tensor
with the activation's *confidence* as its weight.
