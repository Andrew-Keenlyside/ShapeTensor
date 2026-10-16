# Results

The numbers below come from the phantom suite at 20 µm (two seeds unless stated), with the arms and thresholds of
[Phantoms and validation](phantoms.md). The complete tables, the run logs and the calibration files are in the
[results log](research/TCA_FIELD_RESULTS.md) and `scripts/bench/`.

![overview](img/graphical_abstract.png)

## The interface (P3, step 40)

Median angle to the true fibre within 0.1 mm of the step; ρ 8 / 16 / 32 voxels; clear tissue in brackets.

| texture | intensity tensor | anatomix16 TCA | anatomix48 TCA | jet TCA |
|---|---|---|---|---|
| t1 | 7.1 / 3.5 / 2.4 (4.6 / 2.5 / 2.0) | 7.2 / 3.8 / 3.5 (5.4 / 3.4 / 2.8) | 10.8 / 4.6 / 3.3 (4.7 / 3.1 / 2.8) | 10 / 9 / 8 (10.6 / 8.6 / 8.7) |
| t0.5 | **41 / 13 / 7** (5.9 / 2.8 / 2.1) | 56 / 41 / 12 (7.6 / 5.4 / 4.3) | 53 / 12 / 8 (7.7 / 6.4 / 6.6) | 16 / 14 / 11 (13 / 10 / 10) |
| t0.25 | 58 / 53 / 42 (14 / 5 / 3) | 60 / 60 / 57 (27 / 25 / 30) | 61 / 60 / 36 (30 / 32 / 32) | 34 / 33 / 33 (35 / 34 / 34) |

**Gradient excision** (single-seed pilot at t0.25, step 40): 7.9 / 2.7 / 1.8° where the plain tensor is
60 / 50 / 37°; at t0.1: 44 / 15 / 7.4° against 75 / 67 / 60°; clear tissue identical to 0.1°; 0–1% of voxels
excised. The null-calibrated `cl` gate reproduces it except at ρ 32; the TCA-dominance gate does not fire. The
hybrid blend or switch to a shape direction is never better than the intensity tensor.

**Boundary detectors**, precision–recall AUC for "within 0.5 mm of the step" (chance ≈ 0.1), ρ 8 / 16 / 32:

| detector | t1 | t0.5 | t0.25 |
|---|---|---|---|
| coarse gradient magnitude `gm` | 0.50 / 0.85 / 0.90 | 0.77 / 0.96 / 0.95 | 0.88 / 0.95 / 0.95 |
| Westin `cl`, intensity tensor | 0.09 / 0.07 / 0.07 | 0.21 / 0.25 / 0.06 | 0.57 / 0.88 / 0.74 |
| anatomix16 `cl` / TCA gap | 0.09 | 0.30 / 0.15 / 0.05 | 0.83 / 0.97 / 0.92 |
| anatomix48 TCA share / dominance | ≤ 0.17 | ≤ 0.35 | ≤ 0.63 |

## The gyrus (P1 / P2)

Within 0.1 mm of the grey/white boundary, white-matter side; twin without the step in brackets: at t1 the intensity
tensor is unaffected (4.7 / 3.3 / 2.6 vs 3.1 / 1.7 / 2.1); at t0.5 the shell appears but is thin and *shrinks* with
the window (34.5 / 16 / 5.0 vs 5.9 / 2.6 / 2.6); anatomix48 TCA 31 / 15 / 13, anatomix16 63 / 56 / 15, jet ~28.
The boundary-parallel share near the boundary sits *below* the truth's 58% (random ≈ 50%): at this resolution the
artefact is disorder, not wrapping. **U-fibres** (P2, the band 0.25–0.5 mm, true boundary-parallel fibres) are kept
by every method: intensity tensor 1.0 / 0.8 / 2.6° at t1 and 2.3 / 1.3 / 2.5° at t0.5, identical to the twin.

## N populations (P4 / P5)

CSD of the gradient-direction ODF, threshold null-calibrated per engine / ρ / texture (≤ 2% false second
populations on the single-fibre phantom); "found 2 / both within 15°", ρ 16 → 32:

| crossing | intensity ODF | anatomix16 | anatomix48 |
|---|---|---|---|
| 90°, t1 | 100 / 68% → 100 / 97% (5°) | 100 / 82% → 100 / 99% | 100 / 90% → 100 / 100% (5°) |
| 60°, t1 | 38 / 10% → 21 / 6% | 80 / 61% → 82 / 80% (5°) | 98 / 93% → **100 / 100% (4°)** |
| 45°, t1 | 0 | 25 / 13% → 11 / 5% | 44 / 41% → **61 / 61% (3°)** |
| 60° at 4:1, t1 | 0 | 2 / 1% | 52 / 21% → 68 / 29% |
| 60°, t0.5 (threshold 0.4–0.5) | 92 / 29% → 100 / 33% (13°) | 99 / 35% → **100 / 58% (11°)** | 100 / 0% (16–18°) |

On the fan crossed by a bundle (P5), the jet and intensity ODFs lead (100 / 91–100% and 85–99% at ρ 32), anatomix48
72–80%. The other N-selection methods on the same caches: FRT peaks resolve 90° only; **BIC never selects two
populations**; the over-complete CP grouping reports two almost everywhere (false second population on single-fibre
sites up to 100%); CSD + TCA refinement *degrades* the CSD directions; the residual-bootstrap stability filter
removes most true populations. Fractions, where two are found, are within 0.01–0.06 of the truth.

## Orientation inside the network

Per-channel selectivity between two crossing bundles: encoder block 1 — 4 / 16 channels above 0.2, max 0.42;
**encoder block 2 — 18 / 32, max 0.83**; layer 58 — 5 / 48; the output — none (0.06). A tensor on the channels that
prefer each bundle (chosen with the truth) recovers it: 8° / 7°. Label-free: unmixing a window's channel energies
over the tuning curves fails (the response to a crossing is not the sum of the single-population responses); a
per-voxel linear head reads orientation at 73% within ±15° on held-out single-population phantoms and separates a
90° crossing by voting (9° / 6°), not 60°.

## Excision across the families (final pass, 2 seeds)

Within 0.1 mm of the structure, ρ 8 / 16 / 32: plain tensor → excision ungated → excision gated by `cl`
(clear tissue of the same phantom in brackets).

| phantom | plain | excision, ungated | excision, `cl`-gated | clear |
|---|---|---|---|---|
| P3 step 40, half texture | 41 / 13 / 7 | **5.4 / 2.9 / 2.1** | 27 / 9.4 / 6.1 | 5.9 / 2.8 / 2.1 |
| P3 step 40, quarter | 58 / 53 / 42 | **11.9 / 5.5 / 3.1** | 12.1 / 5.5 / 3.5 | 14 / 5 / 3 |
| P1 gyrus, half | 34.5 / 16 / 5.0 | 37 / 18 / 9 | **5.2 / 2.5 / 1.9** | 3.3 / 1.9 / 1.3 |
| P1 gyrus, quarter | 68 / 65 / 13 | 64 / 57 / 53 | **15 / 6.6 / 2.7** | 8.9 / 3.3 / 1.9 |
| P2 U-fibre band, half (0.25–0.5 mm) | 2.3 / 1.3 / 2.5 | 2.3 / 1.5 / 2.9 | 2.8 / 1.7 / 2.4 | — |
| P7 slab face, quarter | 14.8 / 5.8 / 2.6 | 13.7 / 5.8 / 2.8 | 14.5 / 5.8 / 2.9 | — |

On the slab (one interface in textured white matter) the ungated form is right and the gate misses at ρ 8; at the
gyrus (grey matter and CSF in the block) the ungated form *harms*, because its threshold — 3× the block median of the
coarse gradient — is too low when much of the block is smooth, while the gated form removes the shell entirely at
every texture level. The default is therefore the gated excision with the reference median taken over white matter
(a pipeline's mask), under test post hoc as `excision_variants.py`. Excision also removes the vessel halos (P6:
1.3° at ρ 32 against 5.3° for the plain tensor).

![gyrus glyphs](img/P1-t0.5_glyphs.png)

## Against the pre-registered thresholds (`SCORECARD.md`, worst phantom per claim)

| claim | outcome |
|---|---|
| interface fixed (≤ 10° within 0.1 mm, step ≥ 15) | **met**: excision ungated 3.7° (ρ 32), 7.6° (ρ 16); `cl`-gated 6.1° / 9.4°; no shape-space read-out passes (best: jet eigen 7.6° at ρ 32, which is its floor everywhere) |
| nothing lost in clear tissue (≤ plain tensor + 1°, > 2 mm) | met by every excision and hybrid arm (3.1° vs 3.1° at ρ 32) |
| U-fibres kept (≤ 10°) | met by 77 / 84 arms (1.3–3°) |
| vessels (≤ 15° within 2 radii) | met by 52 / 84 arms; best excision ungated 1.3° (ρ 32) |
| crossings resolved at 60° (N ≥ 95%, both ≤ 15° at ≥ 90%) | **met** by anatomix48 CSD (100% at ρ 32, 93% at ρ 16); by no other arm |
| crossings resolved at 45° | not met (anatomix48 CSD 61%); order 12 under test |
| no false populations (≤ 2%) | met with the null-calibrated threshold (27 / 60 arms) |
| fractions at 4:1 (≤ 0.1) | met where two populations are found (0.04) |
| boundary detector (P / R AUC ≥ 0.9) | at the line: coarse gradient 0.90 (ρ 32), 0.85–0.96 at ρ 16 across texture levels |

The follow-ups (order-12 CSD on the 45° / 30° crossings, the PCA variants, the excision-variant test, the real-crop
demonstrations and the speed benchmark) are appended to the [results log](research/TCA_FIELD_RESULTS.md) as they
complete.
