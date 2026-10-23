# Overnight 2026-10-23: generator and training loops

Directive (00:40 Montreal): develop the generator and the training scheme to resolve the identified issues; short
training runs; an independent adversarial review after each run (at least ten loops, training in between); the
strongest and weakest area of each; push the theory towards any shape, context and resolution; until 07:00 Montreal.

Runs: `/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_night/`. Short runs: 64³, batch 4 x accum 4,
2000 steps unless stated, validation on 16 held-out draws of the run's own generator.

## Starting point

| run | data | slot axis (deg) | mixture axis (deg) | N agreement |
|---|---|---|---|---|
| R1 | one line layer, flat, clean | 5.2 | 4.9 | 0.82 |
| R2 | one line layer, flat, HiP-CT 0.5 | 6.1 | 6.1 | 0.80 |
| R3 | crossings, flat, clean | 16.4 | 7.6 | 0.37 |
| R4 | full generator, 0.3 | 42.4 | 18.4 | 0.27 |
| R5 | full generator, 0.15 -> 1.0 | 50.5 | 38.7 | 0.25 |
| O1 | full generator, overfit 8 samples, 1500 steps | 61.7 | 60.5 | 0.51 |

Diagnosis before the loops:

* R4's matched-slot error 41.7 deg, the best of its six slots 21.1 deg on the same cells: about half the error is slot
  ordering. Lines are the heaviest population in only 35% of their cells.
* The orientation loss is not dominated by unlearnable populations: 59% of it is from clearly linear line targets,
  their mean cost two-thirds of the way to an isotropic guess.
* Backgrounds: with a textured background the isotropic remainder and the region interfaces are visible populations,
  and lines are the heaviest in only 51-64% of their cells (100% on a flat background); the structure tensor's error on
  line cells rises from 6.8 deg (flat) to 15-19 deg -- harder but visible.
* The ablation A1 (R2 + backgrounds) was at chance at step 600 and 900, where R2 had learned by 600.
* Gradient on the trunk's last shared layer per loss term (weighted): on R2 the mixture term leads (1.36; presence
  0.16, shape 0.24); on R4 **presence leads (1.01)**, six times the shape term (0.17), its loss stuck at 2.9 (summed
  over six slots). Presence there asks whether each slot holds a population above 0.005 mass -- faint, near-threshold
  populations make it a guess -- and it ramps in at step 300, when orientation should form. Presence is also redundant
  with the mass (min_mass is an inference predicate). Fix tried: a don't-care band (0.002-0.02) and weight 0.2.
* Backgrounds look right but are genuinely harder (`learn_night/backgrounds_targets.png`): tissue texture makes
  elongated blobs at the fibres' scale (orientation visibility 0.55 in the example), cells add bright and dark blobs
  like fibre sections, regions add strong steps whose interface mass is large. On the backgrounds run (L01, step 800)
  every term's gradient at the stem is ~10x smaller than on R2 -- the input features barely train (an input-scale /
  contrast problem: tested by the local-contrast input, --lcn).
* R4d (shape-aware matching) = R4 within noise: matched 41.0 deg, best-of-six 20.4 (R4 41.7 / 21.1). The ordering
  across shapes is not what holds R4 back; "best of six slots" is biased low (a minimum over six axes), so the
  "half the error is ordering" reading was too strong.
* Generator throughput: under four-way GPU sharing one 64³ draw takes ~2.4 s against ~1.1 s for a training step;
  a host-to-device kernel copy in every Gaussian call was 3.0 of 10.5 s (cached now, output identical); the rest is
  per-scalar GPU synchronisations from drawing random scalars on the CUDA generator (changing that changes every
  sample -- deferred).
* Literature (independent survey agent): FOD networks regress SH coefficients (FOD-Net); 4th-order moments separate
  crossings from sheets (Advani & Tucker); unstable matching gives conflicting targets (DN-DETR) -- add a
  matching-free path; contrast invariance inside normalisation-free nets (divisive normalisation, GDN); gradient
  starvation (Pezeshki 2021) -- measure per-head gradients; synthetic-data models train for 10^5 steps (SynthSeg,
  anatomix); arbitrary-resolution decoders (LIIF) conditioned on the query scale.

## Loops

| loop | change | run | slot axis | mixture axis | N agree | reviewer: strongest | reviewer: weakest | next |
|---|---|---|---|---|---|---|---|---|
| 1 | shape-aware matching (--distinct 0.2) | R4d: full generator 0.3 | 41.6 (R4 42.4) | 18.5 (18.4) | 0.29 (0.27) | the targets: exact, matching-free density, peaks find axes to 2.9 deg | the slot output: mass-ranked set + Frobenius loss hedges to round tensors (linearity 0.11 vs 0.88) | fix the metric; SH alone; curriculum; salience order |

## Review 1 and the corrected metric (02:00 Montreal)

Review 1 (independent agent) found: (1) `evaluate()` used the training run's matching options, so `--distinct`
changed the metric itself (same weights at step 0: 53.1 vs 45.4 deg) -- scored under one rule R4d is no better than
R4; (2) "half the error is ordering" is not supported -- best-of-six is biased low (the untrained network scores
28.7), and on single-line cells, where there is no ordering choice, R4 is still 19 deg against 8.5 for a structure
tensor; (3) the gradient-dominance measurement was 2 batches and a stuck term keeps a large gradient by construction;
(4) the overfit test shows the 8-sample run failed, not that learning is blocked; (5) the slot output hedges towards
round tensors on the full generator. Recommended: one reference metric, the SH density alone, a curriculum from the
working regime, salience (anisotropic mass) ordering; hold the presence band and the query decoder.

Evaluation is now under one reference matching rule, on held-out draws each under a fixed cube rotation, with
single-line cells (the line is the only anisotropic population) and shared cells reported apart, next to a structure
tensor (gradient sigma 1, window 5) on the same cells (`ref_*` keys).

**Under the corrected metric the picture changes:**

| model | data scored | slot single | slot shared | mixture single | structure tensor single / shared |
|---|---|---|---|---|---|
| R2 (narrow, flat, 2000) | flat | 5.6 | -- | 5.6 | 5.4 / -- |
| R4 (narrow, full, 2000) | full | 46.0 | 47.7 | 47.5 | 11.3 / 36.5 |
| R2 | full | 13.4 | 39.7 | 10.7 | 13.8 / 37.5 |
| L01 (narrow, backgrounds, 1500) | full | 39.9 | 51.3 | 37.0 | 13.8 / 37.5 |
| **P1 (wide 32/64/128, full + crossings, 1000)** | full | **15.0** | **38.5** | **13.6** | 13.8 / 37.5 |
| R2 | backgrounds | 12.3 | 44.5 | 10.5 | 11.4 / 16.4 |
| **P1** | backgrounds | 19.7 | **19.1** | 16.6 | 11.4 / 16.4 |

* R4's apparent 42 deg was flattered by an orientation bias of the unrotated held-out draws (a constant axis scores
  22 deg there); rotated, it is at chance: the narrow network learned nothing on the full generator.
* **Capacity is the blocker**: the wide network (1.45 M parameters) reached the structure tensor's level on the full
  generator in 1000 steps; the narrow one (0.37 M) learned nothing in 2000. P1 had been stopped on the old metric; it
  is resumed from its step-1000 checkpoint as the long track.
* Training from scratch on hard data with the narrow network fails, while the narrow network trained on flat data
  transfers (13.4 single on the full generator): consistent with capacity plus a curriculum effect.
* Loops 2-4 (narrow, several changes at once) were stopped unfinished; the loops continue on the wide network, one
  factor at a time against a wide baseline (W0).
