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

## While the wide loops train (02:00-03:30 Montreal)

* The generator's line axes are near isotropic over 40 draws (mean |component| z, y, x = 0.47, 0.53, 0.51; share
  within 26 deg of an axis 0.075 / 0.107 / 0.105, isotropic 0.10): the "constant axis scores 22 deg" on unrotated
  draws came from few, coherent draws (8 with lines), not a generator bias -- the rotated metric removes it anyway.
* Built: the exact per-voxel density (`--exact`), descriptor densities (`--desc-lmax`), salience order (`--order
  salience`), query generality evaluation (`scripts/learn/query_eval.py`: scale sweep, off-lattice, half-resolution
  input), real-data density glyphs at several scales (`scripts/learn/real_density.py`), context evaluation
  (`scripts/learn/context_eval.py`).
* Context used by P1 (step 1000): its mixture output stops changing once the input is visible to a half-width of
  ~20 voxels around the scored block (change from full context 15 deg at 12, 5 at 16, 1 at 20, 0 from 24) -- the
  effective context is ~+-20 voxels, inside the receptive field (54); its error against the target is lowest at 16
  (12 deg) and 17.5 with full context (8 draws: noisy, but a hint that distant structure misleads it).
* Context B as a query input (design, not built): train the decoder with the input outside a random cube of
  half-width B around the queries replaced by the mean, B as an input; B beyond the receptive field needs a larger
  trunk (dilations or a coarser level) -- the context evaluation above is the test.

## Review 2 and the fixed bank (02:30-03:00 Montreal)

Review 2 (independent agent) reproduced the re-scored numbers and found: (1) the capacity claim is **not supported**
-- P1's loss schedule scaled with its 12000 steps (scalar terms off for its first 1000 steps, the late terms from 600
not 300, learning rate near 1e-3), its data were easier (strength 0.16-0.23, 2-3 line layers always) and buffer
larger; on the same draws the narrow flat-trained R2 matches P1 on single cells -- the narrow net can represent line
orientation; what fails is learning it from hard data under R4's schedule; (2) the metric is correct but its sampling
noise is large (90% intervals of ~10-20 deg over 16-48 draws; anything above ~45 deg is chance); (3) a salience-order
bug (isotropic ties with empty slots); (4) the context mask's hard step; (5) the half-resolution test too small for
the receptive field; (6) the new tools are ahead of the evidence (none trained yet). Recommended: density-first
training; an absolute schedule and a narrow/wide x schedule test; a fixed validation bank with intervals.

Done: absolute schedule flags (`--late-start`, `--scalar-window`); salience key + 1e-3 m (0 present-after-empty
cells); a tapered context mask, queries scored only where the context covers 1.5 sigma; 96³ half-resolution test;
the fixed bank `learn_night/bank/full_s03_64.pt` (64 line-bearing rotated draws, full generator 0.3, exact density
stored) and `scripts/learn/rescore.py` (pooled medians, 90% bootstrap intervals over draws, order-free slot score,
untrained floor). Stopped W1 (SH on top of all slot terms: does not test the density alone) and W3 (too many changes
at once); launched N1 (narrow, R4's data and settings, only the schedule changed: late terms from 600, scalars off) and
D1 (wide, the exact density only: every slot and scalar weight 0).

**The bank (90% intervals over 64 draws; ~16k single and ~36k shared line cells):**

| model | slot single | order-free single | slot shared | mixture single |
|---|---|---|---|---|
| untrained (narrow) | 59.4 [55.0, 64.1] | 58.9 | 56.5 [51.2, 61.0] | 61.7 |
| R2 (narrow, flat) | 16.0 [12.9, 20.5] | 13.5 [11.3, 16.5] | 43.4 [39.2, 47.6] | 13.2 [11.1, 16.2] |
| R4 (narrow, full) | 44.1 [41.1, 47.0] | 42.9 | 46.9 [43.6, 50.4] | 42.9 |
| R4d (+ --distinct) | 42.9 [39.7, 47.0] | 42.5 | 49.0 [46.8, 51.2] | 41.9 |
| L01 (narrow, backgrounds) | 38.8 [34.8, 43.1] | 38.2 | 46.8 [43.8, 49.9] | 35.6 |
| P1@1000 (wide, full + crossings, long schedule) | 19.3 [16.9, 22.6] | 18.7 | 37.7 [34.0, 42.3] | 18.1 |
| structure tensor | 15.2 [12.4, 19.0] | | 35.1 [32.5, 37.6] | |

R4 and R4d are indistinguishable; R2 is best on single cells, P1 best on shared cells (overlapping the structure
tensor); everything trained on hard data with R4's schedule is far worse than either.

## The curriculum result (03:40 Montreal)

| run (1000 steps unless stated) | change | val single / shared / mixture (48 rotated draws) | bank single / shared / mixture |
|---|---|---|---|
| W0 wide | -- (control) | 58.7 / 57.0 / 60.1 at 750: chance | (pending) |
| **W2 wide** | **curriculum: 0.8 -> 0.1 share of flat single-line draws by step 500** | **18.5 / 32.3 / 16.2** | **21.7 [19.3, 24.4] / 35.0 [31.8, 38.2] / 20.5** |
| N1 narrow, 2000 planned | late schedule only (late terms from 600, scalars off) | 58.5 / 55.9 / 55.6 at 750: chance -- stopped | |
| D1 wide, 1500 | exact density only (all slot / scalar weights 0) | SH nearest-peak 29.5 / 34.1 at 500 (floor 39.1 / 38.7) | (running) |
| structure tensor | | 13.7 / 32.9 | 15.2 / 35.1 |

With the same width and schedule, the curriculum turns chance into structure-tensor-level shared-cell orientation
in 1000 steps (W2 vs W0); neither width alone (W0) nor the late schedule alone (N1, narrow, at 750) did. P1's early
success fits: its data were line-rich and easy (crossings forced, strength 0.16-0.23). Review 3 scrutinises this.
Launched: C1 (narrow + curriculum: is width still needed?) and C2 (wide + curriculum + exact density + queries +
descriptors + context: the general model).
