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

## Review 4 and the round plateau (04:00-04:20 Montreal)

Review 4 (independent agent, on the controls W0 and N1): both collapsed to **round tensors** -- W0's mixture loss is
0.997 of a round (I/3) prediction's, its predicted mixture anisotropy 0.028 against a target 0.45, matched slots'
linearity 0.04 against 0.88; it learned no orientation of anything, interfaces included (a deviatoric cosine by the
cell's heaviest kind: W0 +0.10 / +0.22 / +0.11 / +0.03 for line / interface / sheet / blob, untrained similar, **W2
+0.77 / +0.80 / +0.79 / +0.67**). W0's shape and mixture losses fall to exactly the full generator's round floors by
step ~120 and stay; from step 300 the late terms learn mass, presence, fraction ("where", not "which way"). W2 sat on
its own round plateau for ~250 steps and broke away between 250 and 500. W2 vs W0 is unconfounded (identical
baselines, the same validation draws, W2's full-generator draws a subset of W0's).

Why: the full generator carries 5.3x less non-round target energy than the simple regime (the mixture target is
the mass-weighted mean of 2.3 populations per cell -- blobs, oblate interfaces, crossing partners -- against 0.8);
only 39% of the slot shape term's round-floor energy is from learnable lines; 25% of draws have no line; after step
300 presence, Hellinger and fraction set the update direction (cosine with the total 0.92 / 0.88 / 0.92), the
orientation terms' mean gradient is 19% of the total and near orthogonal to it. Optimisation is more likely than "no
usable gradient", not proven. The targets are learnable and consistent: W2 scores better than W0 on W0's own
objective (2.44 vs 2.54). Missing cell: wide + late schedule without a curriculum.

Recommended: (1) a schedule gated on breaking away from the round floor; (2) sampling by anisotropy instead of a
hand-made regime; (3) drop presence or weight it <= 0.1. Built: `--gate R` (orientation terms only at constant rate
until the mixture loss / its round floor (EMA) < R, then the late terms over 300 steps, the scalars over the next
300, cosine decay) and `--aniso A0 N` (buffer pairs drawn in proportion to their non-round target energy ** alpha,
alpha annealed A0 -> 0 over N steps). Launched G1: wide, full generator, `--gate 0.85 --aniso 1 600`, presence 0.1,
1200 steps -- the category-free replacement for the hand-made curriculum.

Stopped P1 at ~2550 (checkpoint 2000 kept) and D1 at ~900 (checkpoint 750) to free the GPU for the curriculum runs
(C1 narrow, C2 general, C3 crossings) and G1. Bank:

| model | slot single | slot shared | mixture single | SH top-1 single | SH nearest-3 single / shared |
|---|---|---|---|---|---|
| **P1@2000** (wide, line-rich easy data, long schedule) | **15.6 [13.6, 18.1]** | **32.7 [28.7, 35.9]** | **15.1** | | |
| D1@750 (wide, exact density only, no curriculum) | 57.9 | 52.3 | 58.5 | 57.9 (untrained 58.9) | 34.2 / 33.7 (untrained 40.4 / 37.2) |
| structure tensor | 15.2 | 35.1 | | | |

P1 is the best model so far; D1 sat on the round plateau too (density-only training does not escape it without a
curriculum).
* P1@2000 on the real HiP-CT vessel (`learn_pilot/P1_wide_crossing/real_glyphs_step2000_vessel.png`): sticks and
  lobes follow the vessel loop's local direction around its whole length, but faint populations (above the engine's
  min_mass 0.01, presence on) fill the empty dark space -- a presence / mass calibration problem (presence is the
  term reviews 1 and 4 flagged; G1 weights it 0.1).

## Review 3 (04:15 Montreal)

Review 3 (independent agent, on W2): W2 vs W0 is a clean single-factor comparison (same seed stream, same
validation draws; W2 saw *less* hard data: 717 vs 1064 full-generator draws); the effect is real for this seed and
spread over modalities and backgrounds (single cells: W2 trails the structure tensor by 3-8 deg in every stratum;
shared cells: at or better than it in histology, mixed, tissue, HiP-CT, regions; worse in MRI 44.5 vs 34.5, CT 40.4
vs 37.7). Over-claims corrected: "not width" is untested (only a wide curriculum run), "not the schedule" rests on a
narrow run stopped early, curriculum vs a constant share of easy draws is untested -- what is supported is that easy,
line-rich examples in the mix let learning start; one seed. Hidden by the axis metric: W2's matched slots are nearly
round (linearity 0.20 vs target 0.86; P1 0.26, R2 0.71), the kind head collapses to "line", no model predicts a
sheet. Bugs: the curriculum window scaled with --steps; the buffer lags the draw share by ~64 steps; the metrics see
only line cells. Recommended: (1) separate orientation from shape magnitude in the slot loss; (2) an all-kinds simple
regime at a constant floor share, 2x2 x 2 seeds; (3) density-only + curriculum (D2).

Built: `--shape-loss devcos` -- the squared Frobenius distance of deviators is (|Dp| - |Dt|)^2 + 2 |Dp| |Dt| (1 -
cos), its angular part weighted by the *predicted* anisotropy, so a rounder guess pays less for a wrong axis (60 deg
off: sharp 1.50, half-round 0.92, round 0.67); the new cost weights the angle by the *target's* anisotropy, (|Dp| -
|Dt|)^2 + 2 |Dt|^2 (1 - cos) (60 deg off: sharp 1.50, round 2.00; right: 0) -- for slots and mixture, any kind.
`--simple-window` (absolute), the buffer's simple share logged, rescore: matched-slot linearity and false linear
lines on no-line cells (W2 0.20 / 0.01, P1 0.26 / 0.02, R2 0.71 / 0.59 -- R2 sees sharp lines everywhere, W2 and P1
predict round "line" slots). Launched S1 = W2 + `--shape-loss devcos` (absolute window 500).

## Review 5 (04:30 Montreal)

Review 5 (independent agent, on D1 = density only, no curriculum): D1 learned nothing image-dependent -- its l >= 2
per-order losses equal a round prediction's (0.866/0.800/0.662/0.463 vs 0.870/0.800/0.662/0.462), angular energy
0.1% of the target's; the bank SH "gains" were peak-geometry artefacts (top-1 57.9 vs 59.7 against shuffled axes); the
plateau persisted without any presence / Hellinger / fraction terms, which **refutes the term-competition story** of
review 4 -- the signal is weak on hard data. The exact density target is sound (order 0 = sqrt(4 pi) slot mass,
ratio 1.000; top peak 3.5 deg from the slot axis on single-line cells; label axes within 1.9-5.4 deg of the rendered
geometry); it differs from the per-layer form only where orientation visibility varies inside the window (a
definitional difference; the exact form is linear and coarsens exactly). Bugs: the per-batch normaliser of the SH
loss down-weights line-rich batches (an anti-curriculum, correlation -0.46); val loss averages per-draw normalised
losses (empty draws dominate); peaks at rel 0.3 without an absolute gate find 2-3 peaks in noise (detection = false
second = 100% for the untrained net). Recommended: calibrated peaks; a decoupled density loss (order 0 apart,
per-order magnitude + target-weighted angle over running energies, order 2 first); sampling by the density's
angular energy.

Built (aba4c34): `peaks_gated` (a cell counts if its mean density > 0, <= 20% of the sphere negative and the top
peak stands 0.5x the mean above it; a second peak at (f2 - mean) / (f1 - mean) >= 0.42, the target's 10% false
rate) -- with it D1 and the untrained net give no peaks at all; `--sh-loss decoupled` with order 2 first (all orders
once the mass-weighted order-2 cosine passes 0.3); `--aniso` uses the density's angular energy when the target has
one. Launched D2: wide, density only, decoupled loss, curriculum (window 350), angular-energy sampling, 700 steps.

## Review 6 and the final model (04:40 Montreal)

Review 6 (independent agent, on P1@2000, paired bootstrap over draws, extra banks at strength 0.6 and with crossings
forced): P1 matches the structure tensor on single cells (+0.4 [-1.8, 2.0]); its shared-cell gain is real but small
(-2.5 [-4.9, -0.3]) and comes from line + interface cells (-6.5), not crossings (-2.0 [-5.1, +0.3]); R2 (never saw
an interface) is as good there -- a property of any learned model. **P1 does not learn the second population**: its
two most salient slots are 9.4 deg apart against target crossings of median 60 deg, the second a copy of the first
(salience 0.67 of it); pair error rises as half the crossing angle (a one-axis model); no better than R2. N agreement
0.31 is below a constant N = 1 (0.39); a ~0.07 line mass everywhere, the kind head says "line". It generalises to
draws without crossings, not to strength 0.6 (worse than the structure tensor by 4.2 [1.0, 9.2]; it never saw > 0.34).
The resume changed nothing (cosine continued at ~1e-3, no decay ever). Bugs: training evaluation still used the
uncalibrated peaks (fixed, 35d8cc4); `--resume` with `--gate` closes the gate again (use `--init`); the matched-slot
shared metric rewards copying the dominant axis (compare mixture heads and pair / copy / random-second controls).
Recommended: continue P1 rather than start fresh, with a density head (the only matching-free route to second
populations), train at the strength evaluated, de-duplicate slots and decide presence by mass.

Final model launched: F1 = `--init` P1@2000, 900 steps, lr 5e-4 with warmup 50 and the gated schedule (opens at ~50
for a warm start, then decays -- P1 never decayed), buffer 128, strength 0.3 -> 0.7, ~75% crossing draws (`--simple
{"crossing": true} --p-simple 0.5 0.5`), exact density head with the decoupled loss, presence 0.1, checkpoints
every 300 steps. C2 (the five-change general model) stopped at 250 to make room (three reviews called it
confounded; its parts are covered by tests and by D2 / F1).

## C1: narrow + curriculum (05:25 Montreal)

| run (1000 steps, same curriculum) | bank slot single | shared | mixture single | slot linearity |
|---|---|---|---|---|
| C1 narrow 16/32/64 | 42.0 [36.7, 47.7] | 49.0 [45.4, 52.3] | 39.1 [34.1, 44.2] | 0.10 |
| W2 wide 32/64/128 | 21.7 [19.3, 24.4] | 35.0 [31.8, 38.2] | 20.5 [18.0, 23.1] | 0.20 |
| untrained narrow | 59.4 | 56.5 | 61.7 | 0.68 |

Review 7 (independent agent): the C1 - W2 comparison is clean (same pairs, batches, schedule); paired bank
differences C1 - W2: single +20.3 [15.7, 25.0], shared +13.9 [11.3, 16.7], mixture +18.6 [14.3, 22.6] -- width
matters with the curriculum. But C1@1000 is no better than narrow without a curriculum (= R4@2000 within the
intervals): "learns, but slowly" is not supported; the gap opens at steps 325-450, when the late terms ramp in and the
easy share falls, and does not close. Shared cells cannot show separation (the metric does not detect it). Width is
now 32/64/128 by default; narrow runs dropped. (L01's bank row is its step-2000 checkpoint.) Fixed: the untrained
floor per width in rescore (it used the first run's), paired differences in rescore, a code version in the cache key.
* G1 (wide, `--gate 0.85 --aniso 1 600`, presence 0.1, no hand-made curriculum): its mixture loss fell below 0.85 of
  the round floor at step 574 -- it broke away from the round plateau without a hand-made regime (bank pending).
* F1 (P1@2000 continued with the density head): gate open at 51 (warm start), all SH orders on at 51 -- the density
  head on P1's trunk reached a mass-weighted order-2 cosine of 0.36 within 50 steps (P1's features carry it).
* D2 (density only, decoupled loss, curriculum): no order-2 alignment above 0.3 by step 575.

## S1, C3, D2 on the bank (05:50 Montreal; paired against W2)

| run (wide) | change vs W2 | slot single | shared | mixture single | paired single vs W2 | SH head (calibrated peaks) |
|---|---|---|---|---|---|---|
| S1 | `--shape-loss devcos` | 46.9 | 46.1 | 51.5 | **+25.2 [+21.0, +28.4]** | -- |
| C3 | crossing-rich curriculum + exact SH head | 21.5 | 37.2 | 20.9 | -0.2 [-2.1, +2.1] | top-1 single 25.7, false second 0.07, crossings found 0.06 (target 0.52) |
| D2 | density only, decoupled loss, curriculum, 700 steps | (untrained slots) | | | | top-1 87.1 (gated out), false second 0.53, crossings 0.46 (noise) |

* The deviator loss **hurt badly** (orientation +25 deg worse, slots no sharper: linearity 0.18): a negative result
  (review 8 investigates why).
* Crossing-rich easy data did not change the slots (C3 = W2) and its density head learned single-line orientation
  (25.7) but almost no second populations (6% against the target's 52%).
* Density-only training failed again even with the decoupled loss and the curriculum (D2); trained alongside the slot
  terms (C3) the density head learns -- the slot terms seem to bootstrap the trunk.

## Reviews 10 and the gated recipe (06:00 Montreal)

Review 10 (D2): D2 learned order 0 and the order-2 *magnitude* (size 0.65 of the target) but never its direction
(target-weighted cosine +0.09; C3's MSE density head +0.63); S1 vs W2 shows the same target-weighted-angle form
failing on slots and mixture -- both decoupled forms fail from scratch where Frobenius / MSE work (hypothesis: MSE
shrinks to the conditional mean and accumulates weak directional evidence from a round start; the decoupled form fits
size independently of confidence and its direction gradient goes as 1/|p|). The energy sampler worked against the
curriculum: the density-energy weight goes as mass^2 and easy draws carry *less* angular energy (fixed: scale-free
angular share per occupied cell). Untrained high orders (D2 trained l = 2 only) were scored; the order-2-only state
is not saved in checkpoints. Recommended: per-order MSE over a running per-order energy, all orders, alongside the
slot terms; repair or drop the sampler; a late magnitude term for orders >= 4 to recover second peaks.

**G1 (gated schedule + anisotropy-weighted sampling + presence 0.1, no hand-made curriculum, 1200 steps): the best
model of the night** -- bank slot single 14.5 [12.4, 18.0], shared 32.4 [28.7, 35.9], mixture single 14.1 [12.0,
16.8], linearity 0.23; paired against W2: single -7.3 [-9.0, -4.9], shared -2.7 [-4.9, -0.5], mixture -6.4 [-8.1,
-4.4]; better than the structure tensor (15.2 / 35.1) and P1@2000 (15.6 / 32.7 / 15.1). Its gate opened at step 574.
Confounds (1200 vs 1000 steps, a different rate trajectory, presence 0.1, one seed) under review 11. G1's sampler used
the slot-based weight (linear in mass), not the density-based one review 10 faulted.
