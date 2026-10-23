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
* Literature (independent survey agent): FOD networks regress SH coefficients (FOD-Net); 4th-order moments separate
  crossings from sheets (Advani & Tucker); unstable matching gives conflicting targets (DN-DETR) -- add a
  matching-free path; contrast invariance inside normalisation-free nets (divisive normalisation, GDN); gradient
  starvation (Pezeshki 2021) -- measure per-head gradients; synthetic-data models train for 10^5 steps (SynthSeg,
  anatomix); arbitrary-resolution decoders (LIIF) conditioned on the query scale.

## Loops

| loop | change | run | slot axis | mixture axis | N agree | reviewer: strongest | reviewer: weakest | next |
|---|---|---|---|---|---|---|---|---|
