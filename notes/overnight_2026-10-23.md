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
* The ablation A1 (R2 + backgrounds) was at chance at step 600, where R2 had learned.

## Loops

| loop | change | run | slot axis | mixture axis | N agree | reviewer: strongest | reviewer: weakest | next |
|---|---|---|---|---|---|---|---|---|
