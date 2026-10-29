# Evaluation

## Synthetic, held out

ODFs are scored by their **log-likelihood gain over a uniform ODF**, in nats per query, on crops the model never saw.
The scores are broken down by query kind: single populations, crossing cores, regions where the orientation is not
recoverable, and regions without line-like structure. Each score is compared with:

- the uniform ODF (gain 0);
- a fitted structure tensor (its line axis over a uniform floor, fitted on separate crops);
- the true ODF, which is the upper bound.

Calibration is checked as well: the model's sharpness should track its actual angular error. An independent
geometry source for held-out sets is planned, so the model is not only tested on its own generator.

`scripts/eval_density_ce.py`

## Known-truth phantoms

Bundles at several angles, crossings from 30° to 90°, and fibre-free texture, each at several texture levels. For each case:

- the axis error for single populations;
- the share of crossing points where both directions are recovered within 15°;
- how much orientation each method claims in texture that has none.

All are compared with the structure tensor.

`scripts/eval_phantoms.py`, `scripts/figure_odf_glyphs.py`

## Real images (after training only)

Validation against an independent measurement of the same specimen, for example ODFs from HiP-CT compared with
co-registered ex vivo diffusion MRI at matched scale. Real images are never used for training, calibration or model
selection. Visual inspection of real volumes is a sense check only.
