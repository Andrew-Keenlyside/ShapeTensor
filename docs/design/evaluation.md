# Evaluation

Every model is scored the same way, against criteria fixed before training (`scripts/eval_populations.py`, then
`scripts/acceptance.py`). Windows of 3 voxels unless stated; others can be added (`WINDOWS=6,12`).

## Synthetic, held out

Crops the model never saw, at strength 0.3 and 1.0. Targets are each query's populations from construction.

| # | what | criterion |
|---|---|---|
| A | orientation | median axis error of the matched population where one line holds >= 60 % of the window: no worse than a fitted structure tensor's |
| B | crossings | where two lines each hold >= 20 % and lie >= 45 deg apart: both axes found within 15 deg in >= 25 % of queries (the structure tensor reports one axis: 0 %) |
| C | populations | existence F1 above the trivial "every slot present" answer; share error below the trivial "one population holds everything" |
| D | shape | line / sheet / isotropic class of matched populations correct in >= 60 % (chance ~33 %) |
| E | learning | validation loss falls with training; A-D better at the end than halfway |
| F | no collapse | predicted existence rate near the targets'; predicted shapes vary |

Duplicate slots (two slots with nearly the same ellipse) are reported against the targets' rate.

## Known-truth phantoms

Bundles at several angles, crossings at 45 deg, 60 deg and 90 deg, and fibre-free texture, each at two texture
levels. Reported: the bundle's axis error, the share of crossing points where both directions are found within 15 deg,
and whether the texture is called isotropic. All are compared with the structure tensor.

`scripts/eval_populations.py`, `scripts/figure_populations.py`

## Structural tests

`scripts/replicate_pops.py` trains a head for a few hundred steps on fixed crops under random rotations and reports
the ellipse error by shape, the axis error of lines, crossings found and duplicate slots. It takes minutes and answers
one question: can this head read orientation from the images at all?

## Real images (after training only)

Validation against an independent measurement of the same specimen, for example ODFs from HiP-CT compared with
co-registered ex vivo diffusion MRI at matched scale. Real images are never used for training, calibration or model
selection. Visual inspection of real volumes (`scripts/figure_hipct_dec.py`) is a sense check only.
