# Real HiP-CT test of the population readout (27 Oct, 07:00 ET)

Asked: "Run a test in real hipct data".

`scripts/learn/real_populations.py`: the 20 um deep-brain crops of five specimens (`/hdd/andrew/brain_atlas_20um_test_crops`,
SynthSeg labels on the same grid; checks only, never training). Per specimen one crop holding both white matter
(SynthSeg 2/41) and grey matter (cortex, thalamus, caudate, putamen), the default readout on a 3-D grid of points at
window sigma 2.5 voxels. Three models:
- the overnight pilot (13000 steps; trained on synthetic phantom families only);
- yesterday's direct-path pilot (phantoms only);
- the generator model Z4 (6000 steps on the scene generator; given the voxel size).

A structure tensor (gradient sigma 1, window 5) is the classical baseline. Results:
`/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/pilot_entities/overnight/real/hipct_populations{,_x4}.{json,png}`.

## Results (means over the 5 specimens; AUC = white vs grey matter, 0.5 = chance)

| | 20 um, 96^3 crop (1.9 mm) | 80 um (x4 block-averaged), 96^3 (7.7 mm) |
|---|---|---|
| structure tensor: AUC of coherence | 0.48 | 0.64 |
| overnight pilot: AUC of oriented mass | 0.45 | 0.58 |
| direct-path pilot: AUC of oriented mass | 0.48 | 0.52 |
| generator model: AUC of oriented mass | 0.56 | 0.47 |
| overnight pilot: oriented mass WM / GM | 0.68 / 0.66 | 0.55 / 0.52 |
| generator model: oriented mass WM / GM | 0.10 / 0.13 | 0.05 / 0.07 |
| overnight pilot: points with 2+ populations WM / GM | 0.18 / 0.17 | 0.11 / 0.15 |
| line axis agreement with the structure tensor (WM, \|cos\|) | 0.93-0.97 | 0.90-0.98 |

Per crop at 80 um, AUC of oriented mass (structure tensor / overnight / direct-path / generator):

| specimen | ST | overnight | direct-path | generator |
|---|---|---|---|---|
| AUMC-116-24 | 0.76 | 0.57 | 0.70 | 0.56 |
| AUMC-2025-044 | 0.92 | 0.79 | 0.87 | 0.75 |
| LADAF-2021-17 | 0.54 | 0.67 | 0.57 | 0.61 |
| LADAF-2023-69 | 0.38 | 0.49 | 0.20 | 0.16 |
| S-21-46 | 0.62 | 0.40 | 0.26 | 0.25 |

## Reading

1. **The orientation fields are coherent and follow the image's texture.** Where the image shows streaks, the line
   populations' axes agree with the structure tensor (|cos| 0.90-0.98) and form smooth fields.
2. **SynthSeg's white / grey label is a poor test of visible orientation at this resolution.** In these deep-brain
   crops the "grey" nuclei (striatum, thalamus) carry visible fibre-bundle streaks, while compact white matter looks
   almost featureless at 20-80 um (its axons are below the resolution). At 20 um a 96^3 crop spans ~2 SynthSeg voxels
   (labels are 1 mm), so the first run is uninformative (the structure tensor scores 0.48); at 80 um the labels resolve,
   but two specimens invert even for the structure tensor.
3. **The phantom-trained pilot does not transfer: it is over-confident on real tissue.** It assigns 50-70 % of every
   window to oriented material in both tissues, draws lines in featureless regions and finds two populations at 11-18 %
   of points everywhere -- it never answers "no orientation here". It was trained only on synthetic phantoms with no
   voxel size and no real-like textures. The scene-generator model is more conservative (no population at ~60 % of
   points) but no better at separating the tissues.
4. **No model beats the structure tensor at separating the tissues** (0.47-0.58 against 0.64 at 80 um), and none should
   be expected to until the population model is trained on the scene generator (geometry targets, the fixes from this
   night) rather than the phantom families.

## What a better real test needs

- Labels of *visible* orientation, not tissue class: e.g. annotated tracts with known directions (corpus callosum,
  internal capsule, anterior commissure) at a resolution where they show texture, or co-registered dMRI/PLI directions.
- A "featureless" control: homogeneous regions where the right answer is isotropic / no population.
- The population model trained on the scene generator (with `--geometry-targets`, the trunk pre-norm and the split head)
  before judging real-data transfer.
