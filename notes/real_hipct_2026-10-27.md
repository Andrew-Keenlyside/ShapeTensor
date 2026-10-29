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

## Predominant orientation against the structure tensor (`scripts/learn/real_vs_st.py`)

Same crops; at every grid point the 3-D angle between the structure tensor's line direction (smallest-eigenvalue
eigenvector; gradient sigma 1, integration sigma 2.5 = the model's window) and (a) the density's peak direction, (b) the
heaviest line population's axis. Chance: median 60 deg. Median angle (and share within 20 deg) by the structure tensor's
coherence -- where it is low, the structure tensor has no defined direction:

| 80 um (x4) | all points | coh 0.1-0.2 | coh 0.2-0.3 | coh 0.3-0.5 | coh >= 0.5 |
|---|---|---|---|---|---|
| share of points | | 4 % | 14 % | 54 % | 28 % |
| overnight pilot, density peak | 13 | 36 (23 %) | 23 (43 %) | 12 (72 %) | 10 (78 %) |
| overnight pilot, heaviest line population | 12 | 35 (25 %) | 22 (45 %) | 11 (76 %) | 9 (84 %) |
| direct-path pilot, density peak | 13 | 36 (23 %) | 23 (43 %) | 12 (72 %) | 10 (76 %) |
| generator model, density peak | 18 | 43 (18 %) | 30 (32 %) | 17 (58 %) | 14 (70 %) |
| generator model, heaviest line population | 11 | 24 (43 %) | 17 (60 %) | 11 (85 %) | 10 (87 %) |

| 20 um (x1) | all points | coh 0.1-0.2 | coh 0.2-0.3 | coh 0.3-0.5 | coh >= 0.5 |
|---|---|---|---|---|---|
| share of points | | 13 % | 26 % | 44 % | 16 % |
| overnight pilot, density peak | 16 | 38 (23 %) | 25 (41 %) | 13 (69 %) | 9 (84 %) |
| overnight pilot, heaviest line population | 15 | 34 (25 %) | 22 (46 %) | 12 (73 %) | 9 (87 %) |
| generator model, heaviest line population | 12 | 30 (33 %) | 18 (56 %) | 11 (80 %) | 10 (85 %) |

Per crop (density peak, all points): 10-35 deg at 20 um (S-21-46 the outlier at 35), 10-17 deg at 80 um (generator
model 14-26).

**Reading.** Where the image is clearly oriented (coherence >= 0.3: 60-82 % of points), the models' predominant
orientation is within ~10-13 deg (median) of the structure tensor's, 70-87 % within 20 deg -- they read the same
orientation as the classical estimate. Where the image is weakly oriented the angles approach chance, as they must:
neither direction is defined there. The two phantom-trained pilots agree with the structure tensor equally; the
generator model's density peak is a little further off (14-17 deg), but its line populations, where it reports any,
agree best (10-11 deg). Agreement with the structure tensor is consistency with a classical baseline, not correctness:
where the two differ at high coherence (13-30 % of those points beyond 20 deg) this test cannot say which is right.
Figures: `orientation_vs_st_x1.png`, `orientation_vs_st_x4.png`.

## The same comparison against the truth on hard synthetic phantoms (`scripts/learn/synth_vs_truth.py`)

The final set's crossings, fine crossings and mixes (4 each; curves and fans for reference), held-out seeds, window
sigma 2.5, all points every 3 voxels. Truth: the generator's line populations at each point. Single-population points:
21 000+; crossing cores (two line populations, the lighter >= half the heavier): 3266 (cross 1759, fine 1255, mix 252).
The two phantom pilots were trained on these families (other seeds); the generator model never saw them.

| median angle to the truth (share within 20 deg) | single population | crossing core (nearer axis) | cores: both bundles found |
|---|---|---|---|
| structure tensor | 2 (89 %) | 16 (67 %); fine 19, mix 24 | -- (one direction) |
| overnight pilot, density peak | 4 (94 %) | 4 (100 %) | |
| overnight pilot, heaviest line population | 2 (95 %) | 4 (99 %) | **49 %** (cross 31, fine 69, mix 76) |
| direct-path pilot, heaviest line population | 3 (94 %) | 4 (93 %) | 34 % |
| generator model, heaviest line population | 9 (89 %) | 16 (69 %) | 0 % |

**Reading.** On a single bundle the structure tensor is as good as any model (2 deg): the predominant orientation of
one population is an easy classical problem. At crossing cores it is not: the structure tensor's one direction blends the
two bundles (16 deg median, 24 deg in mixes, a third of core points beyond 20 deg), while the phantom-trained models'
predominant direction stays on one true bundle (4 deg, ~100 % within 20 deg) -- and the overnight pilot also reports the
second bundle at 49 % of core points (31 % of the crossings' cores: the contrast confound, §7 of the overnight notes),
which no single-direction estimate can. The generator model, trained on different data, is near the structure tensor
at cores and never finds both. So the real-data agreement (~10-13 deg) is what both get on single structures; the
models' advantage is in multi-population windows, where the real-data test had no truth to show it.

## Separated mass images, overnight pilot only (`scripts/learn/mass_maps.py`)

`mass_maps_overnight_pilot_synthetic.png`, `..._real_x4.png`, `..._real_x1.png` (results/.../overnight/real/).

**Synthetic (final set; truth beside model, model mass matched to each true population):** backgrounds match the truth
closely; curves, fans and blobs (as isotropic objects) are separated cleanly. Crossings show the contrast confound as
mass images: in cross 0 (contrasts 0.32 / 0.71) the dim bundle keeps 96 of 579 units of mass and the bright one takes
790 of 425; in fine 2 the dimmer bundle (0.34) drops out of the core; cross 2 has notches where one bundle takes the
other's core. Cross 1 and fine 0 are shared (589 / 413 and 289 / 287; 274 / 345 and 650 / 548). The bundle + sheet mix
finds the sheet only at the centre (147 / 189, the rest "other"); the sheet phantom is found only in a central strip,
the rest typed as lines ("other") -- the ring rule's cost on unseen sheets.

**Real HiP-CT (no truth; populations ranked by mass at each point):** at both 20 and 80 um the model splits every
window into one dominant oriented population (mass 0.4-0.8, a smooth direction field) and background; a second
population appears only in scattered patches; isotropic objects are ~0 everywhere (vessels are read as oriented
populations, not isotropic objects). At 80 um the featureless zones get more background and less oriented mass; at 20 um
the oriented mass is high almost everywhere -- the over-confidence of a model trained on phantoms only.

## Readout fix: continuous masses and types (27 Oct, afternoon)

The mass images had hard, threshold-like edges. The network's outputs are smooth (the oriented-material map, the
density's anisotropy); the edges came from the readout's discrete decisions:

1. **Mass split by fitted lobe shapes** ("maps": each lobe's share from its fitted Bingham) jumped wherever the
   population count or a lobe's fit changed. Now `mass="basins"` (default): each oriented entity's share of the
   oriented-material map is its directions' density above the floor (a basin integral) -- a new lobe is born with ~0
   mass, nothing is reassigned when one appears. Weak lobes too broad to type (typed isotropic, they left the oriented
   mass unresolved: a hard rim at bundles' edges) are read as lines (`weak_lines`); `lobe_frac` 0.25 -> 0.1 lets
   lobes form lower on the density. Criteria on the selection set unchanged (pilot3x 7/16; mass level 8 -> 9).
2. **Line / sheet typing** cut a smooth field: the ring cover (the lobe moment's in-plane eigenvalue ratio) varies
   smoothly over real tissue (26-33 % of points at 0.7-0.9), and the type switches at 0.8 -- a map drawing lines by
   their axis and sheets by their normal jumped 90 deg there. The readout now also returns `sheet`, a smoothstep of the
   cover over 0.8 +- 0.1 (`ring_soft`; 0.5 at the type boundary), and the maps split each population into a line part
   (mass x (1 - sheet), its axis) and a sheet part (mass x sheet, its normal). The types (`k`) and so the criteria are
   unchanged.
3. **Display matching**: the synthetic maps used the criteria's hard 30-deg one-to-one matching; now a smooth angular
   weight (tau 15 deg, an "other" option at 30 deg), line parts to true lines and sheet parts to true sheets.

What remains discrete: **a population splitting in two** -- lobes joined across a saddle >= 0.8 of the lower peak, or
within 15 deg -- shows as patch edges in the mixing map and the per-direction track maps (total mass stays continuous).
A soft merge (basins kept apart, joined by a weight smooth in the saddle ratio) would remove it; not done.

Figures (results/.../overnight/real/): `mass_maps_v4_synthetic.png`, `mass_maps_v4_real_x4.png`, `mass_maps_v4_real_x1.png`.
On synthetic: fine 2 now carries both bundles through the core (392 / 337 against truth 370 / 438); cross 0's dim
bundle 166 of 579 (was 96: the contrast confound remains); the sheet phantom an hourglass of sheet mass with soft rims.

## The readout across resolution and smoothing (`scripts/learn/real_sweep.py`)

The five specimens, a crop centred on the same point, block-averaged to 20 / 40 / 80 / 160 um voxels (96^3: a 1.9 /
3.8 / 7.7 / 15.4 mm field), read out on the middle slice at window sigma 1.5 / 2.5 / 4 / 6 voxels (30-960 um). The
structure tensor at the same window (gradient sigma 1 voxel). Means over the specimens:

| voxels | sigma (vox) | oriented mass | background | sheet share | 2+ populations | lines vs ST (deg) | sheet normals vs ST | WM/GM AUC |
|---|---|---|---|---|---|---|---|---|
| 20 um | 1.5 / 2.5 / 4 / 6 | 0.68 -> 0.65 | 0.16-0.17 | 7 -> 10 % | 8 -> 7 % | 27 / 15 / 8 / 5 | 17 / 9 / 5 / 4 | 0.51-0.55 |
| 40 um | | 0.62 -> 0.60 | 0.24 | 3 -> 5 % | 4 -> 3 % | 19 / 11 / 6 / 4 | 16 / 10 / 6 / 4 | 0.51-0.55 |
| 80 um | | 0.55 -> 0.53 | 0.30 | 5 -> 7 % | 11 % | 21 / 13 / 8 / 5 | 16 / 9 / 6 / 4 | 0.46-0.48 |
| 160 um | | 0.42 -> 0.41 | 0.47-0.48 | 19 -> 23 % | 28 -> 21 % | 29 / 18 / 12 / 8 | 20 / 11 / 7 / 5 | 0.38-0.41 |

(Lines vs ST: the line parts' mean direction against the structure tensor's line direction where coherence >= 0.3 and
lines carry most of the mass; sheet normals: against its normal where sheets do.)

**Reading.**
1. **Agreement with the structure tensor rises with the window in voxels**: ~15-30 deg at sigma 1.5 voxels, 4-8 deg at
   6 voxels, at every resolution. 40 um agrees best at each window. Per specimen at sigma 4: 5-10 deg at 20-80 um,
   8-15 deg at 160 um.
   *Corrected the same day (DiSCo single-fibre test, `notes/disco_greyscale_2026-10-27.md`)*: the model's density does
   not depend on the query window at all (correlation 0.999-1.000 between sigma 1 and 8; peaks unmoved). So it is the
   structure tensor that changes with the window, converging on the model's fixed-scale direction as rho grows -- not
   "both estimates noisy at small windows" as first read.
2. **Resolution sets how much is oriented**: oriented mass falls 0.67 -> 0.42 and background rises 0.16 -> 0.48 from 20
   to 160 um, nearly independent of the window. At 20 um the phantom-trained pilot still calls ~2/3 of every window
   oriented (the over-confidence above); coarser crops take in featureless tissue and the model reports more
   background there.
3. **Sheets are rare up to 80 um (3-10 %) and ~20 % at 160 um**, and where reported their normals agree with the
   structure tensor's normal (4-7 deg at sigma >= 4): plausible planar texture (e.g. the laminated band in
   AUMC-2025-044), not noise. Before the line / sheet split, sheet normals were compared to the structure tensor's line
   direction, which inflated the 160 um angles (26 -> 12 deg at sigma 6 on AUMC-116-24).
4. **Two populations: 3-11 % of points up to 80 um, 21-28 % at 160 um** (windows 240-960 um, where fibre systems
   meet) -- falling slightly as the window grows at 160 um, where lobes 15 deg apart or across a high saddle merge.
5. **No white / grey separation** at any scale (AUC 0.38-0.55; inverted at 160 um, where grey nuclei carry visible
   streaks and compact white matter looks featureless -- see above). Isotropic objects: 0 everywhere.

Figures: `real/sweep_soft/sweep_<specimen>.png` (rows resolution, columns window; lines by direction, sheets grey) and
`sweep_metrics.png`; numbers in `sweep.json`.
