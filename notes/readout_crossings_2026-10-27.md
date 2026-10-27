# Readout at narrow crossings (27 Oct 2026)

Question: can the readout alone (no retraining) separate crossings the overnight pilot (`pilot3x.pt`) merges? On
DiSCo1 it found both bundles at 0-2 % of 20-40 deg crossing cores, 4-35 % at 40-60 and 30-61 % at 60-90
(`disco_greyscale_2026-10-27.md`).

## Protocol (fixed before the full grid ran)

- **Tuning set** (`scripts/learn/readout_xing.py`, seeds from 3.1M -- not the selection set's 0.9M nor the final set's
  1.9M; DiSCo never): the generator's Monte Carlo crossings ("cross", resolved; "fine", below the grid) with the second
  bundle turned to 20, 25, 30, 35, 40, 50, 60, 75 and 90 deg (two per angle and family, the third population dropped);
  their first bundle alone (four per family); four each of the generator's fans, curves and sheets. Points every 4
  voxels, 6 from the faces; windows sigma 2.5 and 4.
- **Measures**:
  - at cores (two line populations, the lighter >= half the heavier), by the per-point crossing angle (20-30, 30-40,
    40-60, 60-90): **found** -- the two heaviest line populations each within 20 deg of a true axis, one-to-one (as
    `disco_test.py`); **resolved** -- found, each nearer its own axis than the other, and the two at least half the
    true angle apart (at a 20-deg crossing a blended pair is within 20 deg of both axes);
  - where the truth has one population: a second line population (>= 0.03, and >= half the first) on single
    bundles, fans and curves; sheets: the sheet share of the oriented mass.
- **Rule**: the highest resolved share at cores (raw axes), the mean over the four bins and both windows, subject to
  (against the default) a second line population (>= half the first) on single bundles at most 2 points more, on fans
  and on curves at most 3 more each, the sheets' sheet share at most 0.05 less; within 1 point, fewer changed options.
  Then guarded on the selection set's criteria (`readout_lab.py`: the default passes 7/16, criteria C1-C9 9 11 14 13
  13 15 16 15 12) and checked once on the final set (`synth_vs_truth.py`) and DiSCo1 (`disco_test.py`).
- **Options** (`entity_infer.entities_from_query`):
  - `lobe_merge_deg` (new; was fixed at 25): density maxima within it of a heavier one join it;
  - `saddle` (0.8): basins joined across a boundary at least this fraction of the lower peak; `saddle_ring`,
    `saddle_peaks`: the existing guards on that join;
  - `merge_deg` (15): same-type populations within it merged after typing;
  - `sharpen` (new): the density deconvolved by the model's own single-bundle lobe (`sharpen_density`), `sharpen`
    Richardson-Lucy steps, redrawn at `sharpen_to`;
  - the direction grid: 200 (default) or 400.

## Findings before the grid

1. **The model's lobes are broad, at any window.** On the tuning set's single bundles the density above its floor
   falls to half at ~16 deg from its own axis (a Watson fit: kappa 7.6 at sigma 2.5, 7.3 at sigma 4). Its training
   targets are far sharper: the scene generator stores exact fibre tangents, so a straight bundle's target is capped
   only by the direction basis (`DENS_KAPPA_MAX` 40, half at ~8 deg); the evaluation phantoms' conc 0.94 (kappa ~17,
   half at ~12 deg) is theirs alone. The broad lobes are the model's (training), not its targets'. Two such lobes 30 deg apart sum to one hump with a shallow dip,
   25 deg apart to one peak.
2. **Is the second peak in the density?** At cores, along the great circle through the two true axes (smoke test,
   one phantom per angle, sigma 2.5): at 60 deg the model has two maxima at 94 % (valley 0.60 of the lower peak; the
   target 100 %, 0.07); at 30 deg at 48 % (valley 1.00 median; the target 87 %, 0.79).
3. **The direction grid limits the readout.** On noiseless sums of two model-width lobes, the 200-direction grid
   (~10 deg apart; a maximum must beat its 8 neighbours) never shows two maxima 25 deg apart, however sharpened; 30
   deg after 100-300 sharpening steps. 400 directions show 25 deg (300 steps, no redraw), 20 deg only after 1000
   (`tests/test_sharpen_density.py`).
4. **The image refinement drifts on thin interleaved fibres.** On a tuning 60-deg crossing (radii 0.7-1.0 voxel,
   layers ~2 voxels apart) `refine_axes` moved axes seeded 3-4 deg from the truth to 10-17 deg (gradient sigma 0.7:
   8-9; 1.5: 19-25); both found fell 73-83 % -> 31-52 %. On DiSCo at 0.5 um (fibres ~3.4 voxels across) it helped.
   It must not become the default without a resolution gate; here it is reported, not selected on.

## Results (tuning set: 36 crossings, 8 single bundles, 4 each of fans, curves and sheets; ~2,200 points each)

**The density at cores** (along the great circle through the two true axes; sigma 2.5, sigma 4 alike):

| crossing | model: two maxima | valley / lower peak | target: two maxima | valley |
|---|---|---|---|---|
| 20 deg | 0 % | 1.00 | 4 % | 1.00 |
| 25 deg | 1 % | 1.00 | 41 % | 1.00 |
| 30 deg | 6 % | 1.00 | 91 % | 0.76 |
| 35 deg | 23 % | 1.00 | 98 % | 0.56 |
| 40 deg | 73 % | 0.79 | 100 % | 0.37 |
| 60 deg | 79 % | 0.53 | 100 % | 0.04 |
| 90 deg | 100 % | 0.27 | 100 % | 0.00 |

Below 35 deg the second bundle is not in the model's density (one hump); the targets (kappa capped at 40) separate
from ~30 deg. No merge setting can recover what the density does not hold.

**Resolved at cores (%), by crossing angle, raw axes, mean of sigma 2.5 and 4** (20-30 / 30-40 / 40-60 / 60-90):

| readout | resolved | single +2nd | fan | curve | sheet share | rule |
|---|---|---|---|---|---|---|
| default | 0 / 6 / 31 / 73 | -- | -- | -- | -- | -- |
| `lobe_merge_deg=15, saddle_ring=0.6` | 1 / 20 / 63 / 83 | +0.0 | -0.1 | +2.7 | +0.00 | **selected** |
| `lobe_merge_deg=15, saddle_ring=0.8` | 1 / 20 / 63 / 85 | +0.0 | +5.7 | +2.7 | +0.00 | fans |
| `saddle=None` | 1 / 20 / 63 / 86 | +0.0 | +8.8 | +2.7 | -0.87 | sheets |
| sharpened, 200 dirs, 300 steps | 12 / 59 / 78 / 84 | +1.4 | +11.9 | +7.9 | -0.87 | fans, sheets |
| sharpened, 400 dirs, 1000 steps | 24 / 71 / 76 / 83 | +3.7 | +11.6 | +12.5 | -0.88 | all three |

(`lobe_merge_deg` alone changes nothing: the saddle merge joins the same lobes. Sharpening splits fans' arcs and turns
sheets' girdles into lines -- it would need the types read from the unsharpened density.)

**Guard (selection set, `readout_lab.py`): failed.** Pass 7/16 either way; criteria C1-C9 9 11 14 13 13 15 16 15 12 ->
8 11 14 10 14 15 16 16 12 (total 118 -> 116): the selection set's fans split further (a fan's other-direction mass
0.39 -> 0.51, its level 0.94 -> 0.73) and leakage (C4) failed on three more phantoms. By the rule the readout is **not
adopted**; the final set and DiSCo were not run.

**Reading.** The readout can buy ~15-30 points at 30-60 deg, but only by splitting fans, which look the same to the
merge as a 40-deg crossing does once the model draws both 30 deg wide; and it buys nothing below 30 deg, where the
second bundle is not in the density at all. The separation has to come from the model (sharper lobes, trained on the
targets it already has -- kappa up to 40 -- and on narrow crossings), not from the readout.
