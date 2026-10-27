# DiSCo rendered as a greyscale volume: the overnight pilot against the strands' truth (27 Oct, 09:30-10:00 ET)

Asked: "Test in disco greyscale volume".

DiSCo is **test only** (plan D2): nothing here enters training, the generator or model selection. The plan asked for
the protocol to be committed before the first model is scored. This first run wrote and scored the protocol in the
same session, with two changes after seeing results:
- the crop selection, so that it includes 1- and 2-bundle crops (the first draw had none);
- the breakdown by crossing angle, added as analysis.

No model or readout setting was changed.

## Protocol (`scripts/learn/disco_test.py`)

- **Data.** DiSCo1 (Rafael-Patino et al. 2020; doi:10.17632/fgf86jdfg6, CC BY 4.0), downloaded to
  `/hdd/andrew/disco_data/DiSCo1`: strands (.tck), diameters, ROI pairs, and the 25 um bundle-count / volume-fraction
  maps and mask. The meshes and the simulated dMRI were not downloaded.
  - 12 196 strands, 1.4-4.2 um in diameter (median 1.7), in a 1 mm sphere.
  - 25 connections (bundles of 355-733 strands).
  - Coordinates are scaled x40 (1 unit = 25 um).
- **Render.** Each strand is a tube of its own diameter (balls along its polyline) on a 0.5 um grid, averaged down to
  the voxel (true partial volume).
  - The look sits inside the phantom families' ranges:
    - bright fibres: background 0.15, contrast 0.6;
    - dark fibres: 0.8 / -0.5;
    - blur 0.5 voxel, noise 0.02, 8-bit.
  - Three voxel sizes:
    - **1 um**: fibre radius ~0.85 voxel, the cross / curve families' 0.6-1.5;
    - **2 um**: ~0.43, the fine family's 0.35-0.6;
    - **4 um**: ~0.2, fibres unresolved, like white matter in 20 um HiP-CT.
- **Crops.** Six 96^3 crops (96 / 192 / 384 um), seeded, centred >= 125 um inside the sphere where DiSCo's own 25 um
  map counts 4, 4, 4, 3, 2 and 1 bundles. The 4 um crops reach past the sphere (empty corners; those points have no
  truth and are not scored).
- **Truth** at each point and window:
  - each connection's window occupancy and window-mean direction dyad;
  - connections within 20 deg joined into one population (kissing and merging bundles are parallel there).
- **Classes.** As `synth_vs_truth.py`:
  - *single*: one population, any other < 1/5 of it;
  - *core*: two populations, the lighter >= half the heavier;
  - populations counted at mass >= 0.03.
- **Scoring.**
  - Points every 4 voxels, 8 voxels from the faces.
  - Window sigma 2.5 / 4 / 6 voxels.
  - Structure tensor: gradient sigma 1, integration = the window.
  - Model: the overnight pilot (`pilot3x.pt`) with the default readout (continuous masses, soft sheet weight).

## Results (bright fibres, window sigma 4 voxels; dark fibres within 1 deg and 2 % on angles and counts, 0.04 on mass, 0.07 on background)

| | 1 um | 2 um | 4 um |
|---|---|---|---|
| single: median angle, structure tensor | **3** (99 % within 20) | **4** (97 %) | 7 (82 %) |
| single: model, heaviest line population | 4 (97 %) | 5 (95 %) | 7 (79 %) |
| cores: nearer-axis angle, structure tensor | 15 (68 %) | 18 (58 %) | 19 (52 %) |
| cores: model, density peak | **6 (96 %)** | **8 (89 %)** | **11 (76 %)** |
| cores: model, heaviest line population | 9 (88 %) | 13 (75 %) | 16 (64 %) |
| cores: both bundles found (two line populations, each within 20 deg) | 44 % | 30 % | 16 % |
| population count exact | 65 % | 45 % | 37 % |
| true occupancy / model oriented mass (correlation) | 0.24 / 0.36 (0.86) | 0.23 / 0.36 (0.81) | 0.20 / 0.34 (0.87) |

**Both bundles found at cores, by the crossing angle** (sigma 4):

| crossing | 1 um | 2 um | 4 um |
|---|---|---|---|
| 20-40 deg (17-27 % of cores) | 2 % | 1 % | 0 % |
| 40-60 deg | 35 % | 15 % | 4 % |
| 60-90 deg (52-59 % of cores) | **61 %** | **50 %** | 30 % |

At 20-40 deg, the model's and the structure tensor's nearer-axis angles are equal (11-13 deg): one blended direction.
At 60-90 deg the model's heaviest line is 5-10 deg from a true axis at 1 um (window 6 -> 2.5 voxels), against the
structure tensor's 16-19.

**Counts** (sigma 4; rows: truth, columns: model 1 / 2 / 3+):

| | 1 um | 2 um | 4 um |
|---|---|---|---|
| 1 population | 90 / 9 / 1 % | 89 / 11 / 0 % | 78 / 18 / 2 % |
| 2 | 52 / 44 / 4 % | 60 / 37 / 3 % | 64 / 28 / 6 % |
| 3+ | 44 / 39 / 17 % | 56 / 33 / 11 % | 53 / 27 / 19 % |

Over windows 2.5 -> 6 voxels, the structure tensor and the model both improve on single points (1 um: 4 -> 3 and
6 -> 3 deg). At cores the structure tensor gets worse as the window takes in more of both bundles (14 -> 17 deg),
while the model gets better (11 -> 8 deg; both found 37 -> 47 %).

## Reading

1. **On a single bundle the structure tensor is as good or slightly better** (3-7 deg against 4-7). Same as on the
   phantoms and against HiP-CT: one population's orientation is an easy classical problem.
2. **At crossings the model's orientation is clearly better**:
   - nearer-axis 6-9 deg against 15 at 1 um, 88-96 % within 20 deg against 68 %;
   - at 60-90 deg it reports both bundles at 61 % of cores (1 um) and 50 % (2 um) -- in line with the phantoms'
     final set (49 % overall: cross 31, fine 69, mix 76), on geometry it never saw (NFG strands, real-like packing
     and diameters, no lattice).
3. **Narrow crossings are not resolved.** At 20-40 deg the model never separates the bundles (0-2 %); between 40 and
   60 deg it does at 1 um (35 %) but rarely below. The phantoms cross at 30-90 deg, and the readout joins lobes within
   15 deg or across a high saddle. Its angular resolution, not the image, sets the limit: at 1 um the two bundles'
   fibres are individually visible.
4. **Resolution:** separation falls with the fibres' size in voxels (both found 44 -> 30 -> 16 %). At 4 um the fibres
   are unresolved and the structure tensor's single-bundle error also rises (7 deg, 82 % within 20). Some
   orientation survives as partial-volume texture.
5. **Under-counting where bundles meet:**
   - single bundles are counted right (78-90 %);
   - where the truth has two, the model reports one at 52-64 % of points (these include the narrow crossings and
     unbalanced pairs, the lighter below half the heavier);
   - three are rarely all found (11-19 %).
6. **Mass is over-estimated about 1.5x** (oriented mass 0.34-0.36 against occupancy 0.20-0.24; background 0.56-0.63
   against 0.76-0.80). It tracks the occupancy point by point (r 0.81-0.92, higher with larger windows). The same
   over-calling showed on real HiP-CT (0.6-0.7 oriented everywhere). The cause is untested:
   - the 4 um crops include empty space outside the sphere and still over-estimate, so a missing empty region to
     calibrate against is not the whole story;
   - the phantoms' fibre packing (lattice or randomised sections, fvf 0.2-0.35) may read differently from NFG's
     dense packing.
7. **Contrast sign does not matter:** bright and dark fibres agree within 1 deg and 2 %.

## Figures

`results/.../overnight/disco/`:
- `disco1_bright_summary.png`: angles, both found, counts, mass, by the crossing angle;
- `disco1_bright_slices_{1,2,4}um.png`: per crop, the middle slice. Panels: image; truth / model / structure-tensor
  direction; truth / model population count; occupancy against oriented mass; per-point angle error of the model and
  of the structure tensor;
- the same for `dark`;
- numbers in `disco1_{bright,dark}.json`, logs in `.log`.

## Next (not done)

- Crossings below 40 deg: the readout's lobe merging (15 deg, saddle 0.8) and the training crossings' 30 deg floor
  are the candidates. Training crossings down to 20 deg and a softer merge would be tested on the phantoms first,
  never selected on DiSCo.
- Mass calibration in packed tissue: a phantom family packed wall to wall (NFG-like dense sections) would test the
  over-estimation without touching DiSCo.
- DiSCo2 / DiSCo3 and a pre-committed protocol before scoring the next model.
