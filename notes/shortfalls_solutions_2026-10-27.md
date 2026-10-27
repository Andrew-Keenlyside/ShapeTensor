# Every identified shortfall of the population model, and solutions (27 Oct 2026)

Model: the overnight pilot (`pilot3x.pt`, trained on the phantom families by `phantom_ladder.py`). Evidence from today's
notes (`real_hipct_`, `disco_greyscale_`, `readout_crossings_`, `generator_resolution_`, `overnight_2026-10-27b`).
For each: what goes wrong (evidence), why (likely causes), solutions by layer (D data / generator, T targets, L loss and
training, M model, R readout, E evaluation), and the test that would show it fixed. Status: none of these is fixed in a
trained model; the generator items marked *done* are in the scene generator (`ea22e72`), untrained.

---

## 1. Crops show small bundles on an empty background, not tissue packed with fibres

**Evidence.** Tissue extent (smoothed occupancy > 0.15) of training crops, 24 scenes at the training settings: oriented
tissue median **0 %** of the crop, **79 %** of crops under 10 %, **8 %** over 70 % (all structure: median 17 %). With
`dense_p=1`: oriented median 1 %, 12 % of crops over 70 %. The phantom families: 4-18 % of the crop, **no** crop over
70 %. Real white matter at HiP-CT or DiSCo resolution: fibres fill 50-80 % of the volume everywhere; bundles meet
bundles (borders are changes of direction, not of tissue); crossings are interdigitated regions inside packed tissue.

**Why.** The scene generator is object-centric: bundles of radius 20-3000 um drawn as objects (radius capped at a fifth
of the canvas) on a background; dense scenes are off in training (`dense_p` 0); the phantoms are slabs in a 64^3 box.
A model trained on this learns "a bundle is a dark tube with edges", which cannot transfer to tissue with no edges --
and it probably feeds items 2 (lines everywhere on real tissue) and 8 (mass 1.5x high in packed DiSCo).

**Solutions.**
- D: a **tissue-filling family** (a third to half of scenes): a volume-filling direction field with several domains
  (tracts with sharp borders, smooth bends, fanning zones, crossing zones), fibres seeded to fill the canvas at packing
  0.4-0.85 and grown along the field (NFG / MEDUSA style packing; `fibres.py` has dart throwing and relaxation),
  resolved or unresolved by voxel size (the new strand texture for the latter).
- D: **fascicle partition**: Voronoi / supervoxel domains, each its own direction and packing, touching with no gap --
  abutting tracts ("kissing" without background), the commonest real border.
- D: **crossings inside packed tissue**, interleaved at every grain: single fibres alternating, small fascicles,
  sheet-like layers (DiSCo, real pontine and centrum semiovale crossings), and abutting (one tract over another).
- D: bundles larger than the crop: lift the radius cap so a crop can sit entirely inside one bundle (orientation
  everywhere, no edge anywhere).
- D: **what fills the rest**: vessels running through tracts, rows of interfascicular cells along the fibres,
  extracellular space -- real white-matter texture, not empty background.
- D: grey-matter analogues: dense isotropic neuropil with cell bodies and sparse radial / tangential fibres, so "fibres
  everywhere but weakly oriented" exists, as well as "none".
- D: **stratified crop sampling** with quotas by oriented fill (e.g. 30 % > 70 % filled, 30 % partial, 20 % sparse,
  20 % none), recorded per crop.
- T: in packed tissue there are no edges, so orientation evidence must come from texture: switch on
  `orient_from_texture` for these scenes (the visibility from the populations' oriented texture).
- E: a packed-tissue test family (dense sections, known fill and directions) for mass and counting.

**Test.** The oriented-fill histogram of training crops against the quotas; on a packed test family: mass within
+-10 %, direction, counts; on HiP-CT: the share of featureless regions given lines (item 2).

## 2. Real tissue: never answers "no orientation here"

**Evidence.** HiP-CT (5 specimens): 50-70 % of every window called oriented in white and grey matter; lines drawn in
featureless regions; two populations at 11-18 % of points everywhere; no white / grey separation (AUC 0.38-0.55; the
structure tensor 0.64); isotropic objects 0 everywhere.

**Why.** Trained only on phantoms (no voxel size, no acquisition, no real texture); its crops nearly always hold an
oriented object (item 1); targets credit orientation from outlines; nothing teaches "unknown".

**Solutions.**
- D/L: train on the scene generator (acquisition physics, voxel sizes, the new texture), with item 1's packed tissue
  and its grey-matter analogues.
- T: `orient_from_texture` on: where the image holds no oriented texture the target is isotropic, so the model is
  trained to say so.
- D: more direction-free content in dense form: `ISO_SCENE_P` 0.08 -> 0.15-0.2; dense cell fields; texture underlays
  with real HiP-CT power spectra (measured on featureless real regions).
- M: an explicit **evidence output** per query (trained on the generator's `recoverability` / visibility), used by
  the readout to gate populations: "no orientation" becomes an answer, not an absence.
- L: label-free losses on real crops (no targets needed): rotation equivariance (a rotated crop's output is the
  rotated output), window-size consistency (item 5), cross-resolution consistency (a downsampled crop's density is the
  blurred density of the original).
- D: intensity domain randomisation matched to real scans: histogram matching, real noise spectra, ring artefacts,
  phase-contrast fringes (some exist in `physics.py`; calibrate them against the real scans).
- E: replace SynthSeg white / grey with orientation-evidence labels (multi-scale structure-tensor coherence, expert
  "oriented / not" marks on a few hundred real windows, or co-registered dMRI / PLI where available); report "lines in
  featureless regions" as its own metric.

**Test.** On HiP-CT: share of featureless windows given a population (target ~0); agreement with the structure tensor
where it is coherent; separation of tissues by the evidence output.

## 3. Crossings in general: the hard families pass nothing

**Evidence.** Selection set 7/16 pass all criteria; crossing, fine-crossing and mixture phantoms pass none (mass level
C1 and flatness C2 fail on every crossing and mixture); the overnight notes' dominant failure: one rotated crossing is
learned quickly, crossings in all orientations are not; 13000 steps improved 9 -> 11/32 and the selection set was flat
from step 6000.

**Why.** A non-equivariant CNN must learn every rotation of a crossing's texture; few crossing cores per batch (batch
4, accumulation 2; query points rarely at cores); a narrow full-resolution trunk (32 channels at stride 1); short
training.

**Solutions.**
- M: cheap exact augmentation: the 48 cube symmetries (rotations and flips) on the grid, plus test-time averaging
  (`query_tta` exists: measure its gain now -- a one-hour test).
- M: rotation-aware features at strides 1-2: widen the steerable direction bank (`odirect` 256 -> more directions and
  scales), or equivariant layers (spherical-harmonic / e3nn-style steerable convolutions) in the first two levels.
- M: fixed oriented filter banks as extra input channels (second-derivative / Gabor responses at 20-40 directions,
  2-3 scales): they separate crossing directions by construction; the network learns on top.
- M: a wider full-resolution trunk (32 -> 64 channels at stride 1), memory permitting.
- L: query points sampled at crossing cores in proportion (importance by the number of populations in the window);
  crossings in more scenes (`cross_p` 0.3 -> 0.5 with item 1's packed crossings).
- L: dense auxiliary supervision at stride 1: each population's local direction per voxel (or its oriented gradient
  energy) -- an easy, local target that builds crossing features.
- L: longer training with a larger effective batch, weight averaging (EMA), and the learning-rate schedule re-tuned --
  only after the above, since the plateau points to capacity / data, not duration.

**Test.** The hard families' criteria; the crossing-angle tuning set (`readout_xing.py`); a rotation test: one phantom
in 24 orientations, the spread of the outputs.

## 4. Angular resolution: broad lobes, nothing below ~35 deg

**Evidence.** Single bundle drawn ~30 deg wide at half height (kappa ~7.5) against targets of kappa ~17; at 30-deg
crossing cores the density has two maxima at 6 % (target 91 %), at 35 deg 23 % (98 %); the readout cannot recover
what the density lacks (`readout_xing.py`).

**Why.** The log-L1 per-direction loss's optimum under uncertainty is a median (flattened lobes); 32 directions per
query (16 near lobes) barely constrain a sharp peak; phantom targets fixed at kappa ~17; no phantom crossing below 30
deg; short training.

**Solutions.**
- M/T: **sharpness as a query input** (kappa_q, log-uniform 2-50) with targets = the window's true tangent distribution
  convolved with a Watson of kappa_q (exact by spherical-harmonic degree weights; `query.voxel_density` /
  `sample_density` build the window means); any sharpness on request, the model's uncertainty widening it honestly.
- T: one convention for phantoms and scenes: exact tangents (no conc 0.94 floor), or kappa_q everywhere.
- L: 64-128 directions per query, importance-sampled near each lobe at its width and along the arc between
  neighbouring lobes (the valley decides separation); a loss whose optimum is the posterior mean (KL / cross-entropy of
  the normalised density, or L2 on it), tested against log-L1 on the overfit ladder first.
- D: crossings 10-40 deg oversampled in phantoms and scenes (the scene generator draws 5-90 uniformly).
- M: the direction basis: order 16 draws kappa <= 40 (separates ~13-15 deg); lower needs order 24 or the density from
  the per-direction MLP without the SH head (`dens_sh` off) -- test which is sharper.
- R: grid and merge angle scaled to kappa_q (200 directions to kappa 17, 400 to 40, 800 above); `sharpen` (built,
  tested: deconvolution recovers 25-35 deg but splits fans -- usable once fans are told apart, item 7).

**Test.** Overfit one 20-deg crossing: one lobe at kappa_q 8, two at 40; the profile at cores (two maxima by angle,
`readout_xing.py`); the tuning set's resolved share by angle.

## 5. Blind to the window size

**Evidence.** The density is the same from sigma 1 to 8 (correlation 0.999-1.000); peaks never move with the window;
HiP-CT agreement with the structure tensor rose with the window only because the tensor converged on the model's
fixed answer.

**Why.** In homogeneous phantoms the target barely changes with sigma; training windows sigma 1-6; nothing forces the
model to use sigma.

**Solutions.**
- D: content whose window mean changes with scale: item 1's domains and borders, meandering bundles (*done*),
  finite crossings, fans, curvature.
- L: sigma 1-12 (*done* as the default); sampled per query.
- L: **window-size consistency** (label-free, real data too): the density at sigma_big equals the Gaussian-weighted
  mixture of densities at sigma_small over positions (exact for the truth: densities add over voxels).
- M: feed sigma to every decoder layer (FiLM), not only the encoder input, so it cannot be ignored.
- E: the sigma-response test: correlation of the model's density between sigma 1 and 8 against the target's.

## 6. Counting: crossings under-counted, unequal and triple crossings, the brighter bundle wins

**Evidence.** Where the truth has two populations the model reports one at 52-64 % of points; three all found at
11-19 %; the brighter bundle takes crossing cores (overnight notes, section 7).

**Why.** Broad lobes merge (item 4); weak lobes fall below the readout's thresholds; contrasts drawn independently per
population, so the brighter bundle dominates the image; the per-direction loss favours the stronger signal.

**Solutions.**
- D: crossing contrasts drawn as related (same tissue, same contrast) with some probability, and contrast ratios
  sampled explicitly; mass shares 0.1-0.5 sampled explicitly; triple crossings in 40 % of crossing scenes.
- L: weight the minority population (`lobe_mass` exists: raise it); a count head (populations above a mass threshold)
  with its own loss.
- R: a prominence-based lobe merge instead of single-linkage saddles; keep a lobe by its share of the window's
  oriented mass, not of the density mean; `cmax` checked on triple crossings.
- E: scores stratified by contrast ratio and mass ratio; a triple-crossing test family.

## 7. Fans look like crossings

**Evidence.** On the selection set 39 % of a fan's mass sits in other directions (C4), fan direction (C6) fails; every
readout that split more crossings also split fans (`readout_xing.py`: +6 to +19 points).

**Why.** With 30-deg-wide lobes an arc (a fan) and two peaks (a crossing) give the same density; the readout decides
from the density alone.

**Solutions.**
- M: sharper lobes (item 4) make arc and peaks different shapes.
- M/R: decide populations by the **embeddings and votes**, which are trained on population identity, not by density
  maxima: the votes' concentration says "one broad population" for a fan and "two peaked" for a crossing
  (`entities="votes"` and the vote kappas exist) -- weight the vote losses up and read them.
- R: an arc test: a merged group whose directions form an arc and whose votes are broad stays one population.
- D: fans next to crossings in the same scene; fanning crossings (the scene generator's mode exists); more fans.
- E: a fans-beside-crossings test set; C4 and C6 on fans.

## 8. Mass: too high in packed tissue, uneven through crossings

**Evidence.** DiSCo 1-4 um: oriented mass 0.34-0.36 against fill 0.20-0.24 (~1.5x; calibrated at 0.5 um); HiP-CT 0.6-0.7
oriented everywhere; phantoms: fans, curves, mixtures 5-20 % high; crossings and mixtures fail mass level (C1) and
flatness (C2).

**Why.** Never trained on packed tissue (item 1): presence of structure is read as high fill; mass reads image
contrast, whose calibration varies with the look; at crossings the split between populations follows the broad lobes
and the brighter bundle.

**Solutions.**
- D: packed tissue with known fill 0.3-0.9 and contrast decoupled from fill (item 1).
- L: **mass additivity** across scales (label-free): the mass at sigma_big is the Gaussian average of the masses at
  sigma_small.
- L: flatness along a bundle: neighbouring windows along a population's axis should agree on its mass -- a loss on the
  existing overlapping-window pairs (`pairs`, `pair_every`), weighted up through crossings.
- M: keep "how much" apart from "which way": the split head's oriented / isotropic maps trained on fill alone; the
  per-population share from the lobe-mass loss (exists).
- R: a post-hoc calibration curve of oriented mass against true fill from a held-out generator set (fixes the average
  only; a stopgap).
- E: the packed test family; DiSCo2 / DiSCo3.

## 9. Axis precision

**Evidence.** Inside single fibres the structure tensor is 1.1-1.9 deg; the model 2.8-14 deg; the image refinement equals
the tensor where fibres are resolved (>= ~3 voxels) but drifts on thin interleaved fibres (3-4 -> 10-17 deg).

**Solutions.**
- R: **continuous peak refinement**: the density can be queried at any direction, so a few gradient-ascent steps per lobe
  from its grid maximum remove the grid's 1-2 deg -- cheap.
- R: the image refinement behind a resolution gate: only where the fibre scale (the descriptor's size output, or the
  window's gradient coherence) is >= ~2 voxels; elsewhere keep the model's axis, or blend by agreement.
- M/L: axes from the votes (`axes="votes"`, a direct frame per direction) rather than the density's moment; frame-loss
  weight up; sharper lobes (item 4) also place peaks better.
- R: averaging over test-time rotations.
- E: `disco_fibres.py` single-fibre errors; the tuning set's axis errors by angle.

## 10. Fibres below the voxel

**Evidence.** Unresolved fibres recovered 0/6 (overnight runs); the generator drew them featureless until today.

**Solutions.**
- D: the strand texture (*done*); supersampled rendering of unresolved strands (as the phantoms' `fine` family, ss 2-4)
  so per-voxel targets hold the true sub-voxel tangent distribution.
- T: sub-voxel windows (sigma below 1) with targets from the fine grid; the fibre-size descriptor in um as a target.
- E: DiSCo at 2-4 um (fibres below the voxel) crossings and mass.

## 11. Non-fibrous objects

**Evidence.** Blob level and edges (C1, C5) off on the selection set; in the scene generator cells and foams own no
directions and no identity; isotropic objects 0 everywhere on HiP-CT.

**Solutions.**
- T: identity targets for cells, nuclei and foam cells (tree nodes); their isotropic entities kept apart from the
  background.
- D: dense cell fields, cell rows along fibres, nuclei with chromatin texture, varied contrast; isotropic-only scenes up.
- L: boundary-aware mass (edges): a loss on the window mean at object borders.
- E: cell-rich test phantoms; on HiP-CT, cell-dense regions (cerebellar granular layer, choroid plexus) as checks.

## 12. Sheets

**Evidence.** The ring (sheet) threshold was set on 2 sheets; HiP-CT sheets 3-10 % of oriented mass, normals 4-7 deg from
the structure tensor's; sharpening turns sheets into lines; a crossing of many directions in one plane and a sheet both
make a girdle.

**Solutions.**
- D: more sheets: folded, fenestrated, stacked lamellae, membranes crossed by fibres (families exist; weight them up).
- T: make the convention explicit: a sheet is one population with a normal; a planar crossing of k lines is k
  populations -- both in the training data side by side.
- R: type from the votes' second concentration where the density is sharpened; check the ring threshold on >= 8 sheets.
- E: a sheet test family (stacks, folds, sheets crossed by fibres).

## 13. Map artefacts and spatial consistency

**Evidence.** Hard edges in mass maps where the population count changes; each point read out independently.

**Solutions.**
- R: the soft merge (weights from the saddle depth instead of a hard join) -- offered, not built.
- R: spatial linking of entities between neighbouring points (`link()` exists) with a smoothness term (a small
  graph-cut / CRF over the matches) before the maps are drawn.
- L: cross-window consistency (overlapping pairs exist): raise its weight and include crossings.

## 14. Object identity across windows; tractography

**Evidence.** Untested in this work.

**Solutions.** The DiSCo tractography plan (five direction fields on one tracker, connectivity against the truth,
crossing fidelity by angle); `link()` / `follow()` tested on phantoms with known objects; object-level metrics (a
bundle's continuity through a crossing).

## 15. Uncertainty

**Evidence.** The vote uncertainty output exists (`vote_unc`) but has never been checked against actual errors.

**Solutions.** Calibration curves (predicted against observed angle error) on the tuning set and DiSCo; an evidence
output per window (item 2); uncertainty-weighted readout gating and tracking stops.

## 16. Readout limits and cost

**Evidence.** 200 directions cannot show two maxima 25 deg apart; denser grids fragment the clustering; the merge
thresholds suit one lobe width; every point queries 200 directions.

**Solutions.** Adaptive direction sampling (coarse grid, then ascent near maxima, item 9); thresholds scaled to the
requested sharpness (item 4); peak finding on the SH coefficients where the density is SH; the existing `t_rel` gate
(evaluate only where the total ODF has density); measure throughput on a whole HiP-CT volume before choosing.

## 17. Training process

**Evidence.** Selection set flat after step 6000; 32 directions per query; 0.8 s/step; CPU phantom workers too slow
(fine crossings 7-30 s each); GPU 1 unusable by torch; batch 4 x accumulation 2.

**Solutions.** GPU data generation in its own process feeding a queue (scenes are already shared across 8 crops);
more directions per query (decoder-only cost: measure); EMA weights; a larger effective batch; mixed precision checked;
find why torch cannot use GPU 1 (driver / CUDA visibility) to double throughput.

## 18. Evaluation

**Evidence.** 2 phantoms per family in the selection set, and the final set has been looked at; real truth only DiSCo1;
HiP-CT without direction truth; SynthSeg white / grey a poor test; phantoms (kappa ~17) and scenes (up to 40) define
targets differently; histology, EM, MRI untested on real data.

**Solutions.**
- A fresh, larger test set (>= 8 per family) generated once and never looked at during development.
- Scores stratified by crossing angle, contrast ratio, mass ratio, fill, voxel size and blur.
- One target convention across phantoms and scenes (item 4).
- DiSCo2 / DiSCo3 under a protocol written before scoring.
- **Real truth from HiP-CT's own zooms**: the same region scanned at 2-6 um (local zoom) and 20-25 um (overview); the
  zoom's structure tensor at fibre resolution is the reference for the overview -- real tissue, real acquisition, truth
  for exactly the resolution loss the model must handle.
- Real data in other modalities with a reference: PLI or histology with fibre-resolved structure tensors, dMRI where
  co-registered.
