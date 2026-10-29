# Why the scene-trained density stayed flat (27-28 Oct)

The question: the scene-trained PopulationNet's direction density stayed flat (peak/mean 1.004-1.03 at oriented
queries, where the targets' is 6-9) while the phantom pilot's formed. Is that the generator, the model's structure, or
the way it is trained? Synthetic data only throughout (no real data).

## 1. Generator: two rendering artefacts (fixed)

* **Curves swept with gaps** (`render_curves`). The renderer swept each curve's section at its stored samples; where a
  fat or wandering curve moved its section's edge more than a voxel between samples, it drew stripes across the fibre
  or single sections as rays. At the training's settings this hit **29 % of all curve volume** (66 % in bundle
  scenes; `curve_steps.py`). In those layers the rendered geometry varied *along* the labelled direction as much as
  across it: the share of the gradient energy along the target axis was 0.45-0.49, above the 1/3 of no relation. Fix:
  curves whose edge would step more than `SWEEP_STEP` = 0.7 voxel are swept on a Catmull-Rom resampling (2-16x).
  After the fix the worst layers fell to 0.10-0.14, and the image of the giant fibre became a clean tube. Test:
  `test_a_coarsely_sampled_fat_curve_is_swept_without_gaps`.
  The fibre wander is also capped so that a fibre bends no tighter than about four of its radii (rms). A fibre as wide
  as its bundle had zig-zagged and folded its own section.
* **Interleaved crossings teleported fibres.** `Scene.interleave` placed each *sample* in a slab by `floor(h / w)`, so
  a fibre wandering over a slab edge jumped 10-24 voxels between samples. The coarse sweep hid the jump as a gap; swept
  properly, the jump cut through the other bundle (15 % of voxels shared). Now each fibre gets one slab, from its mean
  place in the crossing: 1 % shared, with or without the refinement.
* **Grain at any strength.** The isotropic grain underlay (80 % of crops) had an amplitude up to 2x the image's range
  whatever the strength, so even the curriculum's "easy" start was covered by it. It now scales with the strength, up
  to the range at strength 1.

## 2. Model structure: the density cannot start from zero

* The from-scratch phantom model `famN` (8000 steps) **never formed density**. Its density loss was 0.242 at step 0
  and 0.244 at step 8000. Density appeared only in `famO`, which added the direct oriented path to famN's trunk. The
  overnight pilot `pilot3x` descends from famO. So no model of this architecture has formed density from scratch.
* Overfit test (`train.py --overfit 8`, new): 8 fixed full-difficulty scene crops, the pilot's model options, 800
  steps. The density loss on its own training crops stayed at the flat prediction's (0.4122 against 0.4131), while
  mass, votes, tree and descriptors fell. Oriented queries are 57 % of the queries and carry 96 % of the flat loss.
* The direct path's input does carry the direction. Over directions, the bank's line energy along u correlates with
  the target density at a median of 0.30-0.44, depending on level. But its 5 -> 64 -> 32 MLP squeezes that variation
  to 0.036, and the zero-initialised output (weights ~0.016) to 0.014. Getting O(1) log-density shape needs gains
  ~50x larger, and Adam grows each weight by at most the learning rate per step.
* With augmentation (cube transforms, as in training) nothing beat flat: the density path alone, at lr 3e-4 or 1e-2,
  with or without smoothed targets (`DENS_KAPPA_SMOOTH` 8). Without augmentation the harmonic head did fit, but that
  was memorising each crop through the trunk's point code, not reading orientation. Augmentation itself is
  consistent: the energy-target correlation is unchanged under cube transforms (`aug_od.py`).

## 3. The targets at full strength are not in the image

* The pilot, a trained density reader, applied to the same 8 full-difficulty crops: peak/mean 3.1 (it does draw
  lobes), but correlation with the targets only 0.13 (19 % of queries above 0.5), and a loss *worse* than flat (0.76
  against 0.56). Fine-tuned on them with the full loss, it went flat within 25 steps (direction CV 0.75 -> 0.08).
* On 8 clean crops (bundles and packed tissue; no grain, texture, physics or deformation) it is learnable. The
  pilot, as is, correlates 0.41 (34 % above 0.5). The direct path alone shapes the density (CV 0.24) and beats flat by
  5-9 %. The pilot fine-tuned with the full loss keeps its shape and beats flat by 4-21 %.
* So at full strength the acquisition removes the orientation, yet the targets still ask for sharp lobes. The
  orientation visibility the generator assigns is ~1 regardless. Under the log-L1 shape loss, a partly right lobe costs
  more than a flat answer, so flat is the optimum.

## What this means for the overnight run

* Warm start from `pilot3x` (`launch_overnight.sh`, `INIT`; a resume takes precedence), on the fixed generator. The
  curriculum (strength 0.15 -> 1 and difficulty 0.4 -> 1 over the first half) keeps the early data readable.
* Not fixed yet, and the next work: (a) orientation visibility in the targets that follows the acquisition, so a
  window whose orientation the noise, grain or PSF erased asks for a broad or flat density, not a sharp lobe. (b) A
  direct path that does not start at zero, e.g. a linear skip from the log energies to the log density, initialised
  at the classical answer, so scratch training can form density at all.
