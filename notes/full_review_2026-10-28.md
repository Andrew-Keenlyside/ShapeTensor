# Full review of the generator, targets, model and trainer (28 Oct, evening)

Four independent adversarial reviews (geometry and rendering; acquisition -> targets -> loss; model structure; the
training loop), each asked for measurements, after the three stacked causes of the flat density were found
(`density_learning_2026-10-28.md`). What they found, what was fixed, and what remains. Synthetic data only.

## Training loop (`train.py`, `entity_train.py`) -- all fixed

1. **The warm start's gradient was 93 % the lattice ODF loss.** With the pilot's weights loaded, on full-difficulty
   crops: total gradient norm 158 against the clip of 5; `overall` (the lattice ODF's relative squared error) 147 of
   it, the density path 1.1. After clipping the density path moved as if its learning rate were ~1e-5. The pilot's
   SH head is far from the scene targets (loss 5.4 against 0.3-0.9 for a scratch model after 50 steps: another target
   convention). Fix: `--init-reset total` leaves the head **at zero** on a warm start (its loss then starts at its
   floor, 0.999; a randomly initialised head gave 10-19), and for any `--init` run the `overall` weight ramps in over
   the warmup. `grad_norm` is in `log.jsonl`: watch for values far above 5.
2. **`REGION_DENSITY` was wired in `train_lobes` only.** `train_entities` left `ET.REGION_DENSITY` at its default
   (False), so the region head, whose target is a density (crossings count both, values to 2.6) and whose value is a
   softplus, was trained with BCE on logits: targets above 1 pushed the logits to infinity, and the mass path (the
   window mean of the softplus) fought the BCE on the same logit. The pilot was trained with this too. Fixed: set in
   `train_entities`; `--no-region-density` now reaches the targets as well.
3. **Window pairs were drawn at strength 1 whatever the curriculum** (`EntityPairBuffer` passed no strength). Every
   4th step a third of the entity gradient came from full-noise crops. Fixed: the buffer follows `cfg["strength"]`.
4. Val creation on the GPU before the first step lost two runs to out-of-memory on the shared card (nothing saved yet).
   Fixed: retried four times a minute apart. A resume now logs that `--init` was ignored.
5. Measured, no change: the buffer's items lag the curriculum by ~160-190 steps (mean age 128 plus the worker queue),
   negligible over 16000 steps; the val set is drawn at full strength and difficulty (a fixed benchmark, not an early
   learning signal); `--init` loads all 114 tensors; the optimiser starts fresh (3e-4 is ~20x the pilot's final rate:
   expect an early rise); worker seeds are disjoint from val and pairs; the cube augmentation moves image and every
   target alike; `--overfit` runs the contrast routes at full strength unless `--routes 0`.

## Geometry and rendering (`scene.py`) -- fixed unless marked

1. **`smooth1d` blew up at the ends** (unit-norm kernel, replicated ends): the first and last samples had 3-12x the
   variance (rms 3.1 at sigma 10, 5.4 at 33, 8.9 at 88). Every bundle is grown both ways from its centre
   (`centred_paths`), so the blow-up sat at the bundle centre -- the crossing point, where half the crops are centred:
   curvature there 6-14x the drawn value (bend radius 1.3-2.6 voxels); 19 of 40 bundles had inner fibres reversing,
   14 of 18 resolved bundles had fibres bent tighter than their radius (one seed: 94 of 148 fibres, 99-degree turns);
   trees kinked at every branch start; the kinks also doubled the sweep refinement (render 24 s -> 10 s without them).
   The crossing angle was wrong within a correlation length of the crossing point. Fix: `smooth_noise1d` draws
   `S + 2r` samples and convolves without padding (unit variance everywhere; the turn at the centre is now 1.5x the
   drawn value, the two half-walks meeting, against 6-14x); `smooth1d` stays for deterministic ratios.
2. **Interleaving assumed straight bundles**: the distance to the other bundle's *axis* placed a meandering bundle's
   crossing elsewhere (12-13 % of the smaller bundle's voxels shared, against 0-3.5 % for straight ones). Fix: the
   distance to the other bundle's actual centreline in the plane of the interleaving.
3. **Cell rows sat on the fibre paths** (tract occupancy 0.6-0.7 at the cell voxels, labelled "tract direction plus a
   cell"). Fix: offset beside the fibre by 1-2 cell radii. *Not fixed*: vessels through packed tissue are not carved
   out of the tract fields (tract occupancy 0.45-0.67 in the lumen; 0.01-2 % of the voxels of a tissue scene).
4. **Fanning had a slope discontinuity at the bundle centre** (edge fibres turned by up to 16 degrees exactly where
   crops centre): a smooth knee now.
5. The deformation's inverse residual counted as valid below 0.5 voxel: now 0.1 (20 random deformations: max 0.047).
6. *Not fixed, flagged*: in packed tissue the smooth part of the fill carries the full line tensor although its clean
   geometry shows no orientation (11 of 46 tract layers: axis energy > 0.2 with orientation visibility 0.7-1.0 at
   strength 0). Whether the acquisition's oriented texture makes it visible in the *image* is the targets review's
   question (visibility-aware targets cover it either way).
7. Verified: tensors against curve tangents 0.2-0.4 degrees (0.6-voxel fibres 1.3); under deformation the swept and the
   pulled-back directions agree within 0.1-1.6 degrees; per-family label-geometry consistency (103 layers, 28 scenes)
   bundle 3.4, compose 2.6, tree 4.3, network 7 degrees; the 28 Oct sweep refinement leaves no holes (core cv 0.005)
   and no memory growth; interleave shares 0-3.5 % of voxels in 7 of 9 crossings; draws are bit-identical per seed.

## Acquisition -> targets -> loss -- the blocker, fixed in the targets

1. **The generator's orientation visibility does not track whether the orientation is recoverable.** On 20 crops at
   full strength and difficulty (2570 peaked oriented queries): a structure tensor in the query's own window puts the
   line axis within 20 degrees of the target on **15 %** of them (chance 6 %; median angle 52; the classical density's
   correlation with the target 0.13 -- the pilot's 0.13 exactly). On the clean geometry it is 42 %. The generator's
   `orientation_visibility` at those queries has median 0.71 and the recoverable share is flat across it (14 % below
   0.25, 19 % above 0.9); the presence visibility `e_vis` is 1.0. Causes: the evidence is a t-statistic of the layer's
   fibre-scale texture against the image, maxed over windows of sigma 5-20 (a sigma-20 window pools 64x the voxels of
   a sigma-2 query's), and it says "the texture is there", not "the axis is recoverable at this scale" (a packed-tissue
   crop at strength 0: visibility 0.97, kappa 17, 10 % recoverable). Visibility only softens kappa (0.5 -> peak/mean 5).
   The ceiling by condition: grain on 15 % / off 18 % (clean 35 / 65); PSF rms 0.8-1.5 24 %, > 1.5 4 %; modality mri
   43, ct / em 24, hipct / histology 6, cryoet 4; noise is not the driver.
2. **The loss basin is geometric, and no alternative widens it.** With the real shape loss and sampled directions, a
   kappa-17 lobe over a 40 % isotropic background: a correct lobe 10 degrees off wins against flat on 100 % of queries,
   20 off 85 %, 30 off 63 %, 45 off 22 %; a broadened lobe (kappa / 2 ... / 8) or a half-flat mixture always wins. KL
   and linear L1 have the same or narrower basins. So flat is the optimum whenever the model's lobe is wrong on more
   than ~40 % of oriented queries -- which, by (1), it had to be. Oriented queries (49-58 %) carry 96 % of the flat
   loss; nothing weighted them by visibility.
3. **The pilot's lobes on these crops are wrong lobes**, not blurred right ones: 2-4x the flat loss under every target
   rule, even on the queries the rule calls visible. A classical reader with the target's kappa along its own axis
   beats flat on visible queries (0.19 against 0.31) and loses on the invisible ones (where the new targets say flat).
4. Fix (`entities.orientation_visibility`, `train.entity_queries` -> `e_ovis`, mixed in `entity_loss`): the structure
   tensor at the query's own window on the *acquired* image (gradients at sigma 1, `window_mean` of their products at
   `e_sigma`), its weakest direction against the heaviest line entity's axis -- at a crossing core (second line entity
   at half the mass or more) against the two axes' plane normal -- gives v = 1 within 10 degrees, 0 beyond 40; the
   density target becomes `v rho + (1 - v) mass` (flat at the same mass). Measured on training crops (CPU, 0.4-0.5 s
   per crop): line queries with v > 0.5: 43 % at strength 0, 32 % at 0.5, 41 % at 1 (v = 0: 37 / 52 / 29 %).
   `--no-ovis-targets` restores the old targets. The flat loss on oriented queries then drops (0.54 -> ~0.25), and the
   visible queries become the whole gradient instead of a fifth of it.
5. Verified: `e_R[..., 2]`, `d["tensors"]`, `e_T6` and the image's structure tensor share one frame (0.0-0.5 degrees;
   on a clean bundle crop 71 % of queries within 20); cube transforms leave recoverability unchanged; contrast routes
   degrade it (24 -> 15-28 %, relief the worst) but never invert it; the 200-direction peak/mean is within 7 %.
   Caveat: `dens_check.py` / `dens_floor.py` (the earlier note's numbers) never set `ORIENT_FROM_TEXTURE`, so their
   self-drawn crops used the pre-27-Oct visibility; the overfit items came through `train.py` and are fine.

## Model structure (`model.py`) -- two beliefs corrected, one change

1. **No structural bug makes orientation unreadable**: the bank's frame (identity best of all 24 signed axis
   permutations), `od_sample`'s coordinates against `encode`'s offsets, the query's sampling, the gradient flow to
   `od_net` / `od_dens` (norms 4e-3 / 2e-2), autocast (od in fp32), block-norm / LCN / softplus (none touch the od
   path; the input's directional spread 0.30-0.56 reaches `od_net` unchanged) all check out.
2. **The model does form density from scratch -- on readable crops.** A scratch trunk with augmentation, density loss
   only, 8 easy items, 300 steps: 11 % better than flat, peak/mean 2.5 (target 3.9), correlation 0.48. The "never from
   scratch" result was only ever measured on full-difficulty crops, where nothing forms (0.2 %) -- and where the od
   path's input, fitted to convergence, cannot beat flat either (0.8 %): the targets, not the model.
3. **The classical linear answer is nearly flat under this loss** (a product of the energies with fitted gains beats
   flat by 1.4 %; unit gains are worse than flat), so a skip initialised at the classical answer starts non-flat but
   neither speeds learning nor improves the endpoint (300 steps: 10.7 % against 11.3 % without). Not adopted.
4. **The od code saw only the energy along u, never the point's angular context** (and its "mean" feature is
   identically ~0, the energies being normalised per level already). Fitted to convergence on readable crops, the path
   beat flat by 5 %; with each level's energy relative to the strongest direction at the point, 9 % (peak/mean 2.1 ->
   2.7). Filter length / width and a sharper bank kernel changed nothing. Added: `--od-norm` (the extra inputs; a warm
   start grows `od_net`'s input channels with zeros, so it starts as the pilot did).
5. Zero-initialised `od_dens` grows at ~lr per step (as the note said) but is not what blocks density; the receptive
   field of 182 voxels on 64-voxel crops is not either.

## After the fixes: the warm start still went flat (stress run, 400 steps) -- and what was changed for it

* **Stress run** (`warm_stress`: pilot3x warm start, the fixed generator, visibility-aware targets, full difficulty by
  step 200): val density 0.474 -> 0.125, but on 12 fresh full-strength crops the model's density is flat (peak/mean
  1.10 at visible line queries, 1.02 at invisible; correlation with the raw target 0.17) and its loss 0.051 is *worse*
  than flat's 0.044. The pilot it started from reads the visible queries (peak/mean 4.1, correlation 0.61, 64 % above
  0.5) but draws lobes everywhere (1.6 where the target is flat), so its loss is 0.356.
* **Why.** (1) Under the new targets flat is nearly the global optimum: visible line queries are 7 % of the queries
  with mass (310 of 4608), invisible 16 %, the rest non-line or partial. The cheapest descent from 0.356 is "flatten
  everything"; re-forming lobes only where the image shows them means conditioning on the structure-tensor contrast,
  slower than 400 steps and uphill from the flat basin. (2) The density gradient is swamped: at step 400 the total
  gradient norm is 12.9 (clipped to 5) -- split_ori 9.2, split_iso 5.4, obj 3.9, mvote 2.2, vote 1.4, **density 0.26**.
  (3) A fresh Adam at 3e-4 moves every weight by ~lr for the first steps, scrambling the trained density path.
* **Structural change: a visibility gate** (`--dens-gate`): a scalar per query from the point code, trained with BCE on
  `e_ovis`; the density read out is `g x lobe + (1 - g) x flat`; the lobe is trained only where the orientation shows
  (per-query weight `v + 0.02`), so the lobe path never receives a "go flat" gradient -- flatness is the gate's job and
  its target is explicit. Also `--dens-vis-w` (a per-query weight `1 + w v` on the single-output density) for the
  tuning variant, `LR`, `WORKERS`, `THREADS` in the launcher, and the val set drawn on the CPU when the GPU is out of
  memory (two pilots were lost to that on the shared card).
* **Pilots** (2000 steps, full difficulty from step 1000): C = lr 1e-4, visibility weight 4, density weight 4 (tuning
  of the single output); D = lr 1e-4 with the gate. Evaluated by `dens_eval.py` at steps 1000 and 2000.

## Training speed (28 Oct, local A6000, 64 cores shared with another job)

* 1.7-3.2 s per step at batch 4 x accum 2; the CPU workers deliver 0.2-0.7 new pairs per step against the target of
  1 -- the data generation is the throughput limit, and on the cluster (4 ranks) the demand is 4x.
* Profile of one worker draw (8 crops of one scene, CPU, under load): 89 s per crop. The scene render 300 s per scene
  (`render_curves` -> `splat` 261 s: index_add 42, `torch.unique` 52, cat 21, Python 153), the **visibility
  statistics 289 s per 8 crops** (36 s per crop: 13 049 Gaussian filterings -- presence and orientation t-statistics
  at three windows for every layer), `entity_queries` 7.5 s per crop. Change: with the image-measured visibility the
  generator's orientation t-statistics are skipped (`generator.ORIENT_VIS` off: half the visibility cost). The splat's
  per-call `unique` and the three-window presence ladder are the next targets; on the cluster node (192 idle cores)
  the plan is 24 workers per rank.

## Night of 28-29 Oct: the local-pilot loop (what trained, what did not)

Evaluation throughout: `dens_eval.py` on 10 fresh full-strength crops -- the density loss against the visibility-aware
targets (flat 0.044-0.052), the model's peak/mean at *visible* line queries (`e_ovis` > 0.5; target 4.6) and at
invisible ones (< 0.2; target flat), the correlation with the raw lobe over directions.

| pilot | change | result |
|---|---|---|
| warm stress, 400 steps | targets + generator fixes only | flat everywhere (1.10 / 1.02); loss 0.051, worse than flat |
| C, 1000 steps (local) | lr 1e-4, visibility weight 4, density weight 4 | flat (1.01 / 1.01); loss 0.050 |
| D, cluster, 4 H100 | **the visibility gate**, lr 1e-4 | step 500: lobes where visible (2.11, corr 0.66, 60 % > 0.5), flat where not (1.05) -- the first scene-trained density; step 2000: gone again (1.05, corr 0.24). The gate itself hardly learned (BCE 0.693 -> 0.685) |
| E, cluster | gate fed the classical cue (energy contrast per level, positive init), gate lr x10 | running |
| F, cluster | E + the trunk frozen 1000 steps then at 0.3x, the lobe on the visibility-mixed target, density weight 4 | running |

Reading of D: the lobe branch, supervised only where the orientation shows, still went flat by step 2000 -- its lobes
there are not accurate enough for the sharp raw targets (the basin: beyond ~25 degrees flat wins), and the trunk,
driven by the other heads' 50x larger gradients, drifted from the features the pilot's density reading used. Hence F:
partial credit (the mixed target) and a frozen trunk while the heads adapt.

Cluster: `~/ShapeTensors` (rsync of the working tree), env `~/venvs/shapetensors` (conda python 3.12, torch 2.5.1
cu124, numpy 2.2.6, scipy 1.14.1 -- the newest wheels need a newer glibc than CentOS 7's), `qsub.shapetensor_v3.sh`
(`-pe gpu 4`, h100; STEPS / OUT / VAL_EVERY / CKPT_MIN / WORKERS / NGPU / LR / EXTRA_ARGS from `-v`). Benchmark on 4
H100 (`--benchmark 30`): 1.12 s per step (data 0.26, forward + backward 0.67, optimiser 0.08), 7.1 samples/s/GPU, 15.6
GB; the generators (24 workers per rank) keep up at low difficulty (1.0 new pairs per step) but not at full (0.2):
40 per rank from pilot E on.

### Status at 08:15 ET, 29 Oct (for the 9 am review)

* The session was paused by a usage limit from ~02:15 to ~07:15 ET; no 12-hour cluster run was launched (the gate
  pilots had not yet held their lobes, which was the condition).
* Pilot F (local, gate v2 + frozen trunk + mixed lobe target + density weight 4) at step 1000: the gate learns at
  last (BCE 0.915 -> 0.692 and falling), the lobe loss at visible queries halves (0.52 -> 0.26), but the lobes
  themselves are nearly flat again (peak/mean 1.06 at visible queries, correlation 0.43, 46 % above 0.5; loss 0.0525
  against flat 0.0476). Running to 2000.
* Reading: the lobe branch starts from the pilot's lobes, which are *wrong* on these crops (its loss at visible
  queries is above flat's), so the first thing it learns is to drop them; re-forming correct lobes from the bank has
  to come after, and 1000-2000 steps at lr 1e-4 have not shown it. Next: per-subset losses (visible / partial /
  invisible) to see whether any checkpoint beats flat where the orientation shows; if not, train the lobe branch
  from scratch (no pilot lobes to unlearn) on the visibility-weighted loss alone first, then the rest.
* Cluster: seymour2's gpu.q is subordinate to the owning lab's queue -- pilot E was suspended seconds after it
  started and the profile job likewise; a 12-hour run there can be paused at any time (the trainer checkpoints
  every 15 minutes and resumes, so paused is survivable; killed is not).
* Per-subset density loss (10 crops, seeds 140M; the mixed targets): at *visible* line queries flat costs 0.190, the
  pilot 0.344 (its lobes are wrong there), D at step 500 0.181, F at step 1000 0.187; partial 0.104 / 0.414 / 0.107 /
  0.103; invisible 0.003 / 0.335 / 0.050 / 0.010. So the gated models beat flat where the orientation shows, but by
  2-5 % only, with lobes of peak/mean 1.1-1.4 against the mixed target's 2.45 -- the classical reader with the
  target's own kappa reaches ~40 % below flat there (targets review), so the room exists and the model is not
  reaching it. Pilot G: the density path fresh (`--init-fresh ent.dens,ent.od_net,ent.od_dens`) so nothing wrong has
  to be unlearnt; local G' the same at lr 3e-4.
* 08:50 ET: pilot F's gate converged to a constant (BCE 0.693 = output 0.5; its energy-contrast gains were being
  driven down), so the energy contrast is a weak cue for the visibility. Gate v3 reads the query's own structure
  tensor (anisotropy, line-ness, planarity: the cue the visibility is defined from) -- `st_products` on the image once,
  `window_mean` at each query, in training and inference (`2331e2a`). Pilot H = gate v3 + fresh density path + frozen
  trunk + density weight 4, lr 3e-4 (cluster, queued); local G' = the same with gate v2, running.
* Cluster at 08:50 ET: every job of ours is suspended (E, F, the profile) or queued (G, H): the owning lab's queue
  pre-empts gpu.q on seymour2 and seymour1. The local A6000 (one pilot at a time beside the 15 GB calibrator job) is
  the only dependable resource; the 12-hour run has not been launched.
* Pilot F at step 2000: fully flat (peak/mean 1.01; visible-query loss 0.189 against flat's 0.190; correlation 0.37).
  Starting from the pilot's (wrong) lobes, the gated lobe branch goes flat and stays. G' / H (fresh density path)
  are the remaining tests of the gate structure; the deciding number is the visible-query loss against flat's 0.190.

### ~09:00 ET: why no lobe forms -- the pedestal (the isolated harness, `lobe_learn.py` / `oracle_check.py`)

On the 8 fixed full-difficulty crops with the image-measured visibility (521 visible line queries), per-query density
loss as the trainer computes it (the mixed targets):

| prediction at visible queries | loss |
|---|---|
| flat | 0.458 |
| the dominant line entity's lobe (its axis and kappa) alone | 1.664 |
| ... mixed toward flat by v | 0.841 |
| **that lobe on its pedestal**: share x lobe + (1 - share), share = the entity's mass share (median 0.33) | **0.162** |
| ... mixed by v | 0.079 |
| the structure tensor read along u, exp(-g u^T M u), g = 1.5 / 3 | 0.433 / 0.476 |
| the same on the pedestal, g = 3 / 6 / 12 | 0.412 / 0.392 / 0.382 |

So at a visible query the target is a lobe holding a *third* of the mass over background and other populations. A
lobe alone -- however right -- is 3.6x worse than flat; the same lobe on its pedestal is 65 % better. The density is
multiplicative (softplus of a sum of logits), so no path could make a pedestal: every lobe it tried cost more than
flat, and flat was the optimum -- which is what every pilot found. Added: `--iso-head`, a per-point isotropic
pedestal added to the density. The structure-tensor axis on the right pedestal reaches -17 % (the model has to learn
a better axis than the tensor's for the remaining -48 %); `--st-path` gives it that axis as a start.
Isolated learnability so far: a fresh density path on the bank alone stays flat (+-1 % over 400 steps); with the
structure-tensor path at gain 12 the loss starts at +361 % (too sharp, no pedestal) and falls toward flat; at gain 3,
+69 % falling toward flat -- both consistent with the table.
* Isolated learnability with the pedestal head and the structure-tensor path (gain 12, centred on 1/3): starts at
  +100 % (peak/mean 1.69, the share wrong per query) and the optimiser again kills the lobe (+3 % by step 400,
  peak/mean 1.00) rather than fix the share per point -- the share is not linearly readable from the pilot's point
  code, and shrinking the lobe is the quickest descent. With the oracle share per query the same reader scores -17 %.
* Conclusion for the next structural step: supervise the decomposition explicitly. The target's oriented share at
  each query is known (the line entities' mass over the total); a **share head** trained on it (as the gate is on the
  visibility) and the density read out as ``share x lobe + (1 - share)`` makes the pedestal a regression target
  instead of something the lobe branch has to discover under the shape loss. The lobe branch then only has to get
  the axis right, where the oracle says -65 % and the tensor's axis -17 % are available at visible queries.
* **Share decomposition** (`--dens-share`, `6b9…`/`299a8a3` and after): in the isolated harness (fresh density path,
  gate v3, structure-tensor path, 400 steps) the lobes *hold* for the first time -- peak/mean 2.0 at visible queries
  throughout -- and the loss falls monotonically (+115 % -> +60 % against flat; the gate's loss 0.74 -> 0.14) where
  every earlier form collapsed to flat within 150 steps. Not yet below flat; the trend is ~8 % per 100 steps late.
  Pilot I = gate v3 + share + structure-tensor path + fresh density path + frozen trunk 1000 steps then 0.3x, density
  weight 4, lr 3e-4, 3000 steps: local (09:23 ET) and cluster (queued; G and H suspended again).
* **Pilot I (cluster, 2 H100, share form) at step 500 -- the first checkpoint below flat where the orientation shows**:
  visible line queries loss 0.171 against flat's 0.190 (-10 %), lobes of peak/mean 2.12, correlation with the raw lobe
  0.56 (58 % above 0.5). But it draws lobes at invisible queries too (peak/mean 2.21; loss 0.132 against 0.003), so
  overall it is still worse than flat (0.139 / 0.048): the share head (BCE 0.76 -> 0.21) has not yet learnt to be
  near zero where the tensor's axis is wrong -- which it cannot read from the tensor alone, so it needs the trunk
  (frozen until step 1000). Isolated harness: zeroing the pilot's density logit (`--reset-dens-logit`) made no
  difference (+95 % -> +52 % at 750 steps, plateau ~+60 %); a decomposition of the readout error (share vs lobe) is
  running.
* Decomposition of the harness model's readout at visible queries (loss, flat 0.458): its share x its lobe 0.50; the
  *true* share x its lobe 0.54 (its lobe is worse than the tensor lobe it started from: true share x tensor lobe
  0.37); its share x the oracle lobe 0.33; true share x oracle lobe 0.24; true share x the whole oriented part 0.18.
  Its share has median 0.08 against the true 0.23 and correlates -0.21 with it. On 8 crops with augmentation both
  pieces are poor; the cluster pilot with fresh data does better on the lobe (-10 % at visible queries) and its
  weak piece is the share head. Where the tensor's axis is confident but wrong, nothing in the tensor says so: the
  share head needs the trunk (unfrozen from step 1000) or is fundamentally limited there.
* Pilot I at step 500, axes at visible line queries (8 fresh crops, 226 queries): the model's peak axis is within 20
  degrees of the target on 62 % (median 12), of the tensor's axis on 80 % (median 11); the tensor's own axis is
  within 20 of the target on 73 % (median 9). So the lobe branch reads the axis (the structure-tensor path works);
  the share head predicts ~0.08 everywhere (the base rate of v x share) -- it is the piece that does not discriminate,
  which is why the lobes are weak where they should be strong and present where they should be absent.
* Pilot H (cluster, gate v3 with the tensor cue, fresh density path, *without* the share form) finished 2000 steps:
  gate BCE 0.74 -> 0.68 (the base-rate entropy again), lobe loss at visible queries 0.33 -> 0.31. Superseded by the
  share form (pilot I): without a pedestal the lobe branch cannot win even where it reads the axis.
* Pilot I at step 1500 (trunk unfrozen from 1000): visible 0.167 (flat 0.190, -12 %; correlation 0.63), partial
  0.101 (flat 0.104, below flat for the first time), invisible 0.064 (flat 0.003; was 0.132 at step 500), overall
  0.083 (flat 0.048). Every subset improving; the share head is beginning to discriminate once the trunk trains.
  Running to 3000.
* Evaluation caveat (11:00 ET): `dens_eval.py`, `axis_check.py`, `perf_image.py` drew their crops with the generator's
  own orientation visibility on (the module default), which training switches off (`generator.ORIENT_VIS` False with
  the image-measured targets) -- the generator's visibility softens the entities' kappa, so the "visible line
  queries" (kappa2 >= 2) in those evaluations were the subset the generator also called visible. Comparisons between
  models on them stand (same targets); the absolute counts do not. The scripts now match training; later numbers
  (pilot I at 3000 on) are on the training targets.
