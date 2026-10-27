# The full generator: fibre architecture, image formation and orientation truth (fully synthetic)

Branch `population-model`, planned 2026-10-26 and revised the same day. Two questions:

1. Can PLI fibre simulators and dMRI tract test sets (DiSCo and others) help develop the model as a fully synthetic,
   learned foundation model?
2. How should the full generator be set up, and how is each image made alongside its orientations?

This plan extends `plan_foundation_model.md` §5-6 and §13 and `plan_entity_field_model.md` §5. Licences and formats
marked ✓ were confirmed from the source on 26 Oct; the rest come from a literature scan (sources in §8).

Your instructions (26 Oct):
- **DiSCo is a test dataset only.**
- **No PLI image simulation.**
- **The simulators' concepts are reimplemented natively in our generator, not their packages,** to keep integration simple.

## 1. Short answer

Yes, through their **concepts**. The PLI and dMRI simulators are, at heart, **fibre-architecture generators**: they make dense, collision-free, realistically packed fibre geometry. Their forward models (Jones calculus for PLI, Monte Carlo diffusion for dMRI) are separate stages we do not need.

The architecture ideas fit our generator's S2 stage (skeletons) directly. Everything downstream stays as it is (our renderer, physics, acquisition and targets), so labels and images still come from one code path.

| concept | from | native version here | fixes |
|---|---|---|---|
| bundle sandbox: a trajectory + cross-section filled with fibres, radius per point | fastPLI, NFG, Fiberfox, Phantomas | `Scene.bundle` already does this; replace the hex lattice with **Poisson-disk seeding** and **gamma diameters** | W2, W3 |
| **collision solver**: capsules per segment, overlaps pushed apart under a curvature bound and length control | fastPLI, NFG, CACTUS stage 1 | `relax()` on the GPU (spatial hash), over every fibre and obstacle of a scene | **W1**, W2 |
| **radial growth** into free space after placement (FVF to 0.8-0.9, non-circular sections shaped by neighbours) | CACTUS stage 2, WMG | `grow()`: per sample, radius up to the gap to its neighbours; flattening / angle from the nearest neighbour (`flat`, `phi`, `sexp` exist) | W2 |
| myelin as a layer (g-ratio), nodes of Ranvier, beading | CACTUS, fastPLI layers, MEDUSA | `wall` = r (1 - g), g 0.55-0.8, lumen contents; periodic wall gaps; beading exists | realism |
| fibres yielding to cells and glia | MEDUSA, CATERPillar, WMG | cells / nuclei / vessels as sphere or capsule obstacles inside `relax` (replaces `deflect`) | realism |
| **orientation dispersion** drawn from a distribution (Watson / Bingham, κ set) | ConFiG, NFG | fibre directions within a bundle drawn about the axis at a set κ, then relaxed: the population's dispersion is a known parameter, not a by-product | W5 |
| **growth with crowding**: tips advance along a guidance field toward targets, steering around occupied space; fibres that cannot advance end | ConFiG, WMG | `grow_paths()` on `compose.Field` with an occupancy lookup; tortuosity and endings emerge from crowding | realism |
| **connectivity-defined bundles**: strands joining endpoint regions, merging through bottlenecks and fanning out | DiSCo / NFG / Fiberfox concept | `connections()`: endpoint patches, centrelines through random waypoints, shared trunks; tree strand → connection → region pair. **Our parameters, never DiSCo's layout** | hard cases |

Not taken:
- **PLI image simulation** (your instruction). The existing simple `pli` physics stays one of the contrast routes; no new work on it.
- **External packages.** None of fastPLI, CACTUS or NFG is installed or imported.
- **DWI.** Not an input to a scalar-image model.

**Tests.** DiSCo (✓ CC BY 4.0) is **test only**. Its strands, meshes and maps never enter training, generator or target checks, or model selection; it scores finished candidates under a fixed protocol. The dark-field crossing phantoms (Zenodo) are a second external test. Real data stays checks only.

## 2. How an image and its orientations are made today

**Principle: one geometry, two outputs.** The orientation labels come from the geometry. The image comes from the same geometry through materials, physics and acquisition. The image never makes a label; it only limits what the labels may ask for (visibility, S10).

```
 scene grammar ─► skeletons + sections ─► sweep render (8 samples / voxel) ─► per (voxel, population, object):
   (S1)              (S2)                       (S3)                              volume w, w·t tᵀ, w·descriptors
                       ▲                                                                 │
          new: pack, relax, grow,                                                        │
          disperse, connect (§4)                                                         │
            ┌────────────────────────── crop / zoom (S4) ◄──────────────────────────────┘
            │                                     │
   IMAGE SIDE                                       ORIENTATION SIDE
   masks (S5) → composite (S6) → materials +        entries kept exact (S3), masks applied (S5),
   physics (S7) → contrast routes (S8) →            composite factor per population (S6)
   acquisition: warp, PSF, noise,       ── warp grid + Jacobian ──►  tensors reoriented, points pulled (S9)
   artefacts (S9)                       ── visibility t-maps ─────►  mass × presence, tensor → isotropic where
            │                                                        orientation does not show (S10)
            ▼                                                                    ▼
        8-bit image                                            entity targets at query (x, σ) (S11)
```

| stage | what happens | orientation side | image side | code |
|---|---|---|---|---|
| S1 scene grammar | Families picked by voxel size (`scale_weight`). Crossing scenes (`cross_p`), laminae layout, neuropil underlay. Physical size priors in µm | which populations exist; the tree's parents | — | `scene.Scene.build` |
| S2 skeletons | Bundles of hex-packed fibres (jitter, twist, fan, split), Murray trees, sheets, foams (Voronoi), cells, helices, `compose.py` fields (crimp, ±α helical, laminar rotation, radial / tangential). Crossings angled / kissing / fanning; `interleave` (fascicle slabs, p 0.7 of angled crossings); `deflect` around obstacles | every object's tangent / normal is defined analytically | — | `Scene.bundle`, `compose.population` |
| S3 sweep render | Each cross-section template swept along rotation-minimising frames at step 0.5 voxel. Every sample point carries `t tᵀ` (line), `(I - n nᵀ)/2` (sheet) or a second moment (blob), plus descriptors and contrast. Points are splatted trilinearly | **exact**: per (voxel, population, object) the volume and the volume-weighted tensor; an entity's tensor is the exact sum of its objects' | per-population occupancy and contrast maps | `render_curves`, `Canvas.splat/entries` |
| S4 crop / zoom | 128³ canvas, 64³ crops, zoom by block-averaging (partial volume is linear) | entries mapped to crop coordinates | fields pooled | `draw_scene`, `_pool_fields` |
| S5 masks | Sparse labelling, specimen edge, cut-outs | removed objects leave the targets | ... and the image | `draw_scene` |
| S6 composite | Occlusion; any voxel holding more than one voxel of material is **scaled back to one**; interface layer (region boundaries, sheet tensor); isotropic remainder | composite factor per population (`comp`) | grey levels | `generator._composite` |
| S7 physics | Materials per population. `additive`, `xray` (μ, δ, Paganin, edge fringes), `darkfield` (`1 - gᵀTg`), `histology` (Beer-Lambert), `pli` (simple; unchanged), `mri` (χ anisotropy: R2* and frequency vs B0) | orientation enters the image only through `uᵀTu` terms | physical map → grey | `physics.form_image` |
| S8 routes | Blend of two physics, orientation-dependent term, per-population offsets, inner texture, shared intensities, grain, medium | — | contrast diversity | `draw_scene` |
| S9 acquisition | Warp / affine; acquisition axis; sections (misalignment, drift, chatter, lost sections); bias; PSF; resolution; Gibbs; CTF; noise (Poisson, Gaussian, Rician, low dose); missing wedge; thick slices; CT artefacts; window to 8 bits | the warp moves points; tensors `J⁻¹ S J⁻ᵀ`; descriptor sizes scale with the stretch | degraded image | `generator.nuisance` |
| S10 visibility | Per-layer t-statistics of the image regressed on each layer. Presence (t 3→8) scales mass. Orientation (t 6→18, from each layer's **fibre-scale texture**, now default) blends the tensor toward isotropic | the targets ask only for what shows | — | `generator.visibility`, `ORIENT_FROM_TEXTURE` |
| S11 targets | At query points x and scales σ: Gaussian window over the entries; objects lifted to the first ancestor with radius ≥ max(1.5, 0.5 σ_eff) unless alone; mass × visibility; tensor dispersion floor 0.94; Bingham moment-matched; background, total density, descriptors, tree nodes | the training targets | — | `entities.entity_targets`, `bingham.fit` |
| S12 batch | Cube symmetries (SH rotated exactly), intensity jitter | rotated with the image | | `augment.py` |

### 2.1 Where the chain is weakest (from the code and the 25-27 Oct evidence)

- **W1, crossing cores.**
  - Mechanism: unless `interleave` fires (p 0.7, angled crossings only), the fibres of two crossing bundles overlap in space. S6 then clamps the voxel's material to one, so the image shows a solid, featureless core while the labels keep both populations.
  - Why it matters: real crossings are interwoven. Fibres deflect around one another, so the core keeps fibre-scale texture in both directions; every collision-solving simulator produces this.
  - Implication: our generator asks the model to separate populations whose image the generator itself erased. This is a plausible contributor to the crossing failure (0/9 generator; crossings fail C3-C9 on phantoms). `phantom2` interleaves in one plane only.
- **W2, packing.**
  - Hex spacing (1.15 × 1.05-1.5 × 2r) gives a fibre volume fraction (FVF) of about 0.3-0.65 inside bundles. Myelinated white matter is 0.6-0.8; CACTUS-style growth reaches 0.9.
  - Diameters are log-normal with sd 0.12, narrow against real gamma distributions.
- **W3, regular packing.** A jittered hexagonal lattice gives periodic texture. Simulation papers report artefacts from symmetric packings; for us the risk is that the network learns a lattice cue that real tissue lacks.
- **W5, orientation truth vs dispersion.**
  - The 0.94 dispersion floor is one constant. The geometry's truth (the window's second moment of the real tangents) already carries dispersion; the floor should be the image's resolution limit (σ, voxel size and PSF), not a constant.
  - Within-bundle dispersion is a by-product of smoothed offsets, not a set parameter.
- **W6, no independent check of our targets.** All targets come from our own S3 / S11 code. A bug there would be invisible, because the ground-truth checks feed our targets back as the model. The fix is a second, independent implementation (§4 part 7).

(The old W4 was PLI physics; dropped with PLI simulation.)

## 3. External resources: concepts and tests (scan of 26 Oct)

### 3.1 Where each concept comes from (concepts only; no package is used)

| source | concept taken | key numbers to match |
|---|---|---|
| fastPLI (Matuschke 2019, arXiv:1901.10284; ✓ MIT) | bundle sandbox; capsule collision solver (octree, curvature bound r_min, push 0.01-0.1 µm per step); radius varying along each fibre | FVF 65 % (one population), 52 % (two crossing), 45 % (three) after solving |
| CACTUS (Villarreal-Haro 2023, doi:10.3389/fninf.2023.1208073; ✓ MIT) | two stages: global placement / overlap optimisation, then radial growth into free space; g-ratio myelin; non-circular sections | FVF 0.90-0.95 aligned; 0.91 at 25° dispersion; 0.92-0.94 in 30-90° crossings |
| NFG (Close 2009, doi:10.1016/j.neuroimage.2009.03.077) | strands joining endpoint regions; joint optimisation of overlap, length and curvature; hexagonal subdivision of bundles | — |
| ConFiG (Callaghan 2020, doi:10.1016/j.neuroimage.2020.117107) | growth: tips advance toward targets and avoid collisions; Watson / angular-Gaussian dispersion | FVF 0.71-0.75; up to 3 crossing bundles |
| MEDUSA / CATERPillar / WMG | axons as sphere or ellipsoid chains (cheap collision primitives); beading; glia as obstacles; demyelination | FVF 0.72-0.83 (WMG); up to ~0.75 (CATERPillar) |
| Andersson 2020 XNH (doi:10.1073/pnas.2012533117) | real axon statistics: radius CV along an axon, tortuosity, beading, deflection around cells | statistics to check our draws against, never geometry |

### 3.2 Test sets (never trained on)

| test set | truth | use |
|---|---|---|
| **DiSCo 1-3** ✓ CC BY 4.0 (Mendeley doi:10.17632/fgf86jdfg6), **test only** | 12,196 strands in a 1 mm sphere: centrelines, diameters, end ROIs, meshes; ODF, volume-fraction, count and diameter maps; 16×16 connectivity; up to 82 strands / 5 bundles per 25 µm voxel | Render the strands through our modalities at 1-25 µm (a small reader for its `.tck` + diameters, kept in the test harness, not the generator). Score entities and the hierarchy (strand → bundle → ROI pair) against DiSCo's released maps and strands. Protocol (renders, crops, seeds, metrics) fixed and committed before the first model is scored; scored only for finished candidates |
| **Dark-field crossings** (Huyge 2023, Zenodo doi:10.5281/zenodo.8089557) | GATE Monte Carlo, carbon-fibre crossings at 40-90° | the crossing-angle limit against published CSD (~50°) and Funk-Radon (~70°) |
| ISMRM-2015 Tractometer (Zenodo 572345) | 25 ground-truth bundles (streamlines) | macro scale, hierarchy and following; later |

Selection and development use our own sets: the phantom families (selection and final sets), held-out generator crops, and a new **held-out fibre-architecture family** (§4) with its own seeds.

### 3.3 Lessons applied

1. Regular packings create artefacts (W3): randomised dense packing.
2. The training distribution sets the prior (Gyori 2022, doi:10.1002/mrm.29014): mix uniform parameter sampling with realistic combinations (packing, dispersion, diameters).
3. Every DiSCo challenge team made the same false connections at bottlenecks: geometry-driven hard cases. Generic bottlenecks (converging, passing, diverging bundles) go into our grammar from our own parameters, never copied from DiSCo's layout.
4. Single-orientation readouts are biased by mixtures. With interwoven crossings, the image of a mixture is realistic and the targets keep both populations: the case the entity separation must learn.

## 4. The full generator: design

Everything new happens in S2 (skeletons) and feeds the same S3 renderer, so the target code (S11) does not change.

**Non-fibrous content stays** (your instruction, 26 Oct). The fibre options change bundles only. Cells, granules,
crystals, blobs, sheets, foams and the isotropic-only scenes (8 %) stay in the mix, with cells and vessels as fixed obstacles
in the relaxation. `scripts/learn/content_mix.py` measures the share of mass per kind and the non-oriented share with the
options off and on. Every experiment trains on the default family mix (never bundles only) and reports that table. A new module `learn/fibres.py` holds the parts below. It works on our existing `Curves` (points `P (N, S, 3)`, radius `rad (N, S)`, ids, parents) and is called from `Scene.bundle`, `compose.population` and a new `Scene.connections`.

1. **Packing** (`pack_section`).
   - Poisson-disk seeding of a bundle's cross-section, with radii drawn from a gamma distribution (mean and shape per scene, in µm).
   - Target FVF drawn 0.3-0.85; no lattice.
   - Fascicles from the seeding (clusters), as today, for the deep tree.
2. **Dispersion** (`disperse`).
   - Each fibre's local direction is drawn about the bundle axis from a Watson distribution at κ (drawn 2-200), integrated along the fibre with a correlation length.
   - κ is recorded per population, so the truth's dispersion is a known parameter.
3. **Collision relaxation** (`relax`, GPU).
   - Primitives: capsules per segment (sphere chains as the cheap alternative) for every fibre in the scene, plus cells, nuclei and vessels as obstacles.
   - Each iteration:
     - a spatial hash finds candidate pairs;
     - overlapping pairs are pushed apart, split by radius;
     - curvature is limited (circumradius ≥ r_min);
     - segment length is restored;
     - the fibre is smoothed.
   - A few hundred iterations or until overlap < 1 %.
   - Crossing bundles interweave instead of being clamped, kissing bundles flatten against each other, and fibres yield to cells. This replaces `interleave` and `deflect`, which stay available as fallbacks.
4. **Growth to high FVF** (`grow`).
   - After relaxation, each sample's radius grows toward the free gap to its neighbours, up to the target FVF.
   - Sections flatten across the nearest neighbour's direction (`flat`, `phi`), so axons in dense packing become non-circular.
   - Myelin: `wall` = r (1 - g) with g 0.55-0.8 and distinct lumen contents; nodes as short periodic wall gaps; beading as now.
   - Optional demyelination: patches where g → 1.
5. **Crowded growth** (`grow_paths`).
   - Fibres grown tip by tip along a `compose.Field` toward target regions. Each step checks an occupancy grid (the canvas, coarse) and steers around occupied space; a fibre that cannot advance ends (an endpoint).
   - Tortuosity and endings emerge from crowding. This is the slow path, used in a share of dense scenes only.
6. **Connectivity bundles** (`Scene.connections`).
   - Endpoint patches on the canvas faces, on a sphere or on scene surfaces.
   - Region pairs joined by bundles whose centrelines pass random waypoints; shared trunks through bottlenecks, fanning at the ends.
   - Tree: strand → connection → region pair → root.
   - Endpoints go to the points map. All parameters are ours (DiSCo is never sampled).
7. **Independent target check** (W6).
   - A second, deliberately simple implementation: for each voxel, 8³ supersampled points tested against the capsules directly (inside which fibre, which tangent). It yields per-(voxel, object) volume and `t tᵀ`, compared with `render_curves` + `Canvas.entries` on small scenes.
   - Analytic cases are checked exactly: a straight tube (volume fraction and axis known); two tubes crossing at a known angle.
   - Thresholds are fixed beforehand. The check runs in the test suite, not in training.
8. **Truth upgrades** (S10-S11).
   - The dispersion floor becomes a resolution floor (σ, voxel size, PSF), not 0.94 everywhere (W5).
   - Orientation visibility from oriented texture stays the default.
   - A **standing inferability check** per sample (structure tensor on single-population regions, the 27 Oct check) is recorded in `rec` and summarised per family, so a new family that asks for invisible orientation shows up at once.
9. **Benchmarks.** `scripts/learn/bench_external.py`:
   - the dark-field phantoms and the DiSCo test protocol, through the C1-C9 criteria (ground-truth check first);
   - DiSCo metrics: ODF angular correlation, FVF error, bundles per voxel, strand → bundle grouping (adjusted Rand);
   - baselines: structure tensor + clustering, and CSD on DiSCo's DWI;
   - DiSCo results never feed back into choices.

Throughput budget: `relax` on ~400 fibres × ~500 samples per canvas is ~2×10⁵ capsules; a hashed iteration is milliseconds on the GPU, so a few hundred iterations is ~0.5-1 s per scene, amortised over 8 crops (today 0.07 s per pair amortised). `grow_paths` is the expensive one, hence its limited share.

## 5. Workplan

Effort is agent working time; compute is the A6000. Every phase ends with tests, an adversarial review and a note in `notes/`. New options are off by default until their experiment passes.

| phase | content | deliverables / tests | effort | gate |
|---|---|---|---|---|
| **P0** check first | Independent voxeliser and analytic cases (part 7); inferability summary in `rec` (part 8) | `render_curves` agrees with the voxeliser on straight, curved and crossing tubes within fixed thresholds; any disagreement fixed before new families are trusted | 0.5-1 d | W6 closed |
| **P1** packing + dispersion | `pack_section` (Poisson-disk, gamma radii, target FVF), `disperse` (Watson κ) in `Scene.bundle` and `compose` | FVF reached ±0.05; no lattice peak in the section's pair correlation; drawn κ recovered from the rendered tensors | 1 d | — |
| **P2** `relax` | GPU collision relaxation with obstacles; crossings, kissing, fans and splits through it; `interleave` / `deflect` kept as fallbacks | overlap < 1 % of fibre volume; curvature bound held; FVF at two-population crossings ≥ 0.45 (fastPLI's figure); **structure-tensor inferability at crossing cores up vs off** (the W1 test); time per scene within budget | 1.5 d | — |
| **P3** `grow` + myelin | Radial growth, non-circular sections, g-ratio walls, nodes, demyelination | FVF up to 0.8-0.9 aligned; radius CV, tortuosity and beading compared with the XNH statistics | 1 d | — |
| **P4** connections | `Scene.connections`: endpoint patches, waypoints, bottlenecks, fanning; the tree | tree depth and parents correct; endpoints in the points map; entity unit tests (sum to total, merge rule) | 1 d | — |
| **P5** crowded growth | `grow_paths` on fields with occupancy steering | endings and tortuosity from crowding; cost per scene measured; share set from it | 1 d | after E1 / E2 |
| **P6** truth + benchmarks | Resolution-based dispersion floor; the held-out fibre-architecture family; `bench_external.py` with the DiSCo protocol committed before any model is scored; baselines | ground truth passes all criteria on every family; baseline tables | 1 d | before any P7 run |
| **P7** experiments | **E1** `relax` on / off (families + generator, same seeds, ~6 h each); **E2** + packing / growth / dispersion; **E3** + connections and crowded growth | selection on phantom families, held-out generator crops and the architecture family; crossing angle-resolution curves; **DiSCo and dark-field scored once on the chosen candidates** | ~2 d compute | cluster run decision |

Order:
- P0 first, so every later family is rendered by checked code.
- P1 → P2 → P3 is one line; P4 can run beside P3.
- P6 lands before any P7 run, so nothing is scored on a moving benchmark.
- **E1 is the first experiment.** If interwoven crossings move the crossing criteria, the generator was the bottleneck. If they do not, the next lever is the equivariant trunk (overnight 27 Oct §7).

### 5.1 Crossing track (the generator plan alone does not cover it)

The plan above fixes the **data-side** cause of the crossing failure: generator crossings whose image the generator
erased (W1). The phantom evidence points to a **second, model-side** cause that no generator change reaches:

- `phantom2` crossings are already cleanly interleaved: two fibre lattices offset by half a spacing, no overlap, no
  clamp. Their cores show both directions, yet held-out crossings in random 3-D orientation fail. Axis-aligned
  crossings passed 12/12 but rotated ones 0/4, and the core density is broad.
- One rotated crossing is fitted in ~100 steps, so the architecture can represent it.
- None of these moved it: longer training (fam_F), a wider trunk (famW), the oriented probe, or test-time rotation
  averaging.

So the network does not form direction-resolved features that generalise across orientations. Steps, in order of cost:

| step | what | decides |
|---|---|---|
| **X0** diagnosis (~1 d, no new training) | (a) crossing loss on training draws vs held-out: is it underfitting or generalisation?; (b) error vs the crossing plane's angle to the voxel grid; (c) a classical upper bound: two directions recovered at phantom and generator crossing cores from local evidence (a window's gradient-orientation histogram fitted with two lobes, Kjer / Gray style) | whether the core's orientation is local and recoverable, and whether the failure is fit or transfer |
| **X1** oriented evidence as input (~1 d) | classical, rotation-steerable oriented-filter energies at fine scales (a bank of steerable second-derivative / quadrature filters over ~15-30 directions, or the 4th-order structure tensor) given to the trunk beside the image | the cheapest model lever, and classical maths as you prefer: the network decodes oriented evidence it would otherwise have to discover |
| **X2** multiplicative direction path (~0.5 d) | the vote and embedding conditioned on the direction multiplicatively (FiLM by `Y(u)`, or harmonics of the point code as the density has) -- the saddle noted on 25 Oct | whether the per-direction ownership switch is the bottleneck |
| **X3** equivariant fine trunk (~2-3 d) | group convolutions over the 24 cube rotations (or steerable SO(3) filters) at strides 1-2 | the structural fix if X1 / X2 fall short |

Run X0 alongside P0, before E1, so E1's outcome can be read correctly. If X0(c) shows both directions are locally
recoverable and E1 does not help, X1 → X2 → X3 is the line; if E1 helps generator crossings but phantoms stay at 0/4,
both causes were real.

## 6. Decisions

- **D1 (settled 26 Oct): no PLI image simulation.** The existing `pli` contrast route stays as it is. Say if you want it removed from the physics mix.
- **D2 (settled 26 Oct): DiSCo is test only.**
- **D3 (settled 26 Oct): concepts, not packages.** No external simulator is installed or imported.
- **D4 (open): compute.** All of the above runs locally. The 48 h GPU pilot script is still unsubmitted and needs your go-ahead.

## 7. Limits of this route

- **Tube models.** Fibres are tubes (or sphere chains). Glia, vessels and cells come from our own families as obstacles; real axon statistics (XNH) are the check.
- **Our own implementation.** Reimplementing the concepts means our solver is not the published one. The checks are behavioural: overlap, FVF reached, curvature, the inferability check, and the XNH statistics. Agreement with fastPLI's or CACTUS's exact output is not tested.
- **Scale.** Dense packing matters where fibres are resolved or nearly resolved (0.5-5 µm voxels). At coarser voxels it shapes the partial-volume texture only, and large-scale coherence stays with the tracts, laminae and BigBrain surfaces.
- **One external world.** DiSCo is three phantoms in one 1 mm sphere: a clean final check, not a selection set. The dark-field phantoms add a second, narrower one.
- **No guarantee on crossings.** E1 tests whether the crossing failure is in the data; the architecture levers remain open.

## 8. Sources

- fastPLI: github.com/3d-pli/fastpli ✓ MIT; JOSS doi:10.21105/joss.03042; dense models arXiv:1901.10284.
- CACTUS: github.com/Juanitovh/CACTUS ✓ MIT; doi:10.3389/fninf.2023.1208073.
- NFG doi:10.1016/j.neuroimage.2009.03.077.
- ConFiG doi:10.1016/j.neuroimage.2020.117107.
- MEDUSA doi:10.1016/j.neuroimage.2019.02.055.
- CATERPillar doi:10.1101/2025.06.20.660694.
- WMG doi:10.3389/fninf.2024.1354708.
- Andersson 2020 doi:10.1073/pnas.2012533117.
- DiSCo: data doi:10.17632/fgf86jdfg6 ✓ CC BY 4.0; doi:10.1016/j.dib.2021.107429.
- Huyge 2023 doi:10.3390/polym15132887; Zenodo doi:10.5281/zenodo.8089557.
- ISMRM-2015 doi:10.1038/s41467-017-01285-x; Zenodo 572345.
- Gyori 2022 doi:10.1002/mrm.29014.
- Dohmen 2015 doi:10.1016/j.neuroimage.2015.02.020 (mixture bias; context only).

## 9. Status (2026-10-27, 23:45 ET)

| item | status |
|---|---|
| P0 / P1-P2 (Monte Carlo packing + relaxation) | **built**: `learn/fibres.py`, GPU neighbour search; `--mc-fibres` in the trainer and data workers; off by default |
| Monte Carlo crossings in the phantoms | **built**: `phantom2` kind `fibres`; the `cross` / `fine` families now use them, the lattices kept as `cross_lattice` / `fine_lattice` |
| Non-fibrous content check | **built**: `scripts/learn/content_mix.py`; the kind and non-oriented shares unchanged with the options on |
| X0 diagnosis | **done**: `notes/x0_mc_fibres_2026-10-26.md`; the lattice phantoms put a third (stacking) direction in the image; Monte Carlo crossings are recoverable by a classical filter (19/26 vs 0/26) |
| X1 (filter bank as input channels), X2 (direction harmonics) | **no effect** on fitting 64 crossings |
| **Direct oriented path** (`PopulationNet(odirect=256)`) | **built, works**: held-out crossings 0.023 → 0.112 (target 0.122) in the fast screen; the phantom pilot raises direction (C6) 11 → 24/32; `notes/framing_review_2026-10-27.md` §6-7 |
| Voxel-first ownership | **tested and dropped** (§7 of the framing note) |
| E1 (generator with / without relaxation) | **paused** at ~step 775 of 2000 |
| P3-P8 | not started; next are the density-to-entities readout and mass calibration (framing note §7) |
