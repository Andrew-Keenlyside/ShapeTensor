# BRIDGE: anatomical curvature, cortical containment and structure-tensor limits

**Confidence tags.**
- **[V]**: read in the full text, PDF or official docs.
- **[A]**: abstract, index record or search summary only.
- **[M]**: memory, not re-checked.
- **[D]**: my calculation.

**Formulas.** R = s / (2 sin(θ/2)). A turning rate of θ°/mm equals R = 57.3/θ mm.

## 1. Fibre curvature and tortuosity

**Radii of named bundles.** I found no measured radius of curvature in mm for Meyer's loop, the fornix, the genu/forceps, the arcuate or U-fibres. Do not quote one. What exists:
- **Meyer's loop position.** The tip lies 23.1 mm (left) and 26.4 mm (right) from the temporal pole (Shan 2019, *BMC Neurol*, 10.1186/s12883-019-1537-6) [V].
- **Old MRtrix limit.** MRtrix 0.2 used a 0.2 mm step and a 1 mm minimum radius. This was "too conservative to reconstruct Meyer's loop" (Besseling 2012, *PLoS ONE*, 10.1371/journal.pone.0034125) [V].
- **Curl.** Yeh 2020 (*NeuroImage* 223:117329, 10.1016/j.neuroimage.2020.117329) defines curl = length/span, the inverse of BRIDGE straightness [V]. BRIDGE's 0.29 is a curl of 3.4. Per-bundle values are only in Fig. 4b, which I could not read.

**Tightest documented macroscopic turn: entering the sulcal wall.**
- Schilling 2016 (ISMRM abstract 0924, squirrel monkey, myelin histology) [V]:
  - In WM, wall fibres run at 77° ± 11° to the surface normal. They come within 28° of the normal only about 600 µm into cortex.
  - Crown fibres stay within 20° of the normal.
  - [D] About 49° over ≥0.6 mm gives R ≲ 0.7 mm, and the turn happens inside GM.
- Budde & Annese 2013 (*Front Integr Neurosci* 7:3, 10.3389/fnint.2013.00003) [V]: "sharp bends into the cortex" at sulcal walls.

**Microscopic.**
- **Andersson 2020** (*PNAS* 117:33649, 10.1073/pnas.2012533117), 75 nm XNH, monkey splenium [V]:
  - Bundle dispersion is 7°.
  - Single-axon dispersion is 14° at 1 µm and 7° at 30 µm.
  - Axons deviate 2–9 µm around a 10 µm vessel.
- **Lee 2019** (*Brain Struct Funct* 224:1469, 10.1007/s00429-019-01844-6), mouse 3D SEM [V, preprint text]: dispersion ~17°, sinuosity 1.04.
- **Abdollahzadeh 2021** (*Commun Biol* 4:179, 10.1038/s42003-021-01699-w), rat [V]: tortuosity 1.068 (CC) and 1.072 (cingulum).
- **Kjer 2025** (*eLife* 13:RP94917, 10.7554/eLife.94917) [V]: tortuosity 1.01 in monkey CC and 1.04 in centrum semiovale.
- **Mortazavi 2018** (*Cereb Cortex* 28:1219, 10.1093/cercor/bhx034) [A]: monkey axons make turns of radius ≤15 µm between two primary axes. Fewer than 20% follow smooth curves.

At 0.6 mm sampling, deep WM is nearly straight. [D] HCP-1065's 7°/mm corresponds to R ≈ 8 mm. Sharp turns occur far below the sampling scale (≤15 µm) or at cortical entry (R ≈ 0.5–0.7 mm).

**Tracker defaults, converted to R** [D; parameters V from MRtrix tckgen, DSI Studio CLI and DIPY CLI docs]:

| Tracker | Step | Angle | R |
|---|---|---|---|
| MRtrix iFOD2 | 0.5 voxel | 45° | 0.65 voxel (0.82 mm at 1.25 mm voxels) |
| MRtrix iFOD1 | 0.1 voxel | 15° | 0.38 voxel (0.48 mm) |
| MRtrix SD_STREAM / FACT | 0.1 voxel* | 60° | 0.10 voxel (0.13 mm) |
| DSI Studio (default = random) | 1 voxel | 45–90° | 1.31–0.71 voxel |
| DIPY `dipy_track` | 0.5 mm | 30° | 0.97 mm |
| DIPY `dipy_track_pft` | 0.2 mm | 20° | 0.58 mm |
| HiP-CT ST tractography (Wanjau 2026) | ~0.2 mm | 30° | 0.39 mm |
| **BRIDGE cap** | 0.76 mm | 75° | **0.62 mm** |
| **BRIDGE jitter alone** | 0.76 mm | 29° | **1.52 mm** |

\*The docs give 0.1 voxel for "first-order algorithms"; applying that to SD_STREAM and FACT is my inference.

**Key derivation [D].**
- 29° per 0.76 mm step is 38°/mm. BRIDGE's observed median is 37°/mm, so jitter alone can explain the curl.
- BRIDGE's cap is no more permissive than standard caps. Standard trackers stay smooth because they follow an interpolated, continuous orientation field in short steps, not because of their caps.
- Tightening the cap cannot remove jitter.

**Defensible hard limit.** No paper proposes one.
- My inference: any global cap must allow R ≈ 0.5–0.7 mm, or it blocks cortical entry, yet deep WM is about 10× straighter.
- Better options: a smoothness prior or smoothed path, or a cap that depends on tissue (tight in deep WM, relaxed within about 1 mm of the boundary).

## 2. Containment at the WM–GM interface

**Evidence.**
- **Reveley 2015** (*PNAS* 112:E2820, 10.1073/pnas.1418198112), macaque, 250 µm dMRI [V]:
  - About 50% of cortex was inaccessible to long-range tracking. With WM seeding, 47% was never penetrated.
  - The cause is dense axon sheets running parallel to the surface within about 1 mm of it, mostly sulcal but also on many crowns.
- **Schilling 2018** (*HBM* 39:1449, 10.1002/hbm.23936) [A]: gyral bias appears across algorithms. In the 2016 abstract, the crown:wall density ratio was 1.52 ± 0.69 in histology and 2.1–2.5 in tractography [V].
- **Cottaar 2018** (*NeuroImage* 176:417, 10.1016/j.neuroimage.2018.04.040) [V]:
  - In superficial WM, fibres sit within a median <7° of the tangential plane, mostly along the sulcal axis.
  - Inside GM they turn radial. The transition's offset and width are both under 1.25 mm.
- **Models of the change** [A]:
  - Cottaar 2021 (*NeuroImage* 227:117693, 10.1016/j.neuroimage.2020.117693) models gyral blades as a divergence-free vector field.
  - Asymmetric FODs give "sharper turns into the cortex": Bastiani 2017 (*NeuroImage* 158:205, 10.1016/j.neuroimage.2017.06.050) and Wu 2020 (*MedIA* 59:101543, 10.1016/j.media.2019.101543).
- **Histology** (Budde & Annese 2013) [V]: layer 1 is tangential, layers 2, 3 and 5 radial, layers 4 and 6 mixed. In sulcal depths fibres run parallel to the interface.

**Methods.**
- **ACT** (Smith 2012, *NeuroImage* 62:1924, 10.1016/j.neuroimage.2012.06.005) [A]. Its rules are [M] plus the developer's mailing-list posts: stop on entering cortical GM, and reject streamlines that enter CSF. The MRtrix options `-backtrack`, `-crop_at_gmwmi` and `-seed_gmwmi` exist [V].
- **PFT** (Girard 2014, *NeuroImage* 98:266, 10.1016/j.neuroimage.2014.04.074) [A]: probabilistic stopping from partial-volume maps. Streamlines that stop in WM or CSF backtrack and resample. DIPY defaults: back 2 mm, forward 1 mm [V].
- **SET** (St-Onge 2018, *NeuroImage* 169:524, 10.1016/j.neuroimage.2017.12.036) [A]: seeds and stops on the surface, with a surface flow. It reduces gyral and length bias.

**Rules that work:**
1. Stop on entering GM.
2. Reject or backtrack streamlines that end in WM or CSF.
3. Seed at the interface.
4. Let orientation rotate from tangential to radial within about 1 mm.

**For BRIDGE [D].** Tangential travel in the superficial-WM band is real (U-fibres, Reveley's sheets). Tangential travel *inside GM*, or along the boundary itself, is not.

## 3. Structure-tensor tractography on microscopy and X-ray

**Validation.**
- **Budde & Frank 2012** (*NeuroImage* 63:1, 10.1016/j.neuroimage.2012.06.042) [A]: 2D structure tensor on histology; anisotropy correlated with DTI.
- **Khan 2015** (*NeuroImage* 111:192, PMC4387076) [V]:
  - 0.25 µm confocal data, σ = 2 µm, windows of 12.5–112.5 µm. The window must "span the fiber cross-section".
  - They had to correct depth attenuation, shrinkage and PSF anisotropy.
  - No coherence weighting and no vessel handling.
- **Schilling 2016** (*NeuroImage* 129:185, 10.1016/j.neuroimage.2016.01.022) [A]: CSD against confocal structure-tensor FODs gave ~6° error for parallel fibres and 10–11° for crossings.
- **Kleinnijenhuis 2012** (STIFT, *NeuroImage* 59:3941, 10.1016/j.neuroimage.2011.10.078) [A]: a structure tensor of 0.5 mm gradient-echo images steered dMRI tracking and extended Meyer's loop.
- **Trinkle 2021** (*MRM* 86:1067, 10.1002/mrm.28776) [A]: structure-tensor FODs from 1.17 µm mouse micro-CT. Their ISMRM 2020 abstract reports it beat MRI against tracers even when binned to MRI resolution.
- **Not verified or not identified:**
  - Axer 2011 (*NeuroImage* 54:1091, 3D-PLI ≤100 µm) [A].
  - Dyer 2017 and Mortazavi 2018: I could not confirm they used a structure tensor.
  - "Brazzle": not identified.

**HiP-CT and XPCT.**
- **Walsh 2021** (*Nat Methods* 18:1532, 10.1038/s41592-021-01317-x) [V]: brain at 25.08 µm, zooms at 6.05 and 2.45 µm. Only gyri and sulci are described at 25 µm, and WM orientation was not analysed.
- **Wanjau 2026** (bioRxiv, 10.64898/2026.04.02.715729; Keenlyside co-author) [V]:
  - Structure tensor on the 15 µm hemisphere, σ = 2, ρ = 4 (units unstated), fODFs on 800 µm supervoxels.
  - Vessels were masked at the gradient level, because pre-masking "creates voids in intensity, leading to artificially high gradients".
  - Failure modes: "uniform tissue density or weak phase contrast" gives unstable eigenvectors and spurious streamlines; streaks at sharp density transitions.
  - Agreement with dMRI was only qualitative.
- **Kjer 2025** [V]: multi-scale structure tensor (8 scales) with vessel voxels excluded. Orientation was consistent across scales, but anisotropy depended on FOV and resolution.
- **Gray 2025** (bioRxiv, 10.1101/2025.01.16.633408) [V]:
  - A single orientation is robust to parameters, but crossings are fragile: σg +0.5 px took the error from 4° to 40°.
  - Recommends σw ≈ structure width. Advises against the structure tensor when any dimension is coarser than the structure.

**Edge artefact [D, standard algebra].**
- At a step edge the tensor is rank-1 (λ1 ≫ λ2 ≈ λ3), so e3 can point anywhere in the tangent plane. That is exactly BRIDGE's hypothesis. A fibre texture gives λ1 ≈ λ2 ≫ λ3.
- Remedies:
  - Gate on linearity (λ2−λ3)/λ1 rather than planarity (λ1−λ2)/λ1 (Westin 2002, *MedIA* 6:93) [M].
  - Mask boundary gradients before integration, as Wanjau did for vessels.
  - Use edge-preserving tensors (Brox 2006) [M].
  - Use scale selection (Lindeberg 1998, *IJCV* 30:79, 10.1023/A:1008045108935) [A].
- No cited paper validates edge rejection at cortex.

## 4. Scale and resolution limits

**Contrast.** Chourrout 2022 (*Biomed Opt Express* 13:1620, 10.1364/BOE.438832) [V]:
- In formaldehyde-fixed tissue, WM is "faint hypointense", and "phase contrast alone is not sufficient to enhance white-matter tracts".
- Ethanol dehydration makes myelinated WM hyperintense, and lipid clearing abolishes that signal.
- HiP-CT brains are dehydrated to 70% ethanol (Walsh 2021) [V].
- The WM/GM boundary is therefore a strong edge at any voxel size.

**Texture.**
- Monkey CC laminae are up to 40–45 µm thick and mouse laminae 5–40 µm (Kjer 2025) [V]. I found no human centrum-semiovale fascicle size.
- [D] At 100 µm, σ is 1 voxel and ρ is 3 voxels, and laminae are sub-voxel. Homogeneous-WM texture must therefore come from partial-volume striation, vessels or noise.
- Vessels are a poor proxy: median 28–34° from the nearest fibre, and perpendicular to association pathways (Schilling 2025, *Sci Rep* 15:18396, 10.1038/s41598-025-99724-z) [V].
- Wanjau reports instability in uniform regions even at 15 µm.

**Integration scale.** The window should span the structure (Khan, Gray). [D] With its kernel tail, a 0.3 mm ρ mixes gradients from the boundary and from ~1 mm superficial-WM sheets, so the rank-1 boundary gradient dominates near cortex. This is consistent with the observed wrapping.
