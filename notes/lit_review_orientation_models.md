# Scoping review: fibre orientation / ODF models and their downstream uses (2026-10-25)

Three independent searches: methods, applications by tissue and material, and the analyses that consume orientation
outputs. Every reference was opened by the searcher; items marked ⚑ have an unconfirmed detail (venue, DOI or a
figure taken from an abstract). Purpose: before the cluster run, find downstream uses the entity-field model would
miss, and sort what must go into the generator or training now from what can be added afterwards.

## 1. Where the model sits

- **No published learned model gives per-population ODFs, masses or a hierarchy from structural 3-D images.** Closest:
  - DeepFixel (2025) splits a voxel ODF into fixels, but from dMRI only.
  - Kjer et al. 2025 (eLife) fit Binghams to structure-tensor histograms from synchrotron data; this is classical.
  - Gray et al. 2025: structure tensor with k-means, giving 1–3 populations, sensitive to its parameters at crossings.
  - ComSLI (Georgiadis 2025): up to 3 in-plane populations, from the physics of scattered light.
  - CL-3D (Oberstrass 2024): self-supervised 3D-PLI features that follow crossings, but with no orientation output.
- **Training only on synthetic, contrast-agnostic data** is established, but for segmentation and representation
  learning only (SynthSeg, anatomix, vesselFM, Chollet 2024). Synthetic-only orientation exists for composites (Bleiziffer 2021).
- **dMRI deep learning** regresses fODFs in SH: angular super-resolution (FOD-Net, FOD-Swin-Net), few-direction
  acquisitions (Kebiri 2024), rotation-equivariant sparse ODFs (Elaldi), neural fields with uncertainty (Consagra) and,
  recently, explicit fixels (DeepFixel, PRISM). I found no dMRI foundation model that outputs fODFs.
- **Orientation from microscopy, PLI, SLI, CLARITY, X-ray and PS-OCT is still classical** (structure tensor, Frangi,
  FFT, physics), with populations found afterwards (peaks, k-means, Bingham). Learned work there is mostly
  segmentation or tracing of single fibres (TrailMap, flood-filling networks, X-ray nanoholotomography axons).

**What is new here:**
- one learned, modality-agnostic model with Bingham ODFs and masses per population;
- an any-depth hierarchy, matching for example Kjer's laminar fasciculi within pathways;
- following identities across windows;
- training on synthetic data only.

## 2. Downstream uses, and what each needs

| use | needs from the model | status |
|---|---|---|
| Tractography, microscopy and MRI-style (MRtrix, 3D-PLI style) | SH on a regular grid, mass-weighted; tissue partial volumes for ACT; continuity | export (post hoc); following built |
| Connectomes, SIFT2 / COMMIT | absolute fibre density per fixel; cross-section; true endpoints; tissue maps | **mass calibration (training)**; endpoint meaning (post hoc) |
| Fixel-based analysis (FD, FC, FDC) | fixel directions in RAS, fd = volume fraction, Bingham per lobe (scilpy), cross-section, fixel connectivity | export; **calibrated mass**; connectivity from following |
| Tractometry (AFQ, BUAN), bundle atlases | streamlines in RAS/MNI with values per vertex; profiles along each object | post hoc via following + TRX |
| Superficial white matter, U-fibres, radial/tangential cortex, gyral bias | orientation relative to the surface normal; laminar depth; separate U-fibre / radial / through-going populations at the grey/white boundary | angles post hoc; **generator coverage (training)** |
| Validating dMRI with microscopy | ODFs pooled over the dMRI voxel by volume, truncated to dMRI lmax in the same SH basis, a "predicted dMRI" view | export / derived |
| Forward-simulating dMRI (Monte Carlo substrates) | explicit objects (skeleton + diameter) or ODFs + volume fractions per population | derived from following + descriptors |
| Registration and templates | declared world frame; reorientation by the Jacobian (PPD); mass modulated by det J | export (BRIDGE native axes are not RAS) |
| Biomechanical / FE (brain injury, heart, arteries, muscle, IVD) | fibre field per element; second-moment tensor with two dispersion parameters (Bingham maps onto the generalised structure tensor of Holzapfel 2015); fibre families; volume fraction | export; **two-family structures (generator)** |
| Cardiac architecture | helix / transverse angle in the wall frame, sheetlets, two sheetlet populations | angles post hoc; **rotating laminae + sheetlets (generator)** |
| Arteries (GOH model) | two symmetric helical families ±α + κ | **helical shells (generator)**; ±α pair reporting post hoc |
| IVD annulus, cornea | alternating ±30° lamellae, orthogonal lamellae with branching | **alternating lamellae (generator)** |
| Tendon, ligament, optic nerve head | crimp period and angle, separate from dispersion | **crimp (generator)**; crimp metrics from following |
| Peripheral nerve (vagus mapping for stimulation) | fascicles followed through splits and merges (~every 560 µm) | **split-merge fascicle networks (generator)**; following built |
| Vessels, airways, kidney tubules, xylem | topology, tortuosity, branch angles, lumen, connections between vessels | descriptors and points built; **loops and connections (generator)**; network metrics post hoc |
| Tumour collagen (TACS-3 prognosis), fibrosis | alignment relative to a boundary normal; degree of order | **boundary-referenced arrangements (generator)**; angles post hoc |
| Bone, teeth, insect cuticle (helicoids) | lamella / helicoid pitch and handedness; tubule direction | twisted plywood built; pitch / handedness post hoc from the frame field |
| Muscle | fascicle length, pennation relative to aponeurosis, curvature | following + angles relative to parent / boundary post hoc; **fascicles ending on sheets (generator)** |
| Composites, textiles, paper | 2nd / 4th-order orientation tensors (Advani–Tucker), fibre length distribution, waviness, voids, flattened in-plane fibres | tensors from SH post hoc; voids exist (porous); **short fibres, woven yarns (generator, if in scope)** |

## 3. Triage

**Before the cluster run.** These need generator morphologies or training targets. All are generic arrangements, not tissue models:
1. **Crimp and waviness:** coherent zig-zag or helical crimp across neighbouring fibres, periods ~10–100 µm (scaled), angles 5–15°. Without it, crimp is learned as dispersion.
2. **Two-family and alternating arrangements:**
   - helical shells: tube walls with ±α fibre families, pitch and dispersion varying by layer;
   - alternating lamellae (±α);
   - orthogonal branching lamellae.

   Crossing mode and plywood cover only part of this.
3. **Rotating laminae with sheetlets:** fibre angle rotating through a thick wall, cleavage planes, two sheetlet populations.
4. **Networks that split and merge:** fascicle plexuses, capillary loops, connections between neighbouring tubes. Trees only branch today.
5. **Arrangements referenced to a boundary:**
   - fibres radial or tangential to a blob or surface;
   - superficial white matter sheets running parallel to the cortex;
   - U-fibres;
   - radial cortical fibres crossing tangential layers;
   - fanning gyral blades;
   - fascicles ending on sheets (aponeuroses).
6. **Targets:**
   - Absolute, calibrated volume fraction per entity: in progress with the mass head, and must pass its calibration checks.
   - Cross-section and diameter: the size descriptor; calibrate it.
   - Uncertainty kept separate from dispersion: the vote uncertainty exists; mass and existence uncertainty could be added.
   - Optional: a scale-consistency term, so an ODF pooled over a support equals the volume-weighted mixture.

**After training: exports and derived quantities, with no retraining:**
- MRtrix3 SH (lmax 4/6/8, sidecar with basis and normalisation; unit-integral and mass-weighted versions; pyramids built on SH, never on Bingham parameters).
- Fixel directory (directions in RAS, fd, κ1/κ2/minor axis, cross-section, population and hierarchy ids, fixel connectivity from following).
- Bingham per lobe in the scilpy convention.
- TRX streamlines with values per vertex (diameter, dispersion, uncertainty), object ids, endpoint meaning; zarr-vectors at whole-brain scale.
- 2nd/4th-order orientation tensors.
- Angles in an anatomical frame (helix, radiality to the surface normal, relative to a boundary normal, child relative to parent such as microfibril or pennation angle), ±α family pairs, helicoid pitch and handedness.
- Profiles along objects; network metrics (lengths, tortuosity, branch angles, split/merge events).
- A "predicted dMRI" view.
- Frame handling: native axes to RAS; Jacobian reorientation (PPD) and det J modulation for MNI.

**Evaluation to adopt.** These are the field's standard metrics:
- **Synthetic:**
  - success rate in counting populations per voxel (Daducci 2014);
  - angular error and recall/precision against crossing angle, volume fraction, SNR and resolution anisotropy;
  - smallest crossing angle resolved; false peaks;
  - ACC between ODFs; κ error.
- **Baselines:** a tuned structure tensor + clustering (Gray 2025), Frangi ODFs, a Kjer-style Bingham fit on structure tensors; CSD on dMRI-like data.
- **Real cross-modal checks, for evaluation only, never training:** Menzel 2023 SLI/SAXS/dMRI (Zenodo), BigMac, the DRCMR XNH axon segmentations, DiSCo, HR-Kidney, DTU unidirectional composites, the HAL human left ventricle at 3.5 µm.
- **Also:** uncertainty calibration; Tractometer / ISMRM-2015 bundles for tractography.

**Inputs outside our current scope:** orientation below the voxel that the intensity image does not show (polarised light, polarisation-resolved SHG, SAXS/WAXS and X-ray diffraction are tensor- or vector-valued measurements). The model takes scalar images; those modalities would need a different input.

## 4. Key references

**Methods**
- Nath 2019 MRI doi:10.1016/j.mri.2019.07.012
- Karimi 2021 NeuroImage (PMC8385546)
- FOD-Net bioRxiv 10.1101/2021.01.17.427042
- FOD-Swin-Net arXiv:2402.11775
- Elaldi arXiv:2411.11819
- Kebiri 2024 MedIA
- Consagra arXiv:2307.08138, 2405.13655
- DeepFixel arXiv:2511.03893
- PRISM arXiv:2604.00250
- Riffert 2014 doi:10.1016/j.neuroimage.2014.06.015
- Dhollander 2021 doi:10.1016/j.neuroimage.2021.118417
- TractSeg doi:10.1016/j.neuroimage.2018.07.070
- Schilling 2018 doi:10.1016/j.neuroimage.2017.10.046
- Gray 2025 doi:10.1016/j.jneumeth.2025.110539
- Axer 2016 doi:10.3389/fnana.2016.00040
- Alimi 2020 doi:10.1016/j.media.2020.101760
- Howard 2023 (BigMac) doi:10.1038/s41467-023-39916-1
- Menzel 2023 doi:10.7554/eLife.84024
- ComSLI doi:10.1038/s41467-025-64896-9
- Oberstrass 2024 doi:10.1162/imag_a_00351
- Goubran 2019 doi:10.1038/s41467-019-13374-0
- Leuze 2021 (PMC7953593)
- TrailMap doi:10.1073/pnas.1918465117
- Liebi 2015 Nature doi:10.1038/nature16056
- Georgiadis 2021 doi:10.1038/s41467-021-22719-7
- Andersson 2020 doi:10.1073/pnas.2012533117
- Kjer 2025 doi:10.7554/eLife.94917
- SynthSeg doi:10.1016/j.media.2023.102789
- anatomix arXiv:2411.02372
- vesselFM arXiv:2411.17386
- Bleiziffer 2021 doi:10.1016/j.engappai.2021.104351
- Krause 2010 doi:10.1007/s10853-009-4016-4
- Daducci 2014 doi:10.1109/TMI.2013.2285500
- Maier-Hein 2017 doi:10.1038/s41467-017-01285-x

**Applications**
- Streeter 1969 doi:10.1161/01.res.24.3.339
- LeGrice 1995 doi:10.1152/ajpheart.1995.269.2.h571
- Wang 2024 doi:10.1038/s41598-024-65371-z
- Nielles-Vallespin 2017 doi:10.1016/j.jacc.2016.11.051
- Brunet 2024 doi:10.1148/radiol.232731
- Damon 2017 doi:10.1002/nbm.3563
- Kupczik 2015 doi:10.1016/j.jtbi.2015.06.034
- Spiesz 2018 doi:10.1016/j.actbio.2018.01.034
- Meek 2005 doi:10.1167/iovs.04-1253
- Winkler 2011 doi:10.1167/iovs.11-8070
- Holzapfel 2005 doi:10.1007/s10237-004-0053-8
- Disney 2017 doi:10.1038/s41598-017-16354-w
- Gasser 2006 doi:10.1098/rsif.2005.0073
- Schriefl 2012 doi:10.1098/rsif.2011.0727
- Holzapfel 2015 doi:10.1098/rsif.2015.0188
- Ní Annaidh 2012 doi:10.1016/j.jmbbm.2011.08.016
- Raya 2012 doi:10.1148/radiol.11110821
- Georgiadis 2015 doi:10.1016/j.bone.2014.10.002
- Provenzano 2006 doi:10.1186/1741-7015-4-38
- Conklin 2011 doi:10.1016/j.ajpath.2010.11.076
- Matsuzaki 2023 doi:10.1038/s41598-023-48887-8
- Upadhye 2022 doi:10.1088/1741-2552/ac9643
- Thompson 2023 doi:10.3389/fnins.2023.963503
- Fratini 2015 doi:10.1038/srep08514
- Jan 2017 doi:10.1167/iovs.17-21811
- Kuo 2023 doi:10.1038/s41597-023-02407-5
- Walsh 2021 doi:10.1038/s41592-021-01317-x
- Kirst 2020 doi:10.1016/j.cell.2020.01.028
- Bullitt 2005 doi:10.1016/j.acra.2005.05.027
- Barnett & Bonham 2004 doi:10.1017/S1464793103006377
- Brodersen 2011 doi:10.1111/j.1469-8137.2011.03754.x
- Weaver 2012 doi:10.1126/science.1218764
- Jud 2016 doi:10.1016/j.dental.2016.06.021
- Advani & Tucker 1987 doi:10.1122/1.549945
- Emerson 2018 doi:10.1016/j.compscitech.2018.08.028

**Downstream analyses**
- Tournier 2007 doi:10.1016/j.neuroimage.2007.02.016
- MRtrix3 2019
- Smith 2012 (ACT) doi:10.1016/j.neuroimage.2012.06.005
- Bastiani 2017 doi:10.1016/j.neuroimage.2017.06.050
- SIFT doi:10.1016/j.neuroimage.2012.11.049
- SIFT2 doi:10.1016/j.neuroimage.2015.06.092
- COMMIT doi:10.1109/TMI.2014.2352414
- Raffelt 2012 AFD (PubMed 22036682)
- CFE doi:10.1016/j.neuroimage.2015.05.039
- FBA doi:10.1016/j.neuroimage.2016.09.029
- Raffelt FOD registration doi:10.1002/mrm.23058
- Yeatman 2012 doi:10.1371/journal.pone.0049790
- Chandio 2020 (PMC7555507)
- Garyfallidis 2018 doi:10.1016/j.neuroimage.2017.07.015
- Warrington 2020 doi:10.1016/j.neuroimage.2020.116923
- Reveley 2015 doi:10.1073/pnas.1418198112
- Schilling 2018 gyral bias doi:10.1002/hbm.23936
- Cottaar 2021 doi:10.1016/j.neuroimage.2020.117693
- McNab 2013 (PMC3557597)
- Schilling 2016 doi:10.1016/j.neuroimage.2016.01.022
- Mollink 2017 doi:10.1016/j.neuroimage.2017.06.001
- Fiberfox doi:10.1002/mrm.25045
- ConFiG doi:10.1016/j.neuroimage.2020.117107
- Alexander 2001 doi:10.1109/42.963816
- Giordano & Kleiven 2014 doi:10.4271/2014-22-0002 ⚑
- Bayer 2012 doi:10.1007/s10439-012-0593-5
- Blemker & Delp 2005 doi:10.1007/s10439-005-1433-7
- Behrens 2007
- Jeurissen 2011 doi:10.1002/hbm.21032
- Sotiropoulos 2012 doi:10.1016/j.neuroimage.2012.01.056
- MRtrix SH / fixel formats (mrtrix.readthedocs.io)
- DIPY SH bases
- TRX specification ⚑
- BIDS BEP016 ⚑ (draft)
- OME-NGFF doi:10.1038/s41592-021-01326-w
- zarr-vectors specification
- scilpy Bingham tools
