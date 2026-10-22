# Overnight build, 2026-10-21/22 (22:40 → 08:00 Montreal)

The brief: review the theory and literature for anything not yet tried; build the clean flexible-N pipeline; meet the
acceptance criteria and the speed benchmark; measure tractography on a HiP-CT VOI; rewrite the docs. Progress is logged
at the bottom, newest last.

## 1. Literature and theory review

What was searched: multi-orientation estimation in images (not diffusion MRI), position–orientation liftings, higher-
order tensors, autocorrelation and Fourier methods, tensor voting, fibre orientation from micro-CT in materials, and the
2024–2026 microscopy-tractography preprints.

### Found, not previously considered

1. **Double-orientation (mixed-orientation-parameter) tensors** — Aach, Mota, Stuke, Mühlich, Barth, *Analysis of
   Superimposed Oriented Patterns*, IEEE TIP 15(12) 2006 ([doi](https://doi.org/10.1109/tip.2006.884921); conference
   version [Stuke et al. 2004](https://www.inb.uni-luebeck.de/fileadmin/files/publications/inb-publications/pdfs/StAaBaMo04.pdf);
   Mota et al., *Estimation of multiple orientations in multi-dimensional signals*). Two superimposed oriented
   patterns `f = g1 + g2`, `g1` constant along `d1`, `g2` along `d2`, satisfy the second-order constraint
   `d1ᵀ (∇∇ᵀ f) d2 = 0` exactly. A 6 × 6 "second-order structure tensor" of Hessian products gives, by its least
   eigenvector, the symmetric matrix `sym(d1 d2ᵀ)`, whose eigen-decomposition returns both directions in closed form;
   the eigenvalue hierarchy of the first- and second-order tensors decides N ∈ {0, 1, 2}, and the construction extends
   to N orientations with N-th derivatives in any dimension. It also covers occluding (non-additive) superposition. The
   constraint holds for interleaved fascicles too (each voxel satisfies one of `H d1 = 0`, `H d2 = 0`). Cost: a few
   Gaussian derivative filters and a batched 6 × 6 eigensolution — a few times the structure tensor. *Candidate for the
   fast solve; untested on dispersed textures.*
2. **Orientation scores and crossing-preserving diffusion on SE(3)** — Duits, Janssen, Hannink, Sanguinetti, JMIV 2016
   ([arXiv 1502.08002](https://rduits.win.tue.nl/1502.08002v3.pdf)); Janssen et al., *Design and processing of
   invertible orientation scores of 3D images*, JMIV 2018 ([arXiv 1707.02191](https://arxiv.org/abs/1707.02191));
   Bekkers & Duits, *Orientation scores should be a piece of cake*, 2025 ([arXiv 2504.00702](https://arxiv.org/pdf/2504.00702)).
   The image is lifted to R³ × S² with cake wavelets and smoothed by coherence-enhancing diffusion *along each
   orientation* — the principled form of this pipeline's "momentum". Our momentum is the morphological analogue (a
   directional closing: bridge where both sides hold, never extend a free end), which the SE(3) diffusion does not
   guarantee. Keep the morphological rule; cite the framework.
3. **3D Fourier fibre-family method** — *Fourier transform-based method for quantifying the three-dimensional
   orientation distribution of fibrous units*, 2024 ([PMC11222475](https://pmc.ncbi.nlm.nih.gov/articles/PMC11222475/)):
   spectral magnitude summed in cones per direction on a Fourier shell, deconvolved by the cones' overlap matrix,
   fitted as a mixture of bivariate von Mises families. The Fourier dual of the lag ODF (a population's power lies in
   the plane ⊥ to it; its autocorrelation on the line along it). Tile-level only (256³ tiles), CPU, 60 s per 1024 × 1024
   × 256; no windowed field. Confirms the deconvolution step and the von Mises family description of dispersion.
4. **HiP-CT fODF tractography preprint** — *Bridging the microstructural gap in human connectomics using HiP-CT as a
   reference for diffusion MRI*, bioRxiv 2026.04 ([link](https://www.biorxiv.org/content/10.64898/2026.04.02.715729v1)):
   structure tensor (σ 2, ρ 4) per voxel, eigenvectors pooled per 800 µm supervoxel into a spherical histogram fitted
   with SH (lmax 8), peaks = crossings, DIPY probabilistic tracking, vessels masked at the gradient level with a 2.5D
   U-Net, CPU/Dask out of core. Crossings come only from pooling single-direction voxels; nothing per voxel.
5. **Tensor voting** (Medioni; GPU versions, gap filling of vessel segments) — the stick vote is a straight/circular
   continuation with distance-and-curvature decay: the same role as momentum, without the both-sides rule.
6. Older multi-orientation literature: IRON (isotropic recursive oriented network), k-medoids on the Fourier
   transform, autocorrelation fitting for texture orientation; **hybrid MRI-microscopy tractography** (Huszar et al.,
   Med. Image Anal. 2025); tensor-augmented light-sheet scattering microscopy (bioRxiv 2026.05) with a von Mises mixture
   for crossings at 2.5 µm.

### Where this leaves the design

- The lag ODF is the right lifting for a general array: it is the local autocorrelation read on a shell, the structure
  tensor is its zero-lag second moment, N is free, and nothing is trained. The population-specific carry-forward is a
  directional closing on that lifting; identities are a constrained graph join. All three are classical.
- Not tried before tonight and worth a prototype as the **fast solve**: the double-orientation tensor (closed form, N ≤ 2).
- The learned embedding (anatomix) is not needed by the chain; it stays an optional weighting engine.

## 2. Plan

1. Speed: lift on the lattice (products pooled to cells before the window), cached sampling grids, batched momentum,
   null floor once per volume region. Target ≤ 8 s per 256³ for the whole chain on an A6000.
2. Native-resolution field: lobe tensors interpolated to every voxel; weighted single field (presence × coherence),
   multi-vector field with identities; one-call API `orientations()` / `shapetensor()`.
3. Acceptance suite with the tracking metric (through-crossing rate, bend follow-through, T-junction false extension),
   N / direction / identity metrics, the structure tensor beside; new phantoms: kissing, free end.
4. Chunked volume driver with fODF blending across overlaps and label stitching.
5. HiP-CT VOI: field + tractography against the structure tensor.
6. Docs, README, results log, commit.

## 3. Progress log

- 22:40–23:30 Montreal: literature review (above); plan.
- 23:30–00:15: `lift.py` — integer-lag shell, lattice pooling, direction kernel: 256³ in 2.1 s (was 10 s), sharper raw
  lobes on the crossing (both within 15°: 0.66 vs 0.29 raw).
- 00:15–01:50: `populations.py`, `field.py` (chain + native field); iterated on the two failure modes (weight dropouts in
  the 90° intermixed crossing; y-lobe false seconds in fascicles). Tried and dropped: valley test, lobe-vs-ring ratio on
  the raw lift, ring atoms in the dictionary, lobe-shape test. Kept: N-independent presence (excess mass), high-pass
  0.3 scale, excision inside the correlation. Lattice numbers: bend N1 0.86, straight 0.97, band 90/60 both-15 0.77/0.94.
- 01:50–02:30: acceptance suite (`validate.py`, tracking metric), chunked driver (`volume.py`, exact seams after rounding
  the halo to the lattice step), label stitching, unit tests (7 CPU tests), CLI.
- 02:30–03:40: fixed the tracking scorer (seeds on the stub; through = beyond the crossing; thick grids), acceptance run
  3; VOI run launched (768³ cortex, 384³ chunks, 70–105 s each shared GPU); docs rewritten (index, pipeline/lift,
  populations, field, parameters, api, quickstart, installation, validation), README, CHANGELOG, results log §35,
  rules.md status.
- Open at 03:45: VOI tractography numbers and figure → docs/data.md; a quick test of angular smoothing in the
  carry-forward for the tight bend; the speed bench on an idle GPU; commit.
