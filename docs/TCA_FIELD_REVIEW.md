# TCA on shape features as a multivector orientation field for BRIDGE: review and recommendation

Status 2026-10-16. A review of the concept and the prototypes in this repo, with a recommended design for the field,
its validation, and the package that computes it. Written for approval before any implementation plan.

Decisions already taken by the user (2026-10-16): N populations enter BRIDGE as **interleaved vertices** (one direction
each, vertex-stage change only); real-tissue truth is **phantoms only**; the method lives in a **separate package**
with a reader in BRIDGE; no clustering and no labels as a method input; a **no-network control arm**; validation at
**publication grade** (pre-registered thresholds, several seeds, confidence intervals).

## 0. Verdict in one page

**The concept is sound and the evidence for the single-population case is strong enough to build on.** On the
tract-interface phantom D15, the 48-channel anatomix layer with CP decomposition (TCA, rank 3) gives a median error of
**7° within 0.5 mm of the interface**, where the intensity structure tensor gives 42–62°. It needs no labels, no
no-step twin and no oracle. That is the result this whole line of work was looking for.

Three things in the current evidence should temper the claims, and the design below is built around them:

1. **Most of the gain is the basis, not the decomposition.** At the D15 interface the ordinary eigen read-out of the
   48-D shape tensor already gives 11°; TCA takes it to 7°. The 16-D output layer (64°) shows the basis is what
   matters. TCA's distinctive contributions are (a) *naming* the step component, which is what makes boundary
   extraction and the hybrid possible, and (b) non-orthogonal directions, which is what makes crossings possible.
   Claims should be phrased that way.
2. **Shape space has a floor of 9–12° in clear tissue, where the intensity tensor sits at 2°.** The field therefore
   has to be a **hybrid**: the intensity tensor wherever no interface is detected, shape-space TCA where one is. The
   switch is designed in `IMAGE_TO_VECTORS.md` §6 and has not been run. It is the single most important untested piece.
3. **N-population selection is not yet calibrated.** BIC chooses 3 populations on a clean single rod 100% of the time;
   the ad-hoc floor-and-merge rules rescue the easy cases (rods at 90°+30°: 99.8% correct) but fail at 60°+120°
   (41%, 58% say "3") and at realistic noise (17%). Nothing shipped should rely on BIC.

**Recommended field.** Per site: up to 3 axial directions with fractions, a boundary normal with a strength, a quality
score, and a provenance flag (intensity / shape / blended). The single-direction path is the hybrid above. The
N-population path takes *N and its coarse directions from a constrained spherical deconvolution (CSD) of the
gradient-direction distribution*, which is rank-free by construction and has twenty years of calibration literature,
then *refines each direction with TCA at rank 2N* initialised from those peaks, and *accepts a population only if it is
stable* under the free resamples the method already produces (the grid phases) and a spatial bootstrap. The TCA-native
alternative (over-complete CP, geometric grouping of pieces) is kept as a comparison arm, not the default.

**Recommended package.** `shapetensor`: torch + anatomix in its own container; tiles with discarded margins on the
GPU; a channel-resolved tensor `K` on a coarse grid (the ρ-smoothing makes finer sampling pointless); batched CP / CSD
read-outs; an OME-Zarr orientation-field store. BRIDGE gains one config block, `vertices.orientation_source`, and a
`population` vertex attribute. Measured today on the A6000: anatomix layer 58 runs a 384³ tile in **1.46 s** (bf16,
18.8 GiB), and its receptive field at layer 58 has a half-width of **8 voxels at 1% of peak, 27 at 0.1%**. A 768³ 20 µm
crop is minutes; a whole 20 µm hemisphere is hours on one card and under an hour on an 8-H100 node.

**Validity.** A phantom suite at the target voxel size (20 µm), each phantom with a matched no-step or single-population
twin, scored by distance-to-interface band with CIs over ≥ 5 seeds, against pre-registered thresholds (§7). Three
controls run through the identical read-out: the intensity tensor at matched ρ, the gradient-excision fix A, and a
**no-network basis** (a locally normalised multiscale Gaussian-derivative jet) — the last decides whether the deep
network is needed at all. End-to-end: BRIDGE builds on the phantoms with the field, scored with
`bridge.testing.groundtruth.score_streamlines` and the phantom-shell "A behaves like B" criterion.

## 1. What the concept is, precisely

Per output site, the method looks at the gradients of *every feature channel* in a window:

```
X[p, c, i] = sqrt(G_rho(p)) * d f_c / d x_i (p)        p: voxel in the window, c: channel, i: axis
```

Summed over `p` and `c` this is the familiar 3×3 shape tensor `T`, whose smallest eigenvector is the fibre. TCA keeps
the array unsummed and fits a CP (PARAFAC) model

```
X ~ sum_r  a_r (space) o b_r (channel) o d_r (axis),     R components, d_r unit, not orthogonal
```

Exactly and much smaller: `CP(X) = CP(L^T reshaped (3C, C, 3))` where `K = G_rho * (g g^T)` is the channel-resolved
`3C × 3C` tensor (`g = (channel, axis)`) and `L` its Cholesky factor. So the per-site input is `K`, 48×48 with 16
channels, and the whole method is a function of `K`.

Read-outs:
- **One fibre:** the direction the pieces do not use — the null direction of `sum_r lam_r^2 d_r d_r^T`.
- **N fibres:** two pieces per population (a population's gradients span the plane ⊥ to it); pieces paired by
  geometry (the matching whose pairs are most perpendicular); fibre `= d_r × d_s`; fraction = the pair's share of the
  fitted energy.
- **A step** (interface, grey/white boundary) is one piece: a sheet-like spatial map, one channel signature, one
  direction — its normal. This is what the intensity tensor cannot see: with one channel `X` is a matrix, CP of a
  matrix is not unique (any rotation of the factors fits equally), and the only identifiable read-out is the SVD, i.e.
  the eigendecomposition. With `C ≥ 2` channels CP becomes identifiable (Kruskal's condition `k_A + k_B + k_D ≥ 2R + 2`),
  and non-orthogonal directions, among them a step's normal that is not at right angles to the fibres, can be named.

**Why it helps at an interface.** In the image, a step of 15 grey levels contributes gradient energy that dominates a
0.3 mm window; the tensor goes rank-1 (λ1 ≫ λ2 ≈ λ3) and e3 is an arbitrary direction *in* the interface plane. That
is the mechanism of the boundary shell in `BOUNDARY_PROBLEM.md` (88–93% of e3 in-plane within 0.5 mm, shell thickness
∝ ρ). In the 48-D layer the step is "one small change among many texture channels": the network has learnt
contrast-robust texture descriptors, so the summed shape tensor is texture-dominated (11° at D15), and CP can name the
step's piece and leave it out (7°).

### The evidence as it stands (phantoms, 0.1 mm voxels; from `IMAGE_TO_VECTORS.md` and the result JSONs)

| case | read-out | result | status |
|---|---|---|---|
| D15 interface, within 0.5 mm | intensity ST ρ3 / ρ16 | 62° / 42° | established |
| | 16-D shape tensor, eigen | 64° | established (fails) |
| | 48-D shape tensor, eigen | 11° | established |
| | **48-D TCA R3, ρ16** | **7°** (bands 3 / 8 / 7°) | established, one phantom family |
| D15, 3–8 mm away; D0 | intensity ST ρ16 | 2° | established |
| | 48-D TCA | 9–10° | established (the floor) |
| rods 90°+30° (clean) | TCA 2 pieces/pop, geometric pairing | 4° / 4°, 100% both within 15° | established |
| rods 60°+120°, noise 2 / 8 | same | 6/13° (70%); 5/11° (91%) | partial |
| realistic lines 60°+120°, noise 8 | same | 19° / 13° at 29% of sites | **not working** |
| realistic lines, single 30° | intensity ST / TCA (16-D) | 1.5° / 15° | shape space loses to ST |
| N selection, single rod | BIC alone | picks 3 at 100% | **miscalibrated** |
| N selection, 60°+120° | BIC + floor + merge | 41% correct | not working |
| N by FRT-ODF peaks (intensity gradients) | rods | 100% correct, 2–3° | works on rods, merges lines |
| hybrid ST / TCA switch | — | not run | **untested** |
| anything on real tissue | — | not run | untested |

## 2. Where the structure tensor fails, and what the phantoms must show

The field is only worth building if it fixes measured failures. Each failure below is a phantom in §7 and a figure in
the "where ST fails" set.

| failure | mechanism (gradient tensor, λ1 ≥ λ2 ≥ λ3, fibre = e3) | symptom in BRIDGE | phantom |
|---|---|---|---|
| **Interface / grey-white step** | step energy makes T rank-1; e3 arbitrary in the plane; shell thickness ∝ ρ | boundary wrapping: 27% of streamline length in surface-parallel runs vs 8% in HCP; walks turn onto the sheet | A vs B (gyrus), D8/15/30 (tract interface), C vs C0 (U-fibres that *must* be kept) |
| **Crossings and fans** | two populations' gradient planes sum; e3 is the bisector or the normal to both | low (λ2−λ3)/λ2 at every ρ at gyral-blade junctions (F5); bisector streamlines | rods / lines at 90, 60, 45, 30°, fractions 1:1, 2:1, 4:1; a fan |
| **Scale trade-off** | small ρ: no support in homogeneous WM (median \|z\| 0.50 = random); large ρ: thicker shell, blurred crossings | F1/F2: coverage vs orientation quality; 39.5 / 17.7 / 14.1° at ρ 3 / 9 / 24 | D-series at ρ ∈ {4…80} vox; a homogeneous-WM phantom with matched noise |
| **Vessels / halos** | a dark tube is a strong cylindrical edge; FA islands are vessel halos (F1) | streamlines along vessels (vessel masks exist in post-processing only) | dark tubes crossing a fibre field at random angles |
| **Laminar bundles** | λ3 ≫ λ2 ≫ λ1 with a well-determined e3 but a "sheet" shape class | the Westin shape gate rejects the callosum | a slab with internal fibre texture |

What the figure set must contain for each row: error vs distance-to-structure curves (ST at ρ 3 / 6 / 16 / 80 vs the
field), a map of the failure region, and the same for the hybrid, with the twin phantom as the control. Every claim
"ST fails here" is then a measured difference between a phantom and its twin, not an interpretation.

## 3. Critical review of the prototype

**Strengths.**
- The single-population D15 result is clean, oracle-free and reproduced across windows and ρ.
- The method runs on `K`, which is a smooth field (it is ρ-smoothed), so it can be computed on a coarse grid; the
  Cholesky trick makes the per-site problem 48 × 16 × 3 — small enough to batch millions of sites on a GPU.
- The clean-anatomix recipe (one pass, reflect padding, grid-phase averaging) is correct and well documented; the
  face-effect fix (grown phantoms) was the right call.
- The phantom families (A/B, C/C0, D-series, rods, lines) already cover the failure taxonomy of §2.

**Weaknesses and what to do about them.**

1. *The clear-tissue floor (9–12°).* The network blurs about 7–8 voxels (measured today: receptive-field half-width
   8 vox at 1% of peak) and has direction preferences (grid-aligned rods give 2.2× the gradient energy of oblique
   ones; thick rods drown thin ones). → The hybrid is mandatory. Also test the floor's origin: run the no-network
   basis (§7, arm C3) through the same TCA; if its floor is 2° the floor is the network's, not the method's.
2. *Brightness sensitivity of the 48-D layer.* A regional offset changes its features everywhere. → Per-tile
   intensity standardisation before the network is not enough (the step is regional). Test invariance explicitly:
   D15 with the whole image offset by ±20 and rescaled by ×0.8 / ×1.25; the orientation must not move by more than the
   seed-to-seed spread. Add the 16-D output's own boundary strength `|df/dx|` as a cheap, contrast-trained signal.
3. *ρ is in voxels and the phantoms are at 0.1 mm.* The 7° result needs ρ 16 (1.6 mm); at ρ 6 it is 50°. At 20 µm the
   network's blur is 0.16 mm, the step's reach in feature space scales with it, and the fibre texture is at a
   different voxel scale. → All phantoms are rebuilt at **20 µm voxels** with structure sizes and noise matched to the
   real crops (§7), and ρ is swept there. Nothing from the 0.1 mm phantoms transfers numerically.
4. *CP degeneracy.* ALS on an over-ranked or ill-conditioned array produces "swamps": diverging, nearly collinear
   components whose directions are meaningless. The prototype's "nothing dropped" read-out at rank 3 is safe (rank 3
   in a 3-axis mode is at most the eigen solution's complexity), but the 2N-piece crossings fit at N = 3 is exposed.
   → Report the CP fit *and* the condition of the pairing (the minimum pairwise sine, already computed) as quality
   fields; reject fits whose pieces are within 10° of collinear; bound the rank by the CSD count (§4).
5. *N selection by BIC.* `k = 2N(P + C + I − 2)` parameters against `n = P·C·I` entries assumes independent Gaussian
   residuals; the residuals of a Cholesky-reshaped Gram are neither. → Do not use BIC. Use CSD peaks for N and
   stability for acceptance (§4).
6. *The geometric pairing is exhaustive over matchings:* 15 matchings at rank 6, 105 at rank 8. Fine on a GPU at
   N ≤ 3; do not go beyond.
7. *The grid-phase average costs 8×.* The network's period-2 checkerboard (1.5–3%) must go, but a [1 2 1]/4 filter
   along each axis zeroes Nyquist exactly at the cost of a 0.7-voxel blur, negligible against the network's 8. →
   Test equivalence (D15 score within the seed spread) and, if it holds, save 8× on the dominant cost.
8. *Realistic texture crossings (lines, noise 8) do not work (29%).* This is the result that matters for real
   tissue; rods are a sanity check. → The CSD scaffold is the proposed fix; it is also the one that failed least on
   lines (FRT merges them; CSD is the standard remedy for exactly that).
9. *Nothing has run on real tissue.* → The 20 µm S-21-46 crop (internal capsule between thalamus and pallidum) is the
   real-data demonstration: glyphs, boundary-strength maps, and BRIDGE builds compared by label-free consistency
   metrics. No numeric truth claim is made there (user decision).
10. *Memory and traps in the prototype:* float16 feature caches band the minor PCs; `pkill -f` self-matching; chunkreg's
    silent 0.25-overlap fallback. → The package stores features as float16 only after the PCA projection (the top 16
    PCs have the range), and never tiles through chunkreg.

## 4. N populations without a fixed k

The requirement is a per-site count that is not chosen in advance and does not depend on a miscalibrated criterion.

| approach | how N emerges | status here | assessment |
|---|---|---|---|
| **CP + BIC** | rank with minimum BIC | fails (always 3) | drop |
| **CP + floor/merge rules** | fraction ≥ 10%, merge < 20° | rods only | keep the rules as post-filters, not as the selector |
| **Over-complete CP + geometric grouping** | fit rank R_max (6–8); group pieces into coplanar sets; each plane's normal is a candidate fibre; accept by energy and stability | not tried | TCA-native and rank-free in effect, but exposed to CP swamps at over-rank; comparison arm |
| **Block-term decomposition (L,L,1)** | one signature per population, rank-2 block | tried (`shape_pop12` method 1): N found 0 on crossings | needs the rules fixed; low priority |
| **FRT / q-ball ODF peaks** | peaks of the Funk–Radon transform of the gradient-direction distribution | 100% on rods; merges lines | rank-free; angular resolution limited by the FRT blur |
| **CSD of the gradient-direction ODF** | deconvolve with the single-population response (a great-circle ring); peaks of the non-negative fODF | not tried | rank-free, standard, resolves what FRT merges; the recommended scaffold |
| **Bayesian CP with ARD** | automatic pruning of components | not tried | principled but slow and with its own priors to calibrate; not for v1 |
| **Stability selection** | keep a population only if reproduced under resampling | not tried | the acceptance test, on top of any of the above |

**Recommendation: CSD for N, TCA for the directions, stability for acceptance.**

1. *Signal.* Per site, the distribution of gradient directions over the window and the channels, weighted by `|g|²`
   (the prototype's `odf_coefs`), in even real spherical harmonics. The signal is high *perpendicular* to each fibre
   (a ring) exactly as the diffusion signal is high perpendicular to a fibre in dMRI, so the dMRI machinery transfers.
   With thousands of gradient samples per window the SNR allows `l_max` 12–16, so 30–40° crossings are in reach.
2. *Step excision before deconvolution.* A step contributes a *delta* (one direction), not a ring. Deconvolved with
   a ring response it becomes a smeared ring in the fODF — no sharp peak, so a mild step is harmless. A strong step
   still swamps weak populations, so its share is removed first: TCA at rank 3 names the step piece (one direction,
   sheet-like spatial map, high concentration), and its gradient share is subtracted from the ODF (as `readouts()`
   already subtracts a dropped component's Gram from the exact tensor). This is also the boundary extraction of §5.
3. *Deconvolution.* The response is the ring of one population, measured on the single-population phantoms at the
   chosen ρ (as dMRI estimates its response from high-FA voxels), expressed in rotational harmonics; constrained
   non-negative least squares (Tournier 2007) gives the fODF; N = the number of peaks above a null-calibrated
   fraction of the maximum, at least 25° apart, capped at 3.
4. *Refinement.* With N fixed per site, CP at rank 2N, initialised from the peaks (two directions in each peak's
   perpendicular plane), one ALS run — no random starts and no rank search. Fibre `= d_r × d_s`, fraction from the
   pair's energy share. The pairing is given by the initialisation; the "most perpendicular matching" is only checked.
5. *Acceptance.* A population is kept if its direction is reproduced within 10° and its fraction within 0.1 across
   (a) the grid-phase resamples (free: 8 feature sets per site, or 4 with the [1 2 1] filter and 2 phases) and (b) a
   spatial bootstrap of the window (3 draws). Null-calibrated: the false-population rate on the single-population and
   homogeneous phantoms sets the thresholds (target ≤ 2%).

The over-complete-CP grouping runs in the validation as a comparison arm, because it would keep the pipeline
TCA-only; if it matches the scaffold on the lines phantoms it can replace CSD later.

## 5. Boundary extraction and the hybrid

The gradient-direction distribution in a window is a sum of *rings* (fibres) and *deltas* (interfaces). TCA separates
them by their spatial and channel modes as well: a step is a sheet × one signature × one direction. The boundary field
is therefore a by-product of the same decomposition:

- **boundary normal** `n`: the direction of the unpaired, sheet-like piece (R3 fit; the piece with the largest
  concentration `|d_r · g|` share and a spatial map whose own structure tensor is planar);
- **boundary strength** `s`: that piece's share of the window's gradient energy, in [0, 1];
- two cheap corroborating signals: the 16-D output's `|df/dx|` (what the network was trained to see) and the coarse
  intensity-gradient magnitude (fix A's detector).

All three are label-free. Calibration is null-based, as the eigen-gap gate was: the 99th percentile of `s` on D0 and
on the homogeneous phantom is the "no interface" level; D8 / D15 / D30 give the detection curve. Scored as precision /
recall of "within 0.5 mm of an interface" against the phantom geometry.

**The hybrid (the single-direction field).** Per site, with `w = sigmoid((s − s0) / Δs)`:

- `w ≈ 0`: the intensity structure tensor's e3, at BRIDGE's own σ / ρ (computed in the package with the same
  parameters and a parity test against `bridge.stages._vertex_fields._structure_tensor`);
- `w ≈ 1`: the 48-D TCA direction;
- in between: the null direction of the trace-normalised blend `(1 − w) T_I / tr T_I + w G_shape / tr G_shape`
  (axial vectors cannot be averaged; their Grams can).

**An alternative to test alongside: TCA-excised intensity tensor.** Fix A removed step gradients by magnitude and, on
the real 100 µm brain, ended walks at every surface. The TCA step piece is more selective: deflate `T_I` along `n`
only (`T_I − α n n^T`, α the excess of `n^T T_I n` over the fibre-plane level) and take e3. This keeps the 2° floor
automatically where there is no step and needs no blend. It is arm H2 in §7; whichever of H1 (blend) and H2 scores
better on A/B, C/C0 and the D-series becomes the default.

**What the boundary field is for beyond the switch.** BRIDGE's candidate fix B (`BOUNDARY_PROBLEM.md` §6) wanted a
per-vertex boundary attribute. The reader stores `n` and `s` as vertex attributes. No graph rule uses them in this
work (no walk change), but they are then available for a graph-native rule later, measured on the same phantoms.

## 6. The recommended field and how BRIDGE consumes it

**The store** (`<acq>_desc-shapefield.ome.zarr`, co-registered, on the coarse grid `b × voxel`, `b` = 8 at 20 µm):

| array | shape | dtype | content |
|---|---|---|---|
| `dirs` | (3, 3, Z, Y, X) | float16 | up to 3 axial unit directions, sorted by fraction, canonical sign |
| `frac` | (3, Z, Y, X) | float16 | fractions, sum 1 over the populations present |
| `npop` | (Z, Y, X) | uint8 | 0–3 |
| `bnorm`, `bstrength` | (3, …), (…) | float16 | boundary normal and strength `s` |
| `quality` | (Z, Y, X) | float16 | the single-direction reliability: `(λ2 − λ3)/λ2` of the blended Gram (F4's measure) |
| `source` | (Z, Y, X) | uint8 | 0 intensity, 1 shape, 2 blend |

plus `K` itself as an optional intermediate store (1176 channels, float32), so read-outs can be rerun without the
network. The grid is coarse because `K` is ρ-smoothed: at ρ = 1.6 mm a 0.16 mm grid oversamples it tenfold; vertices
sample it by trilinear interpolation of the Gram (then e3) or nearest cell.

**BRIDGE changes (the reader).**

1. `VertexParams.orientation_source: {"kind": "structure_tensor" | "field", "path": …, "populations":
   "primary" | "interleave", "min_fraction": 0.15}`. Default unchanged.
2. Vertex stage: the tensor pass still runs (it supplies `fa` and the intensity direction the hybrid falls back on
   only inside the package, so for BRIDGE it supplies `fa` alone); after `draw_sample`, `gather_sampled_vertices`
   takes `vec` from the field instead of `fields["vec"]`.
3. **Interleaving.** Each population `k` with `frac_k ≥ min_fraction` is sampled by its *own* stratified stream
   (`chunk_rng(chunk_id, seed, k)`) at the configured density, so crossing regions hold both populations at full
   density and no two vertices coincide (two vertices in one 6-voxel cell land on different voxels). Each vertex
   carries `population` (int8) and `fraction` (float16) attributes; `principal_eigenvector` is its own direction.
   Nothing downstream reads the new columns: the alignment gate, `edge_sides`, the lateral side-rank and the walk are
   untouched, which is the "graph-native, no new walk paradigm" condition.
4. **Risk to test, not assume:** the Gabriel filter sees the other population's vertices as ordinary points. A vertex
   of population 2 offset from a population-1 vertex *towards* its neighbour lies inside that edge's Gabriel sphere
   and blocks it. On a 50/50 crossing this halves the candidate edges per population. The end-to-end crossing phantom
   (§7, with `bridge.testing.fibres.tube_cloud` and `score_streamlines`) measures it. If it bites, the fix is a
   same-site exclusion in the candidate-edge table (an edge between vertices of different populations within one
   field cell is dropped before Gabriel) — a graph-stage filter, not a walk change, to be asked for separately.
5. `bnorm`, `bstrength`, `quality`, `source` are stored as vertex attributes for later rules and for QC renders.

## 7. Validation plan (publication grade)

**Principles.** Every phantom has a *twin* that differs only in the confound (no step, one population, no vessel).
Everything is scored on the twin pair, by distance band, over ≥ 5 seeds, with bootstrap 95% CIs on the medians and
the 90th percentiles. Thresholds are stated here, before the runs. The read-outs all run on the same `K`, so a
comparison is never confounded by preprocessing.

**Phantom suite at 20 µm voxels** (rebuilt; the 0.1 mm versions stay for continuity with the old numbers):

| id | geometry | twin | tests |
|---|---|---|---|
| P1 gyrus | two blades, GM layer, fan crossing the boundary, step like real GW contrast | no-step | shell, wrapping, the hybrid |
| P2 U-fibres | P1 with a superficial boundary-parallel band | no-step | the fix must *keep* true parallel fibres |
| P3 interface | one WM slab, steps 0 / 8 / 15 / 30, fibres at 90 / 60 / 30° | step 0 | the headline result; ρ sweep |
| P4 crossings | lines (realistic texture), 90 / 60 / 45 / 30°, fractions 1:1, 2:1, 4:1; rods as sanity | single | N, directions, fractions |
| P5 fan | a divergence-free fan (Cottaar-style) | — | N where it varies continuously |
| P6 vessels | dark tubes (20–100 µm) crossing a fibre field | no tubes | vessel halo failure |
| P7 laminar slab | slab with internal texture | — | the callosum case |
| P8 homogeneous | fibre texture at matched contrast, no structure | — | the clear-tissue floor; null calibration |
| P9 null | noise only, matched spectrum | — | false populations, `s0` |

Structure sizes, contrast steps (IC vs thalamus) and the noise power spectrum are **measured on the real 20 µm crops**
(label-free: the step from the image's own intensity histogram across the IC, the spectrum from homogeneous blocks)
and the phantoms matched to them, with the matching reported. Every phantom is grown by the network's margin (≥ 32
voxels, §8) and cropped before scoring.

**Arms** (all through the same read-out code):
- S1 intensity ST at BRIDGE's ρ 3; S2 at the matched ρ; S3 fix A (gradient excision);
- T1 48-D TCA (shape only); H1 hybrid blend; H2 TCA-excised intensity tensor;
- N1 CSD scaffold + TCA refinement + stability (recommended); N2 over-complete CP + geometric grouping; N3 FRT peaks;
  N4 the current BIC rules (as the baseline to retire);
- **C3 no-network basis:** channels = multiscale Gaussian derivatives (σ 1, 2, 4 voxels: gradient and Hessian
  components, ~24 channels), each locally contrast-normalised over ρ, through the identical TCA. If C3 meets the
  thresholds the network is dropped; if not, the gap is the network's measured contribution.

**Pre-registered acceptance thresholds** (median over sites, CI upper bound must pass):

| claim | measure | threshold | baseline it must beat |
|---|---|---|---|
| interface fixed | P3 step 15 and 30, within 0.5 mm, all bands | ≤ 10° | S2 ≥ 40° |
| nothing lost in clear tissue | P3 step 0, P8 | hybrid ≤ S2 + 1° | S2 ≈ 2° |
| U-fibres kept | P2, 0.15–1.2 mm band | ≤ 10° to the contour tangent | — |
| GW shell gone | P1: boundary-parallel share 0–0.5 mm | within 5 points of the twin | A 100% vs B 43–49% |
| crossings resolved | P4 lines ≥ 45°, fraction ≥ 0.2 | N correct ≥ 95%; both directions ≤ 10° at ≥ 90% of sites | ST: the bisector |
| no false populations | P4 single, P8, P9 | 2nd population at ≤ 2% of sites | BIC: 100% |
| fractions | P4, 2:1 and 4:1 | fraction error ≤ 0.1 | — |
| boundary field | P3, "within 0.5 mm of the interface" | precision and recall ≥ 0.9 at step ≥ 15 | coarse-gradient detector |
| brightness invariance | P3 ±20 offset, ×0.8 / ×1.25 | direction change ≤ seed spread | — |
| vessels | P6 within 2 radii of a tube | ≤ 15° | S2 |

**End to end in BRIDGE** (the point of the exercise): builds on P1/P2 (A-vs-B wrapping metrics from
`BOUNDARY_PROBLEM.md` §3: length within 1 mm of the boundary, runs ≥ 5 mm parallel, entering GM by > 1 mm) and on P4
(the `tube_cloud` crossing with `score_streamlines`: fibre-correct steps, bundle recall, cross-structure jumps), each
with the ST field and with the shape field, same config otherwise. Acceptance: "A behaves like B" (every wrapping
metric within 5 points of the twin's) and crossing bundle recall ≥ 0.9 with cross-structure jumps not above the
single-population build.

**Real data (qualitative, by decision).** The 20 µm S-21-46 crop and one other atlas brain: glyph renders at the IC /
thalamus interface, `bstrength` maps, `npop` maps, and BRIDGE builds with both fields compared by label-free
consistency metrics (straightness, length distribution, (λ2−λ3)/λ2 of the used direction, cross-structure jumps,
left-right symmetry of the connectome where applicable). Reported as consistency, never as accuracy.

**The "where ST fails" figure set.** One figure per row of §2: the twin pair, error vs distance curves for S1 / S2 /
S3 / H1 / H2 with CIs, the failure map, and the end-to-end streamline render for P1 and P4.

## 8. Package design and compute

**Package `shapetensor`** (this repo, made installable; `src/shapetensor/`, CLI `shapetensor`):

```
features   image tiles -> anatomix layer 58 -> per-volume PCA(16) -> float16 OME-Zarr      (GPU, torch)
tensor     16-ch gradients (σ 1) -> products -> box-reduce by b -> Gaussian ρ/b -> K store   (GPU, torch)
field      K -> ST parity tensor, TCA R3, boundary, hybrid; CSD + TCA 2N + stability -> field store
phantoms   make / score the suite of §7; the "ST fails" figures; the report (JSON + markdown)
bridge     nothing: BRIDGE reads the field store; a `validate-store` subcommand checks its contract
```

- **Tiles with discarded margins.** Measured today: layer-58 perturbation from one voxel falls to 0.1% of peak at 27
  voxels and 0.01% at 86. Tiles of 384³ with a 32-voxel margin (core 320³, 1.73× voxel overhead) are exact to 1e-3,
  verified by an equivalence test against a one-pass block. 448³ fails (INT_MAX elements in the upsampling), so 384³
  is the ceiling per pass.
- **Grid phases.** 8 phases, or the [1 2 1] filter with 1 phase if the equivalence test passes (8× cheaper).
- **Never materialise full-resolution gradients or products.** Per tile: features → 48 gradient maps → 1176
  products → block sum over b³ → accumulate into the coarse `K` store (1176 × (N/b)³ float32: 4 GB for a 768³ crop at
  b 8; 86 GB for a 20 µm whole brain at b 16). The ρ Gaussian then runs on the coarse grid.
- **Read-outs** batched on the GPU: Cholesky of 48×48, CP-ALS (the prototype's `cp_als`), SH projection + CSD. At
  b 16 a whole brain is ~2 × 10⁷ sites — tens of minutes.
- **Scheduling.** Dask (one worker per GPU) over a tile list with per-tile done-markers in the store, so a job that
  dies resumes; the same walltime/USR1 handling as `bridge_sge.sh`. Input: the acquisition's OME-Zarr and mask (the
  mask bounds which tiles run).
- **Container.** `envs/shapetensor-gpu.def` from `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (driver 550 =
  CUDA 12.4 on the H100 nodes), `pip install anatomix monai zarr>=3 dask distributed`, the anatomix weights baked in
  at build (compute nodes may not reach the HF hub), and the `shapetensor` wheel. An SGE script mirroring
  `scripts/cluster/bridge_sge.sh` (`-pe gpu N`, `tmem`, `-notify`, resubmit-to-continue).
- **Projected cost** (A6000 measured; H100 ≈ 2.5×): network 38.8 M voxels/s per phase at 384³ → with 8 phases and the
  tile overhead 2.8 M core voxels/s → a 768³ crop in ~3 min, a masked 20 µm hemisphere (~3 × 10¹⁰ voxels) in ~3 h,
  a whole brain (~7.5 × 10¹⁰) in ~7.5 h on one A6000; ~1 h on an 8-H100 node. The tensor pass is memory-bound and
  comparable; the read-outs are minutes. With the [1 2 1] filter all of this drops by up to 8×.
- **Parity tests:** the package's ST against BRIDGE's `_structure_tensor` (same σ, ρ, truncate, `mode='nearest'`);
  tiled features against one-pass; the field store's contract (`validate-store`).

## 9. Risks, in order

1. **The hybrid has never run.** If the step detector misfires in textured tissue the field is worse than ST in
   exactly the regions BRIDGE covers well. Mitigation: null calibration on P8/P9, the H2 alternative, and the end-to-end
   P1 build before anything touches real data.
2. **ρ at 20 µm is unknown**, and the network's blur fixes a minimum window. The sweep on P3 at 20 µm is the first
   experiment; if no ρ passes both the interface and the clear-tissue thresholds the hybrid carries the whole load.
3. **Realistic-texture crossings** are at 29% today. CSD is a reasoned fix, not a demonstrated one; the acceptance
   threshold may have to be reported as unmet.
4. **CP swamps** on ill-conditioned sites; guarded by the collinearity reject and the quality fields, but it will show
   up as holes in `npop`.
5. **Brightness sensitivity** of the 48-D layer; tested explicitly, with the 16-D `|df/dx|` as a fallback detector.
6. **The interleaved vertices and the Gabriel filter**; measured end to end, with a graph-stage exclusion as the
   fallback (asked for separately).
7. **No real-data truth**: everything on tissue is consistency, by decision; the paper's claims rest on the phantom
   suite matching real statistics, so the matching must be reported honestly.
8. **Cost** is dominated by the network; the [1 2 1] test and the C3 arm are the two ways it could shrink by an order
   of magnitude.

## 11. Addendum after the validation run (2026-10-16; full numbers in `TCA_FIELD_RESULTS.md`)

The package was built as planned (§8, BRIDGE reader deferred) and the phantom suite run at 20 µm with statistics
matched to the real S-21-46 crop. The evidence overturns the central hypothesis and replaces the recommendation.

**What held.** The structure-tensor failure is real and reproducible: next to a 40-level step (the real IC/thalamus
step is ~38) the intensity tensor is at 41 / 13 / 7° (ρ 8 / 16 / 32 voxels) at the realistic texture level and
58 / 53 / 42° at a quarter of it, against 3–6° in clear tissue, with the shell thickening with ρ.

**What did not.** Shape-space TCA does not rescue the orientation at 20 µm. The 16-D output is blind to weak texture
(random in clear tissue at a tenth of the texture); the 48-D layer keeps a 26–30° floor at a quarter of the texture
and still carries the step (59–61° at the interface); the locally normalised derivative jet is genuinely
step-insensitive (its interface error equals its clear-tissue error at every texture level) but orientation-poor.
The 7° result on the 0.1 mm phantoms came from a regime (sub-voxel sparse fibres, a 0.8 mm receptive field) that does
not exist at 20 µm. TCA's pieces also do not name the step at 20 µm (detector PR-AUC ≤ 0.4), so the boundary
extraction of §5 does not come from TCA either.

**What works instead.**
1. *Single direction:* the intensity tensor with **gradient excision** (fix A: drop gradients whose coarse magnitude
   exceeds 3× the median, renormalise the integration): 5.4 / 2.9 / 2.1° at the step (half texture) and
   11.9 / 5.5 / 3.1° (quarter) where the plain tensor is 41 / 13 / 7° and 58 / 53 / 42°, at zero cost in clear
   tissue, excising 0–1% of voxels; 1.3° at a vessel. The full suite added one condition: the reference median must
   be taken over *white matter*, or the excision gated by the null-calibrated `cl` — with the block median the
   threshold is too low at a gyrus (most of the window is smooth grey matter and CSF) and the ungated form harms
   there (37 / 18 / 9°), while the gated form clears the shell (5.2 / 2.5 / 1.9°; 15 / 6.6 / 2.7° at quarter
   texture). Where there is no texture signal to recover (a tenth of the texture at ρ ≤ 16) nothing helps but a
   larger window.
2. *Boundary field:* the **coarse gradient magnitude** is the boundary detector (PR-AUC 0.85–0.96 at ρ 16 at every
   texture level); Westin's **cl** of the intensity tensor fires only when the step dominates the window, which is
   exactly when the orientation is at risk, so it is the right gate / reliability flag; the orientation gap
   (λ2−λ3)/λ2 remains the quality measure.
3. *N populations:* **constrained spherical deconvolution of the gradient-direction ODF** (a ring response), rank-free
   by construction: on a 90° crossing N = 2 at 96–100% of sites with both directions within 15° at 68–100% for
   ρ ≥ 16 (errors 4–9°), on the intensity gradients alone; the shape engines add a little at ρ 16. BIC never selects
   two populations; the over-complete CP grouping overcalls (up to 100% false second populations on single-population
   phantoms); the TCA refinement of the CSD peaks *degrades* them; the residual-bootstrap stability test is too strict.
   The SH order-8 limit is reached at 45°; order 12 is the follow-up.

**Revised recommendation for the BRIDGE input field.** `shapetensor.field.compute_field`: plain and excised
intensity tensors (BRIDGE's σ / ρ, ρ ≈ 16–32 voxels at 20 µm, i.e. 0.3–0.6 mm), `gm` and `cl` as the boundary field,
the gap as quality, and CSD fixels (up to three directions with fractions) for the interleaved-vertex N-population
input. The anatomix engines stay in the package as a tested, documented negative result and as an optional
comparison direction; the TCA machinery stays for the reasons above. The remaining open points are the end-to-end
BRIDGE builds on the gyrus phantoms (the reader is deferred), the gyrus / laminar / vessel families and the excision
arms of the full suite (running), and the real-crop demonstration.

## 10. Decisions wanted before the implementation plan

1. The scaffold for N: CSD + TCA refinement + stability (recommended), with over-complete CP as a comparison arm.
2. The hybrid: run both H1 (blend) and H2 (TCA-excised intensity tensor); default decided by the phantom scores.
3. The field store contract of §6 and the BRIDGE reader (`orientation_source`, `population`/`fraction` attributes,
   per-population stratified streams).
4. The phantom suite and thresholds of §7 as the pre-registration (any change after the runs is reported as such).
5. The package layout of §8: `shapetensor` in this repo, its own container, `K` as a persisted intermediate.
6. The order of work: (i) package skeleton + features/tensor passes with parity tests; (ii) P3 and P8 at 20 µm with
   the ρ sweep and the C3 control; (iii) the hybrid; (iv) N populations on P4/P5; (v) the BRIDGE reader and the
   end-to-end builds; (vi) real-data demonstration; (vii) container + cluster run on a whole crop set.
