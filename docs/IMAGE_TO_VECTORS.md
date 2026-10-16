# From image to vector field: the current best path

Status: 2026-10-16. **Everything here is a prototype, tested on synthetic phantoms only.** Nothing has yet been run on
the 20 µm S-21-46 crop or any real tissue. The numbers below are phantom results; they show which choices win, not how
well any of this will do in a brain.

## In short

The vector field is the fibre direction at each point. Today's best path depends on what the point sits in:

- **Clear tissue (no intensity step nearby): the plain intensity structure tensor.** It is the most accurate thing we
  have (about 2° on the phantoms), and nothing in shape space beats it there.
- **Near an intensity step (a tract interface, the grey/white boundary): anatomix's 48-channel layer plus TCA.** This
  does not wrap along the boundary. At a 15-step interface it is 7° off, against 41–62° for the structure tensor. It
  needs no labels, no paired image and no oracle.
- **Crossing fibres: TCA with two pieces per population.** Experimental. It recovers both directions where the fibres
  are clean, solid structures, but not yet with realistic thin, noisy fibre texture.

The piece still missing is the **switch between the first two**: deciding, without labels, which points are "near a
step". It is designed but not yet tested.

## The path

```
 image (one channel, any contrast)
   │
   │ 1. grow / pad the block, so orientations are never computed near a face
   ▼
 anatomix, run cleanly: one pass, no sliding windows, 8 one-voxel grid phases averaged
   │
   ├── 16-D output (the full model) ─────────► region identity and boundaries (what the network was trained for)
   │
   └── 48-D layer 58 (before the output head) ► local texture + context: keeps the fibre stripes
          │
          │ 2. gradients of every channel: a table per voxel (48 channels x 3 axes)
          ▼
        per point, a window (Gaussian, rho = 16 voxels): an array  voxels x channels x axes
          │
          │ 3. TCA (CP, rank 3): split the array into pieces = spatial pattern x channel mix x ONE direction
          ▼
        fibre direction = the direction the pieces do not use (the null direction of their gradient-axis Gram)
          │
          │ 4. combine with the intensity structure tensor (it wins away from steps)   <- designed, not yet tested
          ▼
 vector field: one direction per point (or, experimentally, N directions + fractions per point: "fixels")
```

## Step by step

### 1. Image and context

- **Give every point real context.** Read the block with a margin, or grow the phantom. Orientations computed
  within the reach of a block face are wrong: both the anatomix network and the tensor window see the edge.
  - The phantoms are grown by 9.6 mm (96 voxels) on each x and z face, then cropped back before scoring
    (`surface_phantoms.make_D_padded`).
  - Removing the face effects made the D15 interface results cleaner and stopped the dark edge strips.
- **Parameters are in voxels, not mm.** anatomix ignores the voxel size, and ρ is in voxels. The phantoms are
  0.1 mm/voxel. At 20 µm the same settings cover a 5× smaller physical area, so they must be re-set for real data,
  relative to the fibre size.

### 2. Run anatomix cleanly

`anatomix_shape.anatomix_clean` (16-D) and `anatomix_shape.anatomix_layer` (48-D) do this:

- **One forward pass, no sliding windows.** Windows leave their stride as seams, and the seams read as straight
  "boundaries".
- **Reflect-pad by 16 voxels and up to a multiple of 32.** The network's own zero padding puts a hard step at the
  faces.
- **Average over the 8 one-voxel grid phases** (shift, run, shift back), then scale each voxel to unit length. This
  removes the network's period-2 checkerboard.
- **The one-pass limit is about 352³ padded** (the GPU's 2³¹-element cap). `anatomix()` now raises an error if
  chunkreg silently falls back to tiles. Large real crops will need "tiles with discarded margins"; that is open.

### 3. Which anatomix space

| | 16-D output (full model) | 48-D layer 58 (head removed) |
|---|---|---|
| what it is | the trained output: 3 more convolutions squeeze 48 → 16 | 16 first-encoder channels (raw local texture, about 7 voxels) + 32 decoder channels (context) |
| what it keeps | which structure / region, and edges between structures | fine texture (93–97% of its variance), including the fibre stripes |
| what it discards | appearance: brightness and texture that doesn't define a structure | nothing yet; it still responds to raw brightness |
| at an intensity step | the step dominates (it looks like a boundary between two structures) | the step is one small change among many texture channels |
| use it for | boundary / region identity | **orientation** |

Why the difference: anatomix was trained so that the same anatomy gives the same 16 numbers however the image looks.
The output head therefore learns to drop appearance, and fibre texture is mostly appearance. The 48-D layer sits
before that head.

### 4. Gradients and the window

- At each voxel, take the Gaussian derivative (σ = 1 voxel) of every channel: a table of 48 channels × 3 axes.
- Around each output point, collect those tables over a Gaussian window of **ρ = 16 voxels** (truncated at 2ρ).
  - The window is what lets the method see past the step: at ρ = 6 the D15 interface is still 50° off.
  - The array is voxels × channels × axes. Summed over voxels and channels it becomes the familiar 3×3 shape
    tensor; TCA keeps it unsummed.
- For TCA the 48 channels are compressed to their top 16 principal components (99% of the variance), so the
  per-point array stays small. Its spatial mode is compressed exactly by a Cholesky factor of the channel-resolved
  tensor K = G_ρ ∗ (g gᵀ).

### 5. Read out the orientation

- **One population (the default): TCA, rank 3, nothing dropped.**
  - The array is written as three pieces. Each piece is a spatial pattern × a channel mix × one direction, and the
    directions need not be at right angles.
  - The fibre is the direction the pieces don't use: the null direction of their gradient-axis Gram.
  - It beats the eigenvector of the 3×3 tensor at the interface (7° vs 11° with 48-D, ρ 16).
- **Several populations (experimental): TCA with 2 pieces per population** (rank 4 for two fibres). Each population's
  gradients lie in the plane perpendicular to it, so:
  - **pair the pieces geometrically**, choosing the pairing whose pairs are most perpendicular (no truth needed);
  - each pair's fibre is d₁ × d₂.
  - Grouping pieces by channel-signature similarity does **not** work: identical-looking populations have identical
    signatures.
- **How many populations per point (open).** dMRI-style rules were tried: BIC between 1, 2 and 3 populations, a 10%
  fraction floor, and merging within 20°. BIC alone is badly calibrated (it always picks 3 on clean rods). The floor
  and merge rules help, but false third populations remain. Next to try: stability under resampling.

### 6. Combine with the intensity structure tensor (designed, not tested)

- **Away from steps** the intensity tensor is far better (1.5–2° vs 9–12° for any shape-space read-out).
- **At steps** it is far worse (41–89°).
- So the field should come from the intensity tensor in clear tissue and from 48-D TCA near steps. The switch must be
  label-free. Candidates:
  - the coarse intensity-gradient magnitude, the detector gradient excision uses;
  - the 16-D shape space's own boundary strength (|df/dx|).
- This step has not been run.

### 7. Output

- One unit vector per point (axial: ±u). This is what BRIDGE's vertex stage consumes today.
- Experimental: N directions with a fraction each, per point (fixels). **Feeding several directions per vertex into
  BRIDGE would change what the walk consumes: ask before doing it** (project rule).

## The evidence for each choice

### Single population at a tract interface

Phantoms D0 (no step) and D15 (step of 15 across a plane, fibres crossing it at 90 / 60 / 30°). Grown phantoms, whole-
block anatomix, median angle to the true fibres. Source: `shape48_orient.json`, `shape_tca*.json`.

| read-out | D15 within 0.5 mm of the interface | D15, 3–8 mm away | D0 within 0.5 mm | needs |
|---|---|---|---|---|
| intensity structure tensor, ρ 3 (BRIDGE now) | 62° | 7° | 7° | — |
| intensity structure tensor, ρ 16 | 42° | **2°** | **2°** | — |
| 16-D shape tensor, all channels, ρ 16 | 64° | 10° | 12° | — |
| 16-D, boundary directions removed (paired), eigen, ρ 16 | 21° | 8° | 10° | the no-step twin |
| 16-D, + TCA, step component dropped (gated oracle), ρ 16 | 13° | 8° | 10° | the twin + the true normal |
| 48-D shape tensor, eigen, ρ 16 | 11° | 11° | 12° | — |
| **48-D TCA, rank 3, ρ 16** | **7°** (bands 3 / 8 / 7°) | 9° | 10° | — |
| gradient excision (fix A; earlier, unpadded phantoms) | 9° | — | — | a step threshold; ends walks at real surfaces |

### Crossings

Rod phantoms and D-texture line phantoms, 16-D shape space, ρ 8, central 6.4 × 6.4 mm. Source: `shape_cross_*.json`,
`shape_npop.json`.

| phantom | intensity tensor | TCA, 2 pieces per population, geometric pairing |
|---|---|---|
| rods, 90° + 30° (noise 2 / 8) | the bisector | **4° / 4°**, both within 15° at 100% of sites |
| rods, 60° + 120° (noise 2 / 8) | the bisector | 6° / 13° (70%); 5° / 11° (91%) |
| D-texture lines, 60° + 120°, noise 8 | the bisector | 19° / 13° (29%) |
| D-texture lines, single 30°, noise 8 | **1.5°** | 15° (the 16-D space represents thin random lines poorly) |

For counting populations, the Funk–Radon (q-ball style) ODF of the **intensity** gradients gets N right at 100% of
sites on the rods, but merges the two populations in the line phantoms. Constrained spherical deconvolution is the dMRI
fix for that and hasn't been tried yet.

## Known problems

- **Phantoms only.** No real-tissue run yet.
- **A floor of 9–12° in clear tissue** for every shape-space read-out. The network blurs about 7 voxels and has its own
  direction preferences.
- **Uneven response in the 16-D space:**
  - grid-aligned rods give about 2.2× the gradient energy of oblique ones;
  - thick rods drown out thin ones;
  - thin random fibres at 30° are 15–19° off even alone.

  In a crossing, the weaker population can vanish.
- **The 48-D layer responds to raw brightness:** a regional intensity offset changes its features everywhere, not
  only at the step.
- **Several directions per point:** N selection is not calibrated.
- **The hybrid switch (step 6) is untested.**
- **Scale:** ρ = 16 voxels is 1.6 mm on the phantoms. The right ρ for 20 µm data is unknown.

## Tried and set aside

| idea | why set aside |
|---|---|
| orientation from 16-D shape gradients alone | the step dominates (64° at D15) |
| boundary/fibre split learned from D15 − D0 | needs a no-step twin, impossible in real data; the step directions also carry 94% of D0's own variance |
| drop the TCA component nearest the true normal | an oracle; harmed D0 until gated |
| 2-way split by low-rank structure along u | the projection along the true fibre is no more low-rank than along the wrong one |
| unpaired scale split of the 48-D space | separated visually, but made no difference to the orientations |
| k-means / shape clusters | stopped at the user's request |
| smaller anatomix windows (8, 16 vox) | 11 h for w8 at 0.75 overlap; mirror-tiled; stopped as too slow |
| unmixing into signature abundance maps (method 2) | cannot separate identical-looking populations; missed the thin rods |

## Scripts and outputs

All under `scripts/`; data and JSON in `/home/andrew/bridge_wmcov/shape_tensor/`; figures in `figures/shape_tensor/`.

| step | script | figure |
|---|---|---|
| clean anatomix (16-D, 48-D) | `anatomix_shape.py` (`anatomix_clean`, `anatomix_layer`) | |
| grown phantoms | `surface_phantoms.py make_pad` | |
| 48-D orientations vs structure tensor | `shape48_orient.py` (+ `maps`, `fig`) | `shape48_glyphs.png` |
| TCA on the shape tensor, drops, gating | `shape_tca.py` | `shape_tca_maps_pad96.png`, `shape_tca_summary_pad96.png` |
| the pipeline step by step (16-D, D0/D15) | `shape_tca_steps.py` | `shape_tca_steps.png` |
| crossings | `shape_cross_binary.py`, `shape_cross_texture.py` | `shape_cross_binary.png`, `shape_cross_texture.png` |
| N populations: Funk–Radon ODF | `shape_npop.py` | `shape_npop.png` |
| N populations: TCA fixels, dMRI rules | `shape_tca_fixels.py` | `shape_tca_fixels.png` |
| N populations: joint vs unmixing | `shape_pop12.py` | (rules need fixing; see the log) |

## Next steps, in order

1. **Build and test the hybrid** (step 6): the intensity tensor in clear tissue, 48-D TCA near steps, switched by a
   label-free step detector. Score it on D0 / D15 / D30.
2. **Run the single-population path on the 20 µm S-21-46 crop.** Choose ρ relative to the fibre size there, and
   compare with the intensity tensor on the internal capsule.
3. **For crossings:**
   - calibrate N with a stability test;
   - try constrained spherical deconvolution on the intensity-gradient ODF for realistic texture;
   - check how strongly the 16-D space responds to different rod sizes and angles.
