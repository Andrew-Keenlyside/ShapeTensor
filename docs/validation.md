# Validation and tracking

The acceptance suite (`shapetensor validate`, `shapetensor.validate`) runs the chain and the excised structure tensor on
phantoms with truth and scores them three ways: the populations on the lattice (N, directions), the identities, and —
the test the design is for — **tracking**: streamlines seeded in one arm of a structure, followed by the same
deterministic tracker in both fields, scored by where they end. All phantoms use the texture matched to a real 20 µm
HiP-CT crop (20° of fibre dispersion within a population, half the fibre darkening), 32 voxels of margin on every face,
the core scored; `scale` 10 (0.2 mm) throughout. Numbers below: one seed, 2026-10-22 (`results/acceptance.json`).

## Tracking: does a streamline leave a crossing on its own bundle?

| phantom | measure | shape | structure tensor |
|---|---|---|---|
| band crossing 90° | through the crossing on its own band / switched to the other | **0.74** / 0.04 | 0.07 / 0.65 |
| band crossing 60° | through / switched | **0.81** / 0.12 | 0.42 / 0.55 |
| band crossing 45° | through / switched | **0.68** / 0.19 | 0.40 / 0.55 |
| T-junction 90° | false continuation past the T | 0.00 | 0.00 |
| free end | streamlines continuing beyond the bundle's end | **0.00** | 0.40 |
| intermixed crossing 90° | stayed within 15° of its population, end to end (pop. 1 / 2) | **0.57 / 0.63** | 0.10 / 0.22 |
| intermixed crossing 60° | same | **0.50 / 0.67** | 0.00 / 0.00 |
| intermixed crossing 45° | same | 0.24 / 0.19 | 0.02 / 0.01 |
| bend, fascicles of radius 0.3 / 0.8 / 1.3 mm | through the 90° turn | 0.52 / 0.83 / **0.99** | **0.92** / 0.85 / 0.99 |
| straight control, three fascicles | through | 0.97 / 0.89 / 0.94 | 0.96 / 0.84 / 0.98 |

The structure tensor's streamlines switch bundle at every crossing (its one axis is the bisector, nearer the stronger
band) and run on through texture past a bundle's end (it always has a direction); the chain's pass through on their own
band and stop where nothing is present. The tightest bend (radius 0.3 mm, 15 voxels, inside a 0.2 mm window) is the
one case the structure tensor tracks better: the lifted field is read on a lattice at the window's scale and the
carry-forward runs straight, so a turn that tight inside one window is partly lost. Both fields are tracked by the same
code with the same seeds, step (0.5 voxel) and turning limit (30° per step).

## Populations on the lattice

| phantom | N correct | directions |
|---|---|---|
| bundle | N = 1 at 0.97 of cells | 3.1° median |
| bend / straight fascicles | N = 1 at 0.90 / 0.96 | 4.3° / 4.1° |
| texture (no structure) | N = 0 at 1.00 (and at 1.00 of voxels) | — |
| interface (a 40-level step) | N = 1 at 0.90 | 6.6° |
| band crossing 90° / 60° / 45°, in the overlap | N = 2 at 0.78 / 0.96 / 0.60 | pair 5.7° / 6.7° / 5.7°; both within 15°: 0.72 / 0.86 / 0.54 |
| band crossing, single-band cells | N = 1 at 0.80 / 0.85 / 0.86 | 5.0° / 4.0° / 5.9° |
| T-junction: cells beyond the T | false extension 0.00 | — |
| intermixed crossing 90° / 60° / 45° | N = 2 at 0.57 / 0.69 / 0.41 | pair 8.1° / 10.0° / 13.1° |

The intermixed crossings are the hard case for any windowed estimate: two dense populations halve each other's
correlation, so the presence of a 90° crossing is close to the floor (a third of its cells fall below the weight
threshold), and at 45° the two lobes sit inside the response's width. The remaining false second populations in
single fascicles (10–18% of cells) lie along the phantom's extrusion axis: the far point of the fascicle boundary's
ring, which the high-pass and the excision remove in most cells and the sheet rule does not reach when a strong fibre
lobe sits on the same ring.

## Identities

| phantom | dominant label covers | leaks into the other bundle |
|---|---|---|
| bend (three fascicles turning 90°) | 0.99 of the bundle's cells, one label through the turn | — |
| band crossing 90° / 60° / 45° | 0.89–0.97 of each band | 0.04–0.15 |
| T-junction | 0.99 (through band) / 0.89 (stub) | 0.13 / 0.00 |

## The single field

Where the chain gives weight ≥ 0.2, its strongest direction is within 3–6° of the truth on every single-population
phantom (bundle 3.0°, bend 4.2°, bands 4–6°, interface 6.5°), the excised structure tensor within 3–6° on the same
voxels; at a crossing the structure tensor reports the bisector and the chain its strongest population, which a
single-vector consumer should prefer.

## Speed

On an RTX A6000 the chain takes about 5 s on a 192 × 160 × 192 block and 7 s on 256³ (the excised structure tensor
1.4 s); a 384³ chunk with its 56-voxel halo (496³) about 70 s in a whole-volume run. See the README for the whole-volume
throughput and `shapetensor bench` for your GPU.

## What is not met yet

Against the acceptance criteria set out before the build: through-crossing ≥ 0.9 at 90° and 60° (0.74 and 0.81),
≥ 0.7 at 45° (0.68); the tightest bend; identity leakage ≤ 2% (4–15%); per-voxel resolution of intermixed crossings at
this dispersion. Multi-resolution and anisotropic-voxel invariance are covered by unit tests, not yet by phantom
scores.
