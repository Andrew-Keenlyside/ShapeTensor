# Validation and tracking

The acceptance suite (`shapetensor validate`, `shapetensor.validate`) runs the chain and the excised structure tensor on
phantoms with truth and scores them three ways: the populations on the lattice (N, directions), the identities, and —
the test the design is for — **tracking**: streamlines seeded in one arm of a structure, followed by the same
deterministic tracker in both fields, scored by where they end. All phantoms use the texture matched to a real 20 µm
HiP-CT crop (20° of fibre dispersion within a population, half the fibre darkening), 32 voxels of margin on every face,
the core scored; `scale` 10 (0.2 mm) throughout. Numbers below: one seed, 2026-10-22, the release defaults (`results/acceptance.json`; the first build's numbers are in
the results log §35).

## Tracking: does a streamline leave a crossing on its own bundle?

| phantom | measure | shape | structure tensor |
|---|---|---|---|
| band crossing 90° | through the crossing on its own band / switched to the other | **0.89** / 0.00 | 0.06 / 0.65 |
| band crossing 60° | through / switched | **0.86** / 0.11 | 0.42 / 0.55 |
| band crossing 45° | through / switched | **0.89** / 0.05 | 0.40 / 0.55 |
| T-junction 90° | false continuation past the T | 0.00 | 0.00 |
| free end | streamlines continuing beyond the bundle's end | **0.02** | 0.40 |
| intermixed crossing 90° | stayed within 15° of its population, end to end (pop. 1 / 2) | **0.99 / 0.99** | 0.10 / 0.22 |
| intermixed crossing 60° | same | **0.67 / 0.91** | 0.00 / 0.00 |
| intermixed crossing 45° | same | 0.79 / 0.34 | 0.02 / 0.01 |
| bend, fascicles of radius 0.3 / 0.8 / 1.3 mm | through the 90° turn | 0.39 / **0.92** / 0.99 | **0.92** / 0.85 / 0.99 |
| straight control, three fascicles | through | 0.99 / 0.93 / 0.96 | 0.96 / 0.84 / 0.98 |

The structure tensor's streamlines switch bundle at every crossing (its one axis is the bisector, nearer the stronger
band) and run on through texture past a bundle's end (it always has a direction); the chain's pass through on their own
band and stop where nothing is present. The tightest bend (radius 0.3 mm, 15 voxels, inside a 0.2 mm window) is the
one case the structure tensor tracks better: the lifted field is read on a lattice at the window's scale and the
carry-forward runs straight, so a turn that tight inside one window is partly lost. Both fields are tracked by the same
code with the same seeds, step (0.5 voxel) and turning limit (30° per step).

## Populations on the lattice

| phantom | N correct | directions |
|---|---|---|
| bundle | N = 1 at 0.94 of cells | 2.1° median |
| bend / straight fascicles | N = 1 at 0.88 / 0.96 | 3.1° / 2.6° |
| texture (no structure) | N = 0 at 1.00 (and at 1.00 of voxels) | — |
| interface (a 40-level step) | N = 1 at 0.83 | 4.3° |
| band crossing 90° / 60° / 45°, in the overlap | N = 2 at 0.96 / 0.99 / 0.91 | pair 3.4° / 6.8° / 5.2°; both within 15°: 0.96 / 0.97 / 0.91 |
| band crossing, single-band cells | N = 1 at 0.89 / 0.83 / 0.84 | 3.0° / 7.0° / 5.2° |
| T-junction: cells beyond the T | false extension 0.00 | — |
| intermixed crossing 90° / 60° / 45° | N = 2 at 0.96 / 0.89 / 0.65 | pair 4.1° / 12.9° / 12.3°; both within 15°: 0.95 / 0.36 / 0.34 |

The remaining false second populations in single fascicles (5–17% of cells) lie along the phantom's extrusion axis:
the far point of the fascicle boundary's ring, which the high-pass and the excision remove in most cells and the sheet
rule does not reach when a strong fibre lobe sits on the same ring. At 60° and 45° the intermixed crossings' two lobes
are read but not both to 15° (the response is about 33° wide at half height).

## Identities

| phantom | dominant label covers | leaks into the other bundle |
|---|---|---|
| bend (three fascicles turning 90°) | 1.00 of the bundle's cells, one label through the turn | — |
| band crossing 90° / 60° / 45° | 0.98–1.00 of each band | 0.05–0.13 |
| T-junction | 1.00 (through band) / 0.90 (stub) | 0.15 / 0.00 |

## The single field

Where the chain gives weight ≥ 0.2, its strongest direction is within 2–7° of the truth on every single-population
phantom (bundle 2.1°, bend 3.1°, bands 3–7°, interface 4.3°), the excised structure tensor within 3–6° on the same
voxels; at a crossing the structure tensor reports the bisector and the chain its strongest population, which a
single-vector consumer should prefer (crossing 60°: 18.5° against 30.1° from the nearer population).

## A second seed

The same suite on a second seed of every phantom (`results/dev_pipeline/acceptance_seeds2`): band crossings through on
their own band 1.00 / 0.99 / 0.67 at 90° / 60° / 45° (seed 1: 0.89 / 0.86 / 0.89), switched 0.00 / 0.00 / 0.17; T-junction
0.00; free end 0.17 (seed 1: 0.02); intermixed crossings within 15° end to end 0.99 & 0.99, 0.87 & 0.83, 0.88 & 0.45; the
bend 0.41 / 0.98 / 0.95; the overlaps' N = 2 at 1.00 / 1.00 / 0.91; texture N = 0 at 1.00. The 45° band and the free end
are the two measures that move between seeds by more than a few percent.

## Resolution

The same phantoms downsampled 2× to 40 µm and run with `scale=6` (the lag shell and the high-pass stop at their
voxel floors: lags 2–3.6 voxels, high-pass 3): bundle N = 1 at 0.96 of cells, 2.2°; band crossing 90° overlap N = 2 at
0.82, both within 15° 0.64; intermixed crossing 60° N = 2 at 0.96, both within 15° 0.64 — the same tissue at 20 µm
and `scale=10`: 0.94 / 2.1°, 0.96 / 0.96, 0.89 / 0.36. Halving `scale` without the floors (lags of 1–2 voxels) lost the
bundle (N = 1 at 0.34): a lag under two voxels measures the point-spread function, not the structure.

## Artefacts

A bundle at 60° with streaks (lines in z slices, 1 voxel wide, half a line per slice, bright or dark) and rings
(concentric about the rotation axis, the same in every slice) at 0 / 50 / 100 / 200 grey levels; the fibre darkening
is 50 (`scripts/artefacts_chain.py`). Median error of the single field, chain / structure tensor:

| artefact | 0 | 50 | 100 | 200 |
|---|---|---|---|---|
| streaks in the fibre plane (along x) | 2.1 / 4.0 | 2.2 / 6.0 | 12.8 / 20.2 | 58 / 43 |
| streaks across it (along y) | 2.1 / 4.0 | 2.2 / 4.5 | 3.0 / 8.2 | 88 / 85 |
| streaks, any direction | 2.1 / 4.0 | 2.1 / 4.9 | 2.5 / 11.9 | 57 / 38 |
| rings | 2.1 / 4.0 | 17.2 / 12.6 | 21.2 / 15.6 | 23.2 / 12.0 |

Streaks up to twice the fibre contrast cost the chain 0–11°, the structure tensor 2–16°; at four times both fail. The
chain reports streaks as a second population (N = 2 in 16–47% of cells at 50, 60–90% at 100–200): weighted and
labelled, not hidden. Rings are the chain's weak case: a ring correlates along the whole plane of the slice, a quarter
to a third of the cells are declared sheets (weight 0) and the rest drift by 17–23°, where the structure tensor
degrades to 12–16°.

## Speed

On an RTX A6000 the chain takes about 5 s on a 192 × 160 × 192 block and 7 s on 256³ (the excised structure tensor
1.4 s); a 384³ chunk with its 56-voxel halo (496³) about 70 s in a whole-volume run. See the README for the whole-volume
throughput and `shapetensor bench` for your GPU.

## What is not met yet

Against the acceptance criteria set out before the build: through-crossing ≥ 0.9 at 90° and 60° (0.89 and 0.86);
the tightest bend (0.39 against the structure tensor's 0.92); identity leakage ≤ 2% (5–15%); the pair at 60° and 45°
intermixed crossings resolved to 15° (0.36 / 0.34 of cells). Multi-resolution is covered by the 40 µm check above and by
unit tests; anisotropic voxels by a unit test. On real HiP-CT data the lift also reads the acquisition's anisotropy
along the rotation axis: [real data](data.md).
