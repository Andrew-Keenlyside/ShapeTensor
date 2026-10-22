# Parameters

Everything is a keyword of `shapetensor.field.Orientations` (and so of `shapetensor()` and `shapetensor run`). One
parameter sets the scale; the others are bounds and switches with defaults that hold across the phantom set.

## The one that matters

| parameter | default | what it is |
|---|---|---|
| `scale` | 10 voxels | the window σ. Tied to it: the lags (`0.2–0.45 scale`), the lattice step (`scale / 2.5`), the high-pass (`0.3 scale`), the carry-forward Gaussian (`1.2 scale`) and reach (`2.4 scale`), the trusted margin (`0.45 scale + 2 scale`). At 20 µm, 10 is a 0.2 mm window and a 0.5 mm reach; at 50 µm the same tissue scale is `scale=4`. |

Choose it as you would the structure tensor's ρ: large enough that a window holds several structures of the
population, small enough that the population does not turn much inside it. Resolution changes mean a proportional
change of `scale`; nothing else.

## Bounds on the populations

| parameter | default | effect |
|---|---|---|
| `nmax` | 2 | the most populations a position can hold |
| `min_sep_deg` | 30° | the least angle between two populations; below the response's width (about 33° at half height on HiP-CT texture) two lobes are not resolved anyway |
| `rel` | 0.3 | a further lobe must hold this fraction of the strongest. Lower (0.2) keeps more second populations in crossings (band crossings: both within 15° 0.77 → 0.85 at 90°, 0.94 → 0.98 at 60°) and admits more false seconds in single fascicles (bend 0.86 → 0.74) |
| `min_share` | 0.15 | a further lobe must hold this share of the position's distribution |
| `min_weight` | 0.2 | the presence weight below which a position has no population (N = 0); the single field is not affected |
| `valley` | 0 | if set, a further lobe must stand above the saddle between it and the strongest by this fraction of its height (a shoulder test; off: shoulders are rare after deconvolution) |

## Switches and rules from the data

| parameter | default | effect |
|---|---|---|
| `momentum` | on | the carry-forward along each direction; off for a per-position reading |
| `excise` | 3.0 | voxels whose coarse gradient exceeds this × the median are left out of the correlation; 0 to disable |
| `mean_scale` | 0.3 | the high-pass σ as a fraction of `scale`. 0.3 keeps the fibre texture (periods of 3–10 voxels at 20 µm) and removes the slow structure of boundaries; 0.8 let a fascicle's boundary lift to a ring whose far point became a false second population (bend fascicles with one population 0.56 → 0.86 from 0.8 to 0.3) |
| `presence_mode` | `"excess"` | the lobe mass above the median (N-independent); `"peak"` the peak height (halves at a crossing) |
| `lags` | (0.2, 0.45) | the shell of lag radii as fractions of `scale` (2–4.5 voxels at the default). The first build used (0.25, 0.65): sharper on the phantoms' intermixed crossings at the cost of reading the acquisition's anisotropy on the HiP-CT volume (48° from the confident structure tensor against 15°); see the results log |
| `radial` | `"count"` | how the shell's lags are weighted per direction: the lattice's own count (`∝ r²`, the longest lags dominate) or `"uniform"` (`1 / r²`, every radius alike) |
| `fine_scale` | 0 | a low-pass (fraction of `scale`) before the high-pass: a band-pass that keeps voxel-scale acquisition structure out of the correlation (0.1 = 1 voxel at the default); little effect on the HiP-CT volume |
| `acquisition_null` | off | subtract the mean lift profile of the positions *below the floor* (what structure-free positions correlate along is the acquisition's): on the HiP-CT block it halves the share of first lobes along the rotation axis (0.64 → 0.41) and costs 10–15% of the phantoms' crossing cells; `min_null_cells` (500) positions are needed, else nothing is subtracted |
| `null` | `"isotropic"` | the surrogate the presence floor is measured on. `"spectrum"` keeps the block's full, anisotropic power spectrum and subtracts the surrogate's mean directional lift from every lift: it removes what the acquisition alone puts into the field (a point-spread function longer along the rotation axis, rings), but also a region's dominant real orientation, so it must be measured on a region with many orientations (`run_volume(null="spectrum")` on its calibration region) |
| `momentum_angular_deg` | 0 | an angular smoothing inside the carry-forward so bends are followed; 8° changed the tight bend by less than 1% |
| `max_ring`, `min_prominence` | 0.35, 5 | the sheet rule |
| `null_q`, `full_factor` | 0.99, 2 | the surrogate quantile that sets the floor, and where the weight reaches 1 |
| `n_dirs`, `kernel_deg` | 160, 8° | the hemisphere grid and the angular kernel the lags are binned with |
| `label_angle_deg`, `min_fixels` | 20°, 8 | identities: the alignment that links neighbouring fixels; the smallest population labelled |
| `iters`, `angular_deg`, `bin_deg` | 300, 6°, 2.5° | the deconvolution's iterations, the smoothing after it, the response's angular bins |

## What is measured, not set

The presence floor (on the block's surrogate) and the response (on the block's own single-lobe positions; when fewer
than 200 qualify the selection is loosened, and failing that a Gaussian profile of 33° half-width, the phantoms'
value, is used). For a whole volume both are measured once on a calibration region; `run_volume(floor=, response=)`
accepts given values, so a run can be reproduced exactly or a volume calibrated on a chosen region.
