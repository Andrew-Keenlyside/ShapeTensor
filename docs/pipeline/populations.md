# 2. Populations

On the deconvolved field `g(x, u)`, scaled so that each position's distribution sums to its presence weight.

## Carry-forward (momentum)

For each direction `u`, the share `g(·, u)` is a scalar field over positions. It is smoothed along `u` (a Gaussian of
`1.2 scale` voxels, truncated at the reach `2.4 scale`) and compared with its **bridge**: the weaker of the two sides'
largest value within the reach along `u`. The result is `min(max(smoothed, bridge), max(g, bridge))`:

- a dip between two sides that both hold the population — a crossing, a partial-volume gap — is lifted to the bridge;
- a spike is smoothed;
- beyond a free end the smoothing's tail is cut by the position's own value, so nothing is extended.

It is a directional closing on the position × orientation field: the population-specific smoothing of the design,
and the counterpart of coherence-enhancing diffusion on orientation scores (Duits et al.), with the both-sides rule
the diffusion does not have. All directions of a batch are sampled in one call per lag (`populations.momentum`).

## Lobes = populations

`populations.peaks` takes each position's distribution and keeps its grid local maxima strongest first, at least
`min_sep_deg` apart, up to `nmax`; each lobe's mass is the distribution over the directions within 20° of its peak
that are nearer it than any other kept peak, and its direction their weighted principal axis (so directions are not
quantised to the grid). A further lobe must hold `rel` of the strongest and `min_share` of the position's total.
N is the number kept, zero where the weight is below `min_weight`.

## Sheets

A sheet — a boundary, a vessel wall, a sulcus — correlates along a whole plane and lifts to a ring, which peak-picking
would report as two crossing populations. `populations.ring_fraction` finds the plane of the distribution's second
moment and the share of its in-plane mass lying away from every kept peak (about 0.6 for a ring, near 0 for lines),
and the prominence of the strongest value over the ring's level; a position with ring above 0.35 and prominence
below 5 is a sheet, N = 0, flagged in the output. The high-pass (`0.3 scale`) and the excision keep most boundaries
from lifting to rings in the first place.

## Identities

`populations.label_populations` gives every lobe (a fixel) a population number shared across space. Candidate links
join a fixel to the fixel of a 26-neighbour that it and that fixel best align with, mutually, within 20°. Links are
taken best-aligned first (Kruskal), and two populations merge only if the merged one would hold two fixels of one
position at no more than `max(2, 2%)` of the smaller's positions: a population follows its bends, but chains of
in-between directions (unresolved crossing positions, noise) cannot join two populations that cross, while a few
noisy fixels cannot keep one population in pieces. Where a population still holds two fixels of a position the weaker
loses its label. Populations are numbered 1, 2, 3, … by size. Over a chunked volume, `volume.stitch_labels` joins the
labels across chunk faces by the same alignment rule.

```python
model = Orientations(scale=10.0)
pop = model.populations(v)        # N, mass, share, dirs_cells, presence, weight, fodf, sheet, labels_cells, table
```
