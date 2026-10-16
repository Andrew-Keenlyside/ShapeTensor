# 2. Separating populations in shape space

## What the stage is for

Where two or more fibre populations share a window — a crossing, a kissing, a bundle passing through a fan — one
direction per site is wrong for all of them. The stage has to decide how many populations there are (N) and which
of the local evidence belongs to which, so that the next stage can fit one tensor per population.

## As drawn: cluster the shape space

The drawn pipeline separates in *feature space*: each voxel's shape-feature vector is a point in R^C; cluster the
points, and each cluster is a population. The assumption underneath is that a voxel's local appearance identifies
its population.

The suite tested this on the crossing phantoms (results log §7; `fig_shape_space_clusters.py`):

![shape space clusters](../img/shape_space_clusters.png)

- On a 60° crossing the two bundles' layer-58 feature vectors overlap completely; k-means clusters follow the
  texture phase and the background, not the bundles.
- A tensor on each cluster's voxels returns one bundle or the bisector.
- Projecting the features onto the channel signature of a TCA pair does not help either (signature |cos| 0.6–0.9
  between bundles).

The reason is structural rather than a matter of tuning: two crossing bundles are the same tissue with the same
stain response and the same texture statistics — they differ **only in orientation** — so a feature space can
separate them only if it encodes orientation, and the engines' output features are nearly orientation-invariant
(previous page). The clustering step has nothing to cluster.

## As built: separate in orientation space

Populations are defined by orientation, so the package separates them where orientation lives: on the sphere of
gradient directions. Per site, the gradient-direction ODF collects the directions of all gradients in the window
(weighted by |∇|²) in even spherical harmonics to order 8. One fibre population contributes a *ring* of gradients
in the plane normal to it; N populations contribute N rings. Deconvolving the ODF with the single-population ring
response (constrained spherical deconvolution) turns each ring into a peak at the fibre direction; the peaks
above a calibrated threshold, at least 25° apart, are the populations, with the peak heights as their fractions.

![from image to several directions](../img/tca_multi_explainer.png)

This is a mixture model in orientation space: N is read from the peaks, not chosen. The shape features enter as a
*gradient source*: the ODF of the `anatomix48` feature gradients is sharper than the intensity ODF at full texture,
which is what resolves tight crossings.

| crossing, full texture | intensity ODF | anatomix48 ODF |
|---|---|---|
| 90° | N = 2 at 100%, both within 15° at 97% (ρ 32) | 100% / 100% |
| 60° | 21% / 6% | **100% / 100%** (ρ 32), 98% / 93% (ρ 16) |
| 45° | 0 | 61% / 61% (order 12: 75%) |

The other N-selection methods tried on the same data — BIC over CP ranks, an over-complete CP grouping, a
residual-bootstrap stability filter — never select two populations, select two everywhere, or remove true ones
(results log §6); they are kept in `readout.multi` as the documented comparison.

## Two conditions found by the null tests

1. **The threshold must be set on dispersed single-fibre texture at the data's texture level**, not on straight
   lines (results log §14): the line-null threshold gives ~50% false second populations on dispersed tissue at half
   texture for anatomix48. At full texture rel 0.5–0.6 is compliant (≤ 2% false seconds) and keeps the 60° crossing;
   at half texture 0.7 (intensity, ρ 16) and ≈ 0.9 (anatomix48), under which only 90° crossings remain detectable.
2. **Contrast interfaces make false populations** (results log §13): a step, a gyral boundary or a vessel wall gives
   the ODF one strong peak, which the ring deconvolution turns into a ring of fibre directions — N ≥ 2 at 30–100% of
   sites within 0.5 mm. The fix is the interface gate of the next stage, not a smaller threshold.

## What would make the drawn stage work

Separation per voxel is possible in principle: encoder block 2 of the network is orientation-selective, and a
linear head on it reads a voxel's orientation at ±15–20° (73% within ±15° on held-out phantoms) — enough to split a
90° crossing by voting, not 60°. The sound form of "clusters" is spatial rather than feature-based: at 20 µm
fascicle boundaries are visible, so a fascicle instance segmentation (orientation-coherence watershed on the
excised tensor, or a learned fascicle map) would assign voxels to populations as connected fascicles, and a tensor
per fascicle would follow with no decomposition at all. That is the next study; today the ODF route is the
estimator.
