# The pipeline, stage by stage

![pipeline](../img/pipeline_overview.png)

The pipeline as drawn has four stages: shape features are computed from the image, the shape space is separated
into fibre populations, a tensor is fitted per population, and the per-population vector fields are merged into one
multi-vector field for a streamline builder. The four pages that follow explain each stage: what it is meant to
do, what the package implements for it, what the phantom suite found when the stage was tested, and — where the
evidence changed the stage's form — what it became.

| stage | as drawn | as built | the page |
|---|---|---|---|
| 1 | anatomix features per voxel | three feature engines (intensity, jet, anatomix) behind one interface, with a per-volume PCA | [Getting the shape features](shape_features.md) |
| 2 | cluster the shape space into populations | separate in **orientation** space: the gradient-direction ODF deconvolved into populations; the shape features serve as a gradient source | [Separating populations](separating.md) |
| 3 | TCA per population | the channel-resolved tensor and its CP decomposition for one direction; CSD peaks for N directions with fractions; the excised intensity tensor as the single-direction field | [Getting tensors from populations](tensors_from_populations.md) |
| 4 | merge the population fields | a set of directions with fractions per site, gated at interfaces, with the tensor's gap as the "no population" verdict; several stains add | [Merging population vector fields](merging.md) |

The [walkthrough](../walkthrough.md) runs all four on one block with a figure and a validation number per step;
the [results](../results.md) page and the [results log](../research/TCA_FIELD_RESULTS.md) hold every number quoted.
