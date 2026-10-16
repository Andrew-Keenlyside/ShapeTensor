# The pipeline, stage by stage

![pipeline](../img/pipeline_overview.png)

The pipeline as drawn has four stages: shape features are computed from the image, the shape space is separated
into fibre populations, a tensor is fitted per population, and the per-population vector fields are merged into one
multi-vector field for a streamline builder. The four pages that follow explain each stage: what it is meant to
do, what the package implements for it, what the phantom suite found when the stage was tested, and what is being
changed *within* the stage to make it work.

| stage | what it does | where it stands | the page |
|---|---|---|---|
| 1 | shape features per voxel | three engines behind one interface; the orientation-carrying layer is encoder block 2, now exposed as `anatomix_enc2` | [Getting the shape features](shape_features.md) |
| 2 | separate the shape space into populations | the stage under construction: label-free separation in the orientation-selective shape space (TCA pieces / channel-signature memberships) | [Separating populations](separating.md) |
| 3 | one tensor per population | the channel-resolved tensor and its CP decomposition (TCA); the excised intensity tensor where there is one population | [Getting tensors from populations](tensors_from_populations.md) |
| 4 | merge the population fields | one set of directions per site for a streamline builder; several stains add | [Merging population vector fields](merging.md) |

!!! note "Archived alternative"
    An ODF / spherical-deconvolution route (`st.fixels_3d`, `readout.multi.n_by_peaks`) was built and validated
    alongside this pipeline and is kept in the package and the results log as an archived alternative. It is not
    part of the pipeline and is not developed further.

The [walkthrough](../walkthrough.md) runs all four on one block with a figure and a validation number per step;
the [results](../results.md) page and the [results log](../research/TCA_FIELD_RESULTS.md) hold every number quoted.
