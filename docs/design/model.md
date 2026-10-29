# Model and readout

> **Status.** Implemented: the trunk, point-and-window queries, and a normalised ODF head for all non-background
> material (cross-entropy on a fixed direction grid). In progress: the per-population set head (grouping levels,
> descriptors, embeddings) and spherical-harmonic export.

## Shape latent

A 3D convolutional trunk maps the input volume to multi-scale features once. Its job is to identify objects and how
they group. The trunk takes no orientation features computed by hand.

## Queries

A light decoder reads the latent at any query:

| knob | meaning |
|---|---|
| point | any position, so any output grid or voxel size |
| window | the spatial pooling scale in voxels, like the integration scale of a structure tensor. Small windows describe local orientation and give sharp spatial transitions between neighbouring populations; large windows describe a wider region and give gradual transitions. Arbitrary in training, set from the known voxel size at use. |
| grouping level | how coarse a population is: single objects, small groups, larger groups, whole populations |
| ODF order | the angular detail of each population's ODF |

Grouping level and order trade off at readout. You can ask for one high-order ODF of everything, for many order-2
ODFs (one direction and its anisotropy per object), or for anything in between.

### Grouping

Groups form a **nested partition** of the objects. At every level each object belongs to exactly one group. Objects
may cross and overlap in space, and so may groups, but two groups never claim the same object. The generator's
construction tree supplies this hierarchy directly.

## Outputs

For each population present at a query:

| output | form |
|---|---|
| existence | probability |
| mass | its share of the material in the window |
| ODF | a normalised, positive density over directions: spherical-harmonic channels up to the requested order |
| descriptors | learned object attributes (shape kind, radius, lumen, brightness, packing), supervised from construction |
| embedding | a population code, to group the same population across queries |

The isotropic remainder (background, unresolved noise) is reported separately. The ODF of one kind of object, for
example only the fibre-like populations, is the mass-weighted sum over the populations whose descriptors match.

The ODF head is a normalised density on a fixed direction grid, with a uniform floor for honest uncertainty, trained by
cross-entropy. Arrays are exported as spherical-harmonic channels in the MRtrix3 real even basis.

## Training

- **Set matching.** Hungarian matching of predicted populations to target populations.
- **Losses.**
  - ODF cross-entropy for matched populations;
  - mass, existence and descriptor terms;
  - cross-entropy of the window's total ODF;
  - an embedding loss across queries of the same crop.
- **Sampling.** Every query draws its window, grouping level and order at random. The targets are computed for exactly
  those values.
