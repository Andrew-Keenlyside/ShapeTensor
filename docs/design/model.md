# Model and readout

> **Status.** Implemented: the trunk; point-and-window queries; the populations head (one elliptical ODF per
> population, with its share and existence). Designed, not yet built: grouping levels as a query input and
> higher-order ODFs per population. The model does not yet beat the structure tensor on held-out data; see the
> experiment log in `notes/core_model_design_2026-10-29.md`.

## Shape latent

A 3D convolutional trunk maps the input volume to multi-scale features once. Its job is to identify objects and how
they group.

## Queries

A light decoder reads the latent at any query:

| knob | meaning |
|---|---|
| point | any position, so any output grid or voxel size |
| window | the spatial pooling scale in voxels, like the integration scale of a structure tensor. Small windows describe local orientation and give sharp spatial transitions between neighbouring populations; large windows describe a wider region and give gradual transitions. Arbitrary in training, set from the known voxel size at use. |
| grouping level | how coarse a population is: single objects, small groups, larger groups, whole populations (designed) |
| ODF order | the angular detail of each population's ODF (order 2 now; higher orders designed) |

### Grouping

Groups form a **nested partition** of the objects. At every level each object belongs to exactly one group. Objects
may cross and overlap in space, and so may groups, but two groups never claim the same object. The generator's
construction tree supplies this hierarchy directly. Today the grouping follows a resolution rule: at window *w* an
object is its own population when its radius is at least about *w* / 2, and otherwise merges into its parent.

## Outputs

For each query, up to K = 4 populations plus the background:

| output | form |
|---|---|
| ODF | a trace-1 elliptical ODF (a 3 x 3 second moment): a needle for a line, a disc for a sheet, a sphere for isotropic material |
| share | its share of the material in the window |
| existence | whether the slot holds a population (a slot exists when its share passes 3 %) |
| background | the share of the isotropic background, reported apart from the populations |

The ODF of one kind of object, for example only the line-like populations, is the share-weighted sum over the
populations of that shape.

## How a population's ODF is formed

Each slot's ellipse is the second moment of learned, non-negative weights over a set of 96 directions (a hemisphere,
randomly rotated in training). The ellipse therefore turns with the image by construction; read from a single feature
vector instead, the head could not learn orientation once training crops were rotated.

Each direction is scored by a small network from:

- the query's learned point code;
- the image's evidence along that direction: the raw energies of a fixed bank of oriented line filters (several
  lengths of pooling, and optionally several cross-section widths), and optionally a learned probe that samples the
  image and learned trunk channels along and around the direction.

The fixed filter bank supplies raw directional energies only. It never estimates an orientation itself, and no
structure tensor or Hessian enters the model. Turning the energies into populations is learned.

The slots are filled in turn. Each slot sees how much of every direction the earlier slots already claim, so two
fibres at a crossing go to two slots instead of being averaged into one disc. Shares come by stick breaking: the
background first, then each slot a fraction of what remains; the remainder is "other".

## Training

- **Set matching.** Each query's predicted slots are matched to its target populations by the cheapest of all K!
  assignments, with the same weights as the loss.
- **Loss.** Existence (binary cross-entropy), shares (L1, including the background and "other") and, for every
  real population, the squared Frobenius distance between ellipses. Each term is minimised by its conditional mean,
  so an ambiguous image is answered with a broad ellipse rather than a confident wrong one.
- **Sampling.** Every query draws its window at random; the targets are computed for exactly that window.
