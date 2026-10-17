<div align="center">

<img src="images/shapetensor_logo.png" alt="ShapeTensor" width="45%" />

</div>

# ShapeTensor

Fibre orientation fields of high-resolution white-matter volumes (HiP-CT at 20 µm, serial histology, light-sheet)
from a pretrained shape-feature space, on the GPU. One pipeline, four stages:

1. **Shape features** — the block through a contrast-agnostic 3-D UNet (anatomix), read at the encoder layer
   whose channels encode fibre orientation.
2. **Separating populations in the shape space** — the orientation code is read without labels (the structure
   tensor at confident sites is the teacher); the populations present are found, and every voxel is unmixed onto
   their signatures: one **activation layer per population**, a full-volume image of that population's shape.
3. **One tensor per population** — the image's structure tensor with the gradients weighted by the population's
   activation.
4. **Merging** — one vector per voxel from the merged population tensors, or every population's vector with its
   activation as weight. No masks anywhere: the weights say where a population is present.

The classic structure tensor and its boundary-robust (*excised*) form are the baseline, at the same σ and ρ as
every tensor in the pipeline, so every comparison is like for like.

<div align="center">
<img src="docs/img/pipeline_overview.png" alt="the pipeline" width="100%" />
<img src="docs/img/overview_S2146.png" alt="the pipeline on a HiP-CT block" width="100%" />
</div>

The idea, and the same stages on a 192³ block of HiP-CT (S-21-46, 20 µm): the image, its voxels in the shape
space, the two populations found there, their activation layers, each population's field, and the combined fields.

## Install

```bash
git clone https://github.com/Andrew-Keenlyside/ShapeTensor.git && cd ShapeTensor
pip install -e ".[anatomix]"       # torch (CUDA), numpy, scipy, zarr >= 3, matplotlib; anatomix + monai (weights from the HF hub)
```

## Use

```python
import shapetensor as st

vec     = st.shapetensor(image, rho=10.0)                               # (3, Z, Y, X): one direction per voxel, (x, y, z)
vecs, w = st.shapetensor(image, rho=10.0, multi=True, weights=True)     # (N, 3, Z, Y, X) + (N, Z, Y, X): every population, by activation
vec     = st.structure_tensor(image, rho=10.0)                          # the excised structure tensor: the baseline
out     = st.population_field(image, rho=10.0)                          # modes, activation layers, both fields, masks, the null
```

```bash
shapetensor run brain.ome.zarr out.zarr --level 0 --rho 10        # a whole volume, chunked with halos, calibrated once
shapetensor structure-tensor brain.ome.zarr st.zarr --rho 10      # the baseline in the structure-tensor package's layout
shapetensor validate --out validation                              # the three phantoms at the texture of the real crop
```

`image` is a `(Z, Y, X)` numpy, cupy or torch array of grey levels; the result comes back as the same kind.
`rho` is the integration window in voxels (0.2 mm at 20 µm); `docs/parameters.md` shows what each parameter does to
the field. Documentation: `mkdocs serve` or the `docs/` pages.

## What the phantoms show

Three phantoms with truth, at 20 µm with the texture matched to a real HiP-CT crop (20° of fibre dispersion within
a population) — a single bundle, a 60° crossing and a bundle across a tissue step. Every figure in the package has
the same columns: the image, the excised structure tensor, the shape space with the populations found in it, one
activation layer per population, the shape tensor's single field, and the first vector of its multi-vector field.

<div align="center">
<img src="docs/img/validation_calibrated.png" alt="validation" width="100%" />
</div>

- At a **tissue step** the plain structure tensor is dragged 35° into the boundary; the excision removes the
  effect, and the pipeline inherits it (the excision is inside every population tensor).
- At a **crossing** the structure tensor reports the bisector, 24° from either population. The pipeline finds the
  two populations on every seed; the first vector of its multi-vector field is 11° from the nearest population at
  the median, the stronger population's own tensor at each voxel. The pair is not resolved per voxel at this
  dispersion: the second vector is indicative.
- On a **single bundle**, and on the real block where the structure tensor is confident, the single field
  reproduces the excised structure tensor.

The numbers are in `docs/validation.md`; the research behind every design decision, with all its numbers, in
`notes/`.

## Layout

```
src/shapetensor/
  __init__.py     shapetensor(), structure_tensor(), population_field()
  features/       the engines: anatomix_enc2 (default), anatomix48, anatomix16, jet, intensity
  separate.py     stage 2: the dictionary, the orientation readout, the unmixing, the null
  tensors.py      stage 3: the population tensor
  merge.py        stage 4: the single field, the multi-vector field, export
  pipeline.py     calibrate / apply / population_field / run_volume
  st.py           the structure tensor (plain, excised), eigen, the structure-tensor layout, run_volume
  phantoms.py     bundle | crossing | interface, with truth
  validate.py     the validation
  tracking.py     a deterministic streamline tracker over multi-vector grids
  viz.py, stats.py, filters.py, pca.py, cli.py
  archive/        the ODF / fixel and TCA routes (importable, not maintained)
tests/            pytest
notes/            the research log, review and plans
```
