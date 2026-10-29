<p align="center"><img src="docs/img/shapetensor_logo.png" alt="ShapeTensor" width="300"></p>

# ShapeTensor

**Object-level orientation distributions from any intensity volume.**

ShapeTensor maps a 3D image to a shape latent, from which the orientation distribution functions (ODFs) of the
objects present at any point can be read: per object or per group of objects, at the window and grouping level you
choose. It is trained only on synthetic volumes whose orientations are known by construction, with contrast,
resolution and artefacts randomised far beyond any single imaging modality, so it learns shape rather than
appearance.

<p align="center"><img src="docs/img/example_orientations.png" alt="A synthetic crop, its ground-truth ODFs and the model's ODFs" width="900"></p>
<p align="center"><em>A synthetic crop (middle slice), its ground-truth ODFs from construction, and the ODFs of an
earlier version of the model (a single density readout, before the per-population head). Glyphs are spherical-harmonic
ODFs coloured by direction (red x, green y, blue z).</em></p>

> **Status: research code, not yet a working orientation model.** The generator and its per-object targets are built
> and verified against construction. The populations head learns lines, sheets and crossings on short structural
> tests, but does not yet beat the structure tensor on held-out data. Progress and every experiment are logged in
> `notes/core_model_design_2026-10-29.md`.

## Install

```bash
pip install -e ".[dev,figures]"
pytest -q tests
```

## Use

```bash
# train the populations model on synthetic data drawn on the fly
bash scripts/train.sh runs/model 1500

# score it on held-out synthetic crops and known-truth phantoms, then check the acceptance criteria
PYTHONPATH=src OUT=runs/final.json python scripts/eval_populations.py runs/model/model.pt 24
PYTHONPATH=src python scripts/figure_populations.py runs/model/model.pt runs/populations.png

# a minutes-long structural test of the head (fits fixed crops under rotations)
PYTHONPATH=src python scripts/replicate_pops.py ITEMS.pt --set pop_dir=96 pop_seq=3 --et POP_TIE=1.0
```

## Design

- [Principles](docs/design/principles.md)
- [Model and readout](docs/design/model.md)
- [Synthetic data](docs/design/data.md)
- [Evaluation](docs/design/evaluation.md)

The classical structure-tensor pipeline (version 0.3), now the benchmark the model has to beat, is still in
`src/shapetensor` and documented in `docs/` (see `docs/archive/` for its design).

## Licence

MIT
