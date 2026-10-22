# Installation

ShapeTensor is a Python package (3.11+) on PyTorch with CUDA. Everything computes on the GPU.

```bash
git clone https://github.com/Andrew-Keenlyside/ShapeTensor.git
cd ShapeTensor
pip install -e .                 # numpy, scipy, torch, zarr >= 3, numba, matplotlib
pip install -e ".[dev]"          # pytest, mkdocs
pip install -e ".[anatomix]"     # only for the 0.2 pipeline's shape-feature network (anatomix, monai)
```

Install a CUDA build of torch matching your driver first if `pip` would not pick one (for example
`pip install torch --index-url https://download.pytorch.org/whl/cu124`). The chain itself has no network and no
weights to download.

## Memory and speed

The chain on a 256³ block peaks at a few GB; a 384³ chunk with its halo (496³) at about 10 GB. `shapetensor bench`
reports the times and the peak memory on your GPU. Everything runs on the CPU as well (`device="cpu"`), slowly; the
tests do.

## Container

`envs/shapetensor-gpu.def` builds an Apptainer image (PyTorch 2.5.1 / CUDA 12.4, the weights baked in):

```bash
apptainer build --fakeroot shapetensor-gpu.sif envs/shapetensor-gpu.def
apptainer exec --nv shapetensor-gpu.sif shapetensor run in.ome.zarr out.zarr --level 0 --scale 10
```

## Data root

Phantoms and results of the validation go under `SHAPETENSOR_DATA` (default `~/shapetensor_data`).

## Tests

```bash
pytest tests          # the structure tensor against scipy; the chain's stages on small fields; chunked == one pass (CPU)
shapetensor demo      # the install check: a band-crossing phantom through the chain, tracked
```

The chain is deterministic: the same block and parameters give the same field.
