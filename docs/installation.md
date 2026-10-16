# Installation

ShapeTensor is a Python package (3.11+) built on PyTorch with CUDA. Everything computes on the GPU; there is no CPU
fallback for volumes of any size.

## From source

```bash
git clone https://github.com/Andrew-Keenlyside/ShapeTensor.git
cd ShapeTensor
pip install -e .                 # numpy, scipy, torch, zarr >= 3, nibabel, matplotlib, tifffile
```

Install a CUDA build of torch that matches your driver first if `pip` would not pick one (for example
`pip install torch --index-url https://download.pytorch.org/whl/cu124` for CUDA 12.4 drivers).

## The shape-feature engines

The `anatomix16` and `anatomix48` engines need the pretrained anatomix UNet:

```bash
pip install -e ".[anatomix]"     # anatomix (from GitHub) and monai
```

The weights (`neeldey/anatomix`) are fetched from the Hugging Face hub on first use and cached under `HF_HOME`.
anatomix's package metadata pins a newer torch than the one it needs; if the resolver complains, install it without
dependencies: `pip install --no-deps git+https://github.com/neel-dey/anatomix.git huggingface_hub`.

The `intensity` and `jet` engines need nothing beyond the base install.

## Container (cluster)

`envs/shapetensor-gpu.def` builds an Apptainer image from the PyTorch 2.5.1 / CUDA 12.4 runtime (drivers ≥ 550),
with the anatomix weights baked in so compute nodes need no network:

```bash
apptainer build --fakeroot shapetensor-gpu.sif envs/shapetensor-gpu.def
apptainer exec --nv shapetensor-gpu.sif shapetensor volume in.ome.zarr --level 0 out.zarr --rho 16
```

`scripts/cluster/shapetensor_sge.sh` is a Grid Engine job that runs one process per GPU of a node over a zarr
volume, splitting the chunk list by rank.

## Data root

Phantoms, caches, results and figures of the validation suite go under `SHAPETENSOR_DATA` (an environment variable;
the default is the author's data directory).

## Tests

```bash
python tests/test_core.py       # filters against scipy, K against the structure tensor, spherical harmonics, CP
python tests/test_st.py         # the volume API: layout, eigen conventions, excision, fixels, the chunked driver
python tests/test_anatomix.py   # the anatomix engine on the GPU: tiling and de-checkerboarding at the read-out level
```

`pytest` runs them too; each file also runs as a script.
