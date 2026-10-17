# Installation

ShapeTensor is a Python package (3.11+) on PyTorch with CUDA. Everything computes on the GPU.

```bash
git clone https://github.com/Andrew-Keenlyside/ShapeTensor.git
cd ShapeTensor
pip install -e .                 # numpy, scipy, torch, zarr >= 3, matplotlib
pip install -e ".[anatomix]"     # the shape-feature network (anatomix, monai); weights come from the HF hub on first use
pip install -e ".[dev]"          # pytest
```

Install a CUDA build of torch matching your driver first if `pip` would not pick one (for example
`pip install torch --index-url https://download.pytorch.org/whl/cu124`). anatomix's package metadata pins a newer
torch than it needs; if the resolver complains, `pip install --no-deps git+https://github.com/neel-dey/anatomix.git
huggingface_hub`. The weights (`neeldey/anatomix`) are cached under `HF_HOME`.

## Memory

The feature engine runs the network on tiles of 256³ with 64 voxels of context; a pass needs about 10 GB at the
encoder layer the pipeline reads. The pipeline on a 192³ block peaks at about 10 GB, on 256³ at about 22 GB. On a
24 GB card use `--chunk 128 128 128 --calibration-size 192`; `shapetensor bench` reports the times and the peak
memory on your GPU. There is no CPU fallback.

## Container

`envs/shapetensor-gpu.def` builds an Apptainer image (PyTorch 2.5.1 / CUDA 12.4, the weights baked in):

```bash
apptainer build --fakeroot shapetensor-gpu.sif envs/shapetensor-gpu.def
apptainer exec --nv shapetensor-gpu.sif shapetensor run in.ome.zarr out.zarr --level 0 --rho 10
```

## Data root

Phantoms and results of the validation go under `SHAPETENSOR_DATA` (default `~/shapetensor_data`).

## Tests

```bash
pytest tests          # the structure tensor against scipy; the pipeline's pieces; the whole pipeline on a phantom (GPU)
shapetensor demo      # the install check: a crossing phantom through the whole pipeline
```

Seeds are fixed, but the readout's training steps on the GPU are not bit-reproducible, so a borderline population
decision can differ between two runs on the same block.
