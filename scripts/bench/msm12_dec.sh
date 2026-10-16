#!/bin/bash
# The real-data DEC demonstration on msm12 (RTX A2000 12 GB): a 160^3 core of the 20 um S-21-46 crop, anatomix in
# 128-voxel tiles with 64-voxel margins. Run on msm12 from the repo root, in the env with torch 2.5.1 + anatomix (chunkreg):
#   git pull && bash scripts/bench/msm12_dec.sh
set -euo pipefail
PY="${PY:-$HOME/.conda/envs/chunkreg/bin/python}"
SP=$($PY -c "import site; print(site.getsitepackages()[0])")
echo "$(pwd)/src" > "$SP/shapetensor.pth"
export SHAPETENSOR_DATA="${SHAPETENSOR_DATA:-$HOME/bridge_wmcov/shapetensor}"
mkdir -p "$SHAPETENSOR_DATA/fields"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" $PY scripts/demo_real_dec.py --core 160 --rho 16 --reduce 4 --tile 128 --margin 64 "$@"
