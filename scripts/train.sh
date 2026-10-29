#!/bin/bash
# A short local training run on synthetic data drawn on the fly.
#   bash scripts/train.sh OUT_DIR [STEPS] [INIT_CHECKPOINT]
# Environment: WORKERS (CPU data workers, default 16), GPU (default 0).
set -eu
OUT=${1:?output directory}; STEPS=${2:-600}; INIT=${3:-}
GEN='{"scene": {"canvas": 128, "crops_per_scene": 8, "cross_p": 0.5, "p_centre": 0.5}}'
INIT_ARG=""; [ -n "$INIT" ] && INIT_ARG="--init $INIT"
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src python -m shapetensor.learn.train \
    --head entities --out "$OUT" --steps "$STEPS" --size 64 --batch 4 --accum 2 --buffer 128 --refresh 1 --lr 3e-4 --warmup 50 \
    --val-n 16 --val-every 300 --log-every 25 --ckpt-minutes 5 --resume auto \
    --strength0 0.0 --strength1 1.0 --level0 0.4 --strength-mix \
    --pairs 16 --pair-every 4 --dir-lmax 16 --vote-unc --deep-tree --lcn 3 --region-dec --e-cross 0.2 --kappa-log \
    --vote-objects --own-objects --dens-same-norm --bg-head --obj-head --block-norm 1 --odirect 256 --od-sigmas 2 4 6 8 \
    --split-head --mass-vote --od-norm --no-producer --cpu-workers "${WORKERS:-16}" --cpu-threads 3 --gen "$GEN" \
    --geometry-targets --dens-ce --ce-no-st --weights '{"density":4}' $INIT_ARG
