#!/bin/bash
# Train the populations model on synthetic data drawn on the fly (the pilot configuration).
#   bash scripts/train.sh OUT_DIR [STEPS] [extra trainer flags...]
# Environment: GPU (a CUDA_VISIBLE_DEVICES value, default 0), WORKERS (CPU data workers, default 20).
set -eu
OUT=${1:?output directory}; STEPS=${2:-1500}; shift $(( $# < 2 ? $# : 2 ))
GEN='{"scene": {"canvas": 128, "crops_per_scene": 8, "cross_p": 0.5, "p_centre": 0.5}}'
WPOP='{"density":0,"mass":0,"region":0,"vote":0,"kappa":0,"desc":0,"tree":0,"overall":0,"fraction":0,"points":0,"contrast":0,"cross":0,"mvote":0,"bg":0,"obj":0,"split":0,"vote_unc":0,"tree_iso":0,"gate":0,"readout":0,"pop":4}'
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src python -m shapetensor.learn.train \
    --head entities --out "$OUT" --steps "$STEPS" --size 64 --batch 4 --accum 2 --buffer 128 --refresh 1 --lr 3e-4 --warmup 100 \
    --val-n 16 --val-every 300 --log-every 25 --ckpt-minutes 4 --resume auto \
    --strength0 0.0 --strength1 1.0 --level0 0.4 --strength-mix \
    --pairs 0 --dir-lmax 16 --vote-unc --deep-tree --lcn 3 --region-dec --e-cross 0.2 --kappa-log \
    --vote-objects --own-objects --dens-same-norm --bg-head --obj-head --block-norm 1 --odirect 256 --od-sigmas 2 4 6 8 \
    --split-head --mass-vote --od-norm --no-producer --cpu-workers "${WORKERS:-20}" --cpu-threads 3 --gen "$GEN" \
    --geometry-targets --pop-k 4 --pop-dir 96 --pop-seq 3 --pop-tie 1.0 --weights "$WPOP" "$@"
