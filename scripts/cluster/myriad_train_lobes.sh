#!/bin/bash -l
# The lobe-mixture model's long training on Myriad (UCL RC, SGE), one GPU, resumable.
# NOT submitted automatically: qsub only after an explicit go-ahead.
#
#   qsub -v SEED=0,STEPS=100000 scripts/cluster/myriad_train_lobes.sh
#
# Expects: the ShapeTensors repo at $REPO, a Python env with torch at $PY. Training is purely synthetic.
#$ -N st_lobes
#$ -l h_rt=48:00:00
#$ -l mem=12G
#$ -l gpu=1
#$ -pe smp 8
#$ -l tmpfs=50G
#$ -cwd
#$ -j y

set -euo pipefail
REPO=${REPO:-$HOME/ShapeTensors}
PY=${PY:-$HOME/.conda/envs/shapetensors/bin/python}
SEED=${SEED:-0}
STEPS=${STEPS:-100000}
OUT=${OUT:-$HOME/Scratch/shapetensor/lobes_s${SEED}}
mkdir -p "$OUT"
cd "$REPO"
RESUME=""
[ -f "$OUT/model.pt" ] && RESUME="--resume $OUT/model.pt"
nvidia-smi
PYTHONPATH=src "$PY" -m shapetensor.learn.train --head lobes --out "$OUT" --seed "$SEED" --steps "$STEPS" \
  --size 128 --batch 4 --accum 4 --buffer 256 --refresh 2 --gate 0.85 --gate-max 3000 --difficulty auto --level0 0.3 \
  --pairs 32 --pair-every 4  --query 128 --strength1 1.0 --val-n 32 --val-every 2000 \
  --gen '{"scene": {"canvas": 256, "crops_per_scene": 8}}' $RESUME
