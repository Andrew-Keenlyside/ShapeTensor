#!/bin/bash -l
# A long entity-model pilot on UCL Myriad (SGE). NOT submitted automatically: check the paths, then `qsub` it only with
# the user's explicit go-ahead for this job (cluster rule). One GPU, 48 h; resumes from OUT/model.pt (--resume auto).
#$ -N st_entity_pilot
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -pe smp 8
#$ -wd /home/USER/Scratch/shapetensors          # <- check
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -j y

set -euo pipefail
ENV_PY=/home/USER/.conda/envs/shapetensors/bin/python          # <- check: the cluster's env
REPO=/home/USER/Scratch/ShapeTensors                            # <- check: the repo checkout (branch population-model)
OUT=/home/USER/Scratch/shapetensors/results/entity_pilot_$JOB_ID
INIT=${INIT:-}                                                   # optional warm start (a phantom-family or generator checkpoint)

cd "$REPO"
GEN='{"scene": {"canvas": 128, "crops_per_scene": 8, "cross_p": 0.6, "p_centre": 0.7}}'
PYTHONPATH=src "$ENV_PY" -m shapetensor.learn.train --head entities --steps 60000 --size 64 --batch 8 --accum 1 --buffer 256 \
    --refresh 1 --lr 3e-4 --warmup 500 --val-n 32 --val-every 1000 --log-every 50 --pairs 16 --pair-every 4 --no-producer \
    --cpu-workers 7 --cpu-threads 1 --dir-lmax 16 --vote-unc --e-sigma 1 6 --deep-tree --lcn 3 --resume auto \
    ${INIT:+--init "$INIT"} --level0 0.4 --region-dec --e-cross 0.2 --gen "$GEN" \
    --kappa-log --vote-objects --own-objects --dens-same-norm --bg-head --obj-head \
    --out "$OUT"
# (orientation visibility from oriented texture and the entity dispersion floor are on by default since 2026-10-27)
