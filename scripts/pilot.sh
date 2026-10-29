#!/bin/bash
# A pilot of the populations model (1500 steps, from scratch or resumed), then its evaluation against the acceptance
# criteria (notes/core_model_design_2026-10-29.md): halfway and final checkpoints scored by eval_populations.py,
# acceptance.py, and the populations figure. Detach it (setsid nohup) so it outlives the shell that started it.
#
#   scripts/pilot.sh NAME [extra trainer flags...]
# env: GPU (a CUDA_VISIBLE_DEVICES value), RESULTS (the results directory), EVAL (a directory for the scores and figures),
#      STEPS (default 1500)
set -u
NAME=$1; shift
cd "$(dirname "$0")/.."
GPU=${GPU:-0}; R=${RESULTS:-/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/scene_v2}; E=${EVAL:-$R/eval}; STEPS=${STEPS:-1500}
PY=/home/andrew/.conda/envs/shapetensors/bin/python; OUT=$R/$NAME
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$E"
GEN='{"scene": {"canvas": 128, "crops_per_scene": 8, "cross_p": 0.5, "p_centre": 0.5}}'
WPOP='{"density":0,"mass":0,"region":0,"vote":0,"kappa":0,"desc":0,"tree":0,"overall":0,"fraction":0,"points":0,"contrast":0,"cross":0,"mvote":0,"bg":0,"obj":0,"split":0,"vote_unc":0,"tree_iso":0,"gate":0,"readout":0,"pop":4}'
echo "=== $NAME started $(TZ=America/New_York date +%H:%M) flags: $*"
$PY -m shapetensor.learn.train \
    --head entities --out $OUT --steps $STEPS --size 64 --batch 4 --accum 2 --buffer 128 --refresh 1 --lr 3e-4 --warmup 100 \
    --val-n 16 --val-every 300 --log-every 25 --ckpt-minutes 4 --resume auto --strength0 0.0 --strength1 1.0 --level0 0.4 --strength-mix \
    --pairs 0 --dir-lmax 16 --vote-unc --deep-tree --lcn 3 --region-dec --e-cross 0.2 --kappa-log \
    --vote-objects --own-objects --dens-same-norm --bg-head --obj-head --block-norm 1 --odirect 256 --od-sigmas 2 4 6 8 --split-head --mass-vote \
    --od-norm --no-producer --cpu-workers 20 --cpu-threads 3 --gen "$GEN" --geometry-targets --pop-k 4 --pop-dir 96 --weights "$WPOP" "$@" \
    >> $OUT.log 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do                                                       # the halfway checkpoint, as it appears
  if [ ! -f $OUT/model_half.pt ]; then
    st=$($PY -c "import torch,sys; print(torch.load('$OUT/model.pt', map_location='cpu', weights_only=False).get('step', 0))" 2>/dev/null || echo 0)
    [ "${st:-0}" -ge $((STEPS / 2)) ] && cp $OUT/model.pt $OUT/model_half.pt && echo "halfway checkpoint at step $st"
  fi
  sleep 60
done
echo "=== $NAME trained $(TZ=America/New_York date +%H:%M)"; grep -v Warning $OUT.log | grep -E "^  val [0-9]|Traceback" | sed -E 's/(val_pop_T [^ ]*).*/\1/' | cut -c1-200 | tail -6
OUT=$E/${NAME}_final.json $PY scripts/eval_populations.py $OUT/model.pt 24 2>&1 | grep -v Warning | tail -12
[ -f $OUT/model_half.pt ] && OUT=$E/${NAME}_half.json $PY scripts/eval_populations.py $OUT/model_half.pt 24 > /dev/null 2>&1
$PY scripts/acceptance.py $E/${NAME}_final.json $E/${NAME}_half.json $OUT.log
$PY scripts/figure_populations.py $OUT/model.pt $E/${NAME}_populations.png 2>&1 | grep -v Warning | tail -1
echo "=== $NAME done $(TZ=America/New_York date +%H:%M)"
