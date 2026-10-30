#!/bin/bash
# The overnight rich-scene pilot (30-31 Oct 2026), unattended on one card: a pool of crops made with the whole card until
# POOL_END, training from it until TRAIN_END (SIGTERM: the trainer saves its checkpoint and stops), then the evaluation
# on the older scenes and on the rich ones (acceptance criteria, halfway vs final) and the populations figures. Times are
# US Eastern, today. Detach it (setsid nohup) so it outlives the shell that started it.
#
#   scripts/overnight.sh            env: NAME, RESULTS, POOL_END (02:10), TRAIN_END (06:30), SPS (expected s/step, 2.3),
#                                        EVAL_N (24), GPU (a CUDA_VISIBLE_DEVICES value)
set -u
cd "$(dirname "$0")/.."
NAME=${NAME:-pilot_rich_overnight}; R=${RESULTS:-/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/scene_v2}; E=$R/eval
POOL_END=${POOL_END:-02:10}; TRAIN_END=${TRAIN_END:-06:30}; SPS=${SPS:-2.3}; EVAL_N=${EVAL_N:-24}
GPU=${GPU:-GPU-924b8625-a187-4201-e24e-f41d91a63fd2}
PY=/home/andrew/.conda/envs/shapetensors/bin/python; OUT=$R/$NAME; POOL=$R/${NAME}_pool; LOG=$R/$NAME.driver.log
export PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU
mkdir -p "$E" "$POOL"
now() { TZ=America/New_York date +%H:%M; }
until_s() { echo $(( $(TZ=America/New_York date -d "$1" +%s) - $(date +%s) )); }
GEN='{"scene": {"canvas": 128, "crops_per_scene": 8, "cross_p": 0.5, "p_centre": 0.5}}'
WPOP='{"density":0,"mass":0,"region":0,"vote":0,"kappa":0,"desc":0,"tree":0,"overall":0,"fraction":0,"points":0,"contrast":0,"cross":0,"mvote":0,"bg":0,"obj":0,"split":0,"vote_unc":0,"tree_iso":0,"gate":0,"readout":0,"pop":4}'
FLAGS=(--head entities --size 64 --batch 4 --accum 2 --buffer 128 --refresh 1 --lr 3e-4 --warmup 100
       --val-n 16 --val-every 500 --log-every 25 --ckpt-minutes 10 --strength0 0.0 --strength1 1.0 --level0 0.4 --strength-mix
       --pairs 0 --dir-lmax 16 --vote-unc --deep-tree --lcn 3 --region-dec --e-cross 0.2 --kappa-log
       --vote-objects --own-objects --dens-same-norm --bg-head --obj-head --block-norm 1 --odirect 256 --od-sigmas 2 4 6 8 --split-head --mass-vote
       --od-norm --no-producer --gen "$GEN" --geometry-targets --pop-k 4 --pop-dir 96 --weights "$WPOP"
       --pop-seq 3 --pop-tie 1.0 --matrix-p 1 --scales random)
{
echo "=== $NAME started $(now) ET: pool until $POOL_END, training until $TRAIN_END ($(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader | grep ${GPU#GPU-}))"

# 1. the pool: crops made with the run's own settings (strength and difficulty drawn per scene), shards of 16
t=$(until_s "$POOL_END")
if [ "$t" -gt 60 ]; then
  timeout -s TERM "$t" $PY -m shapetensor.learn.train "${FLAGS[@]}" --out $R/${NAME}_poolmaker --steps 1 --make-pool 1000000 --pool $POOL \
      >> $R/${NAME}_poolmaker.log 2>&1
fi
echo "=== pool $(now) ET: $(ls $POOL | wc -l) shards ($(du -sh $POOL | cut -f1)); $(grep -v Warning $R/${NAME}_poolmaker.log | grep '^pool:' | tail -1)"

if [ "$(ls $POOL | wc -l)" -eq 0 ]; then echo "=== no pool shards: stopping"; exit 1; fi

# 2. training from the pool until TRAIN_END (steps sized so the cosine schedule ends about then); the halfway checkpoint kept
t=$(until_s "$TRAIN_END"); STEPS=$(( t * 10 / ${SPS/./} ))
echo "=== training $(now) ET: $STEPS steps planned ($t s at $SPS s/step)"
timeout -s TERM "$t" $PY -m shapetensor.learn.train "${FLAGS[@]}" --out $OUT --steps $STEPS --resume auto --pool $POOL >> $OUT.log 2>&1 &
TP=$!
while kill -0 $TP 2>/dev/null; do
  if [ ! -f $OUT/model_half.pt ] && [ -f $OUT/model.pt ]; then
    st=$($PY -c "import torch; print(torch.load('$OUT/model.pt', map_location='cpu', weights_only=False).get('step', 0))" 2>/dev/null || echo 0)
    [ "${st:-0}" -ge $((STEPS / 2)) ] && cp $OUT/model.pt $OUT/model_half.pt && echo "halfway checkpoint at step $st ($(now) ET)"
  fi
  sleep 60
done
wait $TP; echo "=== trained $(now) ET (exit $?): $(grep -v Warning $OUT.log | grep -E '^step [0-9]+' | tail -1 | cut -c1-120)"
grep -v Warning $OUT.log | grep -E "^  val [0-9]|Traceback" | sed -E 's/(val_pop_T [^ ]*).*/\1/' | cut -c1-200 | tail -8

# 3. evaluation: the older scenes (comparable with the earlier pilots) and the rich ones; halfway vs final; figures
for kind in old rich; do
  M=0; [ $kind = rich ] && M=1
  for pair in model:final model_half:half; do
    ck=${pair%%:*}; tag=${pair##*:}
    [ -f $OUT/$ck.pt ] && MATRIX=$M OUT=$E/${NAME}_${kind}_${tag}.json $PY scripts/eval_populations.py $OUT/$ck.pt $EVAL_N 2>&1 | grep -v Warning | tail -3
  done
  echo "--- acceptance, $kind scenes:"
  $PY scripts/acceptance.py $E/${NAME}_${kind}_final.json $E/${NAME}_${kind}_half.json $OUT.log 2>&1 | grep -v Warning
  MATRIX=$M $PY scripts/figure_populations.py $OUT/model.pt $E/${NAME}_${kind}_populations.png 2>&1 | grep -v Warning | tail -1
done
echo "=== $NAME done $(now) ET"
} >> $LOG 2>&1
