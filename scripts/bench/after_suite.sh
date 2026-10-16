#!/bin/bash
# Follow-ups after the full suite: the PCA variants, the order-12 CSD check, the real-crop demonstration, the report.
cd /tmp/claude-1001/-home-andrew-scripts-BRIDGE/2d1436b8-07a9-415d-b480-8d272c810dd5/scratchpad/bench
PY=~/.conda/envs/shapetensors/bin/python
export CUDA_VISIBLE_DEVICES=0
echo "=== after_suite start $(date) ==="
for s in spec_var_pcs.json spec_var_pcs8.json; do
  $PY run_full.py $s > ${s%.json}.log 2>&1; echo "--- $s:"; grep "single P3" ${s%.json}.log | tail -6
done
$PY run_lmax12.py > lmax12.log 2>&1; echo "--- lmax12:"; grep "lmax12\|Traceback" lmax12.log | tail -24
cd /home/andrew/scripts/ShapeTensors && $PY scripts/demo_real_crop.py --rho 16 --core 384 --reduce 4 > /tmp/claude-1001/-home-andrew-scripts-BRIDGE/2d1436b8-07a9-415d-b480-8d272c810dd5/scratchpad/bench/demo.log 2>&1
echo "--- demo:"; grep -v Warning /tmp/claude-1001/-home-andrew-scripts-BRIDGE/2d1436b8-07a9-415d-b480-8d272c810dd5/scratchpad/bench/demo.log | tail -6
$PY -c "from shapetensor.validate import report; print(report.write('main'))"
echo "=== after_suite done $(date) ==="
