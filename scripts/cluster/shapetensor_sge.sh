#!/bin/bash
# shapetensor on a Grid Engine GPU node: structure-tensor-compatible outputs (S, eigenvectors, the excised tensor,
# fixels) for a whole zarr volume, one process per GPU, chunks split by rank. Edit SETTINGS, then: qsub scripts/cluster/shapetensor_sge.sh
#
#   qsub -v SRC=/path/volume.ome.zarr,LEVEL=0,DST=/path/out.zarr scripts/cluster/shapetensor_sge.sh
#
# Resubmitting after a walltime kill is safe: chunks already written are recomputed (idempotent writes).

#$ -N shapetensor
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -o shapetensor.$JOB_ID.log
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 4
#$ -l tmem=64G
#$ -l h_rt=24:00:00
#$ -R y

set -euo pipefail

# ============================== SETTINGS ===================================
SIF="${SIF:-/path/to/shapetensor-gpu.sif}"       # built from envs/shapetensor-gpu.def
SRC="${SRC:?set SRC=/path/volume.zarr (an array, or an OME-Zarr group with LEVEL)}"
LEVEL="${LEVEL:-}"
DST="${DST:?set DST=/path/out.zarr}"
SIGMA="${SIGMA:-1.0}"
RHO="${RHO:-16}"                                  # voxels: 16 = 0.32 mm at 20 um
CHUNK="${CHUNK:-256 256 256}"
OUTPUTS="${OUTPUTS:-S eig excised fixels}"
NGPU="${NGPU:-${NSLOTS:-1}}"
BIND="${BIND:-$(dirname "$SRC"),$(dirname "$DST")}"
# ===========================================================================

echo "host $(hostname)  gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
LEVEL_ARG=""; [ -n "$LEVEL" ] && LEVEL_ARG="--level $LEVEL"
pids=()
for r in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$r apptainer exec --nv -B "$BIND" "$SIF" \
    shapetensor volume "$SRC" "$DST" $LEVEL_ARG --sigma "$SIGMA" --rho "$RHO" --chunk $CHUNK \
    --outputs $OUTPUTS --rank "$r" --world "$NGPU" > "shapetensor.$JOB_ID.rank$r.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "done, status $status"
exit $status
