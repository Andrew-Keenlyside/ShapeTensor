#!/bin/bash
# Copy the ShapeTensors code and the training bundle (shape library, real crops) to pryor. Run locally, only with the
# user's go-ahead (it writes to the cluster's storage). ~350 MB.
set -euo pipefail
HOST=${HOST:-pryor.cs.ucl.ac.uk}
P=/home/akeenlys/storage_main/bridge_project_data/shapetensor
LIB=${LIB:-/home/andrew/bridge_wmcov/shapetensor/shape_library}
BUNDLE=${BUNDLE:-/home/andrew/bridge_wmcov/shapetensor/cluster_bundle}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
tar -cf "$BUNDLE/shape_library.tar" -C "$(dirname "$LIB")" "$(basename "$LIB")/neurons" "$(basename "$LIB")/surfaces"
ssh "$HOST" "mkdir -p $P/ShapeTensors $P/bundle $P/runs $P/bench"
rsync -a --delete --exclude .git --exclude '__pycache__' --exclude '*.pt' --exclude results --exclude notes "$REPO/src" "$REPO/scripts" "$REPO/pyproject.toml" "$HOST:$P/ShapeTensors/"
rsync -a --progress "$BUNDLE/shape_library.tar" "$BUNDLE"/hipct_*_crops.npz "$HOST:$P/bundle/"
echo "staged to $HOST:$P"
