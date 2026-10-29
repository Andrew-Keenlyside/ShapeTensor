#!/bin/bash
# Pilots one after another (one 48 GB card cannot hold two pilots' validation draws): each line of QUEUE_FILE is
# "NAME flag flag ...", run with scripts/pilot.sh; lines are read as they are reached, so more can be appended.
#   setsid nohup scripts/pilot_queue.sh QUEUE_FILE &      (env GPU, RESULTS as for pilot.sh)
cd "$(dirname "$0")/.."
Q=$1; R=${RESULTS:-/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/scene_v2}; i=1
while line=$(sed -n "${i}p" "$Q"); [ -n "$line" ]; do
  set -- $line; name=$1; shift
  scripts/pilot.sh "$name" "$@" > "$R/$name.driver.log" 2>&1
  i=$((i + 1))
done
