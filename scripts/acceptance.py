"""The acceptance suite (``shapetensor validate``): see :mod:`shapetensor.validate`.

    python scripts/acceptance.py [--out DIR] [--seeds 1] [--cases band,bend] [--scale 10]
"""
from __future__ import annotations

import argparse

from shapetensor.validate import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/acceptance")
    ap.add_argument("--seeds", type=int, default=1); ap.add_argument("--cases", default=""); ap.add_argument("--scale", type=float, default=10.0)
    a = ap.parse_args()
    run(a.out, seeds=a.seeds, cases=a.cases, scale=a.scale)
