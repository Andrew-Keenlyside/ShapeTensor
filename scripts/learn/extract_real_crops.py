"""Pre-extract unlabelled real crops for the self-supervised term into one small file a cluster job copies to its node.

    python scripts/learn/extract_real_crops.py --src PATH@LEVEL [PATH@LEVEL ...] --out crops.npz [--n 256] [--size 112]

Crops with at least half tissue (background of either polarity found by ``realdata.background_mask``), normalised by
the tissue's robust range, uint8; the trainer cuts its training-size crops from them at random.
"""
import argparse

import numpy as np
import torch

from shapetensor.learn.lobe_train import RealBuffer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--size", type=int, default=112); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    b = RealBuffer(a.src, a.size, "cpu", n=a.n, seed=a.seed)
    crops = torch.stack(b.items).numpy()
    np.savez_compressed(a.out, crops=crops, sources=np.array(a.src))
    print(f"wrote {a.out}: {crops.shape} uint8 ({crops.nbytes / 2 ** 20:.0f} MB raw)")


if __name__ == "__main__":
    main()
