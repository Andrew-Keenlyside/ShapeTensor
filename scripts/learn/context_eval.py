"""How much context the model uses: outputs with the input masked beyond a half-width B around the scored cells.

    python scripts/learn/context_eval.py --run DIR [--n 8] [--out FILE]

On held-out draws (the run's generator, each under a fixed cube rotation): the central 4³ block of output cells
(16 voxels) is scored; the input outside a cube of half-width 8 + B voxels around it (B = 4 ... 28) is replaced by the
draw's mean intensity. Per B: the mixture head's axis error against the target on line cells, and its change from the
full-context output; with an SH head, the top density peak likewise. The curve flattens at the context the model
actually uses (its receptive field bounds it).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn.generator import eig3
from shapetensor.learn.losses import augment
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.sh import peaks_gated
from shapetensor.learn.train import make, stack

DEV = "cuda"
BS = (4, 8, 12, 16, 20, 24, 28)


def ang(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--out")
    a = ap.parse_args(); run = Path(a.run)
    ck = torch.load(run / "model.pt", map_location=DEV); args = ck["args"]
    net = PopulationNet.from_args(args).to(DEV); net.load_state_dict(ck["model"]); net.eval()
    size = args["size"]; gen = args.get("gen"); sg = args.get("strength1", 1.0); lmax = int(args.get("sh_lmax", 0) or 0)
    c0 = size // 2 - 8; c1 = c0 + 16; k0, k1 = c0 // 4, c1 // 4
    res = {B: {"mix_t": [], "mix_d": [], "sh_t": [], "sh_d": []} for B in BS}
    for i in range(a.n):
        img, t = augment(*make(10_000_000 + i, size, DEV, sg, gen), torch.Generator().manual_seed(i))
        x, T = stack([(img, t)], aug=False)
        tv, tV = eig3(T["slots"][0, 0].float()[:, k0:k1, k0:k1, k0:k1])                                # the heaviest target slot
        lin = (tv[2] - tv[1]) / tv[2].clamp_min(1e-9)
        line = (T["kind"][0, 0][k0:k1, k0:k1, k0:k1] == 0) & (T["mass"][0, 0].float()[k0:k1, k0:k1, k0:k1] > 0.05) & (lin > 0.5)
        if not bool(line.any()):
            continue
        ta = tV[2][:, line]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P0 = net(x)
        _, Vm0 = eig3(P0["mixture"][0].float()[:, k0:k1, k0:k1, k0:k1]); m0 = Vm0[2][:, line]
        s0 = peaks_gated(P0["sh"][0].float()[:, k0:k1, k0:k1, k0:k1].flatten(1)[:, line.flatten()], lmax, nmax=1)[0][0] if lmax else None
        mean = x.mean()
        for B in BS:
            lo, hi = max(c0 - B, 0), min(c1 + B, size)
            xm = torch.full_like(x, float(mean)); xm[..., lo:hi, lo:hi, lo:hi] = x[..., lo:hi, lo:hi, lo:hi]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                P = net(xm)
            _, Vm = eig3(P["mixture"][0].float()[:, k0:k1, k0:k1, k0:k1]); mb = Vm[2][:, line]
            res[B]["mix_t"].append(ang(mb, ta)); res[B]["mix_d"].append(ang(mb, m0))
            if lmax:
                sb = peaks_gated(P["sh"][0].float()[:, k0:k1, k0:k1, k0:k1].flatten(1)[:, line.flatten()], lmax, nmax=1)[0][0]
                res[B]["sh_t"].append(ang(sb, ta)); res[B]["sh_d"].append(ang(sb, s0))
    med = lambda L: float(torch.cat(L).median()) if L else float("nan")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")
    for key, lab, mk in (("mix", "mixture head", "o-"), ("sh", "SH density top peak", "s-")):
        if key == "sh" and not lmax:
            continue
        ax[0].plot([8 + B for B in BS], [med(res[B][f"{key}_t"]) for B in BS], mk, label=lab)
        ax[1].plot([8 + B for B in BS], [med(res[B][f"{key}_d"]) for B in BS], mk, label=lab)
    for B in BS:
        print(f"half-width {8 + B:3d} vox: mixture vs target {med(res[B]['mix_t']):5.1f}, change from full context {med(res[B]['mix_d']):5.1f}"
              + (f"; SH peak vs target {med(res[B]['sh_t']):5.1f}, change {med(res[B]['sh_d']):5.1f}" if lmax else ""), flush=True)
    rf = ck.get("receptive_field")
    for a_ in ax:
        a_.set_xlabel("visible half-width around the scored block (voxels)"); a_.legend()
        if rf:
            a_.axvline(8 + rf / 2, color="#999", ls=":"); a_.text(8 + rf / 2, a_.get_ylim()[1] * 0.9, " RF/2", color="#777")
    ax[0].set_ylabel("axis error vs target (deg)"); ax[1].set_ylabel("change from full-context output (deg)")
    fig.suptitle(f"Context used by {run.name} (step {ck['step']})")
    out = Path(a.out) if a.out else run / "context_eval.png"
    plt.tight_layout(); fig.savefig(out, dpi=70); print("wrote", out)


if __name__ == "__main__":
    main()
