"""QC of a training run: learning curves and, on held-out samples, the model's populations against the target's.

    python scripts/learn/qc_model.py --run DIR [--n 4] [--out FILE]

``DIR`` holds ``model.pt`` and ``log.jsonl`` (from ``python -m shapetensor.learn.train``). The held-out samples are
the run's own validation draws (seeds from 10^7, its generator arguments and final strength). Per sample, a z-x slice
through the middle: the image; the target's and the prediction's line-population count per cell (N, read by the engine
from the slots); the heaviest line population's axis (DEC) weighted by its mass; the mixture head's axis; the shape
fraction; the axis error of the heaviest population where the target is a line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn.engine import PopulationEngine
from shapetensor.learn.generator import eig3
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.train import make, stack
from shapetensor.viz import dec_rgb

DEV = "cuda"
NMAP = matplotlib.colors.ListedColormap(["#1b1b1b", "#2a78d6", "#e34948", "#f0b429", "#ffffff"])


def oracle(T):
    """The target's slots in the network's output form (one block, no batch dim): read by the same engine."""
    return {"slots": T["slots"][0], "mass": T["mass"][0], "root_mass": T["mass"][0].sqrt(), "presence_logit": torch.where(T["mass"][0] > 0.005, 10.0, -10.0),
            "kind": 30.0 * torch.nn.functional.one_hot(T["kind"][0].clamp_min(0), 4).movedim(-1, 1).float(),
            "embed": torch.zeros(T["mass"].shape[1], 8, *T["mass"].shape[2:], device=DEV),
            "fraction_logit": torch.logit(T["fraction"][0].clamp(1e-4, 1 - 1e-4)), "class_logits": torch.log(T["classes"][0] + 1e-8), "log_size": T["size"][0].clamp_min(0.3).log()}


def dec(axis, weight):
    return (dec_rgb(axis.cpu().numpy()) * weight.clamp(0, 1).cpu().numpy()[..., None]).clip(0, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=4); ap.add_argument("--out")
    a = ap.parse_args(); run = Path(a.run)
    ck = torch.load(run / "model.pt", map_location=DEV); args = ck["args"]
    net = PopulationNet().to(DEV); net.load_state_dict(ck["model"]); net.eval()
    eng = PopulationEngine(net, device=DEV)
    size, gen, sg = args["size"], args.get("gen"), args.get("strength1", 1.0)
    rows = [json.loads(l) for l in open(run / "log.jsonl")]
    tr = [r for r in rows if "total" in r]; va = [r for r in rows if "val_axis_deg_median" in r]
    fig = plt.figure(figsize=(30, 4.2 * (a.n + 1)), facecolor="white"); gs = fig.add_gridspec(a.n + 1, 10, hspace=0.35, wspace=0.12)
    # learning curves
    ax = fig.add_subplot(gs[0, 0:3])
    for k, c in (("total", "k"), ("shape", "#2a78d6"), ("mixture", "#7e3ff2"), ("mass", "#e34948"), ("presence", "#f0b429"), ("hellinger", "#1f9d55")):
        if tr and k in tr[0]:
            ax.plot([r["step"] for r in tr], [r[k] for r in tr], color=c, lw=1.5, label=k)
    ax.set_yscale("log"); ax.legend(fontsize=8); ax.set_title(f"training loss parts ({run.name})", fontsize=10); ax.set_xlabel("step")
    ax = fig.add_subplot(gs[0, 3:6])
    if va:
        ax.plot([r["step"] for r in va], [r["val_axis_deg_median"] for r in va], "o-", color="#2a78d6", label="slots: axis error (median)")
        if "val_mixture_axis_deg_median" in va[-1]:
            ax.plot([r["step"] for r in va if "val_mixture_axis_deg_median" in r], [r["val_mixture_axis_deg_median"] for r in va if "val_mixture_axis_deg_median" in r], "s-", color="#7e3ff2", label="mixture head: axis error")
    ax.axhline(57.3, color="#999", ls=":", lw=1); ax.text(0, 58.5, "chance ≈ 57°", fontsize=8, color="#777")
    ax.set_ylim(0, 80); ax.legend(fontsize=8); ax.set_title("held-out orientation (degrees)", fontsize=10); ax.set_xlabel("step")
    ax = fig.add_subplot(gs[0, 6:9])
    for k, c, lab in (("val_fraction_corr", "#e34948", "fraction correlation"), ("val_N_agree", "#1f9d55", "population count agreement")):
        if va and k in va[-1]:
            ax.plot([r["step"] for r in va if k in r], [r[k] for r in va if k in r], "o-", color=c, label=lab)
    ax.set_ylim(-0.1, 1.05); ax.legend(fontsize=8); ax.set_title("held-out presence (0-1)", fontsize=10); ax.set_xlabel("step")
    ax = fig.add_subplot(gs[0, 9]); ax.axis("off")
    ax.text(0, 1, f"checkpoint step {ck['step']}\nsize {size}³, strength {sg}\ngen {json.dumps(gen) if gen else 'full generator'}\n"
                  f"last val: axis {va[-1]['val_axis_deg_median']:.1f}°, mixture {va[-1].get('val_mixture_axis_deg_median', float('nan')):.1f}°" if va else "", fontsize=8, va="top", wrap=True)
    # held-out samples
    c = size // 8; vy = 4 * c + 2
    for i in range(a.n):
        img, t = make(10_000_000 + i, size, DEV, sg, gen)
        x, T = stack([(img, t)], aug=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = net(x)
        P = {k: v.float() for k, v in P.items()}
        pp = eng.populations({k: v[0] for k, v in P.items()}, labels=False); pt = eng.populations(oracle(T), labels=False)
        r = i + 1
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(img[:, vy].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax.set_title(f"held-out {i}: image (z-x)", fontsize=9)
        ax = fig.add_subplot(gs[r, 1]); ax.imshow(pt["N"][:, c].cpu().numpy(), cmap=NMAP, vmin=-0.5, vmax=4.5, interpolation="nearest"); ax.set_title("target N (line populations)", fontsize=9)
        ax = fig.add_subplot(gs[r, 2]); ax.imshow(pp["N"][:, c].cpu().numpy(), cmap=NMAP, vmin=-0.5, vmax=4.5, interpolation="nearest"); ax.set_title("predicted N", fontsize=9)
        ax = fig.add_subplot(gs[r, 3]); ax.imshow(dec(pt["dirs_cells"][0][:, :, c], pt["mass"][0][:, c] / 0.2), interpolation="nearest"); ax.set_title("target: heaviest axis × mass", fontsize=9)
        ax = fig.add_subplot(gs[r, 4]); ax.imshow(dec(pp["dirs_cells"][0][:, :, c], pp["mass"][0][:, c] / 0.2), interpolation="nearest"); ax.set_title("predicted: heaviest axis × mass", fontsize=9)
        _, Vm = eig3(P["mixture"][0][:, :, c]); mixw = (T["mass"][0].sum(0)[:, c] / 0.2)
        ax = fig.add_subplot(gs[r, 5]); ax.imshow(dec(Vm[2], mixw), interpolation="nearest"); ax.set_title("predicted: mixture-head axis", fontsize=9)
        ax = fig.add_subplot(gs[r, 6]); ax.imshow(T["fraction"][0][:, c].cpu().numpy(), cmap="magma", vmin=0, vmax=1, interpolation="nearest"); ax.set_title("target fraction", fontsize=9)
        ax = fig.add_subplot(gs[r, 7]); ax.imshow(pp["fraction"][:, c].cpu().numpy(), cmap="magma", vmin=0, vmax=1, interpolation="nearest"); ax.set_title("predicted fraction", fontsize=9)
        tv, tV = eig3(T["slots"][0, 0][:, :, c]); lin = (tv[2] - tv[1]) / tv[2].clamp_min(1e-9)
        _, pV = eig3(P["slots"][0, 0][:, :, c])
        err = torch.rad2deg(torch.acos((tV[2] * pV[2]).sum(0).abs().clamp(max=1)))
        msk = (T["mass"][0, 0][:, c] > 0.05) & (T["kind"][0, 0][:, c] == 0) & (lin > 0.5)
        em = np.where(msk.cpu().numpy(), err.cpu().numpy(), np.nan)
        ax = fig.add_subplot(gs[r, 8]); h = ax.imshow(em, cmap="RdYlGn_r", vmin=0, vmax=60, interpolation="nearest")
        ax.set_title(f"slot-0 axis error (deg): median {np.nanmedian(em) if msk.any() else float('nan'):.1f}", fontsize=9); plt.colorbar(h, ax=ax, fraction=0.046)
        ax = fig.add_subplot(gs[r, 9]); vals = err[msk].cpu().numpy()
        if vals.size:
            ax.hist(vals, bins=30, range=(0, 90), color="#2a78d6")
        ax.set_title("axis error histogram (this slice)", fontsize=9)
        for k in range(9):
            fig.axes[-10 + k].set_xticks([]); fig.axes[-10 + k].set_yticks([])
    out = Path(a.out) if a.out else run / "qc_model.png"
    fig.suptitle(f"Model QC: {run}", fontsize=12); fig.savefig(out, dpi=55, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
