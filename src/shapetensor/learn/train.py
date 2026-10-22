"""Training the population model on the synthesis, generated on the fly.

    python -m shapetensor.learn.train --out DIR [--steps 20000] [--size 128] [--batch 4] [--buffer 256] [--refresh 1]

Samples are drawn (:func:`~shapetensor.learn.generator.draw`: a modality, matched content, its acquisition) and their
lattice targets computed on the GPU into a replay buffer of (image, target) pairs; every step takes a batch from it,
applies one random cube-group element to each pair (exact on the lattice: the tensors rotate with the volume), and
replaces ``refresh`` pairs with new draws, so each pair is seen a few times under different symmetries. AdamW, a
warm-up then a cosine schedule, bf16 autocast, gradient clipping. A fixed held-out set (seeds from 10^7) is evaluated
every ``val_every`` steps: the loss parts, the axis error of matched populations, the agreement of the population
count, the fraction error. ``log.jsonl`` and ``model.pt`` (state, arguments, step, receptive field) in ``--out``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from ..lift import hemisphere
from .generator import draw, full3
from .losses import augment, gather_slots, loss, match
from .model import PopulationNet, receptive_field
from .targets import target

TKEYS = ("slots", "mass", "kind", "layer", "fraction", "classes", "size", "size_mask")
HALF = ("slots", "mass", "fraction", "classes", "size")


def make(seed: int, size: int, device) -> tuple[torch.Tensor, dict]:
    """One training pair: the 8-bit image and its lattice target (floats in half precision)."""
    s, d = draw((size,) * 3, seed=seed, device=device)
    t = target(s, d)
    out = {k: (t[k].half() if k in HALF else t[k]) for k in TKEYS}
    out["kind"] = out["kind"].to(torch.int8); out["layer"] = out["layer"].to(torch.int16)
    return d["image"].to(torch.uint8), out


def stack(pairs, g=None, aug: bool = True):
    """A batch from (image, target) pairs: the image ``(B, 1, Z, Y, X)`` in 0-1, the targets stacked, cube-augmented."""
    imgs, tgts = [], []
    for img, t in pairs:
        t = {k: (v.float() if k in HALF else v.long() if k in ("kind", "layer") else v) for k, v in t.items()}
        if aug:
            img, t = augment(img, t, g)
        imgs.append(img); tgts.append(t)
    x = torch.stack(imgs)[:, None].float() / 255.0
    return x, {k: torch.stack([t[k] for t in tgts]) for k in TKEYS}


class Buffer:
    def __init__(self, n: int, size: int, device, seed0: int = 0, log=print):
        self.size, self.device, self.next = size, device, seed0
        self.items = []
        t0 = time.time()
        for i in range(n):
            self.items.append(make(self.next, size, device)); self.next += 1
        log(f"buffer: {n} pairs at {size}^3 in {time.time() - t0:.0f} s")

    def refresh(self, k: int, g: torch.Generator):
        for _ in range(k):
            i = int(torch.randint(0, len(self.items), (1,), generator=g))
            self.items[i] = make(self.next, self.size, self.device); self.next += 1

    def batch(self, B: int, g: torch.Generator):
        idx = torch.randint(0, len(self.items), (B,), generator=g).tolist()
        return stack([self.items[i] for i in idx], g)


def _axes(S: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """Axes of tensors ``S (..., 6)``: a line's / blob's largest eigenvector, a sheet's / interface's normal."""
    _, V = torch.linalg.eigh(full3(S))
    planar = ((kind == 1) | (kind == 3))[..., None]
    return torch.where(planar, V[..., 0], V[..., -1])


@torch.no_grad()
def evaluate(model, val, grid, margin: int, device) -> dict:
    """The held-out set: loss parts, the median axis error of matched populations (target mass > 0.05), the share of
    cells whose population count (mass > 0.02) is right, the fraction's mean absolute error."""
    model.eval(); parts, errs, ncorr, ntot, ferr = {}, [], 0.0, 0.0, []
    for img, t in val:
        x, T = stack([(img, t)], aug=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            P = model(x)
        _, p = loss(P, T, grid, margin)
        for k, v in p.items():
            parts[k] = parts.get(k, 0.0) + v / len(val)
        crop = lambda v: v[..., margin:-margin, margin:-margin, margin:-margin]
        Pc = {k: crop(v) for k, v in P.items()}; Tc = {k: crop(v) for k, v in T.items()}
        _, _, perm = match(Pc, Tc)
        Sp = gather_slots(Pc["slots"].float(), perm)                                              # (1, K, 6, c)
        kt = Tc["kind"]; mt = Tc["mass"].float()
        ap = _axes(Sp.movedim(2, -1), kt); at = _axes(Tc["slots"].float().movedim(2, -1), kt)
        ang = torch.rad2deg(torch.acos((ap * at).sum(-1).abs().clamp(max=1.0)))
        sel = mt > 0.05
        if bool(sel.any()):
            errs.append(ang[sel])
        Np = (Pc["mass"].float() > 0.02).sum(1); Nt = (mt > 0.02).sum(1)
        ncorr += float((Np == Nt).float().sum()); ntot += Np.numel()
        ferr.append((torch.sigmoid(Pc["fraction_logit"].float()) - Tc["fraction"].float()).abs().mean())
    model.train()
    e = torch.cat(errs) if errs else torch.zeros(1)
    return {**{f"val_{k}": v for k, v in parts.items()}, "val_axis_deg_median": float(e.median()), "val_axis_deg_p90": float(e.quantile(0.9)),
            "val_N_acc": ncorr / max(ntot, 1), "val_fraction_mae": float(torch.stack(ferr).mean())}


def train(out: str, steps: int = 20000, size: int = 128, batch: int = 4, buffer: int = 256, refresh: int = 1, lr: float = 1e-3,
          warmup: int = 300, margin: int = 3, val_n: int = 16, val_every: int = 500, log_every: int = 25, seed: int = 0, device: str = "cuda",
          resume: str | None = None):
    outd = Path(out); outd.mkdir(parents=True, exist_ok=True)
    logf = open(outd / "log.jsonl", "a")
    def log(msg):
        print(msg, flush=True)
    args = dict(steps=steps, size=size, batch=batch, buffer=buffer, refresh=refresh, lr=lr, warmup=warmup, margin=margin, val_n=val_n, seed=seed)
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = PopulationNet().to(device)
    rf = receptive_field(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start = 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    sched = lambda s: min(1.0, (s + 1) / warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    grid = hemisphere(160, device)
    log(f"model: {sum(p.numel() for p in model.parameters())} parameters, receptive field {rf} voxels; args {args}")
    val = [make(10_000_000 + i, size, device) for i in range(val_n)]
    buf = Buffer(buffer, size, device, seed0=seed * 1_000_000 + start * refresh, log=log)
    t0 = time.time(); run = {}
    for step in range(start, steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * sched(step)
        x, T = buf.batch(batch, g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            P = model(x)
        total, parts = loss(P, T, grid, margin, axis=int(torch.randint(0, 3, (1,), generator=g)))
        opt.zero_grad(set_to_none=True); total.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        buf.refresh(refresh, g)
        for k, v in {**parts, "total": float(total), "grad_norm": gn}.items():
            run[k] = run.get(k, 0.0) + v
        if (step + 1) % log_every == 0:
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1), "lr": lr * sched(step), **{k: v / log_every for k, v in run.items()}}
            logf.write(json.dumps(rec) + "\n"); logf.flush(); run = {}
            log(f"step {step + 1} {rec['time_s']}s total {rec['total']:.4f} shape {rec['shape']:.4f} mass {rec['mass']:.4f} hell {rec['hellinger']:.4f} cls {rec['classes']:.3f} id {rec['identity']:.3f}")
        if (step + 1) % val_every == 0 or step + 1 == steps:
            v = evaluate(model, val, grid, margin, device)
            logf.write(json.dumps({"step": step + 1, **v}) + "\n"); logf.flush()
            log(f"  val {step + 1}: " + " ".join(f"{k} {x:.4f}" for k, x in v.items()))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "args": args, "receptive_field": rf}, outd / "model.pt")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--buffer", type=int, default=256); ap.add_argument("--refresh", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--margin", type=int, default=3); ap.add_argument("--val-n", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=500); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--resume")
    a = ap.parse_args()
    train(a.out, a.steps, a.size, a.batch, a.buffer, a.refresh, a.lr, margin=a.margin, val_n=a.val_n, val_every=a.val_every, seed=a.seed, resume=a.resume)


if __name__ == "__main__":
    main()
