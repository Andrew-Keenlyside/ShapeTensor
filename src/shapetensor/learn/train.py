"""Training the population model on the synthesis, generated on the fly.

    python -m shapetensor.learn.train --out DIR [--steps 20000] [--size 96] [--batch 4] [--accum 4] [--buffer 128]

Samples are drawn (:func:`~shapetensor.learn.generator.draw`: a modality, matched content, its acquisition) and their
lattice targets computed on the GPU into a replay buffer of (image, target) pairs; each step takes ``accum``
micro-batches of ``batch`` from it (an effective batch of ``batch * accum``), applies one random cube-group element to
each pair (exact on the lattice) and a random intensity window jitter, and replaces ``refresh`` pairs with new draws.

A curriculum, against the collapse to constant outputs seen when everything is asked at once:

* **nuisance strength** ramps from ``strength0`` to 1 over the first half of training (clear images first);
* **orientation first** -- the slots' orientation and the mixture tensor from the start; mass, presence, kind and the
  Hellinger term ramped in from max(300, 5% of the steps) over 300 steps (present from the start, they swamp the
  orientation signal and the network never learns it); the cell scalars (classes, fraction, size) and the identity
  term off for the first 20% of steps, ramped in by 40%.

AdamW (``lr`` 3e-4), warm-up then cosine, bf16 autocast, clipping at 5. A fixed held-out set (seeds from 10^7, full
strength, ``strength1``) is scored every ``val_every`` steps with metrics that can see learning: the matched slot axes' error against
the untrained network's (the chance level of a best-of-K match), the spread and correlation of the predicted fraction
(a collapsed network is constant), and the engine's population count against the target's. ``--overfit N``: a fixed
set of N pairs and no refresh, scored on those pairs (a sanity check that the loss can be driven down).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from ..lift import hemisphere
from .generator import draw, eig3
from .losses import WEIGHTS, augment, gather_slots, loss, match
from .model import PopulationNet, receptive_field
from .targets import target

TKEYS = ("slots", "mass", "kind", "layer", "fraction", "classes", "size", "size_mask")
HALF = ("slots", "mass", "fraction", "classes", "size")
SCALARS = ("classes", "fraction", "size", "identity")
LATE = ("mass", "presence", "kind", "hellinger")                                                   # after orientation: they swamp it if present from the start


def make(seed: int, size: int, device, strength: float = 1.0, gen: dict | None = None) -> tuple[torch.Tensor, dict]:
    """One training pair: the 8-bit image and its lattice target (floats in half precision). ``gen``: arguments for
    :func:`~shapetensor.learn.generator.draw` (a modality, a background style, crossing groups, layer kinds, ...)."""
    s, d = draw((size,) * 3, seed=seed, device=device, strength=strength, **(gen or {}))
    t = target(s, d)
    out = {k: (t[k].half() if k in HALF else t[k]) for k in TKEYS}
    out["kind"] = out["kind"].to(torch.int8); out["layer"] = out["layer"].to(torch.int16)
    return d["image"].to(torch.uint8), out


def stack(pairs, g=None, aug: bool = True):
    """A batch from (image, target) pairs: the image ``(B, 1, Z, Y, X)`` in 0-1, the targets stacked; with ``aug``
    each pair cube-transformed and its intensities jittered (a different window: ``a x + b``, clipped)."""
    imgs, tgts = [], []
    for img, t in pairs:
        t = {k: (v.float() if k in HALF else v.long() if k in ("kind", "layer") else v) for k, v in t.items()}
        x = img.float() / 255.0
        if aug:
            x, t = augment(x, t, g)
            a = 0.6 + 0.8 * float(torch.rand(1, generator=g)); b = 0.4 * (float(torch.rand(1, generator=g)) - 0.5)
            x = (a * (x - 0.5) + 0.5 + b).clamp(0, 1)
        imgs.append(x); tgts.append(t)
    return torch.stack(imgs)[:, None], {k: torch.stack([t[k] for t in tgts]) for k in TKEYS}


class Buffer:
    def __init__(self, n: int, size: int, device, seed0: int = 0, strength: float = 1.0, log=print, gen: dict | None = None):
        self.size, self.device, self.next, self.gen = size, device, seed0, gen
        t0 = time.time()
        self.items = [self._make(strength) for _ in range(n)]
        log(f"buffer: {n} pairs at {size}^3 (strength {strength:.2f}) in {time.time() - t0:.0f} s")

    def _make(self, strength):
        item = make(self.next, self.size, self.device, strength, self.gen); self.next += 1
        return item

    def refresh(self, k: int, g: torch.Generator, strength: float):
        for _ in range(k):
            self.items[int(torch.randint(0, len(self.items), (1,), generator=g))] = self._make(strength)

    def batch(self, B: int, g: torch.Generator):
        idx = torch.randint(0, len(self.items), (B,), generator=g).tolist()
        return stack([self.items[i] for i in idx], g)


def _axes(S: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
    """Axes of tensors ``S (..., 6)``: a line's / blob's largest eigenvector, a sheet's / interface's normal."""
    _, V = eig3(S.movedim(-1, 0))
    planar = ((kind == 1) | (kind == 3))[None]
    return torch.where(planar, V[0], V[2]).movedim(0, -1)


@torch.no_grad()
def evaluate(model, val, grid, margin: int, device, engine=None) -> dict:
    """The held-out set: loss parts; the median axis error of matched line populations whose target is linear (mass >
    0.05, linearity > 0.5) and of the mixture head on cells with one line population; the predicted
    fraction's spread over the set and its correlation with the target's; the engine's population count (line
    populations, N) against the target's own (the same engine reading the target slots)."""
    model.eval(); parts, errs, mix_errs, fp, ft, nagree, ntot = {}, [], [], [], [], 0.0, 0.0
    crop = lambda v: v[..., margin:-margin, margin:-margin, margin:-margin]
    for img, t in val:
        x, T = stack([(img, t)], aug=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            P = model(x)
        _, p = loss(P, T, grid, margin)
        for k, v in p.items():
            parts[k] = parts.get(k, 0.0) + v / len(val)
        Pc = {k: crop(v) for k, v in P.items()}; Tc = {k: crop(v) for k, v in T.items()}
        _, _, perm = match(Pc, Tc)
        Sp = gather_slots(Pc["slots"].float(), perm)
        kt = Tc["kind"]; mt = Tc["mass"].float()
        ang = torch.rad2deg(torch.acos((_axes(Sp.movedim(2, -1), kt) * _axes(Tc["slots"].float().movedim(2, -1), kt)).sum(-1).abs().clamp(max=1.0)))
        tv, _ = eig3(Tc["slots"].float().movedim(2, 0))                                                   # (3, 1, K, c): the target's own linearity
        lin = (tv[2] - tv[1]) / tv[2].clamp_min(1e-9)
        sel = (mt > 0.05) & (kt == 0) & (lin > 0.5)                                                       # cells where the target is a line with an axis
        if bool(sel.any()):
            errs.append(ang[sel])
        # the permutation-free mixture head against the target mixture, where the cell holds one line population
        one = sel[:, 0] & ~(mt[:, 1:] > 0.05).any(1)
        if bool(one.any()):
            _, Vm = eig3(Pc["mixture"][0].float()); _, Vt = eig3(Tc["slots"][0, 0].float())
            mix_errs.append(torch.rad2deg(torch.acos((Vm[2] * Vt[2]).sum(0).abs().clamp(max=1.0)))[one[0]])
        fp.append(torch.sigmoid(Pc["fraction_logit"].float()).flatten()); ft.append(Tc["fraction"].float().flatten())
        if engine is not None:
            pe = engine.populations({k: v[0].float() for k, v in P.items()}, labels=False)
            to = {"slots": T["slots"][0], "mass": T["mass"][0], "root_mass": T["mass"][0].sqrt(), "presence_logit": torch.where(T["mass"][0] > 0.005, 10.0, -10.0),
                  "kind": 30.0 * torch.nn.functional.one_hot(T["kind"][0].clamp_min(0), 4).movedim(-1, 1).float(), "embed": P["embed"][0].float(),
                  "fraction_logit": P["fraction_logit"][0].float(), "class_logits": P["class_logits"][0].float(), "log_size": P["log_size"][0].float()}
            pt = engine.populations(to, labels=False)
            Np, Nt = crop(pe["N"]), crop(pt["N"])
            nagree += float((Np == Nt).float().sum()); ntot += Np.numel()
    model.train()
    e = torch.cat(errs) if errs else torch.zeros(1)
    fp, ft = torch.cat(fp), torch.cat(ft)
    corr = float(((fp - fp.mean()) * (ft - ft.mean())).mean() / (fp.std() * ft.std()).clamp_min(1e-9))
    em = torch.cat(mix_errs) if mix_errs else torch.zeros(1)
    out = {**{f"val_{k}": v for k, v in parts.items()}, "val_axis_deg_median": float(e.median()), "val_axis_deg_p90": float(e.quantile(0.9)),
           "val_mixture_axis_deg_median": float(em.median()),
           "val_fraction_std": float(fp.std()), "val_fraction_corr": corr, "val_fraction_mae": float((fp - ft).abs().mean())}
    if engine is not None:
        out["val_N_agree"] = nagree / max(ntot, 1)
    return out


def train(out: str, steps: int = 20000, size: int = 96, batch: int = 4, accum: int = 4, buffer: int = 128, refresh: int = 1, lr: float = 3e-4,
          warmup: int = 500, margin: int = 4, val_n: int = 32, val_every: int = 500, log_every: int = 25, seed: int = 0, device: str = "cuda",
          strength0: float = 0.15, strength1: float = 1.0, overfit: int = 0, gen: dict | None = None, resume: str | None = None,
          widths: tuple = (16, 32, 64), distinct: float | None = None):
    from .engine import PopulationEngine
    from . import losses as _losses
    _losses.DISTINCT = distinct                                                                          # slot order kept only among similar shapes
    outd = Path(out); outd.mkdir(parents=True, exist_ok=True)
    logf = open(outd / "log.jsonl", "a")
    def log(msg):
        print(msg, flush=True)
    args = dict(steps=steps, size=size, batch=batch, accum=accum, buffer=buffer, refresh=refresh, lr=lr, warmup=warmup, margin=margin, val_n=val_n,
                seed=seed, strength0=strength0, strength1=strength1, overfit=overfit, gen=gen, widths=list(widths), distinct=distinct)
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    model = PopulationNet(widths=tuple(widths)).to(device)
    rf = receptive_field(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    start = 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    sched = lambda s: min(1.0, (s + 1) / warmup) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    strength = lambda s: strength1 if overfit else strength0 + (strength1 - strength0) * min(1.0, s / (0.5 * steps))
    ramp = lambda s: min(1.0, max(0.0, (s / steps - 0.2) / 0.2))                                     # the cell scalars: 0 until 20%, full by 40%
    o0 = max(300, int(0.05 * steps))
    oramp = lambda s: min(1.0, max(0.0, (s - o0) / 300))                                               # mass, presence, kind, Hellinger: after orientation (o0 steps)
    grid = hemisphere(160, device)
    log(f"model: {sum(p.numel() for p in model.parameters())} parameters, receptive field {rf} voxels; args {args}")
    buf = Buffer(overfit or buffer, size, device, seed0=seed * 1_000_000 + start * refresh, strength=strength(start), log=log, gen=gen)
    val = buf.items[:min(val_n, overfit)] if overfit else [make(10_000_000 + i, size, device, strength1, gen) for i in range(val_n)]   # overfit: score the training pairs
    engine = PopulationEngine(model, device=device)
    v0 = evaluate(model, val, grid, margin, device, engine)
    logf.write(json.dumps({"step": start, "baseline": True, **v0}) + "\n"); logf.flush()
    log(f"  val {start} (baseline): " + " ".join(f"{k} {x:.4f}" for k, x in v0.items()))
    t0 = time.time(); run = {}
    for step in range(start, steps):
        for pg in opt.param_groups:
            pg["lr"] = lr * sched(step)
        r = ramp(step); ro = oramp(step)
        W = {**{k: WEIGHTS[k] * r for k in SCALARS}, **{k: WEIGHTS[k] * ro for k in LATE}}
        opt.zero_grad(set_to_none=True); tot_v = 0.0; parts_acc = {}
        for _ in range(accum):
            x, T = buf.batch(batch, g)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
                P = model(x)
            total, parts = loss(P, T, grid, margin, axis=int(torch.randint(0, 3, (1,), generator=g)), weights=W, g=g)
            (total / accum).backward(); tot_v += float(total) / accum
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v / accum
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        opt.step()
        if not overfit:
            buf.refresh(refresh, g, strength(step))
        for k, v in {**parts_acc, "total": tot_v, "grad_norm": gn}.items():
            run[k] = run.get(k, 0.0) + v
        if (step + 1) % log_every == 0:
            rec = {"step": step + 1, "time_s": round(time.time() - t0, 1), "lr": lr * sched(step), "strength": strength(step), "scalar_weight": r,
                   **{k: v / log_every for k, v in run.items()}}
            logf.write(json.dumps(rec) + "\n"); logf.flush(); run = {}
            log(f"step {step + 1} {rec['time_s']}s s={rec['strength']:.2f} total {rec['total']:.4f} shape {rec['shape']:.4f} mass {rec['mass']:.4f} "
                f"pres {rec['presence']:.4f} hell {rec['hellinger']:.4f} cls {rec['classes']:.3f} frac {rec['fraction']:.3f}")
        if (step + 1) % val_every == 0 or step + 1 == steps:
            v = evaluate(model, val, grid, margin, device, engine)
            logf.write(json.dumps({"step": step + 1, **v}) + "\n"); logf.flush()
            log(f"  val {step + 1}: " + " ".join(f"{k} {x:.4f}" for k, x in v.items()))
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1, "args": args, "receptive_field": rf}, outd / "model.pt")
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--accum", type=int, default=4); ap.add_argument("--buffer", type=int, default=128)
    ap.add_argument("--refresh", type=int, default=1); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--val-n", type=int, default=32); ap.add_argument("--val-every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength0", type=float, default=0.15); ap.add_argument("--overfit", type=int, default=0); ap.add_argument("--resume")
    ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--strength1", type=float, default=1.0)
    ap.add_argument("--distinct", type=float, default=None, help="slot order kept only among populations whose eigenvalue spectra lie within this distance (e.g. 0.2); default: among all")
    ap.add_argument("--widths", type=int, nargs=3, default=[16, 32, 64], help="channels at strides 1, 2, 4")
    ap.add_argument("--gen", type=json.loads, default=None, help='draw() arguments as JSON, e.g. {"style": "flat", "crossing": false, "k_layers": 1, "kinds": ["line"]}')
    a = ap.parse_args()
    train(a.out, a.steps, a.size, a.batch, a.accum, a.buffer, a.refresh, a.lr, warmup=a.warmup, margin=a.margin, val_n=a.val_n, val_every=a.val_every,
          seed=a.seed, strength0=a.strength0, strength1=a.strength1, overfit=a.overfit, gen=a.gen, resume=a.resume, widths=tuple(a.widths), distinct=a.distinct)


if __name__ == "__main__":
    main()
