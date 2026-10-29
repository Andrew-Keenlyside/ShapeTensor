"""A short structural test of the populations head: the trainer's step (cube transforms, contrast routes, bf16, AdamW,
clipping) on 8 fixed crops for a few hundred steps, from scratch, then the ellipse error by target type, the axis error
of line targets, crossings found (two line targets >= 30 deg apart, each share >= 0.1, both within 15 deg) and the rate
of duplicate slots (two existing slots with ellipses < 0.05 apart, against the targets' rate). Minutes, not hours:
it says whether a head can read the images at all, not how it generalises.

    PYTHONPATH=src python scripts/replicate_pops.py ITEMS.pt [--steps 300] [--set pop_seq=2 pop_dir=96 ...]
ITEMS.pt: the (image, target) pairs a trainer run with --overfit saves (overfit_items.pt); its model.pt beside it gives
the base arguments.
"""
import argparse
import json
import math
import os

import torch

import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser()
ap.add_argument("items"); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--set", nargs="*", default=[], help="model arguments, key=value (JSON values)")
ap.add_argument("--et", nargs="*", default=[], help="entity_train constants, KEY=value (JSON values), e.g. POP_TIE=0.3")
a = ap.parse_args(); dev = "cuda"
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.42)))
ck = torch.load(os.path.join(os.path.dirname(a.items), "model.pt"), map_location="cpu", weights_only=False); args = dict(ck["args"])
for kv in a.set:
    k, v = kv.split("=", 1); args[k] = json.loads(v)
ET.ENT_WEIGHTS.update(args["weights"]); ET.POP_BG = int(args.get("pop_seq", 0) or 0) == 2
ET.POP_MATCH_AS_LOSS = not bool(args.get("pop_match_share", False)); TR.ROUTES = 1.0
for kv in a.et:
    k, v = kv.split("=", 1); setattr(ET, k, json.loads(v))
items = [(i[0].to(dev), {k: v.to(dev) if torch.is_tensor(v) else v for k, v in i[1].items()}) for i in torch.load(a.items, weights_only=False)]
V = [TR.stack(items[:4], aug=False), TR.stack(items[4:], aug=False)]

torch.manual_seed(0); net = PopulationNet.from_args(args).to(dev).train()
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
g = torch.Generator().manual_seed(1); gd = torch.Generator(device=dev).manual_seed(2)
for st in range(a.steps + 1):
    sel = torch.randperm(len(items), generator=g)[:4].tolist(); x, T = TR.stack([items[i] for i in sel], g, aug=True)
    dirs = ET.sample_dirs(T, 32, g=gd)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        P = net(x, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
    for pg in opt.param_groups:
        pg["lr"] = a.lr * min(1.0, (st + 1) / 30)
    total, parts = ET.entity_loss(P, T, dirs, margin=4, g=gd)
    opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    if st % 100 == 0:
        print(f"step {st}: pop {float(parts['pop']):.3f} existence {float(parts['pop_exist']):.3f} share {float(parts['pop_share']):.3f} ellipse {float(parts['pop_T']):.4f}", flush=True)

net.eval(); ang = lambda u, v: torch.rad2deg(torch.acos((u * v).sum(-1).abs().clamp(max=1)))
rows, cross, dup_p, dup_t = [], [], [], []
for x0, T0 in V:
    with torch.no_grad():
        Pv = net(x0, voxel=T0["voxel"], e_points=T0["e_points"], e_sigma=T0["e_sigma"], e_dirs=ET.sample_dirs(T0, 32))
    mt = ET.population_match(Pv["ent"], T0); PT = Pv["ent"]["pop_T"].float(); on = Pv["ent"]["pop_exist"].float() > 0.5
    v = mt["v_m"].bool() & mt["has"][..., None]
    evt, Vt = torch.linalg.eigh(mt["E_m"].double()); evp, Vp = torch.linalg.eigh(PT.double())
    lin_t = (evt[..., 2] - evt[..., 1]).float(); pla_t = (evt[..., 1] - evt[..., 0]).float(); lin_p = (evp[..., 2] - evp[..., 1]).float()
    ax_t, ax_p = Vt[..., 2].float(), Vp[..., 2].float()
    d = ((PT - mt["E_m"]) ** 2).sum((-1, -2))
    rows.append((d[v], lin_t[v], pla_t[v], mt["s_m"][v], ang(ax_p, ax_t)[v], lin_p[v]))
    line = v & (lin_t > 0.6) & (mt["s_m"] >= 0.1)
    for b, q in torch.nonzero(line.sum(-1) >= 2).tolist():
        ks = torch.nonzero(line[b, q])[:, 0][:2]
        if float(ang(ax_t[b, q, ks[0]], ax_t[b, q, ks[1]])) < 30:
            continue
        cross.append(all(float(ang(ax_p[b, q, k], ax_t[b, q, k])) < 15 and float(lin_p[b, q, k]) > 0.3 for k in ks.tolist()))
    for M, ok, acc in ((PT, on, dup_p), (mt["E_m"], v, dup_t)):                                # duplicate pairs per query
        dd = ((M[..., :, None, :, :] - M[..., None, :, :, :]) ** 2).sum((-1, -2))
        pair = ok[..., :, None] & ok[..., None, :] & torch.triu(torch.ones(M.shape[-3], M.shape[-3], dtype=torch.bool, device=dev), 1)
        acc.append(float(((dd < 0.05) & pair).sum()) / max(int(ok.any(-1).sum()), 1))
d, lin, pla, s, ax, plin = (torch.cat([r[i] for r in rows]) for i in range(6))
for name, m in (("line", lin > 0.6), ("sheet", pla > 0.3), ("isotropic", (lin < 0.15) & (pla < 0.15)), ("all", lin > -1)):
    print(f"  {name}: n {int(m.sum())}, ellipse error {float(d[m].mean()):.3f}, predicted line-ness median {float(plin[m].median()):.2f}"
          + (f", axis error median {float(ax[m].median()):.1f} deg" if name == "line" else ""))
print(f"  crossings: n {len(cross)}, both found {100 * sum(cross) / max(len(cross), 1):.0f} %")
print(f"  duplicate slot pairs per query: model {sum(dup_p) / len(dup_p):.2f}, targets {sum(dup_t) / len(dup_t):.2f}")
