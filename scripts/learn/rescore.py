"""A fixed validation bank and a re-scorer with confidence intervals over draws.

    python scripts/learn/rescore.py bank --out BANK.pt [--n 64] [--strength 0.3] [--gen JSON]
    python scripts/learn/rescore.py score BANK.pt RUN_DIR [RUN_DIR ...] [--json OUT]

**bank**: held-out draws (seeds from 2e7, disjoint from training and the runs' own validation) that contain line
targets (mass > 0.05, linearity > 0.5), each under a fixed cube rotation; stored with their lattice targets and the
exact per-voxel density (SH order 8, window 5, the lattice) so every run is scored on the same draws.

**score**: per run, line-cell axis errors pooled over the bank with a 90% bootstrap interval over draws (1000
resamples): matched slot (one reference matching rule) on single cells (the line is the only anisotropic population)
and shared cells; an order-free slot score (the predicted slot of largest anisotropic mass on single cells); the
mixture head on single cells; the SH density's top peak (single) and nearest of its top 3 (single, shared); on
crossing cells (two line targets > 30 deg apart) how often the SH density has two peaks and their mean error; a
structure tensor on the same cells; and the untrained network of the same width (the floor).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from shapetensor.learn import losses as L
from shapetensor.learn.generator import draw, eig3
from shapetensor.learn.losses import augment, gather_slots, match
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.query import voxel_density
from shapetensor.learn.sh import peaks, peaks_gated
from shapetensor.learn.targets import _pool, target
from shapetensor.learn.train import HALF, TKEYS, _axes, st_axis_cells, stack

DEV = "cuda"; M = 3


def make_bank(a):
    gen = json.loads(a.gen) if a.gen else {}
    items, seed = [], 20_000_000
    while len(items) < a.n:
        s, d = draw((a.size,) * 3, seed=seed, device=DEV, strength=a.strength, **gen)
        t = target(s, d)
        tv, _ = eig3(t["slots"].float().movedim(1, 0)); lin = (tv[2] - tv[1]) / tv[2].clamp_min(1e-9)
        line = (t["kind"] == 0) & (t["mass"] > 0.05) & (lin > 0.5)
        crop = line[..., M:-M, M:-M, M:-M]
        if int(crop.sum()) >= 20:
            tt = {k: (t[k].half() if k in HALF else t[k]) for k in TKEYS}
            tt["kind"] = tt["kind"].to(torch.int8); tt["layer"] = tt["layer"].to(torch.int16)
            tt["sh_t"] = _pool(voxel_density(s, d, 8), 5.0, 4).half()
            img, tt = augment(d["image"].to(torch.uint8), tt, torch.Generator().manual_seed(seed))
            items.append({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in {"img": img, **tt}.items()})
            print(f"bank {len(items)}/{a.n} (seed {seed}, {int(crop.sum())} line cells, {d['record']['modality']})", flush=True)
        seed += 1
    torch.save({"items": items, "gen": gen, "strength": a.strength, "size": a.size}, a.out)
    print("wrote", a.out)


def ang(a_, b_):
    return torch.rad2deg(torch.acos((a_ * b_).sum(-1).abs().clamp(max=1)))


def per_draw(net, item, lmax):
    """Cell errors of one draw, by metric."""
    img = item["img"].to(DEV); t = {k: v.to(DEV) for k, v in item.items() if k != "img"}
    x, T = stack([(img, {k: t[k] for k in TKEYS})], aug=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        P = net(x)
    P = {k: v.float() for k, v in P.items()}
    crop = lambda v: v[..., M:-M, M:-M, M:-M]
    Pc = {k: crop(v) for k, v in P.items()}; Tc = {k: crop(v) for k, v in T.items()}
    saved = (L.DISTINCT, L.PRESENCE_BAND); L.DISTINCT = None; L.PRESENCE_BAND = None
    try:
        _, _, perm = match(Pc, Tc)
    finally:
        L.DISTINCT, L.PRESENCE_BAND = saved
    kt, mt = Tc["kind"], Tc["mass"].float()
    tv, _ = eig3(Tc["slots"].float().movedim(2, 0)); lin = (tv[2] - tv[1]) / tv[2].clamp_min(1e-9)
    an = (tv[2] - tv[0]) / tv.sum(0).clamp_min(1e-9)
    sel = (mt > 0.05) & (kt == 0) & (lin > 0.5)
    single_cell = (sel.sum(1) == 1) & (((mt > 0.05) & (an > 0.25)).sum(1) == 1)
    single, shared = sel & single_cell[:, None], sel & ~single_cell[:, None]
    t_ax = _axes(Tc["slots"].float().movedim(2, -1), kt)                                              # (1, K, c, 3)
    Sp = gather_slots(Pc["slots"], perm)
    p_ax = _axes(Sp.movedim(2, -1), kt)
    out = {"slot_single": ang(p_ax, t_ax)[single], "slot_shared": ang(p_ax, t_ax)[shared]}
    sv, _ = eig3(Sp.movedim(2, 0)); plin = (sv[2] - sv[1]) / sv[2].clamp_min(1e-9)                 # matched slot's linearity
    out["slot_linearity_single"] = plin[single]
    # false lines: cells with no line target (mass > 0.05) where a present predicted slot is a line of linearity > 0.5
    pres = (torch.sigmoid(Pc["presence_logit"]) > 0.5) & (Pc["mass"] >= 0.01)
    pv_, _ = eig3(Pc["slots"].movedim(2, 0)); plin_all = (pv_[2] - pv_[1]) / pv_[2].clamp_min(1e-9)
    pline = (pres & (Pc["kind"].argmax(2) == 0) & (plin_all > 0.5)).any(1)                           # (1, c)
    noline = ~((mt > 0.05) & (kt == 0)).any(1)
    out["false_line_noline"] = pline[noline].float()
    # order-free slot: the predicted slot of largest anisotropic mass
    pv, pV = eig3(Pc["slots"].movedim(2, 0)); psal = Pc["mass"] * (pv[2] - pv[0])                     # (1, K, c)
    best = psal.argmax(1)                                                                             # (1, c)
    bax = torch.gather(pV[2].movedim(0, -1), 1, best[:, None, ..., None].expand(-1, 1, *best.shape[1:], 3))   # (1, 1, c, 3)
    out["slot_orderfree_single"] = ang(bax, t_ax)[single]
    _, Vm = eig3(Pc["mixture"][0]); out["mix_single"] = ang(Vm[2].movedim(0, -1)[None, None], t_ax)[single]
    st = crop(st_axis_cells(x[0, 0])); out["st_single"] = ang(st.movedim(0, -1)[None, None], t_ax)[single]
    out["st_shared"] = ang(st.movedim(0, -1)[None, None], t_ax)[shared]
    if lmax:
        cs = Pc["sh"].shape[2:]
        D, V, _ = peaks_gated(Pc["sh"][0].flatten(1), lmax, nmax=3)                                     # calibrated (review 5)
        pk = D.movedim(1, -1).reshape(3, *cs, 3); pkv = V.reshape(3, *cs)
        e = torch.stack([torch.where(pkv[j][None, None] > 0, ang(pk[j][None, None], t_ax), torch.full(t_ax.shape[:-1], 90.0, device=DEV)) for j in range(3)])
        out["sh_top1_single"] = e[0][single]; out["sh_near3_single"] = e.amin(0)[single]; out["sh_near3_shared"] = e.amin(0)[shared]
        out["sh_false_second_single"] = (pkv[1][None, None].expand_as(single) > 0)[single].float()
        two = sel[0, 0] & sel[0, 1] & (ang(t_ax[0, 0], t_ax[0, 1]) > 30)                              # crossing cells
        if bool(two.any()):
            det = (pkv[1] > 0) & two
            out["sh_cross_detect"] = det[two].float()
            if bool(det.any()):
                a1, a2, p1, p2 = t_ax[0, 0][det], t_ax[0, 1][det], pk[0][det], pk[1][det]
                out["sh_cross_err"] = torch.minimum((ang(p1, a1) + ang(p2, a2)) / 2, (ang(p1, a2) + ang(p2, a1)) / 2)
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def summarise(draws, rng):
    keys = sorted({k for d in draws for k in d})
    res = {}
    for k in keys:
        arrs = [d.get(k, np.zeros(0)) for d in draws]
        pooled = np.concatenate(arrs)
        if pooled.size == 0:
            continue
        stat = np.mean if k in ("sh_cross_detect", "false_line_noline", "sh_false_second_single") else np.median
        boots = []
        for _ in range(1000):
            idx = rng.integers(0, len(arrs), len(arrs))
            b = np.concatenate([arrs[i] for i in idx])
            if b.size:
                boots.append(stat(b))
        res[k] = (float(stat(pooled)), float(np.percentile(boots, 5)), float(np.percentile(boots, 95)), int(pooled.size), int(sum(a.size > 0 for a in arrs)))
    return res


def score(a):
    bank = torch.load(a.bank, weights_only=False); items = bank["items"]
    table, per_run = {}, {}
    for run in a.runs:
        ck = torch.load(Path(run) / "model.pt", map_location=DEV); args = ck["args"]; lmax = int(args.get("sh_lmax", 0) or 0)
        net = PopulationNet.from_args(args).to(DEV); net.load_state_dict(ck["model"]); net.eval()
        torch.manual_seed(0); floor = PopulationNet.from_args(args).to(DEV).eval()
        name = f"{Path(run).name}@{ck['step']}"
        draws = [per_draw(net, it, lmax) for it in items]; per_run[name] = draws
        table[name] = summarise(draws, np.random.default_rng(0))
        fk = f"untrained ({tuple(args.get('widths', (16, 32, 64)))}, sh {lmax})"
        if fk not in table:                                                                            # a floor per width / head set
            table[fk] = summarise([per_draw(floor, it, lmax) for it in items], np.random.default_rng(0))
    keys = ["slot_single", "slot_orderfree_single", "slot_shared", "mix_single", "slot_linearity_single", "false_line_noline",
            "sh_top1_single", "sh_near3_single", "sh_near3_shared", "sh_false_second_single", "sh_cross_detect", "sh_cross_err", "st_single", "st_shared"]
    print(f"bank: {len(items)} draws, gen {bank['gen']}, strength {bank['strength']}")
    for name, res in table.items():
        print(f"\n{name}")
        for k in keys:
            if k in res:
                m, lo, hi, n, nd = res[k]
                print(f"  {k:22s} {m:6.2f}  [{lo:6.2f}, {hi:6.2f}]  ({n} cells in {nd} draws)")
    names = list(per_run)
    if len(names) > 1:                                                                                  # paired differences against the first run
        ref = names[0]; rng = np.random.default_rng(1)
        print(f"\npaired differences (run - {ref}), 90% bootstrap over the same draws:")
        for nm in names[1:]:
            for k in ("slot_single", "slot_shared", "mix_single", "sh_top1_single"):
                A, B = per_run[nm], per_run[ref]
                if not all(k in d for d in A) or not all(k in d for d in B):
                    continue
                stat = lambda D, idx: np.median(np.concatenate([D[i][k] for i in idx]))
                full = stat(A, range(len(A))) - stat(B, range(len(B)))
                bs = [stat(A, idx) - stat(B, idx) for idx in (rng.integers(0, len(A), len(A)) for _ in range(500))]
                print(f"  {nm:40s} {k:16s} {full:+6.2f}  [{np.percentile(bs, 5):+6.2f}, {np.percentile(bs, 95):+6.2f}]")
    if a.json:
        Path(a.json).write_text(json.dumps(table, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank"); b.add_argument("--out", required=True); b.add_argument("--n", type=int, default=64); b.add_argument("--strength", type=float, default=0.3)
    b.add_argument("--gen"); b.add_argument("--size", type=int, default=64)
    s = sub.add_parser("score"); s.add_argument("bank"); s.add_argument("runs", nargs="+"); s.add_argument("--json")
    a = ap.parse_args()
    make_bank(a) if a.cmd == "bank" else score(a)


if __name__ == "__main__":
    main()
