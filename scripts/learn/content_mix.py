"""What the scene generator's crops contain, with the Monte Carlo fibre options off and on (:mod:`~shapetensor.learn.fibres`):
the share of shape mass per kind (line / sheet / blob / solid) and how much of it is oriented, so a change to the fibres
cannot quietly turn the training data into fibres only.

Per crop (default family mix, the training's settings otherwise): each shape layer's mass (sum of its occupancy) by kind,
and the mass whose orientation is weak (the layer's per-voxel tensor has linearity and planarity both below 0.3: blobs,
cells, granules, crystals -- material without a direction). Also the share of crops with no oriented material at all.

Usage: python scripts/learn/content_mix.py OUT.json [--n 40]
"""
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, "src")
from shapetensor.learn import scene as SC, fibres as FB

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed0", type=int, default=31_000_000)
a = ap.parse_args(); dev = "cuda"


def westin(t6):
    T = torch.zeros(*t6.shape[1:], 3, 3, device=t6.device)
    for (i, j), c in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
        T[..., i, j] = t6[c]; T[..., j, i] = t6[c]
    ev = torch.linalg.eigvalsh(T).flip(-1).clamp_min(0); ev = ev / ev.sum(-1, keepdim=True).clamp_min(1e-9)
    return ev[..., 0] - ev[..., 1], 2 * (ev[..., 1] - ev[..., 2])                          # linearity, planarity


out = {}
for name, (pk, rx) in {"off (default)": ("hex", False), "MC + relax": ("random", True)}.items():
    FB.PACKING, FB.RELAX = pk, rx
    kinds, weak, total, none_oriented = {}, 0.0, 0.0, 0
    for i in range(a.n):
        SC._CACHE.clear()
        s, d = SC.draw_scene(seed=a.seed0 + i * 8, device=dev, crops_per_scene=8)
        crop_or = crop_tot = 0.0
        for k, L in enumerate(s.layers):
            if L.role != "shape":
                continue
            occ = d["occupancy"][k].float(); m = float(occ.sum())
            if m <= 0:
                continue
            lin, pla = westin(d["tensors"][k].float())
            w = float((occ * ((lin < 0.3) & (pla < 0.3))).sum())
            kinds[L.kind] = kinds.get(L.kind, 0.0) + m; weak += w; total += m; crop_or += m - w; crop_tot += m
        none_oriented += crop_or < 1e-3 * max(crop_tot, 1)
        print(name, i, {k: round(v / max(total, 1e-9), 3) for k, v in kinds.items()}, flush=True)
    out[name] = {"kind_share": {k: round(v / total, 3) for k, v in kinds.items()}, "non_oriented_share": round(weak / total, 3),
                 "crops_without_oriented_material": int(none_oriented), "n": a.n}
    print(name, json.dumps(out[name]), flush=True)
FB.PACKING, FB.RELAX = "hex", False
json.dump(out, open(a.out, "w"), indent=1)
