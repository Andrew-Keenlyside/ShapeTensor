"""Visibility audit of scenes built of objects: for every population in clean renders, how strongly the image shows it --
|correlation| between the population's material and the image, both high-passed (smooth shading removed) -- by role
(ground, matrix, foreground) and by modality; and the share of foreground material whose population is effectively
invisible (|correlation| below 0.1). Synthetic only.

    PYTHONPATH=src python scripts/check_visibility.py [N_SCENES]
"""
import collections
import sys

import numpy as np
import torch

import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False; G.TARGETS_FROM_GEOMETRY = True
from shapetensor.filters import gaussian
from shapetensor.learn import scene as SC, lobe_train as LT

dev = "cuda"; torch.cuda.set_per_process_memory_fraction(0.45); SC.MATRIX_P = 1.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
hp = lambda x: (x.float() - gaussian(x.float()[None, None], 4.0)[0, 0]).flatten()
by_role, by_mod, inv_fore = collections.defaultdict(list), collections.defaultdict(list), []
for i in range(N):
    try:
        s, d = SC.draw_scene((64,) * 3, seed=330_000_000 + 8 * i, device=dev, strength=0.0, entity_ctx=False, **gen)
    except torch.OutOfMemoryError:
        SC._scene_cache().clear(); torch.cuda.empty_cache(); continue
    img = hp(s.image); mod = d["record"].get("modality"); fore_tot = fore_inv = 0.0
    for k, L in enumerate(s.layers):
        role = getattr(L, "mrole", None)
        if role is None:
            continue
        occ = d["occupancy"][k].float()
        if float(occ.sum()) < 50:
            continue
        o = hp(occ)
        c = float(torch.corrcoef(torch.stack([o, img]))[0, 1].abs()) if float(o.std()) > 1e-6 else 0.0
        by_role[role].append(c); by_mod[(mod, role)].append(c)
        if role in ("fore", "faint"):
            fore_tot += float(occ.sum()); fore_inv += float(occ.sum()) * (c < 0.1)
    if fore_tot > 0:
        inv_fore.append(fore_inv / fore_tot)
    SC._scene_cache().clear(); torch.cuda.empty_cache()
    print(f"scene {i}: {mod}", flush=True)
q = lambda v: f"median {np.median(v):.2f} (25 % {np.percentile(v, 25):.2f}, share < 0.1: {np.mean(np.array(v) < 0.1) * 100:.0f} %, n {len(v)})"
print("\nvisibility, |corr(population, image)|, high-passed:")
for role in ("fore", "faint", "matrix", "ground"):
    if by_role[role]:
        print(f"  {role:7s} {q(by_role[role])}")
for rl in ("fore", "faint"):
    print(f"\n{rl} by modality:")
    for (mod, role), v in sorted(by_mod.items()):
        if role == rl:
            print(f"  {mod:12s} {q(v)}")
print(f"\nshare of foreground material in invisible populations (|corr| < 0.1): median {np.median(inv_fore) * 100:.0f} %, mean {np.mean(inv_fore) * 100:.0f} % over {len(inv_fore)} scenes")
