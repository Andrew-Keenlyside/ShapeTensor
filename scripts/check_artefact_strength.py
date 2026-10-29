"""Each acquisition artefact alone, at two strengths, on clean synthetic scenes built of objects: how much of the clean
image survives (|correlation| of the lightly blurred clean and corrupted images; 1 = untouched). An artefact that leaves
little at a moderate strength makes the targets unreadable there. Synthetic only.

    PYTHONPATH=src python scripts/check_artefact_strength.py [N_SCENES]   (env STRENGTHS, default 0.3,1.0; EXTREME=1: the wider
    pool's corruptions, each forced on alone, instead of the profile's; MODALITY=1: each modality's whole profile as drawn in
    training, many seeds)
"""
import os
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4; STR = [float(x) for x in os.environ.get("STRENGTHS", "0.3,1.0").split(",")]
MOD = bool(os.environ.get("MODALITY"))
if not MOD:
    G.CROSS_MIX = 0.0; G.CROSS_KEY = {}
G.SIGNAL_REDRAW = 0.0
keys = sorted(G.MODALITIES["hipct"].keys()); EXT = bool(os.environ.get("EXTREME"))
xkeys = sorted(G.EXTREME_P.keys()); x0 = dict(G.EXTREME_P)
if EXT:
    G.EXTREME_MOD = {}
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
blur = lambda x: (lambda b: (b - gaussian(b[None, None], 6.0)[0, 0]).flatten())(gaussian(x.float()[None, None], 1.0)[0, 0])   # local structure:
                                                                                                   # smooth shading (a bias field) removed
corr = lambda a, b: float(torch.corrcoef(torch.stack([a, b]))[0, 1].abs())
res = {k: {S: [] for S in STR} for k in ([m_ for m_ in G.MODALITIES if not m_.startswith("_")] if MOD else sorted(G.EXTREME_P.keys()) if os.environ.get("EXTREME") else keys)}
for i in range(N):
    s, d = SC.draw_scene((64,) * 3, seed=320_000_000 + 8 * i, device=dev, strength=0.0, entity_ctx=False, **gen)
    clean = blur(s.image)
    if MOD:
        for m_ in res:
            for S in STR:
                for j in range(4):
                    o = G.nuisance(s, seed=5000 + 31 * i + j, device=dev, strength=S, modality=m_, warp=False, fast=True)
                    res[m_][S].append(corr(clean, blur(o["image"])))
        SC._scene_cache().clear(); torch.cuda.empty_cache(); continue
    for k in (xkeys if EXT else keys):
        if EXT:
            G.EXTREME_P = {kk: (100.0 if kk == k else 0.0) for kk in xkeys}
        G.MODALITIES["_only_" + k] = {kk: (1.0 if (kk == k and not EXT) else 0.0) for kk in keys}
        for S in STR:
            try:
                o = G.nuisance(s, seed=1000 + i, device=dev, strength=S, modality="_only_" + k, warp=False, fast=True)
                res[k][S].append(corr(clean, blur(o["image"])))
            except Exception as e:
                res[k][S].append(float("nan")); print(k, S, type(e).__name__, str(e)[:80])
        del G.MODALITIES["_only_" + k]
    SC._scene_cache().clear(); torch.cuda.empty_cache()
print(f"{'artefact':14s} " + "  ".join(f"strength {S:g}" for S in STR) + "   (median |corr| with the clean image over scenes)")
for k in sorted(res, key=lambda k_: np.nanmedian(res[k_][STR[0]])):
    print(f"{k:14s} " + "  ".join(f"{np.nanmedian(res[k][S]):11.2f}" for S in STR) + ("   <-- strong at a moderate strength" if np.nanmedian(res[k][STR[0]]) < 0.7 else ""))
