"""A contact sheet of training crops as the trainer draws them: random image strength and difficulty level per crop,
middle slice of each 64^3 crop, titled with the modality and strength (synthetic only).

    PYTHONPATH=src python scripts/figure_training_variety.py OUT.png [N] [profile]   (env SEED0)
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False
from shapetensor.learn import scene as SC, lobe_train as LT

out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 64; prof = sys.argv[3] if len(sys.argv) > 3 else None
dev = "cuda" if torch.cuda.is_available() else "cpu"; seed0 = int(os.environ.get("SEED0", 180_000_000))
if dev == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.4)))
rng = np.random.default_rng(seed0)
base = dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA)
if prof:
    base["profile"] = prof
nc = int(math.ceil(math.sqrt(N))); nr = int(math.ceil(N / nc))
fig, axs = plt.subplots(nr, nc, figsize=(2.1 * nc, 2.3 * nr)); axs = np.atleast_2d(axs)
for i in range(nr * nc):
    ax = axs.flat[i]; ax.axis("off")
    if i >= N:
        continue
    u = float(rng.random()); st = u; lv = 0.4 + 0.6 * u                                               # as --strength-mix draws them
    if os.environ.get("STRENGTH"):                                                                  # (env STRENGTH: every crop at that strength)
        st = float(os.environ["STRENGTH"])
    d = None
    for k in range(4):                                                                              # an occasional draw asks for tens of GB
        try:                                                                                        # (the trainer retries too): the next seed
            s, d = SC.draw_scene((64,) * 3, seed=seed0 + 8 * i + 1000 * k, device=dev, strength=st, entity_ctx=False, **LT.difficulty_gen(lv, base))
            break
        except torch.OutOfMemoryError:
            SC._scene_cache().clear(); torch.cuda.empty_cache()
    if d is None:
        continue
    img = d["image"].float().cpu().numpy() / 255
    rec = d.get("record", {}) or {}
    ax.imshow(img[32], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    mod = rec.get("modality", rec.get("physics", "?"))
    ax.set_title(f"{mod}  s={st:.2f}", fontsize=6)
    SC._scene_cache().clear()
    if dev == "cuda":
        torch.cuda.empty_cache()
fig.suptitle(f"Training crops as drawn ({'profile ' + prof if prof else 'universal generator'}): random strength and difficulty, middle slice of each 64^3 crop", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=110); print("saved", out)
