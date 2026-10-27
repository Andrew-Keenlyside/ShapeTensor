"""Outcome figures of the generator changes of 27 Oct (packed tissue, crop strata, narrow and related crossings, ...).

1. ``OUT_before_after.png``: random training crops (mid-slices of the acquired image) from the generator as it was this
   morning and as it is now, at the training's settings, and the share of crops by oriented-tissue fill (from
   ``gen_outcome.py``-style draws: ``--old`` / ``--new`` JSON + image pickles).
2. ``OUT_tissue.png``: the packed-tissue family through the training pipeline (``draw_scene``): per crop its
   populations, true fibre direction and acquired image, chosen to cover the crossing grains, the planar crossing, a
   resolved scene and the grey-matter variant (synthetic only: no real data, ever).

Usage: python scripts/learn/generator_outcomes.py OUT_STEM --old OLD_STEM --new NEW_STEM
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import scene as SC

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
a = ap.parse_args(); dev = "cuda"
TR.HEAD = "entities"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"], "font.size": 8})
GREY, BLUE, INK2 = "#a3a29c", "#2a78d6", "#52514e"


def win(x):
    q = np.percentile(x, [1, 99]); return np.clip((x - q[0]) / max(q[1] - q[0], 1e-6), 0, 1)


# ---- 1. before / after ----------------------------------------------------------------------------------------------
old, new = json.load(open(a.old + ".json")), json.load(open(a.new + ".json"))
oi, ni = pickle.load(open(a.old + "_imgs.pkl", "rb")), pickle.load(open(a.new + "_imgs.pkl", "rb"))
fig = plt.figure(figsize=(13, 7.0))
gs = fig.add_gridspec(5, 17, height_ratios=[1, 1, 1, 1, 2.4], hspace=0.06, wspace=0.06)
for side, (imgs, title, c0) in enumerate(((oi, "Before: the generator this morning", 0), (ni, "After: packed tissue, crop strata, narrow and related crossings", 9))):
    for j in range(16):                                                                      # pairs: the acquired image over its oriented fill
        im, ext, vu, cf, ofill = imgs[j][:5] if len(imgs[j]) >= 5 else (*imgs[j], None)
        for r_, (x_, cm) in enumerate(((win(im), "gray"), (ofill, "magma"))):
            ax = fig.add_subplot(gs[2 * (j // 8) + r_, c0 + j % 8])
            if x_ is not None:
                ax.imshow(x_, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r_ == 1:
                ax.text(2, 60, f"{ext * 100:.0f}%", color="white", fontsize=6, va="bottom")
            if j == 0 and r_ == 0:
                ax.set_title(title, fontsize=9, loc="left", pad=4)
ax = fig.add_subplot(gs[4, 1:8])
bins = [(0, 0.1, "< 10 %"), (0.1, 0.3, "10-30 %"), (0.3, 0.7, "30-70 %"), (0.7, 1.01, "> 70 %")]
for k, (d, col, lab) in enumerate(((old, GREY, "before"), (new, BLUE, "after"))):
    ex = np.array([c["ext"] for c in d["crops"]]); sh = [np.mean((ex >= lo) & (ex < hi)) * 100 for lo, hi, _ in bins]
    ax.bar(np.arange(4) + (k - 0.5) * 0.38, sh, 0.36, color=col, label=lab)
    for x_, v in zip(np.arange(4) + (k - 0.5) * 0.38, sh):
        ax.text(x_, v + 1, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
ax.set_xticks(np.arange(4)); ax.set_xticklabels([b[2] for b in bins]); ax.set_ylabel("% of training crops")
ax.set_xlabel("share of the crop within ~2 voxels of oriented tissue"); ax.legend(frameon=False, fontsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax2 = fig.add_subplot(gs[4, 9:17]); ax2.axis("off")
row = lambda d, f: f(d["crops"])
stats = [("oriented fill, median", lambda c: f"{np.median([x['ext'] for x in c]) * 100:.0f} %"),
         ("crossings below 40 deg", lambda c: (lambda aa: f"{np.mean(aa < 40) * 100:.0f} % of {len(aa)}" if len(aa) else "--")(np.array([y for x in c for y in x["angles"]]))),
         ("queries at crossing cores", lambda c: f"{np.mean([x['two_line_share'] for x in c]) * 100:.1f} %"),
         ("populations per query", lambda c: f"{np.mean([x['pops_per_query'] for x in c]):.2f}"),
         ("largest training window", lambda c: f"sigma {max(x['sigma_max'] for x in c):.1f}"),
         ("draw time per crop (median)", lambda c: f"{np.median([x['draw_s'] for x in c]):.2f} s")]
ax2.text(0.0, 0.98, "", transform=ax2.transAxes)
ax2.text(0.42, 0.98, "before", transform=ax2.transAxes, fontsize=8, fontweight="bold", color=INK2, va="top")
ax2.text(0.72, 0.98, "after", transform=ax2.transAxes, fontsize=8, fontweight="bold", color=BLUE, va="top")
for i, (lab, f) in enumerate(stats):
    y = 0.84 - i * 0.14
    ax2.text(0.0, y, lab, transform=ax2.transAxes, fontsize=8, va="top"); ax2.text(0.42, y, row(old, f), transform=ax2.transAxes, fontsize=8, va="top")
    ax2.text(0.72, y, row(new, f), transform=ax2.transAxes, fontsize=8, va="top")
fig.suptitle(f"Training crops ({len(old['crops'])} each, the training's settings): mid-slices of the acquired 64^3 image over its oriented-tissue fill (%: share of the crop)", fontsize=10, x=0.02, ha="left")
plt.savefig(a.out + "_before_after.png", dpi=110, bbox_inches="tight"); plt.close(fig); print("saved", a.out + "_before_after.png", flush=True)

# ---- 2. the packed-tissue family through the pipeline ---------------------------------------------------------------
SC.P_TISSUE = 1.0
want = {"fibre": None, "fascicle": None, "layers": None, "planar": None, "resolved": None, "grey": None}
picked = []
for seed in range(60):
    if all(v is not None for v in want.values()):
        break
    grey = seed % 6 == 5; SC.P_GREY_TISSUE = 1.0 if grey else 0.0
    vu = [4.0, 8.0, 20.0, 1.5][seed % 4] if not grey else 4.0
    s, d = SC.draw_scene((96,) * 3, seed=95_000_000 + 8 * seed, device=dev, canvas=128, crops_per_scene=8, voxel_um=vu, strength=0.35, p_centre=1.0,
                         p_warp=0.0, p_affine=0.0, p_cutout=0.0, p_boundary=0.0, p_sparse=0.0, entity_ctx=False)   # (no distortion: it mirrors the
                                                                                                    # crop's edge -- shown apart)
    rec = d["record"].get("tissue") or {}
    cz = rec.get("crossing") or []
    tag = "grey" if rec.get("grey") else ("planar" if any(z["planar"] for z in cz) else (cz[0]["grain"] if cz else ("resolved" if rec.get("resolved") else None)))
    if rec.get("resolved") and want["resolved"] is None and not rec.get("grey"):
        tag = "resolved"
    if tag in want and want[tag] is None:
        want[tag] = (s, d, vu, rec); print("picked", tag, seed, vu, flush=True)
    SC._scene_cache().clear()
order = [k for k in ("fibre", "fascicle", "layers", "planar", "resolved", "grey") if want[k] is not None]
cols = len(order)
fig, axs = plt.subplots(3, cols, figsize=(2.3 * cols, 7.2))
pal = plt.cm.tab10(np.arange(10))[:, :3]
for j, k in enumerate(order):
    s, d, vu, rec = want[k]; z = 48
    occ = d["occupancy"].float(); T = d["tensors"].float(); K = occ.shape[0]
    shp = [i for i, L in enumerate(s.layers) if L.role == "shape"]
    o_ = occ[shp, z].cpu().numpy(); pop = np.ones(o_.shape[1:] + (3,))
    for i in np.argsort(-o_.reshape(len(o_), -1).sum(1)):
        al = np.clip(o_[i], 0, 1)[..., None] * 0.85; pop = pop * (1 - al) + pal[i % 10] * al
    Tw = (occ[shp][:, None] * T[shp]).sum(0)[:, z]
    M = torch.stack([torch.stack([Tw[0], Tw[3], Tw[4]], -1), torch.stack([Tw[3], Tw[1], Tw[5]], -1), torch.stack([Tw[4], Tw[5], Tw[2]], -1)], -2)
    ev, V = torch.linalg.eigh(M.double().cpu() + 1e-9 * torch.eye(3, dtype=torch.float64)); v = V[..., -1].abs().numpy()
    ani = ((ev[..., 2] - ev[..., 1]) / ev.sum(-1).clamp_min(1e-9)).clamp(0, 1).numpy()
    dec = v[..., ::-1] * np.clip(occ[shp, z].sum(0).cpu().numpy(), 0, 1)[..., None] * ani[..., None]
    axs[0, j].imshow(pop); axs[1, j].imshow(np.clip(dec * 1.3, 0, 1)); axs[2, j].imshow(win(d["image"].float()[z].cpu().numpy()), cmap="gray")
    cz = rec.get("crossing") or [{}]
    sub = {"fibre": "crossing, fibre by fibre", "fascicle": "crossing by fascicles", "layers": "crossing in layers", "planar": "lines in one plane",
           "resolved": "resolved fibres", "grey": "grey-matter-like"}[k]
    ang = f", {cz[0]['angles'][0]:.0f} deg" if k in ("fibre", "fascicle", "layers") and cz[0].get("angles") else ""
    axs[0, j].set_title(f"{sub}{ang}\n{vu:g} um voxels, fibres {rec.get('fibre_um', 0):.2g} um", fontsize=8)
for ax in axs.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
for i, lab in enumerate(("populations", "true direction (rgb = zyx)", "acquired image")):
    axs[i, 0].set_ylabel(lab, fontsize=8)
fig.suptitle("The packed-tissue family through the training pipeline (draw_scene; mid-slices of 96^3 crops)", fontsize=10, x=0.02, ha="left")
plt.tight_layout(); plt.savefig(a.out + "_tissue.png", dpi=110); print("saved", a.out + "_tissue.png")
