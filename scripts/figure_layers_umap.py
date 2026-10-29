"""How detectable populations are in each of anatomix-dev-vit's last three layers: the decoder's three upsampling stages
(216 channels at 1/4 resolution, 120 at 1/2, the 32-channel output at full resolution).

One 128³ tile from each sample volume (straight bundles at the 12 calibration angles, fibre-free texture at grey- and
white-matter level, the 90° and 60° crossings; the phantoms' y extent reflect-padded to 128), one network pass with
forward hooks on the three layers, each upsampled (trilinear) to the tile; voxels sampled from the tile's central 64³.
Per layer: a UMAP of the samples (channels standardised, 50 principal components), coloured by the truth, and
cross-validated k-nearest-neighbour accuracies (k 15, 5 folds): the angle of single bundles (12 classes, and the median
angular error), fibre against texture, crossing against single bundle.

    python scripts/figure_layers_umap.py [OUT_PNG]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from shapetensor.features.devvit import load_devvit
from shapetensor.fibre_sets import iso_textures
from shapetensor.phantoms import Grid, build
from shapetensor.validate import ANGLES

import os
GRIDF = os.environ.get("GRIDF") == "1"                                           # remove the patch grid from every layer (box8 * box2)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/figures/"
                                                        + ("latent_layers_umap_devvit_gridfilter.png" if GRIDF else "latent_layers_umap_devvit.png"))
DEV = "cuda"; N_PER = 400
LAYERS = (("decoder 1: 216 ch, 1/4 resolution", "up_projection.decode.0"), ("decoder 2: 120 ch, 1/2 resolution", "up_projection.decode.1"),
          ("output: 32 ch, full resolution", "out_norm"))
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def hue(a):
    a = np.atleast_1d(np.asarray(a, float)); return hsv_to_rgb(np.stack([(a / 180.0) % 1, np.ones_like(a), np.ones_like(a)], -1))


def tile(v):
    """The central 128³ of a block, reflect-padded where the block is thinner."""
    out = v
    for ax in range(3):
        n = out.shape[ax]
        if n < 128:
            lo = (128 - n) // 2; hi = 128 - n - lo
            pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - ax)] = lo; pad[2 * (2 - ax) + 1] = hi
            out = Fn.pad(out[None, None], pad, mode="reflect")[0, 0]
        else:
            s = (n - 128) // 2; out = out.narrow(ax, s, 128)
    return out


def main():
    tex, texture = P.tex_for("calibrated"); cg = Grid(); rng = np.random.default_rng(0)
    vols = [(f"bundle {a:g}", a, torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float())
            for k, a in enumerate(ANGLES)]
    vols += [(f"texture {i}", np.nan, t) for i, t in enumerate(iso_textures("calibrated", DEV))]
    vols += [("crossing 90", np.nan, torch.as_tensor(build("crossing", 0, cg, tex, texture=texture, device=DEV, cross_deg=90.0)[0], device=DEV).float()),
             ("crossing 60", np.nan, torch.as_tensor(build("crossing", 0, cg, tex, texture=texture, device=DEV, cross_deg=60.0)[0], device=DEV).float())]
    net = load_devvit(DEV)
    store = {}
    hooks = [dict(net.named_modules())[name].register_forward_hook(lambda m, i, o, name=name: store.__setitem__(name, o.detach().float()))
             for _, name in LAYERS]
    idx = rng.integers(32, 96, size=(N_PER, 3))                                                   # voxels in the central 64³
    feats = {name: [] for _, name in LAYERS}; kind, ang = [], []
    for nm, a, v in vols:
        x = (tile(v) / 255.0).clamp(0, 1)[None, None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            net(x)
        for _, name in LAYERS:
            f = store[name]
            if f.shape[-1] != 128:
                f = Fn.interpolate(f, size=(128, 128, 128), mode="trilinear", align_corners=False)
            if GRIDF:
                k9 = torch.tensor([1, 2, 2, 2, 2, 2, 2, 2, 1], dtype=f.dtype, device=f.device) / 16.0
                g = f[0][:, None]
                for axd in range(3):
                    shp = [1, 1, 1, 1, 1]; shp[2 + axd] = 9
                    pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - axd)] = 4; pad[2 * (2 - axd) + 1] = 4
                    g = Fn.conv3d(Fn.pad(g, pad, mode="replicate"), k9.view(shp))
                f = g[:, 0][None]
            feats[name].append(f[0][:, idx[:, 0], idx[:, 1], idx[:, 2]].T.cpu().numpy())
        k = "bundle" if nm.startswith("bundle") else ("texture" if nm.startswith("texture") else nm)
        kind += [k] * N_PER; ang += [a] * N_PER
    for h in hooks:
        h.remove()
    kind = np.array(kind); ang = np.array(ang, float)
    import umap
    fig, axs = plt.subplots(1, len(LAYERS), figsize=(7.2 * len(LAYERS), 7.6), facecolor="white")
    res = {}
    for ax, (label, name) in zip(axs, LAYERS):
        X = np.concatenate(feats[name]); X = (X - X.mean(0)) / X.std(0).clamp(1e-6) if False else (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
        Xp = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
        knn = KNeighborsClassifier(n_neighbors=15)
        b = kind == "bundle"
        pred = cross_val_predict(knn, Xp[b], ang[b].astype(int), cv=5)
        err = np.abs(((pred - ang[b] + 90) % 180) - 90)
        fib = np.isin(kind, ["bundle", "crossing 90", "crossing 60"])
        acc_ft = float((cross_val_predict(knn, Xp, fib.astype(int), cv=5) == fib.astype(int)).mean())
        bc = np.isin(kind, ["bundle", "crossing 90", "crossing 60"]); cr = np.isin(kind[bc], ["crossing 90", "crossing 60"]).astype(int)
        acc_cr = float((cross_val_predict(knn, Xp[bc], cr, cv=5) == cr).mean())
        r = {"angle accuracy (12 classes)": round(float((pred == ang[b]).mean()), 3), "angle median error (deg)": round(float(np.median(err)), 1),
             "fibre vs texture accuracy": round(acc_ft, 3), "crossing vs single bundle accuracy": round(acc_cr, 3)}
        res[label] = r; print(label, json.dumps(r), flush=True)
        emb = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(Xp)
        col = np.zeros((len(kind), 3)); col[b] = hue(ang[b]); col[kind == "texture"] = 0.6
        col[kind == "crossing 90"] = 0.0; col[kind == "crossing 60"] = [0.45, 0.25, 0.1]
        for kd, mk, sz in (("bundle", "o", 3), ("texture", "o", 3), ("crossing 60", "^", 8), ("crossing 90", "s", 8)):
            m = kind == kd
            ax.scatter(emb[m, 0], emb[m, 1], c=col[m], s=sz, marker=mk, linewidths=0.2, edgecolors="w" if kd.startswith("crossing") else "none")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label}\nangle: {100 * r['angle accuracy (12 classes)']:.0f}% of 12 classes (median error {r['angle median error (deg)']}°); "
                     f"fibre vs texture {100 * r['fibre vs texture accuracy']:.0f}%; crossing vs bundle {100 * r['crossing vs single bundle accuracy']:.0f}%",
                     fontsize=9)
    fig.suptitle(("[patch grid removed: box8*box2 on every layer] " if GRIDF else "[single pass, no grid filter] ") + "anatomix-dev-vit, last three layers: UMAP of voxels (hue = bundle angle; texture grey, crossing 90 black, crossing 60 brown), "
                 "k-nearest-neighbour accuracies (k 15, 5-fold)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.94)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=100)
    json.dump(res, open(OUT.with_suffix(".json"), "w"), indent=1); print(OUT)


if __name__ == "__main__":
    main()
