"""Population identities on the phantoms: lag-ODF fixels (self response, momentum) labelled by
``lagodf.label_populations``, scored against each truth population and drawn on the mid-y (z, x) plane.

Per truth population (the bend's three fascicles; the bands of the band crossings and the T-junction; the two
populations of the intermixed crossings): its fixels = the fixels of its core cells within 20° of its direction; the
**dominant label** among them; **coverage** = the share of its cells holding a fixel with that label; **consistency**
= the share of its fixels carrying it; **purity** = the share of that label's fixels (core) lying in the population's
cells (dilated by one cell). **separate** = every truth population has a different dominant label.

    python scripts/test_population_labels.py [OUT_DIR] [--max-angle 20] [--min-fixels 20]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi

from shapetensor.lagodf import LagODF, label_populations
from shapetensor.phantoms import BEND_GRID, Grid, build, direction
from shapetensor.viz import identity_palette, identity_rgb

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("tp", str(HERE / "test_populations.py"))
TP = importlib.util.module_from_spec(spec); spec.loader.exec_module(TP)
DEV = "cuda"
CASES = (("bend", "bend", {}), ("straight", "bend", {"straight": True}), ("band crossing 90", "band_crossing", {"cross_deg": 90.0}),
         ("band crossing 60", "band_crossing", {"cross_deg": 60.0}), ("T-junction 90", "band_crossing", {"cross_deg": 90.0, "t_junction": True}),
         ("crossing 90", "crossing", {"cross_deg": 90.0}), ("crossing 60", "crossing", {"cross_deg": 60.0}))


def truths(kind, tr, meta, cs):
    """``[(name, mask (cz, cx) bool, direction (3, cz, cx))]`` on the cell lattice's (z, x) plane."""
    Z, X = tr["npop"].shape
    zc = np.clip(np.arange(cs[0]) * 4 + 2, 0, Z - 1); xc = np.clip(np.arange(cs[2]) * 4 + 2, 0, X - 1)
    at = lambda a: a[np.ix_(zc, xc)]
    d1 = np.moveaxis(np.nan_to_num(at(tr["dir1"])), -1, 0)
    if kind == "bend":
        fid = at(tr["fid"]); return [(f"fascicle {k}", fid == k, d1) for k in (1, 2, 3)]
    if kind == "band_crossing":
        a1, a2 = meta["angles"]
        full = lambda a: np.broadcast_to(np.asarray(direction(a), np.float32)[:, None, None], (3,) + d1.shape[1:])
        return [("band 1", at(tr["band1"]), full(a1)), ("band 2", at(tr["band2"]), full(a2))]
    d2 = np.moveaxis(np.nan_to_num(at(tr["dir2"])), -1, 0)
    return [("population 1", at(tr["npop"]) >= 1, d1), ("population 2", at(tr["npop"]) >= 2, d2)]


def score(labels, dirs, truth, inner, tol=20.0):
    K = labels.shape[0]; lab = labels.cpu().numpy(); D = dirs.cpu().numpy()
    out = {}; dominant = []
    for name, m2, d in truth:
        m = np.broadcast_to(m2[:, None, :], lab.shape[1:]) & inner
        cosv = np.abs(np.einsum("kc...,c...->k...", D, np.broadcast_to(d[:, :, None, :], (3,) + lab.shape[1:])))
        fx = (lab > 0) & (cosv >= np.cos(np.radians(tol))) & m[None]
        vals = lab[fx]
        if vals.size == 0:
            out[name] = {"cells": int(m.sum()), "dominant": 0, "coverage": 0.0}; dominant.append(0); continue
        dom = int(np.bincount(vals).argmax()); dominant.append(dom)
        cov = float(((lab == dom) & fx).any(0)[m].mean())
        cons = float((vals == dom).mean())
        near = ndi.binary_dilation(m, iterations=1)
        own = (lab == dom) & inner[None]
        pur = float((own & near[None]).sum() / max(own.sum(), 1))
        out[name] = {"cells": int(m.sum()), "dominant": dom, "coverage": round(cov, 3), "consistency": round(cons, 3), "purity": round(pur, 3)}
    out["separate"] = len(set(dominant)) == len(dominant) and 0 not in dominant
    out["labels in core"] = int(len(np.unique(lab[:, inner][lab[:, inner] > 0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/population_labels")
    ap.add_argument("--max-angle", type=float, default=20.0); ap.add_argument("--min-fixels", type=int, default=20)
    ap.add_argument("--conflict-tol", type=float, default=0.02)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)
    tex, texture = TP.P.tex_for("calibrated")
    model = LagODF(momentum=True, device=DEV)
    res = {"settings": vars(a), "cases": {}}
    fig, axs = plt.subplots(2, len(CASES), figsize=(3.1 * len(CASES), 6.6), facecolor="black")
    for j, (name, kind, kw) in enumerate(CASES):
        img, tr, meta = build(kind, 1, tex=tex, texture=texture, device=DEV, **kw)
        v = torch.as_tensor(img, device=DEV).float(); grid = BEND_GRID if kind in ("bend", "band_crossing") else Grid()
        o = model.populations(v)
        labels, table = label_populations(o["dirs_cells"], o["mass"], max_angle_deg=a.max_angle, min_fixels=a.min_fixels, conflict_tol=a.conflict_tol)
        cs = tuple(o["N"].shape)
        inner = np.zeros(cs, bool); zs, ys, xs = grid.inner
        inner[zs.start // 4:zs.stop // 4, ys.start // 4:ys.stop // 4, xs.start // 4:xs.stop // 4] = True
        r = score(labels, o["dirs_cells"], truths(kind, tr, meta, cs), inner)
        r["largest labels"] = table[:6]
        res["cases"][name] = r; log(name, json.dumps({k: v_ for k, v_ in r.items() if k != "largest labels"}))
        yc = cs[1] // 2; yv = yc * 4 + 2
        g = img[:, yv, :].astype(np.float32); g = np.clip((g - np.percentile(g, 1)) / max(np.ptp(np.percentile(g, [1, 99])), 1e-6), 0, 1)
        lab = labels[:, :, yc, :].cpu().numpy()
        pal = identity_palette(max(int(labels.max()), 1))
        axs[0, j].imshow(g, cmap="gray"); axs[0, j].set_title(name, color="white", fontsize=10)
        axs[1, j].imshow(identity_rgb(lab, 4, pal, shape=g.shape))
        for ax in axs[:, j]:
            ax.add_patch(plt.Rectangle((xs.start - 0.5, zs.start - 0.5), xs.stop - xs.start, zs.stop - zs.start, fill=False, ec="#888888", lw=0.6))
            ax.set_xticks([]); ax.set_yticks([])
        sep = "separate" if r["separate"] else "MERGED"
        cov = ", ".join(f"{r[t]['coverage']:.2f}" for t in r if isinstance(r[t], dict))
        axs[1, j].set_title(f"{sep}; coverage {cov}", color="white", fontsize=8)
        del o, labels; torch.cuda.empty_cache()
        json.dump(res, open(out / f"population_labels_tol{a.conflict_tol}.json", "w"), indent=1)
    fig.suptitle("Population identities (one colour per population; a cell with two split on the diagonal), mid-y (z, x) plane; "
                 "grey box = scored core", color="white", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / f"population_labels_tol{a.conflict_tol}.png", dpi=110, facecolor="black")
    log("wrote", out / f"population_labels_tol{a.conflict_tol}.png")


if __name__ == "__main__":
    main()
