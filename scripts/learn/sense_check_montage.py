"""Montages of the sense-check collection, one page per category (brain, body organs, vascular, plants, animals, cells,
subcellular): per source the middle slice of a crop and an orthogonal slice through it.

    python scripts/learn/sense_check_montage.py --data DIR --out PREFIX [--cols 6] [--crop 0]

Writes ``PREFIX_<category>.png`` and ``PREFIX_all.png`` (one tile per source). A catalogue entry's ``category`` wins
over the keyword rules in :func:`category`.
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = ("brain", "body organs", "vascular", "plants", "animals", "cells", "subcellular")
PLANT = ("leaf", "wood", "xylem", "flower", "nut", "seed", "root", "pollen", "stem", "plant", "arabidopsis", "maize")
VESSEL = ("arter", "vessel", "angio", "coronar", "vascul", "capillar")
BRAIN = ("brain", "cortex", "white matter", "neuropil", "nerve", "cerebell", "hippocamp", "spinal", "retina", "neuron")
BODY = ("liver", "kidney", "pancrea", "spleen", "lung", "heart", "abdom", "thorax", "muscle", "intestin", "colon",
        "stomach", "skin", "bone", "knee", "tooth", "airway", "breast", "prostate", "bladder")


def _um(meta) -> float:
    v = meta.get("voxel_um")
    try:
        return float(v)
    except (TypeError, ValueError):
        try:
            return float(str(v).split("x")[0])
        except ValueError:
            return float("nan")


def category(meta: dict) -> str:
    if meta.get("category") in CATEGORIES:
        return meta["category"]
    head = str(meta.get("tissue", "")).split("(")[0].split(":")[0].split(";")[0]                # the main tissue, not its listed parts
    txt = " ".join((str(meta.get("name", "")), str(meta.get("organism", "")), head, str(meta.get("modality", "")))).lower()
    um = _um(meta)
    if um < 0.1 or any(k in txt for k in ("cryo-et", "tomogram", "fib-sem", "electron", " em ", "sem", "tem")) and um < 0.2:
        return "subcellular"
    if any(k in txt for k in PLANT):
        return "plants"
    if any(k in txt for k in VESSEL) or any(k in str(meta.get("name", "")).lower() for k in ("mra_", "angio", "arter", "coronar")):
        return "vascular"
    if any(k in txt for k in ("light-sheet", "confocal", "two-photon", "fluoresc", "lightsheet", "expansion")) and um < 2:
        return "cells"
    human = any(k in str(meta.get("organism", "")).lower() for k in ("human", "homo", "macaque", "mouse", "mus ", "rat"))
    if any(k in txt for k in BRAIN):
        return "brain"
    if any(k in txt for k in BODY) and (human or "pig" in txt):
        return "body organs"
    return "animals"


def _show(a, img, title=None):
    lo, hi = np.percentile(img, [0.5, 99.5])
    a.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray"); a.axis("off")
    if title:
        a.set_title(title, fontsize=7, loc="left")


def page(items, out: Path, cols: int, title: str):
    n = len(items); rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, 2 * cols, figsize=(2.0 * 2 * cols, 2.35 * rows), squeeze=False)
    for a in ax.flat:
        a.axis("off")
    for k, (meta, x) in enumerate(items):
        r, c = divmod(k, cols); x = x.astype(np.float32)
        t = f"{meta['name'][:34]}\n{str(meta.get('organism', ''))[:18]} | {str(meta.get('modality', ''))[:22]}\n{str(meta.get('tissue', ''))[:30]} | {meta.get('voxel_um', '')} um"
        _show(ax[r, 2 * c], x[x.shape[0] // 2], t); _show(ax[r, 2 * c + 1], x[:, x.shape[1] // 2])
    fig.suptitle(f"{title}  ({n} sources; per source: a slice of a 96^3 crop and the orthogonal slice through it)", fontsize=10, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (2.35 * rows))); fig.savefig(out, dpi=75); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=5); ap.add_argument("--crop", type=int, default=0)
    a = ap.parse_args()
    first = OrderedDict()
    for f in sorted(Path(a.data).glob("*.npz")):
        z = np.load(f); meta = json.loads(str(z["meta"])); idx = int(f.stem.rsplit("_", 1)[1])
        if meta["name"] not in first or idx == a.crop:
            first[meta["name"]] = (meta, z["image"])
    groups = OrderedDict((c, []) for c in CATEGORIES)
    for meta, x in first.values():
        groups[category(meta)].append((meta, x))
    for c, items in groups.items():
        items.sort(key=lambda t: _um(t[0]))
        if items:
            page(items, Path(f"{a.out}_{c.replace(' ', '_')}.png"), a.cols, c)
    counts = {c: len(v) for c, v in groups.items()}
    json.dump({c: [m["name"] for m, _ in v] for c, v in groups.items()}, open(f"{a.out}_categories.json", "w"), indent=1)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
