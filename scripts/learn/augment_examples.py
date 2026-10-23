"""Examples of the contrast routes: one geometry through many routes, and random training inputs as the trainer sees them.

    python scripts/learn/augment_examples.py [--out DIR]

1. ``routes_one_scene.png``: one 96³ crop (middle slice) -- its target (the exact density's top direction, colour by
   direction, brightness by mass), then the same geometry through each image formation, the draw-time routes (a
   physics blend, an orientation-dependent term, per-population scrambling) and the batch-time routes (a non-monotone
   and a spatially varying transfer function, relief, unsharp / Laplacian edges, missing lines, missing blocks), and
   four full random draws. Every panel has the same target.
2. ``training_inputs.png``: 24 random training inputs at full difficulty, exactly as the trainer sees them (scene,
   physics, routes, acquisition, batch-time route), titled with the voxel size, physics and routes.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn import augment as AU
from shapetensor.learn.scene import Scene, draw_scene

import os
DEV = os.environ.get("ST_DEV", "cuda")
CANVAS = int(os.environ.get("ST_CANVAS", "192"))


def only(op: str, x: torch.Tensor, seed: int) -> torch.Tensor:
    """``x`` through one batch-time operation (forced on, the rest off)."""
    saved = dict(AU.P)
    try:
        for k in AU.P:
            AU.P[k] = 0.0
        AU.P[op] = 1.0
        if op == "spatial":
            AU.P["lut"] = 1.0
        if op == "nonmono":
            AU.P["lut"] = 1.0; AU.P["nonmono"] = 1.0
        return AU.contrast_route(x, np.random.default_rng(seed), 1.0)
    finally:
        AU.P.update(saved)


def target_rgb(t: dict, size: int) -> np.ndarray:
    """The target density's top direction per cell (DEC colour x mass), middle cell slice upsampled to voxels."""
    from shapetensor.learn.lobe_train import sh_peaks
    ct = t["sh_t"][:45].float()
    D, V = sh_peaks(ct, 8)
    cs = ct.shape[1:]; d = D[0].reshape(3, *cs).abs(); m = (ct[0] / math.sqrt(4 * math.pi)).clamp(0, 1)
    rgb = (d * (m / 0.3).clamp(max=1)[None] * (V[0].reshape(cs) > 0)[None]).permute(1, 2, 3, 0)[cs[0] // 2]
    return np.kron(rgb.cpu().numpy(), np.ones((4, 4, 1)))


def pick_seed(fam, canvas, size, n_groups, voxel_um, tries: int = 12, start: int = 4321) -> int:
    """A scene whose central crop's middle slice is full of structure (the largest shape occupancy among a few)."""
    from shapetensor.learn.scene import render_scene
    best, best_v = start, -1.0
    c0 = (canvas - size) // 2
    for sd in range(start, start + tries):
        sc, ren, _, _ = render_scene((canvas,) * 3, 7_000_000 + sd, DEV, voxel_um, fam, tuple(n_groups), 0.3, "general", None)
        occ = sum(g["occupancy"][c0 + size // 2, c0:c0 + size, c0:c0 + size] for g in ren["groups"])
        v = float((occ > 0.3).float().mean()) if torch.is_tensor(occ) else 0.0
        if v > best_v:
            best, best_v = sd, v
    print("picked scene", best, "slice coverage", round(best_v, 3), flush=True)
    return best


def main_routes(out: Path, seed: int | None = None):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from capability_bank import _set_lobe_targets
    from shapetensor.learn import train as TR
    _set_lobe_targets()
    size, canvas = 96, CANVAS
    fam = {k: (2.0 if k == "bundle" else 1.0 if k in ("cells", "tree") else 0.0) for k in Scene.FAMILIES}
    seed = seed if seed is not None else pick_seed(fam, canvas, size, [3, 4], 4.0)
    common = dict(canvas=canvas, crops_per_scene=1, families=fam, n_groups=[3, 4], voxel_um=4.0, p_warp=0.0, crop=((canvas - size) // 2,) * 3, modality="mixed")
    img0, t0 = TR._make_pair(seed, size, DEV, 0.15, {"scene": {**common, "physics": "additive", "p_blend": 0, "p_aniso": 0, "p_scramble": 0}})
    panels = [("target: top direction x mass", target_rgb(t0, size)), ("additive (the old compositor)", img0[size // 2].float().cpu().numpy())]
    for ph in ("xray", "darkfield", "histology", "pli", "mri"):
        s, d = draw_scene((size,) * 3, seed=seed, device=DEV, strength=0.15, physics=ph, p_blend=0, p_aniso=0, p_scramble=0, **common)
        panels.append((ph, d["image"][size // 2].float().cpu().numpy()))
    for name, kw in (("blend (+ a second physics)", dict(p_blend=1.0, p_aniso=0, p_scramble=0)), ("orientation term", dict(p_blend=0, p_aniso=1.0, p_scramble=0)),
                     ("scramble (per population)", dict(p_blend=0, p_aniso=0, p_scramble=1.0))):
        s, d = draw_scene((size,) * 3, seed=seed, device=DEV, strength=0.15, physics="xray", **kw, **common)
        panels.append((f"{name}: {', '.join(d['record']['routes'])}", d["image"][size // 2].float().cpu().numpy()))
    base = img0.float() / 255
    for op, title in (("nonmono", "non-monotone transfer fn"), ("spatial", "spatially varying transfer fn"), ("relief", "relief (DIC-like)"),
                      ("unsharp", "unsharp / blur"), ("log", "Laplacian edges"), ("lines", "missing lines"), ("blocks", "missing blocks")):
        panels.append((title, only(op, base, 11)[size // 2].cpu().numpy()))
    for k in range(3):
        rng = np.random.default_rng(100 + k); rec = {}
        s, d = draw_scene((size,) * 3, seed=seed, device=DEV, strength=0.6, acq_seed=500 + k, **{**common, "modality": None})
        y = AU.contrast_route(d["image"].float() / 255, rng, 1.0, rec=rec)
        panels.append((f"full route {k + 1}: {d['record']['physics']} + {', '.join(rec)}"[:60], y[size // 2].cpu().numpy()))
    n = len(panels); c = 5; r = (n + c - 1) // c
    fig, ax = plt.subplots(r, c, figsize=(3.4 * c, 3.6 * r))
    for a_, (t, im) in zip(ax.flat, panels):
        a_.imshow(im, cmap=None if im.ndim == 3 else "gray"); a_.set_title(t, fontsize=8); a_.axis("off")
    for a_ in ax.flat[n:]:
        a_.axis("off")
    fig.suptitle("one geometry, one target, many contrast routes (middle slice of a 96³ crop)")
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / "routes_one_scene.png", dpi=60); plt.close(fig)


def training_inputs(out: Path, n: int = 24):
    from shapetensor.learn.lobe_train import difficulty_gen
    gen = difficulty_gen(1.0, {"canvas": CANVAS, "crops_per_scene": 1})
    fig, ax = plt.subplots(4, 6, figsize=(19, 13.5))
    for i, a_ in enumerate(ax.flat[:n]):
        rng = np.random.default_rng(7000 + i); rec = {}
        st = float(rng.uniform(0.3, 1.0))
        s, d = draw_scene((96,) * 3, seed=9_100_000 + 8 * i, device=DEV, strength=st, **gen)
        y = AU.contrast_route(d["image"].float() / 255, rng, 1.0, rec=rec)
        fams = ",".join(sorted({list(f)[0] for f in d["record"]["families"]}))[:28]
        a_.imshow(y[48].cpu().numpy(), cmap="gray")
        a_.set_title(f"{d['voxel_um']:.1f} um {d['record']['modality']}/{'+'.join(d['record']['routes'])}\n{fams} | {','.join(rec)[:34]}", fontsize=7)
        a_.axis("off")
    fig.suptitle("random training inputs at full difficulty (middle slices), as the trainer sees them")
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / "training_inputs.png", dpi=55); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/scene_generator")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.only or "routes" in a.only:
        main_routes(out); print("wrote routes_one_scene", flush=True)
    if not a.only or "inputs" in a.only:
        training_inputs(out); print("wrote training_inputs", flush=True)


if __name__ == "__main__":
    main()
