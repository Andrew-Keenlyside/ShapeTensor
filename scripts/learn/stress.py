"""Stress test of the synthesis: a gallery of edge cases under every acquisition, and a quantitative sweep.

    python scripts/learn/stress.py [--out DIR] [--n 200] [--size 64]

Gallery (``stress_gallery_{1,2}.png``): one row per case -- clean (two orthogonal planes), the acquisitions (HiP-CT at
half and full strength, CT, MRI, histology), the labels (occupancy per population; directions as DEC × presence ×
orientation visibility of the full-strength HiP-CT draw) and the target's line-population count per cell.

Sweep (``stress_sweep.json``, ``stress_sweep.png``): ``--n`` random draws over modalities and strengths. Crashes and
non-finite values; label-image agreement (the labels against a structure tensor of the clean image at voxels holding
one population); whether orientation visibility tracks what a structure tensor can recover from the acquired image;
the distributions actually produced (crossing angles, radii, layer fill, populations per cell, interface share,
saturation, time per draw).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from examples import label_panels                                                              # noqa: E402

from shapetensor.filters import gaussian                                                        # noqa: E402
from shapetensor.learn.generator import (Layer, Sample, _composite, arc_field, boundary_sample, companion_field, density_for,  # noqa: E402
                                         direction_field, draw, eig3, nuisance, render_layer, sample)
from shapetensor.learn.targets import target                                                    # noqa: E402
from shapetensor.viz import dec_rgb                                                             # noqa: E402

DEV = "cuda"
ACQ = (("hipct", 0.5), ("hipct", 1.0), ("ct", 1.0), ("mri", 1.0), ("histology", 1.0))


def custom(shape, layers_spec, seed, style_value=110.0):
    """A sample from explicit layers: ``[(kind, field, radius, length, fraction, contrast), ...]`` on a flat background."""
    g = torch.Generator(device=DEV).manual_seed(seed); ones = torch.ones(shape, dtype=torch.bool, device=DEV); layers = []
    for kind, fld, r, L, phi, c in layers_spec:
        Lr = Layer(kind, fld, r, L, density_for(kind, r, L, phi), ones, c)
        Lr.occupancy, Lr.tensor = render_layer(Lr, shape, g, DEV); layers.append(Lr)
    bg = torch.full(shape, style_value, device=DEV) + 2 * torch.randn(shape, device=DEV, generator=g)
    R = torch.full(shape, style_value, device=DEV)
    img = _composite(bg, R, layers, DEV)
    return Sample(img, layers, bg, torch.zeros(shape, dtype=torch.long, device=DEV), "flat", 1.0, "general", {}, R)


def find(pred, kw, start=0, tries=200, size=96):
    """The first seed whose sample satisfies ``pred``."""
    for sd in range(start, start + tries):
        s = sample((size,) * 3, seed=sd, device=DEV, **kw)
        if pred(s):
            return s, sd
    raise RuntimeError(f"no seed in {tries} satisfies the case {kw}")


def cases(size):
    S = (size,) * 3; g = lambda k: torch.Generator(device=DEV).manual_seed(k)
    zx = torch.tensor([0.0, 0.0, 1.0], device=DEV).view(3, 1, 1, 1).expand(3, *S).contiguous()
    out = []
    out.append(("one straight population (flat)", custom(S, [("line", zx, 1.2, 60.0, 0.2, 60.0)], 1)))
    d1 = direction_field(S, 40.0, g(2), DEV)
    for th in (30.0, 60.0, 90.0):
        out.append((f"crossing {th:.0f}° (companion fields)", custom(S, [("line", d1, 1.0, 60.0, 0.12, 60.0), ("line", companion_field(d1, th, 40.0, g(3), DEV), 1.0, 60.0, 0.12, 60.0)], 3)))
    out.append(("faint population 1:10 at 60°", custom(S, [("line", d1, 1.0, 60.0, 0.2, 60.0), ("line", companion_field(d1, 60.0, 40.0, g(4), DEV), 1.0, 60.0, 0.02, 60.0)], 4)))
    out.append(("tight bend R = 8 voxels", custom(S, [("line", arc_field(S, 8.0, g(5), DEV), 1.0, 80.0, 0.3, 70.0)], 5)))
    out.append(("dense fascicle, fill 0.5", custom(S, [("line", d1, 1.0, 80.0, 0.5, 60.0)], 6)))
    out.append(("thin fibres inside a thick tract", custom(S, [("line", d1, 4.0, 80.0, 0.3, 30.0), ("line", d1, 0.7, 80.0, 0.15, 50.0)], 7)))
    s, _ = find(lambda s: s.record.get("crossing", {}).get("independent"), dict(crossing=True, k_layers=0, style="tissue"), size=size)
    out.append(("independent-field crossing (tissue)", s))
    s, _ = find(lambda s: s.record.get("crossing", {}).get("n") == 3, dict(crossing=True, k_layers=0, style="regions"), start=50, size=size)
    out.append(("three populations (regions)", s))
    out.append(("cell stain, laminar (histology content)", find(lambda s: s.style == "cells", dict(content="histology", crossing=False, k_layers=1), start=100, size=size)[0]))
    out.append(("small scale 0.5", sample(S, seed=11, device=DEV, scale=0.5, crossing=True, k_layers=1)))
    out.append(("large scale 2.5", sample(S, seed=12, device=DEV, scale=2.5, crossing=True, k_layers=1)))
    out.append(("MRI content", sample(S, seed=13, device=DEV, content="mri", crossing=True, k_layers=2)))
    out.append(("occluding layers", find(lambda s: any(L.occluding for L in s.layers), dict(crossing=False, k_layers=3), start=200, size=size)[0]))
    out.append(("hollow vessels", find(lambda s: any(L.hollow for L in s.layers), dict(crossing=False, k_layers=2, kinds=("line",)), start=300, size=size)[0]))
    out.append(("boundary + faint fibres (step 60, contrast 15)", boundary_sample(S, step=60.0, contrast=15.0, seed=7, device=DEV)))
    return out


def gallery(out_dir, size):
    cs = cases(size)
    per = 8
    for part in range(math.ceil(len(cs) / per)):
        rows = cs[part * per:(part + 1) * per]
        fig, axs = plt.subplots(len(rows), 10, figsize=(36, 3.7 * len(rows)), facecolor="white", squeeze=False)
        for r, (name, s) in enumerate(rows):
            ax = axs[r]; ky = size // 2
            ax[0].imshow(s.image[:, ky].cpu().numpy(), cmap="gray"); ax[0].set_title(f"{name}\nclean, z-x plane", fontsize=8)
            ax[1].imshow(s.image[ky].cpu().numpy(), cmap="gray"); ax[1].set_title("clean, y-x plane", fontsize=8)
            draws = [nuisance(s, seed=1000 + 10 * r + j, device=DEV, modality=m, strength=sg) for j, (m, sg) in enumerate(ACQ)]
            for j, ((m, sg), d) in enumerate(zip(ACQ, draws)):
                keys = [k for k in d["record"] if k not in ("modality", "gamma", "psf", "window", "visibility", "orientation_visibility", "axis")]
                ax[2 + j].imshow(d["image"][:, ky].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
                pv = [v for v, L in zip(d["record"]["visibility"], s.layers) if L.role == "shape"]
                ov = [v for v, L in zip(d["record"]["orientation_visibility"], s.layers) if L.role == "shape"]
                ax[2 + j].set_title(f"{m} s={sg} (axis {'zyx'[d['record']['axis']]}): " + ", ".join(keys)[:44] + f"\npresence {pv}  orient {ov}", fontsize=6)
            label_panels(ax[7], ax[8], s, draws[1], ky)
            ax[7].set_title("labels: occupancy per population", fontsize=8); ax[8].set_title("labels: DEC × presence × orientation vis. (hipct s=1)", fontsize=8)
            t = target(s, draws[1]); c = ky // 4
            N = ((t["mass"] > 0.02) & (t["kind"] == 0)).sum(0)[:, c].cpu().numpy()
            ax[9].imshow(N, cmap=matplotlib.colors.ListedColormap(["#1b1b1b", "#2a78d6", "#e34948", "#f0b429", "#ffffff"]), vmin=-0.5, vmax=4.5, interpolation="nearest")
            ax[9].set_title("target: line populations per cell (hipct s=1)\nblack 0 · blue 1 · red 2 · yellow 3 · white 4", fontsize=8)
            for a_ in ax:
                a_.set_xticks([]); a_.set_yticks([])
            print(f"gallery: {name}", flush=True)
        plt.tight_layout(); fig.savefig(out_dir / f"stress_gallery_{part + 1}.png", dpi=60); plt.close(fig)


def st_axes(img, s_int=2.0):
    """A gradient structure tensor: the least-gradient axis (a line's), the most-gradient axis (a sheet's normal), and
    its line coherence ``(λ_mid - λ_min) / λ_max`` (high where one direction is clearly the line's)."""
    g = [gaussian(img, 1.0, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    T6 = torch.stack([gaussian(g[i] * g[j], s_int) for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
    val, V = eig3(T6)
    return V[0], V[2], (val[1] - val[0]) / val[2].clamp_min(1e-12)


def sweep(out_dir, n, size):
    rng = np.random.default_rng(0); rec = {"crashes": [], "nonfinite": 0, "time_s": [], "agree_clean_line": [], "agree_clean_sheet": [],
                                          "acq_err_by_ovis": {"<0.2": [], "0.2-0.8": [], ">0.8": []}, "theta": [], "radius": [], "fill": [],
                                          "N_cells": [0] * 7, "interface_share": [], "saturation": [], "presence": [], "orientation": []}
    angle = lambda a, b: torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1)))
    for i in range(n):
        m = ("hipct", "ct", "mri", "histology", "mixed")[i % 5]; sg = (0.5, 1.0)[(i // 5) % 2]
        t0 = time.time()
        try:
            s, d = draw((size,) * 3, seed=50_000 + i, device=DEV, modality=m, strength=sg); t = target(s, d); torch.cuda.synchronize()
        except Exception as e:
            rec["crashes"].append({"seed": 50_000 + i, "modality": m, "error": repr(e), "trace": traceback.format_exc()[-800:]}); continue
        rec["time_s"].append(time.time() - t0)
        dw = nuisance(s, seed=50_000 + i, device=DEV, modality=m, strength=sg, warp=False)            # the visibility check needs one frame
        if not (torch.isfinite(d["image"]).all() and torch.isfinite(d["tensors"]).all() and torch.isfinite(t["slots"]).all()):
            rec["nonfinite"] += 1
        occ = torch.stack([L.occupancy for L in s.layers]); shapes = [k for k, L in enumerate(s.layers) if L.role == "shape"]
        others = occ[shapes].sum(0) if shapes else torch.zeros_like(occ[0])
        inner = torch.zeros_like(occ[0], dtype=torch.bool); inner[6:-6, 6:-6, 6:-6] = True
        for k in shapes:
            L = s.layers[k]; one = (occ[k] > 0.5) & (others - occ[k] < 0.05) & inner
            if int(one.sum()) < 50:
                continue
            si = max(2.0, 1.5 * L.radius)                                                           # the integration matched to the shape's size
            lv, V = eig3(L.tensor)
            if L.kind == "line":
                lin = (lv[2] - lv[1]) / lv[2].clamp_min(1e-9)
                g_line, _, g_coh = st_axes(L.occupancy, si)                                          # the layer's own geometry: no clutter
                q = one & (lin > 0.5) & (g_coh > 0.2)
                if int(q.sum()) >= 50:
                    rec["agree_clean_line"].append(float(angle(V[2], g_line)[q].median()))
                c_line, _, c_coh = st_axes(s.image.float(), si); b_line, _, _ = st_axes(dw["image"].float(), si)   # dw: the same acquisition, unwarped
                ov = dw["orientation_visibility"][k]
                err = angle(V[2], b_line)
                base = one & (lin > 0.5) & (c_coh > 0.2) & (angle(V[2], c_line) < 20)                 # where the clean image shows it
                for key, sel in (("<0.2", ov < 0.2), ("0.2-0.8", (ov >= 0.2) & (ov <= 0.8)), (">0.8", ov > 0.8)):
                    q = base & sel
                    if int(q.sum()) >= 50:
                        rec["acq_err_by_ovis"][key].append(float(err[q].median()))
            elif L.kind == "sheet":
                pla = (lv[1] - lv[0]) / lv[2].clamp_min(1e-9)
                _, g_norm, _ = st_axes(L.occupancy, si)
                q = one & (pla > 0.5)
                if int(q.sum()) >= 50:
                    rec["agree_clean_sheet"].append(float(angle(V[0], g_norm)[q].median()))
            rec["radius"].append(L.radius); rec["fill"].append(float(L.occupancy[L.region].mean()))
            rec["presence"].append(float((d["visibility"][k] * occ[k]).sum() / occ[k].sum().clamp_min(1e-6)))
            rec["orientation"].append(float((d["orientation_visibility"][k] * occ[k]).sum() / occ[k].sum().clamp_min(1e-6)))
        cr = s.record.get("crossing")
        if cr and cr.get("theta_deg") is not None:
            rec["theta"].append(cr["theta_deg"])
        N = ((t["mass"] > 0.02) & (t["kind"] == 0)).sum(0).flatten().cpu().numpy()
        for v in range(7):
            rec["N_cells"][v] += int((N == v).sum())
        iface = [k for k, L in enumerate(s.layers) if L.role == "interface"]
        rec["interface_share"].append(float(t["classes"][3].mean()))
        rec["saturation"].append(float(((d["image"] == 0) | (d["image"] == 255)).float().mean()))
        if (i + 1) % 25 == 0:
            print(f"sweep: {i + 1}/{n}", flush=True)
    summ = lambda v: {"n": len(v), "median": float(np.median(v)) if v else None, "p10": float(np.percentile(v, 10)) if v else None, "p90": float(np.percentile(v, 90)) if v else None}
    out = {"draws": n, "crashes": len(rec["crashes"]), "crash_examples": rec["crashes"][:3], "nonfinite": rec["nonfinite"],
           "time_s": summ(rec["time_s"]), "label_vs_own_geometry_line_deg": summ(rec["agree_clean_line"]), "label_vs_own_geometry_sheet_deg": summ(rec["agree_clean_sheet"]),
           "acquired_ST_error_by_orientation_visibility_deg": {k: summ(v) for k, v in rec["acq_err_by_ovis"].items()},
           "crossing_theta_deg": summ(rec["theta"]), "radius_vox": summ(rec["radius"]), "layer_fill": summ(rec["fill"]),
           "line_populations_per_cell": {str(v): c / max(sum(rec["N_cells"]), 1) for v, c in enumerate(rec["N_cells"])},
           "interface_share": summ(rec["interface_share"]), "saturation": summ(rec["saturation"]),
           "presence_visibility": summ(rec["presence"]), "orientation_visibility": summ(rec["orientation"])}
    json.dump(out, open(out_dir / "stress_sweep.json", "w"), indent=1)
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), facecolor="white")
    hist = [("label vs clean ST, lines (deg)", rec["agree_clean_line"]), ("label vs clean ST, sheets (deg)", rec["agree_clean_sheet"]),
            ("crossing angle (deg)", rec["theta"]), ("radius (voxels)", rec["radius"]), ("layer fill", rec["fill"]),
            ("presence visibility", rec["presence"]), ("orientation visibility", rec["orientation"]), ("saturated share", rec["saturation"])]
    for a_, (name, v) in zip(axs.flat, hist):
        a_.hist(v, bins=30, color="#2a78d6"); a_.set_title(f"{name} (n={len(v)})", fontsize=10)
    plt.tight_layout(); fig.savefig(out_dir / "stress_sweep.png", dpi=70); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/learn_stress")
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--size", type=int, default=64); ap.add_argument("--gallery-size", type=int, default=96)
    ap.add_argument("--skip-gallery", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.skip_gallery:
        gallery(out, a.gallery_size)
    print(json.dumps(sweep(out, a.n, a.size), indent=1))


if __name__ == "__main__":
    main()
