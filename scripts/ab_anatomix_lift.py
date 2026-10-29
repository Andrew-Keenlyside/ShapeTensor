"""A/B: the lift from the intensity (A) against the lift from anatomix features (B1 enc2, B2 dev-vit), the chain after it
identical, on three simple phantoms; one QC image per phantom.

    python scripts/ab_anatomix_lift.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from shapetensor.features import make_engine
from shapetensor.field import Orientations
from shapetensor.phantoms import Grid, build, direction
from shapetensor.st import excision_weights
from shapetensor.validate import lattice_scores, tex_for
from shapetensor.viz import dec_rgb

DEV = "cuda"
CASES = (("bundle 60", "bundle", {"angle": 60.0}, Grid()), ("crossing 90", "crossing", {"cross_deg": 90.0}, Grid()),
         ("band crossing 90", "band_crossing", {"cross_deg": 90.0}, Grid(core=(192, 32, 192))))


def run_on(model, field, v, mask):
    """The chain on ``field`` (the intensity, or a feature field (K, Z, Y, X)); excision from the intensity."""
    w = excision_weights(v, factor=model.excise) if model.excise > 0 else None
    O = model.lift(field, weights=w)
    pop = model.populations(field, O=O, mask=mask, labels=False)
    fld = model.field(pop, v.shape)
    pop["vec"], pop["weight"], pop["N_vox"] = fld["vec"], fld["weight"], fld["N"]
    return pop


def centre_profile(O, grid, cz, cy, cx):
    """The lifted distribution at a cell, read on the great circle of the x-z plane (0 = along x, 90 = along z)."""
    g = grid.cpu().numpy(); o = O[:, cz, cy, cx].cpu().numpy()
    phi = np.arange(0, 180, 1.0); circ = np.stack([np.sin(np.radians(phi)), np.zeros_like(phi), np.cos(np.radians(phi))], 1)
    K = np.exp((np.abs(circ @ g.T) ** 2 - 1) / (2 * math.sin(math.radians(9)) ** 2)); K /= K.sum(1, keepdims=True)
    return phi, K @ o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/ab_anatomix_raw")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tex, texture = tex_for("calibrated")
    engines = {"B1 enc2 raw": make_engine("anatomix_enc2", device=DEV, l2=False, out_dtype=torch.float32),
               "B1s enc2 standardised": make_engine("anatomix_enc2", device=DEV, l2=False, out_dtype=torch.float32),
               "B2 dev-vit raw": make_engine("anatomix_devvit", device=DEV, l2=False, out_dtype=torch.float32),
               "B2s dev-vit standardised": make_engine("anatomix_devvit", device=DEV, l2=False, out_dtype=torch.float32)}
    model = Orientations(device=DEV)
    results = {}
    cmapN = ListedColormap(["#222222", "#2a78d6", "#e34948"])
    for name, kind, kw, grid in CASES:
        img, tr, meta = build(kind, 1, grid, tex=tex, texture=texture, device=DEV, **kw)
        v = torch.as_tensor(img, device=DEV).float(); Z, Y, X = v.shape; ky = Y // 2
        fields = {"A intensity": v}
        cache = {}
        for ename, eng in engines.items():
            key = eng.name
            if key not in cache:
                cache[key] = eng(v).float()
            F = cache[key]
            if "standardised" in ename:                                                           # every channel counts alike (the lift's high-pass removes the means)
                F = F / F.reshape(F.shape[0], -1).std(1).clamp_min(1e-6).view(-1, 1, 1, 1)
            fields[ename] = F
        runs = {}
        for fname, field in fields.items():
            runs[fname] = run_on(model, field, v, None)
            sc = lattice_scores(runs[fname], tr, grid, meta, model)
            results.setdefault(name, {})[fname] = {"floor": round(runs[fname]["floor"], 4), **{k: {kk: vv for kk, vv in val.items() if kk != "cells"} if isinstance(val, dict) else val for k, val in sc.items()}}
            print(f"{name:18s} {fname:20s} floor {runs[fname]['floor']:.4f} {json.dumps(results[name][fname])}", flush=True)
        # --- the QC image
        cols = 5 + (1 if kind == "band_crossing" else 0)
        fig, axs = plt.subplots(len(fields), cols, figsize=(4.2 * cols, 3.6 * len(fields)), facecolor="white")
        step, org = model.step, model.origin; cz, cy, cx = (Z // 2 - org) // step, (ky - org) // step, (X // 2 - org) // step
        tis = (tr["region"] == 2) if "region" in tr else np.ones((Z, X), bool)
        for i, (fname, field) in enumerate(fields.items()):
            r = runs[fname]; ax = axs[i]
            if field.dim() == 3:
                ax[0].imshow(field[:, ky].cpu().numpy(), cmap="gray", vmin=40, vmax=220); ax[0].set_title(f"{fname}: the image", fontsize=10)
            else:
                Fc = field[:, :, ky, :].cpu().numpy(); Fc = Fc - Fc.mean((1, 2), keepdims=True)
                U, S, Vt = np.linalg.svd(Fc.reshape(Fc.shape[0], -1), full_matrices=False)
                pcs = (Vt[:3] * S[:3, None]).reshape(3, Z, X); pcs = (pcs - pcs.min((1, 2), keepdims=True)) / (np.ptp(pcs, axis=(1, 2))[:, None, None] + 1e-9)
                ax[0].imshow(np.moveaxis(pcs, 0, -1)); ax[0].set_title(f"{fname}: the features (3 PCs)", fontsize=10)
            ax[0].plot([X / 2], [Z / 2], "o", mfc="none", mec="#ffd400", mew=2, ms=12)
            phi, o = centre_profile(r["lift"], model.grid, cz, cy, cx); _, f = centre_profile(r["fodf"], model.grid, cz, cy, cx)
            ax[1].plot(phi, o - o.min(), color="#888888", lw=1.6, label="lift − min"); ax[1].plot(phi, f / max(f.max(), 1e-9) * (o - o.min()).max(), color="#2a5db0", lw=2.2, label="deconvolved")
            dc = r["dirs_cells"][:, :, cz, cy, cx].cpu().numpy(); mc = r["mass"][:, cz, cy, cx].cpu().numpy()
            for k in range(dc.shape[0]):
                if mc[k] > 0:
                    ax[1].axvline(math.degrees(math.atan2(dc[k][0], dc[k][2])) % 180.0, color="#e34948", ls="--", lw=1.2)
            ax[1].set_xlim(0, 180); ax[1].set_xticks([0, 45, 90, 135, 180]); ax[1].legend(fontsize=8); ax[1].set_title("the lifted distribution at the circle", fontsize=10)
            p = r["presence"][:, cy, :].cpu().numpy(); fl = r["floor"]
            ax[2].imshow(p, cmap="magma", vmin=0, vmax=max(float(np.percentile(p, 99)), 2 * fl)); ax[2].contour(p, levels=[fl], colors=["#4fd1ff"], linewidths=1.0)
            ax[2].set_title(f"presence (floor {fl:.3f}, cyan)", fontsize=10)
            ax[3].imshow(r["N"][:, cy, :].cpu().numpy(), cmap=cmapN, vmin=-0.5, vmax=2.5, interpolation="nearest"); ax[3].set_title("N per cell (nmax 2, ≥ 30° apart)", fontsize=10)
            ax[4].imshow(dec_rgb(r["vec"][:, :, ky, :].cpu().numpy(), r["weight"][:, ky, :].cpu().numpy()) * tis[..., None]); ax[4].set_title("the first vector, DEC × weight", fontsize=10)
            if kind == "band_crossing":
                g = model.grid.cpu().numpy(); d2 = direction(meta["angles"][1]); lobe2 = np.abs(g @ d2) >= math.cos(math.radians(20))
                F_ = r["fodf"][:, cz, cy, :]; sh = (F_[torch.as_tensor(lobe2, device=DEV)].sum(0) / F_.sum(0).clamp_min(1e-12)).cpu().numpy()
                xc = (org + step * np.arange(sh.shape[0]) - X / 2) * grid.vox
                ax[5].axvspan(-meta["width_mm"] / 2, meta["width_mm"] / 2, color="#f3d9d9"); ax[5].plot(xc, sh, color="#2a5db0", lw=2)
                ax[5].set_ylim(0, 1); ax[5].set_xlim(-1.5, 1.5); ax[5].set_title("the crossed band's share along its axis", fontsize=10); ax[5].set_xlabel("mm from the crossing", fontsize=9)
            for a_ in ax[:5]:
                if a_ is not ax[1]:
                    a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f"{name}: the lift from the intensity (A) against the lift from anatomix features (B1, B2); the chain after it identical", fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / f"ab_{kind}.png", dpi=90); plt.close(fig)
        json.dump(results, open(out / "ab_anatomix.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
