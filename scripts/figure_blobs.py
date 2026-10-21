"""Regions instead of partitions: each direction as a bounded blob in the shape space, texture as another, and a voxel
outside every blob left unassigned (a generative model with a reject option -- classical: Gaussian class-conditional
densities and a chi-squared radius).

Blobs (whitened features, the dictionary's residual covariance as the metric): one per calibration angle, centred on
that direction's signature, its covariance the samples' own about it; texture: the fibre-free texture samples' mean and
covariance. A voxel belongs to the blob of smallest Mahalanobis distance if that distance is within the blob's 99% radius
(chi-squared, C degrees of freedom), otherwise to none. Shown: a 3D view (orientation PC 1, PC 2, the presence axis) with
each blob's 2-sigma ellipsoid (its covariance projected), the samples, 2D projections, and per sample kind the shares
assigned to a direction (and their error), to texture, and to none.

    python scripts/figure_blobs.py [OUT_PNG]        ENGINE=anatomix_devvit (default) | anatomix_enc2
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from scipy.stats import chi2

from shapetensor.classical import whitener
from shapetensor.fibre_sets import iso_textures
from shapetensor.phantoms import Grid, build, direction
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES

ENGINE = os.environ.get("ENGINE", "anatomix_devvit")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(f"/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/figures/latent_blobs_{ENGINE}.png")
DEV = "cuda"; N_PER = 500
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def hue(a):
    a = np.atleast_1d(a); return hsv_to_rgb(np.stack([(a / 180.0) % 1, np.ones_like(a), np.ones_like(a)], -1))


def ellipsoid(ax, c, C, color, n=14, s=2.0):
    w, V = np.linalg.eigh(C); w = np.sqrt(np.clip(w, 0, None)) * s
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2))
    sph = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1)
    pts = sph @ (V * w).T + c
    ax.plot_wireframe(pts[..., 0], pts[..., 1], pts[..., 2], color=color, linewidth=0.5, alpha=0.6)


def ellipse(ax, c, C, color, s=2.0, lw=1.2):
    w, V = np.linalg.eigh(C); t = np.linspace(0, 2 * np.pi, 100)
    pts = (np.stack([np.cos(t), np.sin(t)], 1) * (np.sqrt(np.clip(w, 0, None)) * s)) @ V.T + c
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw)


def main():
    tex, texture = P.tex_for("calibrated"); cg = Grid(); g = torch.Generator(device="cpu").manual_seed(0)
    bundles = [(a, torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float())
               for k, a in enumerate(ANGLES)]
    c90 = torch.as_tensor(build("crossing", 0, cg, tex, texture=texture, device=DEV, cross_deg=90.0)[0], device=DEV).float()
    c60 = torch.as_tensor(build("crossing", 0, cg, tex, texture=texture, device=DEV, cross_deg=60.0)[0], device=DEV).float()
    texv = iso_textures("calibrated", DEV)
    cal = calibrate([b for _, b in bundles] + [c90, c60], sigma=1.0, rho=10.0, truncate=3.0, margin=cg.margin[0], engine=ENGINE, engine_kwargs={},
                    smooth=3, seed=0, device=DEV, iso=texv)
    D = cal.dictionary; Wh = whitener(D); mu = D.D.float().mean(0)

    def take(v, n=N_PER):
        Fp = D.prep(cal.features(v)).float(); C = Fp.shape[0]
        sub = Fp[(slice(None),) + cg.inner].reshape(C, -1)
        X = sub[:, torch.randperm(sub.shape[1], generator=g)[:n].to(DEV)].T
        return ((X - D.mean) / D.std - mu) @ Wh.T                                                   # whitened, centred

    Zb = [(a, take(b)) for a, b in bundles]; Zt = torch.cat([take(t) for t in texv]); Z90, Z60 = take(c90), take(c60)
    C = Zt.shape[1]
    # blobs: centre = the direction's signature (whitened), covariance = its samples' about it; texture = its samples'
    blobs = []
    for a, Z in Zb:
        d = torch.as_tensor(direction(a), device=DEV)
        c = (D.signature(d).float() - mu) @ Wh.T
        R = Z - c; S = (R.T @ R) / max(1, len(R) - 1) + 1e-3 * torch.eye(C, device=DEV)
        blobs.append(("dir", a, c, S))
    ct = Zt.mean(0); St = torch.cov(Zt.T) + 1e-3 * torch.eye(C, device=DEV); blobs.append(("texture", None, ct, St))
    r99 = float(chi2.ppf(0.99, C))

    def assign(Z):
        d2 = torch.stack([(((Z - c) @ torch.linalg.inv(S)) * (Z - c)).sum(1) for _, _, c, S in blobs], 1)        # (n, n_blobs)
        m, k = d2.min(1)
        k = torch.where(m <= r99, k, torch.full_like(k, -1))
        return k.cpu().numpy()

    stats = {"engine": ENGINE, "chi2 99% radius^2": round(r99, 1)}
    nb = len(blobs)
    def summarise(name, k, true_angle=None):
        out = {"to a direction": round(float((k >= 0) & (k < nb - 1)).mean() if False else float(((k >= 0) & (k < nb - 1)).mean()), 3),
               "to texture": round(float((k == nb - 1).mean()), 3), "to none": round(float((k < 0).mean()), 3)}
        if true_angle is not None:
            sel = (k >= 0) & (k < nb - 1); dk = k[sel]
            if len(dk):
                err = np.abs(((np.array([blobs[i][1] for i in dk]) - true_angle[sel] + 90) % 180) - 90)
                out["direction error median (deg)"] = round(float(np.median(err)), 1)
        stats[name] = out
    kb = np.concatenate([assign(Z) for _, Z in Zb]); ab = np.concatenate([np.full(len(Z), a) for a, Z in Zb])
    summarise("bundles", kb, ab)
    summarise("texture", assign(Zt)); summarise("crossing 90", assign(Z90)); summarise("crossing 60", assign(Z60))
    print(json.dumps(stats, indent=1))
    # the 3D frame: orientation PCs of the signatures and the presence axis (texture -> bundles)
    Sg = (D.D.float() - mu) @ Wh.T
    _, _, Vt = torch.linalg.svd(Sg - Sg.mean(0), full_matrices=False)
    Zball = torch.cat([Z for _, Z in Zb])
    pa = Zball.mean(0) - ct; pa = pa / pa.norm()
    e1 = Vt[0] - (Vt[0] @ pa) * pa; e1 /= e1.norm(); e2 = Vt[1] - (Vt[1] @ pa) * pa - (Vt[1] @ e1) * e1; e2 /= e2.norm()
    E = torch.stack([e1, e2, pa])                                                                 # (3, C)
    proj = lambda Z: (Z @ E.T).cpu().numpy()
    fig = plt.figure(figsize=(22, 8.5), facecolor="white")
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    for a, Z in Zb:
        p = proj(Z[:150]); ax3.scatter(p[:, 0], p[:, 1], p[:, 2], s=3, c=hue(a).repeat(len(p), 0), alpha=0.5)
    for Zs, cc in ((Zt[:300], [0.55] * 3), (Z90[:200], [0, 0, 0]), (Z60[:200], [0.45, 0.25, 0.1])):
        p = proj(Zs); ax3.scatter(p[:, 0], p[:, 1], p[:, 2], s=4, c=[cc], alpha=0.6)
    for kind, a, c, S in blobs:
        ellipsoid(ax3, proj(c[None])[0], (E @ S @ E.T).cpu().numpy(), hue(a)[0] if kind == "dir" else (0.4, 0.4, 0.4))
    ax3.set_xlabel("orientation PC 1"); ax3.set_ylabel("orientation PC 2"); ax3.set_zlabel("presence axis"); ax3.view_init(elev=22, azim=-60)
    ax3.set_title("blobs (2σ ellipsoids, covariance projected) and samples", fontsize=10)
    for col, (i, j, xl, yl) in enumerate(((0, 1, "orientation PC 1", "orientation PC 2"), (2, 0, "presence axis (texture -> bundles)", "orientation PC 1"))):
        ax = fig.add_subplot(1, 3, 2 + col)
        for a, Z in Zb:
            p = proj(Z); ax.scatter(p[:, i], p[:, j], s=3, c=hue(a).repeat(len(p), 0), alpha=0.35)
        for Zs, cc, mk, lab in ((Zt, [0.55] * 3, "o", "texture"), (Z90, [0, 0, 0], "s", "crossing 90"), (Z60, [0.45, 0.25, 0.1], "^", "crossing 60")):
            p = proj(Zs); ax.scatter(p[:, i], p[:, j], s=8, c=[cc], marker=mk, alpha=0.7, label=lab)
        for kind, a, c, S in blobs:
            Cp = (E @ S @ E.T).cpu().numpy()[np.ix_([i, j], [i, j])]
            ellipse(ax, proj(c[None])[0][[i, j]], Cp, hue(a)[0] if kind == "dir" else (0.3, 0.3, 0.3), lw=1.6 if kind != "dir" else 1.0)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=8, loc="lower right", markerscale=2); ax.set_xticks([]); ax.set_yticks([])
    s = stats
    fig.suptitle(f"Regions in the shape space ({ENGINE}): a blob per calibration direction (hue = angle) and for texture (grey), 2σ.  Assigned to a direction / texture / none —  "
                 f"bundles {s['bundles']['to a direction']:.2f}/{s['bundles']['to texture']:.2f}/{s['bundles']['to none']:.2f} (error {s['bundles'].get('direction error median (deg)', float('nan'))}°);  "
                 f"texture {s['texture']['to a direction']:.2f}/{s['texture']['to texture']:.2f}/{s['texture']['to none']:.2f};  "
                 f"crossing 90 {s['crossing 90']['to a direction']:.2f}/{s['crossing 90']['to texture']:.2f}/{s['crossing 90']['to none']:.2f};  "
                 f"crossing 60 {s['crossing 60']['to a direction']:.2f}/{s['crossing 60']['to texture']:.2f}/{s['crossing 60']['to none']:.2f}", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.94)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=100)
    json.dump(stats, open(OUT.with_suffix(".json"), "w"), indent=1); print(OUT)


if __name__ == "__main__":
    main()
