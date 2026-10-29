"""The partitions each approach draws in the shape space. The plane shown: the first two principal components of the
direction signatures (centred, whitened by the dictionary's residual covariance) -- where orientation lives in the
features. Samples (voxels, the readers' box-averaged features): straight bundles at the 12 calibration angles, fibre-free
texture, the 90° and 60° crossings. Panels per engine:

  1. truth: each sample's direction (in-plane angle as hue; texture grey; crossings black / brown);
  2. teacher: the excised structure tensor's direction at the sample (what the readers are calibrated on);
  3. dev readout (the small network): its decision regions over the plane (the most probable direction as hue, its
     probability as brightness), the samples on top coloured by their own prediction;
  4. classical matched filter: the direction whose whitened signature correlates best, regions and samples;
  5. presence, in its own plane (presence decides along a different axis than orientation): the axis from the texture
     samples' mean to the bundles' (whitened) against orientation PC 1; the distance from fibre-free texture (contour
     at the texture voxels' 99th percentile) and the linear probe's boundary (p = 0.5), samples by kind.
The regions are a slice of the decision functions through the plane at the samples' mean (the other components held
there), an illustration of the partition, not the whole of it.

    python scripts/figure_latent.py [OUT_PNG]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb

from shapetensor import st
from shapetensor.classical import whitener
from shapetensor.fibre_sets import iso_textures
from shapetensor.phantoms import Grid, build
from shapetensor.pipeline import calibrate
from shapetensor.validate import ANGLES

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/figures/latent_partitions.png")
DEV = "cuda"; N_PER = 400
spec = importlib.util.spec_from_file_location("proto", str(Path(__file__).with_name("prototype_fibre_sets.py")))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def angle(d):
    """In-plane angle (deg, 0-180) of directions ``(n, 3)`` (z, y, x): the phantoms' convention, (sin a, 0, cos a)."""
    return (torch.rad2deg(torch.atan2(d[:, 0], d[:, 2])) % 180).cpu().numpy()


def hue_rgb(a, v=None):
    h = (np.asarray(a) / 180.0) % 1.0
    s = np.ones_like(h); val = np.ones_like(h) if v is None else np.clip(v, 0, 1)
    return hsv_to_rgb(np.stack([h, s, val], -1))


def main():
    tex, texture = P.tex_for("calibrated"); cg = Grid(); g = torch.Generator(device="cpu").manual_seed(0)
    bundles = [(a, torch.as_tensor(build("bundle", 100 + k, cg, tex, texture=texture, angle=float(a), device=DEV)[0], device=DEV).float())
               for k, a in enumerate(ANGLES)]
    c90 = torch.as_tensor(build("crossing", 0, cg, tex, texture=texture, device=DEV, cross_deg=90.0)[0], device=DEV).float()
    c60 = torch.as_tensor(build("crossing", 0, cg, tex, texture=texture, device=DEV, cross_deg=60.0)[0], device=DEV).float()
    texv = iso_textures("calibrated", DEV)
    inner = cg.inner
    engines = (("anatomix_enc2", "encoder block 2 (released)"), ("anatomix_devvit", "dev-vit (blended tiles, grid filter)"))
    fig, axs = plt.subplots(len(engines), 5, figsize=(26, 5.4 * len(engines)), facecolor="white")
    for row, (eng, label) in enumerate(engines):
        cal = calibrate([b for _, b in bundles] + [c90, c60], sigma=1.0, rho=10.0, truncate=3.0, margin=cg.margin[0], engine=eng, engine_kwargs={},
                        smooth=3, seed=0, device=DEV, iso=texv)
        D = cal.dictionary; Wh = whitener(D); Whi = torch.linalg.inv(Wh)
        S = D.D.float(); mu = S.mean(0); Sw = (S - mu) @ Wh.T
        _, _, Vt = torch.linalg.svd(Sw - Sw.mean(0), full_matrices=False); E = Vt[:2]                  # (2, C) the plane, whitened space

        def take(v, n=N_PER, with_teacher=False):
            Fp = D.prep(cal.features(v)).float(); C = Fp.shape[0]
            sub = Fp[(slice(None),) + inner].reshape(C, -1)
            idx = torch.randperm(sub.shape[1], generator=g)[:n].to(DEV)
            X = sub[:, idx].T
            tdir = None
            if with_teacher:
                d = st.structure_tensor(v, sigma=1.0, rho=10.0, truncate=3.0, excise=True, order="zyx", device=DEV)[(slice(None),) + inner].reshape(3, -1)
                tdir = d[:, idx].T
            return X, tdir

        samples = []                                                                                   # (X raw prepped, kind, true angle, teacher dir)
        for a, b in bundles:
            X, td = take(b, with_teacher=True); samples.append((X, "bundle", np.full(len(X), a), td))
        for t in texv:
            X, td = take(t, with_teacher=True); samples.append((X, "texture", np.full(len(X), np.nan), td))
        X, td = take(c90, with_teacher=True); samples.append((X, "crossing 90", np.full(len(X), np.nan), td))
        X, td = take(c60, with_teacher=True); samples.append((X, "crossing 60", np.full(len(X), np.nan), td))
        Xall = torch.cat([s[0] for s in samples])
        Z = ((Xall - D.mean) / D.std - mu) @ Wh.T                                                      # whitened, centred on the mean signature
        zbar = Z.mean(0)
        P2 = ((Z - zbar) @ E.T).cpu().numpy()                                                          # the samples in the plane
        lo, hi = np.percentile(P2, 1, 0), np.percentile(P2, 99, 0); pad = 0.15 * (hi - lo); lo, hi = lo - pad, hi + pad
        ga, gb = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220))
        Zg = zbar[None] + torch.as_tensor(np.stack([ga.ravel(), gb.ravel()], 1), device=DEV, dtype=torch.float32) @ E
        Fg = ((Zg @ Whi.T) + mu) * D.std + D.mean                                                       # back to the readers' features
        with torch.no_grad():
            Pr = cal.readout.probs((Fg - cal.readout.mean) / cal.readout.std)                          # (G, n_dirs)
            Ps = cal.readout.probs((Xall - cal.readout.mean) / cal.readout.std)
        u = cal.readout.grid
        reg_ro = hue_rgb(angle(u[Pr.argmax(1)]), (Pr.max(1).values / Pr.max()).cpu().numpy() ** 0.5).reshape(220, 220, 3)
        mf_g = Zg @ Sw.T; mf_s = Z @ Sw.T
        reg_mf = hue_rgb(angle(D.grid[mf_g.argmax(1)]), (mf_g.max(1).values / mf_g.max(1).values.max()).clamp_min(0).cpu().numpy() ** 0.5).reshape(220, 220, 3)
        dist_g = cal.texture_model.predict(Fg.T.reshape(-1, 220, 220, 1), prepped=True).reshape(220, 220).cpu().numpy()
        dist_s = cal.texture_model.predict(Xall.T.reshape(Xall.shape[1], -1, 1, 1), prepped=True).flatten().cpu().numpy()
        prob_g = cal.presence.predict(Fg.T.reshape(-1, 220, 220, 1), prepped=True).reshape(220, 220).cpu().numpy()
        kinds = np.concatenate([[s[1]] * len(s[0]) for s in samples]); truth = np.concatenate([s[2] for s in samples])
        teach = torch.cat([s[3] for s in samples]); ext = (lo[0], hi[0], lo[1], hi[1])
        null99 = float(np.percentile(dist_s[kinds == "texture"], 99))
        bmask = kinds == "bundle"

        def scatter(ax, colours, title, bg=None):
            if bg is not None:
                ax.imshow(bg, origin="lower", extent=ext, aspect="auto", alpha=0.85)
            for kd, mk, sz in (("bundle", "o", 5), ("texture", "o", 4), ("crossing 60", "^", 9), ("crossing 90", "s", 9)):
                m = kinds == kd
                ax.scatter(P2[m, 0], P2[m, 1], c=colours[m], s=sz, marker=mk, linewidths=0.2, edgecolors="k" if kd.startswith("crossing") else "none")
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])

        col_truth = np.zeros((len(kinds), 3)); col_truth[bmask] = hue_rgb(truth[bmask])
        col_truth[kinds == "texture"] = 0.6; col_truth[kinds == "crossing 90"] = 0.0; col_truth[kinds == "crossing 60"] = [0.45, 0.25, 0.1]
        scatter(axs[row, 0], col_truth, f"{label}\ntruth (hue = in-plane angle; texture grey, crossing 90 black, 60 brown)")
        scatter(axs[row, 1], hue_rgb(angle(teach)), "teacher: excised structure tensor's direction")
        scatter(axs[row, 2], hue_rgb(angle(u[Ps.argmax(1)])), f"dev readout network: regions and predictions (fit {cal.readout_overlap:.2f})", reg_ro)
        scatter(axs[row, 3], hue_rgb(angle(D.grid[mf_s.argmax(1)])), "classical matched filter: regions and predictions", reg_mf)
        ax = axs[row, 4]
        tz = torch.as_tensor(kinds == "texture", device=DEV); bz = torch.as_tensor(bmask, device=DEV)
        pa = Z[bz].mean(0) - Z[tz].mean(0); pa = pa / pa.norm()                                        # the presence axis (whitened)
        e1 = E[0] - (E[0] @ pa) * pa; e1 = e1 / e1.norm(); E2 = torch.stack([pa, e1])
        Q = ((Z - zbar) @ E2.T).cpu().numpy()
        qlo, qhi = np.percentile(Q, 0.5, 0), np.percentile(Q, 99.5, 0); qp = 0.15 * (qhi - qlo); qlo, qhi = qlo - qp, qhi + qp
        qa, qb = np.meshgrid(np.linspace(qlo[0], qhi[0], 220), np.linspace(qlo[1], qhi[1], 220))
        Zq = zbar[None] + torch.as_tensor(np.stack([qa.ravel(), qb.ravel()], 1), device=DEV, dtype=torch.float32) @ E2
        Fq = ((Zq @ Whi.T) + mu) * D.std + D.mean
        dist_q = cal.texture_model.predict(Fq.T.reshape(-1, 220, 220, 1), prepped=True).reshape(220, 220).cpu().numpy()
        prob_q = cal.presence.predict(Fq.T.reshape(-1, 220, 220, 1), prepped=True).reshape(220, 220).cpu().numpy()
        for kd, cc, mk, sz in (("bundle", [0.1, 0.5, 0.9], "o", 5), ("crossing 60", [0.45, 0.25, 0.1], "^", 9), ("crossing 90", [0.0, 0.0, 0.0], "s", 9),
                               ("texture", [0.55, 0.55, 0.55], "o", 4)):
            m = kinds == kd
            ax.scatter(Q[m, 0], Q[m, 1], c=[cc], s=sz, marker=mk, linewidths=0.2, edgecolors="w" if kd.startswith("crossing") else "none", label=kd)
        ax.contour(qa, qb, dist_q, levels=[null99], colors="cyan", linewidths=1.8)
        ax.contour(qa, qb, prob_q, levels=[0.5], colors="magenta", linewidths=1.8)
        ax.set_xlim(qlo[0], qhi[0]); ax.set_ylim(qlo[1], qhi[1]); ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=7, loc="lower right", markerscale=2)
        ax.set_title("presence, in its own plane: distance from texture (cyan = texture 99th pct),\nlinear probe p = 0.5 (magenta)", fontsize=10)
        for ax in axs[row, :4]:
            ax.set_xlabel("orientation PC 1"); ax.set_ylabel("orientation PC 2")
        axs[row, 4].set_xlabel("presence axis (texture -> bundles)"); axs[row, 4].set_ylabel("orientation PC 1 (orthogonalised)")
        del cal; torch.cuda.empty_cache()
    # the hue key
    cax = fig.add_axes([0.92, 0.02, 0.07, 0.07], projection="polar")
    th = np.linspace(0, np.pi, 180); cax.scatter(th, np.ones_like(th), c=hue_rgb(np.degrees(th)), s=20); cax.set_thetamax(180)
    cax.set_yticks([]); cax.set_title("in-plane angle", fontsize=8)
    fig.suptitle("The partitions in the shape space: the plane where orientation lives (first two principal components of the whitened direction signatures)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 0.92, 0.96)); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=100); print(OUT)


if __name__ == "__main__":
    main()
