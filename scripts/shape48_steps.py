"""PROTOTYPE: steps 1-4 of the shape-space pipeline with anatomix's 48-channel full-resolution layer (layer 58: the 32
decoder channels upsampled from 1/2 resolution, concatenated with the 16 first-encoder channels), on D0 and D15 (grown
by 9.6 mm per x and z face, cropped back to the original 20 x 20 mm for display), and an UNPAIRED boundary / fibre
division: learned from each image alone, no twin, no labels, no clustering.

The division is by spatial scale. Each channel is split into a coarse part (Gaussian, sigma SIG_C) and a fine part
(the rest); the generalised eigenvectors of C_coarse w = lambda C_fine w over the image's own voxels order feature
directions from mm-scale-dominated (lambda > 1: regions and their boundaries, here the step and the fibre bands) to
fine-scale-dominated (lambda < 1: texture, here the fibres). Boundary component = the lambda > 1 directions; fibre
component = the rest (an oblique split: f = V_b c_b + V_f c_f).

  1  image
  2a shape space, RGB = top 3 PCs of the 48-D features (per image)
  2b the 48-D space on the boundary axis (largest lambda) and the fibre axis (fibre component's PC 1); colour = 2a
  3  boundary component, RGB = its top 3 directions (largest lambda)
  4  fibre component, RGB = its top 3 PCs
Evaluation only (uses the twin; nothing is learned from it): the share of D15 - D0 within 3 mm of the interface that
falls in D15's boundary component.

    python shape48_steps.py [sigma_c_vox]     (shapetensors env, one GPU)
"""
import json, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import linalg
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import edge_aware_phantom_fig as EPF
import shape_tensor as ST
import surface_phantoms as SP

PAD = SP.D_PAD
JY = ST.JY
YS = slice(JY - 40, JY + 41)                                  # the y slab kept (enough for rho 16 later)
CACHE = ST.OUT / f"f48_pad{PAD}"
NAMES = ("D0", "D15")
SIG_C = 10.0                                                  # coarse scale (voxels): 1 mm


def feats48(n):
    p = CACHE / f"{n}.npy"
    if p.exists():
        return np.load(p)
    v = np.asarray(nib.load(SP.PAD_ROOT / f"{n}.nii.gz").dataobj).astype(np.float32)
    t0 = time.time(); F = AS.anatomix_layer(v, ys=YS).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True); np.save(p, F); print(n, "48-ch features", F.shape, f"{time.time() - t0:.0f} s", flush=True)
    return F


def crop_xz(a):
    return a[..., PAD:PAD + SP.NX, :, PAD:PAD + SP.NZ]


def split(F, sig=SIG_C, yh=10):
    """Unpaired scale split of one image's features F (C, x, y, z), statistics over the original region, y = JY +- yh."""
    C = F.shape[0]; jy = JY - YS.start
    coarse = np.stack([ndi.gaussian_filter(f, sig, mode="nearest", truncate=2.0) for f in F])
    region = (slice(None), slice(PAD, PAD + SP.NX), slice(jy - yh, jy + yh + 1), slice(PAD, PAD + SP.NZ))
    Xc = coarse[region].reshape(C, -1); Xf = (F - coarse)[region].reshape(C, -1)
    Cc = np.cov(Xc); Cf = np.cov(Xf) + 1e-9 * np.trace(np.cov(Xf)) / C * np.eye(C)
    lam, W = linalg.eigh(Cc, Cf); o = np.argsort(lam)[::-1]; lam, W = lam[o], W[:, o]       # W^T Cf W = I
    V = np.linalg.inv(W).T                                                                    # dual basis: f = V c
    kb = int((lam > 1).sum())
    return {"lam": lam, "W": W, "V": V, "kb": kb, "mu": F[region].reshape(C, -1).mean(1, keepdims=True)}


def main(sig=SIG_C):
    F = {n: feats48(n) for n in NAMES}
    S = {n: split(F[n], sig) for n in NAMES}
    jy = JY - YS.start
    sl = {n: F[n][:, PAD:PAD + SP.NX, jy, PAD:PAD + SP.NZ].reshape(F[n].shape[0], -1) for n in NAMES}   # (48, NX * NZ)
    res = {"sigma_c_vox": sig}
    panels = {}
    for n in NAMES:
        s = S[n]; X = sl[n] - s["mu"]; c = s["W"].T @ X; kb = s["kb"]
        fb = s["V"][:, :kb] @ c[:kb]; ff = s["V"][:, kb:] @ c[kb:]                            # boundary, fibre parts
        tot = (X ** 2).sum()

        def pcs(Y, k=3):
            w, U = np.linalg.eigh(np.cov(Y)); return U[:, np.argsort(w)[::-1][:k]].T
        P2 = pcs(X) @ X; Pf = pcs(ff) @ (ff - ff.mean(1, keepdims=True)); Pb = c[:3]
        panels[n] = {"rgb": P2, "b": Pb, "f": Pf}
        res[n] = {"k_boundary": kb, "lambda": [round(float(x), 3) for x in s["lam"]],
                  "var_boundary_part": float((fb ** 2).sum() / tot), "var_fibre_part": float((ff ** 2).sum() / tot)}
    # evaluation only: where does the true step change go in D15's own split?
    d = crop_xz(F["D15"][:, :, jy - 10:jy + 11, :] - F["D0"][:, :, jy - 10:jy + 11, :])
    near = np.abs((np.arange(SP.NZ) + 0.5) * SP.VOX - SP.D_PLANE_Z) < 3.0
    dX = d[:, :, :, near].reshape(d.shape[0], -1); s = S["D15"]; c = s["W"].T @ dX; kb = s["kb"]
    db = s["V"][:, :kb] @ c[:kb]
    res["eval_step_change_in_D15_boundary_part"] = float((db ** 2).sum() / (dX ** 2).sum())
    (ST.OUT / "shape48_split.json").write_text(json.dumps(res, indent=1))
    for n in NAMES:
        print(n, "boundary dirs (lambda > 1):", res[n]["k_boundary"], "of 48; top lambda", res[n]["lambda"][:6],
              f"| variance in boundary part {res[n]['var_boundary_part']:.0%}, fibre part {res[n]['var_fibre_part']:.0%}")
    print(f"evaluation only: share of the D15 - D0 step change (within 3 mm) in D15's boundary part: "
          f"{res['eval_step_change_in_D15_boundary_part']:.0%}")
    # figure
    def rgb(P):
        lo, hi = np.percentile(P, 1, axis=1), np.percentile(P, 99, axis=1)
        return np.clip((P - lo[:, None]) / np.maximum(hi - lo, 1e-12)[:, None], 0, 1).T.reshape(SP.NX, SP.NZ, 3)
    cols = ("1  image", "2a  anatomix 48-D shape space\n(RGB = top 3 PCs)", "2b  48-D space: boundary vs fibre axes\n(each dot = one voxel)",
            f"3  boundary component (mm-scale)\n(RGB = top 3 directions, λ > 1)", "4  fibre component (fine-scale)\n(RGB = its top 3 PCs)")
    fig, axs = plt.subplots(2, 5, figsize=(26, 10.8), dpi=90)
    rng = np.random.default_rng(0); samp = rng.choice(SP.NX * SP.NZ, 6000, replace=False)
    dep = np.abs((np.arange(SP.NZ) + 0.5) * SP.VOX - SP.D_PLANE_Z)[None, :].repeat(SP.NX, 0).ravel()[samp]
    for r, n in enumerate(NAMES):
        _, draw, _ = EPF.truth_and_bounds(n)
        v = np.asarray(nib.load(SP.PAD_ROOT / f"{n}.nii.gz").dataobj).astype(np.float32)
        EPF.base_image(axs[r, 0], crop_xz(v)[:, JY, :], draw)
        p = panels[n]; img2 = rgb(p["rgb"])
        for c, im in ((1, img2), (3, rgb(p["b"])), (4, rgb(p["f"]))):
            ax = axs[r, c]; ax.imshow(np.transpose(im, (1, 0, 2)), origin="lower", extent=EPF.EXT); draw(ax)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axs[r, 2]; o = np.argsort(dep < 0.5)
        ax.scatter(p["b"][0, samp][o], p["f"][0, samp][o], c=img2.reshape(-1, 3)[samp][o], s=4, lw=0, alpha=0.8)
        ax.set_xlabel("boundary axis (largest λ)", fontsize=9); ax.set_ylabel("fibre axis (fibre component PC 1)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.98, f"{res[n]['k_boundary']} of 48 directions have λ > 1\nvariance: boundary part "
                f"{res[n]['var_boundary_part']:.0%}, fibre part {res[n]['var_fibre_part']:.0%}",
                transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(boxstyle="round", fc="w", alpha=0.85, lw=0))
        axs[r, 0].set_ylabel({"D0": "D0\nno step", "D15": "D15\nstep 15"}[n], fontsize=14, rotation=0, labelpad=40)
    for c, t in enumerate(cols):
        axs[0, c].set_title(t, fontsize=11)
    fig.suptitle(f"PROTOTYPE. Steps 1-4 with anatomix's 48-channel full-resolution layer, D0 and D15, each image split on its own "
                 f"(no twin, no labels, no clustering): directions dominated by mm-scale structure (σ {sig * 0.1:g} mm) = boundary, the "
                 f"rest = fibre.\nEvaluation only (uses the twin): {res['eval_step_change_in_D15_boundary_part']:.0%} of the true D15 - D0 "
                 "step change within 3 mm of the interface falls in D15's boundary component. Phantoms grown 9.6 mm per x and z face, "
                 "cropped for display; cyan = interface.", fontsize=12, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.85, wspace=0.25, hspace=0.15, left=0.05, right=0.99)
    out = ST.FIGS / "shape48_steps.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else SIG_C)
