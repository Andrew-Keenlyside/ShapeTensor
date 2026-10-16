"""PROTOTYPE: steps 5-7 on the unpaired 48-D split of shape48_steps.py, against the intensity structure tensor.
D0 and D15 (grown by 9.6 mm per x and z face; orientations computed on the grown block, cropped before scoring).

  intensity ST      the standard structure tensor of the image (sigma 1 vox), e3, rho 3 (BRIDGE now), 6, 16
  48 all eigen      e3 of sum_c grad f_c grad f_c^T over all 48 channels (exact)
  48 fibre eigen    the same on the fibre component (oblique projection onto the fine-scale directions, 48-D metric)
  TCA all / fibre   CP rank 3 of (space x channel x axis), orientation from the components (nothing dropped), on the
                    top 16 PCs of all 48 channels / of the fibre component (K would be 144 x 144 per pixel otherwise)

    python shape48_orient.py [rho,...]     (shapetensors env, one GPU)
"""
import json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import shape48_steps as S48
import shape_tca as T
import surface_phantoms as SP

T.PAD = SP.D_PAD
JS = T.ST.JY - S48.YS.start                                           # mid y inside the cached 48-D slab


def e3_eigen(Z, rho, jy, mask=None):
    """e3 of sum_c G_rho * (grad Z_c grad Z_c^T) at y = jy on the scored region (via the 3C x 3C path, channel-summed)."""
    acc = None
    for a in range(0, len(Z), 8):                                       # 8 channels at a time keeps K small
        K = T.k_slice(Z[a:a + 8], rho, jy)
        C = K.shape[-1] // 3
        t = K.reshape(K.shape[0], K.shape[1], C, 3, C, 3).diagonal(dim1=2, dim2=4).sum(-1)
        acc = t if acc is None else acc + t
    mask = T.MASK if mask is None else mask
    acc = T.crop(acc, (0, 1))[torch.from_numpy(mask).to(T.DEV)].double()
    e = np.full(mask.shape + (3,), np.nan, np.float32); e[mask] = T.null_dir(acc).float().cpu().numpy()
    return e


def top_pcs(Z, k, jy):
    X = T.crop(Z[:, :, jy, :], (1, 2)).reshape(len(Z), -1)
    w, U = np.linalg.eigh(np.cov(X)); o = np.argsort(w)[::-1]
    return U[:, o[:k]].T.astype(np.float32), float(w[o[:k]].sum() / w.sum())


def main(rhos=(6.0, 16.0)):
    res = {}
    for n in S48.NAMES:
        F = S48.feats48(n); s = S48.split(F)
        Pf = (s["V"][:, s["kb"]:] @ s["W"][:, s["kb"]:].T).astype(np.float32)         # oblique projector, fibre part
        Ff = np.tensordot(Pf, F, axes=(1, 0))
        v = np.asarray(nib.load(SP.PAD_ROOT / f"{n}.nii.gz").dataobj).astype(np.float32)
        r = res[n] = {}
        for rho in (3.0, 6.0, 16.0):
            r[f"intensity ST rho{rho:g}"] = T.score(e3_eigen(v[None], rho, T.ST.JY))
        for rho in rhos:
            r[f"48 all eigen rho{rho:g}"] = T.score(e3_eigen(F, rho, JS))
            r[f"48 fibre eigen rho{rho:g}"] = T.score(e3_eigen(Ff, rho, JS))
            for lab, Z in (("all", F), ("fibre", Ff)):
                U, kept = top_pcs(Z, 16, JS)
                K = T.crop(T.k_slice(np.tensordot(U, Z, axes=(1, 0)), rho, JS), (0, 1))
                e, fit, _ = T.readouts(K, T.MASK, (3,))
                r[f"TCA {lab} rho{rho:g}"] = T.score(e["tca_recon_R3"])
                r[f"TCA {lab} rho{rho:g}"]["pcs16_variance_kept"] = round(kept, 3)
                r[f"TCA {lab} rho{rho:g}"]["cp_rel_err"] = round(fit[3], 3)
        for k, v_ in r.items():
            print(f"{n:4s} {k:24s} <0.5mm {v_['within_0.5mm']:5.1f}  3-8mm {v_['deep_3-8mm']:5.1f}   bands<0.5 90/60/30: "
                  f"{v_['90deg'][0]}/{v_['60deg'][0]}/{v_['30deg'][0]}" + (f"  (16 PCs keep {v_['pcs16_variance_kept']:.0%})" if "pcs16_variance_kept" in v_ else ""), flush=True)
    (T.ST.OUT / "shape48_orient.json").write_text(json.dumps(res, indent=1))


COLS = (("int_rho3", "intensity structure tensor, ρ 3\n(BRIDGE now)"), ("int_rho16", "intensity structure tensor, ρ 16"),
        ("e48_rho16", "48-D shape tensor, eigen e3, ρ 16"), ("tca48_rho16", "48-D TCA (rank 3), ρ 16"))


def maps():
    """Full-slice vectors for the glyph figure -> shape48_orient_maps.npz."""
    full = np.ones(T.MASK.shape, bool); out = {}
    for n in S48.NAMES:
        F = S48.feats48(n)
        v = np.asarray(nib.load(SP.PAD_ROOT / f"{n}.nii.gz").dataobj).astype(np.float32)
        out[f"{n}_int_rho3"] = e3_eigen(v[None], 3.0, T.ST.JY, full)
        out[f"{n}_int_rho16"] = e3_eigen(v[None], 16.0, T.ST.JY, full)
        out[f"{n}_e48_rho16"] = e3_eigen(F, 16.0, JS, full)
        U, _ = top_pcs(F, 16, JS)
        e, _, _ = T.readouts(T.crop(T.k_slice(np.tensordot(U, F, axes=(1, 0)), 16.0, JS), (0, 1)), full, (3,))
        out[f"{n}_tca48_rho16"] = e["tca_recon_R3"]; print(n, "done", flush=True)
    np.savez_compressed(T.ST.OUT / "shape48_orient_maps.npz", **out)


def glyph_fig():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    import edge_aware_phantom_fig as EPF
    M = np.load(T.ST.OUT / "shape48_orient_maps.npz")
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    cmap.set_bad("#e1e0d9")
    rows = (("D0", "D0, no step", None, 7), ("D15", "D15, step 15", None, 7),
            ("D15", "D15 zoom: interface,\n90° and 60° bands", (7.0, 13.0), 3))
    XR = (1.0, 11.0)
    fig, axs = plt.subplots(len(rows), 5, figsize=(29, 5.6 * 2.6 + 0.6), dpi=90,
                            gridspec_kw={"height_ratios": [1, 1, 0.6]})
    for r, (n, lab, zr, st) in enumerate(rows):
        tru, draw, _ = EPF.truth_and_bounds(n)
        img = T.crop(np.asarray(nib.load(SP.PAD_ROOT / f"{n}.nii.gz").dataobj).astype(np.float32), (0, 2))[:, T.ST.JY, :]
        EPF.base_image(axs[r, 0], img, draw)
        for c, (k, t) in enumerate(COLS, 1):
            ax = axs[r, c]; e = M[f"{n}_{k}"]
            a = np.where((T.XD >= 1) & (T.XD < 19), EPF.angle(e, tru), np.nan)
            im = ax.imshow(np.ma.masked_invalid(a.T), cmap=cmap, vmin=0, vmax=90, origin="lower", extent=EPF.EXT)
            xi, zi = np.meshgrid(np.arange(1, SP.NX, st), np.arange(1, SP.NZ, st), indexing="ij")
            keep = (T.XD[xi, zi] >= 1) & (T.XD[xi, zi] < 19); xi, zi = xi[keep], zi[keep]
            ee = e[xi, zi]; inpl = np.hypot(ee[:, 0], ee[:, 2]); d = ee[:, [0, 2]] / np.maximum(inpl[:, None], 1e-9)
            L = 0.85 * st * SP.VOX * inpl; cx, cz = (xi + 0.5) * SP.VOX, (zi + 0.5) * SP.VOX
            segs = np.stack([np.stack([cx - d[:, 0] * L / 2, cz - d[:, 1] * L / 2], 1),
                             np.stack([cx + d[:, 0] * L / 2, cz + d[:, 1] * L / 2], 1)], 1)
            lw = 1.3 if zr is None else 1.6
            ax.add_collection(LineCollection(segs, colors="white", linewidths=lw + 1.6, capstyle="round"))
            ax.add_collection(LineCollection(segs, colors="#0b0b0b", linewidths=lw, capstyle="round"))
            draw(ax); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(XR if zr else EPF.EXT[:2]); ax.set_ylim(zr if zr else EPF.EXT[2:])
            m = np.isfinite(a) & (T.DEP < 0.5); m2 = np.isfinite(a) & (T.DEP >= 3) & (T.DEP < 8)
            if zr is None:
                EPF.label(ax, f"median error {np.median(a[m]):.0f}° within 0.5 mm, {np.median(a[m2]):.0f}° at 3-8 mm")
            if r == 0:
                ax.set_title(t, fontsize=12)
        axs[r, 0].set_ylim(zr if zr else EPF.EXT[2:]); axs[r, 0].set_xlim(XR if zr else EPF.EXT[:2])
        axs[r, 0].set_ylabel(lab, fontsize=13)
    axs[0, 0].set_title("image (x-z, mid y); cyan = interface", fontsize=12)
    fig.subplots_adjust(top=0.9, wspace=0.04, hspace=0.06, left=0.03, right=0.92)
    cb = fig.colorbar(im, cax=fig.add_axes([0.93, 0.3, 0.006, 0.5])); cb.set_label("angle to the true fibre direction (deg)")
    fig.suptitle("PROTOTYPE. Fibre-direction glyphs: intensity structure tensor vs anatomix 48-channel shape space (layer 58, no output "
                 "head), D0 and D15. Sticks = estimated fibre direction (in-plane part; short = out of plane); background = its error.\n"
                 "Truth: fibres cross the interface at 90° (left band), 60° and 30° (right band). No labels, no paired data, no oracle. "
                 "Phantoms grown 9.6 mm per x and z face; vectors computed on the grown block, cropped for display.",
                 fontsize=12.5, x=0.01, ha="left", y=0.985)
    out = T.ST.FIGS / "shape48_glyphs.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "maps":
        maps(); glyph_fig(); sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        glyph_fig(); sys.exit()
    main(tuple(float(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else (6.0, 16.0))
