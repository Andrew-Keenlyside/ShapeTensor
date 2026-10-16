"""The pipeline step by step on one block, one figure and one set of validation numbers per step
(docs/walkthrough.md). A phantom gives every step a number against the truth; a real block gives the same figures
as a consistency check.

    python scripts/walkthrough.py --source phantom:P3-40-t0.5 --rho 10 [--seed 0]
    python scripts/walkthrough.py --source s2146 --rho 10 [--core 256]
    python scripts/walkthrough.py --source bigbrain --rho 10 [--core 256]

Writes figures/walkthrough/<name>/step<N>_*.png and validation.json. GPU.
"""
import argparse, json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor import DATA_ROOT, st, tracking
from shapetensor.features import make_engine
from shapetensor.filters import gaussian
from shapetensor.odf import fit_ring_width, n_coef, real_sh
from shapetensor.pca import PCA
from shapetensor.phantoms import make, stats
from shapetensor.readout import common, single
from shapetensor.tensor import excision_weights, gradients, k_dense, odf_dense
from shapetensor.validate import score
from shapetensor.viz import CAT, SEQ, dec_rgb, eig_field, fixel_field, sticks, streamline_segments

CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
VOI = Path(DATA_ROOT) / "bigbrain" / "bigbrain_20um_x2336_y3936_z2464_768.npy"
VOX = 0.02
BANDS = ("0-0.1mm", "0.1-0.25mm", "0.25-0.5mm", ">2mm")
# CSD settings by texture level (results log §14): response width and the dispersed-tissue-compliant threshold
CSD_BY_TEX = {"t1": (22.9, 0.5), "t0.5": (28.9, 0.7), "t0.25": (28.9, 0.8)}
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)


class Block:
    """The volume, the slab to draw, the site grid, and (phantom) the truth at the sites."""

    def __init__(self, a):
        self.a = a; self.b = 2; self.rho = a.rho; self.sigma = a.sigma; self.tr = 4.0
        if a.source.startswith("phantom:"):
            pid = a.source.split(":", 1)[1]
            img, tr, meta, grid = make.load(pid, a.seed)
            self.name = f"phantom_{pid}"; self.v = torch.as_tensor(img, device="cuda").float(); self.truth = tr; self.grid = grid
            (z0, z1), (x0, x1) = grid.core_zx
            half = 64
            self.core = ((z0, z1), (grid.y_mid - half, grid.y_mid + half), (x0, x1))
            self.y_draw = grid.y_mid
            self.tissue_thr = None
            self.tex = "t0.5" if "t0.5" in pid else ("t0.25" if "t0.25" in pid else "t1")
        else:
            m = int(math.ceil((self.tr * (a.sigma + a.rho) + 64) / self.b) * self.b)
            if a.source == "s2146":
                import zarr
                z0 = zarr.open(CROP, mode="r")["0"]; n = z0.shape[0]
                lo = (n - a.core) // 2 - m; hi = lo + a.core + 2 * m
                vol = np.asarray(z0[lo:hi, lo:hi, lo:hi]).astype(np.float32); self.name = "real_S2146"; self.tissue_thr = (">=", 142.5)
            else:
                full = np.load(VOI, mmap_mode="r"); n = full.shape[0]
                lo = (n - a.core) // 2 - m; hi = lo + a.core + 2 * m
                vol = np.asarray(full[lo:hi, lo:hi, lo:hi]).astype(np.float32); self.name = "bigbrain"; self.tissue_thr = ("<", 250.0)
            self.v = torch.as_tensor(vol, device="cuda"); self.truth = None; self.grid = None
            Z, Y, X = self.v.shape
            self.core = ((m, Z - m), (m, Y - m), (m, X - m)); self.y_draw = Y // 2
            self.tex = "t0.5"
        self.name += f"_rho{a.rho:g}"
        self.out = Path(__file__).resolve().parents[1] / "figures" / "walkthrough" / self.name
        self.out.mkdir(parents=True, exist_ok=True)
        (z0, z1), (y0, y1), (x0, x1) = self.core
        self.zs, self.ys, self.xs = (np.arange(z0, z1, self.b) + 0.5, np.arange(y0, y1, self.b) + 0.5, np.arange(x0, x1, self.b) + 0.5)
        self.jy = int(np.argmin(np.abs(self.ys - self.y_draw)))                   # the site row to draw
        if self.truth is not None:
            self.t = score.sample_truth(self.truth, self.zs - 0.5, self.xs - 0.5)   # (nz*nx) truth on the drawn row
        self.val = {"source": a.source, "rho": a.rho, "sigma": a.sigma, "reduce": self.b, "texture_level": self.tex}
        self.img = self.v[z0:z1, self.y_draw, x0:x1].cpu().numpy()
        self.ext = (0, self.img.shape[1] * VOX, 0, self.img.shape[0] * VOX)

    def tissue_sites(self):
        (z0, z1), (y0, y1), (x0, x1) = self.core
        sub = self.v[z0:z1:self.b, y0:y1:self.b, x0:x1:self.b]
        if self.tissue_thr is None:
            return torch.ones_like(sub, dtype=torch.bool)
        return sub >= self.tissue_thr[1] if self.tissue_thr[0] == ">=" else sub < self.tissue_thr[1]

    def bands(self, d_row):
        """Phantom: median angle by band on the drawn row for directions ``(nz, nx, 3)`` in (z, y, x)."""
        if self.truth is None:
            return None
        err = score.angle_err(d_row.reshape(-1, 3), self.t["dir1"])
        if np.isfinite(self.t["dir2"][:, 0]).any():
            err = np.fmin(err, score.angle_err(d_row.reshape(-1, 3), self.t["dir2"]))
        if not np.isfinite(self.t["dist"]).any():
            return {"all": round(float(np.nanmedian(err[self.t["region"] == 2])), 2)}
        bs = score.band_summary(err, self.t["dist"], self.t["region"])
        return {k: round(float(bs[k]["median"]), 2) for k in BANDS if k in bs and "median" in bs[k]}

    def share_by_band(self, flag_row):
        if self.truth is None or not np.isfinite(self.t["dist"]).any():
            return None
        out = {}
        lo_hi = {"0-0.1mm": (0, 0.1), "0.1-0.25mm": (0.1, 0.25), "0.25-0.5mm": (0.25, 0.5), ">2mm": (2.0, 1e9)}
        f = flag_row.reshape(-1)
        for k, (lo, hi) in lo_hi.items():
            m = (self.t["dist"] >= lo) & (self.t["dist"] < hi) & (self.t["region"] == 2)
            if m.sum() >= 20:
                out[k] = round(float(f[m].mean()), 3)
        return out

    def save(self, fig, name):
        fig.savefig(self.out / name, bbox_inches="tight"); plt.close(fig); print("wrote", self.out / name, flush=True)

    def dump(self):
        json.dump(self.val, open(self.out / "validation.json", "w"), indent=1, default=str)


def row(t, jy):
    """Site-grid tensor ``(..., nz, ny, nx)`` -> the drawn row ``(..., nz, nx)`` as numpy."""
    return t[..., :, jy, :].cpu().numpy()


def axis_off(ax):
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------------------------------------------
def step0(B):
    v = B.v; Z, Y, X = v.shape
    cz, cy, cx = Z // 2, Y // 2, X // 2
    sub = v[cz - 128:cz + 128, cy - 128:cy + 128, cx - 128:cx + 128]
    subn = sub.cpu().numpy()
    mask = None if B.tissue_thr is None else ((subn >= B.tissue_thr[1]) if B.tissue_thr[0] == ">=" else (subn < B.tissue_thr[1]))
    prof = stats.profile(subn, mask)
    real = json.load(open(Path(DATA_ROOT) / "results" / "real_stats_S2146.json"))["all"]
    rms = {k: float(prof["dog_rms"][k]["rms_texture"]) for k in prof["dog_rms"]}
    rms_real = {k: float(real["dog_rms"][k]["rms_texture"]) for k in real["dog_rms"]}
    B.val["step0"] = {"noise_sd": float(prof["noise_sd"]), "texture_rms": rms, "real_crop_texture_rms": rms_real,
                      "texture_ratio_to_real_crop_1-2vox": round(rms["1-2"] / rms_real["1-2"], 3)}
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.5), dpi=100)
    axs[0].imshow(B.img, cmap="gray", origin="lower", extent=B.ext); axs[0].set_title(f"{B.name}: axial plane (z, x) at the drawn row"); axis_off(axs[0])
    z0, x0 = B.img.shape[0] // 2 - 64, B.img.shape[1] // 2 - 64
    axs[1].imshow(B.img[z0:z0 + 128, x0:x0 + 128], cmap="gray", origin="lower", extent=(0, 2.56, 0, 2.56), interpolation="nearest"); axs[1].set_title("centre 2.56 mm at full 20 µm"); axis_off(axs[1])
    axs[2].hist(sub.cpu().numpy().ravel(), bins=128, color=CAT[0]); axs[2].set_title("intensity histogram (central 256³)"); axs[2].set_yticks([])
    ks = list(rms); axs[3].plot(ks, [rms[k] for k in ks], "o-", color=CAT[0], label="this block"); axs[3].plot(ks, [rms_real[k] for k in ks], "s--", color=CAT[1], label="HiP-CT S-21-46 crop")
    axs[3].set_ylabel("texture RMS (noise-corrected)"); axs[3].set_xlabel("DoG scale (voxels)"); axs[3].legend(); axs[3].set_title(f"texture profile; noise SD {prof['noise_sd']:.1f}")
    fig.suptitle("Step 0 — the block and its texture", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step0_block.png")


def step1(B):
    G = gradients(B.v[None], B.sigma, B.tr)                                  # (3, Z, Y, X): z, y, x derivatives
    (z0, z1), _, (x0, x1) = B.core
    g = G[:, z0:z1, B.y_draw, x0:x1]
    mag = g.norm(dim=0).cpu().numpy(); d = (g / g.norm(dim=0, keepdim=True).clamp_min(1e-9)).cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    im = axs[0].imshow(mag, cmap=cmap, origin="lower", extent=B.ext, vmax=np.percentile(mag, 99)); fig.colorbar(im, ax=axs[0], fraction=0.04); axs[0].set_title(f"|∇I| at σ {B.sigma:g}"); axis_off(axs[0])
    axs[1].imshow(dec_rgb(d, np.clip(mag / np.percentile(mag, 95), 0, 1)), origin="lower", extent=B.ext, interpolation="nearest"); axs[1].set_title("gradient direction (DEC: R=|x| G=|y| B=|z|) × magnitude"); axis_off(axs[1])
    pz, px = mag.shape[0] // 2 - 24, mag.shape[1] // 2 - 24
    patch = B.img[pz:pz + 48, px:px + 48]
    axs[2].imshow(patch, cmap="gray", origin="lower", extent=(0, 48 * VOX, 0, 48 * VOX))
    zz, xx = np.meshgrid((np.arange(0, 48, 3) + 0.5) * VOX, (np.arange(0, 48, 3) + 0.5) * VOX, indexing="ij")
    gz, gx = d[0, pz:pz + 48:3, px:px + 48:3], d[2, pz:pz + 48:3, px:px + 48:3]
    axs[2].quiver(xx, zz, gx, gz, color=CAT[1], scale=18, width=0.004, headwidth=3); axs[2].set_title("gradients on a 0.96 mm patch"); axis_off(axs[2])
    val = {"grad_mag_median": float(np.median(mag)), "grad_mag_p99": float(np.percentile(mag, 99))}
    if B.truth is not None:
        dz, dx = B.zs.astype(int), B.xs.astype(int)
        dd = d[:, dz - z0, :][:, :, dx - x0].transpose(1, 2, 0).reshape(-1, 3)
        ang = score.angle_err(dd, B.t["dir1"]); wm = (B.t["region"] == 2) & np.isfinite(ang)
        val["gradient_vs_fibre_angle_median_wm"] = round(float(np.median(ang[wm])), 1)
    B.val["step1"] = val
    fig.suptitle("Step 1 — gradients at σ (a fibre leaves gradients in its normal plane; an interface one strong gradient)", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step1_gradients.png")
    return G


def step2(B):
    gm = sum(gaussian(B.v, 3.0, order=o, truncate=B.tr) ** 2 for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))).sqrt()
    w = excision_weights(B.v, 3.0, 3.0, B.tr)
    (z0, z1), _, (x0, x1) = B.core
    gmr = gm[z0:z1, B.y_draw, x0:x1].cpu().numpy(); mask = (w[z0:z1, B.y_draw, x0:x1] == 0).cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    im = axs[0].imshow(gmr, cmap=cmap, origin="lower", extent=B.ext, vmax=np.percentile(gmr, 99.5)); fig.colorbar(im, ax=axs[0], fraction=0.04); axs[0].set_title("coarse gradient magnitude gm (σ 3)"); axis_off(axs[0])
    axs[1].imshow(B.img, cmap="gray", origin="lower", extent=B.ext); axs[1].imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([CAT[1]]), origin="lower", extent=B.ext, alpha=0.8, interpolation="nearest")
    axs[1].set_title(f"excision mask: gm > 3 × median ({float(mask.mean()):.1%} of this plane, {float(1 - w.mean()):.1%} of the block)"); axis_off(axs[1])
    axs[2].hist(gmr.ravel(), bins=100, color=CAT[0], log=True); axs[2].axvline(3 * float(gm.median()), color=CAT[1]); axs[2].set_title("gm histogram and the 3 × median threshold"); axs[2].set_yticks([])
    val = {"excised_share_block": float(1 - w.mean())}
    if B.truth is not None and np.isfinite(B.truth["dist"]).any():
        dist = B.truth["dist"]; near = (dist < 0.1)[z0:z1, x0:x1]
        tp = (mask & near).sum(); val["mask_precision_vs_within_0.1mm"] = round(float(tp / max(1, mask.sum())), 3); val["mask_recall_of_within_0.1mm"] = round(float(tp / max(1, near.sum())), 3)
        from sklearn.metrics import average_precision_score
        val["gm_pr_auc_within_0.1mm"] = round(float(average_precision_score(near.ravel(), gmr.ravel())), 3)
    B.val["step2"] = val
    fig.suptitle("Step 2 — the interface map and the excision mask (scale separates the interface from the texture)", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step2_interface.png")
    return w, gm


def step3_4(B, G, w, gm):
    S, sites = k_dense(G, B.rho, B.tr, B.b, B.core); d_st, gap_st, cl_st = eig_field(S.permute(3, 0, 1, 2).contiguous()); del S
    Sx, _ = k_dense(G, B.rho, B.tr, B.b, B.core, weights=w.half()); d_ex, gap_ex, cl_ex = eig_field(Sx.permute(3, 0, 1, 2).contiguous()); del Sx
    Kg, _ = k_dense(gm[None].half(), B.rho, B.tr, B.b, B.core); gm_s = Kg[..., 0].clamp_min(0).sqrt(); del Kg
    diff = torch.rad2deg(torch.arccos((d_st[0] * d_ex[0]).sum(0).abs().clamp(max=1)))
    jy = B.jy
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    im = axs[0].imshow(row(diff, jy), cmap=cmap, vmin=0, vmax=45, origin="lower", extent=B.ext, interpolation="nearest"); fig.colorbar(im, ax=axs[0], fraction=0.04); axs[0].set_title("angle: excised vs plain tensor (deg)"); axis_off(axs[0])
    im = axs[1].imshow(row(gm_s, jy), cmap=cmap, origin="lower", extent=B.ext, interpolation="nearest"); fig.colorbar(im, ax=axs[1], fraction=0.04); axs[1].set_title("boundary strength at the sites (gm integrated at ρ)"); axis_off(axs[1])
    tis = B.tissue_sites()
    q = torch.quantile(gm_s[tis], 0.9)
    val = {"plain_vs_excised_median_deg": float(diff[tis].median()), "plain_vs_excised_median_high_gm": float(diff[tis & (gm_s > q)].median()), "plain_vs_excised_median_low_gm": float(diff[tis & (gm_s <= q)].median())}
    if B.truth is not None:
        val["plain_bands"] = B.bands(row(d_st[0], jy).transpose(1, 2, 0)); val["excised_bands"] = B.bands(row(d_ex[0], jy).transpose(1, 2, 0))
        txt = "\n".join(f"{k}: plain {val['plain_bands'].get(k, float('nan')):.1f}°, excised {val['excised_bands'].get(k, float('nan')):.1f}°" for k in val["plain_bands"])
    else:
        txt = f"plain vs excised: {val['plain_vs_excised_median_deg']:.1f}° median; {val['plain_vs_excised_median_high_gm']:.1f}° in the top gm decile"
    axs[2].axis("off"); axs[2].text(0.02, 0.95, "Validation\n" + txt, va="top", fontsize=12, family="monospace")
    B.val["step3"] = val
    fig.suptitle(f"Step 3 — the structure tensor and the excised tensor at ρ {B.rho:g} voxels ({B.rho * VOX:.2f} mm)", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step3_tensors.png")
    # step 4: direction, gap, cl, sticks
    fig, axs = plt.subplots(2, 3, figsize=(18, 11), dpi=100)
    for ax, (d, g, name) in zip(axs[0], ((d_st, gap_st, "structure tensor"), (d_ex, gap_ex, "excised tensor"))):
        ax.imshow(dec_rgb(row(d[0], jy), row(g, jy) ** 0.5), origin="lower", extent=B.ext, interpolation="nearest"); ax.set_title(f"DEC: {name} (× √gap)"); axis_off(ax)
    ax = axs[0, 2]; ax.imshow(B.img, cmap="gray", origin="lower", extent=B.ext, alpha=0.6)
    dd = row(d_ex[0], jy); nz, nx = dd.shape[1:]
    sel = np.zeros((nz, nx), bool); sel[::6, ::6] = True
    Zc, Xc = np.meshgrid((B.zs - B.core[0][0]) * VOX, (B.xs - B.core[2][0]) * VOX, indexing="ij")
    dsel = dd[:, sel].T
    if B.truth is not None:
        err = score.angle_err(dsel, B.t["dir1"][sel.ravel()]); cols = cmap(np.clip(err / 90, 0, 1))
        ax.set_title("excised direction sticks, coloured by error to the truth")
    else:
        cols = dec_rgb(dsel.T[:, :, None])[:, 0]; ax.set_title("excised direction sticks (DEC colour)")
    sticks(ax, Zc[sel], Xc[sel], dsel, 6 * B.b * VOX * 0.9, cols); axis_off(ax)
    for ax, (g, name, vmax) in zip(axs[1], ((gap_ex, "orientation gap (λ2−λ3)/λ2, excised", 1.0), (cl_ex, "Westin cl (sheet-likeness), excised", 0.6))):
        im = ax.imshow(row(g, jy), cmap=cmap, vmin=0, vmax=vmax, origin="lower", extent=B.ext, interpolation="nearest"); fig.colorbar(im, ax=ax, fraction=0.04); ax.set_title(name); axis_off(ax)
    axs[1, 2].axis("off")
    v4 = {"gap_median_tissue": float(gap_ex[tis].median()), "cl_median_tissue": float(cl_ex[tis].median())}
    if B.truth is not None:
        v4["excised_bands"] = val["excised_bands"]
    axs[1, 2].text(0.02, 0.95, "Validation\n" + "\n".join(f"{k}: {v}" for k, v in v4.items()), va="top", fontsize=11, family="monospace")
    B.val["step4"] = v4
    fig.suptitle("Step 4 — one direction per site and its quality (dim colour = undetermined; high cl = a sheet, not a fibre)", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step4_direction.png")
    return d_st, d_ex, gap_ex, gm_s, tis


def step5(B, G):
    O, sites = odf_dense(G, B.rho, B.tr, B.b, B.core, lmax=8)             # (nz, ny, nx, 45)
    jy = B.jy; Or = O[:, jy]                                                 # (nz, nx, 45)
    # order-by-order power
    l_of = np.concatenate([[l] * (2 * l + 1) for l in range(0, 9, 2)])
    power = {int(l): float((Or[..., l_of == l] ** 2).sum(-1).mean()) for l in (0, 2, 4, 6, 8)}
    # polar glyphs in the drawn (z, x) plane at a few sites
    ang = torch.linspace(0, 2 * math.pi, 181, device="cuda")
    u = torch.stack([torch.cos(ang), torch.zeros_like(ang), torch.sin(ang)], 1)         # (z, y, x) unit vectors in the plane
    Y = real_sh(u, 8)                                                                      # (181, 45)
    nz, nx = Or.shape[:2]
    picks = [(nz // 4, nx // 4), (nz // 4, 3 * nx // 4), (nz // 2, nx // 2), (3 * nz // 4, nx // 4), (3 * nz // 4, 3 * nx // 4), (nz // 2, nx // 8)]
    fig = plt.figure(figsize=(20, 6), dpi=100)
    ax0 = fig.add_subplot(1, 4, 1); ax0.imshow(B.img, cmap="gray", origin="lower", extent=B.ext)
    for k, (iz, ix) in enumerate(picks):
        ax0.plot((B.xs[ix] - B.core[2][0]) * VOX, (B.zs[iz] - B.core[0][0]) * VOX, "o", color=CAT[k % 4], ms=8, mec="white")
    ax0.set_title("sites of the glyphs"); axis_off(ax0)
    for k, (iz, ix) in enumerate(picks):
        ax = fig.add_subplot(2, 6, 3 + k + (3 if k >= 3 else 0), projection="polar")
        f = (Y @ Or[iz, ix]).clamp_min(0).cpu().numpy()
        ax.plot(ang.cpu().numpy(), f, color=CAT[k % 4]); ax.fill(ang.cpu().numpy(), f, color=CAT[k % 4], alpha=0.3); ax.set_yticks([]); ax.set_xticks([])
        if B.truth is not None:
            t1 = B.t["dir1"].reshape(nz, nx, 3)[iz, ix]
            if np.isfinite(t1[0]):
                th = math.atan2(t1[2], t1[0]); ax.plot([th, th + math.pi], [f.max(), f.max()], color="k", lw=1)
        ax.set_title(f"site {k + 1}", fontsize=9)
    ax2 = fig.add_subplot(1, 4, 4); ax2.bar([str(l) for l in power], [power[l] for l in power], color=CAT[0]); ax2.set_xlabel("SH order l"); ax2.set_ylabel("mean power"); ax2.set_title("ODF power by order (l ≤ 2 = the tensor)")
    val = {"power_by_order": power}
    if B.truth is not None:
        wm = (B.t["region"] == 2) & np.isfinite(B.t["dir1"][:, 0])
        if B.t["dist"] is not None and np.isfinite(B.t["dist"]).any():
            wm &= B.t["dist"] > 0.5
        width = fit_ring_width(Or.reshape(-1, 45)[torch.as_tensor(wm, device="cuda")], torch.as_tensor(B.t["dir1"][wm], device="cuda"), 8)
        val["ring_width_fitted_deg"] = round(float(np.degrees(width)), 1); val["ring_width_calibrated_deg"] = CSD_BY_TEX[B.tex][0]
    B.val["step5"] = val
    fig.suptitle("Step 5 — the gradient-direction ODF (order 8); glyphs in the drawn plane, the black bar is the true fibre (gradients peak at 90° to it)", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step5_odf.png")
    return O


def step6_7(B, O, gm_s, tis, d_ex):
    width, rel = CSD_BY_TEX[B.tex]
    dirs, fracs, npop = fixel_field(O, width, rel)
    gate = gm_s > 3.0 * gm_s[tis].median()
    dirs_g, fracs_g, npop_g = fixel_field(O, width, rel, gate)
    jy = B.jy
    ncm = ListedColormap(["#ffffff", "#cde2fb", "#6da7ec", "#184f95"])
    Zc, Xc = np.meshgrid((B.zs - B.core[0][0]) * VOX, (B.xs - B.core[2][0]) * VOX, indexing="ij")
    for name, (D, Fr, N), fname, title in (("step6", (dirs, fracs, npop), "step6_fixels.png", f"Step 6 — CSD fixels (response {width:g}°, threshold {rel:g})"),
                                           ("step7", (dirs_g, fracs_g, npop_g), "step7_gate.png", "Step 7 — the interface gate on the fixels (gm > 3 × median → no population)")):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
        axs[0].imshow(row(N, jy), cmap=ncm, vmin=-0.5, vmax=3.5, origin="lower", extent=B.ext, interpolation="nearest"); axs[0].set_title("N populations (white 0, light 1, mid 2, dark 3)"); axis_off(axs[0])
        axs[1].imshow(B.img, cmap="gray", origin="lower", extent=B.ext, alpha=0.6)
        nz, nx = N.shape[0], N.shape[2]; sel = np.zeros((nz, nx), bool); sel[::6, ::6] = True
        for k in range(3):
            dk = row(D[k], jy); fk = row(Fr[k], jy); ok = sel & (fk > 0)
            sticks(axs[1], Zc[ok], Xc[ok], dk[:, ok].T, 6 * B.b * VOX * 0.9, CAT[k], lw=1.0)
        axs[1].set_title("fixel sticks (colour = rank)"); axis_off(axs[1])
        f2 = row(Fr[1], jy); im = axs[2].imshow(np.ma.masked_where(f2 <= 0, f2), cmap=cmap, vmin=0, vmax=0.5, origin="lower", extent=B.ext, interpolation="nearest"); fig.colorbar(im, ax=axs[2], fraction=0.04); axs[2].set_title("fraction of the second population"); axis_off(axs[2])
        val = {"npop_shares_tissue": [float((N[tis] == k).float().mean()) for k in range(4)]}
        if name == "step7":
            val["gated_share_tissue"] = float(gate[tis].float().mean())
        if B.truth is not None:
            val["n_ge2_share_by_band"] = B.share_by_band((row(N, jy) >= 2))
            if np.isfinite(B.t["dir2"][:, 0]).any():
                d1 = row(D[0], jy).transpose(1, 2, 0).reshape(-1, 3); d2 = row(D[1], jy).transpose(1, 2, 0).reshape(-1, 3); cnt = row(N, jy).reshape(-1)
                sc = score.multi_scores(np.stack([d1, d2, np.zeros_like(d1)], 1), np.stack([row(Fr[0], jy).reshape(-1), row(Fr[1], jy).reshape(-1), np.zeros(len(d1))], 1), cnt, B.t["dir1"], B.t["dir2"], B.t["npop"], B.t["frac1"])
                val["crossing"] = {k: sc.get("two", {}).get(k) for k in ("found2_share", "both_within_15_share")}
            else:
                val["false_second_share_far"] = (val["n_ge2_share_by_band"] or {}).get(">2mm")
        B.val[name] = val
        fig.suptitle(title, x=0.01, ha="left", fontsize=12)
        B.save(fig, fname)
    return dirs_g, fracs_g, npop_g


def step8(B, d_st, tis):
    F = make_engine("anatomix48", nyquist="none", n_pcs=8)(B.v)
    (z0, z1), _, (x0, x1) = B.core
    rgb = F[:3, z0:z1, B.y_draw, x0:x1].float().cpu().numpy()
    lo, hi = np.percentile(rgb, 1, axis=(1, 2)), np.percentile(rgb, 99, axis=(1, 2))
    rgb = np.clip((rgb - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1).transpose(1, 2, 0)
    G = gradients(F, B.sigma, B.tr); del F
    K, _ = k_dense(G, B.rho, B.tr, B.b, B.core); del G
    Kp = K.reshape(-1, K.shape[-1]); C3 = common.nc3_from_packed(Kp.shape[-1]); Kf = common.unpack(Kp, C3)
    S3 = sum(Kf[:, 3 * c:3 * c + 3, 3 * c:3 * c + 3] for c in range(C3 // 3))
    shp = K.shape[:3]
    e = single.eigen(S3); d_sh = e["dir"].T.reshape((3,) + shp); gap_sh = e["gap"].reshape(shp)
    tc = single.tca(Kf, rank=3); d_tc = tc["dir"].T.reshape((3,) + shp); del Kf, Kp, K
    jy = B.jy
    diff = torch.rad2deg(torch.arccos((d_st[0] * d_sh).sum(0).abs().clamp(max=1)))
    fig, axs = plt.subplots(1, 4, figsize=(23, 6), dpi=100)
    axs[0].imshow(rgb, origin="lower", extent=B.ext, interpolation="nearest"); axs[0].set_title("feature RGB: anatomix48 PCs 1–3"); axis_off(axs[0])
    axs[1].imshow(dec_rgb(row(d_sh, jy), row(gap_sh, jy) ** 0.5), origin="lower", extent=B.ext, interpolation="nearest"); axs[1].set_title("DEC: shape tensor (summed channel tensor)"); axis_off(axs[1])
    axs[2].imshow(dec_rgb(row(d_tc, jy)), origin="lower", extent=B.ext, interpolation="nearest"); axs[2].set_title("DEC: TCA rank 3 on the channel-resolved tensor"); axis_off(axs[2])
    im = axs[3].imshow(row(diff, jy), cmap=cmap, vmin=0, vmax=90, origin="lower", extent=B.ext, interpolation="nearest"); fig.colorbar(im, ax=axs[3], fraction=0.04); axs[3].set_title("angle: shape tensor vs structure tensor"); axis_off(axs[3])
    val = {"shape_vs_structure_median_deg": float(diff[tis].median()), "shape_gap_median": float(gap_sh[tis].median())}
    if B.truth is not None:
        val["shape_eigen_bands"] = B.bands(row(d_sh, jy).transpose(1, 2, 0)); val["tca_bands"] = B.bands(row(d_tc, jy).transpose(1, 2, 0))
    B.val["step8"] = val
    fig.suptitle("Step 8 — the shape engine: features, the shape tensor and TCA (a negative result at the interface; a gradient source for the ODF)", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step8_shape.png")
    torch.cuda.empty_cache()


def step9(B, d_ex, dirs_g, tis):
    shp = tis.shape
    g = torch.Generator(device="cuda").manual_seed(0)
    cand = tis.clone(); cand[:4] = cand[-4:] = False; cand[:, :2] = cand[:, -2:] = False; cand[:, :, :4] = cand[:, :, -4:] = False
    idx = cand.nonzero(); sel = idx[torch.randperm(idx.shape[0], generator=g, device="cuda")[:6000]].float()
    seeds = sel + torch.rand(sel.shape, generator=g, device="cuda") - 0.5
    site_mm = B.b * VOX
    res = {}
    for name, D in (("excised tensor", d_ex), ("fixels", dirs_g)):
        r = tracking.track(D, tis, seeds, step=0.5, max_angle_deg=15.0, max_steps=600)
        L = r["length"] * site_mm; keep = L >= 0.2
        res[name] = {"r": r, "keep": keep, "den": tracking.density(r["points"][keep], shp), "L": L}
    ref, fx = res["excised tensor"], res["fixels"]
    both = ref["keep"] & fx["keep"]
    div = tracking.endpoint_divergence(ref["r"], fx["r"], 0.5) * site_mm
    val = {name: {"kept_share": float(x["keep"].float().mean()), "length_mm_median": float(x["L"][x["keep"]].median()) if x["keep"].any() else 0.0,
                  "secondary_fixel_streamlines": float((x["r"]["used_secondary"].sum(1)[x["keep"]] > 0).float().mean()) if x["keep"].any() else 0.0} for name, x in res.items()}
    val["fixels_vs_excised"] = {"density_dice": tracking.dice(ref["den"], fx["den"]), "divergence_mm_median": float(div[both].median()) if both.any() else 0.0}
    B.val["step9"] = val
    jy = B.jy
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    for ax, (name, x) in zip(axs[:2], res.items()):
        ax.imshow(B.img, cmap="gray", origin="lower", extent=B.ext, alpha=0.7)
        P = x["r"]["points"][x["keep"]].cpu().numpy()
        P = P[:, :, :, [0, 1, 2]]
        # project tracks within ±2 sites of the drawn row onto the (x, z) plane: swap so that y is the slab axis
        Pz = P.copy(); Pz[..., [0, 1]] = P[..., [1, 0]]                      # (y, z, x) so that streamline_segments' z-selection uses y
        streamline_segments(ax, Pz, lambda q: np.abs(q[:, 0] - jy).min() <= 2.0, site_mm)
        ax.set_xlim(B.ext[0], B.ext[1]); ax.set_ylim(B.ext[2], B.ext[3]); axis_off(ax)
        ax.set_title(f"{name}: {val[name]['kept_share']:.0%} seeds kept, median {val[name]['length_mm_median']:.2f} mm")
    d = np.log1p(row(fx["den"], jy)) - np.log1p(row(ref["den"], jy))
    im = axs[2].imshow(d, cmap="RdBu_r", vmin=-3, vmax=3, origin="lower", extent=B.ext, interpolation="nearest"); fig.colorbar(im, ax=axs[2], fraction=0.04)
    axs[2].set_title(f"fixels − excised (log density); Dice {val['fixels_vs_excised']['density_dice']:.2f}, divergence {val['fixels_vs_excised']['divergence_mm_median']:.2f} mm"); axis_off(axs[2])
    fig.suptitle("Step 9 — a first tractography on the single-vector and the fixel field (same seeds, same tracker)", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step9_tracks.png")


def step10(B, d_st):
    res = {}
    jy = B.jy
    fig, axs = plt.subplots(2, 3, figsize=(18, 11), dpi=100)
    axs[0, 0].imshow(dec_rgb(row(d_st[0], jy)), origin="lower", extent=B.ext, interpolation="nearest"); axs[0, 0].set_title("isotropic structure tensor"); axis_off(axs[0, 0])
    axs[1, 0].axis("off")
    for r_, (axis, aname) in enumerate(((1, "y (across the drawn plane)"), (0, "z (within the drawn plane)"))):
        size = list(B.v.shape); size[axis] = int(round(size[axis] / 2.5))
        va = Fn.interpolate(B.v[None, None], size=size, mode="area")[0, 0]
        sp = [1.0, 1.0, 1.0]; sp[axis] = 2.5
        core_a = list(B.core); core_a[axis] = (int(round(B.core[axis][0] / 2.5)), int(round(B.core[axis][1] / 2.5)))
        Ga = gradients(va[None], B.sigma, B.tr)
        out = {}
        for name, kw in (("naive", {}), ("spacing", {"spacing": tuple(sp)})):
            S = st.structure_tensor_3d(va, B.sigma, B.rho, B.tr, **kw)
            Sc = S[(slice(None),) + tuple(slice(c[0], c[1], B.b) for c in core_a)]
            d, _, _ = eig_field(Sc.contiguous()); del S, Sc
            # compare with the isotropic field sampled onto the coarse site grid
            n_a = d.shape[2 + axis]
            ref = Fn.interpolate(d_st[0][None].float(), size=[d_st.shape[2 + k] if k != axis else n_a for k in range(3)], mode="nearest")[0]
            ang = torch.rad2deg(torch.arccos((ref * d[0]).sum(0).abs().clamp(max=1)))
            out[name] = {"vs_isotropic_median_deg": float(ang.median())}
            jr = jy if axis == 0 else int(round(jy / 2.5))
            ax = axs[r_, 1 if name == "naive" else 2]
            ax.imshow(dec_rgb(row(d[0], jr)), origin="lower", extent=B.ext, interpolation="nearest", aspect="auto"); axis_off(ax)
            if B.truth is not None and axis == 1:
                out[name]["bands"] = B.bands(row(d[0], jr).transpose(1, 2, 0))
            ax.set_title(f"coarse axis {aname}: {name} — {out[name]['vs_isotropic_median_deg']:.1f}° to isotropic" + (f", bands {out[name]['bands']}" if "bands" in out[name] else ""), fontsize=9)
        res[aname] = out; del Ga, va; torch.cuda.empty_cache()
    B.val["step10"] = res
    fig.suptitle("Step 10 — anisotropic voxels (2.5× coarser along one axis): naive voxel-unit filters vs spacing-aware physical filters", x=0.01, ha="left", fontsize=12)
    B.save(fig, "step10_anisotropy.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="phantom:P3-40-t0.5"); ap.add_argument("--rho", type=float, default=10.0); ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--core", type=int, default=256); ap.add_argument("--no-shape", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    B = Block(a); print(f"block {tuple(B.v.shape)}, core {B.core}, drawn row {B.y_draw}; {B.name}", flush=True)
    step0(B); G = step1(B); w, gm = step2(B)
    d_st, d_ex, gap_ex, gm_s, tis = step3_4(B, G, w, gm)
    O = step5(B, G); del G, w; torch.cuda.empty_cache()
    dirs_g, fracs_g, npop_g = step6_7(B, O, gm_s, tis, d_ex); del O; torch.cuda.empty_cache()
    if not a.no_shape:
        step8(B, d_st, tis)
    step9(B, d_ex, dirs_g, tis)
    step10(B, d_st)
    B.dump(); print(f"WALKTHROUGH DONE {time.time() - t0:.0f} s -> {B.out}", flush=True)


if __name__ == "__main__":
    main()
