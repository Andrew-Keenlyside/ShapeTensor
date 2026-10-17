"""The activation walkthrough on real HiP-CT data (S-21-46 crop), populations unknown and everything self-calibrated
on the data: the teacher is the excised structure tensor at its confident sites in bright tissue (sampled evenly
over orientation), the background signature is the lowest-gap tissue, the dictionary and the per-voxel distribution
readout are fitted on those, the populations of the analysis block are the modes of its summed distribution, each
voxel is unmixed onto their signatures, and the masks are the activations above the wrong-population null.

    python scripts/fig_activation_real.py [--calib 384] [--block 192] [--rho 10]
      -> figures/tca_field/activation_real_S2146.png, activation_real_S2146_field.png, results/activation_real_S2146.json
"""
import argparse, json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn.functional as Fn
import zarr

from shapetensor import DATA_ROOT, st
from shapetensor.features import make_engine
from shapetensor.features.cache import cached_features
from shapetensor.readout.layers import (SignatureDictionary, activation_null, compress_activation, export_population_field, layer_tensor_direction,
                                        nnls_batched, population_masks, unmix)
from shapetensor.readout.orientation import OrientationReadoutDist, modes_from_density
from shapetensor.viz import sticks

CROP = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
SIGMA, TR, RHO_TEACHER, VOX, BRIGHT = 1.0, 4.0, 8.0, 0.02, 142.5


def q_of(x, q, n=2_000_000):
    x = x.reshape(-1)
    if x.numel() > n:
        x = x[torch.randint(0, x.numel(), (n,), device=x.device)]
    return torch.quantile(x.float(), q)


def teacher(v):
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO_TEACHER, TR)
    val, vec = st.eig_special_3d(S)
    return vec.flip(0), ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1)


def stratified_weights(d, gap, tissue, q=0.7, n_bins=24):
    """Confident single-population voxels, weighted so that every orientation bin (on a hemisphere grid) contributes
    the same: the dictionary and the readout must not be dominated by the block's main bundle."""
    from shapetensor.readout.orientation import hemisphere
    u = hemisphere(n_bins, d.device)
    conf = tissue & (gap > q_of(gap[tissue], q))
    idx = conf.nonzero()
    dv = d[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    b = ((dv @ u.T) ** 2).argmax(1)
    counts = torch.bincount(b, minlength=len(u)).float()
    w = torch.zeros_like(gap); w[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0 / counts[b].clamp_min(1)
    return w, counts


def lambert(u):
    """Hemisphere directions (z, y, x) with z >= 0 -> 2-D Lambert azimuthal coordinates (x', y')."""
    z, y, x = u[:, 0], u[:, 1], u[:, 2]
    r = torch.sqrt(2 * (1 - z).clamp_min(0)) / math.sqrt(2)
    phi = torch.atan2(y, x)
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi)], 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calib", type=int, default=384); ap.add_argument("--block", type=int, default=192); ap.add_argument("--rho", type=float, default=10.0); ap.add_argument("--nmax", type=int, default=4); ap.add_argument("--smooth", type=int, default=3)
    ap.add_argument("--transform", default="none", choices=["none", "sqrt", "log1p", "unit"], help="compression of the features before the dictionary"); ap.add_argument("--act-transform", default="none", choices=["none", "sqrt", "log"], help="compression of the activations after the unmixing")
    a = ap.parse_args(); t0 = time.time(); sfx = ("" if a.transform == "none" else f"_t{a.transform}") + ("" if a.act_transform == "none" else f"_a{a.act_transform}")
    z0 = zarr.open(CROP, mode="r")["0"]; n = z0.shape[0]
    lo = (n - a.calib) // 2; hi = lo + a.calib
    vc = torch.as_tensor(np.asarray(z0[lo:hi, lo:hi, lo:hi]).astype(np.float32), device="cuda")
    eng = make_engine("anatomix_enc2", nyquist="none")
    Fc = cached_features(eng, vc, tag="S2146_calib")
    d, gap = teacher(vc); tissue = vc >= BRIGHT
    inner = (slice(32, a.calib - 32),) * 3
    w = torch.zeros_like(gap); wi, counts = stratified_weights(d[(slice(None),) + inner], gap[inner], tissue[inner]); w[inner] = wi
    wb = torch.zeros_like(gap); wb[inner] = ((gap[inner] < q_of(gap[inner][tissue[inner]], 0.3)) & tissue[inner]).float()
    dic = SignatureDictionary(smooth=a.smooth, transform=a.transform).fit([(Fc, d, w)], background=[(Fc, wb)])
    ro = OrientationReadoutDist(smooth=a.smooth).fit([(Fc, d, w)], n_sample=240_000)
    tau = activation_null(dic, [(Fc, d, w)])
    print(f"[{time.time() - t0:4.0f}s] self-calibrated on the {a.calib}^3 block: orientation bins occupied {int((counts > 0).sum())}/{len(counts)}, readout target overlap {ro.fit_r2:.2f}, tau {tau:.3f}", flush=True)
    # ---- the analysis block: the centre of the calibration block
    o = (a.calib - a.block) // 2; blk = (slice(o, o + a.block),) * 3
    v = vc[blk]; F = Fc[(slice(None),) + blk]; f = dic._prep(F); tis = tissue[blk]; d_b, gap_b = d[(slice(None),) + blk], gap[blk]
    del Fc; torch.cuda.empty_cache()
    Z = a.block; y_mid = Z // 2; m = 24
    row = (slice(m, Z - m), y_mid, slice(m, Z - m)); img = v[row].cpu().numpy()
    P = ro.predict_dist(F)
    sl = (slice(m, Z - m), slice(y_mid - 4, y_mid + 5), slice(m, Z - m))
    dens = (P[(slice(None),) + sl] * tis[sl].float()[None]).reshape(P.shape[0], -1).sum(1)
    modes, h, nmod = modes_from_density(dens, ro.grid, rel=0.3, min_sep_deg=25.0, nmax=a.nmax, mass_floor=0.1)
    # keep the modes that own a mask (the N rule of results log §23)
    keep = list(range(nmod))
    while keep:
        A, R = unmix(F, dic, modes[keep]); As = Fn.avg_pool3d(A[None], 3, stride=1, padding=1)[0]
        masks, union, strongest = population_masks(As, tau)
        share = torch.stack([(mk & tis).float().sum() / tis.float().sum().clamp_min(1) for mk in masks])
        worst = int(share.argmin())
        if float(share[worst]) >= 0.05 or len(keep) == 1:
            break
        keep.pop(worst)
    modes = modes[keep]; N = len(keep); A = compress_activation(A, a.act_transform)                     # masks stay from the raw averaged activations
    S_list = [dic.signature(mm) for mm in modes]
    print(f"[{time.time() - t0:4.0f}s] block: {nmod} candidate modes, {N} kept (mask shares {[round(float(s), 2) for s in share]}); residual median {float(R[tis].median()):.2f}", flush=True)
    # per-window N over the slab
    half = int(a.rho * 2); step = 8
    zs = list(range(m + half, Z - m - half, step)); xs = list(range(m + half, Z - m - half, step))
    Nmap = np.zeros((len(zs), len(xs)), int)
    for i, zz in enumerate(zs):
        for j, xx in enumerate(xs):
            s2 = (slice(None), slice(zz - half, zz + half + 1), slice(y_mid - half // 2, y_mid + half // 2 + 1), slice(xx - half, xx + half + 1))
            dd = (P[s2].reshape(P.shape[0], -1) * tis[s2[1:]].float().reshape(-1)[None]).sum(1)
            _, _, nn = modes_from_density(dd, ro.grid, rel=0.64, mass_floor=0.1); Nmap[i, j] = nn
    del P; torch.cuda.empty_cache()
    # layer directions and their agreement with the plain tensor inside each mask
    S_img = st.structure_tensor_3d(v, SIGMA, a.rho, TR); _, vec = st.eig_special_3d(S_img); d_img = vec.flip(0); del S_img
    layers = []
    for k in range(N):
        dl, gl = layer_tensor_direction(A[k], SIGMA, a.rho, TR)
        ang_img = torch.rad2deg(torch.arccos((dl * d_img).sum(0).abs().clamp(max=1)))
        ang_mode = torch.rad2deg(torch.arccos((dl * modes[k][:, None, None, None]).sum(0).abs().clamp(max=1)))
        mk = masks[k] & tis
        ak = A[k]; d2 = (ak[2:] - ak[:-2]).abs()[tis[1:-1]].mean()
        layers.append({"dirs": dl, "gap": gl, "mask_share": round(float(mk.float().sum() / tis.float().sum()), 3),
                       "spike_p99_over_median_in_mask": round(float(q_of(ak[mk], 0.99) / ak[mk].median().clamp_min(1e-6)), 2) if mk.any() else None, "rel_2vox_diff_in_tissue": round(float(d2 / ak[tis].mean().clamp_min(1e-6)), 3),
                       "layer_vs_image_tensor_in_mask_deg": round(float(ang_img[mk].median()), 1), "layer_vs_mode_in_mask_deg": round(float(ang_mode[mk].median()), 1),
                       "mode_zyx": [round(float(c), 3) for c in modes[k]]})
    res = {"calib": a.calib, "block": a.block, "rho": a.rho, "smooth": a.smooth, "transform": a.transform, "act_transform": a.act_transform, "tau": tau, "n_candidates": int(nmod), "N": N, "residual_median": round(float(R[tis].median()), 3),
           "union_share_of_tissue": round(float((union & tis).float().sum() / tis.float().sum()), 3), "window_N_shares": [round(float((Nmap == k).mean()), 3) for k in range(5)],
           "layers": [{k2: v2 for k2, v2 in L.items() if k2 not in ("dirs", "gap")} for L in layers]}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(f"{DATA_ROOT}/results/activation_real_S2146_nmax{a.nmax}_s{a.smooth}{sfx}.json", "w"), indent=1)
    export_population_field(Path(DATA_ROOT) / "fields" / f"S2146_population_field_block{a.block}_nmax{a.nmax}_s{a.smooth}{sfx}.zarr", modes, A, torch.stack([L["dirs"] for L in layers]),
                            torch.stack([L["gap"] for L in layers]), masks, union, strongest, VOX, {"source": "S-21-46 crop centre", "tau": tau, "smooth": a.smooth, "transform": a.transform, "act_transform": a.act_transform, "residual_median": res["residual_median"]})
    # ---------------------------------------------------------------- figure 1: the walkthrough
    chan = [int(S.argmax()) for S in S_list[:2]]
    Dang = dic.D; mu = Dang.mean(0); _, _, Vt = torch.linalg.svd(Dang - mu, full_matrices=False); E = Vt[:2].T
    grid_xy = ((Dang - mu) @ E).cpu().numpy()
    Xrow = dic.standardise(f[(slice(None),) + row].reshape(f.shape[0], -1).T)
    vox_xy = ((Xrow - mu) @ E).cpu().numpy()
    tisr = tis[row].cpu().numpy().ravel()
    # colour the voxels by the teacher's in-plane angle (the only orientation reference on real data)
    dr = d_b[(slice(None),) + row].reshape(3, -1).T; ang_t = (torch.rad2deg(torch.atan2(dr[:, 0], dr[:, 2])) % 180).cpu().numpy()
    conf_r = (gap_b[row].reshape(-1) > q_of(gap_b[row][tis[row]], 0.5)).cpu().numpy() & tisr
    s_xy = [((S - mu) @ E).cpu().numpy() for S in S_list]
    # the example voxel: in mask 1 with a strong activation
    cand = torch.nonzero(masks[0][row] & (As[0][row] > q_of(As[0][row][masks[0][row]], 0.9))) if masks[0][row].any() else torch.nonzero(tis[row])
    iz, ix = [int(c) for c in cand[len(cand) // 2]]
    x_vox = Xrow[iz * (Z - 2 * m) + ix]
    Sm = torch.stack(S_list + [dic.background], 1); coef = nnls_batched(Sm, x_vox[None])[0]; resid = x_vox - Sm @ coef
    rgb = f[:3][(slice(None),) + row].cpu().numpy(); lo_, hi_ = np.percentile(rgb, 1, axis=(1, 2)), np.percentile(rgb, 99, axis=(1, 2))
    rgb = np.clip((rgb - lo_[:, None, None]) / (hi_ - lo_)[:, None, None], 0, 1).transpose(1, 2, 0)
    fig, axs = plt.subplots(3, 4, figsize=(24, 17), dpi=100)
    axs[0, 0].imshow(img, cmap="gray", origin="lower"); axs[0, 0].set_title(f"1a. S-21-46 crop, {a.block}³ block, axial mid plane (no ground truth)")
    for ax, c, name in ((axs[0, 1], chan[0], "channel most excited by population 1's signature"), (axs[0, 2], chan[1] if len(chan) > 1 else chan[0], "channel most excited by population 2's signature")):
        ax.imshow(f[c][row].cpu().numpy(), cmap="magma", origin="lower"); ax.set_title(f"1b–c. shape space — channel {c}: {name}", fontsize=10)
    axs[0, 3].imshow(rgb, origin="lower", interpolation="nearest"); axs[0, 3].set_title("1d. shape space as RGB (first three encoder-2 channels)")
    ax = axs[1, 0]
    for k, S in enumerate(S_list):
        ax.plot(S.cpu().numpy(), "o-", color=CAT[k], ms=3, lw=1, label=f"signature of population {k + 1}")
    ax.plot(dic.background.cpu().numpy(), "o-", color="0.4", ms=3, lw=1, label="background signature"); ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("encoder-2 channel"); ax.legend(fontsize=9); ax.set_title("2a. the dictionary, self-calibrated on this crop (teacher: excised tensor at confident sites)", fontsize=10)
    ax = axs[1, 1]
    sc = ax.scatter(vox_xy[conf_r, 0], vox_xy[conf_r, 1], s=2, c=ang_t[conf_r], cmap="hsv", vmin=0, vmax=180, alpha=0.5)
    ax.scatter(grid_xy[:, 0], grid_xy[:, 1], s=6, color="k", alpha=0.5, label="the dictionary's directions")
    for k, xy in enumerate(s_xy):
        ax.annotate("", xy=xy, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=CAT[k], lw=2.5))
    fig.colorbar(sc, ax=ax, fraction=0.04, label="teacher in-plane angle (deg)")
    ax.set_title("2b. the orientation plane (two principal directions of the dictionary): confident tissue voxels\ncoloured by the excised tensor's in-plane angle; black = the dictionary's directions; arrows = the populations found", fontsize=9)
    ax = axs[1, 2]
    xy = lambert(ro.grid).cpu().numpy(); dn = (dens / dens.max()).cpu().numpy()
    ax.scatter(xy[:, 0], xy[:, 1], c=dn, s=18, cmap="Blues", vmin=0, vmax=1)
    for k, mm in enumerate(modes):
        mxy = lambert(mm[None] if mm[0] >= 0 else -mm[None]).cpu().numpy()[0]; ax.plot(mxy[0], mxy[1], "o", mfc="none", mec=CAT[k], ms=14, mew=2.5)
    ax.set_aspect("equal"); ax.set_title(f"2c. the clustering: the block's summed per-voxel orientation distribution on the hemisphere\n(Lambert view, z = centre); circles = the {N} populations kept of {nmod} candidates", fontsize=9)
    ax = axs[1, 3]
    im = ax.imshow(Nmap, cmap=ListedColormap(["#ffffff", "#cde2fb", "#6da7ec", "#184f95", "#0d366b"]), vmin=-0.5, vmax=4.5, origin="lower", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.04, ticks=range(5)); ax.set_title(f"2d. N per window ({2 * half * VOX:.2f} mm): shares {res['window_N_shares']}", fontsize=9)
    ax = axs[2, 0]; C = len(x_vox); idx = np.arange(C); wdt = 0.8 / (N + 1)
    ax.bar(idx - 0.4, x_vox.cpu().numpy(), width=wdt, color="k", label="voxel's standardised pattern")
    for k in range(N):
        ax.bar(idx - 0.4 + (k + 1) * wdt, (coef[k] * S_list[k]).cpu().numpy(), width=wdt, color=CAT[k], label=f"a{k + 1}·S{k + 1} (a{k + 1} = {float(coef[k]):.2f})")
    ax.plot(idx, (coef[N] * dic.background).cpu().numpy(), "o", color="0.4", ms=3, label=f"a_bg·S_bg ({float(coef[N]):.2f})"); ax.plot(idx, resid.cpu().numpy(), "x", color=CAT[2], ms=4, label="residual")
    ax.legend(fontsize=8); ax.set_xlabel("encoder-2 channel"); ax.set_title(f"3a. the unmixing at one voxel (z {iz}, x {ix})", fontsize=10)
    for k, ax in enumerate(axs[2, 1:3]):
        if k < N:
            ax.imshow(A[k][row].cpu().numpy(), cmap="Blues", vmin=0, vmax=float(q_of(A[k][tis], 0.99)), origin="lower", interpolation="nearest"); ax.plot(ix, iz, "o", mfc="none", mec=CAT[2], ms=10, mew=2)
            ax.set_title(f"3b/c. activation a{k + 1}(v): population {k + 1}'s own image from the shape space", fontsize=10)
        else:
            ax.axis("off")
    ax = axs[2, 3]; ax.imshow(img, cmap="gray", origin="lower", alpha=0.45)
    sr = strongest[row].cpu().numpy()
    ax.imshow(np.ma.masked_where(sr == 0, sr), cmap=ListedColormap(CAT[:max(N, 1)]), vmin=1, vmax=max(N, 1), origin="lower", alpha=0.75, interpolation="nearest")
    ax.set_title(f"3d. strongest population per voxel (colour) and the union of the masks ({res['union_share_of_tissue']:.0%} of bright tissue)", fontsize=10)
    for ax in list(axs[0]) + list(axs[2, 1:]):
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"The activation walkthrough on real HiP-CT data (S-21-46, populations unknown, everything self-calibrated): {N} populations found; layer vs image tensor inside each mask: "
                 + ", ".join(f"{L['layer_vs_image_tensor_in_mask_deg']}°" for L in res["layers"]), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIGS / f"activation_real_S2146_nmax{a.nmax}_s{a.smooth}{sfx}.png"); plt.close(fig)
    # ---------------------------------------------------------------- figure 2: the population direction fields in their masks
    fig, axs = plt.subplots(1, N + 1, figsize=(6.5 * (N + 1), 6.5), dpi=100)
    axs = np.atleast_1d(axs)
    nz, nx = img.shape; Zc, Xc = np.meshgrid(np.arange(nz) * VOX, np.arange(nx) * VOX, indexing="ij")
    axs[0].imshow(img, cmap="gray", origin="lower", extent=(0, nx * VOX, 0, nz * VOX), alpha=0.6)
    dd = d_img[(slice(None),) + row].cpu().numpy(); sel = np.zeros((nz, nx), bool); sel[::6, ::6] = True; sel &= tis[row].cpu().numpy()
    sticks(axs[0], Zc[sel], Xc[sel], dd[:, sel].T, 6 * VOX * 0.9, "k"); axs[0].set_title("the image's own structure tensor (one direction everywhere)")
    for k in range(N):
        ax = axs[k + 1]; ax.imshow(img, cmap="gray", origin="lower", extent=(0, nx * VOX, 0, nz * VOX), alpha=0.6)
        dl = layers[k]["dirs"][(slice(None),) + row].cpu().numpy(); mk = (masks[k] & tis)[row].cpu().numpy()
        selk = sel & mk; sticks(ax, Zc[selk], Xc[selk], dl[:, selk].T, 6 * VOX * 0.9, CAT[k])
        ax.set_title(f"population {k + 1}: its layer's direction field inside its mask ({layers[k]['mask_share']:.0%} of tissue)\nvs the image tensor there: {layers[k]['layer_vs_image_tensor_in_mask_deg']}° median", fontsize=10)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Merged population vector field on the real crop: one direction field per population, placed by its mask", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / f"activation_real_S2146_field_nmax{a.nmax}_s{a.smooth}{sfx}.png"); plt.close(fig)
    # ---------------------------------------------------------------- figure 3: the real-data equivalent of the pipeline illustration
    Ab = unmix_background(F, dic, modes)                                                  # the background's own activation volume
    fig, axs = plt.subplots(2, 4, figsize=(26, 12.5), dpi=100)
    ax = axs[0, 0]; ax.imshow(img, cmap="gray", origin="lower"); ax.set_title("1. brain image (HiP-CT, 20 µm)", fontsize=12)
    ax = axs[0, 1]
    assign = strongest[row].cpu().numpy().ravel()
    bgm = ((Ab[row] > As[:, row[0], row[1], row[2]].max(0).values) & tis[row]).cpu().numpy().ravel() if False else ((strongest[row] == 0) & tis[row]).cpu().numpy().ravel()
    ax.scatter(vox_xy[~tisr, 0], vox_xy[~tisr, 1], s=1, color="0.85", alpha=0.3)
    ax.scatter(vox_xy[bgm, 0], vox_xy[bgm, 1], s=2, color="#e8975a", alpha=0.35, label="background / boundary (discard)")
    for k in range(N):
        mk = (assign == k + 1) & tisr; ax.scatter(vox_xy[mk, 0], vox_xy[mk, 1], s=2, color=CAT[k], alpha=0.35, label=f"fibre population {k + 1}")
        pts = vox_xy[mk]
        if len(pts) > 50:
            from scipy.spatial import ConvexHull
            sub = pts[np.argsort(np.linalg.norm(pts - pts.mean(0), axis=1))[:int(0.8 * len(pts))]]
            hull = ConvexHull(sub); hp = sub[hull.vertices]; ax.plot(np.append(hp[:, 0], hp[0, 0]), np.append(hp[:, 1], hp[0, 1]), "--", color=CAT[k], lw=2)
    ax.set_title("2. shape space (the orientation plane of the encoder-2 features)\nvoxels coloured by the population they are assigned to", fontsize=11); ax.legend(fontsize=9, markerscale=5, loc="lower left"); ax.set_xticks([]); ax.set_yticks([])
    for k in range(2):
        ax = axs[0, 2 + k]
        if k < N:
            ax.imshow(A[k][row].cpu().numpy(), cmap=["Blues", "Greens", "Oranges", "Purples"][k], vmin=0, vmax=float(q_of(A[k][tis], 0.99)), origin="lower", interpolation="nearest")
            ax.set_title(f"3. feature activation volume — fibre population {k + 1}\n(its signature's weight at every voxel; a full volume, no mask)", fontsize=11)
        else:
            ax.axis("off")
    ax = axs[1, 0]; ax.imshow(Ab[row].cpu().numpy(), cmap="Oranges", vmin=0, vmax=float(q_of(Ab[tis], 0.99)), origin="lower", interpolation="nearest")
    ax.set_title("3′. background activation volume (grey–white boundary, no fibre) — discarded", fontsize=11)
    nz, nx = img.shape; ext = (0, nx * VOX, 0, nz * VOX); Zc, Xc = np.meshgrid(np.arange(nz) * VOX, np.arange(nx) * VOX, indexing="ij")
    sel = np.zeros((nz, nx), bool); sel[::5, ::5] = True; sel &= tis[row].cpu().numpy()
    for k in range(2):
        ax = axs[1, 1 + k]; ax.imshow(img, cmap="gray", origin="lower", extent=ext, alpha=0.55)
        if k < N:
            dl = layers[k]["dirs"][(slice(None),) + row].cpu().numpy(); mk = (masks[k] & tis)[row].cpu().numpy(); selk = sel & mk
            sticks(ax, Zc[selk], Xc[selk], dl[:, selk].T, 5 * VOX * 0.9, CAT[k], lw=1.4)
            ax.set_title(f"4. vec{k + 1}: population {k + 1}'s vector field (its layer's own structure tensor, ρ {a.rho:g})\nplaced where its activation is above the null ({layers[k]['mask_share']:.0%} of tissue)", fontsize=11)
        else:
            ax.set_title("4. (one population only)", fontsize=11)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax = axs[1, 3]; ax.imshow(img, cmap="gray", origin="lower", extent=ext, alpha=0.55)
    for k in range(N):
        dl = layers[k]["dirs"][(slice(None),) + row].cpu().numpy(); mk = (masks[k] & tis)[row].cpu().numpy(); selk = sel & mk
        sticks(ax, Zc[selk], Xc[selk], dl[:, selk].T, 5 * VOX * 0.9, CAT[k], lw=1.4)
    both = (masks[0] & masks[1] & tis)[row].cpu().numpy() if N > 1 else np.zeros((nz, nx), bool)
    ax.set_title(f"5. combined multi-vector field: every population's vectors together\n({both.mean():.0%} of the plane holds two vectors)", fontsize=11)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    for ax in [axs[0, 0], axs[0, 2], axs[0, 3], axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3]]:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The pipeline on real HiP-CT data (S-21-46 crop, populations unknown): brain image → shape space → per-population feature activations (+ background, discarded) → per-population vector fields → combined multi-vector field", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / f"activation_real_S2146_illustration_nmax{a.nmax}_s{a.smooth}{sfx}.png"); plt.close(fig)
    print(f"REAL WALKTHROUGH DONE {time.time() - t0:.0f} s", flush=True)


def unmix_background(F, dic, modes, z_chunk=16):
    """The background signature's own activation volume (the last coefficient of the same unmixing)."""
    f = dic._prep(F); C, Z, Y, X = f.shape
    S = torch.stack([dic.signature(m) for m in modes] + [dic.background], 1)
    out = torch.empty((Z, Y, X), device=F.device)
    for z in range(0, Z, z_chunk):
        fz = f[:, z:z + z_chunk]; Xn = dic.standardise(fz.reshape(C, -1).T)
        out[z:z + z_chunk] = nnls_batched(S, Xn)[:, -1].reshape(fz.shape[1:])
    return out


if __name__ == "__main__":
    main()
