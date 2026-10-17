"""The documentation figures: the activation layers on a crossing phantom, the real HiP-CT block (fields and
streamlines) and the BigBrain volume of interest.

    python scripts/figures.py all            -> docs/img/{layers_phantom, real_S2146, real_S2146_tracks, bigbrain}.png
    python scripts/figures.py real           -> one of them

Data: the S-21-46 20 µm crop (HIPCT) and the BigBrain VOI (BIGBRAIN, see scripts/bigbrain_fetch.py).
"""
import argparse, json, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor import st
from shapetensor.phantoms import Grid, TextureParams, build
from shapetensor.pipeline import apply, calibrate
from shapetensor.tracking import track
from shapetensor.validate import ANGLES, angle_deg
from shapetensor.viz import dec_activation_overlay, dec_rgb, streamline_segments

HIPCT = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
BIGBRAIN = Path.home() / "bridge_wmcov/shapetensor/bigbrain/bigbrain_20um_x2336_y3936_z2464_768.npy"
OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
RHO, SIGMA, BRIGHT = 10.0, 1.0, 142.5
dev = "cuda"


def _panel(ax, img, title):
    ax.imshow(img, cmap="gray", origin="lower"); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def _dec(ax, d, title, w=None):
    ax.imshow(dec_rgb(d, w), origin="lower", interpolation="nearest"); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def fig_layers_phantom():
    g = Grid(); tex = TextureParams(dispersion_deg=8.0, pitch_deg=4.0)
    cal = calibrate([torch.as_tensor(build("bundle", 100 + k, g, tex, angle=float(a))[0], device=dev).float() for k, a in enumerate(ANGLES)], margin=32, sigma=SIGMA, rho=RHO)
    img, tr, _ = build("crossing", 0, g, tex, cross_deg=60.0); v = torch.as_tensor(img, device=dev).float()
    out = apply(cal, v); d_st = st.structure_tensor(v, sigma=SIGMA, rho=RHO, order="zyx")
    zs, y, xs = g.inner[0], g.shape[1] // 2, g.inner[2]; im = img[zs, y, xs].astype(np.float32); N = out["modes"].shape[0]
    fig, axs = plt.subplots(1, 4 + 2 * min(N, 2), figsize=(4.4 * (4 + 2 * min(N, 2)), 4.8), dpi=100)
    _panel(axs[0], im, "60° crossing phantom (tight bundles)")
    _dec(axs[1], d_st[:, zs, y, xs].cpu().numpy(), "excised structure tensor: the bisector")
    for k in range(min(N, 2)):
        act = out["activation"][k][zs, y, xs].cpu().numpy()
        axs[2 + 2 * k].imshow(act, cmap="Blues", vmin=0, vmax=np.percentile(act, 99), origin="lower", interpolation="nearest"); axs[2 + 2 * k].set_title(f"activation layer of population {k + 1}", fontsize=10)
        axs[3 + 2 * k].imshow(dec_activation_overlay(out["dirs"][k][:, zs, y, xs].cpu().numpy(), act, im, "opacity"), origin="lower", interpolation="nearest")
        axs[3 + 2 * k].set_title(f"its tensor's direction, opacity = activation", fontsize=10)
    _dec(axs[-2], out["vec"][:, zs, y, xs].cpu().numpy(), "shape tensor: one vector per voxel")
    W = out["weights"]; wref = float(W[0][zs, :, xs].float().quantile(0.99))
    _dec(axs[-1], out["vecs"][min(1, N - 1)][:, zs, y, xs].cpu().numpy(), "second vector, brightness = its activation", (W[min(1, N - 1)][zs, y, xs] / wref).clamp(0, 1).cpu().numpy())
    for ax in axs: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Stage 2 on a crossing: each population's activation layer, its own tensor, and the merged fields (DEC: R = |x|, G = |y|, B = |z|)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "layers_phantom.png"); plt.close(fig); print("wrote layers_phantom.png")


def _hipct_blocks():
    import zarr
    z0 = zarr.open(HIPCT, mode="r")["0"]; n = z0.shape[0]; lo = (n - 384) // 2
    vc = torch.as_tensor(np.asarray(z0[lo:lo + 384, lo:lo + 384, lo:lo + 384]).astype(np.float32), device=dev)
    o = 96; v = vc[o:o + 192, o:o + 192, o:o + 192].contiguous()
    return vc, v


def fig_real():
    vc, v = _hipct_blocks(); tc, t = vc >= BRIGHT, v >= BRIGHT
    t0 = time.time(); cal = calibrate(vc, tc, sigma=SIGMA, rho=RHO); del vc
    out = apply(cal, v, t); d_st = st.structure_tensor(v, sigma=SIGMA, rho=RHO, mask=t, order="zyx")
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO, mask=t); val, _ = st.eig_special_3d(S); gap_st = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1); del S
    m = 24; sl = (slice(m, 192 - m), 96, slice(m, 192 - m)); sel = torch.zeros_like(t); sel[m:-m, m:-m, m:-m] = t[m:-m, m:-m, m:-m]
    e = angle_deg(d_st, out["vec"]); conf = sel & (gap_st > gap_st[sel].quantile(0.75))
    N = out["modes"].shape[0]; im = v[sl].cpu().numpy()
    stats = {"calibration_s": round(time.time() - t0, 1), "n_populations": N, "tau": round(out["tau"], 3), "readout_bins": cal.orientation_bins_occupied,
             "agreement_median_deg": round(float(e[sel].median()), 1), "agreement_where_tensor_confident_deg": round(float(e[conf].median()), 1),
             "agreement_within_15_where_confident": round(float((e[conf] <= 15).float().mean()), 3), "structure_tensor_gap_median": round(float(gap_st[sel].median()), 3)}
    print(json.dumps(stats)); json.dump(stats, open(OUT / "real_S2146.json", "w"), indent=1)
    cols = 4 + 2 * min(N, 2)
    fig, axs = plt.subplots(1, cols, figsize=(4.4 * cols, 4.8), dpi=100)
    _panel(axs[0], im, "S-21-46, 20 µm HiP-CT, 192³ block")
    _dec(axs[1], d_st[(slice(None),) + sl].cpu().numpy(), "excised structure tensor")
    _dec(axs[2], out["vec"][(slice(None),) + sl].cpu().numpy(), "shape tensor, one vector per voxel")
    wn = (out["weight"][sl] / out["weight"][sl].float().quantile(0.99)).clamp(0, 1).cpu().numpy()
    _dec(axs[3], out["vec"][(slice(None),) + sl].cpu().numpy(), "the same, brightness = total activation", wn)
    for k in range(min(N, 2)):
        act = out["activation"][k][sl].cpu().numpy()
        axs[4 + 2 * k].imshow(act, cmap="Blues", vmin=0, vmax=np.percentile(act, 99), origin="lower", interpolation="nearest"); axs[4 + 2 * k].set_title(f"activation layer of population {k + 1}", fontsize=10)
        axs[5 + 2 * k].imshow(dec_activation_overlay(out["dirs"][k][(slice(None),) + sl].cpu().numpy(), act, im, "opacity"), origin="lower", interpolation="nearest"); axs[5 + 2 * k].set_title("its tensor's direction, opacity = activation", fontsize=10)
    for ax in axs: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"The real block: {N} populations found; the fields agree to {stats['agreement_where_tensor_confident_deg']}° where the structure tensor is confident "
                 f"(median gap {stats['structure_tensor_gap_median']}), {stats['agreement_median_deg']}° over the block", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "real_S2146.png"); plt.close(fig); print("wrote real_S2146.png")
    return v, t, d_st, out


def fig_tracks(v=None, t=None, d_st=None, out=None):
    if out is None:
        v, t, d_st, out = fig_real()
    b = 4; pool = lambda x: Fn.avg_pool3d(x[None], b)[0]                                       # vectors to a site grid (axial: average the dyads' principal axis ≈ the centre vector here)
    def sites(d):
        return d[:, b // 2::b, b // 2::b, b // 2::b].contiguous()
    mask = pool(t[None].float())[0] > 0.5
    fields = {"excised structure tensor": sites(d_st)[None], "shape tensor, single": sites(out["vec"])[None], "shape tensor, multi": torch.stack([sites(out["vecs"][k]) for k in range(out["vecs"].shape[0])])}
    gen = torch.Generator(device=dev).manual_seed(0)
    idx = mask.nonzero(); seeds = idx[torch.randperm(len(idx), generator=gen, device=dev)[:1500]].float()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), dpi=100)
    nz, ny, nx = mask.shape; im = v[:, 96].cpu().numpy()
    for ax, (name, D) in zip(axs, fields.items()):
        r = track(D, mask, seeds, step=0.5, max_angle_deg=30.0, max_steps=300)
        P = r["points"].cpu().numpy()
        ax.imshow(im, cmap="gray", origin="lower", extent=(0, nx * b, 0, nz * b), alpha=0.6)
        streamline_segments(ax, P, lambda q: abs(q[:, 1].mean() - ny / 2) < 4, b, lw=0.8)
        L = r["length"].float(); ax.set_title(f"{name}: streamlines (median length {float(L.median()) * b:.0f} voxels)", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, nx * b); ax.set_ylim(0, nz * b)
    fig.suptitle("Deterministic streamlines from the same seeds on each field (a slab around the mid plane; colour = segment direction)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT / "real_S2146_tracks.png"); plt.close(fig); print("wrote real_S2146_tracks.png")


def fig_bigbrain():
    vol = np.load(BIGBRAIN, mmap_mode="r"); n = vol.shape[0]; lo = (n - 256) // 2
    v = torch.as_tensor(np.asarray(vol[lo:lo + 256, lo:lo + 256, lo:lo + 256]).astype(np.float32), device=dev)
    cal = calibrate(v, sigma=SIGMA, rho=RHO); out = apply(cal, v); d_st = st.structure_tensor(v, sigma=SIGMA, rho=RHO, order="zyx")
    sl = (slice(32, 224), 128, slice(32, 224)); im = v[sl].cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8), dpi=100)
    _panel(axs[0], im, "BigBrain VOI, 20 µm (cell stain; white matter dark)")
    _dec(axs[1], d_st[(slice(None),) + sl].cpu().numpy(), "excised structure tensor")
    _dec(axs[2], out["vec"][(slice(None),) + sl].cpu().numpy(), f"shape tensor ({out['modes'].shape[0]} populations), brightness = activation", (out["weight"][sl] / out["weight"][sl].float().quantile(0.99)).clamp(0, 1).cpu().numpy())
    fig.tight_layout(); fig.savefig(OUT / "bigbrain.png"); plt.close(fig); print("wrote bigbrain.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["all", "phantom", "real", "tracks", "bigbrain"]); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.which in ("all", "phantom"):
        fig_layers_phantom()
    if a.which in ("all", "real", "tracks"):
        fig_tracks() if a.which != "real" else fig_real()
    if a.which in ("all", "bigbrain"):
        fig_bigbrain()
