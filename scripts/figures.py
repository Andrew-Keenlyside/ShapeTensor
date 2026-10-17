"""The documentation figures on real data: the HiP-CT block (the standard row, the streamlines, the pipeline
illustration's partner) and the BigBrain volume of interest (the standard row). The phantom figure is the
validation's (``shapetensor validate``).

    python scripts/figures.py all            -> docs/img/{real_S2146, real_S2146_tracks, overview_S2146, bigbrain}.png
    python scripts/figures.py real           -> one of them (overview: the illustration's partner on the HiP-CT block)

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
from shapetensor.pipeline import apply, calibrate, teacher
from shapetensor.tracking import track
from shapetensor.validate import angle_deg
from shapetensor.viz import CMAPS, POP, dec_activation_overlay, dec_rgb, pipeline_row, scatter_space, shape_space, sticks, streamline_segments

HIPCT = "/hdd/andrew/BRIDGE/atlas20/crops/S-21-46.ome.zarr"
BIGBRAIN = Path.home() / "bridge_wmcov/shapetensor/bigbrain/bigbrain_20um_x2336_y3936_z2464_768.npy"
OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
RHO, SIGMA, BRIGHT = 10.0, 1.0, 142.5
dev = "cuda"


def _panel(ax, img, title):
    ax.imshow(img, cmap="gray", origin="lower"); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def _dec(ax, d, title, w=None):
    ax.imshow(dec_rgb(d, w), origin="lower", interpolation="nearest"); ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def _hipct_blocks():
    import zarr
    z0 = zarr.open(HIPCT, mode="r")["0"]; n = z0.shape[0]; lo = (n - 384) // 2
    vc = torch.as_tensor(np.asarray(z0[lo:lo + 384, lo:lo + 384, lo:lo + 384]).astype(np.float32), device=dev)
    o = 96; v = vc[o:o + 192, o:o + 192, o:o + 192].contiguous()
    return vc, v


def fig_real():
    vc, v = _hipct_blocks(); tc, t = vc >= BRIGHT, v >= BRIGHT
    t0 = time.time(); cal = calibrate(vc, tc, sigma=SIGMA, rho=RHO); del vc
    F = cal.features(v); out = apply(cal, v, t, features=F); d_st = st.structure_tensor(v, sigma=SIGMA, rho=RHO, mask=t, order="zyx")
    S = st.excised_structure_tensor_3d(v, SIGMA, RHO, mask=t); val, _ = st.eig_special_3d(S); gap_st = ((val[1] - val[2]) / val[1].clamp_min(1e-12)).clamp(0, 1); del S
    m = 24; sl = (slice(m, 192 - m), 96, slice(m, 192 - m)); dsl = (slice(None),) + sl; sel = torch.zeros_like(t); sel[m:-m, m:-m, m:-m] = t[m:-m, m:-m, m:-m]
    e = angle_deg(d_st, out["vec"]); conf = sel & (gap_st > gap_st[sel].quantile(0.75))
    N = out["modes"].shape[0]; im = v[sl].cpu().numpy()
    stats = {"calibration_s": round(time.time() - t0, 1), "n_populations": N, "tau": round(out["tau"], 3), "readout_bins": cal.orientation_bins_occupied,
             "agreement_median_deg": round(float(e[sel].median()), 1), "agreement_where_tensor_confident_deg": round(float(e[conf].median()), 1),
             "agreement_within_15_where_confident": round(float((e[conf] <= 15).float().mean()), 3), "structure_tensor_gap_median": round(float(gap_st[sel].median()), 3)}
    print(json.dumps(stats)); json.dump(stats, open(OUT / "real_S2146.json", "w"), indent=1)
    space = shape_space(F, cal.dictionary, out["activation"], out["tau"], sel)[:3]; del F
    _standard_row(OUT / "real_S2146.png", "S-21-46, 20 µm HiP-CT, 192³ block", v, d_st, out, space, sl,
                  f"The real block: {N} populations found; the single field and the excised structure tensor agree to {stats['agreement_where_tensor_confident_deg']}° where the tensor is confident "
                  f"(median gap {stats['structure_tensor_gap_median']}), {stats['agreement_median_deg']}° over the block")
    print("wrote real_S2146.png")
    return v, t, d_st, out


def _standard_row(path, label, v, d_st, out, space, sl, title):
    """One block in the standard layout: image | excised structure tensor | shape space | activation layers | shape
    tensor, single | multi-vector field, first vector."""
    N = min(out["modes"].shape[0], 3); dsl = (slice(None),) + sl; zs, _, xs = sl
    A, W = out["activation"], out["weight"]
    aref = float(A[:, zs, :, xs].float().reshape(-1)[::53].quantile(0.99).clamp_min(1e-6)); wref = float(W[zs, :, xs].float().quantile(0.99).clamp_min(1e-6))
    fig, axs = plt.subplots(1, 5 + N, figsize=(4.2 * (5 + N), 4.9), dpi=100)
    pipeline_row(axs, v[sl].cpu().numpy(), d_st[dsl].cpu().numpy(), space, [A[k][sl].cpu().numpy() for k in range(N)],
                 out["vec"][dsl].cpu().numpy(), (W[sl] / wref).clamp(0, 1).cpu().numpy(), out["vecs"][0][dsl].cpu().numpy(), (out["weights"][0][sl] / wref).clamp(0, 1).cpu().numpy(),
                 aref=aref, label=label)
    fig.suptitle(title + " (DEC: R = |x|, G = |y|, B = |z|; the shape tensor's brightness = activation)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(path); plt.close(fig)


def fig_tracks(v=None, t=None, d_st=None, out=None):
    if out is None:
        v, t, d_st, out = fig_real()
    b = 4; pool = lambda x: Fn.avg_pool3d(x[None], b)[0]                                       # vectors to a site grid (axial: average the dyads' principal axis ≈ the centre vector here)
    def sites(d):
        return d[:, b // 2::b, b // 2::b, b // 2::b].contiguous()
    mask = pool(t[None].float())[0] > 0.5
    fields = {"excised structure tensor": (sites(d_st)[None], None), "shape tensor, single": (sites(out["vec"])[None], None),
              "shape tensor, multi": (torch.stack([sites(out["vecs"][k]) for k in range(out["vecs"].shape[0])]), torch.stack([sites(out["weights"][k][None])[0] for k in range(out["weights"].shape[0])]))}
    gen = torch.Generator(device=dev).manual_seed(0)
    idx = mask.nonzero(); seeds = idx[torch.randperm(len(idx), generator=gen, device=dev)[:1500]].float()
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.6), dpi=100)
    nz, ny, nx = mask.shape; im = v[:, 96].cpu().numpy()
    for ax, (name, (D, W)) in zip(axs, fields.items()):
        r = track(D, mask, seeds, step=0.5, max_angle_deg=30.0, max_steps=300, weights=W, min_weight=out["tau"] if W is not None else 0.0)
        P = r["points"].cpu().numpy()
        ax.imshow(im, cmap="gray", origin="lower", extent=(0, nx * b, 0, nz * b), alpha=0.6)
        streamline_segments(ax, P, lambda q: abs(q[:, 1].mean() - ny / 2) < 4, b, lw=0.8)
        L = r["length"].float(); ax.set_title(f"{name}: streamlines (median length {float(L.median()) * b:.0f} voxels)", fontsize=10); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, nx * b); ax.set_ylim(0, nz * b)
    fig.suptitle("Deterministic streamlines from the same seeds on each field (a slab around the mid plane; colour = segment direction; the multi-vector field followed only where a population's activation exceeds the null)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT / "real_S2146_tracks.png"); plt.close(fig); print("wrote real_S2146_tracks.png")


def fig_overview_real():
    """The pipeline illustration's partner on the HiP-CT block, the same six columns left to right: the image, the
    shape space, the populations in it, their activation layers, their fields, and the combined fields."""
    from matplotlib.patches import FancyArrowPatch
    vc, v = _hipct_blocks(); tc, t = vc >= BRIGHT, v >= BRIGHT
    cal = calibrate(vc, tc, sigma=SIGMA, rho=RHO); del vc
    F = cal.features(v); out = apply(cal, v, t, features=F); N = min(out["modes"].shape[0], 2)
    m = 24; sl = (slice(m, 192 - m), 96, slice(m, 192 - m)); im = v[sl].cpu().numpy()
    # the shape space: the standardised (3³-averaged) features of 12 000 tissue voxels on their first two principal axes
    reg = torch.zeros_like(t); reg[m:-m, m:-m, m:-m] = t[m:-m, m:-m, m:-m]
    P, c_sep, above, pick = shape_space(F, cal.dictionary, out["activation"], out["tau"], reg)
    d_t, gap_t = teacher(v, SIGMA, RHO); gp = gap_t.reshape(-1)[pick]; dv = d_t.reshape(3, -1)[:, pick]
    hi, lo = gap_t[reg].quantile(0.7), gap_t[reg].quantile(0.3)
    c_ss = np.full((len(pick), 3), 0.82); conf = (gp > hi).cpu().numpy(); bg = (gp < lo).cpu().numpy()
    c_ss[conf] = dec_rgb(dv.cpu().numpy())[conf]; c_ss[bg] = (0.93, 0.58, 0.30)
    A = out["activation"]
    aref = float(A[:, m:-m, m:-m, m:-m].float().reshape(-1)[::97].quantile(0.99))
    # the figure: six columns, the first three one panel, the last three two stacked panels
    fig = plt.figure(figsize=(28, 10), dpi=100)
    gs = fig.add_gridspec(4, 6, left=0.012, right=0.988, top=0.84, bottom=0.025, wspace=0.28, hspace=0.16)
    ax_img, ax_ss, ax_sep = (fig.add_subplot(gs[1:3, c]) for c in range(3))
    stacked = [[fig.add_subplot(gs[0:2, c]), fig.add_subplot(gs[2:4, c])] for c in range(3, 6)]
    _panel(ax_img, im, "mid-plane slice of the block")
    scatter_space(ax_ss, P, c_ss, conf | bg); scatter_space(ax_sep, P, c_sep, above)
    ax_ss.set_title("teacher's orientation as DEC; orange = least oriented", fontsize=10)
    ax_sep.set_title("largest activation above the null, per population", fontsize=10)
    for k in range(2):
        axa, axd, axc = stacked[0][k], stacked[1][k], stacked[2][k]
        if k < N:
            act = out["activation"][k][sl].cpu().numpy()
            axa.imshow(act, cmap=CMAPS[k], vmin=0, vmax=aref, origin="lower", interpolation="nearest"); (axa.set_title if k == 0 else axa.set_xlabel)(f"population {k + 1}", color=POP[k], fontsize=11, loc="left")
            axd.imshow(dec_activation_overlay(out["dirs"][k][(slice(None),) + sl].cpu().numpy(), act, im, "opacity"), origin="lower", interpolation="nearest")
            (axd.set_title if k == 0 else axd.set_xlabel)(f"population {k + 1}: DEC, opacity = activation", color=POP[k], fontsize=11, loc="left")
        else:
            for ax in (axa, axd): ax.text(0.5, 0.5, "one population found", ha="center", va="center", transform=ax.transAxes); ax.set_facecolor("0.95")
        for ax in (axa, axd):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_edgecolor(POP[k]); s.set_linewidth(1.5)
    wn = (out["weight"][sl] / out["weight"][sl].float().quantile(0.99)).clamp(0, 1).cpu().numpy()
    _dec(stacked[2][0], out["vec"][(slice(None),) + sl].cpu().numpy(), "one vector per voxel: DEC, brightness = total activation", wn)
    c0, w_ = 64, 64; sl2 = (slice(c0, c0 + w_), 96, slice(c0, c0 + w_)); im2 = v[sl2].cpu().numpy(); ax = stacked[2][1]
    ax.imshow(im2, cmap="gray", origin="lower", alpha=0.75); step = 3
    cz, cx = np.meshgrid(np.arange(1, w_, step), np.arange(1, w_, step), indexing="ij"); cz, cx = cz.ravel(), cx.ravel()
    for k in range(N):
        d = out["dirs"][k][:, c0 + cz, 96, c0 + cx].T.cpu().numpy(); a = (out["activation"][k][c0 + cz, 96, c0 + cx] / (0.5 * aref)).clamp(0, 1).cpu().numpy()
        sticks(ax, cz, cx, d, step * 1.3 * a, POP[k], lw=1.8)
    ax.set_xlim(-0.5, w_ - 0.5); ax.set_ylim(-0.5, w_ - 0.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("all populations at every voxel; stick length = activation", fontsize=10)
    for ax in (ax_img, ax_ss, ax_sep): ax.set_xticks([]); ax.set_yticks([])
    heads = ("HiP-CT image", "Shape space", "Populations in the shape space", "Activation layers", "Per-population fields", "Combined fields")
    subs = ("S-21-46, 20 µm", "the voxels' shape features\n(two principal axes of 32 channels)", "found without labels, every voxel\nunmixed onto their signatures",
            "one full-volume layer per population", "the image's tensor weighted\nby each population's activation", "one vector per voxel, or\nevery population with its weight")
    tops = [ax_img, ax_ss, ax_sep] + [s[0] for s in stacked]
    for ax, h, s in zip(tops, heads, subs):
        b = ax.get_position(); fig.text((b.x0 + b.x1) / 2, 0.955, h, ha="center", va="center", fontsize=15, weight="bold"); fig.text((b.x0 + b.x1) / 2, 0.905, s, ha="center", va="center", fontsize=10.5, color="0.35")
    for a0, a1 in zip(tops[:-1], tops[1:]):
        x0, x1 = a0.get_position().x1 + 0.004, a1.get_position().x0 - 0.004
        fig.add_artist(FancyArrowPatch((x0, 0.43), (x1, 0.43), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20, lw=2.0, color="0.25"))
    fig.savefig(OUT / "overview_S2146.png"); plt.close(fig); print(f"wrote overview_S2146.png ({out['modes'].shape[0]} populations, tau {out['tau']:.2f})")


def fig_bigbrain():
    vol = np.load(BIGBRAIN, mmap_mode="r"); n = vol.shape[0]; lo = (n - 256) // 2
    v = torch.as_tensor(np.asarray(vol[lo:lo + 256, lo:lo + 256, lo:lo + 256]).astype(np.float32), device=dev)
    cal = calibrate(v, sigma=SIGMA, rho=RHO); F = cal.features(v); out = apply(cal, v, features=F); d_st = st.structure_tensor(v, sigma=SIGMA, rho=RHO, order="zyx")
    sl = (slice(32, 224), 128, slice(32, 224)); reg = torch.zeros(v.shape, dtype=torch.bool, device=dev); reg[32:224, 32:224, 32:224] = True
    space = shape_space(F, cal.dictionary, out["activation"], out["tau"], reg)[:3]; del F
    _standard_row(OUT / "bigbrain.png", "BigBrain VOI, 20 µm (cell stain; white matter dark)", v, d_st, out, space, sl,
                  f"BigBrain: {out['modes'].shape[0]} populations found; the same pipeline on a different contrast")
    print("wrote bigbrain.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["all", "real", "tracks", "overview", "bigbrain"]); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.which in ("all", "real", "tracks"):
        fig_tracks() if a.which != "real" else fig_real()
    if a.which in ("all", "overview"):
        fig_overview_real()
    if a.which in ("all", "bigbrain"):
        fig_bigbrain()
