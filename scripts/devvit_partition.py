"""Separable populations from the full 32-D dev-vit output, per voxel, no window mean.

1. Separability with the truth (noise-free phantoms): a Fisher discriminant between band-1-only and band-2-only voxels,
   applied to the overlap -- does the overlap split into two modes? The intermixed crossing is the control.
2. Label-free partition per window: a two-way k-means split of the window's voxel cloud (n x 32), accepted when its
   between/within ratio exceeds the same split on a Gaussian cloud with the window's covariance by ``gain``;
   recursive up to ``nmax``. N per window = the clusters accepted; each cluster's share = its strength.
3. The N heatmap on the phantom slice and on a HiP-CT block (cortex + white matter), with the clusters' shares.

    python scripts/devvit_partition.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

from shapetensor.features import make_engine
from shapetensor.phantoms import Grid, TextureParams, build
from shapetensor.validate import SETTINGS

DEV = "cuda"
SRC = "/hdd/andrew/BRIDGE/20um_cortex_S2146/sub-S2146/micr/sub-S2146_sample-cortex_acq-20um_XPCT.ome.zarr"


def kmeans2(X: torch.Tensor, iters: int = 15, seed: int = 0):
    """Two-way k-means on ``X (n, d)`` (initialised on the principal axis): labels, between/within ratio."""
    Xc = X - X.mean(0)
    _, _, V = torch.pca_lowrank(Xc, q=1, center=False); proj = Xc @ V[:, 0]
    lab = (proj > proj.median()).long()
    for _ in range(iters):
        c = torch.stack([X[lab == k].mean(0) if (lab == k).any() else X.mean(0) for k in (0, 1)])
        lab = torch.cdist(X, c).argmin(1)
    c = torch.stack([X[lab == k].mean(0) if (lab == k).any() else X.mean(0) for k in (0, 1)])
    within = sum(float(((X[lab == k] - c[k]) ** 2).sum()) for k in (0, 1))
    between = sum(float((lab == k).sum()) * float(((c[k] - X.mean(0)) ** 2).sum()) for k in (0, 1))
    return lab, between / max(within, 1e-9)


def split_gain(X: torch.Tensor, seed: int = 0):
    """The data's two-way split ratio over the same ratio on a Gaussian cloud with the data's mean and covariance."""
    lab, r = kmeans2(X)
    g = torch.Generator(device=X.device).manual_seed(seed)
    Xc = X - X.mean(0); C = Xc.T @ Xc / max(X.shape[0] - 1, 1)
    L = torch.linalg.cholesky(C + 1e-6 * torch.eye(C.shape[0], device=X.device))
    G = torch.randn(X.shape, device=X.device, generator=g) @ L.T + X.mean(0)
    _, rg = kmeans2(G)
    return lab, r / max(rg, 1e-9)


def partition(X: torch.Tensor, nmax: int = 4, gain: float = 1.5, min_frac: float = 0.05):
    """Recursive splits of ``X (n, d)`` accepted above ``gain``: labels (n,) in 0..N-1 and N."""
    labels = torch.zeros(X.shape[0], dtype=torch.long, device=X.device); n = 1
    queue = [0]
    while queue and n < nmax:
        k = queue.pop(0); idx = (labels == k).nonzero().flatten()
        if idx.numel() < 2 * max(50, int(min_frac * X.shape[0])):
            continue
        lab, gn = split_gain(X[idx])
        if gn < gain or min(int((lab == 0).sum()), int((lab == 1).sum())) < max(50, int(min_frac * X.shape[0])):
            continue
        labels[idx[lab == 1]] = n; queue += [k, n]; n += 1
    return labels, n


def features(eng, v):
    return eng(v).float()


def n_heatmap(F: torch.Tensor, step: int = 8, win: int = 24, nmax: int = 4, gain: float = 1.5, ky: int | None = None):
    """Per window (``win``^3 voxels every ``step`` voxels) on the slice ``ky``: N, the clusters' shares (nmax,)."""
    K, Z, Y, X = F.shape; ky = Y // 2 if ky is None else ky
    zs = list(range(win // 2, Z - win // 2, step)); xs = list(range(win // 2, X - win // 2, step))
    N = np.zeros((len(zs), len(xs)), np.int32); shares = np.zeros((len(zs), len(xs), nmax), np.float32)
    y0, y1 = max(0, ky - win // 2), min(Y, ky + win // 2)
    for i, z in enumerate(zs):
        for j, x in enumerate(xs):
            blk = F[:, z - win // 2:z + win // 2, y0:y1, x - win // 2:x + win // 2].reshape(K, -1).T
            lab, n = partition(blk, nmax=nmax, gain=gain)
            N[i, j] = n
            cnt = torch.bincount(lab, minlength=nmax).float() / lab.numel()
            shares[i, j] = np.sort(cnt.cpu().numpy())[::-1][:nmax]
    return N, shares, zs, xs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/devvit_partition")
    ap.add_argument("--gain", type=float, default=1.5)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eng = make_engine("anatomix_devvit", device=DEV, l2=False, out_dtype=torch.float32)
    p = SETTINGS["calibrated"]; tex0 = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"], noise_sd=0.0)   # noise-free
    res = {}
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.5), facecolor="white")
    # ---- 1. separability in the overlap, noise-free band crossing (and the intermixed control)
    for row, (name, kind, kw, g) in enumerate((("band crossing 90 (noise-free)", "band_crossing", {"cross_deg": 90.0}, Grid(core=(192, 32, 192))),
                                               ("intermixed crossing 60 (noise-free)", "crossing", {"cross_deg": 60.0}, Grid()))):
        img, tr, meta = build(kind, 1, g, tex=tex0, texture=p["texture"], device=DEV, **kw); v = torch.as_tensor(img, device=DEV).float()
        F = features(eng, v); K, Z, Y, X = F.shape; ky = Y // 2
        zs_, ys_, xs_ = g.inner
        core = torch.zeros((Z, Y, X), dtype=torch.bool, device=DEV); core[zs_, ys_, xs_] = True
        if kind == "band_crossing":
            b1 = torch.as_tensor(tr["band1"], device=DEV)[:, None, :].expand(Z, Y, X); b2 = torch.as_tensor(tr["band2"], device=DEV)[:, None, :].expand(Z, Y, X)
            only1, only2, both = b1 & ~b2 & core, b2 & ~b1 & core, b1 & b2 & core
        else:                                                                                   # the intermixed crossing: the "bands" are two halves of the block (no true partition)
            only1 = core.clone(); only1[:, :, X // 2:] = False; only2 = core & ~only1; both = core.clone(); both[:Z // 2] = False; both[Z // 2 + 32:] = False
        Xf = F.reshape(K, -1).T
        g_ = torch.Generator(device=DEV).manual_seed(0)
        def sample(m, n=20000):
            idx = m.reshape(-1).nonzero().flatten(); return Xf[idx[torch.randperm(idx.numel(), device=DEV, generator=g_)[:n]]]
        A, B, O = sample(only1), sample(only2), sample(both)
        mu = torch.cat([A, B]).mean(0); Sw = ((A - A.mean(0)).T @ (A - A.mean(0)) + (B - B.mean(0)).T @ (B - B.mean(0))) / (A.shape[0] + B.shape[0])
        w = torch.linalg.solve(Sw + 1e-3 * torch.trace(Sw) / K * torch.eye(K, device=DEV), A.mean(0) - B.mean(0))
        proj = lambda Xq: (Xq - mu) @ w / (A.mean(0) - B.mean(0)).__matmul__(w)                   # 0.5 at A's mean, -0.5 at B's
        pa, pb, po = proj(A), proj(B), proj(O)
        thr = 0.0; acc = 0.5 * (float((pa > thr).float().mean()) + float((pb < thr).float().mean()))
        fisher = float((pa.mean() - pb.mean()) ** 2 / (pa.var() + pb.var()))
        # bimodality of the overlap's projection: the share of its mass away from the middle
        lab_o, gain_o = split_gain(O)
        res[name] = {"held-in accuracy single bands": round(acc, 3), "Fisher ratio": round(fisher, 2), "overlap: share of voxels > 0.25 toward band 1": round(float((po > 0.25).float().mean()), 3),
                     "overlap: share < -0.25 toward band 2": round(float((po < -0.25).float().mean()), 3), "overlap: two-way split gain over a Gaussian": round(gain_o, 2)}
        print(name, json.dumps(res[name]), flush=True)
        ax = axs[row]
        ax[0].hist(pa.cpu().numpy(), 60, color="#2a78d6", alpha=0.6, label="band 1 only"); ax[0].hist(pb.cpu().numpy(), 60, color="#e34948", alpha=0.6, label="band 2 only")
        ax[0].hist(po.cpu().numpy(), 60, color="#444444", alpha=0.6, label="overlap"); ax[0].legend(fontsize=8); ax[0].set_title(f"{name}\nFisher projection of per-voxel 32-D features", fontsize=10)
        # PCA scatter of the overlap and the single bands (2 PCs of the pooled cloud)
        P = torch.cat([A[:3000], B[:3000], O[:3000]]); Pc = P - P.mean(0); _, _, V = torch.pca_lowrank(Pc, q=2, center=False); Q = (Pc @ V[:, :2]).cpu().numpy()
        for k, (lab, col) in enumerate((("band 1", "#2a78d6"), ("band 2", "#e34948"), ("overlap", "#444444"))):
            ax[1].scatter(Q[k * 3000:(k + 1) * 3000, 0], Q[k * 3000:(k + 1) * 3000, 1], s=2, c=col, alpha=0.35, label=lab)
        ax[1].legend(fontsize=8, markerscale=5); ax[1].set_title("the voxel cloud, 2 PCs (display only)", fontsize=10)
        # 2-3. the N heatmap from the label-free partition
        N, shares, zz, xx = n_heatmap(F, gain=a.gain, ky=ky)
        ax[2].imshow(v[:, ky].cpu().numpy(), cmap="gray"); ax[2].set_title("the phantom (mid slice)", fontsize=10)
        h = ax[3].imshow(N, cmap="viridis", vmin=1, vmax=4, interpolation="nearest"); plt.colorbar(h, ax=ax[3], fraction=0.046)
        ax[3].set_title(f"N per window from the label-free partition (gain {a.gain})", fontsize=10)
        res[name]["N histogram (windows)"] = {int(k): int((N == k).sum()) for k in range(1, 5)}
        print("   N histogram:", res[name]["N histogram (windows)"], flush=True)
        for a_ in ax[1:]:
            a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout(); fig.savefig(out / "partition_phantoms.png", dpi=85); plt.close(fig)
    # ---- the HiP-CT block: cortex + white matter
    z = zarr.open_group(SRC, mode="r")["0"]; c = 384
    v = torch.as_tensor(np.asarray(z[c - 128:c + 128, c - 32:c + 32, c - 128:c + 128]).astype(np.float32), device=DEV)      # 256 x 64 x 256
    F = features(eng, v); N, shares, zz, xx = n_heatmap(F, gain=a.gain, ky=32)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6), facecolor="white")
    axs[0].imshow(v[:, 32].cpu().numpy(), cmap="gray"); axs[0].set_title("HiP-CT S-21-46 cortex VOI, 256 x 256 slice (white matter bright)", fontsize=10)
    h = axs[1].imshow(N, cmap="viridis", vmin=1, vmax=4, interpolation="nearest"); plt.colorbar(h, ax=axs[1], fraction=0.046); axs[1].set_title("N per window (label-free partition of the 32-D voxel cloud)", fontsize=10)
    h = axs[2].imshow(shares[..., 1], cmap="magma", vmin=0, vmax=0.5, interpolation="nearest"); plt.colorbar(h, ax=axs[2], fraction=0.046); axs[2].set_title("share of the second cluster (0 where N = 1)", fontsize=10)
    for a_ in axs:
        a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout(); fig.savefig(out / "partition_hipct.png", dpi=85)
    wm = (v[:, 32] >= 141).cpu().numpy(); tissue = (v[:, 32] >= 80).cpu().numpy()
    Nw = np.array([[N[i, j] for j, x in enumerate(xx)] for i, z_ in enumerate(zz)]); wmw = np.array([[wm[z_, x] for x in xx] for z_ in zz]); tw = np.array([[tissue[z_, x] for x in xx] for z_ in zz])
    res["hipct"] = {"N mean, white matter windows": round(float(Nw[wmw].mean()), 2), "N mean, darker tissue windows": round(float(Nw[tw & ~wmw].mean()), 2),
                    "N histogram WM": {int(k): int((Nw[wmw] == k).sum()) for k in range(1, 5)}, "N histogram darker": {int(k): int((Nw[tw & ~wmw] == k).sum()) for k in range(1, 5)}}
    print("hipct", json.dumps(res["hipct"]), flush=True)
    json.dump(res, open(out / "partition.json", "w"), indent=1); print("wrote", out)


if __name__ == "__main__":
    main()
