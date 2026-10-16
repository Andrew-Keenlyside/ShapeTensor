"""Does shape space separate fibre POPULATIONS (as opposed to tissue types)? A test of the "cluster the shape space,
then one tensor / TCA per population" model on a 60-degree crossing phantom with a 40-level step through the window:
anatomix layer-58 features (8 PCs) of every voxel in the window, coloured by what the voxel truly belongs to
(bundle 1 / bundle 2 / both / boundary band / background), k-means on the same features, and the structure tensor of
each k-means cluster against the two true directions. CPU only.

    python scripts/fig_shape_space_clusters.py        (shapetensors env)
"""
import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from shapetensor.phantoms import make
from shapetensor.features.anatomix import AnatomixEngine
from shapetensor.tensor import gradients
from shapetensor.readout import multi

OUT = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
COL = {"bundle 1": "#2a78d6", "bundle 2": "#e34948", "both": "#6250d6", "boundary band": "#eb6834", "background": "#b9b8b1"}
torch.set_num_threads(16)


def main():
    tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]
    tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(128, 32, 128), margin=(32, 32, 32))
    rng = np.random.default_rng(5)
    a1, a2 = 120.0, 60.0
    # the two populations rendered separately, so every voxel's membership is known
    sp1 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp1, a1, 1.0); d1 = sp1.render()
    sp2 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp2, a2, 1.0); d2 = sp2.render()
    zc = g.coords(0); xc = g.coords(2)
    Zm, Xm = np.meshgrid(zc, xc, indexing="ij")
    z_mid = g.core_mm[0] / 2
    base2d = np.where(Zm >= z_mid, 164.0 + 40.0, 164.0).astype(np.float32)
    amp2d = np.full(Zm.shape, tex.amp, np.float32)
    img = make._finish(g, tex, base2d, amp2d, (d1 + d2), rng)
    m1, m2 = (d1 > 0.25).numpy(), (d2 > 0.25).numpy()
    vol = torch.as_tensor(img).float()
    F = AnatomixEngine(layer="l58", device="cpu", amp=False, nyquist="none", tile=128, margin=32, n_pcs=8, l2=True,
                       out_dtype=torch.float32)(vol)
    Z, Y, X = img.shape
    cz, cy, cx, r = int(np.argmin(np.abs(zc - z_mid))), Y // 2, X // 2, 20
    sl = (slice(cz - r, cz + r + 1), slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
    Fw = F[(slice(None),) + sl].numpy().reshape(F.shape[0], -1).T                 # (P, 8)
    w1, w2 = m1[sl].ravel(), m2[sl].ravel()
    zz = np.broadcast_to(np.arange(cz - r, cz + r + 1)[:, None, None], (2 * r + 1,) * 3).ravel()
    band = np.abs(zz - cz) <= 3
    lab = np.full(len(w1), "background", object)
    lab[w1 & ~w2] = "bundle 1"; lab[w2 & ~w1] = "bundle 2"; lab[w1 & w2] = "both"; lab[band] = "boundary band"
    # 2-D view of shape space: PCA of the window's feature vectors
    mu = Fw.mean(0); U, S, Vt = np.linalg.svd(Fw - mu, full_matrices=False)
    P2 = (Fw - mu) @ Vt[:2].T
    # k-means on the 8-D features
    from sklearn.cluster import KMeans
    km = KMeans(3, n_init=10, random_state=0).fit(Fw)
    cl = km.labels_
    names = ["bundle 1", "bundle 2", "both", "boundary band", "background"]
    cont = np.array([[int(((cl == k) & (lab == n)).sum()) for n in names] for k in range(3)])
    # per-cluster structure tensor vs the truths; and the joint TCA (two pieces per population) on the same window
    G = gradients(F, 1.0, 3.0, out_dtype=torch.float32)[(slice(None),) + sl].numpy().reshape(F.shape[0], 3, -1)
    tru = [make._dir(a1), make._dir(a2)]
    ang = lambda u, v: np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v))))))
    per_cluster = []
    for k in range(3):
        sel = cl == k
        T = np.einsum("cip,cjp->ij", G[:, :, sel], G[:, :, sel])
        e3 = np.linalg.eigh(T)[1][:, 0]
        per_cluster.append((int(sel.sum()), ang(e3, tru[0]), ang(e3, tru[1])))
    w = np.exp(-0.5 * ((zz - cz) ** 2 + 0.0) / 10.0 ** 2)  # (weights only along z here would bias; use isotropic below)
    zi, yi, xi = np.meshgrid(*(np.arange(-r, r + 1),) * 3, indexing="ij")
    wg = np.exp(-0.5 * (zi ** 2 + yi ** 2 + xi ** 2) / 10.0 ** 2).ravel().astype(np.float32)
    Xt = torch.as_tensor(np.ascontiguousarray((G * np.sqrt(wg)[None, None]).transpose(2, 0, 1)))[None]
    Xt = Xt / Xt.norm()
    A, B, D, err = multi.cp_als(Xt, 4, None, starts=6, iters=400, seed=1)
    fib, en, msine, best = multi.pair_geometric(D, multi.block_gram(A, B, D))
    fib = fib[0].numpy()
    joint = [min(ang(fib[q], tru[0]), ang(fib[q], tru[1])) for q in range(2)]
    print("k-means contingency (rows clusters, cols " + ", ".join(names) + "):\n", cont)
    print("per-cluster ST e3 angle to (truth 1, truth 2):", [(n, round(a, 0), round(b, 0)) for n, a, b in per_cluster])
    print("joint TCA pair fibres, error to nearest truth:", [round(x, 1) for x in joint], flush=True)

    # ------------------------------------------------------------ figure
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.edgecolor": GRID})
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.9), dpi=130, gridspec_kw={"width_ratios": [1, 1.25, 1, 1.15]})
    fig.patch.set_facecolor("white")
    # (a) image + membership
    ax = axs[0]
    ext = (-r * 0.02, r * 0.02, -r * 0.02, r * 0.02)
    ax.imshow(img[sl][:, r, :], cmap="gray", origin="lower", extent=ext)
    over = np.zeros((2 * r + 1, 2 * r + 1, 4))
    L = lab.reshape((2 * r + 1,) * 3)[:, r, :]
    for n, c in COL.items():
        if n == "background":
            continue
        rgb = np.array([int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        over[L == n, :3] = rgb; over[L == n, 3] = 0.45
    ax.imshow(over, origin="lower", extent=ext, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("a  window on a 60° crossing with a 40-level step\n(true membership overlaid)", fontsize=9, loc="left")
    for i, (n, c) in enumerate(COL.items()):
        ax.text(0.02, 0.97 - 0.07 * i, n, transform=ax.transAxes, color=c, fontsize=8, va="top", fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    # (b) shape-space scatter
    ax = axs[1]
    rng2 = np.random.default_rng(0)
    for n in ("background", "bundle 1", "bundle 2", "both", "boundary band"):
        sel = np.flatnonzero(lab == n)
        sel = rng2.choice(sel, min(len(sel), 2500), replace=False)
        ax.scatter(P2[sel, 0], P2[sel, 1], s=3, c=COL[n], alpha=0.5 if n == "background" else 0.7, lw=0, label=n)
    ax.set_xlabel("shape-space PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("b  the same voxels in shape space (anatomix layer 58, 8 PCs → 2)\nbundle 1, bundle 2 and the boundary band all overlap", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, markerscale=3, frameon=False, loc="lower right")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # (c) k-means clusters as a map + contingency
    ax = axs[2]
    ax.imshow(cl.reshape((2 * r + 1,) * 3)[:, r, :], cmap=plt.matplotlib.colors.ListedColormap(["#cde2fb", "#fdebe3", "#e0f5ed"]), origin="lower", extent=ext, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("c  k-means (k = 3) on the 8-D features\nclusters = texture phase / background,\nnot bundles, not the step", fontsize=9, loc="left")
    txt = "voxels per cluster:\n cl  " + " ".join(f"{n[:6]:>6s}" for n in names) + "\n" + "\n".join(
        f"  {k}  " + " ".join(f"{cont[k, j]:6d}" for j in range(len(names))) for k in range(3))
    # (d) per-cluster tensor vs joint TCA
    ax = axs[3]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    rows = [("per-cluster structure tensor (your model)", None)] + [(f"cluster {k} ({n:,} voxels)", f"{a:.0f}° / {b:.0f}° from truth 1 / truth 2") for k, (n, a, b) in enumerate(per_cluster)]
    rows += [("", None), ("joint TCA, pairs of pieces (what the package does)", None),
             ("fibre 1 / fibre 2", f"{joint[0]:.0f}° / {joint[1]:.0f}° from the nearest truth")]
    y = 0.95
    for t, v in rows:
        if t and v is None:
            ax.text(0.0, y, t, fontsize=9, fontweight="bold", va="top")
        elif t:
            ax.text(0.04, y, t, fontsize=8.5, va="top"); ax.text(0.98, y, v, fontsize=8.5, va="top", ha="right")
        y -= 0.1
    ax.text(0.0, 0.30, txt, fontsize=7.2, family="monospace", va="top")
    ax.text(0.0, 0.0, "Every cluster mixes both bundles (and the step), so its tensor gives one of the\ntwo directions or the bisector. "
                      "The joint decomposition separates the bundles by\nDIRECTION (the gradient-axis mode), which the feature vectors do not carry.", fontsize=8, va="bottom", color=INK)
    ax.set_title("d  what each route recovers", fontsize=9, loc="left")
    fig.suptitle("Test of the 'cluster the shape space, then one tensor per population' model: the texture layer's shape space does not separate the two fibre "
                 "populations (nor, in this window, the step)", fontsize=10.5, x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.01, "Crossing phantom at 20 µm (calibrated texture), 41³ window centred on the step. The earlier 100 µm work found the 16-D OUTPUT layer separates tissue classes and "
             "boundaries (a region encoder), but its unmixing into population abundances found no crossings either (method 2, `IMAGE_TO_VECTORS.md`).", fontsize=7.6, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "shape_space_clusters.png", facecolor="white")
    print("wrote", OUT / "shape_space_clusters.png")


if __name__ == "__main__":
    main()
