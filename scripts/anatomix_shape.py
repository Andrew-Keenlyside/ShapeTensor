"""anatomix features on the surface phantoms and two real 100 um blocks: what does a pretrained shape encoder see at the
grey/white boundary and at tract interfaces?

Samples: phantom A (gyrus, grey/white step, fan crossing it), phantom D15 (WM tract interface, step 15, fibres crossing
at 90/60/30 deg), and the cortical and deep-WM (most internal interfaces) 100 um blocks of glyphs_real_blocks.png,
192^3 voxels each. Intensities are /255 (all uint8), the same window for every sample; each block is reflect-padded
before the network (see anatomix()).

anatomix (pretrained 3D UNet, 16 channels, L2-normalised per voxel, as chunkreg runs it) gives each voxel a feature
vector. Per sample:
  RGB          the three leading principal components of the sample's tissue feature vectors as R, G, B
  shape space  feature vectors projected on PC1-PC2 / PC2-PC3, coloured by their RGB, by region (boundary shell,
               interface, interior) and by the structure tensor's e3 (rho 0.6 mm; |x| red |y| green |z| blue)
  separability 10-NN balanced accuracy, shell/interface vs interior, from the 16 anatomix channels and, as a baseline,
               from three gradient-magnitude features (|g| at sigma 0.1, 0.3 and 0.6 mm)
Resolution: each sample is also mean-binned 2x and 4x (0.2 and 0.4 mm, same field of view) and run again; labels are
the native ones taken at the bin centres, physical scales (border, rho, baseline sigmas) are kept in mm.
<name>_resolution.png sets the three resolutions side by side.

    python anatomix_shape.py OUTDIR [BINS]    (chunkreg env: torch + anatomix; BINS default 1,2,4)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/andrew/scripts/scalable-reg")
import phantom_shell as PS
import surface_phantoms as SP
import tensor_variants as TV

OMEZ = "/home/andrew/bridge_wmcov/hundred_um/new/sub-S2146/micr/sub-S2146_sample-wholebrain_acq-100um_XPCT.ome.zarr"
THR, VOX, HALF, BORDER = 144.5, 0.1, 96, 12
BLOCKS = {"cortex": (1392, 1423, 922), "deep": (675, 577, 775)}       # x, y, z voxels; as in glyphs_real_blocks.png
STEP_GC = 8 / (3 * np.sqrt(2 * np.pi))                                # coarse |g| (sigma 3 vox) of a step of 8
LABEL_COLOURS = ("#888888", "#e41a1c", "#ff7f00", "#377eb8", "#4daf4a", "#984ea3")


def read_block(c, half=HALF):
    z0 = zarr.open(OMEZ, mode="r")["0"]
    b = np.asarray(z0[c[2] - half:c[2] + half, c[1] - half:c[1] + half, c[0] - half:c[0] + half])
    return b.transpose(2, 1, 0).astype(np.float32)                    # x, y, z


def gmag(v, s):
    return np.sqrt(sum(ndi.gaussian_filter(v, s, order=o, mode="nearest", truncate=2.0) ** 2
                       for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))))


def sample_A():
    v = np.asarray(nib.load(PS.ROOT / "A.nii.gz").dataobj).astype(np.float32)
    t = np.load(PS.ROOT / "truth.npz"); wm, gm, sdf = t["wm"], t["gm"], t["sdf"]
    lab2 = np.select([~(wm | gm), gm, wm & (sdf < 0.5), wm & (sdf < 1.5)], [0, 1, 2, 3], 4)
    names = ("CSF", "grey matter", "WM 0-0.5 mm (shell)", "WM 0.5-1.5 mm", "WM > 1.5 mm")
    return v, np.broadcast_to(lab2[:, None, :], v.shape), names, {2}, {4}


def sample_D():
    v = np.asarray(nib.load(SP.ROOT / "D15.nii.gz").dataobj).astype(np.float32)
    z = (np.arange(SP.NZ) + 0.5) * VOX; dep = np.abs(z - SP.D_PLANE_Z)
    lab1 = np.digitize(dep, [0.25, 0.5, 1.0, 2.0])
    names = ("< 0.25 mm from interface", "0.25-0.5 mm", "0.5-1 mm", "1-2 mm", "> 2 mm")
    return v, np.broadcast_to(lab1[None, None, :], v.shape), names, {0, 1}, {4}


def sample_real(c):
    v = read_block(c)
    m = ndi.gaussian_filter(v, 1.0) >= THR
    din = ndi.distance_transform_edt(m) * VOX                         # mm inside the mask to its edge
    step = gmag(v, 3.0) > STEP_GC
    lab = np.select([~m, din <= 0.5, step], [0, 1, 2], 3)
    names = ("below threshold (GM / other)", "WM <= 0.5 mm from mask edge (shell)", "WM interface (step >= 8)",
             "WM interior")
    return v, lab, names, {1, 2}, {3}


def e3_at(v, idx, rho=6.0):
    """Structure-tensor e3 (sigma 1, rho vox) at flat indices idx; |components| as RGB."""
    g = TV.grads(v)
    T = np.stack([TV.integ(g[i] * g[j], rho).ravel()[idx] for i, j in TV.PAIRS], -1)
    M = np.empty((len(idx), 3, 3), np.float32)
    for k, (i, j) in enumerate(TV.PAIRS):
        M[:, i, j] = M[:, j, i] = T[:, k]
    return np.abs(np.linalg.eigh(M)[1][:, :, 0])


def binned(v, b):
    if b == 1:
        return v
    n = [d // b for d in v.shape]
    return v[:n[0] * b, :n[1] * b, :n[2] * b].reshape(n[0], b, n[1], b, n[2], b).mean((1, 3, 5))


NET_MIN = 32         # 4 max-pools to a 2^3 bottleneck: the net's own reflect-padded convs fail on a 1^3 one (16^3 input)


def _PadTile(net):
    """Wrap the UNet so a tile smaller than NET_MIN is mirror-tiled up to it (repeated reflection: torch reflects at most
    n - 1 per call) and the centre cropped back: the network sees only the tile's own content, mirrored, so the window
    still sets the context."""
    import torch
    import torch.nn.functional as tF

    class PadTile(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.net = net

        def forward(self, x):
            s = x.shape[2:]; lo0 = [(NET_MIN - n) // 2 for n in s]
            lo, hi = list(lo0), [NET_MIN - n - a for n, a in zip(s, lo0)]
            while any(lo) or any(hi):
                n = x.shape[2:]
                a = [min(l, m - 1) for l, m in zip(lo, n)]; b = [min(h, m - 1) for h, m in zip(hi, n)]
                x = tF.pad(x, [p for i in (2, 1, 0) for p in (a[i], b[i])], mode="reflect")
                lo = [l - d for l, d in zip(lo, a)]; hi = [h - d for h, d in zip(hi, b)]
            y = self.net(x)
            return y[..., lo0[0]:lo0[0] + s[0], lo0[1]:lo0[1] + s[1], lo0[2]:lo0[2] + s[2]]
    return PadTile()


def anatomix(v, spacing, margin=16, **kw):
    """Features of v. The block is reflect-padded by margin and up to a multiple of 32 first, and the padding cropped
    off after: the extractor's own zero padding puts a hard step at the block faces, which at 0.4 mm (50-voxel
    blocks) dominated the leading components."""
    from chunkreg.features.anatomix import AnatomixFeatures
    ex = AnatomixFeatures(normalisation="l2", **kw)                    # kw: e.g. window, overlap for big blocks
    if kw.get("window") is not None and kw["window"] < NET_MIN:
        ex.setup(); ex._model = _PadTile(ex._model)                     # windows below the net's minimum input
    pad = [(margin, margin + (-(n + 2 * margin)) % 32) for n in v.shape]
    f = ex(np.pad(np.clip(v / 255.0, 0, 1), pad, mode="reflect"), spacing)
    if kw.get("window") is None and ex._tile_from is not None:          # chunkreg fell back to 0.25-overlap tiles
        raise RuntimeError("anatomix: block too big for one pass; it was tiled (seams). Use device='cpu', amp=False")
    if not isinstance(f, np.ndarray):
        f = f.float().cpu().numpy()
    return np.ascontiguousarray(f[(slice(None),) + tuple(slice(margin, margin + n) for n in v.shape)])   # (16, x, y, z)



def anatomix_clean(v, spacing, shifts=2, **kw):
    """anatomix without the two run artifacts: no sliding window (one pass per call; pass device="cpu", amp=False when
    the block is too big for the GPU: 320^3 takes ~22 s and 28 GB on the CPU), and averaged over the shifts^3
    one-voxel grid phases, each rolled back, then re-normalised. The network leaves a period-2 checkerboard (1.5-3%
    even/odd alternation along every axis) tied to its own voxel grid; the phase average cancels it. The roll wraps
    only the outermost voxels, so the caller crops a margin."""
    import itertools
    acc = None
    for s in itertools.product(range(shifts), repeat=3):
        f = np.roll(anatomix(np.roll(v, s, axis=(0, 1, 2)), spacing, **kw), [-x for x in s], axis=(1, 2, 3))
        acc = f if acc is None else acc + f
    return acc / np.maximum(np.linalg.norm(acc, axis=0, keepdims=True), 1e-6)

LAYER48 = 58         # decoder output at full resolution after the concat with the first encoder skip: 32 + 16 = 48 channels


def anatomix_layer(v, layer=LAYER48, shifts=2, margin=16, ys=None, device="cuda"):
    """PROTOTYPE. An internal full-resolution anatomix activation (default the 48-channel layer feeding the final 48 -> 16
    convolution), run cleanly as anatomix_clean: one pass (no windows), reflect-padded by margin and up to a multiple
    of 32, averaged over the shifts^3 one-voxel grid phases, L2-normalised per voxel. bf16 autocast, as chunkreg runs
    it. ys: keep only this slice of axis 1 (to save memory). Returns (C, x, y, z) float32."""
    import itertools
    import torch
    from chunkreg.features.anatomix import load_backbone
    net = load_backbone("anatomix", device)
    acc = None
    for s in itertools.product(range(shifts), repeat=3):
        vr = np.roll(v, s, axis=(0, 1, 2))
        pad = [(margin, margin + (-(n + 2 * margin)) % 32) for n in v.shape]
        x = torch.from_numpy(np.pad(np.clip(vr / 255.0, 0, 1), pad, mode="reflect").astype(np.float32))[None, None].to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = net(x, layers=[layer], encode_only=True)[0][0].float()
        f = f[(slice(None),) + tuple(slice(margin, margin + n) for n in v.shape)]
        f = torch.roll(f, [-k for k in s], dims=(1, 2, 3))
        if ys is not None:
            f = f[:, :, ys, :]
        f = f.cpu().numpy(); acc = f if acc is None else acc + f
        del x
    return acc / np.maximum(np.linalg.norm(acc, axis=0, keepdims=True), 1e-6)


def pca(F, tissue_idx, rng, n=200_000):
    X = F.reshape(F.shape[0], -1)[:, rng.choice(tissue_idx, min(n, len(tissue_idx)), replace=False)].astype(np.float64)
    mu = X.mean(1, keepdims=True); C = np.cov(X - mu)
    w, V = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, V = np.clip(w[o], 0, None), V[:, o]
    B = V[:, :3].T; B *= np.sign(B[np.arange(3), np.abs(B).argmax(1)])[:, None]
    P = B @ (X - mu); lo, hi = np.percentile(P, 1, axis=1), np.percentile(P, 99, axis=1)
    return {"B": B.astype(np.float32), "mu": mu.astype(np.float32), "lo": lo, "hi": hi,
            "explained": [float(x) for x in w[:3] / w.sum()], "explained_16": [float(x) for x in w / w.sum()]}


def project(F_cols, fit):
    return fit["B"] @ (F_cols - fit["mu"])


def to_rgb(P, fit):
    return np.clip((P - fit["lo"][:, None]) / np.maximum(fit["hi"] - fit["lo"], 1e-12)[:, None], 0, 1).T


def knn_acc(X, y, rng, per=3000):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    sel = np.concatenate([rng.choice(np.flatnonzero(y == c), min(per, (y == c).sum()), replace=False) for c in (0, 1)])
    return float(cross_val_score(KNeighborsClassifier(10), X[sel], y[sel], cv=5, scoring="balanced_accuracy").mean())


def run(name, title, v, lab, names, edge, interior, out, rng, b=1):
    """One sample at bin b (v, lab native); writes the per-sample figures, returns stats and what the overview needs."""
    res_mm = VOX * b; tag = "" if b == 1 else f"_bin{b}"
    title = f"{title}, {res_mm:g} mm voxels" + ("" if b == 1 else f" (mean-binned {b}x)")
    lab = np.asarray(lab)[b // 2::b, b // 2::b, b // 2::b]
    v = binned(v, b); lab = lab[:v.shape[0], :v.shape[1], :v.shape[2]]
    sh = v.shape; bd = max(BORDER // b, 3); rho = max(6.0 / b, 1.0)
    F = anatomix(v, res_mm); C = F.shape[0]
    core = np.zeros(sh, bool); core[bd:-bd, bd:-bd, bd:-bd] = True
    tissue = np.flatnonzero((core & (v / 255.0 > 0.05)).ravel())
    fit = pca(F, tissue, rng)
    Ff = F.reshape(C, -1)
    # stratified sample for the shape space: up to 5000 voxels per label
    labf = np.asarray(lab).ravel()
    samp = np.concatenate([rng.choice(ix, min(5000, len(ix)), replace=False)
                           for k in range(len(names)) for ix in [np.intersect1d(tissue, np.flatnonzero(labf == k))] if len(ix)])
    rng.shuffle(samp)
    Ps = project(Ff[:, samp], fit); rgb_s = to_rgb(Ps, fit); e3_s = e3_at(v, samp, rho)
    # separability: shell/interface vs interior, anatomix vs gradient-magnitude baseline
    y = np.full(len(samp), -1); y[np.isin(labf[samp], list(edge))] = 1; y[np.isin(labf[samp], list(interior))] = 0
    ok = y >= 0
    base = np.stack([gmag(v, max(s / b, 0.7)).ravel()[samp] for s in (1.0, 3.0, 6.0)], 1)
    base = (base - base.mean(0)) / base.std(0)
    sep = {"anatomix_16ch": knn_acc(Ff[:, samp].T[ok], y[ok], rng), "gradient_magnitude_3": knn_acc(base[ok], y[ok], rng),
           "edge_labels": [names[k] for k in sorted(edge)], "interior_labels": [names[k] for k in sorted(interior)]}
    # planes through the centre
    cx, cy, cz = (n // 2 for n in sh)
    planes = (("x-z plane (mid y)", np.s_[:, cy, :]), ("x-y plane (mid z)", np.s_[:, :, cz]), ("y-z plane (mid x)", np.s_[cx, :, :]))
    fig = plt.figure(figsize=(22, 17), dpi=100)
    gs = fig.add_gridspec(3, 4)
    lo, hi = np.percentile(v[core], (0.5, 99.5))
    for c, (pt, s) in enumerate(planes):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(v[s].T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        ax.set_title(f"intensity, {pt}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[1, c]); fp = F[(slice(None),) + s]
        img = to_rgb(project(fp.reshape(C, -1), fit), fit).reshape(fp.shape[1:] + (3,))
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower")
        ax.set_title(f"anatomix RGB = PC1/PC2/PC3, {pt}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    # label map and e3 map on the x-z plane
    ax = fig.add_subplot(gs[0, 3]); lm = np.asarray(lab)[:, cy, :]
    from matplotlib.colors import ListedColormap
    ax.imshow(lm.T, cmap=ListedColormap(LABEL_COLOURS[:len(names)]), vmin=-0.5, vmax=len(names) - 0.5, origin="lower",
              interpolation="nearest")
    ax.set_title("regions (x-z plane), colours as in the shape space", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, 3]); pidx = np.ravel_multi_index(np.meshgrid(np.arange(sh[0]), [cy], np.arange(sh[2]), indexing="ij"), sh).ravel()
    e3p = e3_at(v, pidx, rho).reshape(sh[0], sh[2], 3)
    ax.imshow(np.transpose(e3p, (1, 0, 2)), origin="lower")
    ax.set_title(f"structure tensor e3, rho {rho * res_mm:.2g} mm (x-z): |x| R |y| G |z| B", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    # shape space
    lc = np.array([matplotlib.colors.to_rgb(LABEL_COLOURS[k]) for k in labf[samp]])
    for c, (i, j, col, t) in enumerate(((0, 1, rgb_s, "coloured by their RGB"), (0, 1, lc, "coloured by region"),
                                        (1, 2, lc, "coloured by region"), (0, 1, e3_s, "coloured by tensor e3 |x|R |y|G |z|B"))):
        ax = fig.add_subplot(gs[2, c]); ax.scatter(Ps[i], Ps[j], c=col, s=2, alpha=0.6, lw=0)
        ax.set_xlabel(f"PC{i + 1} ({fit['explained'][i]:.0%})"); ax.set_ylabel(f"PC{j + 1} ({fit['explained'][j]:.0%})")
        ax.set_title(f"shape space PC{i + 1}-PC{j + 1}, {t}", fontsize=10); ax.grid(alpha=0.3)
        if c == 1:
            for k, nm in enumerate(names):
                ax.scatter([], [], c=LABEL_COLOURS[k], s=20, label=nm)
            ax.legend(fontsize=8, markerscale=1.5, loc="best")
    fig.suptitle(f"anatomix on {title}.  PC1-3 carry {sum(fit['explained']):.0%} of the feature variance.  "
                 f"10-NN balanced accuracy, edge vs interior: anatomix {sep['anatomix_16ch']:.2f}, "
                 f"gradient magnitude {sep['gradient_magnitude_3']:.2f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out / f"{name}{tag}_shape_rgb.png"); plt.close(fig)
    # every channel on the x-z plane
    fig, axs = plt.subplots(4, 4, figsize=(16, 16), dpi=80)
    for k, ax in enumerate(axs.ravel()):
        ch = F[k, :, cy, :]; c0, c1 = np.percentile(ch, (1, 99))
        ax.imshow(ch.T, cmap="gray", origin="lower", vmin=c0, vmax=c1); ax.set_title(f"channel {k}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"anatomix channels, {title}, x-z plane (mid y)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out / f"{name}{tag}_channels.png"); plt.close(fig)
    print(name, b, "explained", [round(x, 3) for x in fit["explained"]], "sep", sep, flush=True)
    xz = F[:, :, cy, :]
    viz = {"img": v[:, cy, :], "lohi": (lo, hi), "rgb": to_rgb(project(xz.reshape(C, -1), fit), fit).reshape(xz.shape[1:] + (3,)),
           "P": Ps, "rgb_s": rgb_s, "lc": lc, "names": names}
    return {"title": title, "voxel_mm": res_mm, "shape": list(sh), "explained_top3": fit["explained"],
            "explained_16": fit["explained_16"], "separability": sep,
            "label_counts": {names[k]: int((labf[tissue] == k).sum()) for k in range(len(names))}}, viz


def overview(name, title, runs, out):
    """Rows: resolutions; columns: intensity, anatomix RGB (x-z plane), shape space by RGB and by region."""
    fig, axs = plt.subplots(len(runs), 4, figsize=(20, 5.0 * len(runs)), dpi=100, squeeze=False)
    for r, (st, vz) in enumerate(runs):
        ax = axs[r, 0]; ax.imshow(vz["img"].T, cmap="gray", origin="lower", vmin=vz["lohi"][0], vmax=vz["lohi"][1])
        ax.set_ylabel(f"{st['voxel_mm']:g} mm voxels\n{st['shape'][0]} x {st['shape'][1]} x {st['shape'][2]}", fontsize=12)
        ax.set_title("intensity, x-z plane" if r == 0 else "", fontsize=11)
        ax = axs[r, 1]; ax.imshow(np.transpose(vz["rgb"], (1, 0, 2)), origin="lower")
        ax.set_title(f"anatomix RGB (PC1-3 carry {sum(st['explained_top3']):.0%})", fontsize=11)
        for a in axs[r, :2]:
            a.set_xticks([]); a.set_yticks([])
        sep = st["separability"]
        for c, col, t in ((2, vz["rgb_s"], "by RGB"), (3, vz["lc"], "by region")):
            ax = axs[r, c]; ax.scatter(vz["P"][0], vz["P"][1], c=col, s=2, alpha=0.6, lw=0); ax.grid(alpha=0.3)
            ax.set_xlabel(f"PC1 ({st['explained_top3'][0]:.0%})"); ax.set_ylabel(f"PC2 ({st['explained_top3'][1]:.0%})")
            ax.set_title(f"shape space {t}" + (f"; edge vs interior 10-NN: anatomix {sep['anatomix_16ch']:.2f}, "
                                                f"|g| {sep['gradient_magnitude_3']:.2f}" if c == 3 else ""), fontsize=10)
        if r == 0:
            for k, nm in enumerate(vz["names"]):
                axs[r, 3].scatter([], [], c=LABEL_COLOURS[k], s=20, label=nm)
            axs[r, 3].legend(fontsize=8, markerscale=1.5, loc="best")
    fig.suptitle(f"anatomix vs resolution: {title} (same field of view, mean-binned)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out / f"{name}_resolution.png"); plt.close(fig)


def main(out, bins=(1, 2, 4)):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    jobs = (("phantom_A", "phantom A (gyrus, grey/white step, fibres crossing it)", sample_A),
            ("phantom_D15", "phantom D15 (WM tract interface, step 15; fibres at 90/60/30 deg)", sample_D),
            ("real_cortex", f"real 100 um cortical block {BLOCKS['cortex']}", lambda: sample_real(BLOCKS["cortex"])),
            ("real_deep", f"real 100 um deep-WM block {BLOCKS['deep']} (most internal interfaces)", lambda: sample_real(BLOCKS["deep"])))
    for name, title, fn in jobs:
        sample = fn(); runs = []
        for b in bins:
            st, vz = run(name, title, *sample, out, np.random.default_rng(b), b)
            res[f"{name}@{st['voxel_mm']:g}mm"] = st; runs.append((st, vz))
        if len(runs) > 1:
            overview(name, title, runs, out)
    (out / "summary.json").write_text(json.dumps(res, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (1, 2, 4))
