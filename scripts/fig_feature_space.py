"""The shape feature space of fibre phantoms, by orientation.

Single-population phantoms with straight fibres at 12 in-plane angles (two texture realisations each) and the
45 / 60 / 90 degree crossings; per-voxel feature vectors from four depths of the anatomix network (encoder block 1,
encoder block 2, layer 58, the output) and from the Gaussian derivative jet (the no-network control). Per layer:
a PCA embedding of fibre voxels coloured by orientation with the crossing voxels overlaid; the channel tuning
heatmap (energy vs angle, channels sorted by preferred angle); the silhouette score over the 12 orientation
classes; a linear probe (multinomial logistic regression) trained on seed A and tested on seed B, scored within
+-15 deg; and the probe's per-voxel orientation map on the 60 deg crossing, with the share of voxels within 15 deg
of either true direction. CPU (small phantoms).

    python scripts/fig_feature_space.py [--smooth 3]     -> figures/tca_field/feature_space_*.png, results/feature_space.json
"""
import argparse, json, math, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor import DATA_ROOT
from shapetensor.features.anatomix import load_net
from shapetensor.features import make_engine
from shapetensor.phantoms import make

torch.set_num_threads(16)
FIGS = Path(__file__).resolve().parents[1] / "figures" / "tca_field"
CAT = ["#6250d6", "#e34948", "#eda100", "#2a9d8f"]
ANG = np.arange(0, 180, 15.0)
LAYERS = [("encoder 1 (layer 8)", 8), ("encoder 2 (layer 15)", 15), ("layer 58", 58), ("output (16 ch)", "out")]
CROSS = [(30.0, 120.0), (30.0, 90.0), (30.0, 75.0)]        # 90, 60, 45 degree crossings


def build(angles, seed, tex, g):
    rng = np.random.default_rng(seed)
    sp = make.Splat(g, tex, rng, "cpu")
    for a, f in angles:
        make._add_population(sp, a, f)
    d = sp.render()
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng)
    return img, (d > 0.25).numpy()


def net_feats(net, img, layers, smooth):
    x = torch.as_tensor(img).float()[None, None] / 255.0
    out = {}
    with torch.no_grad():
        want = [l for _, l in layers if l != "out"]
        fs = net(x, layers=want, encode_only=True)
        for (name, l), f in zip([(n, l) for n, l in layers if l != "out"], fs):
            f = f[0].float() if f.dim() == 5 else f.float()
            out[name] = Fn.interpolate(f[None], size=img.shape, mode="trilinear", align_corners=False)[0]
        if any(l == "out" for _, l in layers):
            y = net(x)
            y = y[0] if isinstance(y, (list, tuple)) else y
            out["output (16 ch)"] = y[0].float() if y.dim() == 5 else y.float()
    if smooth > 1:
        for k in out:
            out[k] = Fn.avg_pool3d(out[k][None], smooth, stride=1, padding=smooth // 2)[0]
    return out


def jet_feats(img, smooth):
    eng = make_engine("jet")
    f = eng(torch.as_tensor(img).float()).float()
    if smooth > 1:
        f = Fn.avg_pool3d(f[None], smooth, stride=1, padding=smooth // 2)[0]
    return f


def sample(f, mask, core, n, rng):
    """``n`` fibre voxels inside the core: (n, C) vectors and their (z, y, x) indices."""
    c = (slice(core[0], f.shape[1] - core[0]), slice(core[1], f.shape[2] - core[1]), slice(core[2], f.shape[3] - core[2]))
    m = torch.as_tensor(mask)
    sel = torch.zeros_like(m); sel[c] = True
    idx = (m & sel).nonzero()
    idx = idx[torch.as_tensor(rng.choice(len(idx), size=min(n, len(idx)), replace=False))]
    return f[:, idx[:, 0], idx[:, 1], idx[:, 2]].T.numpy(), idx.numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smooth", type=int, default=3); ap.add_argument("--n", type=int, default=1200)
    a = ap.parse_args(); t0 = time.time()
    tex = json.load(open(f"{DATA_ROOT}/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
    tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
    g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32)); core = (36, 12, 36)
    net = load_net("cpu")
    names = [n for n, _ in LAYERS] + ["jet (27 ch)"]
    rng = np.random.default_rng(0)
    X = {n: {"A": [], "B": []} for n in names}; Y = {"A": [], "B": []}
    for k, ang in enumerate(ANG):
        for seed_name, seed in (("A", 100 + k), ("B", 200 + k)):
            img, mask = build([(ang, 1.0)], seed, tex, g)
            F = net_feats(net, img, LAYERS, a.smooth); F["jet (27 ch)"] = jet_feats(img, a.smooth)
            for n in names:
                v, _ = sample(F[n], mask, core, a.n, rng); X[n][seed_name].append(v)
            Y[seed_name].append(np.full(a.n, ang))
        print(f"[{time.time() - t0:4.0f}s] angle {ang:3.0f}", flush=True)
    for n in names:
        for s in ("A", "B"):
            X[n][s] = np.concatenate(X[n][s])
    Y = {s: np.concatenate(Y[s]) for s in Y}
    # crossings: features and the fibre masks per population (the render of each population alone gives its mask)
    crossings = {}
    for a1, a2 in CROSS:
        img, mask = build([(a1, 0.5), (a2, 0.5)], 300 + int(a2 - a1), tex, g)
        # population 1 is drawn first in both renders (same seed), so its render alone is its mask; population 2's
        # voxels are the combined render's fibre voxels outside it (a fresh draw of population 2 would land elsewhere)
        _, m1 = build([(a1, 0.5)], 300 + int(a2 - a1), tex, g)
        F = net_feats(net, img, LAYERS, a.smooth); F["jet (27 ch)"] = jet_feats(img, a.smooth)
        crossings[int(a2 - a1)] = {"img": img, "mask": mask, "m1": m1, "m2": mask & ~m1, "F": F, "angles": (a1, a2)}
        print(f"[{time.time() - t0:4.0f}s] crossing {a2 - a1:.0f}", flush=True)
    # ---- analyses per layer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    res = {}
    ang_err = lambda p, t: np.minimum(np.abs(p - t) % 180, 180 - np.abs(p - t) % 180)
    fig1, axs1 = plt.subplots(1, 5, figsize=(26, 5.5), dpi=100)
    fig2, axs2 = plt.subplots(1, 5, figsize=(26, 5.5), dpi=100)
    fig3, axs3 = plt.subplots(2, 5, figsize=(26, 10), dpi=100)
    hues = plt.cm.hsv(ANG / 180.0)
    for j, n in enumerate(names):
        sc = StandardScaler().fit(X[n]["A"])
        XA, XB = sc.transform(X[n]["A"]), sc.transform(X[n]["B"])
        sil = float(silhouette_score(XA[::4], Y["A"][::4]))
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(XA, Y["A"].astype(int))
        pred = clf.predict(XB).astype(float)
        acc = float((ang_err(pred, Y["B"]) <= 15).mean()); exact = float((pred == Y["B"]).mean())
        # per-class confusion in angle: mean error
        r = {"silhouette": round(sil, 3), "probe_within_15deg": round(acc, 3), "probe_exact": round(exact, 3), "channels": int(XA.shape[1])}
        # tuning heatmap: mean energy per channel per angle (seed A), normalised per channel
        E = np.stack([(X[n]["A"][Y["A"] == ang] ** 2).mean(0) for ang in ANG], 1)          # (C, n_angles)
        En = E / E.mean(1, keepdims=True).clip(1e-9); depth = (En.max(1) - En.min(1)) / (En.max(1) + En.min(1) + 1e-9)
        order = np.argsort(ANG[En.argmax(1)] + 1e-3 * np.arange(len(depth)))
        r["tuning_depth_median"] = round(float(np.median(depth)), 3); r["channels_depth_gt_0.3"] = int((depth > 0.3).sum())
        # embedding
        pca = PCA(2).fit(XA); P = pca.transform(XA)
        ax = axs1[j]
        for ang, col in zip(ANG, hues):
            m = Y["A"] == ang; ax.scatter(P[m, 0], P[m, 1], s=3, color=col, alpha=0.5, label=f"{ang:.0f}°")
        c60 = crossings[60]; Fc = c60["F"][n]
        v1, _ = sample(Fc, c60["m1"], core, 300, rng); v2, _ = sample(Fc, c60["m2"], core, 300, rng)
        Pc1, Pc2 = pca.transform(sc.transform(v1)), pca.transform(sc.transform(v2))
        ax.scatter(Pc1[:, 0], Pc1[:, 1], s=6, color="k", marker="x", alpha=0.6, label="60° crossing, pop. 1 (30°)")
        ax.scatter(Pc2[:, 0], Pc2[:, 1], s=6, color="0.4", marker="+", alpha=0.8, label="pop. 2 (90°)")
        ax.set_title(f"{n}: silhouette {sil:.2f}, probe ±15° {acc:.0%}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.legend(fontsize=7, markerscale=3, ncol=2, loc="lower left")
        ax = axs2[j]
        im = ax.imshow(En[order], aspect="auto", cmap="viridis", extent=(ANG[0] - 7.5, ANG[-1] + 7.5, len(order), 0)); ax.set_xlabel("fibre angle (deg)"); ax.set_ylabel("channel (sorted by preferred angle)")
        ax.set_title(f"{n}: tuning depth median {np.median(depth):.2f}, {int((depth > 0.3).sum())} / {len(depth)} channels > 0.3", fontsize=10); fig2.colorbar(im, ax=ax, fraction=0.04)
        # the probe on the crossings: per-voxel predicted angle on the 60 deg crossing, spatial map and histogram
        for ci, cross in enumerate((60, 90)):
            c = crossings[cross]; Fc = c["F"][n]
            zc, yc, xc = Fc.shape[1:]
            sl = (slice(None), slice(core[0], zc - core[0]), yc // 2, slice(core[2], xc - core[2]))
            V = sc.transform(Fc[sl].reshape(Fc.shape[0], -1).T.numpy())
            pm = clf.predict(V).astype(float).reshape(zc - 2 * core[0], xc - 2 * core[2])
            msk = c["mask"][core[0]:zc - core[0], yc // 2, core[2]:xc - core[2]]
            a1, a2 = c["angles"]
            near1, near2 = ang_err(pm, a1) <= 15, ang_err(pm, a2) <= 15
            r[f"cross{cross}_share_within_15_of_either"] = round(float((near1 | near2)[msk].mean()), 3)
            r[f"cross{cross}_share_pop1"] = round(float(near1[msk].mean()), 3); r[f"cross{cross}_share_pop2"] = round(float(near2[msk].mean()), 3)
            # does the probe's assignment follow the true population of each voxel (the fascicle it lies in)?
            m1 = c["m1"][core[0]:zc - core[0], yc // 2, core[2]:xc - core[2]]; m2 = c["m2"][core[0]:zc - core[0], yc // 2, core[2]:xc - core[2]]
            if m1.sum() > 50 and m2.sum() > 50:
                r[f"cross{cross}_pop1_voxels_predicted_pop1"] = round(float(near1[m1].mean()), 3); r[f"cross{cross}_pop2_voxels_predicted_pop2"] = round(float(near2[m2].mean()), 3)
            if cross == 60:
                ax = axs3[0, j]
                ax.imshow(c["img"][core[0]:zc - core[0], yc // 2, core[2]:xc - core[2]], cmap="gray", origin="lower")
                ax.imshow(np.ma.masked_where(~msk, pm), cmap="hsv", vmin=0, vmax=180, origin="lower", alpha=0.75, interpolation="nearest")
                ax.set_title(f"{n}: probe angle on the 60° crossing (30° / 90°)\nwithin 15° of either: {r['cross60_share_within_15_of_either']:.0%}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
                ax = axs3[1, j]
                ax.hist(pm[msk], bins=np.arange(-7.5, 180, 15), color=CAT[0]); ax.axvline(a1, color="k"); ax.axvline(a2, color="0.4"); ax.set_xlabel("predicted angle (deg)"); ax.set_yticks([])
                ax.set_title(f"pop.1 voxels → 30°: {r.get('cross60_pop1_voxels_predicted_pop1', float('nan')):.0%}, pop.2 voxels → 90°: {r.get('cross60_pop2_voxels_predicted_pop2', float('nan')):.0%}", fontsize=10)
        res[n] = r
        print(f"[{time.time() - t0:4.0f}s] {n}: {r}", flush=True)
    for fig, name, title in ((fig1, "feature_space_embedding.png", "The shape feature space of single-population phantoms (PCA of standardised features, fibre voxels, 12 in-plane angles) with the 60° crossing's voxels overlaid"),
                             (fig2, "feature_space_tuning.png", "Channel tuning by orientation (mean energy per channel per angle, normalised per channel)"),
                             (fig3, "feature_space_crossing.png", "A linear probe (12 angle classes, trained on seed A) applied per voxel to the 60° crossing")):
        fig.suptitle(title + f" — features averaged over {a.smooth}³ voxels", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIGS / name); plt.close(fig); print("wrote", FIGS / name, flush=True)
    json.dump({"smooth": a.smooth, "n_per_phantom": a.n, "angles": ANG.tolist(), "layers": res}, open(f"{DATA_ROOT}/results/feature_space.json", "w"), indent=1)
    print(f"FEATURE SPACE DONE {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
