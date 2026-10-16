"""PROTOTYPE: unpaired ways to divide anatomix's 16-D shape space (the full model's output) into boundary and fibre
portions, compared on D0 (no step) and D15 (tract interface, step 15). Each method sees ONE image: no twin, no labels,
no k-means; every split is 2-way. Phantoms grown by 9.6 mm per x and z face (whole-block features, 8 grid phases),
cropped to the original 20 x 20 mm for statistics, display and scoring.

  M1 top PCs          boundary = the image's own top 4 principal components (naive baseline)
  M2 scale split      generalised eigenvectors of C_coarse vs C_fine of the feature VALUES (coarse = Gaussian
                      sigma 1 mm); boundary = lambda > 1 (mm-scale dominated)
  M3 gradient         generalised eigenvectors of the gradient energy of the 1 mm-smoothed features vs of the raw
     coherence        features: a step's gradient keeps its sign over mm and survives smoothing, fibre-stripe gradients
                      flip and cancel; boundary = the directions above the largest gap in log lambda
  M4 edge voxels      voxels whose coarse shape-gradient magnitude is above an Otsu threshold = boundary voxels;
                      boundary directions = generalised eigenvectors of the feature covariance in those voxels vs the
                      rest, lambda > 1
  M5 spectral         2-way spectral partition of the voxels (kNN cosine graph, QR assignment, not k-means); boundary
     bipartition      direction = the Fisher discriminant between the two groups
  "Mostly fibre" (the bulk of the distribution is fibre; boundary = what falls outside it):
  M6 robust core      robust covariance of the central 75% of voxels (MCD, top-10 PCs); outliers beyond chi^2 97.5% =
                      boundary voxels; boundary directions = generalised eigenvectors of the outliers' scatter about the
                      core mean vs the core covariance, lambda > 1
  M7 natural          density-based clustering (HDBSCAN: as many groups as the data has, plus noise; no k) on the
     clusters         top-10 PCs; the largest group = fibre, every other group and the noise = non-fibre; boundary
                      directions as M6
Boundary and fibre components are the (oblique) projections f = V_b c_b + V_f c_f; for M1 orthogonal.

Metrics: downstream = e3 of the fibre-component shape tensor (rho 16 vox) against the truth; step capture (evaluation
only, uses the twin) = share of D15 - D0 within 3 mm of the interface in D15's boundary part; fibre texture kept = share
of D15's fine-scale texture (f - G_1mm f, > 5 mm from the interface) in its fibre part; D0 boundary share = share of
D0's variance its own split calls boundary (D0 has no step).

    python shape16_split.py        (shapetensors env, one GPU)
"""
import json, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy import linalg
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import edge_aware_phantom_fig as EPF
import shape48_orient as O48
import shape_tca as T
import shape_tensor as ST
import surface_phantoms as SP
import tensor_variants as TV

T.PAD = P = SP.D_PAD
NAMES = ("D0", "D15")
NX, NZ, JY = SP.NX, SP.NZ, ST.JY
SIG = 10.0                                            # 1 mm
YH = 10                                               # statistics over y = JY +- YH
ZC = (np.arange(NZ) + 0.5) * SP.VOX - SP.D_PLANE_Z
METHODS = ("M1 top PCs", "M2 scale split", "M3 gradient coherence", "M4 edge voxels", "M5 spectral bipartition",
           "M6 robust core", "M7 natural clusters")


def region(a, yh=YH):
    """(C, x, y, z) on the grown grid -> (C, N) over the original x-z region, y = JY +- yh (z fastest)."""
    return a[:, P:P + NX, JY - yh:JY + yh + 1, P:P + NZ].reshape(a.shape[0], -1)


def zmask(sel, yh=YH):
    return np.broadcast_to(sel[None, None, :], (NX, 2 * yh + 1, NZ)).ravel()


def smooth(F):
    return np.stack([ndi.gaussian_filter(f, SIG, mode="nearest", truncate=2.0) for f in F])


def grad_energy(F):
    """sum_i E[d_i f d_i f^T] over the region: (C, C)."""
    S = 0
    for f_axes in zip(*[TV.grads(f) for f in F]):                     # per axis i: tuple of C arrays
        g = region(np.stack(f_axes)); S = S + g @ g.T / g.shape[1]
    return S


def geig(A, B):
    C = len(A); B = B + 1e-9 * np.trace(B) / C * np.eye(C)
    lam, W = linalg.eigh(A, B); o = np.argsort(lam)[::-1]; lam, W = lam[o], W[:, o]
    return lam, W, np.linalg.inv(W).T


def projectors(W, V, kb):
    return V[:, :kb] @ W[:, :kb].T, V[:, kb:] @ W[:, kb:].T


def otsu(x, bins=256):
    h, e = np.histogram(x, bins); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    w1 = w[-1] - w; between = (m[-1] * w / w[-1] - m) ** 2 / np.maximum(w * w1, 1)
    return c[np.argmax(between[:-1])]


def methods(F):
    """All five splits of one image's features F (16, x, y, z). Returns {name: dict(Pb, Pf, kb, lam, cls)}."""
    X = region(F); mu = X.mean(1, keepdims=True); C = np.cov(X); out = {}
    w, U = np.linalg.eigh(C); U = U[:, np.argsort(w)[::-1]]
    out["M1 top PCs"] = dict(Pb=U[:, :4] @ U[:, :4].T, Pf=U[:, 4:] @ U[:, 4:].T, kb=4, lam=np.sort(w)[::-1] / w.sum(), cls=None)
    Fs = smooth(F)
    lam, W, V = geig(np.cov(region(Fs)), np.cov(region(F - Fs))); kb = max(1, int((lam > 1).sum()))
    out["M2 scale split"] = dict(zip(("Pb", "Pf"), projectors(W, V, kb)), kb=kb, lam=lam, cls=None)
    lam, W, V = geig(grad_energy(Fs), grad_energy(F))
    gaps = -np.diff(np.log(np.maximum(lam, 1e-12)))[:8]; kb = int(np.argmax(gaps)) + 1      # 1..8 boundary directions
    out["M3 gradient coherence"] = dict(zip(("Pb", "Pf"), projectors(W, V, kb)), kb=kb, lam=lam, cls=None)
    gm = np.sqrt(sum((region(np.stack(ax)) ** 2).sum(0) for ax in zip(*[TV.grads(f) for f in Fs])))
    edge = np.log(gm + 1e-12) > otsu(np.log(gm + 1e-12))
    lam, W, V = geig(np.cov(X[:, edge]), np.cov(X[:, ~edge])); kb = max(1, int((lam > 1).sum()))
    out["M4 edge voxels"] = dict(zip(("Pb", "Pf"), projectors(W, V, kb)), kb=kb, lam=lam, cls=edge)
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(0); sub = rng.choice(X.shape[1], 4000, replace=False)
    Xn = (X - mu) / np.linalg.norm(X - mu, axis=0, keepdims=True)
    lab = SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=15, assign_labels="cluster_qr",
                             random_state=0).fit_predict(Xn[:, sub].T)
    cls = KNeighborsClassifier(15).fit(Xn[:, sub].T, lab).predict(Xn.T).astype(bool)
    Sw = np.cov(X[:, cls]) + np.cov(X[:, ~cls]); d = np.linalg.solve(Sw, X[:, cls].mean(1) - X[:, ~cls].mean(1))
    Wf = np.linalg.cholesky(np.linalg.inv(Sw + 1e-9 * np.eye(16)))                         # whitening basis
    lam, W, V = geig(np.outer(d, d) @ Sw @ np.outer(d, d) + 1e-12 * np.eye(16), Sw)        # 1 discriminant direction
    out["M5 spectral bipartition"] = dict(zip(("Pb", "Pf"), projectors(W, V, 1)), kb=1, lam=lam, cls=cls)
    from sklearn.neighbors import KNeighborsClassifier
    # "mostly fibre": the bulk of the distribution is fibre
    from scipy.stats import chi2
    from sklearn.cluster import HDBSCAN
    from sklearn.covariance import MinCovDet
    Y10 = U[:, :10].T @ (X - mu); rng = np.random.default_rng(0); sub = rng.choice(X.shape[1], 6000, replace=False)
    mcd = MinCovDet(support_fraction=0.75, random_state=0).fit(Y10[:, sub].T)
    out_v = mcd.mahalanobis(Y10.T) > chi2.ppf(0.975, 10)
    core = ~out_v; mc = X[:, core].mean(1, keepdims=True); D = X[:, out_v] - mc
    lam, W, V = geig(D @ D.T / D.shape[1], np.cov(X[:, core])); kb = max(1, int((lam > 1).sum()))
    out["M6 robust core"] = dict(zip(("Pb", "Pf"), projectors(W, V, kb)), kb=kb, lam=lam, cls=out_v, labels=None)
    hl = HDBSCAN(min_cluster_size=60, min_samples=10).fit_predict(Y10[:, sub].T)
    good = hl >= 0
    full_l = KNeighborsClassifier(15).fit(Y10[:, sub][:, good].T, hl[good]).predict(Y10.T) if good.sum() > 15 else np.zeros(X.shape[1], int)
    # noise in the sample stays noise for voxels whose nearest sample point is noise
    nn = KNeighborsClassifier(1).fit(Y10[:, sub].T, hl).predict(Y10.T); full_l[nn < 0] = -1
    ids, cnt = np.unique(full_l[full_l >= 0], return_counts=True); fib = ids[np.argmax(cnt)] if len(ids) else 0
    nonf = full_l != fib; mc = X[:, ~nonf].mean(1, keepdims=True); D = X[:, nonf] - mc
    lam, W, V = geig(D @ D.T / max(D.shape[1], 1), np.cov(X[:, ~nonf])); kb = max(1, int((lam > 1).sum()))
    lab = np.where(full_l == fib, 0, np.where(full_l < 0, -1, 1 + np.searchsorted(ids[ids != fib], full_l)))
    out["M7 natural clusters"] = dict(zip(("Pb", "Pf"), projectors(W, V, kb)), kb=kb, lam=lam, cls=nonf, labels=lab,
                                      n_groups=int(len(ids)), noise=float((full_l < 0).mean()))
    for m in out.values():
        m["mu"] = mu; m.setdefault("labels", None)
    return out


def main(rho=16.0):
    F = {n: T.feats(n, None) for n in NAMES}                              # (16, 392, 120, 392) float32
    S = {}
    for n in NAMES:
        t0 = time.time(); S[n] = methods(F[n]); print(n, "splits", f"{time.time() - t0:.0f} s", flush=True)
    near, far = zmask(np.abs(ZC) < 3.0), zmask(np.abs(ZC) > 5.0)
    dX = region(F["D15"] - F["D0"])[:, near]
    Ff = smooth(F["D15"]); tex = region(F["D15"] - Ff)[:, far]; del Ff
    res = {}
    for m in METHODS:
        r = res[m] = {}
        b15, b0 = S["D15"][m], S["D0"][m]
        r["k_boundary"] = {n: int(S[n][m]["kb"]) for n in NAMES}
        r["step_capture_eval"] = float(((b15["Pb"] @ dX) ** 2).sum() / (dX ** 2).sum())
        r["fibre_texture_kept"] = float(((b15["Pf"] @ tex) ** 2).sum() / (tex ** 2).sum())
        X0 = region(F["D0"]) - b0["mu"]
        r["D0_boundary_share"] = float(((b0["Pb"] @ X0) ** 2).sum() / (X0 ** 2).sum())
        for n in NAMES:
            Zf = np.tensordot(S[n][m]["Pf"].astype(np.float32), F[n], axes=(1, 0))
            r[f"{n}_fibre_e3"] = T.score(O48.e3_eigen(Zf, rho, JY))
        if m == "M7 natural clusters":
            r["n_groups"] = {n: S[n][m]["n_groups"] for n in NAMES}; r["noise"] = {n: S[n][m]["noise"] for n in NAMES}
        print(f"{m:24s} k_b {r['k_boundary']}  step capture {r['step_capture_eval']:.0%}  texture kept {r['fibre_texture_kept']:.0%}"
              f"  D0 boundary share {r['D0_boundary_share']:.0%} | fibre e3 <0.5mm/3-8mm: D0 {r['D0_fibre_e3']['within_0.5mm']}"
              f"/{r['D0_fibre_e3']['deep_3-8mm']}  D15 {r['D15_fibre_e3']['within_0.5mm']}/{r['D15_fibre_e3']['deep_3-8mm']}", flush=True)
    for n in NAMES:                                                       # reference: no split
        res.setdefault("no split (all 16)", {})[f"{n}_fibre_e3"] = T.score(O48.e3_eigen(F[n], rho, JY))
    print("no split (all 16): D0", res["no split (all 16)"]["D0_fibre_e3"]["within_0.5mm"], "D15",
          res["no split (all 16)"]["D15_fibre_e3"]["within_0.5mm"], flush=True)
    (ST.OUT / "shape16_split.json").write_text(json.dumps(res, indent=1))
    figure(F, S, res)


def figure(F, S, res):
    SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    seq = LinearSegmentedColormap.from_list("seq_blue", SEQ); two = ListedColormap(["#2a78d6", "#eb6834"])
    sl = {n: F[n][:, P:P + NX, JY, P:P + NZ].reshape(16, -1) for n in NAMES}
    rng = np.random.default_rng(1); samp = rng.choice(NX * NZ, 5000, replace=False)

    def rgb(Pm):
        lo, hi = np.percentile(Pm, 1, axis=1), np.percentile(Pm, 99, axis=1)
        return np.clip((Pm - lo[:, None]) / np.maximum(hi - lo, 1e-12)[:, None], 0, 1).T.reshape(NX, NZ, 3)

    def top3(Y):
        w, U = np.linalg.eigh(np.cov(Y)); return U[:, np.argsort(w)[::-1][:3]].T @ (Y - Y.mean(1, keepdims=True))
    pc = {n: top3(sl[n]) for n in NAMES}
    fig, axs = plt.subplots(1 + len(METHODS), 8, figsize=(30, 3.5 * (1 + len(METHODS)) + 1.5), dpi=72)
    for j, n in enumerate(NAMES):
        _, draw, _ = EPF.truth_and_bounds(n); c0 = 4 * j
        img = T.crop(np.asarray(nib.load(SP.PAD_ROOT / f"{n}.nii.gz").dataobj).astype(np.float32), (0, 2))[:, JY, :]
        EPF.base_image(axs[0, c0], img, draw); axs[0, c0].set_title(f"{n}: image", fontsize=11)
        ax = axs[0, c0 + 1]; ax.imshow(np.transpose(rgb(pc[n]), (1, 0, 2)), origin="lower", extent=EPF.EXT); draw(ax)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{n}: shape space (16-D), RGB = PC 1-3", fontsize=11)
        ax = axs[0, c0 + 2]; ax.scatter(pc[n][0, samp], pc[n][1, samp], c=rgb(pc[n]).reshape(-1, 3)[samp], s=3, lw=0)
        ax.set_title(f"{n}: shape space, PC1 vs PC2", fontsize=11); ax.grid(alpha=0.3); ax.set_xticks([]); ax.set_yticks([])
        axs[0, c0 + 3].axis("off")
        for r, m in enumerate(METHODS, 1):
            s = S[n][m]; X = sl[n] - s["mu"]; xb, xf = s["Pb"] @ X, s["Pf"] @ X
            share = (xb ** 2).sum(0) / np.maximum((xb ** 2).sum(0) + (xf ** 2).sum(0), 1e-12)
            if s.get("labels") is not None:
                cat = ListedColormap(["#a3a29b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
                lb = s["labels"].reshape(NX, 2 * YH + 1, NZ)[:, YH, :].ravel()
                v_ = np.clip(lb + 1, 0, 8); col = cat(v_ / 8.0); mp = v_.reshape(NX, NZ) / 8.0; cm = cat
                lab = f"{s['n_groups']} natural groups: blue = largest (fibre), grey = noise ({s['noise']:.0%})"
            elif s["cls"] is not None:
                cl = s["cls"].reshape(NX, 2 * YH + 1, NZ)[:, YH, :].ravel()
                col = two(cl.astype(float)); mp = cl.reshape(NX, NZ).astype(float); cm, lab = two, "class: orange = boundary voxels"
            else:
                col = seq(share); mp = share.reshape(NX, NZ); cm, lab = seq, "boundary share of each voxel (0-1)"
            ax = axs[r, c0]; ax.scatter(pc[n][0, samp], pc[n][1, samp], c=col[samp], s=3, lw=0)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.3)
            ax.set_title(f"{n}: classified space (PC1 vs PC2)", fontsize=9.5)
            ax = axs[r, c0 + 1]; ax.imshow(mp.T, cmap=cm, vmin=0, vmax=1, origin="lower", extent=EPF.EXT, interpolation="nearest")
            draw(ax); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{n}: {lab}", fontsize=9.5)
            cb = np.zeros((3, X.shape[1])); kb = s["kb"]
            wb, Ub = np.linalg.eigh(np.cov(xb)); cb[:min(3, kb)] = (Ub[:, np.argsort(wb)[::-1][:min(3, kb)]].T @ xb)
            for c, (Y, t) in ((2, (cb, f"boundary component ({kb} dir{'s' if kb > 1 else ''})")), (3, (top3(xf), "fibre component"))):
                ax = axs[r, c0 + c]; ax.imshow(np.transpose(rgb(Y) if Y.any() else np.zeros((NX, NZ, 3)), (1, 0, 2)), origin="lower", extent=EPF.EXT)
                draw(ax); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{n}: {t}", fontsize=9.5)
            if j == 0:
                q = res[m]
                axs[r, 0].set_ylabel(f"{m}\nstep capture {q['step_capture_eval']:.0%} (eval)\ntexture kept {q['fibre_texture_kept']:.0%}\n"
                                     f"D0 boundary share {q['D0_boundary_share']:.0%}\nfibre e3 <0.5 mm: D0 {q['D0_fibre_e3']['within_0.5mm']:.0f}°, "
                                     f"D15 {q['D15_fibre_e3']['within_0.5mm']:.0f}°", fontsize=10.5, rotation=0, ha="right", va="center", labelpad=10)
    ns = res["no split (all 16)"]
    fig.suptitle("PROTOTYPE. Unpaired boundary / fibre division of the 16-D anatomix shape space (whole block, D0 and D15). Every method "
                 "sees one image (no twin, no labels, no k-means), 2-way.\nRows: the method's classified space (scatter + map), its boundary "
                 f"and fibre components. Fibre e3 = orientation of the fibre-component shape tensor (ρ 16), median error; no split: "
                 f"D0 {ns['D0_fibre_e3']['within_0.5mm']:.0f}°, D15 {ns['D15_fibre_e3']['within_0.5mm']:.0f}°. Step capture uses the twin for "
                 "evaluation only.", fontsize=13, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.93, wspace=0.06, hspace=0.22, left=0.12, right=0.99)
    out = ST.FIGS / "shape16_split.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
