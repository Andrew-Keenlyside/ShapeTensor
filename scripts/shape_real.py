"""PROTOTYPE shape tensor on the real subcortical VOI: the deep-WM 100 um block (675, 577, 775) with the most internal
interfaces (the deep block of glyphs_real_blocks.png). Tensors only.

1. Shapes. anatomix features of a 192^3 block (reflect-padded), k-means (K clusters) on the central 100^3. Each
   cluster is called
     grey      if most of it is below the WM threshold,
     boundary  if it is enriched >= 2x in boundary voxels (within 0.3 mm of an internal interface or of the mask
               edge; internal interface = coarse-gradient step, as edge_aware_real.py),
     fibre     otherwise (tract interiors).
2. Boundary directions of shape space: the generalised eigenvectors of S_b u = lambda S_w u, S_b the scatter of the
   boundary-cluster centroids about the fibre mean, S_w the fibre clusters' covariance. The fibre component is
   P_k f, P_k = I - Q Q^T over the top k.
3. Tensors (e3; sigma 1 voxel): std (BRIDGE, intensity) at rho 3 and 6; shape (all of shape space) at rho 6;
   fibre (P_k f) at rho 3 and 6. Compared with each other (angle difference by region) and with HCP-1065 trunk
   tangents (an outside population reference, not truth).

    python shape_real.py OUTDIR [K]        (chunkreg env; needs deep_voi_hcp.npz from the HCP extraction)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
from scipy import linalg, ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import shape_tensor as ST
import tensor_variants as TV

C = np.array([675, 577, 775]); HALF = 96; EV = slice(46, 146)            # 192^3 read, central 100^3 evaluated
THR, GMED = 144.5, 0.8949740529060364                                    # global median coarse gradient (real_blocks_abs)
HCP = ST.OUT / "deep_voi_hcp.npz"


def tensor_e3(T6, rho):
    """e3 on the evaluated crop from six full-block T0 components."""
    S = [TV.integ(t, rho)[EV, EV, EV] for t in T6]
    M = np.empty(S[0].shape + (3, 3), np.float32)
    for k, (i, j) in enumerate(TV.PAIRS):
        M[..., i, j] = M[..., j, i] = S[k]
    w, v = np.linalg.eigh(M)
    return v[..., :, 0], w[..., ::-1]


def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def main(out, K=12):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    v = AS.read_block(tuple(C), HALF)
    F = AS.anatomix(v, 0.1)
    # regions (as edge_aware_real.block)
    mask = v >= THR
    sdf = (ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)) * 0.1
    gcv = np.stack([ndi.gaussian_filter(v, 3.0, order=o, mode="nearest", truncate=2.0) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], -1)
    gcm = np.linalg.norm(gcv, axis=-1); nc = gcv / np.maximum(gcm[..., None], 1e-9)
    iface = (gcm > 3.0 * GMED) & (sdf > 1.5)
    near_if = ndi.binary_dilation(iface, iterations=5)
    bnd = ndi.binary_dilation(iface, iterations=3) | (np.abs(sdf) < 0.3)
    E = (EV, EV, EV)
    mask_e, sdf_e, near_e, bnd_e, nc_e = mask[E], sdf[E], near_if[E], bnd[E], nc[E]
    deep_e = mask_e & (sdf_e > 2.0)
    # 1. shapes: k-means on the evaluated crop
    from sklearn.cluster import MiniBatchKMeans
    Fe = F[(slice(None),) + E].reshape(16, -1).T
    rng = np.random.default_rng(0)
    km = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=8192).fit(Fe[rng.choice(len(Fe), 200_000, replace=False)])
    lab = km.predict(Fe).reshape(mask_e.shape)
    base_b = bnd_e[mask_e].mean()
    kinds, info = [], []
    for k in range(K):
        m = lab == k; inm = mask_e[m].mean(); b = bnd_e[m & mask_e].mean() if (m & mask_e).any() else 0.0
        kind = "grey" if inm < 0.5 else ("boundary" if b >= 2 * base_b else "fibre")
        kinds.append(kind); info.append({"cluster": k, "kind": kind, "voxels": int(m.sum()), "in_wm_mask": round(float(inm), 3),
                                         "boundary_share": round(float(b), 3), "enrichment": round(float(b / base_b), 2)})
    print("boundary share in WM", round(float(base_b), 3)); [print(i) for i in info]
    labf = lab.ravel()
    fib = np.isin(labf, [k for k in range(K) if kinds[k] == "fibre"])
    mu_f = Fe[fib].mean(0); Sw = np.cov(Fe[fib].T) + 1e-8 * np.eye(16)
    Sb = np.zeros((16, 16))
    for k in range(K):
        if kinds[k] == "boundary":
            d = Fe[labf == k].mean(0) - mu_f; Sb += (labf == k).sum() * np.outer(d, d)
    lam, U = linalg.eigh(Sb / max(1, sum((labf == k).sum() for k in range(K) if kinds[k] == "boundary")), Sw)
    o = np.argsort(lam)[::-1]; lam, U = lam[o], U[:, o]
    print("boundary directions, generalised eigenvalues", np.round(lam[:6], 3))
    # 2./3. tensors
    g = TV.grads(v); T_I = [g[i] * g[j] for i, j in TV.PAIRS]; del g
    e = {}; lamd = {}
    e["std_r3"], lamd["std_r3"] = tensor_e3(T_I, 3.0)
    e["std_r6"], lamd["std_r6"] = tensor_e3(T_I, 6.0)
    T_F = ST.t0_shape(F); e["shape_r6"], _ = tensor_e3(T_F, 6.0); del T_F
    stats_k = {}
    for kk in (1, 2, 4):
        Q, _ = np.linalg.qr(U[:, :kk]); P = np.eye(16, dtype=np.float32) - (Q @ Q.T).astype(np.float32)
        PF = np.tensordot(P, F, axes=(1, 0))
        Xb = Fe[np.isin(labf, [k for k in range(K) if kinds[k] == "boundary"])] - mu_f
        stats_k[kk] = {"boundary_offset_removed": round(float(1 - ((Xb @ P.T).mean(0) ** 2).sum() / (Xb.mean(0) ** 2).sum()), 3),
                       "fibre_variance_kept": round(float(np.trace(P @ Sw @ P.T) / np.trace(Sw)), 3)}
        T_P = ST.t0_shape(PF); del PF
        for rho in (3.0, 6.0):
            e[f"fibre{kk}_r{rho:g}"], _ = tensor_e3(T_P, rho)
        del T_P
    # metrics
    d = np.load(HCP); A100 = d["A100"]
    q = np.rint(nib.affines.apply_affine(np.linalg.inv(A100), d["pts"])).astype(int) - (C - HALF) - 46
    inb = np.all((q >= 0) & (q < 100), 1); q, tan = q[inb], d["tan"][inb]; qi = tuple(q.T)
    sh = sdf_e[qi] < 1.0; nf = near_e[qi] & ~sh; el = ~sh & ~nf
    res = {"clusters": info, "boundary_eigenvalues": [round(float(x), 4) for x in lam], "projection": stats_k,
           "n_hcp": int(len(q)), "tensors": {}}
    regions = {"near_iface": deep_e & near_e, "deep_elsewhere": deep_e & ~near_e, "shell_0-1mm": mask_e & (sdf_e > 0) & (sdf_e < 1.0)}
    for name, ev in e.items():
        a = ang(ev[qi], tan)
        r = {"hcp_near_iface": round(float(np.median(a[nf])), 1), "hcp_elsewhere": round(float(np.median(a[el])), 1),
             "hcp_shell": round(float(np.median(a[sh])), 1) if sh.any() else None,
             "iface_in_plane": round(float((np.abs((ev * nc_e).sum(-1)) < 0.5)[deep_e & near_e].mean()), 3)}
        for ref in ("std_r3", "std_r6"):
            if name != ref:
                dd = ang(ev, e[ref])
                r[f"vs_{ref}"] = {rg: round(float(np.median(dd[m])), 1) for rg, m in regions.items()}
        res["tensors"][name] = r
        print(f"{name:12s}", json.dumps(r), flush=True)
    (out / "shape_real.json").write_text(json.dumps(res, indent=1))
    figures(out, v[E], lab, kinds, F[(slice(None),) + E], e, mask_e, bnd_e, iface[E], res)


def figures(out, v, lab, kinds, F, e, mask, bnd, iface, res):
    H = v.shape[1] // 2
    sl = v[:, H, :]; lo, hi = np.percentile(v, (0.5, 99.5))
    K = len(kinds)
    # --- shapes
    fig, axs = plt.subplots(1, 4, figsize=(24, 6.6), dpi=100)
    axs[0].imshow(sl.T, cmap="gray", origin="lower", vmin=lo, vmax=hi); axs[0].set_title("intensity (x-z, mid y)")
    ifc = ndi.binary_dilation(iface, iterations=2)[:, H, :].T.astype(float)
    axs[0].contour(ifc, [0.5], colors="c", linewidths=0.7)
    axs[1].imshow(lab[:, H, :].T, cmap=ListedColormap(plt.get_cmap("tab20").colors[:K]), origin="lower", interpolation="nearest",
                  vmin=-0.5, vmax=K - 0.5)
    axs[1].set_title(f"{K} shape clusters (k-means on anatomix features)")
    kind_map = np.vectorize({"grey": 0, "fibre": 1, "boundary": 2}.get)(np.asarray(kinds)[lab[:, H, :]])
    axs[2].imshow(kind_map.T, cmap=ListedColormap(["#555555", "#4daf4a", "#e41a1c"]), origin="lower", vmin=-0.5, vmax=2.5,
                  interpolation="nearest")
    axs[2].contour(ifc, [0.5], colors="c", linewidths=0.7)
    axs[2].set_title("shapes: grey / fibre (green) / boundary (red); cyan = internal interface")
    ci = res["clusters"]; x = np.arange(K)
    axs[3].bar(x, [c["enrichment"] for c in ci], color=[{"grey": "#555555", "fibre": "#4daf4a", "boundary": "#e41a1c"}[c["kind"]] for c in ci])
    axs[3].axhline(2, color="k", ls="--", lw=0.8); axs[3].set_xticks(x)
    axs[3].set_xlabel("cluster"); axs[3].set_ylabel("boundary-voxel enrichment (x WM base rate)")
    axs[3].set_title("which shapes sit on boundaries")
    for a in axs[:3]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Subcortical VOI (deep WM, 10 x 10 x 10 mm at 100 um): fibre and boundary shapes in anatomix shape space", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out / "voi_shapes.png"); plt.close(fig)
    # --- orientations and differences (each shape-space tensor against the standard tensor at the same rho)
    ifc = ndi.binary_dilation(iface, iterations=2)[:, H, :].T.astype(float)
    cols = (("std_r3", None), ("std_r6", "std_r3"), ("fibre2_r3", "std_r3"), ("fibre2_r6", "std_r6"), ("shape_r6", "std_r6"))
    titles = {"std_r3": "standard tensor, rho 0.3 mm\n(BRIDGE now)", "std_r6": "standard tensor, rho 0.6 mm",
              "fibre2_r3": "fibre tensor (boundary directions\nremoved), rho 0.3 mm", "fibre2_r6": "fibre tensor (boundary directions\nremoved), rho 0.6 mm",
              "shape_r6": "full shape tensor, rho 0.6 mm"}
    fig, axs = plt.subplots(2, len(cols), figsize=(5.6 * len(cols), 12.5), dpi=100)
    st = 3; xi, zi = np.meshgrid(np.arange(2, sl.shape[0] - 2, st), np.arange(2, sl.shape[1] - 2, st), indexing="ij")
    for c, (name, ref) in enumerate(cols):
        ax = axs[0, c]; ax.imshow(sl.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        ee = e[name][xi, H, zi]; inpl = np.hypot(ee[..., 0], ee[..., 2]); dd = ee[..., [0, 2]] / np.maximum(inpl[..., None], 1e-9)
        Lg = 1.3 * inpl
        segs = np.stack([np.stack([xi - dd[..., 0] * Lg, zi - dd[..., 1] * Lg], -1), np.stack([xi + dd[..., 0] * Lg, zi + dd[..., 1] * Lg], -1)], -2)
        ax.add_collection(LineCollection(segs.reshape(-1, 2, 2), colors=np.clip(np.abs(ee).reshape(-1, 3), 0, 1), linewidths=0.9))
        ax.contour(ifc, [0.5], colors="c", linewidths=0.7)
        r = res["tensors"][name]
        ax.set_title(f"{titles[name]}\nto HCP: near interfaces {r['hcp_near_iface']:.0f}, elsewhere {r['hcp_elsewhere']:.0f} deg\n"
                     f"e3 in the interface plane: {r['iface_in_plane']:.0%}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axs[1, c]
        if ref is None:
            km = np.vectorize({"grey": 0, "fibre": 1, "boundary": 2}.get)(np.asarray(kinds)[lab[:, H, :]])
            ax.imshow(km.T, cmap=ListedColormap(["#555555", "#4daf4a", "#e41a1c"]), origin="lower", vmin=-0.5, vmax=2.5, interpolation="nearest")
            ax.set_title("shapes: grey / fibre (green) / boundary (red)", fontsize=10)
        else:
            dmap = ang(e[name][:, H, :], e[ref][:, H, :])
            im = ax.imshow(dmap.T, cmap="inferno", origin="lower", vmin=0, vmax=90)
            vr = res["tensors"][name][f"vs_{ref}"]
            ax.set_title(f"angle to the standard tensor, rho {ref[-1:] == '6' and '0.6' or '0.3'} mm\nmedian: near interfaces {vr['near_iface']:.0f}, "
                         f"deep elsewhere {vr['deep_elsewhere']:.0f},\nshell {vr['shell_0-1mm']:.0f} deg", fontsize=10)
        ax.contour(ifc, [0.5], colors="c", linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axs[1, 1:].tolist(), shrink=0.7, label="angle (deg)")
    fig.suptitle("Subcortical VOI: fibre-component shape tensor vs the standard tensor (glyphs |x| R |y| G |z| B; x-z plane, mid y; "
                 "cyan = internal interfaces)", fontsize=14)
    fig.savefig(out / "voi_fibre_vs_standard.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
