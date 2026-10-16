"""PROTOTYPE: fibre orientation from the fibre-only shape tensor by tensor component analysis (TCA: CP / PARAFAC) instead
of its eigenvectors. Tract-interface phantoms D0 / D15 / D30 (0.1 mm voxels), cached anatomix features (shape_window.py).

The fibre-only shape tensor T = G_rho * sum_c grad(f_c) grad(f_c)^T (f = the 12 fibre-only channels: shape space with
the top 4 step directions of D15 - D0 removed, as in shape_window.py) is the gradient-axis Gram of a 3-way array per
window, X[voxel, channel, axis] = sqrt(G_rho) * df_c/dx_i. Its eigenvectors are that mode's orthogonal factors. TCA
fits X ~ sum_r a_r o b_r o d_r instead: R components, each a spatial map a_r, a channel signature b_r and a gradient
direction d_r, with the d_r free to be non-orthogonal. Exactly, and much smaller: CP of X equals CP of
K^(1/2) reshaped (36, 12, 3), where K = G_rho * (g g^T) is the channel-resolved (36 x 36) tensor, g = (channel, axis).

Read-outs, per pixel of the mid-y x-z slice (the fibre is the direction least represented by the kept gradients):
  eigen        e3 of T (= the fibre tensor of shape_window.py; the control)
  tca_all      null direction of sum_r lam_r^2 d_r d_r^T over all R components
  tca_oracle   the same after dropping the component closest to the true interface normal (z): an upper bound
  tca_strong   the same after dropping the strongest component (label-free)
Scored as shape_tensor.score_e3 (median angle to the true fibres within 0.5 mm of the interface, and 3-8 mm away), on
the pixels it uses: every pixel within 2 mm of the interface, a stride-2 grid 3-8 mm away.
CP: batched ALS on the GPU; one structured start (directions from the eigenvectors of T, channel signatures from the
channel Gram) and two random, best fit kept per pixel. K^(1/2) is replaced by its Cholesky factor (same Gram, so the
same CP up to an orthogonal transform of the spatial mode): batched eigh of 36 x 36 on the GPU runs one matrix at a time.

    python shape_tca.py [w32,w64,w128,full] [rho,...]    scores (shapetensors env, one GPU) -> shape_tca.json
    python shape_tca.py maps w64 16                     full-slice read-outs (R 3) -> shape_tca_maps_<win>_rho<rho>.npz
    python shape_tca.py fig [w64 16]                    figures/shape_tensor/shape_tca_summary.png, shape_tca_maps.png
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import shape_tensor as ST
import shape_window as SW
import tensor_variants as TV

NAMES = ("D0", "D15", "D30")
RANKS = (3, 4)
STARTS, ITERS = 3, 150
NORMAL = torch.tensor([0.0, 0.0, 1.0])                                # interface normal: the plane z = D_PLANE_Z
OUT = ST.OUT / "shape_tca.json"
DEV = "cuda"
PAD = 0              # "--pad": phantoms grown by SP.D_PAD voxels per x and z face; orientations cropped before scoring


def feats(name, w):
    """Features of phantom name at window w: the shape_window cache, or (PAD) the grown phantom's, float32."""
    if not PAD:
        return SW.feats(name, w)
    p = ST.OUT / f"win_pad{PAD}" / f"{name}_{SW.wname(w)}.npy"
    if p.exists():
        return np.load(p)
    import nibabel as nib
    import anatomix_shape as AS
    v = np.asarray(nib.load(ST.SP.PAD_ROOT / f"{name}.nii.gz").dataobj).astype(np.float32)
    kw = {} if w is None else {"window": w, "overlap": 0.75, "sw_batch": 8}
    F = AS.anatomix_clean(v, ST.SP.VOX, **kw).astype(np.float32)
    p.parent.mkdir(parents=True, exist_ok=True); np.save(p, F)
    return F


def crop(a, axes):
    """The original NX x NZ region of a grown-phantom array (axes = its x and z axes)."""
    if not PAD:
        return a
    sl = [slice(None)] * a.ndim
    sl[axes[0]] = slice(PAD, PAD + ST.SP.NX); sl[axes[1]] = slice(PAD, PAD + ST.SP.NZ)
    return a[tuple(sl)]


def k_slice(Z, rho, jy=None):
    """Z (C, x, y, z) channels -> K (x, z, 3C, 3C) = G_rho * (g g^T) on the mid-y slice, g index c * 3 + i.
    Exactly the 3-D filter of shape_window.angle at y = JY: the y pass is the kernel's weighted sum at the centre (the
    slab is wider than the kernel), then x and z are filtered as there (mode nearest, truncate 2)."""
    m = int(np.ceil(2 * rho)) + 4; jy = ST.JY if jy is None else jy
    G = np.stack([g for f in Z[:, :, jy - m:jy + m + 1, :] for g in TV.grads(f)])              # (3C, x, 2m+1, z)
    r = int(TV.TR * rho + 0.5); y = np.arange(-r, r + 1); w = np.exp(-0.5 * (y / rho) ** 2); w /= w.sum()
    Gt = torch.from_numpy(G).to(DEV)
    ng = G.shape[0]; K = torch.zeros((ng, ng) + (G.shape[1], G.shape[3]), device=DEV)
    for k, wy in zip(range(m - r, m + r + 1), w):
        g = Gt[:, :, k, :]; K += float(wy) * g[:, None] * g[None]
    K = K.cpu().numpy()
    for a in range(ng):
        for b in range(a, ng):
            K[a, b] = ndi.gaussian_filter(K[a, b], rho, mode="nearest", truncate=TV.TR); K[b, a] = K[a, b]
    return torch.from_numpy(np.ascontiguousarray(K.transpose(2, 3, 0, 1))).to(DEV)          # (x, z, 3C, 3C)


TD = np.load(ST.SP.ROOT / "D_truth.npz")
TRU, DEP, XD = TD["tru"], TD["depth"], TD["X"]
_sub = np.zeros(DEP.shape, bool); _sub[::2, ::2] = True
MASK = (XD >= 1) & (XD < 19) & ((DEP < 2.0) | ((DEP >= 3.0) & (DEP < 8.0) & _sub))       # the pixels the scores use


def score(e):
    """shape_tensor.score_e3 on the MASK pixels (NaN elsewhere)."""
    a = TV.angle3(e, TRU)
    r = {f"{ang:.0f}deg": [round(float(np.nanmedian(a[(XD >= x0 + 1) & (XD < x1 - 1) & (DEP >= b0) & (DEP < b1)])), 1)
                           for b0, b1 in ST.BINS] for x0, x1, ang in ST.SP.D_BANDS}
    inner = (DEP < 0.5) & (XD >= 1) & (XD < 19); deep = (DEP >= 3) & (DEP < 8) & (XD >= 1) & (XD < 19)
    r["within_0.5mm"] = round(float(np.nanmedian(a[inner])), 1); r["deep_3-8mm"] = round(float(np.nanmedian(a[deep])), 1)
    return r


def null_dir(M):
    """Unit eigenvector of the smallest eigenvalue of symmetric (..., 3, 3) M."""
    return torch.linalg.eigh(M)[1][..., :, 0]


def cp_als(Y, R, init, starts=STARTS, iters=ITERS, seed=0):
    """Batched CP-ALS of Y (N, P, C, I), best of `starts` starts per item (the first from init = (B0 (N, C, >=R),
    D0 (N, I, 3)), padded with random columns; the rest random). Returns A (N, P, R), B (N, C, R), D (N, I, R) (B, D
    unit columns) and the relative error (N,)."""
    N, P, C, I = Y.shape
    g = torch.Generator(device=DEV).manual_seed(seed)
    yy = (Y ** 2).sum((1, 2, 3))
    best = torch.full((N,), float("inf"), device=DEV)
    bA, bB, bD = (torch.zeros(N, n, R, device=DEV) for n in (P, C, I))
    eye = torch.eye(R, device=DEV)

    def solve(Gm, M):                                                     # M (N, X, R) <- M @ inv(Gm)
        reg = 1e-6 * Gm.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye
        return torch.linalg.solve(Gm + reg, M.transpose(1, 2)).transpose(1, 2)

    for s in range(starts):
        B = torch.randn(N, C, R, device=DEV, generator=g); D = torch.randn(N, I, R, device=DEV, generator=g)
        if s == 0:
            B[:, :, :R] = init[0][:, :, :R]; k = min(R, init[1].shape[2]); D[:, :, :k] = init[1][:, :, :k]
        for _ in range(iters):
            A = solve((B.mT @ B) * (D.mT @ D), torch.einsum("npci,ncr,nir->npr", Y, B, D))
            B = solve((A.mT @ A) * (D.mT @ D), torch.einsum("npci,npr,nir->ncr", Y, A, D))
            D = solve((A.mT @ A) * (B.mT @ B), torch.einsum("npci,npr,ncr->nir", Y, A, B))
            nb = B.norm(dim=1, keepdim=True).clamp_min(1e-12); nd = D.norm(dim=1, keepdim=True).clamp_min(1e-12)
            B = B / nb; D = D / nd; A = A * nb * nd
        err = ((Y - torch.einsum("npr,ncr,nir->npci", A, B, D)) ** 2).sum((1, 2, 3)) / yy.clamp_min(1e-30)
        take = err < best
        best = torch.where(take, err, best); bA[take] = A[take]; bB[take] = B[take]; bD[take] = D[take]
    return bA, bB, bD, best


GATES = (10, 20, 30)          # deg: the oracle drops the component nearest the interface normal only within this cone


def readouts(K, mask=MASK, ranks=RANKS):
    """All read-outs for K (x, z, 36, 36) on the mask pixels; returns {name: e (x, z, 3), NaN off the mask},
    {R: median relative CP error}, {name: per-pixel diagnostic (x, z)}.

    A drop never rebuilds the tensor from the CP components alone (they explain only part of the gradient energy):
    it removes the dropped component's share from the exact tensor, T - G(all) + G(kept), where G(S) is the gradient-
    axis Gram of the reconstruction from components S (cross terms included). With nothing dropped this is T."""
    X, Zn = K.shape[:2]
    Kf = K[torch.from_numpy(mask).to(DEV)].double()
    Kf = Kf / Kf.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-30)[:, None, None]       # unit trace: scale-free ALS
    C = Kf.shape[-1] // 3                                                              # channels (12 fibre-only, 16 all)
    T = Kf.reshape(-1, C, 3, C, 3).diagonal(dim1=1, dim2=3).sum(-1)                     # sum over channels: (N, 3, 3)
    out = {"eigen": null_dir(T)}; diag = {}
    eye = torch.eye(3 * C, device=DEV, dtype=Kf.dtype)
    L, info = torch.linalg.cholesky_ex(Kf + 1e-9 * eye)
    if (info > 0).any():
        L[info > 0] = torch.linalg.cholesky(Kf[info > 0] + 1e-6 * eye)
    Y = L.mT.float().reshape(-1, 3 * C, C, 3)                                          # Y^T Y = K: (N, P, C, I)
    Cg = Kf.reshape(-1, C, 3, C, 3).diagonal(dim1=2, dim2=4).sum(-1)                    # channel Gram (N, C, C)
    init = (torch.linalg.eigh(Cg)[1].flip(-1).float(), torch.linalg.eigh(T)[1].flip(-1).float())
    fit = {}; Tf = T.float(); idx = torch.arange(len(T), device=DEV)
    for R in ranks:
        A, B, D, err = cp_als(Y, R, init)
        fit[R] = float(err.median())
        W = (A.mT @ A) * (B.mT @ B)                                                    # (N, R, R)

        def gram(keep):                                                                # keep (N, R) 0/1
            return D @ (W * keep[:, :, None] * keep[:, None, :]) @ D.mT
        g_all = gram(torch.ones(len(T), R, device=DEV))
        out[f"tca_recon_R{R}"] = null_dir(g_all)                                       # CP components only
        share = W.diagonal(dim1=1, dim2=2) / W.diagonal(dim1=1, dim2=2).sum(1, keepdim=True)
        cosn = (D * NORMAL.to(DEV)[None, :, None]).sum(1).abs()                        # |d_r . n|
        k_o = cosn.argmax(1); ang_o = torch.rad2deg(torch.arccos(cosn[idx, k_o].clamp(max=1)))
        k_s = share.argmax(1)
        diag[f"oracle_angle_R{R}"] = ang_o; diag[f"oracle_share_R{R}"] = share[idx, k_o]

        def dropped(k, ok=None):
            keep = torch.ones(len(T), R, device=DEV); keep[idx, k] = 0
            if ok is not None:
                keep[~ok] = 1
            return null_dir(Tf - g_all + gram(keep))
        out[f"tca_oracle_R{R}"] = dropped(k_o)
        for gdeg in GATES:
            out[f"tca_oracle_g{gdeg}_R{R}"] = dropped(k_o, ang_o < gdeg)
        out[f"tca_strong_R{R}"] = dropped(k_s)
    full, dfull = {}, {}
    for k, v in out.items():
        e = np.full((X, Zn, 3), np.nan, np.float32); e[mask] = v.float().cpu().numpy(); full[k] = e
    for k, v in diag.items():
        a = np.full((X, Zn), np.nan, np.float32); a[mask] = v.float().cpu().numpy(); dfull[k] = a
    return full, fit, dfull


def fib_cap(n, cap_deg=90.0):
    """n near-uniform unit vectors in the cap within cap_deg of +z (90: the upper hemisphere), Fibonacci lattice."""
    k = np.arange(n) + 0.5; c = 1 - (1 - np.cos(np.radians(cap_deg))) * k / n
    r = np.sqrt(1 - c ** 2); ph = k * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(ph), r * np.sin(ph), c], 1)


def split2(K, mask=MASK, Ls=(0, 1, 2, 4), n_coarse=1500, n_fine=300, cap=6.0, chunk=50):
    """PROTOTYPE 2-way split of the gradient energy into fibre and non-fibre, no clusters and no oracle. The fibre
    direction u is the one whose gradient projection u . grad f, as a (space x channel) matrix, has the least energy
    OUTSIDE its best rank-L part: along the fibre there should be either nothing (fibre gradients are perpendicular to
    it) or one coherent low-rank structure (a step: a sheet with a few shape-space signatures). L = 0 is the standard
    tensor's e3. Exhaustive search on the hemisphere, then a finer cap around the best direction.
    Returns {f"split_L{L}": e (x, z, 3), NaN off the mask}."""
    X, Zn = K.shape[:2]; C = K.shape[-1] // 3
    Kf = K[torch.from_numpy(mask).to(DEV)].double()
    Kf = Kf / Kf.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-30)[:, None, None]
    eye = torch.eye(3 * C, device=DEV, dtype=Kf.dtype)
    Lc, info = torch.linalg.cholesky_ex(Kf + 1e-9 * eye)
    if (info > 0).any():
        Lc[info > 0] = torch.linalg.cholesky(Kf[info > 0] + 1e-6 * eye)
    Y = Lc.mT.float().reshape(len(Kf), 3 * C, C, 3)                                  # (N, P, C, I), Y^T Y = K
    Q = torch.einsum("npci,npdj->nijcd", Y, Y)                                        # (N, 3, 3, C, C)
    N = len(Q); out = {}

    def tails(U):                                                                     # U (N or 1, m, 3) -> (N, m, C) eigenvalues
        ev = []
        for a in range(0, U.shape[1], chunk):
            Uc = U[:, a:a + chunk]
            G = torch.einsum("nui,nuj,nijcd->nucd", Uc.expand(N, -1, -1), Uc.expand(N, -1, -1), Q)
            ev.append(torch.linalg.eigvalsh(G))                                       # ascending
        return torch.cat(ev, 1)
    U0 = torch.from_numpy(fib_cap(n_coarse)).float().to(DEV)[None]
    ev0 = tails(U0)
    cap_dirs = torch.from_numpy(fib_cap(n_fine, cap)).float().to(DEV)                 # around +z
    for L in Ls:
        t0 = ev0[..., :C - L].sum(-1)                                                 # energy outside the top L
        u = U0[0, t0.argmin(1)]                                                       # (N, 3) coarse best
        # rotate the fine cap from +z to each u (Rodrigues), evaluate, keep the best
        zax = torch.tensor([0.0, 0.0, 1.0], device=DEV); v = torch.cross(zax.expand_as(u), u, dim=1)
        s_ = v.norm(dim=1, keepdim=True); c_ = u[:, 2:3]
        vx = torch.zeros(N, 3, 3, device=DEV)
        vx[:, 0, 1], vx[:, 0, 2], vx[:, 1, 0], vx[:, 1, 2], vx[:, 2, 0], vx[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
        Rm = torch.eye(3, device=DEV) + vx + vx @ vx * ((1 - c_) / s_.clamp_min(1e-9) ** 2)[:, :, None]
        Rm[s_[:, 0] < 1e-6] = torch.eye(3, device=DEV)
        Uf = torch.einsum("nij,mj->nmi", Rm, cap_dirs)
        tf = tails(Uf)[..., :C - L].sum(-1)
        out[f"split_L{L}"] = Uf[torch.arange(N, device=DEV), tf.argmin(1)]
    full = {}
    for k, v in out.items():
        e = np.full((X, Zn, 3), np.nan, np.float32); e[mask] = v.float().cpu().numpy(); full[k] = e
    return full


def split_main(wins=("full",), rhos=(16.0,)):
    """2-way split scores, on the fibre-only 12 channels and on all 16 (no step directions removed: no twin needed)."""
    p = ST.OUT / f"shape_split{tag()}.json"; res = json.loads(p.read_text()) if p.exists() else {}
    for wn in wins:
        w = None if wn == "full" else int(wn[1:])
        F = {n: feats(n, w) for n in NAMES}
        E, _ = SW.model({n: crop(f, (1, 3)) for n, f in F.items()})
        for chans, Qc in (("fibre12", E[:, SW.K:].T.astype(np.float32)), ("all16", np.eye(16, dtype=np.float32))):
            for rho in rhos:
                key = f"{wn}_rho{rho:g}_{chans}"; res[key] = {}
                for n in NAMES:
                    t0 = time.time()
                    e = split2(crop(k_slice(np.tensordot(Qc, F[n], axes=(1, 0)), rho), (0, 1)))
                    res[key][n] = {k: score(v) for k, v in e.items()}
                    print(f"{key:22s} {n:4s} <0.5mm / 3-8mm  " + "  ".join(
                        f"{k} {v['within_0.5mm']:4.1f}/{v['deep_3-8mm']:4.1f}" for k, v in res[key][n].items())
                          + f"  ({time.time() - t0:.0f} s)", flush=True)
                p.write_text(json.dumps(res, indent=1))


def main(wins=("w32", "w64", "w128", "full"), rhos=(8.0, 16.0)):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for wn in wins:
        w = None if wn == "full" else int(wn[1:])
        t0 = time.time()
        F = {n: feats(n, w) for n in NAMES}
        E, _ = SW.model({n: crop(f, (1, 3)) for n, f in F.items()})      # boundary directions, as in shape_window
        Qc = E[:, SW.K:].T.astype(np.float32)                             # (12, 16): the fibre-only space
        for rho in rhos:
            key = f"{wn}_rho{rho:g}"; res[key] = {"window": wn, "rho": rho, "scores": {}, "cp_rel_err": {}}
            for n in NAMES:
                Z = np.tensordot(Qc, F[n], axes=(1, 0))
                e, fit, _ = readouts(crop(k_slice(Z, rho), (0, 1)))
                res[key]["scores"][n] = {k: score(v) for k, v in e.items()}
                res[key]["cp_rel_err"][n] = fit
                s = res[key]["scores"][n]
                print(f"{key:14s} {n:4s} <0.5mm " + "  ".join(f"{k} {v['within_0.5mm']:4.1f}" for k, v in s.items())
                      + f" | deep eigen {s['eigen']['deep_3-8mm']:.1f} | cp err {fit}", flush=True)
            OUT.write_text(json.dumps(res, indent=1))
        print(wn, f"{time.time() - t0:.0f} s", flush=True)
        del F


def maps(wn="w64", rho=16.0):
    """Full-slice read-outs (eigen, TCA R 3) for one window and rho, for the map figure."""
    w = None if wn == "full" else int(wn[1:])
    F = {n: feats(n, w) for n in NAMES}
    E, _ = SW.model({n: crop(f, (1, 3)) for n, f in F.items()}); Qc = E[:, SW.K:].T.astype(np.float32)
    out = {}
    for n in NAMES:
        e, fit, _ = readouts(crop(k_slice(np.tensordot(Qc, F[n], axes=(1, 0)), rho), (0, 1)), np.ones(MASK.shape, bool), (3,))
        out.update({f"{n}_{k}": v for k, v in e.items()}); print(n, "cp err", fit, flush=True)
    np.savez_compressed(ST.OUT / f"shape_tca_maps_{wn}_rho{rho:g}{tag()}.npz", **out)


def tag():
    return f"_pad{PAD}" if PAD else ""


def volume(n):
    """Intensity of phantom n on the scored grid (the grown phantom's crop when PAD)."""
    if not PAD:
        return ST.volume(n)
    import nibabel as nib
    return crop(np.asarray(nib.load(ST.SP.PAD_ROOT / f"{n}.nii.gz").dataobj).astype(np.float32), (0, 2))


SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # palette blue 100 -> 700
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
METHODS = (("eigen", "standard\n(eigen e3)"), ("tca_recon_R3", "TCA R3\nCP only"),
           ("tca_oracle_R3", "TCA R3\nnearest dropped,\nno gate*"), ("tca_oracle_g20_R3", "TCA R3\ndropped if\n< 20° from n*"),
           ("tca_oracle_g30_R3", "TCA R3\ndropped if\n< 30° from n*"), ("tca_oracle_g20_R4", "TCA R4\ndropped if\n< 20° from n*"))


def fig(wn="w64", rho=16.0):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import patheffects as pe
    import edge_aware_phantom_fig as EPF
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ); cmap.set_bad(GRID)
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": GRID})
    res = json.loads(OUT.read_text())
    keys = [k for k in res if k != "w32_rho6"]
    keys = ["w32_rho6"] + keys if "w32_rho6" in res else keys
    lab = [f"w{res[k]['window'][1:]} ρ{res[k]['rho']:g}" if res[k]["window"] != "full" else f"whole ρ{res[k]['rho']:g}"
           for k in keys]
    lab = [l + f"  ({int(res[k]['window'][1:]) / res[k]['rho']:.0f}ρ)" if res[k]["window"] != "full" else l
           for l, k in zip(lab, keys)]
    # 1. summary: heat tables, rows = window / rho, columns = read-out
    fig, axs = plt.subplots(2, 3, figsize=(19, 12.5), dpi=110)
    for r, (sc, sname) in enumerate((("within_0.5mm", "within 0.5 mm of the interface"), ("deep_3-8mm", "3-8 mm from it"))):
        for c, n in enumerate(NAMES):
            ax = axs[r, c]
            A = np.array([[res[k]["scores"][n][m][sc] for m, _ in METHODS] for k in keys])
            ax.imshow(A, cmap=cmap, vmin=0, vmax=90, aspect="auto")
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    ax.text(j, i, f"{A[i, j]:.0f}°", ha="center", va="center", fontsize=10,
                            color="white" if A[i, j] > 40 else INK, fontweight="bold" if A[i, j] < 12 else "normal")
            ax.set_xticks(range(len(METHODS))); ax.set_xticklabels([t for _, t in METHODS] if r == 1 else [], fontsize=8.5)
            ax.set_yticks(range(len(keys))); ax.set_yticklabels(lab if c == 0 else [], fontsize=9.5)
            ax.set_xticks(np.arange(len(METHODS) + 1) - 0.5, minor=True); ax.set_yticks(np.arange(len(keys) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
            ref = {"D0": "no step (control)", "D15": "step 15; gradient excision 9°, BRIDGE now 59°", "D30": "step 30"}[n]
            ax.set_title(f"{n}: {ref}\nmedian angle to the true fibres, {sname}", fontsize=10.5, loc="left")
    fig.suptitle("PROTOTYPE. Fibre orientation from the fibre-only shape tensor: eigenvectors vs tensor component analysis "
                 "(CP), per anatomix window and integration ρ (voxels, 0.1 mm).\n* uses the TRUE interface normal n to pick the "
                 "step component (an oracle upper bound). A dropped component is removed from the exact tensor (nothing dropped = "
                 "standard). Light = close to the truth; bold = under 12°."
                 + (f"\nPhantoms grown by {PAD * 0.1:g} mm on each x and z face; orientations cropped to the original 20 x 20 mm "
                    "before scoring." if PAD else ""), fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(ST.FIGS / f"shape_tca_summary{tag()}.png"); plt.close(fig)
    # 2. maps for one configuration
    M = np.load(ST.OUT / f"shape_tca_maps_{wn}_rho{rho:g}{tag()}.npz")
    cols = (("eigen", "standard tensor (eigen e3)"), ("tca_oracle_R3", "TCA R3: z-nearest component dropped,\nno gate (old)"),
            ("tca_oracle_g20_R3", "TCA R3: dropped only if within 20°\nof the normal (step-like)"))
    fig, axs = plt.subplots(3, 4, figsize=(20, 15.5), dpi=100)
    for r, n in enumerate(NAMES):
        tru, draw, _ = EPF.truth_and_bounds(n)
        EPF.base_image(axs[r, 0], volume(n)[:, ST.JY, :], draw)
        axs[r, 0].set_ylabel(n, fontsize=14, rotation=0, labelpad=25)
        if r == 0:
            axs[r, 0].set_title("intensity, x-z (mid y); cyan = interface,\ndotted = 90 / 60 / 30° fibre bands", fontsize=10.5)
        for c, (m, t) in enumerate(cols, 1):
            ax = axs[r, c]; e = M[f"{n}_{m}"]; a = EPF.angle(e, tru)
            a = np.where((XD >= 1) & (XD < 19), a, np.nan)                       # the scored x range
            im = ax.imshow(np.ma.masked_invalid(a.T), cmap=cmap, vmin=0, vmax=90, origin="lower", extent=EPF.EXT)
            xi, zi = np.meshgrid(np.arange(3, SP_NX, 7), np.arange(3, SP_NZ, 7), indexing="ij")
            keep = (XD[xi, zi] >= 1) & (XD[xi, zi] < 19); xi, zi = xi[keep], zi[keep]
            ee = e[xi, zi]; inpl = np.hypot(ee[..., 0], ee[..., 2]); d = ee[..., [0, 2]] / np.maximum(inpl[..., None], 1e-9)
            L = 0.55 * inpl; cx, cz = (xi + 0.5) * EPF.VOX, (zi + 0.5) * EPF.VOX
            for xs, zs in zip(np.stack([cx - d[..., 0] * L / 2, cx + d[..., 0] * L / 2], -1).reshape(-1, 2),
                              np.stack([cz - d[..., 1] * L / 2, cz + d[..., 1] * L / 2], -1).reshape(-1, 2)):
                ax.plot(xs, zs, color=INK, lw=1.1, solid_capstyle="round",
                        path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
            draw(ax); ax.set_xlim(EPF.EXT[:2]); ax.set_ylim(EPF.EXT[2:]); ax.set_xticks([]); ax.set_yticks([])
            s = score(e)                                                          # of this map (its own CP fit)
            EPF.label(ax, f"median error: {s['within_0.5mm']:.0f}° within 0.5 mm, {s['deep_3-8mm']:.0f}° at 3-8 mm")
            if r == 0:
                ax.set_title(t, fontsize=11)
    cb = fig.colorbar(im, ax=axs[:, 1:], fraction=0.02, pad=0.01); cb.set_label("angle to the true fibre direction (deg)")
    win = "whole block" if wn == "full" else f"window {wn[1:]} vox ({int(wn[1:]) * 0.1:g} mm)"
    fig.suptitle(f"PROTOTYPE. Fibre direction from the fibre-only shape tensor, {win}, ρ {rho:g} vox ({rho * 0.1:g} mm). "
                 "Sticks = estimated fibre direction (in-plane part; short = out of plane); colour = its error.\n"
                 "The truth: fibres cross the interface at 90° (left band), 60° and 30° (right band). The oracle drop uses the "
                 "true interface normal n to choose the step component. Grey: outside the scored x range."
                 + (f"\nPhantoms grown by {PAD * 0.1:g} mm on each x and z face; orientations computed on the grown block, then "
                    "cropped to the original 20 x 20 mm before scoring." if PAD else ""),
                 fontsize=12, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.9)
    fig.savefig(ST.FIGS / f"shape_tca_maps{tag()}.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", ST.FIGS / f"shape_tca_summary{tag()}.png", ST.FIGS / f"shape_tca_maps{tag()}.png")


SP_NX, SP_NZ = ST.SP.NX, ST.SP.NZ


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--pad" in a:
        a.remove("--pad"); PAD = ST.SP.D_PAD; OUT = ST.OUT / f"shape_tca_pad{PAD}.json"
    if a and a[0] == "feats":
        for wn in a[1].split(","):
            for n in NAMES:
                t0 = time.time(); feats(n, None if wn == "full" else int(wn[1:])); print(wn, n, f"{time.time() - t0:.0f} s", flush=True)
    elif a and a[0] == "split":
        split_main(*([a[1].split(",")] if len(a) > 1 else []), *([tuple(float(x) for x in a[2].split(","))] if len(a) > 2 else []))
    elif a and a[0] == "maps":
        maps(a[1], float(a[2]))
    elif a and a[0] == "fig":
        fig(*(a[1:2] + [float(a[2])] if len(a) > 2 else []))
    else:
        main(*([a[0].split(",")] if a else []), *([tuple(float(x) for x in a[1].split(","))] if len(a) > 1 else []))
