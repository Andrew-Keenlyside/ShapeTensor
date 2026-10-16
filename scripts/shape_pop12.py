"""PROTOTYPE: two shape-derived ways to output N fibre populations per site, compared on rod and line phantoms.

Method 1 (joint; linked-factor TCA = rank-(2, 2, 1) block term decomposition): per site (window rho), the array
  X[p, c, i] (space x 16 shape channels x gradient axis) ~ sum_k b_k[c] M_k[p, i],  M_k = A_k D_k^T (P x 2)(2 x 3):
  population k = ONE shape signature b_k shared by its two pieces, and a rank-2 space x axis block (its gradients span
  the plane perpendicular to its fibre). Fitted by ALS for N = 1, 2, 3. Population k's own tensor T_k = M_k^T M_k;
  fibre u_k = its least-energy direction; fraction ~ trace(T_k).
Method 2 (unmixing, then a tensor per population): per image, each voxel's 16-D feature = a convex mix of K endmember
  signatures (abundances >= 0, summing to 1; signatures signed), fitted by alternating projected gradient / least
  squares for K = 1..4; each abundance map's structure tensor (sigma 1, rho) gives population k's tensor T_k per site;
  fraction = the window's mean abundance.
Shared output rules (dMRI-style): N (method 1) / K (method 2) = add a population only if the relative fit error falls by
  more than DELTA; per site keep populations with fraction >= F_MIN and ring-ness (l2 - l3) / l1 >= RING_MIN (a single
  fibre population's gradients fill one plane: l1 ~ l2 >> l3; as an FA threshold); merge populations within SEP deg.

    python shape_pop12.py     (shapetensors env, one GPU)
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import shape_cross_binary as B
import shape_tca as T
import shape_tensor as ST
import tensor_variants as TV

N, VOX, C0, W, JY, RHO = B.N, B.VOX, B.C0, B.W, B.JY, B.RHO
DEV = T.DEV
DELTA, F_MIN, RING_MIN, SEP, NMAX, KMAX = 0.15, 0.10, 0.5, 20.0, 3, 4
MB = 20                                                    # margin around the scored region for method 2's block
ROWS = (("same rods, single 90°", "rods", 2.0, ((90.0, 1.5),)),
        ("same rods, 90° + 30°", "rods", 2.0, ((90.0, 1.5), (30.0, 1.5))),
        ("same rods, 60° + 120°", "rods", 2.0, ((60.0, 1.5), (120.0, 1.5))),
        ("thin + thick rods, 90° + 30°", "rods", 2.0, ((90.0, 1.0), (30.0, 2.5))),
        ("thin + thick rods, 60° + 120°", "rods", 2.0, ((60.0, 1.0), (120.0, 2.5))),
        ("thin + thick lines, noise 8, 60° + 120°", "lines", 8.0, ((60.0, 0.6), (120.0, 1.4))))


# ---------------------------------------------------------------- phantoms
def rods(th, r, phase):
    x, y, z = np.meshgrid(*(np.arange(N, dtype=np.float32) + 0.5,) * 3, indexing="ij")
    u = B.unit(th); nrm = np.array([-u[2], 0.0, u[0]])
    dn = ((x * nrm[0] + z * nrm[2] + phase) % B.S) - B.S / 2; dy = (y % B.S) - B.S / 2
    return np.clip(r + 0.5 - np.sqrt(dn ** 2 + dy ** 2), 0, 1)


def make(kind, noise, pops, rng):
    if kind == "rods":
        v = B.BG - (B.BG - B.ROD) * np.max(np.stack([rods(t, r, 3.3 * k) for k, (t, r) in enumerate(pops)]), 0)
    else:
        import shape_cross_texture as CT
        v = CT.BASE - CT.AMP * sum(np.clip(lines_sig(t, sg, rng, 1.0 / len(pops)), 0, 1.5) for t, sg in pops)
    return np.clip(np.rint(v + rng.normal(0, noise, v.shape)), 0, 255).astype(np.float32)


def lines_sig(theta, sig, rng, frac):
    """shape_cross_texture.lines with a chosen blur (fibre thickness)."""
    import shape_cross_texture as CT
    a = np.radians(theta); d = np.array([np.cos(a), np.sin(a)]); nrm = np.array([-d[1], d[0]]); L = N * VOX
    pr = np.array([[0, 0], [L, 0], [0, L], [L, L]]) @ nrm; lo, hi = pr.min(), pr.max()
    n = int(CT.DENS * frac * (hi - lo) * L); s = np.arange(-25, 25, 0.05); cen = np.array([L / 2, L / 2])
    dens = np.zeros((N, N, N), np.float32)
    for _ in range(n):
        p = (rng.uniform(lo, hi) * nrm + (cen @ d) * d)[None] + s[:, None] * d[None]
        k = (p[:, 0] >= 0) & (p[:, 0] < L) & (p[:, 1] >= 0) & (p[:, 1] < L)
        ix = (p[k, 0] / VOX).astype(int); iz = (p[k, 1] / VOX).astype(int)
        np.add.at(dens, (ix, np.full(ix.shape, int(rng.uniform(0, L) / VOX)), iz), 1.0)
    dens = ndi.gaussian_filter(dens, sig)
    return dens / np.percentile(dens[dens > 0], 99)


# ---------------------------------------------------------------- shared output rules
def tensor_stats(Tk):
    """Tk (..., 3, 3) -> fibre u (least-energy direction), ring-ness (l2 - l3) / l1, trace."""
    w, V = torch.linalg.eigh(Tk)                                                       # ascending
    l3, l2, l1 = w[..., 0], w[..., 1], w[..., 2]
    return V[..., :, 0], (l2 - l3) / l1.clamp_min(1e-30), w.sum(-1)


def select(U, ring, frac):
    """Per site: populations (U (S, n, 3), ring (S, n), frac (S, n)) -> list of [(u, fraction)] after the rules."""
    U, ring, frac = (x.cpu().numpy() if torch.is_tensor(x) else x for x in (U, ring, frac))
    out = []
    for s in range(len(U)):
        keep = []
        for q in np.argsort(frac[s])[::-1]:
            if frac[s, q] < F_MIN or ring[s, q] < RING_MIN:
                continue
            if all(abs(U[s, q] @ U[s, j]) < np.cos(np.radians(SEP)) for j in keep):
                keep.append(q)
        f = frac[s, keep]; out.append([(U[s, q], float(x / max(f.sum(), 1e-12))) for q, x in zip(keep, f)])
    return out


def elbow(errs):
    n = 1
    while n < len(errs) and errs[n] < errs[n - 1] * (1 - DELTA):
        n += 1
    return n


# ---------------------------------------------------------------- method 1: linked-factor TCA per site
def solve(Gm, M):
    eye = torch.eye(Gm.shape[-1], device=DEV)
    return torch.linalg.solve(Gm + 1e-6 * Gm.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye, M.transpose(1, 2)).transpose(1, 2)


def btd_als(Y, Np, starts=3, iters=200, seed=0):
    n, P, C, I = Y.shape; R = 2 * Np; tie = torch.arange(R, device=DEV) // 2
    g = torch.Generator(device=DEV).manual_seed(seed)
    Xc = Y.permute(0, 2, 1, 3).reshape(n, C, P * I); yy = (Y ** 2).sum((1, 2, 3))
    best = torch.full((n,), float("inf"), device=DEV); keep = None
    for _ in range(starts):
        Bb = torch.randn(n, C, Np, device=DEV, generator=g); Bb = Bb / Bb.norm(dim=1, keepdim=True)
        D = torch.randn(n, I, R, device=DEV, generator=g); D = D / D.norm(dim=1, keepdim=True)
        for _ in range(iters):
            Be = Bb[:, :, tie]
            A = solve((Be.mT @ Be) * (D.mT @ D), torch.einsum("npci,ncr,nir->npr", Y, Be, D))
            D = solve((A.mT @ A) * (Be.mT @ Be), torch.einsum("npci,npr,ncr->nir", Y, A, Be))
            M = torch.einsum("npr,nir->nrpi", A, D).reshape(n, Np, 2, P * I).sum(2)           # (n, Np, P I)
            Bb = solve(M @ M.mT, Xc @ M.mT)                                                  # Xc ~ Bb M
            nb = Bb.norm(dim=1, keepdim=True).clamp_min(1e-12); Bb = Bb / nb; A = A * nb[:, :, tie]
            nd = D.norm(dim=1, keepdim=True).clamp_min(1e-12); D = D / nd; A = A * nd
        err = ((Y - torch.einsum("npr,ncr,nir->npci", A, Bb[:, :, tie], D)) ** 2).sum((1, 2, 3)) / yy.clamp_min(1e-30)
        Mk = torch.einsum("npr,nir->nrpi", A, D).reshape(n, Np, 2, P, I).sum(2)               # (n, Np, P, I)
        take = err < best; best = torch.where(take, err, best)
        cur = (Mk, Bb)
        keep = cur if keep is None else tuple(torch.where(take.view(-1, *([1] * (a.dim() - 1))), a, b) for a, b in zip(cur, keep))
    return keep[0], keep[1], best


def method1(F):
    K = T.k_slice(F, RHO, JY)[C0:C0 + W, C0:C0 + W]
    Kf = K.reshape(-1, 48, 48).double(); Kf = Kf / Kf.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]
    eye = torch.eye(48, device=DEV, dtype=Kf.dtype)
    L, info = torch.linalg.cholesky_ex(Kf + 1e-9 * eye)
    if (info > 0).any():
        L[info > 0] = torch.linalg.cholesky(Kf[info > 0] + 1e-6 * eye)
    Y = L.mT.float().reshape(-1, 48, 16, 3)
    fits = {Np: btd_als(Y, Np) for Np in range(1, NMAX + 1)}
    errs = torch.stack([fits[k][2] for k in range(1, NMAX + 1)], 1).cpu().numpy()
    nsel = np.array([elbow(e) for e in errs])
    U = torch.zeros(len(Y), NMAX, 3, device=DEV); ring = torch.zeros(len(Y), NMAX, device=DEV); frac = torch.zeros(len(Y), NMAX, device=DEV)
    for Np in range(1, NMAX + 1):
        Mk = fits[Np][0]; Tk = Mk.mT @ Mk; u, rg, tr = tensor_stats(Tk.double())
        sel = torch.from_numpy(nsel == Np).to(DEV)
        U[sel, :Np] = u[sel].float(); ring[sel, :Np] = rg[sel].float(); frac[sel, :Np] = (tr / tr.sum(1, keepdim=True))[sel].float()
    return select(U, ring, frac), {"N by elbow": np.bincount(nsel, minlength=NMAX + 1)[1:] / len(nsel), "rel err": errs.mean(0).tolist()}


# ---------------------------------------------------------------- method 2: unmixing, then a tensor per abundance map
def proj_simplex(V):
    """Rows of V (n, K) onto the probability simplex."""
    U, _ = torch.sort(V, dim=1, descending=True); css = U.cumsum(1) - 1
    k = torch.arange(1, V.shape[1] + 1, device=V.device, dtype=V.dtype)
    rho_ = ((U - css / k) > 0).sum(1, keepdim=True)
    th = css.gather(1, rho_ - 1) / rho_.to(V.dtype)
    return (V - th).clamp_min(0)


def unmix(X, K, iters=400):
    """X (n, C) -> endmembers S (K, C), abundances (n, K), relative error vs the centred variance."""
    Xt = torch.from_numpy(X).to(DEV); mu = Xt.mean(0, keepdim=True)
    idx = [int(((Xt - mu) ** 2).sum(1).argmax())]
    for _ in range(1, K):                                   # furthest point from the affine span of the chosen ones
        S0 = Xt[idx]; Q = torch.linalg.qr((S0[1:] - S0[:1]).T)[0] if len(idx) > 1 else torch.zeros(X.shape[1], 0, device=DEV)
        R_ = (Xt - S0[:1]); R_ = R_ - (R_ @ Q) @ Q.T; idx.append(int((R_ ** 2).sum(1).argmax()))
    S = Xt[idx].clone(); A = torch.full((len(Xt), K), 1.0 / K, device=DEV)
    for _ in range(iters):
        step = 1.0 / torch.linalg.matrix_norm(S @ S.T, 2).clamp_min(1e-12)
        A = proj_simplex(A - step * (A @ S - Xt) @ S.T)
        S = torch.linalg.solve(A.T @ A + 1e-6 * torch.eye(K, device=DEV), A.T @ Xt)
    err = float(((A @ S - Xt) ** 2).sum() / ((Xt - mu) ** 2).sum())
    return S, A, err


def method2(F):
    xs = slice(C0 - MB, C0 + W + MB); m = int(np.ceil(2 * RHO)) + 4; ys = slice(JY - m, JY + m + 1)
    blk = F[:, xs, ys, xs]; sh = blk.shape[1:]; X = blk.reshape(16, -1).T.astype(np.float32)
    fits = {K: unmix(X, K) for K in range(1, KMAX + 1)}
    Ksel = elbow([fits[k][2] for k in range(1, KMAX + 1)])
    S, Aab, _ = fits[Ksel]; ab = Aab.T.reshape(Ksel, *sh).cpu().numpy()
    U = np.zeros((W * W, Ksel, 3)); ring = np.zeros((W * W, Ksel)); frac = np.zeros((W * W, Ksel))
    for k in range(Ksel):
        Tk = T.k_slice(ab[k][None], RHO, m)[MB:MB + W, MB:MB + W].reshape(-1, 3, 3).double()
        u, rg, _ = tensor_stats(Tk); U[:, k] = u.cpu().numpy(); ring[:, k] = rg.cpu().numpy()
        frac[:, k] = ndi.gaussian_filter(ab[k][:, m, :], RHO, mode="nearest", truncate=TV.TR)[MB:MB + W, MB:MB + W].ravel()
    frac = frac / np.maximum(frac.sum(1, keepdims=True), 1e-12)
    return select(U, ring, frac), {"K by elbow": Ksel, "rel err": [fits[k][2] for k in range(1, KMAX + 1)], "abund": ab[:, MB:MB + W, m, MB:MB + W]}


# ---------------------------------------------------------------- scoring and figure
def score(fx, tru):
    nf = np.array([len(x) for x in fx]); r = {"N correct": float(np.mean(nf == len(tru))), "N found": {int(k): float(np.mean(nf == k)) for k in range(0, 4)}}
    if len(tru) == 1:
        r["error main"] = float(np.median([B.ang(x[0][0], tru[0]) for x in fx if x])) if any(fx) else None
    else:
        e2 = np.array([min((B.ang(x[0][0], tru[0]), B.ang(x[1][0], tru[1])), (B.ang(x[0][0], tru[1]), B.ang(x[1][0], tru[0])), key=sum)
                       for x in fx if len(x) >= 2]).reshape(-1, 2)
        r["errors when 2 found"] = [float(np.median(e2[:, 0])), float(np.median(e2[:, 1]))] if len(e2) else None
        r["both within 15° (all sites)"] = float(np.sum((e2 < 15).all(1)) / len(fx)) if len(e2) else 0.0
    return r


def main():
    rng = np.random.default_rng(41); res, keep = {}, {}
    for name, kind, noise, pops in ROWS:
        v = make(kind, noise, pops, rng); tru = np.stack([B.unit(t) for t, _ in pops]); F = AS.anatomix_clean(v, VOX)
        f1, d1 = method1(F); f2, d2 = method2(F)
        res[name] = {"method 1": {**score(f1, tru), "N by elbow": d1["N by elbow"].tolist(), "rel err": d1["rel err"]},
                     "method 2": {**score(f2, tru), "K by elbow": d2["K by elbow"], "rel err": d2["rel err"]}}
        keep[name] = (v, F, f1, f2, d2["abund"], tru)
        for k in ("method 1", "method 2"):
            print(f"{name:42s} {k}: {res[name][k]}", flush=True)
    (ST.OUT / "shape_pop12.json").write_text(json.dumps(res, indent=1))
    figure(keep, res)


def figure(keep, res):
    ext = (C0 * VOX, (C0 + W) * VOX, C0 * VOX, (C0 + W) * VOX)
    xi, zi = np.meshgrid(np.arange(2, W, 5), np.arange(2, W, 5), indexing="ij"); site = (xi * W + zi).ravel()
    cx, cz = (C0 + xi.ravel() + 0.5) * VOX, (C0 + zi.ravel() + 0.5) * VOX
    ncm = ListedColormap(["#ffffff", "#cde2fb", "#86b6ef", "#3987e5"]); pcol = ("#0b0b0b", "#eb6834", "#1baf7a")
    fig, axs = plt.subplots(len(keep), 5, figsize=(24, 4.8 * len(keep) + 1.4), dpi=80)

    def fix_panel(ax, fx, q, ntrue):
        ax.imshow(np.array([len(x) for x in fx]).reshape(W, W).T, cmap=ncm, vmin=-0.5, vmax=3.5, origin="lower", extent=ext, interpolation="nearest")
        for j in range(3):
            segs = []
            for s_, x_, z_ in zip(site, cx, cz):
                if len(fx[s_]) > j:
                    e, fr = fx[s_][j]; inpl = np.hypot(e[0], e[2]); d = np.array([e[0], e[2]]) / max(inpl, 1e-9); h = 0.21 * inpl * (0.4 + 0.6 * fr / fx[s_][0][1])
                    segs.append([[x_ - d[0] * h, z_ - d[1] * h], [x_ + d[0] * h, z_ + d[1] * h]])
            if segs:
                ax.add_collection(LineCollection(segs, colors="white", linewidths=3.2)); ax.add_collection(LineCollection(segs, colors=pcol[j], linewidths=1.6))
        txt = f"N = {ntrue} at {q['N correct']:.0%}; " + (f"error {q['error main']:.0f}°" if q.get("error main") is not None else
              (f"errors {q['errors when 2 found'][0]:.0f}° / {q['errors when 2 found'][1]:.0f}°, both < 15°: {q['both within 15° (all sites)']:.0%}"
               if q.get("errors when 2 found") else "no 2-fibre sites"))
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round", fc="w", alpha=0.9, lw=0))
    for r, (name, (v, F, f1, f2, ab, tru)) in enumerate(keep.items()):
        axs[r, 0].imshow(v[C0:C0 + W, JY, C0:C0 + W].T, cmap="gray", origin="lower", extent=ext); axs[r, 0].set_ylabel(name, fontsize=11)
        X = F[:, C0:C0 + W, JY, C0:C0 + W].reshape(16, -1); X = X - X.mean(1, keepdims=True)
        w, U = np.linalg.eigh(np.cov(X)); P = U[:, np.argsort(w)[::-1][:3]].T @ X
        lo, hi = np.percentile(P, 1, axis=1), np.percentile(P, 99, axis=1)
        axs[r, 1].imshow(np.transpose(np.clip((P - lo[:, None]) / np.maximum(hi - lo, 1e-12)[:, None], 0, 1).T.reshape(W, W, 3), (1, 0, 2)), origin="lower", extent=ext)
        fix_panel(axs[r, 2], f1, res[name]["method 1"], len(tru)); fix_panel(axs[r, 3], f2, res[name]["method 2"], len(tru))
        rgb = np.zeros((W, W, 3)); rgb[..., :min(3, len(ab))] = np.transpose(ab[:3], (1, 2, 0))
        axs[r, 4].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext)
        axs[r, 4].text(0.02, 0.02, f"K = {res[name]['method 2']['K by elbow']} signatures", transform=axs[r, 4].transAxes, fontsize=9,
                       bbox=dict(boxstyle="round", fc="w", alpha=0.9, lw=0))
        for c in range(5):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    for c, t in enumerate(("image (x-z, mid y)", "shape space, RGB = PC 1-3", "method 1: joint\n(one signature + one ring per population)",
                           "method 2: unmix, then a tensor\nper abundance map", "method 2: abundance maps\n(RGB = signatures 1-3)")):
        axs[0, c].set_title(t, fontsize=12)
    fig.suptitle("PROTOTYPE. Shape-derived N populations per site: method 1 (linked-factor TCA: each population one shape signature and one "
                 "ring of gradients) vs method 2 (unmixing into signature abundance maps, a tensor per map).\nShared dMRI-style rules: N by "
                 f"the error elbow (> {DELTA:.0%} better), fraction ≥ {F_MIN:.0%}, ring-ness ≥ {RING_MIN}, merge within {SEP:g}°. Background = N "
                 "output; sticks = fibres (length ~ fraction).", fontsize=12, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(top=0.93, wspace=0.04, hspace=0.1, left=0.08, right=0.99)
    out = ST.FIGS / "shape_pop12.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
