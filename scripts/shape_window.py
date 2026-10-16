"""PROTOTYPE: the boundary / fibre separation of anatomix shape space (shape_separate.py) against the anatomix inference
window, i.e. how much context the network sees. Tract-interface phantoms D0 / D15 / D30 (0.1 mm voxels).

Windows 8, 16, 32, 64, 96, 128 voxels (0.8-12.8 mm; Gaussian-blended at 0.75 overlap) and the whole block in one pass
(232 x 160 x 232 padded, ~23 mm); every run averaged over the 8 one-voxel grid phases (anatomix_clean). The UNet's
smallest input is 32^3 (anatomix_shape.NET_MIN), so each 8^3 / 16^3 tile is mirror-tiled up to 32^3 and the centre
cropped back (anatomix_shape._PadTile): the network sees only the tile's own content, but beyond the tile it sees a
mirror image of the tile rather than nothing. Per window:
  boundary directions   principal components of f_D15 - f_D0 within 3 mm of the interface (as rmE in shape_separate)
  boundary RGB          a voxel's coordinates on the top 3 boundary directions (one colour scale for all phantoms)
  fibre RGB             PCA (shared over D0 / D15 / D30) of the features with the top 4 boundary directions removed
  overlap               share of the step's change in the top 4 boundary directions, and share of D0's own variance
                        in the same directions (the bar chart of shape_space_separation.png, summed): a clean
                        separation would be high / low
  reach                 distance from the interface where mean |f_Dstep - f_D0| falls to 1.5 x the far floor (mm)
  angle                 fibre-component shape tensor (rho 6 vox), median angle to the true fibres within 0.5 mm

    python shape_window.py run [--all]  (shapetensors env; caches features in bridge_wmcov/shape_tensor/win/; windows
                                         already in shape_window.json are kept unless --all)
    python shape_window.py fig

Caches: 32 and up were written as float16 (their boundary RGB bands; their JSON metrics came from float32 before
caching); 8 and 16, and any cache written from now on, are float32.
"""
import json, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import edge_aware_phantom_fig as EPF
import shape_separate as SS
import shape_tensor as ST
import surface_phantoms as SP

WIN = (8, 16, 32, 64, 96, 128, None)
NAMES = ("D0", "D15", "D30")
CACHE = ST.OUT / "win"
FIGS = ST.FIGS
K = 4
OVERLAP = {8: 0.5}       # window -> overlap where not 0.75. w8 at 0.75 is ~11 h on an A6000 (every 8^3 tile is a 32^3 pass)


def wname(w):
    return "full" if w is None else f"w{w}"


def mirrored(w):
    return w is not None and w < AS.NET_MIN


def wlabel(w):
    return "whole block (23 mm)" if w is None else f"{w} vox ({w * SP.VOX:g} mm)" + (", mirror-tiled" if mirrored(w) else "")


def feats(name, w):
    p = CACHE / f"{name}_{wname(w)}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    kw = {} if w is None else {"window": w, "overlap": OVERLAP.get(w, 0.75), "sw_batch": 512 if mirrored(w) else 8}
    F = AS.anatomix_clean(ST.volume(name), SP.VOX, **kw)
    CACHE.mkdir(parents=True, exist_ok=True); np.save(p, F.astype(np.float32))
    return F


def model(F):
    """Boundary directions (columns, by change energy) from D15 - D0, and the overlap numbers."""
    dF = SS.core(F["D15"]) - SS.core(F["D0"]); dz = SS.depth()[SS.B:-SS.B]
    near = np.broadcast_to(dz[None, None, :] < SS.NEAR, dF.shape[1:])
    Xn = dF[:, near]; Cn = Xn @ Xn.T / Xn.shape[1]
    we, E = np.linalg.eigh(Cn); o = np.argsort(we)[::-1]; we, E = we[o] / we.sum(), E[:, o]
    X0 = SS.core(F["D0"]).reshape(16, -1); X0 = X0 - X0.mean(1, keepdims=True)
    d0 = ((E.T @ X0) ** 2).mean(1); d0 /= d0.sum()
    dF30 = SS.core(F["D30"]) - SS.core(F["D0"]); X30 = dF30[:, near]
    e30 = ((E.T @ X30) ** 2).mean(1); e30 /= e30.sum()
    return E, {"step_change_in_top4": float(we[:K].sum()), "d0_variance_in_top4": float(d0[:K].sum()),
               "d30_change_in_top4_d15dirs": float(e30[:K].sum()), "step_share": we.tolist(), "d0_share": d0.tolist()}


def reach(F, name):
    z = (np.arange(SP.NZ) + 0.5) * SP.VOX - SP.D_PLANE_Z
    ys = slice(ST.JY - 20, ST.JY + 21); xs = slice(10, SP.NX - 10)
    prof = np.linalg.norm(F[name][:, xs, ys, :] - F["D0"][:, xs, ys, :], axis=0).mean((0, 1))
    floor = np.median(prof[(np.abs(z) > 6) & (np.abs(z) < 8.5)])               # far from the interface, clear of the faces
    above = np.abs(z)[(prof > 1.5 * floor) & (np.abs(z) < 8.5)]
    return float(above.max()) if len(above) else 0.0, prof, z


def angle(F, name, P):
    import tensor_variants as TV
    from scipy import ndimage as ndi
    PF = np.tensordot(P, F[name], axes=(1, 0)); rho = 6.0; m = int(np.ceil(2 * rho)) + 4
    T0 = ST.t0_shape(PF[:, :, ST.JY - m:ST.JY + m + 1, :])
    S = [ndi.gaussian_filter(t, rho, mode="nearest", truncate=2.0)[:, m, :] for t in T0]
    Mx = np.empty((SP.NX, SP.NZ, 3, 3), np.float32)
    for k, (i, j) in enumerate(TV.PAIRS):
        Mx[..., i, j] = Mx[..., j, i] = S[k]
    r, _ = ST.score_e3(np.linalg.eigh(Mx)[1][..., :, 0])
    return r


def run(redo=False):
    p = ST.OUT / "shape_window.json"
    old = {} if redo or not p.exists() else json.loads(p.read_text())
    res = {}
    for w in WIN:
        if wname(w) in old:
            res[wname(w)] = old[wname(w)]; print(wname(w), "kept from", p.name, flush=True)
            continue
        F = {n: feats(n, w) for n in NAMES}
        E, ov = model(F)
        Q, _ = np.linalg.qr(E[:, :K]); P = np.eye(16, dtype=np.float32) - (Q @ Q.T).astype(np.float32)
        r = {"overlap": ov, "reach_mm": {n: reach(F, n)[0] for n in ("D15", "D30")},
             "angle": {n: angle(F, n, P) for n in NAMES},
             "sliding_overlap": None if w is None else OVERLAP.get(w, 0.75), "mirror_tiled": mirrored(w)}
        res[wname(w)] = r
        print(wname(w), {k: round(v, 3) for k, v in ov.items() if not isinstance(v, list)}, "reach", r["reach_mm"],
              "angle <0.5mm", {n: r["angle"][n]["within_0.5mm"] for n in NAMES}, "D0 deep", r["angle"]["D0"]["deep_3-8mm"], flush=True)
        p.write_text(json.dumps({**old, **res}, indent=1))
    p.write_text(json.dumps(res, indent=1))


def fig():
    res = json.loads((ST.OUT / "shape_window.json").read_text())
    tru, draw, _ = EPF.truth_and_bounds("D15")
    cols = [("D15", "boundary"), ("D15", "fibre"), ("D30", "boundary"), ("D30", "fibre"), ("D0", "fibre")]
    fig, axs = plt.subplots(len(WIN), len(cols), figsize=(4.3 * len(cols), 4.4 * len(WIN)), dpi=100)
    for r, w in enumerate(WIN):
        F = {n: feats(n, w) for n in NAMES}
        E, _ = model(F)
        Q, _ = np.linalg.qr(E[:, :K]); P = np.eye(16, dtype=np.float32) - (Q @ Q.T).astype(np.float32)
        mu0 = SS.core(F["D0"]).reshape(16, -1).mean(1, keepdims=True)
        # one colour scale per window for each component, pooled over the three phantoms
        bsl = {n: (E[:, :3].T @ (F[n][:, :, ST.JY, :].reshape(16, -1) - mu0)) for n in NAMES}
        pool = np.concatenate(list(bsl.values()), 1); blo, bhi = np.percentile(pool, 1, axis=1), np.percentile(pool, 99, axis=1)
        pf = {n: P @ F[n][:, :, ST.JY, :].reshape(16, -1) for n in NAMES}
        pp = np.concatenate(list(pf.values()), 1); pm = pp.mean(1, keepdims=True)
        ww, V = np.linalg.eigh(np.cov(pp - pm)); V = V[:, np.argsort(ww)[::-1][:3]].T
        q = V @ (pp - pm); flo, fhi = np.percentile(q, 1, axis=1), np.percentile(q, 99, axis=1)
        for c, (n, kind) in enumerate(cols):
            if kind == "boundary":
                img = np.clip((bsl[n] - blo[:, None]) / (bhi - blo)[:, None], 0, 1)
            else:
                img = np.clip((V @ (pf[n] - pm) - flo[:, None]) / (fhi - flo)[:, None], 0, 1)
            ax = axs[r, c]; ax.imshow(np.transpose(img.T.reshape(SP.NX, SP.NZ, 3), (1, 0, 2)), origin="lower", extent=EPF.EXT)
            draw(ax); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{n}: {kind} component RGB", fontsize=11)
            if c == 0:
                ov = res[wname(w)]["overlap"]
                ax.set_ylabel(f"window {wlabel(w)}\n"
                              f"step change in top 4: {ov['step_change_in_top4']:.0%}\nD0 fibre variance there: "
                              f"{ov['d0_variance_in_top4']:.0%}", fontsize=10)
            a = res[wname(w)]["angle"][n]
            if kind == "fibre":
                EPF.label(ax, f"fibre tensor: {a['within_0.5mm']:.0f} deg at the interface")
            elif n in ("D15", "D30"):
                EPF.label(ax, f"step reaches {res[wname(w)]['reach_mm'][n]:.1f} mm")
    small = [w for w in WIN if mirrored(w)]
    fig.suptitle("PROTOTYPE. Boundary and fibre components of anatomix shape space against the anatomix window (context). "
                 "Boundary RGB = top-3 step directions (learned D15 - D0 per window); fibre RGB = the rest. x-z, mid y; "
                 f"cyan = interface.\nThe {' and '.join(map(str, small))}-voxel windows are below the UNet's {AS.NET_MIN}^3 "
                 f"minimum input: each tile is mirror-tiled up to {AS.NET_MIN}^3, so beyond the tile the network sees a mirror "
                 "image of the tile itself, not real context. Rows 32 and up come from float16 caches (banding in the "
                 "boundary RGB); 8 and 16 from float32.", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(FIGS / "shape_window_rgb.png"); plt.close(fig)
    # metrics against the window
    x = np.arange(len(WIN))
    lab = [f"{w}\n({w * SP.VOX:g} mm)" + ("\nmirror" if mirrored(w) else "") if w else "whole\n(23 mm)" for w in WIN]
    fig, axs = plt.subplots(1, 4, figsize=(24, 5.8), dpi=100)
    ax = axs[0]
    ax.plot(x, [res[wname(w)]["overlap"]["step_change_in_top4"] for w in WIN], "-o", color="C0", label="step change (D15) in the top 4 step directions")
    ax.plot(x, [res[wname(w)]["overlap"]["d0_variance_in_top4"] for w in WIN], "-o", color="C1", label="D0 fibre variance in the same 4 directions")
    ax.plot(x, [res[wname(w)]["overlap"]["d30_change_in_top4_d15dirs"] for w in WIN], "--o", color="C3", label="D30's step change in D15's 4 directions")
    ax.set_title("separability: high blue + low orange = boundary and fibre in different directions", fontsize=10)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[1]
    for n, cc in (("D15", "C1"), ("D30", "C3")):
        ax.plot(x, [res[wname(w)]["reach_mm"][n] for w in WIN], "-o", color=cc, label=n)
    ax.set_ylabel("mm"); ax.set_title("how far the step reaches into shape space", fontsize=10); ax.legend(); ax.grid(alpha=0.3)
    ax = axs[2]
    for n, cc in (("D0", "C2"), ("D15", "C1"), ("D30", "C3")):
        ax.plot(x, [res[wname(w)]["angle"][n]["within_0.5mm"] for w in WIN], "-o", color=cc, label=f"{n}, within 0.5 mm")
    ax.plot(x, [res[wname(w)]["angle"]["D0"]["deep_3-8mm"] for w in WIN], ":o", color="C2", label="D0, 3-8 mm")
    ax.axhline(59.1, color="grey", ls=":", label="BRIDGE now, D15"); ax.axhline(8.8, color="k", ls="--", label="gradient excision, D15")
    ax.set_ylim(0, 90); ax.set_ylabel("median angle to the true fibres (deg)"); ax.set_title("fibre-component shape tensor (rho 0.6 mm)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axs[3]
    for w, cc in ((WIN[0], "C0"), (32, "C2"), (None, "C3")):
        ov = res[wname(w)]["overlap"]; k = np.arange(1, 9)
        ax.plot(k, ov["step_share"][:8], "-o", color=cc, label=f"step change, {wname(w)}")
        ax.plot(k, ov["d0_share"][:8], "--s", color=cc, label=f"D0 fibre variance, {wname(w)}")
    ax.set_xlabel("step direction"); ax.set_ylabel("share"); ax.set_title("the overlap bar chart: smallest, 32 and whole window", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for a in axs[:3]:
        a.set_xticks(x); a.set_xticklabels(lab); a.set_xlabel("anatomix window (voxels)")
        if small:
            a.axvspan(-0.5, len(small) - 0.5, color="0.85", zorder=0)
    fig.suptitle("PROTOTYPE. Boundary / fibre separation against the anatomix window. Grey band: windows below the UNet's "
                 f"{AS.NET_MIN}^3 minimum, mirror-tiled up to it,\nso beyond the tile the network sees a mirror image of the "
                 "tile itself, not real context", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIGS / "shape_window_metrics.png"); plt.close(fig)
    print("wrote", FIGS)


if __name__ == "__main__":
    run("--all" in sys.argv) if sys.argv[1] == "run" else fig()
