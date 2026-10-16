"""Orientation from gradients, built differently: which constructions follow fibres rather than surfaces?

Every variant uses BRIDGE's derivative scale (sigma 1 voxel) and total integration window (rho 3 voxels, truncate 2).

  base      S = G_rho * (g g^T); e3 (least-change direction)                              BRIDGE as is
  unit      S = G_rho * (g^ g^T), unit gradients: every voxel one vote, |g|^2 weighting gone
  p1        S = G_rho * (g g^T / |g|): |g|^1 weighting
  gm<f>     gradients where the coarse (sigma = rho) gradient exceeds f x its in-mask median are dropped;
            the integration is normalised by the surviving weight (normalised convolution)
  tw<r0>    two-scale, tube-weighted: a fine tensor S0 at rho0; each voxel's S0 / tr(S0) is weighted by its
            tube-ness cp = (L2 - L3) / L1 and integrated over rho' = sqrt(rho^2 - rho0^2); e3 of the result.
            A sheet (step, interface) has L1 >> L2 ~ L3, cp ~ 0, so it carries no weight however strong it is.
  gf<k>     the same, but excising FINE-scale gradient-magnitude outliers (|g| > k x its in-mask median)
  gmf<k>    both tests (a gradient is excised if either flags it)
  twh/voteh the same with a hard Westin gate (cp > cl, threshold-free) instead of the cp weight
  vote<r0>  two-scale fibre-axis vote: F = G_rho' * (cp e3_0 e3_0^T); e1 of F (most-voted axis)

    python tensor_variants.py score [variant ...]
"""
import json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import phantom_shell as PS
import surface_phantoms as SP

OUT = Path("/home/andrew/bridge_wmcov/edge_aware")
SIG, RHO, TR = 1.0, 3.0, 2.0
PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
DEFAULT = ("base", "unit", "p1", "gm5", "tw1", "tw1.5", "vote1", "vote1.5")


def integ(a, r):
    return ndi.gaussian_filter(a, r, mode="nearest", truncate=TR)


def grads(vol):
    return [ndi.gaussian_filter(vol, SIG, order=o, mode="nearest", truncate=TR) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]


def assemble(comp, shape):
    T = np.empty(shape + (3, 3), np.float32)
    for (i, j), c in zip(PAIRS, comp):
        T[..., i, j] = T[..., j, i] = c
    return T


def orientation(vol, variant, mask=None):
    """Unit orientation (..., 3) over the whole block, plus the integrated tensor's eigenvalues (desc)."""
    vol = vol.astype(np.float32)
    g = grads(vol)
    if variant == "base":
        T = assemble([integ(g[i] * g[j], RHO) for i, j in PAIRS], vol.shape)
    elif variant in ("unit", "p1"):
        n = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) + 1e-6
        p = 2.0 if variant == "unit" else 1.0
        T = assemble([integ(g[i] * g[j] / n ** p, RHO) for i, j in PAIRS], vol.shape)
    elif variant.startswith(("gm", "gf")):
        m = np.ones(vol.shape, bool) if mask is None else mask
        k = float(variant.lstrip("gmf"))
        w = np.ones(vol.shape, bool)
        if variant.startswith("gm"):                                          # coarse-scale coherence
            gc = np.sqrt(sum(ndi.gaussian_filter(vol, RHO, order=o, mode="nearest", truncate=TR) ** 2
                             for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))))
            w &= gc <= k * np.median(gc[m])
        if variant.startswith(("gf", "gmf")):                                 # fine-scale magnitude outliers
            gn = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
            w &= gn <= k * np.median(gn[m])
        w = w.astype(np.float32)
        den = np.maximum(integ(w, RHO), 1e-6)
        T = assemble([integ(w * g[i] * g[j], RHO) / den for i, j in PAIRS], vol.shape)
    elif variant.startswith(("tw", "vote")):
        hard = "h" in variant
        r0 = float(variant.lstrip("twvoeh")); r1 = np.sqrt(RHO ** 2 - r0 ** 2)
        S0 = assemble([integ(g[i] * g[j], r0) for i, j in PAIRS], vol.shape)
        w0, v0 = np.linalg.eigh(S0)                                         # ascending: L3, L2, L1
        cp = ((w0[..., 1] - w0[..., 0]) / np.maximum(w0[..., 2], 1e-12)).astype(np.float32)
        if hard:                                                            # Westin: planar (tube) beats linear (sheet)
            cp = (3.0 * w0[..., 1] > w0[..., 2] + 2.0 * w0[..., 0]).astype(np.float32)
        den = np.maximum(integ(cp, r1), 1e-6)
        if variant.startswith("tw"):
            tr = np.maximum(w0.sum(-1), 1e-12)
            T = assemble([integ(cp * S0[..., i, j] / tr, r1) / den for i, j in PAIRS], vol.shape)
        else:
            e = v0[..., :, 0]
            T = assemble([integ(cp * e[..., i] * e[..., j], r1) / den for i, j in PAIRS], vol.shape)
            w, v = np.linalg.eigh(T)
            return v[..., :, 2], w[..., ::-1]
    else:
        raise ValueError(variant)
    w, v = np.linalg.eigh(T)
    return v[..., :, 0], w[..., ::-1]


def angle3(e, t2):
    """Angle (deg) between unit e (x, y, z) and an in-slice truth (tx, tz)."""
    c = np.abs(e[..., 0] * t2[..., 0] + e[..., 2] * t2[..., 1])
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def slab(name, root):
    vol = np.asarray(nib.load(root / f"{name}.nii.gz").dataobj).astype(np.float32)
    jy = SP.NY // 2; H = 20
    return vol[:, jy - H:jy + H + 1, :], H


def score(variants):
    res = {}
    # --- A / B: the gyrus with a fan crossing the boundary
    t = np.load(PS.ROOT / "truth.npz"); sdf = t["sdf"]
    gx, gz = np.gradient(ndi.gaussian_filter(sdf, 5.0)); nrm = np.stack([gx, gz], -1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
    X, Z = np.meshgrid((np.arange(SP.NX) + 0.5) * SP.VOX, (np.arange(SP.NZ) + 0.5) * SP.VOX, indexing="ij")
    truAB = PS.truth_dir(X, Z)
    binsAB = [(-1.0, 0.0), (0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
    # --- C / C0: U-fibres
    tc = np.load(SP.ROOT / "C_truth.npz"); truC, sdfC = tc["tru"], tc["sdf"]
    # --- D: tract interface
    td = np.load(SP.ROOT / "D_truth.npz"); truD, depD, XD = td["tru"], td["depth"], td["X"]
    binsD = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (3.0, 8.0)]
    for variant in variants:
        r = res[variant] = {}
        for name, root in (("A", PS.ROOT), ("B", PS.ROOT), ("C", SP.ROOT), ("C0", SP.ROOT)) + tuple((f"D{d}", SP.ROOT) for d in SP.D_STEPS):
            vol, H = slab(name, root)
            thr = PS.THRESH[name] if name in PS.THRESH else SP.THRESH[name]
            m3 = vol >= thr
            e, lam = orientation(vol, variant, m3)
            e = e[:, H]; m = m3[:, H]
            if name in ("A", "B"):
                inpl = np.hypot(e[..., 0], e[..., 2])
                par = np.abs((e[..., 0] * nrm[..., 0] + e[..., 2] * nrm[..., 1])) < 0.5 * np.maximum(inpl, 1e-9)
                a = angle3(e, truAB)
                r[name] = [{"bin": b, "ang": round(float(np.median(a[s])), 1), "par_inslice": round(float(par[s & (inpl > 0.5)].mean()), 2),
                            "along_y": round(float((inpl[s] <= 0.5).mean()), 2)}
                           for b in binsAB for s in [m & (sdf >= b[0]) & (sdf < b[1])]]
            elif name in ("C", "C0"):
                a = angle3(e, np.nan_to_num(truC))
                r[name] = {k: round(float(np.median(a[s])), 1) for k, s in
                           (("band_0.15-0.5", m & (sdfC >= 0.15) & (sdfC < 0.5)), ("band_0.5-1.2", m & (sdfC >= 0.5) & (sdfC <= 1.2)),
                            ("deep", m & (sdfC > 2.0)))}
            else:
                a = angle3(e, truD)
                r[name] = {f"{ang:.0f}deg": [round(float(np.median(a[s])), 1) for b in binsD
                                             for s in [(XD >= x0 + 1) & (XD < x1 - 1) & (depD >= b[0]) & (depD < b[1])]]
                           for x0, x1, ang in SP.D_BANDS}
        A = r["A"]; B = r["B"]
        print(f"{variant:8s} A ang {[p['ang'] for p in A]} y {[p['along_y'] for p in A]} par {[p['par_inslice'] for p in A]}"
              f" | B ang {[p['ang'] for p in B]}", flush=True)
        print(f"{'':8s} C {r['C']} | C0 {r['C0']}", flush=True)
        for d in SP.D_STEPS:
            print(f"{'':8s} D{d:<2d} " + "  ".join(f"{k} {v}" for k, v in r[f'D{d}'].items()), flush=True)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "score":
        variants = sys.argv[2:] or DEFAULT
        res = score(variants)
        p = OUT / "tensor_variants.json"
        old = json.loads(p.read_text()) if p.exists() else {}
        old.update(res); p.write_text(json.dumps(old, indent=1))
