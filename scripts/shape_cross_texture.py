"""PROTOTYPE: the two-fibre TCA test of shape_cross_binary.py, stepped from clean rods towards the D phantoms' fibres.
128^3 at 0.1 mm, central 64 x 64 of the mid-y slice scored, rho 8 vox, CP rank 4 paired geometrically (most
perpendicular pairs; no truth) and by the oracle.

  rods        dark rods (radius 1.5 vox) on a 1 mm lattice, intersecting (as shape_cross_binary)
  lines       the D phantoms' fibres: straight lines at random positions and y, blurred 0.6 vox, darkening by 14 at
              the 99th percentile, D-phantom line density in total
  noise       Gaussian, sd 2 or 8 (the D phantoms: 8)

    python shape_cross_texture.py     (shapetensors env, one GPU)
"""
import json, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import shape_cross_binary as B
import shape_tensor as ST

N, VOX = B.N, B.VOX
DENS, AMP, BASE = 13.7, 14.0, 180.0
ROWS = (("rods, noise 8, 90° + 30°", "rods", 8.0, (90.0, 30.0)), ("rods, noise 8, 60° + 120°", "rods", 8.0, (60.0, 120.0)),
        ("lines, noise 2, 60° + 120°", "lines", 2.0, (60.0, 120.0)), ("lines, noise 8, 90° + 30°", "lines", 8.0, (90.0, 30.0)),
        ("lines, noise 8, 60° + 120°", "lines", 8.0, (60.0, 120.0)), ("lines, noise 8, single 30°", "lines", 8.0, (30.0,)))


def lines(theta, rng, frac):
    """One population of straight lines (density frac x DENS), each at one random y, blurred, normalised."""
    a = np.radians(theta); d = np.array([np.cos(a), np.sin(a)]); nrm = np.array([-d[1], d[0]]); L = N * VOX
    pr = np.array([[0, 0], [L, 0], [0, L], [L, L]]) @ nrm; lo, hi = pr.min(), pr.max()
    n = int(DENS * frac * (hi - lo) * L); s = np.arange(-25, 25, 0.05); cen = np.array([L / 2, L / 2])
    dens = np.zeros((N, N, N), np.float32)
    for _ in range(n):
        p = (rng.uniform(lo, hi) * nrm + (cen @ d) * d)[None] + s[:, None] * d[None]
        k = (p[:, 0] >= 0) & (p[:, 0] < L) & (p[:, 1] >= 0) & (p[:, 1] < L)
        ix = (p[k, 0] / VOX).astype(int); iz = (p[k, 1] / VOX).astype(int)
        np.add.at(dens, (ix, np.full(ix.shape, int(rng.uniform(0, L) / VOX)), iz), 1.0)
    dens = ndi.gaussian_filter(dens, 0.6)
    return dens / np.percentile(dens[dens > 0], 99)


def make(kind, noise, ths, rng):
    if kind == "rods":
        occ = np.max(np.stack([B.rods(t, 3.3 * k) for k, t in enumerate(ths)]), 0); v = B.BG - (B.BG - B.ROD) * occ
    else:
        v = BASE - AMP * sum(np.clip(lines(t, rng, 1.0 / len(ths)), 0, 1.5) for t in ths)
    return np.clip(np.rint(v + rng.normal(0, noise, v.shape)), 0, 255).astype(np.float32)


def main():
    rng = np.random.default_rng(31); res, keep = {}, {}
    for name, kind, noise, ths in ROWS:
        v = make(kind, noise, ths, rng); tru = np.stack([B.unit(t) for t in ths])
        F, out, fit = B.analyse(v, tru); r = res[name] = {"cp_rel_err": round(fit, 3)}; keep[name] = (v, F, out, tru)
        for k, e in out.items():
            if e.ndim == 2:
                q = {f"to {t:g}°": round(float(np.median(B.ang(e, B.unit(t)))), 1) for t in ths}
                if len(ths) == 2:
                    b = np.stack([tru[0] + tru[1], tru[0] - tru[1]]); b /= np.linalg.norm(b, axis=1, keepdims=True)
                    q["to bisector"] = round(float(np.median(np.minimum(B.ang(e, b[0]), B.ang(e, b[1])))), 1)
            else:
                errs, _ = B.pair_err(e, tru)
                q = {f"fibre {ths[0]:g}°": round(float(np.median(errs[:, 0])), 1), f"fibre {ths[1]:g}°": round(float(np.median(errs[:, 1])), 1),
                     "both within 15°": round(float(np.mean((errs < 15).all(1))), 3)}
            r[k] = q; print(f"{name:28s} {k:22s} {q}", flush=True)
        print(f"{name:28s} CP relative error {fit:.3f}", flush=True)
    (ST.OUT / "shape_cross_texture.json").write_text(json.dumps(res, indent=1))
    B.figure(keep, res)
    import shutil; shutil.move(ST.FIGS / "shape_cross_binary.png", ST.FIGS / "shape_cross_texture.png")
    print("wrote", ST.FIGS / "shape_cross_texture.png")


if __name__ == "__main__":
    main()
