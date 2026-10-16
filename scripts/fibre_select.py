"""PROTOTYPE fibre-selective tensors on the tract-interface phantoms D0/D8/D15/D30: orientation from the fibres, not
from the step between the two tracts. Tensors only (no builds, no streamlines). Scored like shape_tensor.py: median
angle of the fibre direction to the truth by band (90/60/30 deg) and distance from the interface, on the mid-y slice.

  I_r<rho>            BRIDGE's tensor: G_rho * (g g^T), e3                                         (reference)
  M<w>_r<rho>         robust background removed: the gradient tensor of R = I - median_w(I) (w^3 voxel window), e3.
                      A median follows a step edge, partial-volume blur included (a monotone profile is a median
                      root), but not thin fibres, so R keeps the fibre texture and loses the step.
  L<s>_r<rho>         Hessian line vote: per voxel, Hessian at scale s with eigenvalues sorted by magnitude
                      |m1| >= |m2| >= |m3|; a line (fibre) has m1, m2 of one sign, a step edge only m1. Weight
                      w = max(|m2| - |m3|, 0) where sign m1 = sign m2, else 0; the line axis t = eigenvector of m3.
                      V = G_rho * (w t t^T); fibre direction = e1 of V (the most-voted axis).
  M<w>L<s>_r<rho>     the line vote on the median residual R

    python fibre_select.py score [VARIANT ...]
"""
import json, re, sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import shape_tensor as ST
import surface_phantoms as SP
import tensor_variants as TV

OUT = ST.OUT
NAMES = ST.NAMES
JY = ST.JY
DEFAULT = ("I_r3", "I_r6", "M5_r3", "M7_r3", "M9_r3", "M7_r6", "M9_r6", "L1_r3", "L1.5_r3", "L1.5_r6", "L2_r6",
           "M7L1.5_r6", "M9L1.5_r6")


def parse(v):
    m = re.fullmatch(r"(I|M(\d+)|)(?:L([\d.]+))?_r([\d.]+)", v)
    if not m:
        raise ValueError(v)
    return (int(m.group(2)) if m.group(2) else None), (float(m.group(3)) if m.group(3) else None), float(m.group(4))


def slab(vol, half):
    """The y slab JY +- half of vol (clipped to the volume) and the index of JY in it."""
    y0, y1 = max(JY - half, 0), min(JY + half + 1, vol.shape[1])
    return vol[:, y0:y1, :], JY - y0


def residual(name, w, half, cache, half_max=21):
    """I - median_w(I) on the y slab JY +- half (computed once per window, on the widest slab any variant needs)."""
    key = ("M", w)
    if key not in cache:
        v, c = slab(ST.volume(name), half_max + w // 2)
        cache[key] = (v - ndi.median_filter(v, size=w, mode="nearest"))[:, c - half_max:c + half_max + 1, :]
    assert half <= half_max
    return cache[key][:, half_max - half:half_max + half + 1, :]


def hess(v, s):
    o = {(0, 0): (2, 0, 0), (1, 1): (0, 2, 0), (2, 2): (0, 0, 2), (0, 1): (1, 1, 0), (0, 2): (1, 0, 1), (1, 2): (0, 1, 1)}
    M = np.empty(v.shape + (3, 3), np.float32)
    for (i, j), od in o.items():
        M[..., i, j] = M[..., j, i] = ndi.gaussian_filter(v, s, order=od, mode="nearest", truncate=3.0)
    return M


def line_vote(v, s):
    """(w, t): line weight and line axis per voxel."""
    lam, vec = np.linalg.eigh(hess(v, s))
    order = np.argsort(-np.abs(lam), axis=-1)
    m = np.take_along_axis(lam, order, -1)
    t = np.take_along_axis(vec, order[..., None, :], -1)[..., :, 2]
    w = np.where(np.sign(m[..., 0]) == np.sign(m[..., 1]), np.maximum(np.abs(m[..., 1]) - np.abs(m[..., 2]), 0), 0)
    return w.astype(np.float32), t.astype(np.float32)


def orient(name, variant, cache):
    w_med, s_line, rho = parse(variant)
    m = int(np.ceil(TV.TR * rho)) + 1
    half = m + 8                                                           # derivative support on top of the kernel
    if w_med is not None:
        v = residual(name, w_med, half, cache)
    else:
        v, _ = slab(ST.volume(name), half)
    c = half

    def mid(a):
        return ndi.gaussian_filter(a, rho, mode="nearest", truncate=TV.TR)[:, c, :]
    if s_line is None:
        g = TV.grads(v)
        S = [mid(g[i] * g[j]) for i, j in TV.PAIRS]
        pick = 0                                                           # e3
    else:
        w, t = line_vote(v, s_line)
        S = [mid(w * t[..., i] * t[..., j]) for i, j in TV.PAIRS]
        pick = 2                                                           # e1 of the vote
    M = np.empty((SP.NX, SP.NZ, 3, 3), np.float32)
    for k, (i, j) in enumerate(TV.PAIRS):
        M[..., i, j] = M[..., j, i] = S[k]
    return np.linalg.eigh(M)[1][..., :, pick]


def score(variants):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fibre_select.json"; res = json.loads(p.read_text()) if p.exists() else {}
    for name in NAMES:
        cache = {}
        for v in variants:
            r, _ = ST.score_e3(orient(name, v, cache))
            res.setdefault(v, {})[name] = r
            print(f"{name:4s} {v:12s} <0.5mm {r['within_0.5mm']:5.1f}  deep {r['deep_3-8mm']:5.1f}  "
                  + "  ".join(f"{k} {r[k]}" for k in ("90deg", "60deg", "30deg")), flush=True)
        p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2:] or DEFAULT)
