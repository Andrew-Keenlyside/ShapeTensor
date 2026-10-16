"""PROTOTYPE shape tensor: orientation from the gradients of the anatomix feature field ("shape space") instead of the
intensity, and integration over shape-space clusters instead of (or within) a spatial rho. Tract-interface phantoms
D0 (no step), D8, D15, D30 (WM interface at z = 10 mm, brighter above by the step; fibres cross it at 90/60/30 deg).

Tensor variants (sigma 1 voxel derivatives, truncate 2; e3 = least-change direction, scored on the mid-y slice):
  I_r<rho>               intensity: T0 = g g^T, Gaussian integration rho (voxels). I_r3 is BRIDGE as is.
  F_r<rho>               shape: T0 = sum over the 16 anatomix channels of grad f_c grad f_c^T (each channel is
                         L2-normalised per voxel across channels, so no intensity scale enters)
  <I|F>_k<K>_r<rho>      integrated only over voxels of the same shape cluster (k-means of the feature vectors,
                         K clusters): S(x) = G_rho*(m_k T0) / G_rho*(m_k) at x, k = the cluster of x
  <I|F>_k<K>s<s>_r<rho>  soft clusters, the shape scale: p_k(x) ~ exp(-|f(x) - mu_k|^2 / 2 h^2), h = s x the median
                         distance to the nearest centre; S(x) = sum_k p_k(x) G_rho*(p_k T0)(x) / G_rho*(p_k)(x).
                         s -> 0 is the hard clustering, s -> inf the plain spatial rho.
  R<K>_...               contrast removed with shape space: T0 = g g^T of the residual I - mean_k(I), mean_k = the
                         intensity mean of the voxel's hard shape cluster (K clusters); Ro = oracle, the true split
  H_r<rho>               shape tensor of high-passed features, f - G_3 * f (coarse shape removed)

    python shape_tensor.py score [VARIANT ...]     (chunkreg env: torch + anatomix)
    python shape_tensor.py fig VARIANT ...
"""
import json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
import anatomix_shape as AS
import surface_phantoms as SP
import tensor_variants as TV

OUT = Path("/home/andrew/bridge_wmcov/shape_tensor")
FIGS = Path(__file__).resolve().parents[1] / "figures" / "shape_tensor"
NAMES = ("D0", "D8", "D15", "D30")
JY = SP.NY // 2
BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (3.0, 8.0))
DEFAULT = ("I_r3", "I_r6", "F_r3", "F_r6", "I_k8_r6", "F_k8_r6", "F_k4_r6", "F_k16_r6",
           "F_k8s0.5_r6", "F_k8s1_r6", "F_k8s2_r6", "I_k8s1_r6", "F_k8_r9", "F_k16_r9")


def volume(name):
    return np.asarray(nib.load(SP.ROOT / f"{name}.nii.gz").dataobj).astype(np.float32)


_FEATS = {}


def features(name):
    if name not in _FEATS:
        _FEATS[name] = AS.anatomix(volume(name), SP.VOX)                    # (16, x, y, z), reflect-padded
    return _FEATS[name]


def t0_intensity(v):
    g = TV.grads(v)
    return np.stack([g[i] * g[j] for i, j in TV.PAIRS])


def t0_shape(F):
    acc = np.zeros((6,) + F.shape[1:], np.float32)
    for f in F:
        g = TV.grads(f)
        for k, (i, j) in enumerate(TV.PAIRS):
            acc[k] += g[i] * g[j]
    return acc


def at_centre(a, rho):
    """G_rho * a, evaluated on the mid-y slice only (exact: the y crop is wider than the kernel)."""
    m = int(np.ceil(TV.TR * rho)) + 1
    return ndi.gaussian_filter(a[:, JY - m:JY + m + 1, :], rho, mode="nearest", truncate=TV.TR)[:, m, :]


def clusters(F, K, seed=0, n=200_000):
    from sklearn.cluster import MiniBatchKMeans
    X = F.reshape(F.shape[0], -1).T
    rng = np.random.default_rng(seed)
    km = MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=8192).fit(X[rng.choice(len(X), n, replace=False)])
    mu = km.cluster_centers_.astype(np.float32)
    d2 = (X ** 2).sum(1)[:, None] + (mu ** 2).sum(1)[None] - 2 * X @ mu.T
    return np.maximum(d2, 0).reshape(F.shape[1:] + (K,)), mu


def memberships(d2, s):
    if s is None:                                                           # hard
        return np.moveaxis(np.eye(d2.shape[-1], dtype=np.float32)[d2.argmin(-1)], -1, 0)
    h2 = (s * np.median(np.sqrt(d2.min(-1)))) ** 2
    z = -(d2 - d2.min(-1, keepdims=True)) / (2 * h2)
    p = np.exp(z); p /= p.sum(-1, keepdims=True)
    return np.moveaxis(p.astype(np.float32), -1, 0)


def parse(variant):
    src, *rest = variant.split("_")
    rho = float(rest[-1][1:]); K = s = None
    if len(rest) == 2:
        kk = rest[0][1:]
        K, s = (int(kk.split("s")[0]), float(kk.split("s")[1])) if "s" in kk else (int(kk), None)
    return src, K, s, rho


def residual(name, src, cache):
    """Intensity minus its shape-cluster means (hard clusters), or minus the true per-side means (Ro)."""
    v = volume(name)
    if src == "Ro":
        lab = np.broadcast_to(((np.arange(SP.NZ) + 0.5) * SP.VOX >= SP.D_PLANE_Z)[None, None, :], v.shape).astype(int)
    else:
        K = int(src[1:])
        if ("km", K) not in cache:
            cache[("km", K)] = clusters(features(name), K)
        lab = cache[("km", K)][0].argmin(-1)
    means = ndi.mean(v, lab, np.arange(lab.max() + 1))
    return v - np.asarray(means, np.float32)[lab]


def tensor(name, variant, cache):
    src, K, s, rho = parse(variant)
    key = ("T0", src)
    if key not in cache:
        if src == "I":
            cache[key] = t0_intensity(volume(name))
        elif src.startswith("R"):
            cache[key] = t0_intensity(residual(name, src, cache))
        elif src == "H":
            F = features(name)
            cache[key] = t0_shape(np.stack([f - ndi.gaussian_filter(f, 3.0, mode="nearest", truncate=TV.TR) for f in F]))
        else:
            cache[key] = t0_shape(features(name))
    T0 = cache[key]
    if K is None:
        S = np.stack([at_centre(c, rho) for c in T0])
    else:
        if ("km", K) not in cache:
            cache[("km", K)] = clusters(features(name), K)
        m = int(np.ceil(TV.TR * rho)) + 1; ys = slice(JY - m, JY + m + 1)          # the y slab the kernel reaches
        P = memberships(cache[("km", K)][0][:, ys], s); T0s = T0[:, :, ys]

        def mid(a):
            return ndi.gaussian_filter(a, rho, mode="nearest", truncate=TV.TR)[:, m, :]
        S = np.zeros((6, SP.NX, SP.NZ), np.float32)
        for p in P:
            den = np.maximum(mid(p), 1e-6); pc = p[:, m, :]
            for c in range(6):
                S[c] += pc * mid(p * T0s[c]) / den
    M = np.empty((SP.NX, SP.NZ, 3, 3), np.float32)
    for k, (i, j) in enumerate(TV.PAIRS):
        M[..., i, j] = M[..., j, i] = S[k]
    w, v = np.linalg.eigh(M)
    return v[..., :, 0], w[..., ::-1]


def score_e3(e):
    td = np.load(SP.ROOT / "D_truth.npz"); tru, dep, XD = td["tru"], td["depth"], td["X"]
    a = TV.angle3(e, tru)
    r = {f"{ang:.0f}deg": [round(float(np.median(a[(XD >= x0 + 1) & (XD < x1 - 1) & (dep >= b0) & (dep < b1)])), 1)
                           for b0, b1 in BINS] for x0, x1, ang in SP.D_BANDS}
    inner = (dep < 0.5) & (XD >= 1) & (XD < 19); deep = (dep >= 3) & (dep < 8) & (XD >= 1) & (XD < 19)
    r["within_0.5mm"] = round(float(np.median(a[inner])), 1); r["deep_3-8mm"] = round(float(np.median(a[deep])), 1)
    return r, a


def score(variants):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "scores.json"; res = json.loads(p.read_text()) if p.exists() else {}
    for name in NAMES:
        cache = {}
        for v in variants:
            e, _ = tensor(name, v, cache)
            r, _ = score_e3(e)
            res.setdefault(v, {})[name] = r
            print(f"{name:4s} {v:14s} <0.5mm {r['within_0.5mm']:5.1f}  deep {r['deep_3-8mm']:5.1f}  "
                  + "  ".join(f"{k} {r[k]}" for k in ("90deg", "60deg", "30deg")), flush=True)
        _FEATS.pop(name, None)
        p.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2:] or DEFAULT)
    elif sys.argv[1] == "fig":
        import shape_tensor_fig
        shape_tensor_fig.main(sys.argv[2:])
