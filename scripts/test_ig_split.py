"""Test of the per-population shape-space formulation, no training anywhere:

1. populations detected from the image's own spectrum: a straight fibre population is invariant along its axis, so its
   texture's power lies on the plane across that axis; the plane energy over a hemisphere of directions peaks at each
   population's axis (no readers, no calibration);
2. the fibre texture split linearly into one component per population by soft spectral-plane filters
   (I = I0 + sum_r I_r + residual, I0 = the image smoothed past the fibre scale);
3. anatomix-dev-vit's features attributed to each component by integrated gradients along I0 -> I
   (F_r = mean_k J(I0 + a_k (I - I0)) . I_r, central finite differences, midpoint rule; patch grid removed by averaging
   the (0,0,0) and (4,4,4) window offsets);
4. each population's direction from its own feature field: the channel-summed gradient tensor (FEAT: smallest
   eigenvector) and a rank-3 TCA (CP) of voxels x channels x gradient axes (the null direction of its axis factors).

Controls: the unsplit features (the known failure on whole-block crossings), the network run on each component alone
(f(I0 + I_r): not a partition), and the image structure tensor of each component (no network at all). Detected and
true directions both used for the split, so detection error and formulation error are told apart.

    python scripts/test_ig_split.py [OUT_JSON]      (CPU by default: DEVICE=cuda to use the GPU)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor import st, tensors
from shapetensor.features.devvit import load_devvit
from shapetensor.filters import gaussian
from shapetensor.phantoms import Grid, TextureParams, build
from shapetensor.separate import hemisphere
from shapetensor.validate import SETTINGS

DEV = os.environ.get("DEVICE", "cpu")
if DEV == "cpu":
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/ig_split_test.json")
M_STEPS, H, TILE, PAD = 4, 0.25, 128, 4
PLANE_SIG = math.sin(math.radians(10.0))          # spectral-plane half-width for the split
DET_SIG = math.sin(math.radians(6.0))             # sharper for detection
BAND = (0.04, 0.30)                               # cycles / voxel: the fibre texture band
CASES = [("bundle", {"angle": 60.0}), ("crossing", {"cross_deg": 90.0}), ("crossing", {"cross_deg": 60.0}), ("crossing", {"cross_deg": 45.0})]
net = load_devvit(DEV)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def forward(x):
    with torch.no_grad():
        return net(x[None, None]).float()[0]


def f(B):
    """Shift-averaged dev-vit features of a (132,)^3 block in [0, 1] on content voxels [4, 128)^3 -> (32, 124, 124, 124)."""
    a = forward(B[:TILE, :TILE, :TILE])[:, PAD:, PAD:, PAD:]
    b = forward(B[PAD:, PAD:, PAD:])[:, :TILE - PAD, :TILE - PAD, :TILE - PAD]
    return 0.5 * (a + b)


def kgrid(shape):
    ks = [torch.fft.fftfreq(n, device=DEV) for n in shape]
    K = torch.stack(torch.meshgrid(*ks, indexing="ij"))                                   # (3, Z, Y, X) cycles/voxel
    return K, K.norm(dim=0)


def detect(T, u, nmax=3, min_sep=25.0, rel=0.35):
    win = [torch.hann_window(n, periodic=False, device=DEV) for n in T.shape]
    P = torch.fft.fftn(T * win[0][:, None, None] * win[1][None, :, None] * win[2][None, None, :]).abs() ** 2
    K, kn = kgrid(T.shape)
    band = (kn >= BAND[0]) & (kn <= BAND[1])
    kh = (K[:, band] / kn[band]).T                                                         # (n, 3) unit k
    w = P[band]
    E = torch.stack([(w * torch.exp(-((kh @ d) ** 2) / (2 * DET_SIG ** 2))).sum() for d in u]) / w.sum()
    order = torch.argsort(E, descending=True); med, top = float(E.median()), float(E.max())
    peaks = []
    for i in order.tolist():
        if (float(E[i]) - med) / max(top - med, 1e-12) < rel or len(peaks) >= nmax:
            break
        if all(math.degrees(math.acos(min(1.0, abs(float(u[i] @ u[j]))))) >= min_sep for j in peaks):
            peaks.append(i)
    return [u[i] for i in peaks], [round((float(E[i]) - med) / max(top - med, 1e-12), 3) for i in peaks]


def split(T, dirs):
    K, kn = kgrid(T.shape)
    kh = K / kn.clamp_min(1e-12)
    W = torch.stack([torch.exp(-((kh * d.view(3, 1, 1, 1)).sum(0) ** 2) / (2 * PLANE_SIG ** 2)) for d in dirs])
    W = W / W.sum(0).clamp_min(1.0)[None]; W[:, 0, 0, 0] = 0
    FT = torch.fft.fftn(T)
    return [torch.fft.ifftn(FT * W[r]).real for r in range(len(dirs))]


def angle(a, b):
    a = a / a.norm(); b = b / b.norm()
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))


def feat_dir(G, core, rho=10.0):
    """FEAT: the region-mean channel-summed gradient tensor's smallest eigenvector, and the per-voxel median error later."""
    P = tensors.feature_products(G, 1.0, 3.0)
    S = P[(slice(None),) + core].mean((1, 2, 3)).view(6, 1, 1, 1)
    d, gap = tensors.direction_and_gap(S)
    _, dv, _ = tensors.shape_tensor(P, torch.ones(G.shape[1:], device=G.device), rho=rho, truncate=3.0)
    return d[:, 0, 0, 0], float(gap), dv[(slice(None),) + core]


def tca_dir(G, core, n=20000, seed=0):
    """Rank-3 CP of voxels x channels x gradient axes (z, y, x); the population axis is the null direction of the
    axis factors weighted by their components' norms."""
    import tensorly as tl
    from tensorly.decomposition import parafac
    Gs = gaussian(G.float(), 1.0) if G.dim() == 4 else G
    gr = torch.stack(torch.gradient(Gs, dim=(1, 2, 3)), -1)[(slice(None),) + core]       # (C, z, y, x, 3)
    X = gr.reshape(gr.shape[0], -1, 3).permute(1, 0, 2)                                    # (voxels, C, 3)
    g = torch.Generator().manual_seed(seed)
    X = X[torch.randperm(X.shape[0], generator=g)[:n]].double().cpu().numpy()
    X = X / np.sqrt((X ** 2).mean())
    w, (A, B, C) = parafac(tl.tensor(X), rank=3, init="svd", n_iter_max=300, tol=1e-8, normalize_factors=True)
    M = sum(float(w[k]) ** 2 * np.outer(C[:, k], C[:, k]) for k in range(3))
    val, vec = np.linalg.eigh(M)
    return torch.as_tensor(vec[:, 0], dtype=torch.float32)


def st_dir(I, core):
    S = st.structure_tensor_3d(I, 1.0, 10.0, 3.0, device=I.device)
    d, _ = tensors.direction_and_gap(S[(slice(None),) + core].mean((1, 2, 3)).view(6, 1, 1, 1))
    return d[:, 0, 0, 0]


def evaluate(name, G, core, truths, assigned):
    fd, fgap, dv = feat_dir(G, core)
    td = tca_dir(G, core)
    pv = [float(torch.rad2deg(torch.arccos((dv * t.view(3, 1, 1, 1)).sum(0).abs().clamp(max=1))).median()) for t in truths]
    return {"feat_deg_to_assigned": round(angle(fd, assigned), 1), "tca_deg_to_assigned": round(angle(td, assigned), 1),
            "feat_deg_to_each_truth": [round(angle(fd, t), 1) for t in truths], "tca_deg_to_each_truth": [round(angle(td, t), 1) for t in truths],
            "feat_voxel_median_deg_to_each_truth": [round(x, 1) for x in pv], "feat_gap": round(fgap, 3)}


def main():
    p = SETTINGS["calibrated"]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"]); g = Grid()
    u = hemisphere(400, DEV)
    res = {"settings": {"m_steps": M_STEPS, "h": H, "plane_half_width_deg": 10.0, "detection_half_width_deg": 6.0, "band_cyc_per_vox": BAND,
                        "texture": "calibrated", "device": DEV}, "cases": []}
    for name, kw in CASES:
        img, tr, meta = build(name, 0, g, tex, texture=p["texture"], device=DEV, **kw)
        v = torch.as_tensor(img, device=DEV).float() / 255.0
        Zc, Yc, Xc = v.shape
        z0 = (Zc - (TILE + PAD)) // 2; x0 = (Xc - (TILE + PAD)) // 2
        blk = v[z0:z0 + TILE + PAD, :, x0:x0 + TILE + PAD]
        py = (TILE + PAD - Yc) // 2
        blk = Fn.pad(blk[None, None], (0, 0, py, TILE + PAD - Yc - py, 0, 0), mode="reflect")[0, 0]          # y: 96 -> 132
        inner = g.inner
        core = tuple(slice(s.start - o - PAD, s.stop - o - PAD) for s, o in zip(inner, (z0, -py, x0)))      # in feature coordinates
        truths = [torch.as_tensor(tr["dir1"][g.margin[0], g.margin[2]], device=DEV).float()]
        if name == "crossing":
            truths.append(torch.as_tensor(tr["dir2"][g.margin[0], g.margin[2]], device=DEV).float())
        I0 = gaussian(blk, 6.0); T = blk - I0
        det, prom = detect(T, u)
        case = {"case": f"{name} {kw}", "detected": [[round(angle(d, t), 1) for t in truths] for d in det], "detected_prominence": prom}
        log(case["case"], "detected (deg to each truth):", case["detected"])
        F = f(blk); F0 = f(I0); dF = F - F0
        case["unsplit"] = evaluate("unsplit", F, core, truths, truths[0])
        case["image_structure_tensor_unsplit"] = [round(angle(st_dir(blk, tuple(slice(c.start + PAD, c.stop + PAD) for c in core)), t), 1) for t in truths]
        for mode, dirs in (("detected", det), ("oracle", truths)):
            if mode == "oracle" and name == "bundle":
                continue
            comps = split(T, dirs)
            out = []
            Fr_all = []
            for r, (d, Ir) in enumerate(zip(dirs, comps)):
                Fr = torch.zeros_like(F)
                for k in range(M_STEPS):
                    a = (k + 0.5) / M_STEPS
                    base = I0 + a * T
                    Fr += (f(base + H * Ir) - f(base - H * Ir)) / (2 * H)
                Fr /= M_STEPS; Fr_all.append(Fr)
                assigned = min(truths, key=lambda t: angle(d, t))
                naive = f(I0 + Ir)
                imgcore = tuple(slice(c.start + PAD, c.stop + PAD) for c in core)
                out.append({"population": r, "direction_deg_to_each_truth": [round(angle(d, t), 1) for t in truths],
                            "ig_partition": evaluate("ig", Fr, core, truths, assigned),
                            "network_on_component": evaluate("naive", naive - F0, core, truths, assigned),
                            "image_structure_tensor_of_component_deg": [round(angle(st_dir(Ir, imgcore), t), 1) for t in truths]})
                log(f"  {mode} pop {r}: IG feat {out[-1]['ig_partition']['feat_deg_to_assigned']} tca {out[-1]['ig_partition']['tca_deg_to_assigned']} | "
                    f"naive feat {out[-1]['network_on_component']['feat_deg_to_assigned']} | image ST {out[-1]['image_structure_tensor_of_component_deg']}")
            S = torch.stack(Fr_all).sum(0)
            case[f"split_{mode}"] = out
            case[f"split_{mode}_completeness"] = round(float(1 - ((dF - S) ** 2).sum() / (dF ** 2).sum()), 3)
        res["cases"].append(case)
        OUT.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(OUT, "w"), indent=1)
    log("wrote", OUT)


if __name__ == "__main__":
    main()
