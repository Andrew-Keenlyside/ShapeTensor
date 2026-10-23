"""Validation on real data with independent labels: does the model's density agree with anatomy it never saw?

    python scripts/learn/real_validation.py --models A.pt [B.pt ...] [--out FILE.json] [--size 224]

1. **HiP-CT 20 um deep brain + SynthSeg on the same grid** (``/hdd/andrew/brain_atlas_20um_test_crops/<ID>``, five
   specimens): per lattice cell the density's anisotropy -- the share of its energy in orders >= 2 (a GFA in
   harmonics) -- and the agreement of the top peak with the neighbouring cells' (the orientation coherence of a tract).
   Scores: the AUC with which anisotropy separates cerebral white matter (SynthSeg 2/41) from grey matter (cortex 3/42,
   thalamus 10/49, caudate 11/50, putamen 12/51), and the coherence in white vs grey matter. A structure tensor (gradient sigma 1,
   window 5) is scored the same way as a baseline.
2. **BigBrain 2015 coronal 20 um + its cortical layer labels** (same grid): the cortical normal from the smoothed
   depth (layer index), and per layer the share of cells whose top peak is radial (< 30 deg from the normal) or
   tangential (> 60 deg). Expectation: tangential in layer I, radial (columns) in the deeper layers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from shapetensor.filters import gaussian
from shapetensor.learn.lobe_train import predict_sh, sh_peaks
from shapetensor.learn.realdata import normalise
from shapetensor.learn.sh import n_coef

DEV = "cuda"
CROPS = Path("/hdd/andrew/brain_atlas_20um_test_crops")
BB_IMG = "/hdd/andrew/Data/BigBrain2015/BigBrain2015_coronal_20um.ome.zarr"
BB_LAB = "/hdd/andrew/Data/BigBrain2015/BigBrain2015_coronal_layers_20um.ome.zarr"
WM = (2, 41); GM = (3, 42, 10, 49, 11, 50, 12, 51)      # cortex, thalamus, caudate, putamen (the deep-brain crops hold
                                                          # white matter ~40 %, thalamus ~35 %, pallidum ~14 % (fibre-rich: left out), putamen ~2 %)


def load_model(path):
    from shapetensor.learn.model import PopulationNet
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = PopulationNet.from_args(ck["args"]).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    return m


def read_stack(folder: Path, sl):
    """A sub-volume of a TIFF slice stack (one file per z, sorted)."""
    import tifffile
    files = sorted(folder.glob("*.tif*"))
    return np.stack([tifffile.imread(files[z])[sl[1], sl[2]] for z in range(sl[0].start, sl[0].stop)])


def anisotropy(f: torch.Tensor) -> torch.Tensor:
    """The share of a density's energy in orders >= 2 (``f (c, ...)``)."""
    return (f[1:] ** 2).sum(0).sqrt() / (f ** 2).sum(0).sqrt().clamp_min(1e-9)


def st_field(x: torch.Tensor):
    """Structure tensor per 4^3 cell (gradient sigma 1, window 5): the coherence and the least-gradient direction."""
    xs = gaussian(x.float(), 1.0); gz, gy, gx = torch.gradient(xs)
    G = torch.stack([Fn.avg_pool3d(gaussian(c, 5.0)[None, None], 4)[0, 0] for c in (gz * gz, gy * gy, gx * gx, gz * gy, gz * gx, gy * gx)])
    M = torch.stack([torch.stack([G[0], G[3], G[4]]), torch.stack([G[3], G[1], G[5]]), torch.stack([G[4], G[5], G[2]])]).permute(2, 3, 4, 0, 1)
    ev, V = torch.linalg.eigh(M)
    coh = (ev[..., 2] - ev[..., 0]) / ev.sum(-1).clamp_min(1e-9)
    return coh, V[..., :, 0].permute(3, 0, 1, 2)


def coherence(d: torch.Tensor, valid: torch.Tensor, lag: int = 3) -> torch.Tensor:
    """Mean |cos| between each cell's direction ``d (3, z, y, x)`` and the cells ``lag`` away along the axes (valid cells
    only) -- at a lag beyond the window's overlap (sigma 5 voxels vs 4-voxel cells), so it measures the tissue."""
    acc = torch.zeros(d.shape[1:], device=d.device); n = torch.zeros_like(acc)
    for a in range(3):
        for s in (lag, -lag):
            dn = torch.roll(d, s, dims=a + 1); vn = torch.roll(valid, s, dims=a)
            c = (d * dn).sum(0).abs()
            acc += c * vn; n += vn.float()
    return acc / n.clamp_min(1)


def auc(pos: torch.Tensor, neg: torch.Tensor, n: int = 4000) -> float:
    g = torch.Generator(device="cpu").manual_seed(0)
    p = pos[torch.randint(0, pos.numel(), (n,), generator=g).to(pos.device)]; q = neg[torch.randint(0, neg.numel(), (n,), generator=g).to(neg.device)]
    return float((p[:, None] > q[None]).float().mean() + 0.5 * (p[:, None] == q[None]).float().mean())


@torch.no_grad()
def model_fields(m, x: torch.Tensor, voxel_um: float, edge: int):
    """The model's density on ``x (Z, Y, X)`` (0-1), anisotropy and top peak per cell (``edge`` cells dropped)."""
    v = torch.tensor([math.log2(voxel_um)], device=DEV)
    f, _, _, _ = predict_sh(m, x[None, None], voxel=v)
    f = f[0][:, edge:-edge, edge:-edge, edge:-edge]
    D, V = sh_peaks(f, 8 if f.shape[0] >= n_coef(8) else 4)
    return f, anisotropy(f), D[0].reshape(3, *f.shape[1:]), (V[0] > 0).reshape(f.shape[1:])


def _pool_crop(a: np.ndarray, f: int, mode: str = "mean") -> np.ndarray:
    """Block-average (images) or block-pick (labels) a crop by ``f``."""
    if f == 1:
        return a
    n = [s // f * f for s in a.shape]; a = a[:n[0], :n[1], :n[2]]
    if mode == "pick":
        return a[f // 2::f, f // 2::f, f // 2::f]
    return a.reshape(n[0] // f, f, n[1] // f, f, n[2] // f, f).mean((1, 3, 5))


def hipct_synthseg(models: dict, size: int = 224, per_specimen: int = 3, scales=(1,)) -> dict:
    res = {}
    for f in scales:
        rng = np.random.default_rng(0); ext = size * f
        for spec in sorted(CROPS.iterdir()):
            img_dir = next(spec.glob("*_deepbrain_L_20um_768"), None); lab_dir = next(spec.glob("*_deepbrain_L_synthseg_768"), None)
            if img_dir is None or lab_dir is None or ext > 768:
                continue
            for k in range(per_specimen):
                for _ in range(40):                                                                # both tissues present
                    o = [int(rng.integers(0, 768 - ext + 1)) for _ in range(3)]; sl = tuple(slice(a, a + ext) for a in o)
                    lab = read_stack(lab_dir, sl)
                    if np.isin(lab, WM).mean() >= 0.05 and np.isin(lab, GM).mean() >= 0.05:
                        break
                img = _pool_crop(read_stack(img_dir, sl).astype(np.float32), f); lab = _pool_crop(lab, f, "pick")
                x, _ = normalise(torch.tensor(img, device=DEV))                                     # the tissue's range (background either polarity)
                edge = 12
                labc = torch.tensor(lab[1::4, 1::4, 1::4].astype(np.int64), device=DEV)[edge:-edge, edge:-edge, edge:-edge]
                wm = torch.isin(labc, torch.tensor(WM, device=DEV)); gm = torch.isin(labc, torch.tensor(GM, device=DEV))
                key = f"{spec.name}_{k}_x{f}"; res[key] = {"origin": o}
                coh, sd = st_field(x)
                coh = coh[edge:-edge, edge:-edge, edge:-edge]; sd = sd[:, edge:-edge, edge:-edge, edge:-edge]
                if bool(wm.any()) and bool(gm.any()):
                    res[key]["st_auc_wm_gm"] = auc(coh[wm], coh[gm])
                    ch = coherence(sd, torch.ones_like(wm)); res[key]["st_coherence_wm"] = float(ch[wm].mean()); res[key]["st_coherence_gm"] = float(ch[gm].mean())
                for name, m in models.items():
                    _, ga, d, ok = model_fields(m, x, 20.0 * f, edge)
                    if bool(wm.any()) and bool(gm.any()):
                        res[key][f"{name}_auc_wm_gm"] = auc(ga[wm], ga[gm])
                        ch = coherence(d, ok)
                        res[key][f"{name}_coherence_wm"] = float(ch[wm & ok].mean()) if bool((wm & ok).any()) else float("nan")
                        res[key][f"{name}_coherence_gm"] = float(ch[gm & ok].mean()) if bool((gm & ok).any()) else float("nan")
                        res[key][f"{name}_peak_share_wm"] = float(ok[wm].float().mean()); res[key][f"{name}_peak_share_gm"] = float(ok[gm].float().mean())
                print(key, {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in res[key].items()}, flush=True)
    return res


def bigbrain_layers(models: dict, size: int = 224, n: int = 4, scales=(1,)) -> dict:
    import zarr
    img = zarr.open_group(BB_IMG, mode="r")["0"]; lab = zarr.open_group(BB_LAB, mode="r")["0"]
    res = {}; Z, Y, X = img.shape
    for f in scales:
        res.update(_bigbrain_scale(models, img, lab, size, n, f, Z, Y, X))
    return res


def _bigbrain_scale(models, img, lab, size, n, f, Z, Y, X) -> dict:
    rng = np.random.default_rng(1); res = {}; got = 0; tries = 0; ext = size * f
    while got < n and tries < 400:
        tries += 1
        o = [int(rng.integers(Z // 4, 3 * Z // 4 - ext)), int(rng.integers(0, Y - ext)), int(rng.integers(0, X - ext))]
        sl = tuple(slice(a, a + ext) for a in o)
        L = np.asarray(lab[sl])
        if not ((L >= 1) & (L <= 6)).mean() > 0.25:
            continue
        L = _pool_crop(L, f, "pick")
        I = _pool_crop(np.asarray(img[sl]).astype(np.float32), f)
        x, _ = normalise(torch.tensor(I, device=DEV))                                               # a white slide is background, not tissue
        Lt = torch.tensor(L.astype(np.float32), device=DEV)
        depth = torch.where(Lt == 0, torch.zeros_like(Lt), Lt)                                             # outside 0, layers 1-6, white matter 7
        dsm = gaussian(depth, 6.0); g = torch.stack(torch.gradient(dsm))
        edge = 12
        nrm = Fn.normalize(Fn.avg_pool3d(g[None], 4)[0], dim=0)[:, edge:-edge, edge:-edge, edge:-edge]
        labc = torch.tensor(L[1::4, 1::4, 1::4].astype(np.int64), device=DEV)[edge:-edge, edge:-edge, edge:-edge]
        key = f"bb_{got}_x{f}"; res[key] = {"origin": o}
        coh, sd = st_field(x); sd = sd[:, edge:-edge, edge:-edge, edge:-edge]
        dirs = {"st": (sd, torch.ones_like(labc, dtype=torch.bool))}
        for name, m in models.items():
            _, _, d, ok = model_fields(m, x, 20.0 * f, edge); dirs[name] = (d, ok)
        for name, (d, ok) in dirs.items():
            ang = torch.rad2deg(torch.acos((d * nrm).sum(0).abs().clamp(max=1)))
            for layer in range(1, 8):
                sel = (labc == layer) & ok
                if int(sel.sum()) >= 20:
                    res[key][f"{name}_L{layer}_radial"] = float((ang[sel] < 30).float().mean())
                    res[key][f"{name}_L{layer}_tangential"] = float((ang[sel] > 60).float().mean())
        got += 1
        print(key, {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in res[key].items()}, flush=True)
    return res


def summarise(res: dict) -> dict:
    """Means per scale (keys suffixed _x<f>)."""
    out = {}
    for f in sorted({k.rsplit("_x", 1)[1] for k in res}):
        sub = {k: r for k, r in res.items() if k.endswith(f"_x{f}")}
        keys = sorted({k for r in sub.values() for k in r if k != "origin"})
        out.update({f"x{f}_{k}": float(np.nanmean([r[k] for r in sub.values() if k in r])) for k in keys})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--out"); ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--per-specimen", type=int, default=3); ap.add_argument("--bigbrain", type=int, default=4)
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 2, 3], help="input block-averaging factors (20 um x f)")
    a = ap.parse_args()
    models = {Path(p).parent.name: load_model(p) for p in a.models}
    h = hipct_synthseg(models, a.size, a.per_specimen, a.scales) if a.per_specimen else {}
    b = bigbrain_layers(models, a.size, a.bigbrain, a.scales) if a.bigbrain else {}
    out = {"hipct_synthseg": h, "bigbrain": b, "summary_hipct": summarise(h), "summary_bigbrain": summarise(b)}
    print(json.dumps(out["summary_hipct"], indent=1)); print(json.dumps(out["summary_bigbrain"], indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
