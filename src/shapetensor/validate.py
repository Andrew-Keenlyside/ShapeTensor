"""The validation: the structure tensor (plain and excised) against the shape tensor's single and multi-vector
fields on the three phantoms, at the same ``sigma`` and ``rho``, over several seeds; a JSON of the scores and one
figure of DEC maps.

    shapetensor validate --out results/validation     (or: python -m shapetensor.validate)

Scores (degrees, axial): on the bundle the error to the population's direction; on the crossing the error of the
single field to the *nearest* population and, for the multi-vector field, the error of the two strongest vectors
matched to the two populations (and the share of voxels where both are within 15°); on the interface the error to
the population's direction by distance from the step (within 0.1 mm, 0.1–0.5 mm, beyond 0.5 mm). The shape
pipeline is calibrated once on single-population bundles at twelve orientations (a phantom's own teacher would
have one orientation only).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import st
from .phantoms import Grid, TextureParams, build, truth_field
from .pipeline import apply, calibrate
from .viz import dec_rgb

ANGLES = np.arange(0, 180, 15.0)
CASES = (("bundle", {"angle": 60.0}), ("crossing", {"cross_deg": 90.0}), ("crossing", {"cross_deg": 60.0}), ("interface", {"step": 40.0}))


def angle_deg(d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Axial angle (deg) between direction fields ``(3, ...)``; NaN where the truth is absent."""
    return torch.rad2deg(torch.arccos((d * t).sum(0).abs().clamp(max=1)))


def _summary(e: torch.Tensor, sel: torch.Tensor) -> dict:
    e = e[sel & torch.isfinite(e)]
    return {"median_deg": round(float(e.median()), 1), "within_15": round(float((e <= 15).float().mean()), 3), "n": int(e.numel())}


def _fields(img: np.ndarray, cal, sigma: float, rho: float, device: str) -> dict:
    v = torch.as_tensor(img, device=device).float()
    d_plain = st.structure_tensor(v, sigma=sigma, rho=rho, excise=False, order="zyx", device=device)
    d_exc = st.structure_tensor(v, sigma=sigma, rho=rho, excise=True, order="zyx", device=device)
    out = apply(cal, v, device=device)
    return {"image": v, "structure tensor": d_plain, "excised structure tensor": d_exc, "shape tensor": out["vec"], "weight": out["weight"],
            "shape tensor, multi": out["vecs"], "weights": out["weights"], "n_populations": int(out["modes"].shape[0])}


def score(name: str, tr: dict, f: dict, inner, vox: float) -> dict:
    ny = f["image"].shape[1]; dev = f["image"].device
    sel = torch.zeros(f["image"].shape, dtype=torch.bool, device=dev); sel[inner] = True
    t1 = truth_field(tr, "dir1", ny).to(dev)
    r = {"n_populations": f["n_populations"]}
    if name == "crossing":
        t2 = truth_field(tr, "dir2", ny).to(dev)
        for k in ("structure tensor", "excised structure tensor", "shape tensor"):
            r[k + " (nearest population)"] = _summary(torch.minimum(angle_deg(f[k], t1), angle_deg(f[k], t2)), sel)
        V = f["shape tensor, multi"]
        if V.shape[0] >= 2:
            e11, e12, e21, e22 = angle_deg(V[0], t1), angle_deg(V[0], t2), angle_deg(V[1], t1), angle_deg(V[1], t2)
            swap = (e12 + e21) < (e11 + e22)
            ea, eb = torch.where(swap, e12, e11), torch.where(swap, e21, e22)
            r["shape tensor, multi (pair)"] = {**_summary((ea + eb) / 2, sel), "both_within_15": round(float(((ea <= 15) & (eb <= 15))[sel].float().mean()), 3)}
        else:
            r["shape tensor, multi (pair)"] = None
    else:
        bands = {"all": sel}
        if name == "interface":
            dist = torch.as_tensor(tr["dist"], device=dev)[:, None, :].expand_as(sel)
            bands = {"within 0.1 mm of the step": sel & (dist < 0.1), "0.1-0.5 mm": sel & (dist >= 0.1) & (dist < 0.5), "beyond 0.5 mm": sel & (dist >= 0.5)}
        for k in ("structure tensor", "excised structure tensor", "shape tensor"):
            e = angle_deg(f[k], t1)
            r[k] = {b: _summary(e, s) for b, s in bands.items()}
    return r


SETTINGS = {"coherent": {"dispersion_deg": 8.0, "pitch_deg": 4.0, "texture": 1.0},        # tight bundles at the full texture level
            "calibrated": {"dispersion_deg": 20.0, "pitch_deg": 10.0, "texture": 0.5}}    # the dispersion and texture level matched to the real crop


def run(out_dir=None, seeds=(0, 1, 2), sigma: float = 1.0, rho: float = 10.0, engine: str = "anatomix_enc2", grid: Grid | None = None,
        settings=("coherent", "calibrated"), device: str = "cuda", log=print) -> dict:
    """The validation at each texture setting: a calibration on twelve bundles, then the four cases over the seeds.
    Returns the scores; writes ``validation.json`` and one figure per setting into ``out_dir``."""
    t0 = time.time()
    grid = grid or Grid()
    res = {"sigma": sigma, "rho": rho, "engine": engine, "seeds": list(seeds), "grid": {"core": grid.core, "margin": grid.margin, "vox": grid.vox}, "settings": {}}
    for setting in settings:
        p = SETTINGS[setting]; tex = TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"]); texture = p["texture"]
        cal_imgs = [build("bundle", 100 + k, grid, tex, texture=texture, angle=float(a), device=device)[0] for k, a in enumerate(ANGLES)]
        cal = calibrate([torch.as_tensor(im, device=device).float() for im in cal_imgs], engine=engine, sigma=sigma, rho=rho, margin=grid.margin[0], device=device)
        del cal_imgs
        log(f"[{time.time() - t0:4.0f}s] {setting}: calibrated on {len(ANGLES)} bundles (dispersion {p['dispersion_deg']:g}°, texture {texture:g}): readout overlap {cal.readout_overlap:.2f}, tau {cal.tau:.3f}")
        cases, panels = {}, []
        for name, kw in CASES:
            label = name + "".join(f" {v:g}" for v in kw.values())
            per_seed = []
            for s in seeds:
                img, tr, meta = build(name, s, grid, tex, texture=texture, device=device, **kw)
                f = _fields(img, cal, sigma, rho, device)
                per_seed.append(score(name, tr, f, grid.inner, grid.vox))
                if s == seeds[0]:
                    panels.append((label, f, tr))
                log(f"[{time.time() - t0:4.0f}s] {setting} {label} seed {s}: {json.dumps(per_seed[-1])}")
            cases[label] = {"per_seed": per_seed, "mean": _mean_over_seeds(per_seed)}
        res["settings"][setting] = {**p, "tau": cal.tau, "readout_overlap": cal.readout_overlap, "cases": cases}
        if out_dir is not None:
            out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
            figure(panels, grid, out / f"validation_{setting}.png", sigma, rho, setting)
    if out_dir is not None:
        json.dump(res, open(Path(out_dir) / "validation.json", "w"), indent=1)
        log(f"wrote {Path(out_dir) / 'validation.json'}")
    return res


def _mean_over_seeds(rows: list) -> dict:
    """The mean of every numeric leaf over the seeds (same structure as one row)."""
    def rec(vals):
        if all(isinstance(v, dict) for v in vals):
            return {k: rec([v[k] for v in vals]) for k in vals[0]}
        if all(isinstance(v, (int, float)) for v in vals):
            return round(float(np.mean(vals)), 3)
        return vals[0]
    return rec(rows)


def figure(panels, grid: Grid, path, sigma: float, rho: float, setting: str = ""):
    """DEC maps on the mid-y plane of the scored core: image, plain and excised structure tensor, the shape tensor's
    single field, and the two strongest vectors of its multi-vector field with their activation as brightness."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["image", "structure tensor", "excised structure tensor", "shape tensor", "multi-vector field: strongest", "multi-vector field: second"]
    fig, axs = plt.subplots(len(panels), len(cols), figsize=(4.2 * len(cols), 4.4 * len(panels)), dpi=100)
    y = grid.shape[1] // 2; zs, _, xs = grid.inner
    for i, (label, f, tr) in enumerate(panels):
        im = f["image"][zs, y, xs].cpu().numpy()
        axs[i, 0].imshow(im, cmap="gray", origin="lower"); axs[i, 0].set_title(f"{label}: image", fontsize=10)
        for j, k in enumerate(("structure tensor", "excised structure tensor", "shape tensor")):
            axs[i, 1 + j].imshow(dec_rgb(f[k][:, zs, y, xs].cpu().numpy()), origin="lower", interpolation="nearest"); axs[i, 1 + j].set_title(k, fontsize=10)
        V, W = f["shape tensor, multi"], f["weights"]
        wref = float(W[0][zs, :, xs].float().quantile(0.99).clamp_min(1e-6))                 # one scale for every panel of the row
        for j in range(2):
            if j < V.shape[0]:
                axs[i, 4 + j].imshow(dec_rgb(V[j][:, zs, y, xs].cpu().numpy(), (W[j][zs, y, xs] / wref).clamp(0, 1).cpu().numpy()), origin="lower", interpolation="nearest")
            else:
                axs[i, 4 + j].imshow(np.zeros((zs.stop - zs.start, xs.stop - xs.start, 3)), origin="lower")
            axs[i, 4 + j].set_title(cols[4 + j] + " (brightness = activation)", fontsize=10)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Phantom validation, {setting} texture, σ {sigma:g}, ρ {rho:g} voxels: DEC of each field (R = |x|, G = |y|, B = |z|) on the mid plane", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "validation")
