"""The validation: the structure tensor (plain and excised) against the shape tensor's single and multi-vector
fields on the three phantoms, at the same ``sigma`` and ``rho``, over several seeds; a JSON of the scores and one
figure in the package's standard layout (image | excised structure tensor | shape space | activation layers |
shape tensor | multi-vector field, first vector).

    shapetensor validate --out results/validation     (or: python -m shapetensor.validate)

Scores (degrees, axial): on the bundle the error to the population's direction; on the crossing the error of each
field to the *nearest* population and, for the multi-vector field, the error of the two strongest vectors matched
to the two populations (and the share of voxels where both are within 15°); on the interface the error to the
population's direction by distance from the step (within 0.1 mm, 0.1–0.5 mm, beyond 0.5 mm). The shape pipeline is
calibrated once on single-population bundles at twelve orientations (a phantom's own teacher would have one
orientation only). The texture is the one matched to the real HiP-CT crop (``calibrated``); ``coherent`` (tight
bundles at the full texture level) is kept as an option.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import st
from ..phantoms import Grid, TextureParams, build, truth_field
from ..pipeline import apply, calibrate
from ..viz import pipeline_row, shape_space

ANGLES = np.arange(0, 180, 15.0)
CASES = (("bundle", {"angle": 60.0}), ("crossing", {"cross_deg": 60.0}), ("interface", {"step": 40.0}))
FIELDS = ("structure tensor", "excised structure tensor", "shape tensor", "shape tensor, multi (first)")
SETTINGS = {"calibrated": {"dispersion_deg": 20.0, "pitch_deg": 10.0, "texture": 0.5},    # the dispersion and texture level matched to the real crop
            "coherent": {"dispersion_deg": 8.0, "pitch_deg": 4.0, "texture": 1.0}}        # tight bundles at the full texture level (optional)


def angle_deg(d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Axial angle (deg) between direction fields ``(3, ...)``; NaN where the truth is absent."""
    return torch.rad2deg(torch.arccos((d * t).sum(0).abs().clamp(max=1)))


def _summary(e: torch.Tensor, sel: torch.Tensor) -> dict:
    e = e[sel & torch.isfinite(e)]
    return {"median_deg": round(float(e.median()), 1), "within_15": round(float((e <= 15).float().mean()), 3), "n": int(e.numel())}


def _fields(img: np.ndarray, cal, sigma: float, rho: float, region: torch.Tensor, device: str) -> dict:
    v = torch.as_tensor(img, device=device).float()
    d_plain = st.structure_tensor(v, sigma=sigma, rho=rho, excise=False, order="zyx", device=device)
    d_exc = st.structure_tensor(v, sigma=sigma, rho=rho, excise=True, order="zyx", device=device)
    F = cal.features(v); out = apply(cal, v, features=F, device=device)
    space = shape_space(F, cal.dictionary, out["activation"], out["tau"], region)[:3]; del F
    return {"image": v, "structure tensor": d_plain, "excised structure tensor": d_exc, "shape tensor": out["vec"], "weight": out["weight"],
            "shape tensor, multi": out["vecs"], "shape tensor, multi (first)": out["vecs"][0], "weights": out["weights"], "activation": out["activation"],
            "shape space": space, "n_populations": int(out["modes"].shape[0])}


def score(name: str, tr: dict, f: dict, inner, vox: float) -> dict:
    ny = f["image"].shape[1]; dev = f["image"].device
    sel = torch.zeros(f["image"].shape, dtype=torch.bool, device=dev); sel[inner] = True
    t1 = truth_field(tr, "dir1", ny).to(dev)
    r = {"n_populations": f["n_populations"]}
    if name == "crossing":
        t2 = truth_field(tr, "dir2", ny).to(dev)
        for k in FIELDS:
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
        for k in FIELDS:
            e = angle_deg(f[k], t1)
            r[k] = {b: _summary(e, s) for b, s in bands.items()}
    return r


def run(out_dir=None, seeds=(0, 1, 2), sigma: float = 1.0, rho: float = 10.0, engine: str = "anatomix_enc2", grid: Grid | None = None,
        settings=("calibrated",), device: str = "cuda", log=print) -> dict:
    """The validation at each texture setting: a calibration on twelve bundles, then the three cases over the seeds.
    Returns the scores; writes ``validation.json`` and one figure per setting into ``out_dir``."""
    t0 = time.time()
    grid = grid or Grid()
    region = torch.zeros(grid.shape, dtype=torch.bool, device=device); region[grid.inner] = True
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
                f = _fields(img, cal, sigma, rho, region, device)
                per_seed.append(score(name, tr, f, grid.inner, grid.vox))
                if s == seeds[0]:
                    panels.append((name, label, f, per_seed[-1]))
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


def _error_text(name: str, r: dict, key: str) -> str:
    """The panel's median error against the truth, from one seed's scores."""
    if name == "crossing":
        return f"({r[key + ' (nearest population)']['median_deg']:g}° to the nearest)"
    if name == "interface":
        return f"({r[key]['within 0.1 mm of the step']['median_deg']:g}° at the step)"
    return f"({r[key]['all']['median_deg']:g}°)"


def figure(panels, grid: Grid, path, sigma: float, rho: float, setting: str = ""):
    """The standard row per case on the mid-y plane of the scored core, with each field's median error in its title."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_act = min(max(max(f["n_populations"] for _, _, f, _ in panels), 1), 3)
    cols = 5 + n_act
    fig, axs = plt.subplots(len(panels), cols, figsize=(4.2 * cols, 4.7 * len(panels) + 0.4), dpi=100, squeeze=False, constrained_layout=True)
    y = grid.shape[1] // 2; zs, _, xs = grid.inner; sl = (zs, y, xs); dsl = (slice(None),) + sl
    for i, (name, label, f, r) in enumerate(panels):
        A = f["activation"]
        aref = float(A[:, zs, :, xs].float().reshape(-1)[::53].quantile(0.99).clamp_min(1e-6))
        acts = [A[k][sl].cpu().numpy() for k in range(min(f["n_populations"], n_act))]
        extra = {k: _error_text(name, r, key) for k, key in (("st", "excised structure tensor"), ("single", "shape tensor"), ("first", "shape tensor, multi (first)"))}
        pipeline_row(axs[i], f["image"][sl].cpu().numpy(), f["excised structure tensor"][dsl].cpu().numpy(), f["shape space"], acts,
                     f["shape tensor"][dsl].cpu().numpy(), f["shape tensor, multi (first)"][dsl].cpu().numpy(),
                     aref=aref, label=label.replace("bundle 60", "bundle at 60°").replace("crossing 60", "crossing at 60°").replace("interface 40", "interface, step 40"), extra=extra)
    fig.suptitle(f"Phantom validation ({setting} texture), σ {sigma:g}, ρ {rho:g} voxels: every field as DEC (R = |x|, G = |y|, B = |z|) on the mid plane; "
                 "median error against the truth in brackets", x=0.01, ha="left", fontsize=12)
    fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "validation")
