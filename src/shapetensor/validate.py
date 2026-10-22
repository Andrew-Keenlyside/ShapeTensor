"""The acceptance suite: phantoms with truth, the chain against the excised structure tensor, field metrics and the
tracking metrics the design is for -- a tracker leaves a crossing on the bundle it entered.

    shapetensor validate [--out DIR] [--seeds 1] [--cases band,bend] [--scale 10]

Per phantom (the core scored, the margins kept as context): N per truth category on the lattice, directions (single:
median error; crossings: pair error and both within 15°), identities (dominant label coverage per truth bundle and its
leakage into the other), the single field at every voxel against the structure tensor, and tracking: streamlines
seeded in one arm and scored by where they end -- *through* (the same bundle's far arm), *switched* (the other
bundle), stopped in the overlap; the structure tensor is tracked by the same tracker from the same seeds. The texture
is the one matched to a real 20 µm HiP-CT crop (``SETTINGS["calibrated"]``).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import st
from .field import Orientations
from .phantoms import Grid, TextureParams, bend_geometry, build, direction
from .tracking import REASONS, track

ANGLES = np.arange(0, 180, 15.0)
SETTINGS = {"calibrated": {"dispersion_deg": 20.0, "pitch_deg": 10.0, "texture": 0.5},    # the dispersion and texture level matched to the real crop
            "coherent": {"dispersion_deg": 8.0, "pitch_deg": 4.0, "texture": 1.0}}        # tight bundles at the full texture level (optional)
GRID_BEND = Grid(core=(192, 96, 192))                                                      # the bend / band layout, thick enough in y to track in
GRID = Grid(core=(128, 96, 128))

CASES = (("bundle 60", "bundle", {"angle": 60.0}), ("bend", "bend", {}), ("straight", "bend", {"straight": True}),
         ("band crossing 90", "band_crossing", {"cross_deg": 90.0}), ("band crossing 60", "band_crossing", {"cross_deg": 60.0}),
         ("band crossing 45", "band_crossing", {"cross_deg": 45.0}),
         ("T-junction 90", "band_crossing", {"cross_deg": 90.0, "t_junction": True}),
         ("free end", "band_crossing", {"cross_deg": 90.0, "t_junction": True, "only_second": True}),
         ("crossing 90", "crossing", {"cross_deg": 90.0}), ("crossing 60", "crossing", {"cross_deg": 60.0}), ("crossing 45", "crossing", {"cross_deg": 45.0}),
         ("interface 40", "interface", {"step": 40.0}), ("texture", "texture", {}))


def tex_for(setting: str = "calibrated"):
    p = SETTINGS[setting]
    return TextureParams(dispersion_deg=p["dispersion_deg"], pitch_deg=p["pitch_deg"]), p["texture"]


def angle_deg(d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Axial angle (deg) between direction fields ``(3, ...)``; NaN where the truth is absent."""
    return torch.rad2deg(torch.arccos((d * t).sum(0).abs().clamp(max=1)))


# ---------------------------------------------------------------------------------------------------------------
# the lattice
# ---------------------------------------------------------------------------------------------------------------

def cell_truth(tr: dict, cs, step: int, origin: int, dev):
    """The truth at the lattice cells: ``npop``, ``d1``, ``d2`` ``(3, ...)`` (zero where absent) and extras."""
    zc = np.clip(np.arange(cs[0]) * step + origin, 0, tr["npop"].shape[0] - 1); xc = np.clip(np.arange(cs[2]) * step + origin, 0, tr["npop"].shape[1] - 1)
    npop = torch.as_tensor(tr["npop"][np.ix_(zc, xc)], device=dev).long()[:, None, :].expand(cs)
    d1 = torch.as_tensor(tr["dir1"][np.ix_(zc, xc)], device=dev).permute(2, 0, 1)[:, :, None, :].expand(3, *cs).nan_to_num(0)
    d2 = torch.as_tensor(tr["dir2"][np.ix_(zc, xc)], device=dev).permute(2, 0, 1)[:, :, None, :].expand(3, *cs).nan_to_num(0)
    extra = {}
    if "corridor2" in tr:
        extra["beyond"] = torch.as_tensor(tr["corridor2"][np.ix_(zc, xc)] & ~tr["band1"][np.ix_(zc, xc)] & ~tr["band2"][np.ix_(zc, xc)], device=dev)[:, None, :].expand(cs)
    for k in ("band1", "band2"):
        if k in tr:
            extra[k] = torch.as_tensor(tr[k][np.ix_(zc, xc)], device=dev)[:, None, :].expand(cs)
    if "fid" in tr:
        extra["fid"] = torch.as_tensor(tr["fid"][np.ix_(zc, xc)], device=dev)[:, None, :].expand(cs)
    return npop, d1, d2, extra


def _inner_cells(grid: Grid, cs, step: int, dev):
    zs, ys, xs = grid.inner
    m = torch.zeros(cs, dtype=torch.bool, device=dev)
    m[zs.start // step:zs.stop // step, ys.start // step:ys.stop // step, xs.start // step:xs.stop // step] = True
    return m


def lattice_scores(pop: dict, tr: dict, grid: Grid, meta: dict, model: Orientations) -> dict:
    """N per truth category, the top lobe's error on single-population cells, the pair error and both-within-15° on
    two-population cells, the false extension beyond a T."""
    N, mass, D = pop["N_cells"] if "N_cells" in pop else pop["N"], pop["mass"], pop["dirs_cells"]; cs = N.shape; dev = N.device
    inner = _inner_cells(grid, cs, model.step, dev)
    npop, d1, d2, extra = cell_truth(tr, cs, model.step, model.origin, dev)
    r = {}
    for cat, k in (("none", 0), ("one", 1), ("two", 2)):
        sel = inner & (npop == k)
        if int(sel.sum()) == 0:
            continue
        rr = {"cells": int(sel.sum()), **{f"N={j}": round(float((N[sel] == j).float().mean()), 3) for j in range(4)}}
        if k == 1:
            s1 = sel & (N >= 1)
            if int(s1.sum()):
                rr["top lobe error median deg"] = round(float(angle_deg(D[0], d1)[s1].median()), 1)
        if k == 2:
            s2 = sel & (N >= 2)
            if int(s2.sum()):
                e11, e12, e21, e22 = angle_deg(D[0], d1), angle_deg(D[0], d2), angle_deg(D[1], d1), angle_deg(D[1], d2)
                sw = (e12 + e21) < (e11 + e22); ea, eb = torch.where(sw, e12, e11), torch.where(sw, e21, e22)
                rr["pair error median deg"] = round(float(((ea + eb) / 2)[s2].median()), 1)
                rr["both within 15 deg"] = round(float(((ea <= 15) & (eb <= 15) & (N >= 2))[sel].float().mean()), 3)
        r[cat] = rr
    if "beyond" in extra:
        sel = inner & extra["beyond"]
        if int(sel.sum()):
            b2 = torch.as_tensor(direction(meta["angles"][1]), device=dev).view(3, 1, 1, 1).expand(3, *cs)
            hit = torch.zeros(cs, dtype=torch.bool, device=dev)
            for j in range(D.shape[0]):
                hit |= (mass[j] > 0) & (angle_deg(D[j], b2) <= 20)
            r["false extension beyond the T"] = round(float(hit[sel].float().mean()), 3)
    return r


def identity_scores(pop: dict, tr: dict, kind: str, grid: Grid, model: Orientations) -> dict:
    """Dominant label per truth bundle on the lattice: its coverage of the bundle's cells and its leakage into the
    other bundle's cells (band crossings), or the bend's fascicles as one bundle."""
    if "labels_cells" not in pop or kind not in ("band_crossing", "bend"):
        return {}
    L = pop["labels_cells"]; cs = L.shape[1:]; dev = L.device
    inner = _inner_cells(grid, cs, model.step, dev)
    _, _, _, extra = cell_truth(tr, cs, model.step, model.origin, dev)
    regions = {"bend": (extra["fid"] > 0, None)} if kind == "bend" else {"band1": (extra["band1"], extra["band2"]), "band2": (extra["band2"], extra["band1"])}
    out = {}
    for name, (reg, other) in regions.items():
        sel = inner & reg
        labs = L[:, sel].flatten(); labs = labs[labs > 0]
        if labs.numel() == 0:
            out[name] = {"coverage": 0.0}; continue
        dom = int(torch.mode(labs).values)
        r = {"label": dom, "coverage": round(float(((L == dom).any(0) & sel).sum() / sel.sum().clamp_min(1)), 3), "labels in bundle": int(labs.unique().numel())}
        if other is not None:
            o = inner & other & ~reg
            r["leakage"] = round(float(((L == dom).any(0) & o).sum() / o.sum().clamp_min(1)), 3)
        out[name] = r
    return out


# ---------------------------------------------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------------------------------------------

def _core(grid: Grid, shape, dev):
    m = torch.zeros(shape, dtype=torch.bool, device=dev); zs, ys, xs = grid.inner; m[zs, ys, xs] = True
    return m


def _plane(a, ny, dev):
    a = torch.as_tensor(np.asarray(a), device=dev)
    return a[:, None, :].expand(a.shape[0], ny, a.shape[1])


def _sample(mask: torch.Tensor, n: int, g: torch.Generator) -> torch.Tensor:
    idx = mask.nonzero()
    if idx.shape[0] == 0:
        return idx.float()
    return idx[torch.randperm(idx.shape[0], generator=g)[:n].to(idx.device)].float()


def _ends(res):
    pts, n = res["points"], res["n_steps"]; N = pts.shape[0]
    return torch.stack([pts[torch.arange(N), h, n[:, h]] for h in (0, 1)], 1)


def _at(mask, e):
    return mask[e[..., 0], e[..., 1], e[..., 2]]


def tracking_scores(kind: str, meta: dict, tr: dict, fld: dict, d_st: torch.Tensor, grid: Grid, g: torch.Generator, *, n_seeds: int = 400,
                    step: float = 0.5, max_angle: float = 30.0) -> dict:
    """Streamlines from one arm in the chain's co-located field (a vector is followed where its weight is above zero)
    and in the structure tensor's field, scored by where they end."""
    shape = tuple(fld["vec"].shape[1:]); ny = shape[1]; dev = fld["vec"].device; core = _core(grid, shape, dev); vox = grid.vox
    Zm, Xm = torch.meshgrid(torch.as_tensor(grid.coords(0), device=dev).float(), torch.as_tensor(grid.coords(2), device=dev).float(), indexing="ij")
    fields = (("shape", fld["vecs"], fld["weights"]), ("structure tensor", d_st[None], None))
    run = lambda dirs, w, seeds, **kw: track(dirs, core, seeds, step=step, max_angle_deg=max_angle, max_steps=int(6 * max(shape) / step),
                                             weights=w, min_weight=1e-6 if w is not None else 0.0, **kw)
    clamp = lambda e: torch.stack([e[..., i].round().long().clamp(0, shape[i] - 1) for i in range(3)], -1)
    out = {}
    if kind == "band_crossing":
        a1, a2 = meta["angles"]; w_mm = meta["width_mm"]
        d1 = torch.as_tensor(direction(a1), device=dev); d2 = torch.as_tensor(direction(a2), device=dev)
        zc, xc = grid.core_mm[0] / 2, grid.core_mm[2] / 2
        t2 = (Zm - zc) * d2[0] + (Xm - xc) * d2[2]                                                  # position along band 2 (0 at the crossing)
        b1 = _plane(tr["band1"], ny, dev); b2 = _plane(tr["band2"], ny, dev); corr2 = _plane(tr["corridor2"], ny, dev)
        T2 = _plane(t2.cpu().numpy(), ny, dev)
        stub = b2 & ~b1 & core & (T2.abs() > w_mm)                                                 # band 2 away from the crossing
        if meta.get("t_junction"):
            near = stub                                                                            # the stopped band itself
            far = corr2 & ~b2 & ~b1 & (T2.abs() > w_mm / 2)                                        # where it would have continued
        else:
            near = stub & (T2 < 0); far = (T2 > w_mm) & ~b1                                        # beyond the crossing, on band 2's side
        seeds = _sample(near, n_seeds, g)
        if seeds.shape[0] == 0:
            return out
        for name, dirs, w in fields:
            res = run(dirs, w, seeds); e = clamp(_ends(res))
            in_far, in_b1, in_over = _at(far, e), _at(b1 & ~b2, e), _at(b1 & b2, e)
            why = {REASONS[k]: round(float((res["reason"] == k).float().mean()), 3) for k in range(len(REASONS)) if bool((res["reason"] == k).any())}
            through = float(in_far.any(1).float().mean())
            r = {"seeds": int(seeds.shape[0]), "length mm median": round(float(res["length"].median()) * vox, 3), "stop reasons": why}
            if meta.get("only_second"):
                r["beyond the end"] = round(through, 3)
            elif meta.get("t_junction"):
                r["false continuation"] = round(through, 3); r["into band 1"] = round(float(in_b1.any(1).float().mean()), 3)
            else:
                r.update({"through": round(through, 3), "switched": round(float((~in_far.any(1) & in_b1.any(1)).float().mean()), 3),
                          "stopped in overlap": round(float((~in_far.any(1) & in_over.any(1)).float().mean()), 3)})
            out[name] = r
    elif kind == "bend":
        fid, tan, s_, arc = bend_geometry(grid, straight=meta.get("straight", False))
        fid_t = _plane(fid, ny, dev); zc_, xc_ = meta["centre_mm"]
        beyond = _plane(((Xm > xc_ + 0.4) & (Zm > zc_)).cpu().numpy(), ny, dev) if not meta.get("straight") else _plane((Zm > zc_ + 1.2).cpu().numpy(), ny, dev)
        s_t = _plane(np.nan_to_num(s_, nan=1e9), ny, dev)
        for k, r0 in enumerate(meta["radii_mm"], 1):
            arm_in = (fid_t == k) & (s_t < -0.4) & core
            seeds = _sample(arm_in, n_seeds // 3, g)
            if seeds.shape[0] == 0:
                continue
            for name, dirs, w in fields:
                res = run(dirs, w, seeds); e = clamp(_ends(res))
                out.setdefault(name, {})[f"fascicle {k}"] = {"seeds": int(seeds.shape[0]), "through the turn": round(float(_at(beyond, e).any(1).float().mean()), 3),
                                                             "other fascicle": round(float(_at((fid_t > 0) & (fid_t != k), e).any(1).float().mean()), 3),
                                                             "length mm median": round(float(res["length"].median()) * vox, 2)}
    elif kind == "crossing":
        a1, a2 = meta["angles"]; zs, ys, xs = grid.inner
        inner = core.clone(); inner[:zs.start + 16] = False; inner[zs.stop - 16:] = False; inner[:, :, :xs.start + 16] = False; inner[:, :, xs.stop - 16:] = False
        seeds = _sample(inner, n_seeds, g)
        for pname, a in (("population 1", a1), ("population 2", a2)):
            d = torch.as_tensor(direction(a), device=dev); h = d[None].expand(seeds.shape[0], 3)
            for name, dirs, w in fields:
                res = run(dirs, w, seeds, headings=h); e = _ends(res)
                v = e[:, 0] - e[:, 1]; L = v.norm(dim=1)
                dev_deg = torch.rad2deg(torch.acos(((v / L.clamp_min(1e-9)[:, None]) * d[None]).sum(1).abs().clamp(max=1.0)))
                long = L * vox >= 0.5 * min(grid.core_mm[0], grid.core_mm[2])
                out.setdefault(name, {})[pname] = {"seeds": int(seeds.shape[0]), "within 15 deg of the population end to end": round(float((dev_deg <= 15).float().mean()), 3),
                                                   "and at least half the core long": round(float((long & (dev_deg <= 15)).float().mean()), 3),
                                                   "end-to-end deviation median deg": round(float(dev_deg.median()), 1), "length mm median": round(float(L.median()) * vox, 2)}
    return out


# ---------------------------------------------------------------------------------------------------------------

def run(out_dir=None, *, seeds: int = 1, cases: str = "", scale: float = 10.0, setting: str = "calibrated", device: str = "cuda", log=print, **params) -> dict:
    """The suite; writes ``acceptance.json`` to ``out_dir`` and returns the results. ``params``: :class:`Orientations` overrides."""
    t0 = time.time(); out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    tex, texture = tex_for(setting)
    sel_cases = [c for c in CASES if not cases or any(k in c[0] for k in cases.split(","))]
    model = Orientations(scale=scale, device=device, **params)
    results = {"settings": {"seeds": seeds, "scale": scale, "setting": setting, **params}, "cases": {}}
    for seed in range(1, 1 + seeds):
        for name, kind, kw in sel_cases:
            grid = GRID_BEND if kind in ("bend", "band_crossing") else GRID
            img, tr, meta = build(kind, seed, grid, tex=tex, texture=texture, device=device, **kw)
            v = torch.as_tensor(img, device=device).float()
            t1 = time.time(); fld = model.run(v); t_chain = time.time() - t1
            t1 = time.time(); d_st = st.structure_tensor(v, sigma=1.0, rho=scale, truncate=3.0, excise=True, order="zyx", device=device); t_st = time.time() - t1
            r = {"seconds": {"chain": round(t_chain, 1), "structure tensor": round(t_st, 2)}, "floor": round(fld["floor"], 4),
                 "lattice": lattice_scores(fld, tr, grid, meta, model)}
            if kind != "texture":
                r["identities"] = identity_scores(fld, tr, kind, grid, model)
                core = _core(grid, v.shape, device)
                d1 = torch.as_tensor(tr["dir1"], device=device).permute(2, 0, 1)[:, :, None, :].expand(3, *v.shape)
                ok = core & ~torch.isnan(d1[0]); e_sh = angle_deg(fld["vec"], d1.nan_to_num(0)); e_st = angle_deg(d_st, d1.nan_to_num(0))
                s = ok & (fld["weight"] >= 0.2)
                r["single field"] = {"shape median deg (weight >= 0.2)": round(float(e_sh[s].median()), 1) if bool(s.any()) else None,
                                     "structure tensor median deg (same voxels)": round(float(e_st[s].median()), 1) if bool(s.any()) else None,
                                     "shape median deg (all truth voxels)": round(float(e_sh[ok].median()), 1), "structure tensor (all)": round(float(e_st[ok].median()), 1),
                                     "voxels": int(s.sum())}
                r["tracking"] = tracking_scores(kind, meta, tr, fld, d_st, grid, torch.Generator().manual_seed(seed))
            else:
                r["N on texture (voxels)"] = {f"N={k}": round(float((fld["N"][_core(grid, v.shape, device)] == k).float().mean()), 4) for k in range(3)}
            results["cases"][f"{name} (seed {seed})"] = r
            log(f"[{time.time() - t0:5.0f}s] {name} seed {seed}: chain {t_chain:.1f}s | lattice {json.dumps(r['lattice'])}")
            for k in ("single field", "identities", "tracking", "N on texture (voxels)"):
                if r.get(k):
                    log(f"        {k}: {json.dumps(r[k])}")
            if out:
                json.dump(results, open(out / "acceptance.json", "w"), indent=1)
            del fld, d_st; torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else None)
