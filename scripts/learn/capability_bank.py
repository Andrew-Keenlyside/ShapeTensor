"""Capability banks: fixed sets of scene draws, one per capability, scored for any model that gives a density.

    python scripts/learn/capability_bank.py build --out DIR [--n 16] [--size 96]
    python scripts/learn/capability_bank.py score --bank DIR --models A.pt B.pt ... [--out FILE.json]

Banks (each ``n`` draws at ``size``³ from the object-scene generator, with the exact density, descriptor densities,
identities, points and voxel size of training):

* ``crossing`` -- two bundles crossing at an angle (interleaved or passing), kissing, fanning;
* ``collapse`` -- vessel trees with stenoses and collapses (one population whatever the lumen does);
* ``branching`` -- vessel trees, dendrites, glands, capillary networks (branch points);
* ``planar`` -- sheets, foams, porous walls, plywood lamellae, trabeculae, epithelia;
* ``blobs`` -- cells and crystals (no false lines);
* ``unresolved`` -- bundles of fibres far below the voxel (partial volume);
* ``physics_<mode>`` -- the same scenes under each imaging physics (additive, X-ray, dark-field, histology, PLI, MRI);
* ``snr_low`` / ``snr_high`` -- mixed scenes at acquisition strength 1.0 / 0.3;
* ``routes`` -- the same scenes through four contrast routes each (physics blends, orientation terms, scrambles,
  transfer functions, filters, missing lines and blocks): accuracy, and how far the top axis moves between routes.

Scores per bank (from the model's density -- lobes rendered, an SH head read directly -- with the calibrated peak
rule against the exact target's peaks): the single-population axis error and false-second rate, the crossing
detection rate and error, the dominant axis at crossings, the unexplained energy in order 2 and orders 4+; for a lobe
model also the lobes' own peaks, the descriptors at the target's top peak, identity separation and points.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from shapetensor.learn import train as TR
from shapetensor.learn import losses as _L
from shapetensor.learn.lobe_train import LB, peak_metrics, predict_sh, sh_peaks
from shapetensor.learn.losses import _cells
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.scene import Scene
from shapetensor.learn.sh import n_coef

DEV = "cuda"
ONLY = lambda *names: {k: (1.0 if k in names else 0.0) for k in Scene.FAMILIES}
BANKS = {
    "crossing": dict(families=ONLY("bundle"), n_groups=(2, 3), voxel_um=None, strength=0.5),
    "collapse": dict(families=ONLY("tree"), n_groups=(1, 2), voxel_um=None, strength=0.5),
    "branching": dict(families={**ONLY("tree", "dendrite", "gland", "network")}, n_groups=(1, 2), voxel_um=None, strength=0.5),
    "planar": dict(families=ONLY("sheet", "foam", "porous", "plywood", "trabecular", "epithelium"), n_groups=(1, 2), voxel_um=None, strength=0.5),
    "blobs": dict(families=ONLY("cells", "crystals"), n_groups=(1, 2), voxel_um=None, strength=0.5),
    "unresolved": dict(families=ONLY("bundle"), n_groups=(1, 3), voxel_um=40.0, strength=0.5),
    "snr_low": dict(families=None, n_groups=(2, 4), voxel_um=None, strength=1.0),
    "snr_high": dict(families=None, n_groups=(2, 4), voxel_um=None, strength=0.3),
}
PHYSICS = ("additive", "xray", "darkfield", "histology", "pli", "mri")


def _set_lobe_targets(lmax=8, desc_lmax=4):
    TR.HEAD, TR.DESC_KEY, TR.SH_EXACT, TR.DESC_LMAX, TR.QUERY, TR.CACHE = "lobes", "descriptors5", lmax, desc_lmax, None, None
    _L.DESC_K = 5; _L.DESC_NC = n_coef(desc_lmax)


def _crossing_cells(t, lmax=8, sep=30.0):
    """How many lattice cells of a target hold two calibrated peaks more than ``sep`` degrees apart."""
    ct = _cells(t["sh_t"][:n_coef(lmax)].float()[None], 4)[0]
    D, V = sh_peaks(ct, lmax)
    ang = torch.rad2deg(torch.acos((D[0] * D[1]).sum(0).abs().clamp(max=1)))
    return int(((V[1] > 0) & (ang > sep) & (ct[0].flatten() > 0.05 * math.sqrt(4 * math.pi))).sum())


def build(out: str, n: int = 16, size: int = 96, canvas: int = 192):
    _set_lobe_targets(); od = Path(out); od.mkdir(parents=True, exist_ok=True)
    for name, cfg in BANKS.items():
        if (od / f"{name}.pt").exists():
            print(name, "exists"); continue
        items, seed = [], 90_000_000 + 10_000 * list(BANKS).index(name)
        while len(items) < n and seed < 90_000_000 + 10_000 * (list(BANKS).index(name) + 1):
            gen = {"scene": {"canvas": canvas, "crops_per_scene": 1, "families": cfg["families"], "n_groups": list(cfg["n_groups"]),
                             "voxel_um": cfg["voxel_um"], "p_warp": 0.3}}
            img, t = TR._make_pair(seed, size, DEV, cfg["strength"], gen); seed += 1
            if name == "crossing" and _crossing_cells(t) < 30:                                        # the crossing must be in the crop
                continue
            if float(t["sh_t"][0].float().mean()) < 0.01 * math.sqrt(4 * math.pi):                  # nearly empty
                continue
            items.append((img.cpu(), {k: v.cpu() for k, v in t.items()}))
        torch.save(items, od / f"{name}.pt"); print(name, len(items), "draws", flush=True)
    # contrast routes: the same scene through four independent routes (draw-time blends, orientation terms, scrambles;
    # batch-time transfer functions, filters, fields, noise, missing lines and blocks), for accuracy and invariance
    if not (od / "routes.pt").exists():
        from shapetensor.learn.augment import contrast_route
        items = []
        for i in range(n):
            for k in range(4):
                gen = {"scene": {"canvas": canvas, "crops_per_scene": 1, "n_groups": [2, 4], "p_warp": 0.0, "p_blend": 0.5, "p_aniso": 0.5,
                                 "p_scramble": 0.5, "acq_seed": 91_000_000 + 10 * i + k, "crop": (canvas // 4,) * 3}}
                img, t = TR._make_pair(90_700_000 + i, size, DEV, 0.5, gen)
                x = contrast_route(img.float() / 255, np.random.default_rng(1000 * i + k), 1.0)
                t["route_scene"] = torch.tensor(i)
                items.append(((x * 255).round().to(torch.uint8).cpu(), {k_: v.cpu() for k_, v in t.items()}))
        torch.save(items, od / "routes.pt"); print("routes", len(items), flush=True)
    # the same scenes under each physics
    base = [90_500_000 + i for i in range(n)]
    for ph in PHYSICS:
        if (od / f"physics_{ph}.pt").exists():
            print("physics", ph, "exists"); continue
        items = []
        for sd in base:
            gen = {"scene": {"canvas": canvas, "crops_per_scene": 1, "n_groups": [2, 4], "physics": ph, "p_warp": 0.0, "modality": "mixed"}}
            img, t = TR._make_pair(sd, size, DEV, 0.5, gen)
            items.append((img.cpu(), {k: v.cpu() for k, v in t.items()}))
        torch.save(items, od / f"physics_{ph}.pt"); print("physics", ph, len(items), flush=True)


def load_model(path: str):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = PopulationNet.from_args(ck["args"]).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    return m, ck["args"]


@torch.no_grad()
def score_model(m, items, lmax=8, desc_lmax=4, margin=4) -> dict:
    from shapetensor.learn import lobe_train as LT
    lobes = getattr(m, "head", "slots") == "lobes" and LT.READOUT != "aux"
    P_sh, T_sh, M_t, L_d, L_m = [], [], [], [], []
    num2 = den2 = num4 = den4 = 0.0; desc_err = []; id_auc = []; pts_c = []
    nsh = n_coef(lmax)
    for img, t in items:
        x = img.to(DEV).float()[None, None] / 255
        v = t.get("voxel"); v = v.to(DEV).view(1) if v is not None else None
        f, fq, lob, P = predict_sh(m, x, voxel=v, lmax=lmax, desc_lmax=desc_lmax)
        lm = min(lmax, m.sh_lmax) if not lobes else lmax
        f = _cells(f, margin)[0]; ct = _cells(t["sh_t"].to(DEV)[None, :nsh].float(), margin)[0]
        if f.shape[0] < nsh:                                                                          # a lower-order SH head: pad
            f = torch.cat([f, torch.zeros(nsh - f.shape[0], *f.shape[1:], device=DEV)])
        num2 += float(((f[1:6] - ct[1:6]) ** 2).sum()); den2 += float((ct[1:6] ** 2).sum())
        num4 += float(((f[6:] - ct[6:]) ** 2).sum()); den4 += float((ct[6:] ** 2).sum())
        P_sh.append(f.flatten(1)); T_sh.append(ct.flatten(1)); M_t.append(ct[0].flatten() / math.sqrt(4 * math.pi))
        if lobes:
            if lob is not None:
                d, mm = LB.lobe_peaks({k: _cells(v_, margin) for k, v_ in lob.items()})
                L_d.append(d[0].flatten(2)[:3]); L_m.append(mm[0].flatten(1)[:3])
            if fq is not None:
                from shapetensor.learn.sh import peaks as _pk
                nd_ = n_coef(desc_lmax); ctf = _cells(t["sh_t"].to(DEV)[None].float(), margin)[0].flatten(1)
                strong = ctf[0] > 0.1 * math.sqrt(4 * math.pi)
                if bool(strong.any()):
                    cpf = torch.cat([f.flatten(1), _cells(fq, margin)[0].flatten(1)], 0)
                    td_, _ = _pk(ctf[:nsh, strong], lmax, nmax=1, rel=0.3); Yd = LB.sh_poly(td_[0].T, desc_lmax)
                    def at(cf):
                        f0 = (Yd * cf[:nd_, strong].T).sum(1).clamp_min(1e-6)
                        return torch.stack([(Yd * cf[nsh + j * nd_: nsh + (j + 1) * nd_, strong].T).sum(1) / f0 for j in range(m.nd)])
                    desc_err.append(at(cpf) - at(ctf))
            if "identity" in P and "identity_t" in t:
                e = _cells(P["identity"], 2 * margin)[0].float().flatten(1); ids = _cells(t["identity_t"].to(DEV)[None], 2 * margin)[0].flatten().long()
                ok = torch.nonzero(ids >= 0)[:, 0]
                if ok.numel() > 100:
                    gi = torch.Generator().manual_seed(0)
                    i = ok[torch.randint(0, ok.numel(), (4000,), generator=gi).to(DEV)]; j = ok[torch.randint(0, ok.numel(), (4000,), generator=gi).to(DEV)]
                    same = ids[i] == ids[j]; dist = (e[:, i] - e[:, j]).norm(dim=0)
                    if bool(same.any()) and bool((~same).any()):
                        id_auc.append(float((dist[~same][None, :1000] > dist[same][:1000, None]).float().mean()))
            if "points" in P and "points_t" in t:
                a_, b_ = _cells(P["points"], margin).float().flatten(), _cells(t["points_t"].to(DEV)[None], margin).float().flatten()
                if float(b_.std()) > 0:
                    pts_c.append(float(torch.corrcoef(torch.stack([a_, b_]))[0, 1]))
    out = {"unexplained_l2": num2 / max(den2, 1e-9), "unexplained_l4p": num4 / max(den4, 1e-9)}
    if all("route_scene" in t for _, t in items):                                                       # invariance across the routes of one scene
        from shapetensor.learn.sh import peaks_gated
        groups = {}
        for (img, t), f_ in zip(items, P_sh):
            groups.setdefault(int(t["route_scene"]), []).append(f_)
        diffs = []
        for fs in groups.values():
            tops = []
            for f_ in fs:
                D, V, _ = peaks_gated(f_, lmax, nmax=1); tops.append((D[0], V[0] > 0))
            for a in range(len(tops)):
                for b in range(a + 1, len(tops)):
                    ok = tops[a][1] & tops[b][1]
                    if bool(ok.any()):
                        diffs.append(torch.rad2deg(torch.acos((tops[a][0][:, ok] * tops[b][0][:, ok]).sum(0).abs().clamp(max=1))))
        if diffs:
            out["route_axis_change_deg"] = float(torch.cat(diffs).median())
    Pc, Tc, Mc = torch.cat(P_sh, 1), torch.cat(T_sh, 1), torch.cat(M_t)
    Dt, Vt = sh_peaks(Tc, lmax); Dp, Vp = sh_peaks(Pc, lmax)
    out.update({f"sh_{k}": v for k, v in peak_metrics(Dp, Vp, Dt, Vt, Mc).items()})
    if L_d:
        out.update({f"lobe_{k}": v for k, v in peak_metrics(torch.cat(L_d, -1), torch.cat(L_m, -1), Dt, Vt, Mc).items()})
    if desc_err:
        de = torch.cat(desc_err, 1)
        for j, nm in enumerate(("logsize", "wall", "lumen", "flat", "angular")[:de.shape[0]]):
            out[f"desc_{nm}_mae"] = float(de[j].abs().median())
    if id_auc:
        out["identity_auc"] = float(np.mean(id_auc))
    if pts_c:
        out["points_corr"] = float(np.mean(pts_c))
    return out


def score(bank: str, models: list, out: str | None = None):
    _set_lobe_targets()
    res = {}
    from shapetensor.learn import lobe_train as LT
    for spec in models:
        mp, _, ro = spec.partition(":")
        LT.READOUT = ro or "lobes"
        m, args = load_model(mp); key = Path(mp).parent.name + (f":{ro}" if ro else "")
        res[key] = {}
        for f in sorted(Path(bank).glob("*.pt")):
            items = torch.load(f, weights_only=False)
            res[key][f.stem] = score_model(m, items)
            r = res[key][f.stem]
            print(f"{key:28s} {f.stem:18s} " + " ".join(f"{k} {v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)
    if out:
        json.dump(res, open(out, "w"), indent=1); print("wrote", out)
    return res


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True); b.add_argument("--n", type=int, default=16); b.add_argument("--size", type=int, default=96)
    s = sub.add_parser("score"); s.add_argument("--bank", required=True); s.add_argument("--models", nargs="+", required=True); s.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.n, a.size)
    else:
        score(a.bank, a.models, a.out)


if __name__ == "__main__":
    main()
