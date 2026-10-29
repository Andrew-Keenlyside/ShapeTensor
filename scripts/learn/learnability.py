"""Is the training targets' orientation in the image? (27 Oct: the scene-trained density stayed flat while it formed on
phantoms and on easy scene data.) No model: at queries holding one line population (>= 80 % of the window's entity
mass), a classical structure tensor of the image at the query's window (gradient sigma 1, integration sigma = the
window) against the target's axis -- **consistent** where under a tenth of its gradient energy lies along the axis (the
image does not vary along the line: a planar boundary or a texture of strands both allow it), **determined** where the
tensor is linear and its line direction within 20 deg of the axis (28 Oct: the first measure, ``supported``, took the
least-gradient direction of planar evidence too, which is arbitrary within the plane). A target the image does not support is one no model can learn from that image; where
most are unsupported the per-direction loss's best answer is a flat density.

Crops drawn as training draws them (``draw_scene`` + ``entity_queries``, the curriculum's family lists), optionally
through a training contrast route; broken down by the crop's acquisition and content. Two mixes: ``full`` (difficulty
1, strength 1, routes 1) and ``easy`` (bundles and packed tissue, strength 0.3, no routes).

Usage: python scripts/learn/learnability.py OUT_STEM [--n 48]
"""
import argparse, json, math, sys
import numpy as np
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import scene as SC, lobe_train as LT
from shapetensor.learn.augment import contrast_route
from shapetensor.filters import gaussian

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=48); ap.add_argument("--mixes", nargs="+", default=["full", "easy"])
ap.add_argument("--no-refine", action="store_true", help="the renderer as before 28 Oct (no finer sweep of fat or wandering curves)")
a = ap.parse_args(); dev = "cuda"
if a.no_refine:
    SC.SWEEP_STEP = 1e9
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True


def st_M(x, pts, sig):
    """The image's structure tensor ``(Q, 3, 3)`` (gradient sigma 1) integrated over each query's window."""
    xs = gaussian(x[None, None], 1.0)[0, 0]; Gr = torch.stack(torch.gradient(xs), -1)
    prods = [Gr[..., i] * Gr[..., j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
    out = torch.zeros(pts.shape[0], 3, 3, dtype=torch.float64); sr = sig.round().clamp(1, 12)
    for s_ in torch.unique(sr).tolist():
        sel = sr == s_; P = pts[sel].round().long().clamp(0, x.shape[0] - 1)
        v = torch.stack([gaussian(p_[None, None], float(s_))[0, 0][P[:, 0], P[:, 1], P[:, 2]] for p_ in prods], -1).double().cpu()
        out[sel.cpu()] = torch.stack([torch.stack([v[:, 0], v[:, 3], v[:, 4]], -1), torch.stack([v[:, 3], v[:, 1], v[:, 5]], -1),
                                      torch.stack([v[:, 4], v[:, 5], v[:, 2]], -1)], -2)
    return out


def evidence(M, ax):
    """Per query: ``axis`` the share of the image's gradient energy along the target axis (0: the image never varies
    along it -- consistent with a line there; 1/3: no relation), ``determined`` the image's line direction (a linear
    structure tensor: two strong eigenvalues, (l2 - l1) / sum >= 0.15) within 20 deg of the axis."""
    ev, V = torch.linalg.eigh(M + 1e-12 * torch.eye(3, dtype=torch.float64)); tr = ev.sum(-1).clamp_min(1e-12)
    a = ax.double().cpu(); axis = torch.einsum("qi,qij,qj->q", a, M, a) / tr
    ang = torch.rad2deg(torch.acos((V[..., 0] * a).sum(-1).abs().clamp(max=1)))
    return axis.float(), (((ev[:, 1] - ev[:, 0]) / tr >= 0.15) & (ang < 20)).float()


def st_at(x, pts, sig):
    """Structure tensor's line direction and coherence at ``pts`` with per-point windows (grouped by rounded sigma)."""
    xs = gaussian(x[None, None], 1.0)[0, 0]; Gr = torch.stack(torch.gradient(xs), -1)
    prods = [Gr[..., i] * Gr[..., j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))]
    out_v = torch.zeros(pts.shape[0], 3, device=x.device); out_c = torch.zeros(pts.shape[0], device=x.device)
    sr = sig.round().clamp(1, 12)
    for s_ in torch.unique(sr).tolist():
        sel = sr == s_; P = pts[sel].round().long().clamp(0, x.shape[0] - 1)
        v = torch.stack([gaussian(p_[None, None], float(s_))[0, 0][P[:, 0], P[:, 1], P[:, 2]] for p_ in prods], -1)
        M = torch.stack([torch.stack([v[:, 0], v[:, 3], v[:, 4]], -1), torch.stack([v[:, 3], v[:, 1], v[:, 5]], -1),
                         torch.stack([v[:, 4], v[:, 5], v[:, 2]], -1)], -2).double().cpu()
        ev, V = torch.linalg.eigh(M + 1e-12 * torch.eye(3, dtype=torch.float64))
        out_v[sel] = V[..., 0].float().to(x.device); out_c[sel] = ((ev[:, 2] - ev[:, 0]) / ev.sum(-1).clamp_min(1e-12)).float().to(x.device)
    return out_v, out_c


base = dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA)
fw = {k: 0.0 for k in SC.Scene.FAMILIES}; fw["bundle"] = 1.0; fw["tissue"] = 1.0
easy = LT.difficulty_gen(1.0, {**base, "cross_p": 1.0, "p_centre": 1.0, "family_w": fw, "underlay_p": 0.0})
ALL = {"full": (LT.difficulty_gen(1.0, base), 1.0, 1.0), "easy": (easy, 0.3, 0.0),
       "easy_nograin": ({**easy, "p_grain": 0.0}, 0.3, 0.0),
       "easy_clean": ({**easy, "p_grain": 0.0, "p_texture": 0.0, "p_physics": 0.0, "p_warp": 0.0, "p_affine": 0.0}, 0.0, 0.0),
       "phantom": (None, 0.0, 0.0)}
mixes = {k: ALL[k] for k in a.mixes}
rows = []
from shapetensor.learn import phantom2 as P2
for mix, (gen, strength, routes) in mixes.items():
    for i in range(a.n):
        if mix == "phantom":                                                                       # the pilot's data: the phantom families
            fam = ("cross", "fan", "curve", "fine", "mix")[i % 5]
            spec, look = P2.random_spec(fam, torch.Generator().manual_seed(160_000_000 + i))
            ph = P2.make(spec, seed=160_100_000 + i, look=look, device=dev, ss=2 if fam == "fine" else 1); n = ph["n"]
            pts, sg = P2.draw_queries(ph, torch.Generator(device=dev).manual_seed(i), 384)
            tq = P2.targets(ph, pts, sg); K = ph["occ"].shape[0]
            t = {"e_points": pts, "e_sigma": sg, "e_mass": tq["e_mass"], "e_kappa": tq["e_kappa"], "e_R": tq["e_R"], "e_vis": torch.ones_like(tq["e_mass"]),
                 "e_node": torch.zeros_like(tq["e_mass"], dtype=torch.long)}
            rec = {"modality": "phantom", "physics": fam}; x = ph["image"].float()
        else:
            s, d = SC.draw_scene((64,) * 3, seed=150_000_000 + 8 * i, device=dev, strength=strength, entity_ctx=True, **gen)
            t = TR.entity_queries(s, d, 64, i, dev); rec = d["record"]
            x = d["image"].float() / 255
        if routes > 0:
            x = contrast_route(x, np.random.default_rng(i), routes)
        m = (t["e_mass"].float() * (t["e_node"] >= 0)).clamp_min(0); kap = t["e_kappa"].float(); vis = t["e_vis"].float()
        line = (m >= 0.03) & (kap[..., 1] >= 2.0)
        ml = torch.where(line, m, torch.zeros_like(m)); top, o = ml.max(1)
        single = (top > 0.05) & (top >= 0.8 * m.sum(1))
        if not bool(single.any()):
            SC._scene_cache().clear(); continue
        ar = torch.arange(m.shape[0], device=dev)
        ax = t["e_R"].float()[ar, o, :, 2][single]; vq = vis[ar, o][single]; kq = kap[ar, o, 1][single]
        v, c = st_at(x, t["e_points"].float()[single], t["e_sigma"].float()[single])
        ang = torch.rad2deg(torch.acos((v * ax).sum(-1).abs().clamp(max=1)))
        sup = (ang < 20) & (c >= 0.15)
        axe, det = evidence(st_M(x, t["e_points"].float()[single], t["e_sigma"].float()[single]), ax)
        dfm = rec.get("deform") or {}
        tags = {"mix": mix, "modality": rec.get("modality"), "physics": rec.get("physics"), "tissue": "tissue" in rec,
                "voxel_um": rec.get("voxel_um"), "heavy_psf": "heavy_psf" in rec, "grain": "grain" in (rec.get("routes") or []),
                "psf": float(np.sqrt(np.mean(np.square(rec.get("psf") or [0, 0, 0])))), "resolution": rec.get("resolution", 1.0),
                "deform": bool(dfm)}
        rows.append({**tags, "n": int(single.sum()), "supported": float(sup.float().mean()), "ang_med": float(ang.median()),
                     "consistent": float((axe < 0.1).float().mean()), "axis_med": float(axe.median()), "determined": float(det.mean()),
                     "coh_med": float(c.median()), "vis_med": float(vq.median()), "kappa2_med": float(kq.median()),
                     "sup_vis_hi": float(sup[vq > 0.8].float().mean()) if bool((vq > 0.8).any()) else float("nan")})
        SC._scene_cache().clear()
    print(f"{mix} done", flush=True)
json.dump(rows, open(a.out + ".json", "w"), default=str, indent=0)


def summ(sel, label):
    r = [x for x in rows if sel(x)]
    if not r:
        return
    w = np.array([x["n"] for x in r], dtype=float); avg = lambda k: np.average(np.array([x.get(k, np.nan) for x in r]), weights=w) * 100
    print(f"  {label:34s} crops {len(r):3d}  queries {int(w.sum()):5d}  consistent {avg('consistent'):5.1f} %  determined {avg('determined'):5.1f} %"
          f"  axis energy median {np.median([x.get('axis_med', np.nan) for x in r]):.3f}  (old: supported {avg('supported'):5.1f} %, angle {np.median([x['ang_med'] for x in r]):5.1f})")


for mix in mixes:
    print(f"== {mix}")
    summ(lambda x: x["mix"] == mix, "all")
    for mod in sorted({x["modality"] for x in rows if x["mix"] == mix}):
        summ(lambda x: x["mix"] == mix and x["modality"] == mod, f"modality {mod}")
    for ph in sorted({str(x["physics"]) for x in rows if x["mix"] == mix}):
        summ(lambda x: x["mix"] == mix and str(x["physics"]) == ph, f"physics {ph}")
    summ(lambda x: x["mix"] == mix and x["tissue"], "packed tissue")
    summ(lambda x: x["mix"] == mix and not x["tissue"], "other scenes")
    summ(lambda x: x["mix"] == mix and x["grain"], "grain on")
    summ(lambda x: x["mix"] == mix and not x["grain"], "grain off")
    summ(lambda x: x["mix"] == mix and x["psf"] > 1.0, "PSF rms > 1 voxel")
    summ(lambda x: x["mix"] == mix and x["psf"] <= 1.0, "PSF rms <= 1 voxel")
    for lo, hi in ((0, 1), (1, 10), (10, 100), (100, 1e9)):
        summ(lambda x: x["mix"] == mix and x["voxel_um"] is not None and lo <= x["voxel_um"] < hi, f"voxel {lo}-{hi} um")
    vh = [x["sup_vis_hi"] for x in rows if x["mix"] == mix and np.isfinite(x["sup_vis_hi"])]
    print(f"  supported where the generator calls the orientation visible (vis > 0.8): {np.mean(vh) * 100:.1f} % (crop mean)")
