"""QC of the scene generator's training targets before a long run (27 Oct): crops drawn exactly as training draws them
(``draw_scene`` with the training's scene settings, then ``train.entity_queries``), and per crop:

* every target finite; the image finite and in range;
* **mass**: each query's summed entity mass (mass x visibility) against the window mean of the rendered fill of the
  same layers (the targets are built from the canvas entries, the fill from the rendered layers: two routes to one
  quantity);
* **direction**: at queries holding one line population (any other below a fifth of it), the target's axis against
  the principal direction of the occupancy-weighted rendered tensors in the same window -- deformed scenes apart;
* **crossings**: the share of queries at crossing cores (two line populations, the lighter >= half the heavier) and,
  there, the angle between the two target axes;
* the crop inside the deformed canvas; the scene's kind (packed tissue, deformation) recorded.

Usage: python scripts/learn/qc_generator_targets.py OUT_STEM [--n 64]
"""
import argparse, json, math, sys
import numpy as np
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import scene as SC
from shapetensor.learn.model import window_mean

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=64); ap.add_argument("--seed0", type=int, default=110_000_000)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
TR.ENTITY["p_cross"] = 0.2
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True                                                                  # as train_entities sets it
from shapetensor.learn import lobe_train as LT                                              # the training's own settings: its
gen = LT.difficulty_gen(1.0, dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))   # family lists (full difficulty)


def sym(T6):
    return torch.stack([torch.stack([T6[..., 0], T6[..., 3], T6[..., 4]], -1), torch.stack([T6[..., 3], T6[..., 1], T6[..., 5]], -1),
                        torch.stack([T6[..., 4], T6[..., 5], T6[..., 2]], -1)], -2)


rows = []; bad = []
for i in range(a.n):
    seed = a.seed0 + 8 * (i // 2) + (i % 2)
    s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=1.0, entity_ctx=True, **gen)
    t = TR.entity_queries(s, d, 64, i, dev)
    rec = d["record"]; sc = SC._scene_cache()[next(reversed(SC._scene_cache()))][0]
    fin = all(bool(torch.isfinite(v.float()).all()) for k, v in t.items() if torch.is_tensor(v) and v.is_floating_point())
    img_ok = bool(torch.isfinite(d["image"].float()).all()) and float(d["image"].min()) >= 0 and float(d["image"].max()) <= 255
    if not (fin and img_ok):
        bad.append((seed, fin, img_ok))
    pts, sig = t["e_points"].float(), t["e_sigma"].float(); Q = pts.shape[0]
    node = t["e_node"]; m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * (node >= 0)
    ent_layers = [k for k, L in enumerate(s.layers) if L.role not in TR.NOT_ENTITY]
    occ = d["occupancy"][ent_layers].float()
    if len(ent_layers):
        wm = window_mean(occ, pts[None].expand(len(ent_layers), -1, -1).contiguous(), sig[None].expand(len(ent_layers), -1).contiguous()).T   # (Q, K)
        fill = wm.sum(1)
    else:
        fill = torch.zeros(Q, device=dev)
    msum = m.sum(1)
    ok = fill > 0.05
    ratio = (msum[ok] / fill[ok]).cpu().numpy() if bool(ok.any()) else np.array([])
    # direction at single-population line queries
    kap = t["e_kappa"].float(); line = (m >= 0.03) & (kap[..., 1] >= 2.0)
    ml = torch.where(line, m, torch.zeros_like(m)); srt, o = ml.sort(1, descending=True)
    single = (srt[:, 0] > 0.05) & (srt[:, 1] < 0.2 * srt[:, 0]) & ((m.sum(1) - srt[:, 0]) < 0.25 * srt[:, 0])
    core = (srt[:, 1] > 0.03) & (srt[:, 1] >= 0.5 * srt[:, 0])
    ar = torch.arange(Q, device=dev); ax1 = t["e_R"].float()[ar, o[:, 0], :, 2]; ax2 = t["e_R"].float()[ar, o[:, 1], :, 2]
    shp = [k for k, L in enumerate(s.layers) if L.role == "shape"]
    ang_single = np.array([]); ang_core = np.array([])
    if shp and bool(single.any()):
        Tw = (d["occupancy"][shp].float()[:, None] * d["tensors"][shp].float()).sum(0)        # (6, Z, Y, X) occupancy-weighted
        qs = pts[single]; ss_ = sig[single]
        tw = window_mean(Tw, qs[None].expand(6, -1, -1).contiguous(), ss_[None].expand(6, -1).contiguous()).T   # (q, 6)
        ev, V = torch.linalg.eigh(sym(tw).double().cpu() + 1e-12 * torch.eye(3, dtype=torch.float64)); v = V[..., -1].float().to(dev)
        ang_single = torch.rad2deg(torch.acos((v * ax1[single]).sum(-1).abs().clamp(max=1))).cpu().numpy()
    if bool(core.any()):
        ang_core = torch.rad2deg(torch.acos((ax1[core] * ax2[core]).sum(-1).abs().clamp(max=1))).cpu().numpy()
    o_ = rec["crop"]; z = rec["zoom"]
    valid = float(sc.deform.valid[tuple(slice(x, x + 64 * z) for x in o_)].float().mean()) if getattr(sc, "deform", None) is not None else 1.0
    rows.append({"seed": seed, "tissue": "tissue" in rec, "deform": bool(rec.get("deform")), "valid": valid, "finite": fin and img_ok,
                 "mass_ratio": ratio.tolist(), "ang_single": ang_single.tolist(), "core_share": float(core.float().mean()), "ang_core": ang_core.tolist(),
                 "voxel_um": rec.get("voxel_um"), "crop_fill": rec.get("crop_fill"), "sigma_max": float(sig.max())})
    if i % 2 == 1:
        SC._scene_cache().clear()
    if i % 8 == 7:
        print(f"{i + 1} crops", flush=True)

json.dump(rows, open(a.out + ".json", "w"), default=str)
cat = lambda key, sel=lambda r: True: np.concatenate([np.array(r[key]) for r in rows if sel(r)] or [np.array([])])
print(f"== {len(rows)} crops: tissue scenes {sum(r['tissue'] for r in rows)}, deformed {sum(r['deform'] for r in rows)}; non-finite {len(bad)} {bad[:3]}")
print(f"   crops fully inside the deformed canvas: {np.mean([r['valid'] > 0.999 for r in rows]) * 100:.0f} % (min {min(r['valid'] for r in rows):.3f})")
for nm, sel in (("all", lambda r: True), ("tissue", lambda r: r["tissue"]), ("deformed", lambda r: r["deform"]), ("undeformed", lambda r: not r["deform"])):
    mr = cat("mass_ratio", sel); an = cat("ang_single", sel)
    if len(mr) or len(an):
        print(f"   {nm:10s}: target mass / rendered fill median {np.median(mr):.3f} (p10 {np.quantile(mr, 0.1):.2f}, p90 {np.quantile(mr, 0.9):.2f}, n {len(mr)})"
              f" | single-population direction error median {np.median(an) if len(an) else float('nan'):.1f} deg (p90 {np.quantile(an, 0.9) if len(an) else float('nan'):.1f}, n {len(an)})")
ac = cat("ang_core")
print(f"   crossing-core queries {np.mean([r['core_share'] for r in rows]) * 100:.1f} %; angle between their two target axes median {np.median(ac) if len(ac) else float('nan'):.0f} deg,"
      f" share < 10 deg {np.mean(ac < 10) * 100 if len(ac) else float('nan'):.0f} % (n {len(ac)})")
print(f"   largest window sigma {max(r['sigma_max'] for r in rows):.1f}")
