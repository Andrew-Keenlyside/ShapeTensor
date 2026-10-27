"""Monte Carlo random fibre architecture in the scene generator (:mod:`~shapetensor.learn.fibres`): example input images
and the crossing-core check (plan_full_generator.md, W1).

Each crossing scene is drawn three ways, crops centred on the crossing:

* ``current``  -- the generator as trained so far (hexagonal packing; ``interleave`` on some angled crossings; crossing
  fibres otherwise share voxels that the compositor clamps to one);
* ``MC``       -- Monte Carlo packing (dart throwing, gamma radii) and wandering fibres, **not** relaxed (same geometry
  as the next row before the relaxation);
* ``MC+relax`` -- the same fibres after the collision relaxation (crossing bundles interweave).

Per draw, on the plane through the crossing core spanned by both bundles' axes (resampled): the labels' direction map
(per-voxel principal axis of the populations' occupancy-weighted tensors as RGB, z y x -> R G B, brightness the
material) and the clean image (additive physics, no acquisition); and two numbers -- the rendered overlap (share of fibre
volume in voxels holding more than one voxel of material) and the classical check at the crossing core (Hessian line
directions at the fibre scale: each bundle's axis error, :mod:`~shapetensor.learn.inferability`) on the clean image and
on the acquired one (HiP-CT, ``--strength``). MC+relax is also shown acquired under several modalities. Scenes whose
crossing bundles' fibres are not resolved (radius outside 0.6-3 voxels) or cross below ``--min-angle`` are skipped.

Usage: python scripts/learn/mc_fibres_examples.py OUT_STEM [--scenes 4] [--stats 12] [--voxel-um 1.0]
"""
import argparse, json, math, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from shapetensor.filters import gaussian
from shapetensor.learn import scene as SC, fibres as FB, inferability as IF, entity_infer as EI

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--scenes", type=int, default=4); ap.add_argument("--stats", type=int, default=12)
ap.add_argument("--voxel-um", type=float, default=1.0); ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--strength", type=float, default=0.5)
ap.add_argument("--min-angle", type=float, default=40.0)
a = ap.parse_args(); dev = "cuda"
SPH = EI.sphere_dirs(1000, dev)
MODES = ("current", "MC", "MC+relax"); EXTRA = ("histology", "mri", "fluorescence")
KW = dict(shape=(64,) * 3, device=dev, canvas=128, crops_per_scene=1, voxel_um=a.voxel_um, families={"bundle": 1.0}, n_groups=(2, 2),
          cross_p=1.0, p_centre=1.0, underlay_p=0.0, p_sparse=0.0, p_cutout=0.0, p_boundary=0.0, p_warp=0.0, p_affine=0.0)
CLEAN = dict(physics="additive", strength=0.0, p_blend=0.0, p_aniso=0.0, p_scramble=0.0, p_texture=0.0, p_merge=0.0, p_grain=0.0)


def set_mode(m):
    FB.PACKING, FB.RELAX = ("hex", False) if m == "current" else ("random", True)
    FB._SKIP = (m == "MC")


_orig = FB.relax_scene
FB._SKIP = False
FB.relax_scene = lambda sc, **kw: {} if FB._SKIP else _orig(sc, **kw)


def draw(seed, mode, modality="hipct", clean=False):
    set_mode(mode); SC._CACHE.clear()
    s, d = SC.draw_scene(seed=seed, modality=modality, **KW, **(CLEAN if clean else {"strength": a.strength}))
    sc = SC._scene_cache()[next(iter(SC._scene_cache()))][0]
    return s, d, sc


def tensor33(t6):
    T = torch.zeros(*t6.shape[1:], 3, 3, device=t6.device)
    for (i, j), c in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
        T[..., i, j] = t6[c]; T[..., j, i] = t6[c]
    return T


def crossing(s, d):
    """The two heaviest line populations, their core (both smoothed occupancies above a quarter of their maxima), each
    one's axis there, and the slice axis most across both."""
    keep = [k for k, L_ in enumerate(s.layers) if L_.role == "shape" and L_.kind == "line"]
    if len(keep) < 2:
        return None
    occ = d["occupancy"][keep].float()
    mass = occ.sum((1, 2, 3)); top = mass.argsort(descending=True)[:2].tolist()
    A, B = keep[top[0]], keep[top[1]]
    sm = [gaussian(d["occupancy"][k].float()[None, None], 2.0)[0, 0] for k in (A, B)]
    core = (sm[0] > 0.25 * sm[0].max()) & (sm[1] > 0.25 * sm[1].max())
    if int(core.sum()) < 50:
        return None
    ax = []
    for k in (A, B):
        T = (d["occupancy"][k].float()[core][:, None, None] * tensor33(d["tensors"][k].float())[core]).sum(0)
        ax.append(torch.linalg.eigh(T)[1][:, -1])
    ang = math.degrees(math.acos(min(1.0, abs(float(ax[0] @ ax[1])))))
    sa = int(torch.minimum(1 - ax[0] ** 2, 1 - ax[1] ** 2).argmax())
    cz = torch.nonzero(core).float().mean(0)
    return {"A": A, "B": B, "core": core, "axes": torch.stack(ax), "angle": ang, "sa": sa, "si": int(round(float(cz[sa]))), "centre": cz,
            "radius": float(np.mean([s.layers[A].radius, s.layers[B].radius])), "radii": (s.layers[A].radius, s.layers[B].radius)}


def plane(vol, cx, half=31):
    """``vol (C, Z, Y, X)`` resampled on the plane through the core's centre spanned by the two axes: ``(2h+1, 2h+1, C)``."""
    A = cx["axes"][0]; B = cx["axes"][1] - (cx["axes"][1] @ A) * A; B = B / B.norm()
    t = torch.arange(-half, half + 1, device=dev, dtype=torch.float32)
    P = cx["centre"] + t[:, None, None] * A + t[None, :, None] * B                       # (u, v, 3) voxel coordinates (z, y, x)
    n = torch.tensor(vol.shape[-3:], device=dev, dtype=torch.float32)
    grid = (2 * P / (n - 1) - 1).flip(-1)[None, None]                                     # grid_sample wants (x, y, z) in [-1, 1]
    out = torch.nn.functional.grid_sample(vol[None].float(), grid, align_corners=True, padding_mode="zeros")[0, :, 0]
    return out.permute(1, 2, 0).cpu().numpy()


def dec_vol(s, d):
    keep = [k for k, L_ in enumerate(s.layers) if L_.role == "shape"]
    occ = d["occupancy"][keep].float(); T = (occ[:, None] * d["tensors"][keep].float()).sum(0)
    v = torch.linalg.eigh(tensor33(T))[1][..., -1].abs()
    return (v * occ.sum(0).clamp(0, 1)[..., None]).permute(3, 0, 1, 2)


def overlap(sc):
    cur = [c for g_ in sc.groups if g_.name in FB.MOVABLE for c in g_.curves if c.fibre_r is None]
    return FB.overlap_fraction(cur, sc.shape, dev) if cur else float("nan")


def check(d, cx):
    sig = float(np.clip(cx["radius"], 0.7, 2.0))
    dens = IF.hessian_line_dirs(d["image"].float(), cx["core"], SPH, sigma=sig)
    est, _ = IF.peak_axes(dens, SPH, k=2)
    at = lambda u: float(dens[(SPH @ u).abs().argmax()]) * SPH.shape[0]
    return {"err": [round(e, 1) for e in IF.axis_errors(est, cx["axes"])], "dens": [round(at(cx["axes"][0]), 1), round(at(cx["axes"][1]), 1)]}


# ---- find crossing scenes (angle >= min_angle, resolved fibres, in every mode), stats and figure rows
found, stats, seed = [], [], a.seed0
t0 = time.time()
while len(stats) < max(a.stats, a.scenes) and seed < a.seed0 + 600:
    seed += 1
    rec = {"seed": seed}; ok = True; rows = {}
    for m in MODES:
        t = time.time(); s, d, sc = draw(seed, m, clean=True)
        cx = crossing(s, d)
        if cx is None or cx["angle"] < a.min_angle or not all(0.6 <= r_ <= 3.0 for r_ in cx["radii"]):
            ok = False; break
        s2, d2, _ = draw(seed, m)
        cx2 = crossing(s2, d2) or cx
        rec[m] = {"angle": round(cx["angle"], 1), "overlap": round(overlap(sc), 4), "clean": check(d, cx), "acquired": check(d2, cx2),
                  "s": round(time.time() - t, 1), "relax": sc.record.get("relax")}
        rows[m] = (plane(dec_vol(s, d), cx), plane(d["image"][None], cx)[..., 0], plane(d2["image"][None], cx2)[..., 0])
        if m == "MC+relax":
            for mod in EXTRA:
                s3, d3, _ = draw(seed, m, modality=mod)
                rows[mod] = plane(d3["image"][None], crossing(s3, d3) or cx)[..., 0]
    if not ok:
        continue
    stats.append(rec)
    print(json.dumps(rec), flush=True)
    if len(found) < a.scenes:
        found.append((seed, rec, rows))

summ = {}
for m in MODES:
    summ[m] = {"overlap_median": float(np.median([r[m]["overlap"] for r in stats])), "n": len(stats),
               "seconds_median": float(np.median([r[m]["s"] for r in stats]))}
    for k in ("clean", "acquired"):
        er = [max(r[m][k]["err"]) for r in stats]
        summ[m][k] = {"worse_axis_err_median": float(np.median(er)), "both_within_15deg": int(sum(e <= 15 for e in er))}
print("summary", json.dumps(summ), f"({time.time() - t0:.0f}s)", flush=True)
json.dump({"scenes": stats, "summary": summ}, open(a.out + ".json", "w"), indent=1)

# ---- figure: a row per scene
cols = [(m, "dirs") for m in MODES] + [(m, "clean") for m in MODES] + [("MC+relax", "hipct")] + [(mod, "img") for mod in EXTRA]
fig, axs = plt.subplots(len(found), len(cols), figsize=(2.2 * len(cols), 2.4 * len(found) + 0.7), squeeze=False, layout="constrained")
for i, (seed, rec, rows) in enumerate(found):
    for j, (m, what) in enumerate(cols):
        ax_ = axs[i, j]
        if what == "dirs":
            ax_.imshow(np.clip(rows[m][0], 0, 1), interpolation="nearest"); ax_.set_title(f"{m}: labels\noverlap {rec[m]['overlap']:.3f}", fontsize=7)
        elif what == "clean":
            ax_.imshow(rows[m][1], cmap="gray", interpolation="nearest"); ax_.set_title(f"{m}: clean image\ncore err {rec[m]['clean']['err']}", fontsize=7)
        elif what == "hipct":
            ax_.imshow(rows[m][2], cmap="gray", interpolation="nearest"); ax_.set_title(f"MC+relax: HiP-CT\ncore err {rec[m]['acquired']['err']}", fontsize=7)
        else:
            ax_.imshow(rows[m], cmap="gray", interpolation="nearest"); ax_.set_title(f"MC+relax: {m}", fontsize=7)
        ax_.set_xticks([]); ax_.set_yticks([])
    axs[i, 0].set_ylabel(f"seed {seed}\ncross {rec['MC+relax']['angle']:.0f} deg", fontsize=8)
fig.suptitle("Crossing scenes on the plane of both bundles: current generator vs Monte Carlo random fibres (unrelaxed, relaxed). "
             "Labels: direction RGB. Core err: classical Hessian line-direction error per bundle (deg) at the crossing core", fontsize=9)
plt.savefig(a.out + ".png", dpi=85); print("saved", a.out + ".png")
