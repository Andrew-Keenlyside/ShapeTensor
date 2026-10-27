"""Predominant orientation against the TRUTH on hard synthetic phantoms (the final set's crossings, fine crossings and
mixes; curves and fans for reference), the same measures as `real_vs_st.py` on real HiP-CT -- here the models and the
structure tensor are both scored against the generator's true populations.

At a 3-D grid of points (every 3 voxels, 6 from the faces), window sigma ``--sigma``, the truth's line populations
(target mass >= 0.03, peaked: kappa2 >= 2) give the classes:

* **single**: one line population (any other below a fifth of it) -- angle of each estimate to its axis;
* **core**: two line populations, the lighter at least half the heavier -- angle of each estimate's predominant
  direction to the NEARER true axis, and for the models whether their two heaviest line populations match both true
  axes (each within 20 deg, one-to-one); the structure tensor gives one direction only.

Estimates: the structure tensor's line direction (gradient sigma 1, integration sigma = the window), each model's
density peak (argmax over 200 directions) and heaviest line population (the default readout). Chance: median 60 deg.

Usage: python scripts/learn/synth_vs_truth.py OUT_STEM --models NAME=CKPT[:um] ... [--families cross fine mix curve fan]
"""
import argparse, json, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.filters import gaussian
from shapetensor.learn import entity_infer as EI, entity_train as ET, phantom2 as P2

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--families", nargs="+", default=["cross", "fine", "mix", "curve", "fan"]); ap.add_argument("--n", type=int, default=4)
ap.add_argument("--sigma", type=float, default=2.5); ap.add_argument("--step", type=int, default=3); ap.add_argument("--count-min", type=float, default=0.03)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
from shapetensor.learn.model import PopulationNet
models = []
for spec in a.models:
    name, path = spec.split("=", 1); um = path.endswith(":um"); path = path[:-3] if um else path
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = PopulationNet.from_args(ck["args"]).to(dev).eval(); m.load_state_dict(ck["model"]); models.append((name, m))
sph = EI.sphere_dirs(200, dev)
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine")


def structure_tensor(x, pts, rho):
    xs = gaussian(x.float()[None, None], 1.0)[0, 0]; gz, gy, gx = torch.gradient(xs)
    comps = [gaussian((u * v)[None, None], rho)[0, 0] for u, v in ((gz, gz), (gy, gy), (gx, gx), (gz, gy), (gz, gx), (gy, gx))]
    idx = pts.round().long(); G = torch.stack([c[idx[:, 0], idx[:, 1], idx[:, 2]] for c in comps], -1)
    M = torch.stack([torch.stack([G[:, 0], G[:, 3], G[:, 4]], -1), torch.stack([G[:, 3], G[:, 1], G[:, 5]], -1), torch.stack([G[:, 4], G[:, 5], G[:, 2]], -1)], -2)
    ev, V = torch.linalg.eigh(M)
    return (ev[:, 2] - ev[:, 0]) / ev.sum(-1).clamp_min(1e-9), V[:, :, 0]


def ang(u, v):
    u = torch.nn.functional.normalize(u, dim=-1); v = torch.nn.functional.normalize(v, dim=-1)
    return torch.rad2deg(torch.acos((u * v).sum(-1).abs().clamp(max=1)))


def stats(x):
    x = x[torch.isfinite(x)]
    if x.numel() < 5:
        return {"n": int(x.numel())}
    return {"n": int(x.numel()), "median": float(x.median()), "within10": float((x <= 10).float().mean()), "within20": float((x <= 20).float().mean())}


acc = {}                                                                                       # (family, class, estimator) -> angles
both = {}                                                                                      # (family, model) -> core points with both axes found
for fam in a.families:
    fi = FAMS.index(fam)
    for i in range(a.n):
        spec, look = P2.random_spec(fam, torch.Generator().manual_seed(1_900_000 + 100 * fi + i))
        ph = P2.make(spec, seed=1_950_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1); n = ph["n"]
        g1 = torch.arange(6, n - 6, a.step, device=dev, dtype=torch.float32)
        pts = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
        sig = torch.full((pts.shape[0],), a.sigma, device=dev)
        t = P2.targets(ph, pts, sig); K = ph["occ"].shape[0]
        mT = t["e_mass"][:, :K]; kT = t["e_kappa"][:, :K]; RT = t["e_R"][:, :K]
        line = (mT >= a.count_min) & (kT[..., 1] >= 2.0)
        mL = torch.where(line, mT, torch.zeros_like(mT)); order = mL.argsort(1, descending=True)
        m1 = mL.gather(1, order[:, :1])[:, 0]; m2 = mL.gather(1, order[:, 1:2])[:, 0] if K > 1 else torch.zeros_like(m1)
        ax1 = RT[torch.arange(len(pts)), order[:, 0], :, 2]; ax2 = RT[torch.arange(len(pts)), order[:, 1], :, 2] if K > 1 else ax1
        single = (m1 > 0) & (m2 < 0.2 * m1); core = (m2 > 0) & (m2 >= 0.5 * m1)
        _, st = structure_tensor(ph["image"], pts, a.sigma)
        ests = {"structure tensor": (st, None)}
        for name, m in models:
            with torch.no_grad():
                feats = EI.features(m, ph["image"])
                f = torch.cat([EI.query(m, feats, pts[j:j + 2048], sig[j:j + 2048], sph, vol_shape=ph["image"].shape)["f"].float() for j in range(0, pts.shape[0], 2048)])
                ro = EI.readout(m, ph["image"], pts, sig, sph, feats=feats, iso_k=ET.VOTE_KAPPA_MIN)
            mass = ro["mass"].float(); kk = ro["k"].float(); R = ro["R"].float()
            isl = (kk[..., 1] >= 2.0) & (mass >= a.count_min); mm = torch.where(isl, mass, torch.zeros_like(mass)); o2 = mm.argsort(1, descending=True)
            ar = torch.arange(len(pts), device=dev)
            p1 = torch.where((mm.gather(1, o2[:, :1]) > 0), R[ar, o2[:, 0], :, 2], torch.full_like(R[:, 0, :, 2], float("nan")))
            p2 = torch.where((mm.gather(1, o2[:, 1:2]) > 0), R[ar, o2[:, 1], :, 2], torch.full_like(R[:, 0, :, 2], float("nan")))
            ests[f"{name} density peak"] = (sph[f.argmax(1)], None)
            ests[f"{name} heaviest line pop."] = (p1, p2)
        for en, (d1, d2) in ests.items():
            a1 = ang(d1, ax1); a_near = torch.minimum(ang(d1, ax1), ang(d1, ax2))
            acc.setdefault((fam, "single", en), []).append(a1[single].cpu())
            acc.setdefault((fam, "core", en), []).append(a_near[core].cpu())
            if d2 is not None and bool(core.any()):                                              # both true axes found, one-to-one within 20 deg
                ok = ((ang(d1, ax1) <= 20) & (ang(d2, ax2) <= 20)) | ((ang(d1, ax2) <= 20) & (ang(d2, ax1) <= 20))
                both.setdefault((fam, en), []).append(ok[core].float().cpu())
        print(fam, i, f"single {int(single.sum())} core {int(core.sum())} points;", " | ".join(
            f"{en}: single {stats(acc[(fam, 'single', en)][-1]).get('median', float('nan')):.0f}, core {stats(acc[(fam, 'core', en)][-1]).get('median', float('nan')):.0f}"
            for en in ests), flush=True)

summary = {}
ens = list(dict.fromkeys(k[2] for k in acc))
for cls in ("single", "core"):
    for fam in a.families + ["hard (cross, fine, mix)"]:
        fams = ["cross", "fine", "mix"] if fam.startswith("hard") else [fam]
        for en in ens:
            parts = [x for f_ in fams for x in acc.get((f_, cls, en), [])]
            if parts:
                summary[f"{cls}|{fam}|{en}"] = stats(torch.cat(parts))
for fam in a.families + ["hard (cross, fine, mix)"]:
    fams = ["cross", "fine", "mix"] if fam.startswith("hard") else [fam]
    for en in ens:
        parts = [x for f_ in fams for x in both.get((f_, en), [])]
        if parts:
            summary[f"core both found|{fam}|{en}"] = float(torch.cat(parts).mean())
json.dump(summary, open(a.out + ".json", "w"), indent=1)
for cls in ("single", "core"):
    print(f"== {cls}: median angle to the truth ({'its axis' if cls == 'single' else 'the nearer of the two axes'}), share within 20 deg")
    for fam in a.families + ["hard (cross, fine, mix)"]:
        print(f"  {fam}: " + " | ".join(f"{en} {summary[k]['median']:.0f} ({summary[k]['within20'] * 100:.0f} %)" for en in ens
                                         if (k := f"{cls}|{fam}|{en}") in summary and "median" in summary[k]))
print("== cores: both true axes found by the two heaviest line populations (each within 20 deg)")
for fam in a.families + ["hard (cross, fine, mix)"]:
    vals = {en: summary[f"core both found|{fam}|{en}"] for en in ens if f"core both found|{fam}|{en}" in summary}
    if vals:
        print(f"  {fam}: " + " | ".join(f"{en.replace(' heaviest line pop.', '')} {v * 100:.0f} %" for en, v in vals.items()))

# figure: median angle per family and estimator, single points and cores; and the both-found share at cores
cols = {"structure tensor": "#555555"}; pal = ["#2a78d6", "#eb6834", "#1baf7a"]
for j, (name, _) in enumerate(models):
    cols[f"{name} density peak"] = pal[j % 3]; cols[f"{name} heaviest line pop."] = pal[j % 3]
fams_p = a.families + ["hard (cross, fine, mix)"]
fig, axs = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
for ax, cls in zip(axs[:2], ("single", "core")):
    w = 0.8 / len(ens)
    for e_i, en in enumerate(ens):
        ys = [summary.get(f"{cls}|{f}|{en}", {}).get("median", np.nan) for f in fams_p]
        ax.bar(np.arange(len(fams_p)) + e_i * w, ys, w, color=cols[en], hatch="//" if "density peak" in en else None, edgecolor="white", label=en)
    ax.axhline(60, color="#999", lw=0.8, ls=":"); ax.set_xticks(np.arange(len(fams_p)) + 0.4 - w / 2); ax.set_xticklabels([f.split(" (")[0] for f in fams_p], fontsize=8)
    ax.set_ylabel("median angle to the truth (deg)"); ax.set_title("single-population points: angle to the true axis" if cls == "single" else "crossing cores: angle to the nearer true axis", fontsize=10)
axs[0].legend(fontsize=7)
ax = axs[2]; mods = [en for en in ens if "line pop." in en]; w = 0.8 / max(len(mods), 1)
for e_i, en in enumerate(mods):
    ys = [summary.get(f"core both found|{f}|{en}", np.nan) * 100 for f in fams_p]
    ax.bar(np.arange(len(fams_p)) + e_i * w, ys, w, color=cols[en], label=en.replace(" heaviest line pop.", ""))
ax.set_xticks(np.arange(len(fams_p)) + 0.4 - w / 2); ax.set_xticklabels([f.split(" (")[0] for f in fams_p], fontsize=8); ax.set_ylim(0, 100)
ax.set_ylabel("% of core points"); ax.set_title("crossing cores: BOTH bundles found (two line populations, each within 20 deg)\n(a structure tensor gives one direction: 0 %)", fontsize=10); ax.legend(fontsize=7)
fig.suptitle(f"Predominant orientation against the truth on the final set's synthetic phantoms, window sigma {a.sigma} voxels", fontsize=11)
plt.savefig(a.out + ".png", dpi=75); print("saved", a.out + ".png")
