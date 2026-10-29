"""How inference depends on the output resolution, and whether the model gives fibre super-resolution estimates, on the
composable phantoms (:mod:`~shapetensor.learn.phantom2`):

1. Output scale: the held-out phantoms scored with the acceptance criteria (crossing_acceptance.evaluate_p2) at query
   windows sigma = 1, 1.5, 2.5, 4, 6 voxels, each against the targets at that sigma.
2. Sub-voxel output: a crossing queried on a half-voxel grid at sigma 1 against the fibre-scale truth: the correlation of
   each population's model and target mass maps (does the model place individual fibres finer than the input grid?).
3. Unresolved fibres: crossings whose fibres are finer and closer than the grid resolves (spacing 1.6-2.4, radius
   0.4-0.7 voxels): the criteria at sigma 2.5 (orientation and fraction of sub-voxel fibres from partial volume).

Usage: python scripts/learn/resolution_study.py OUT_STEM CKPT
"""
import argparse, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, losses as L, entity_infer as EI, phantom2 as P2, crossing_acceptance as CA
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--floor", type=float, default=0.05)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.REGION_DENSITY = True; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
sph = EI.sphere_dirs(200, dev)
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty")


def frame(ph):
    spec = ph["spec"]
    if spec and "normal" in spec[0]:
        nrm = torch.nn.functional.normalize(torch.tensor(spec[0]["normal"], dtype=torch.float32), dim=0); e1, e2 = P2._frame(nrm)
        return torch.stack([nrm, e1, e2], 1).to(dev), (spec[0].get("spacing", 0.0) / 4 if spec[0]["kind"] != "sheet" else 0.0)
    return torch.eye(3, device=dev), 0.0


def maps(ph, sigma, step=1.0, half=28.0):
    """Targets and model on the slice: a grid of ``step`` voxels within a disc of radius ``half`` (any orientation)."""
    n = ph["n"]; c = n / 2.0; R, off = frame(ph)
    ax_ = torch.arange(-half, half + 1e-6, step, device=dev); m = ax_.numel()
    gu, gv = torch.meshgrid(ax_, ax_, indexing="ij")
    loc = torch.stack([torch.full_like(gu, off), gu, gv], -1).view(-1, 3); pts = c + loc @ R.T
    disc = ((gu ** 2 + gv ** 2) <= half ** 2).cpu().numpy()
    sig = torch.full((pts.shape[0],), float(sigma), device=dev)
    t = P2.targets(ph, pts, sig); K = ph["occ"].shape[0]
    tgt = {"mass": t["e_mass"][:, :K].cpu().numpy(), "k": t["e_kappa"][:, :K].cpu().numpy(), "R": t["e_R"][:, :K].cpu().numpy(),
           "bg": t["e_mass"][:, K].cpu().numpy(), "kinds": ph["kinds"]}
    parts = []
    with torch.no_grad():
        f = EI.features(net, ph["image"], voxel_um=None)
        for a0 in range(0, pts.shape[0], 2048):                                                      # in chunks: clustering is memory-hungry
            sl = slice(a0, a0 + 2048)
            r = EI.query(net, f, pts[sl], sig[sl], sph, vol_shape=ph["image"].shape)
            e = EI.cluster(r, sph, floor=a.floor); sp = EI.split_mass(r, sph, e)
            ms = (sp["mass"] * e["live"]).float(); kv = e["kappa"].float(); isoe = kv[..., 0] < ET.VOTE_KAPPA_MIN
            iso_tot = sp["floor"].float() + (ms * isoe).sum(1)
            bgm = r["bgm"].float().clamp(0, 1) if "bgm" in r else iso_tot
            bg = torch.minimum(iso_tot, bgm)
            parts.append({"mass": (ms * ~isoe).cpu().numpy(), "k": kv.cpu().numpy(), "R": e["R"].float().cpu().numpy(),
                          "iso": (iso_tot - bg).clamp_min(0).cpu().numpy(), "bg": bg.cpu().numpy(), "bgm": bgm.cpu().numpy()})
    C = max(p_["mass"].shape[1] for p_ in parts)
    pad = lambda x, k: np.concatenate([x, np.zeros((x.shape[0], C - x.shape[1]) + x.shape[2:])], 1) if x.shape[1] < C else x
    mdl = {k: np.concatenate([pad(p_[k], k) if k in ("mass", "k", "R") else p_[k] for p_ in parts]) for k in parts[0]}
    face = torch.minimum(pts, (n - 1) - pts).amin(1).view(m, m).cpu().numpy()
    return m, tgt, mdl, face, disc


def oracle(tgt):
    iso = CA._types(tgt["k"][..., 0], tgt["k"][..., 1]) == 2
    return {"mass": np.where(iso, 0.0, tgt["mass"]), "k": tgt["k"], "R": tgt["R"], "iso": (tgt["mass"] * iso).sum(1), "bg": tgt["bg"], "bgm": tgt["bg"]}


def score(m, tgt, mdl, face, disc):
    """The criteria, or None where the ground truth itself fails the measurement (unscoreable)."""
    return CA.evaluate_p2(m, tgt, mdl, face, disc) if CA.evaluate_p2(m, tgt, oracle(tgt), face, disc)["PASS"] else None


tests = []
for fi, fam in enumerate(FAMS):
    for i in range(2):
        spec, look = P2.random_spec(fam, torch.Generator().manual_seed(900_000 + 100 * fi + i))
        tests.append((fam, P2.make(spec, seed=950_000 + 100 * fi + i, look=look, device=dev)))

# 1. output scale
sigmas = (1.0, 1.5, 2.5, 4.0, 6.0); tab = {}
for s_ in sigmas:
    rows = []
    for fam, ph in tests:
        m, tgt, mdl, face, disc = maps(ph, s_)
        sc = score(m, tgt, mdl, face, disc)
        if sc is not None:
            rows.append((fam, sc))
    tab[s_] = rows
    print(f"sigma {s_}: {sum(r['PASS'] for _, r in rows)}/{len(rows)} pass (of {len(tests)} phantoms, {len(tests) - len(rows)} unscoreable) | per criterion: " +
          " ".join(f"{k.split()[0]} {sum(r[k][0] for _, r in rows)}" for k in CA.NAMES8), flush=True)

# 2. sub-voxel output on a crossing: half-voxel grid at sigma 1
fam, ph = tests[0]
m, tgt, mdl, face, disc = maps(ph, 1.0, step=0.5)
mm = CA.match_p2(tgt, mdl); ok = disc.reshape(-1) & (face.reshape(-1) >= 2)
cor = [float(np.corrcoef(tgt["mass"][ok, k], mm["m"][ok, k])[0, 1]) for k in range(tgt["mass"].shape[1])]
print(f"sub-voxel (half-voxel grid, sigma 1) on {fam}: model/target mass-map correlation per population {[round(c_, 2) for c_ in cor]}", flush=True)

# 3. unresolved fibres
unres = []
for i in range(6):
    g = torch.Generator().manual_seed(990_000 + i)
    spec, look = P2.random_spec("cross", g)
    for p in spec:
        p["spacing"] = 1.6 + 0.8 * float(torch.rand(1, generator=g)); p["radius"] = 0.4 + 0.3 * float(torch.rand(1, generator=g))
    spec[1]["z0"] = spec[0]["spacing"] / 2
    ph_u = P2.make(spec, seed=991_000 + i, look=look, device=dev)
    mu, tu, md, fu, du = maps(ph_u, 2.5)
    sc = score(mu, tu, md, fu, du)
    if sc is not None:
        unres.append((f"unresolved {i}", sc))
print(f"unresolved fibres (spacing 1.6-2.4, radius 0.4-0.7): {sum(r['PASS'] for _, r in unres)}/{len(unres)} pass | " +
      " ".join(f"{k.split()[0]} {sum(r[k][0] for _, r in unres)}" for k in CA.NAMES8), flush=True)
CA.table_figure(unres, a.out + "_unresolved_acceptance.png", "unresolved fibres (spacing 1.6-2.4, radius 0.4-0.7 voxels), sigma 2.5")

# figure: criteria passed vs sigma, and the sub-voxel maps
fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), layout="constrained")
for k in CA.NAMES8:
    axs[0].plot(sigmas, [sum(r[k][0] for _, r in tab[s_]) for s_ in sigmas], "-o", ms=3, label=k)
axs[0].plot(sigmas, [sum(r["PASS"] for _, r in tab[s_]) for s_ in sigmas], "k-o", lw=2.5, label="all criteria")
axs[0].set_xscale("log"); axs[0].set_xticks(sigmas); axs[0].set_xticklabels([str(s_) for s_ in sigmas]); axs[0].set_xlabel("query window sigma (voxels)")
axs[0].set_ylabel("held-out phantoms passing"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3); axs[0].set_title("output scale", fontsize=10)
for j in range(min(2, tgt["mass"].shape[1])):
    axs[1 + j].imshow(np.where(disc, mm["m"][:, j].reshape(m, m), np.nan), vmin=0, vmax=1); axs[1 + j].set_title(f"model pop {j + 1}, half-voxel grid, sigma 1 (r={cor[j]:.2f})", fontsize=9)
axs[3].imshow(np.where(disc, tgt["mass"][:, 0].reshape(m, m), np.nan), vmin=0, vmax=1); axs[3].set_title("target pop 1 at sigma 1 (fibre scale)", fontsize=9)
for ax_ in axs[1:]:
    ax_.set_xticks([]); ax_.set_yticks([])
plt.savefig(a.out + ".png", dpi=75); print("saved", a.out + ".png")
