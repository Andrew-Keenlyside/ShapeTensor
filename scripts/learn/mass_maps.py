"""Separated mass images of one model: each population's mass in the window (sigma ``--sigma``) at every point of a slice.

* Synthetic (the final set's phantoms, the slice through the phantom's structure as phantom_ladder.py cuts it): per
  true population its target mass beside the model's mass matched to it (crossing_acceptance.match_p2: the same
  matching the criteria use); then the model's mass in other directions, its isotropic objects and its background.
* Real HiP-CT (the 20 um deep-brain crops, as real_populations.py; no truth): the model's heaviest and second oriented
  populations at each point -- colour = the population's 3-D direction (red x, green y, blue z), brightness = its mass --
  and its isotropic objects, background and total oriented mass; SynthSeg's white matter outlined on the image.

Usage: python scripts/learn/mass_maps.py OUT_STEM CKPT [--rows cross:0 cross:1 fine:0 fine:2 mix:1 curve:0 sheet:0 blob:0] [--pool 4]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_infer as EI, entity_train as ET, phantom2 as P2, crossing_acceptance as CA
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.realdata import normalise

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--name", default="overnight pilot")
ap.add_argument("--rows", nargs="+", default=["cross:0", "cross:1", "cross:2", "fine:0", "fine:2", "mix:1", "fan:0", "curve:0", "sheet:0", "blob:0"])
ap.add_argument("--pool", type=int, nargs="+", default=[4, 1]); ap.add_argument("--size", type=int, default=96); ap.add_argument("--sigma", type=float, default=2.5)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); sph = EI.sphere_dirs(200, dev)
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine")
NORM = PowerNorm(0.5, vmin=0, vmax=1); CMAP = matplotlib.colormaps["viridis"].copy(); CMAP.set_over("red")


def readout(x, pts):
    sig = torch.full((pts.shape[0],), a.sigma, device=dev)
    with torch.no_grad():
        return {k: v.float().cpu().numpy() for k, v in EI.readout(net, x, pts, sig, sph, iso_k=ET.VOTE_KAPPA_MIN).items()}


def frame(ph):                                                                                  # phantom_ladder.py's slice
    n = ph["n"]; spec = ph["spec"]
    if spec and "normal" in spec[0]:
        nrm = torch.nn.functional.normalize(torch.tensor(spec[0]["normal"], dtype=torch.float32), dim=0); e1, e2 = P2._frame(nrm)
        if spec[0]["kind"] == "sheet":
            R = torch.stack([e1, nrm, e2], 1); off = 0.0
        else:
            R = torch.stack([nrm, e1, e2], 1); off = spec[0].get("spacing", 0.0) / 4
    else:
        R = torch.eye(3); off = 0.0
    return R.to(dev), off


# ---- synthetic --------------------------------------------------------------------------------------------------
cols = ["image", "truth: population 1", "model: population 1", "truth: population 2", "model: population 2",
        "model: other directions", "model: isotropic objects", "truth: background", "model: background"]
fig, axs = plt.subplots(len(a.rows), len(cols), figsize=(2.2 * len(cols) + 0.8, 2.3 * len(a.rows)), squeeze=False, layout="constrained")
for r_, rs in enumerate(a.rows):
    fam, i = rs.split(":"); i = int(i); fi = FAMS.index(fam)
    spec, look = P2.random_spec(fam, torch.Generator().manual_seed(1_900_000 + 100 * fi + i))
    ph = P2.make(spec, seed=1_950_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1); n = ph["n"]; c = n / 2.0
    R, off = frame(ph)
    gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
    pts = c + torch.stack([torch.full_like(gu, off), gu - c, gv - c], -1).view(-1, 3) @ R.T
    inside = ((pts >= 2) & (pts <= n - 3)).all(1).cpu().numpy()
    t = P2.targets(ph, pts, torch.full((n * n,), a.sigma, device=dev)); K = ph["occ"].shape[0]
    tgt = {"mass": t["e_mass"][:, :K].cpu().numpy(), "k": t["e_kappa"][:, :K].cpu().numpy(), "R": t["e_R"][:, :K].cpu().numpy(), "bg": t["e_mass"][:, K].cpu().numpy(), "kinds": ph["kinds"]}
    mdl = readout(ph["image"], pts); m = CA.match_p2(tgt, mdl)
    img = torch.nn.functional.grid_sample(ph["image"][None, None], (pts / (n - 1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3), align_corners=True).view(n, n).cpu().numpy()
    sh = lambda v: np.where(inside.reshape(n, n), v.reshape(n, n), np.nan)
    iso_t = np.where(CA._types(tgt["k"][..., 0], tgt["k"][..., 1]) == 2, tgt["mass"], 0).sum(1) if K else np.zeros(n * n)
    panels = [(img, "gray", None)]
    for k in range(2):
        if k < K and fam != "blob":
            panels += [(sh(tgt["mass"][:, k]), CMAP, NORM), (sh(m["m"][:, k]), CMAP, NORM)]
        else:
            panels += [(None, None, None), (None, None, None)]
    panels += [(sh(m["other"]), CMAP, NORM), (sh(mdl["iso"]), CMAP, NORM), (sh(tgt["bg"]), CMAP, NORM), (sh(mdl["bg"]), CMAP, NORM)]
    for j, (v, cm, nm) in enumerate(panels):
        ax = axs[r_, j]
        if v is not None:
            im = ax.imshow(v, cmap=cm, norm=nm) if nm is not None else ax.imshow(v, cmap=cm)
            if j in (1, 2, 3, 4) and nm is not None:
                ax.text(2, n - 4, f"sum {np.nansum(v):.0f}", color="w", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    if fam == "blob":                                                                            # blobs: the isotropic truth in the population column
        axs[r_, 1].imshow(sh(iso_t), cmap=CMAP, norm=NORM); axs[r_, 1].text(2, n - 4, "isotropic objects (truth)", color="w", fontsize=6)
    axs[r_, 0].set_ylabel(f"{fam} {i}\n{', '.join(ph['kinds'])}\ncontrast {', '.join(f'{abs(x):.2f}' for x in look['contrast'])}", fontsize=8)
for j, t_ in enumerate(cols):
    axs[0, j].set_title(t_, fontsize=9)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=axs[:, 1:].ravel().tolist(), shrink=0.4, label="mass in the window (square-root scale)")
fig.suptitle(f"{a.name}: separated mass images on held-out synthetic phantoms (final set), window sigma {a.sigma} voxels -- truth beside model, "
             f"population by population (model mass matched to each true population as the criteria match it)", fontsize=10)
plt.savefig(a.out + "_synthetic.png", dpi=70); plt.close(fig); print("saved", a.out + "_synthetic.png")

# ---- real -------------------------------------------------------------------------------------------------------
CROPS = Path("/hdd/andrew/brain_atlas_20um_test_crops"); WM = (2, 41); GM = (3, 42, 10, 49, 11, 50, 12, 51)


def read_stack(folder, sl):
    import tifffile
    files = sorted(folder.glob("*.tif*"))
    return np.stack([tifffile.imread(files[z])[sl[1], sl[2]] for z in range(sl[0].start, sl[0].stop)])


def dec(mass, axis, scale=0.5):
    rgb = np.abs(axis[..., [2, 1, 0]]) * np.clip(mass / scale, 0, 1)[..., None]
    return np.clip(rgb, 0, 1)


for pool in a.pool:
    specs = [s for s in sorted(CROPS.iterdir()) if next(s.glob("*_deepbrain_L_20um_768"), None) is not None]
    rcols = ["image (yellow: SynthSeg white matter)", "heaviest population\n(colour = direction, brightness = mass)", "second population",
             "heaviest population: mass", "second population: mass", "isotropic objects", "background", "total oriented mass"]
    fig, axs = plt.subplots(len(specs), len(rcols), figsize=(2.3 * len(rcols) + 0.8, 2.4 * len(specs)), squeeze=False, layout="constrained")
    for r_, spec in enumerate(specs):
        img_dir = next(spec.glob("*_deepbrain_L_20um_768")); lab_dir = next(spec.glob("*_deepbrain_L_synthseg_768"))
        rng = np.random.default_rng(0); ext = a.size * pool; best = None
        for _ in range(60):                                                                       # the crops of real_populations.py
            o = [int(rng.integers(0, 768 - ext + 1)) for _ in range(3)]; sl = tuple(slice(v, v + ext) for v in o)
            lab = read_stack(lab_dir, sl)[pool // 2::pool, pool // 2::pool, pool // 2::pool]
            shr = min(np.isin(lab, WM).mean(), np.isin(lab, GM).mean())
            if best is None or shr > best[0]:
                best = (shr, o, sl, lab)
            if shr >= (0.10 if pool == 1 else 0.2):
                break
        _, o, sl, lab = best
        img = read_stack(img_dir, sl).astype(np.float32)
        if pool > 1:
            img = img.reshape(a.size, pool, a.size, pool, a.size, pool).mean((1, 3, 5))
        x, _ = normalise(torch.tensor(img, device=dev)); n = a.size; zc = n // 2
        gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
        pts = torch.stack([torch.full_like(gu, float(zc)), gu, gv], -1).view(-1, 3)
        mdl = readout(x, pts)
        mass = mdl["mass"]; kk = mdl["k"]; Rm = mdl["R"]
        typ = np.where(kk[..., 1] < 2.0, 1, 0)                                                   # girdle: its normal; line: its axis
        ax_ = np.where((typ == 1)[..., None], Rm[..., :, 0], Rm[..., :, 2])
        order = np.argsort(-mass, 1); ar = np.arange(mass.shape[0])
        m1, m2 = mass[ar, order[:, 0]], mass[ar, order[:, 1]] if mass.shape[1] > 1 else np.zeros(len(ar))
        a1, a2 = ax_[ar, order[:, 0]], ax_[ar, order[:, 1]] if mass.shape[1] > 1 else ax_[ar, order[:, 0]]
        rsh = lambda v: v.reshape(n, n, *v.shape[1:])
        panels = [(x[zc].cpu().numpy(), "gray"), (rsh(dec(m1, a1)), None), (rsh(dec(m2, a2)), None), (rsh(m1), CMAP), (rsh(m2), CMAP),
                  (rsh(mdl["iso"]), CMAP), (rsh(mdl["bg"]), CMAP), (rsh(mass.sum(1)), CMAP)]
        for j, (v, cm) in enumerate(panels):
            ax = axs[r_, j]
            if cm is None:
                ax.imshow(v)
            elif cm == "gray":
                ax.imshow(v, cmap="gray"); ax.contour(np.isin(lab[zc], WM).astype(float), levels=[0.5], colors="#eda100", linewidths=0.8)
            else:
                ax.imshow(v, cmap=cm, norm=NORM)
            ax.set_xticks([]); ax.set_yticks([])
        axs[r_, 0].set_ylabel(f"{spec.name}\n{20 * pool} um voxels", fontsize=8)
    for j, t_ in enumerate(rcols):
        axs[0, j].set_title(t_, fontsize=8)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=axs[:, 3:].ravel().tolist(), shrink=0.4, label="mass in the window (square-root scale)")
    fig.suptitle(f"{a.name}: separated mass images on real HiP-CT ({20 * pool} um voxels: 20 um deep brain{' block-averaged x' + str(pool) if pool > 1 else ''}), "
                 f"window sigma {a.sigma} voxels ({a.sigma * 20 * pool:.0f} um); no truth -- populations ranked by mass at each point", fontsize=10)
    plt.savefig(f"{a.out}_real_x{pool}.png", dpi=70); plt.close(fig); print("saved", f"{a.out}_real_x{pool}.png")
