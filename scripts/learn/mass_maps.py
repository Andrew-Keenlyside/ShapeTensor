"""Separated mass images of one model: each population's mass in the window (sigma ``--sigma``) at every point of a slice.

* Synthetic (the final set's phantoms, the slice through the phantom's structure as phantom_ladder.py cuts it): per
  true population its target mass beside the model's mass assigned to it (:func:`soft_assign`: a smooth angular weight,
  lines to lines and sheets to sheets); then the model's mass in other directions, its isotropic objects and its
  background.
* Real HiP-CT (the 20 um deep-brain crops, as real_populations.py; no truth): the oriented mass -- lines coloured by
  their mean 3-D direction (red x, green y, blue z), sheets grey, brightness = mass -- the mixing, three soft line
  tracks (:func:`direction_tracks`), the sheet mass, isotropic objects and background; SynthSeg's white matter outlined.
  Types enter as the readout's sheet weight (:func:`parts_of`), so no map jumps where a type changes.

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


def axes_of(k, R):
    """Each entity's representative axis: a line's axis, a sheet's normal (``k (..., 2)``, ``R (..., 3, 3)``)."""
    return np.where((k[..., 1] < 2.0)[..., None], R[..., :, 0], R[..., :, 2])


def parts_of(m):
    """The model's entities split by their sheet weight (the readout's ``sheet``: smooth in the ring cover) into a line
    part (mass x (1 - sheet), the axis) and a sheet part (mass x sheet, the normal): ``(mass (P, 2C), axis (P, 2C, 3),
    is_sheet (P, 2C))`` -- the type as a weight, so a population turning into a sheet fades rather than jumps 90 deg."""
    s = m["sheet"]; R = m["R"]
    return (np.concatenate([m["mass"] * (1 - s), m["mass"] * s], 1), np.concatenate([R[..., :, 2], R[..., :, 0]], 1),
            np.concatenate([np.zeros_like(s, dtype=bool), np.ones_like(s, dtype=bool)], 1))


def soft_assign(tgt, mdl, tau=15.0, other=30.0):
    """The model's mass per true population for display: each entity's mass split among the true populations present
    (and an "other directions" option at ``other`` deg) by a smooth angular weight exp(-angle^2 / 2 tau^2) -- no hard
    matching cut, so the images move continuously (the criteria keep their own one-to-one 30-deg matching); its line
    and sheet parts (:func:`parts_of`) go to the true lines and sheets respectively."""
    tm = tgt["mass"]; K = tm.shape[1]; mm, ma, msh = parts_of(mdl)
    if K == 0:
        return {"m": np.zeros((mm.shape[0], 0)), "other": mm.sum(1)}
    ta = axes_of(tgt["k"], tgt["R"])                                                              # (P, K, 3); the model's (P, 2C, 3)
    ttype = CA._types(tgt["k"][..., 0], tgt["k"][..., 1])
    ang = np.degrees(np.arccos(np.clip(np.abs(np.einsum("pkd,pcd->pkc", ta, ma)), 0, 1)))
    w = np.exp(-0.5 * (ang / tau) ** 2) * (tm[:, :, None] > 0.02) * (ttype[:, :, None] < 2)      # oriented true populations present
    w = w * ((ttype[:, :, None] == 1) == msh[:, None, :])                                         # of the part's type
    w0 = np.full((mm.shape[0], 1, mm.shape[1]), np.exp(-0.5 * (other / tau) ** 2))
    W = np.concatenate([w, w0], 1); W = W / W.sum(1, keepdims=True)
    share = W * mm[:, None, :]
    return {"m": share[:, :K].sum(2), "other": share[:, K].sum(1)}


def direction_tracks(mass, axes, K=3, iters=20):
    """Soft direction tracks over a slice: the entities' ``axes (P, C, 3)`` clustered (weighted by ``mass (P, C)``) into
    ``K`` axial directions (spherical k-means on the dyads), each entity's mass split among the tracks by exp(8 cos^2);
    returns the tracks' mass maps ``(P, K)`` and their directions ``(K, 3)``."""
    ax = axes.reshape(-1, 3); w = mass.reshape(-1); keep = w > 1e-3
    A = ax[keep]; W = w[keep]
    if len(A) < K:
        return np.zeros((mass.shape[0], K)), np.eye(3)[:K]
    rng = np.random.default_rng(0); C = A[rng.choice(len(A), K, replace=False, p=W / W.sum())]
    for _ in range(iters):
        lab = np.argmax(np.abs(A @ C.T), 1)
        for j in range(K):
            sel = lab == j
            if sel.any():
                M = (W[sel, None, None] * A[sel, :, None] * A[sel, None, :]).sum(0); C[j] = np.linalg.eigh(M)[1][:, -1]
    cs = np.abs(np.einsum("pcd,kd->pck", axes, C))
    r_ = np.exp(8 * cs ** 2); r_ = r_ / r_.sum(-1, keepdims=True)
    return (mass[..., None] * r_).sum(1), C


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
    mdl = readout(ph["image"], pts); m = soft_assign(tgt, mdl)
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
             f"population by population (each model population's mass shared among the true ones by a smooth angular weight)", fontsize=10)
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
    rcols = ["image (yellow: SynthSeg white matter)", "oriented mass: lines coloured by\ntheir mean direction, sheets grey", "mixing: 1 - sum share^2\n(0 one population, 0.5 two)",
             "line track 1: mass", "line track 2: mass", "line track 3: mass", "sheet mass\n(soft: weight of the ring cover)", "isotropic objects", "background"]
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
        mass = mdl["mass"]; C_ = mass.shape[1]; pm, pax, _ = parts_of(mdl); lm, lax = pm[:, :C_], pax[:, :C_]
        tr, Cd = direction_tracks(lm, lax)                                                        # soft direction tracks of the line parts
        order = np.argsort(-tr.sum(0)); tr = tr[:, order]; Cd = Cd[order]
        rsh = lambda v: v.reshape(n, n, *v.shape[1:])
        dy = (lm[..., None, None] * lax[..., :, None] * lax[..., None, :]).sum(1)                  # the line parts' mass-weighted mean dyad
        dom = np.linalg.eigh(dy)[1][..., -1]; tot = mass.sum(1); smass = pm[:, C_:].sum(1)
        simpson = 1 - (mass ** 2).sum(1) / np.maximum(tot, 1e-9) ** 2                               # 0: one population, 0.5: two equal
        rgb = np.clip(dec(lm.sum(1), dom) + 0.6 * np.clip(smass / 0.5, 0, 1)[:, None], 0, 1)          # sheets: grey
        panels = [(x[zc].cpu().numpy(), "gray"), (rsh(rgb), None), (rsh(simpson), "magma")]
        panels += [(rsh(tr[:, j]), CMAP) for j in range(3)]
        panels += [(rsh(smass), CMAP), (rsh(mdl["iso"]), CMAP), (rsh(mdl["bg"]), CMAP)]
        track_titles = [f"track {j + 1}: ({', '.join(f'{v:+.2f}' for v in Cd[j][[2, 1, 0]])}) xyz" for j in range(3)]
        for j, (v, cm) in enumerate(panels):
            ax = axs[r_, j]
            if cm is None:
                ax.imshow(v)
            elif cm == "magma":
                ax.imshow(v, cmap="magma", vmin=0, vmax=0.6)
            elif cm == "gray":
                ax.imshow(v, cmap="gray"); ax.contour(np.isin(lab[zc], WM).astype(float), levels=[0.5], colors="#eda100", linewidths=0.8)
            else:
                ax.imshow(v, cmap=cm, norm=NORM)
            ax.set_xticks([]); ax.set_yticks([])
        axs[r_, 0].set_ylabel(f"{spec.name}\n{20 * pool} um voxels", fontsize=8)
        for j in range(3):
            axs[r_, 3 + j].text(2, n - 4, track_titles[j], color="w", fontsize=5)
    for j, t_ in enumerate(rcols):
        axs[0, j].set_title(t_, fontsize=8)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP), ax=axs[:, 3:].ravel().tolist(), shrink=0.4, label="mass in the window (square-root scale)")
    fig.suptitle(f"{a.name}: separated mass images (continuous readout; populations followed by direction) on real HiP-CT ({20 * pool} um voxels: 20 um deep brain{' block-averaged x' + str(pool) if pool > 1 else ''}), "
                 f"window sigma {a.sigma} voxels ({a.sigma * 20 * pool:.0f} um); no truth", fontsize=10)
    plt.savefig(f"{a.out}_real_x{pool}.png", dpi=70); plt.close(fig); print("saved", f"{a.out}_real_x{pool}.png")
