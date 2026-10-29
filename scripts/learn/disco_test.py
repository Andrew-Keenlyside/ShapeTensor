"""The readout on DiSCo rendered as a greyscale volume (test only: DiSCo never enters training, the generator or model
selection -- notes/plan_full_generator.md, D2).

DiSCo1 (Rafael-Patino et al. 2020; data doi:10.17632/fgf86jdfg6, CC BY 4.0): 12 196 strands (1.4-4.2 um diameter) in a
1 mm sphere, 25 connections between 16 ROIs. Its coordinates are scaled x40 (1 unit = 25 um, the 40^3 maps' voxel).

* **Render**: each strand a tube of its own diameter (a union of balls along its polyline) on a grid of 0.5-um cells
  than the voxel, averaged down (true partial volume), at ``--voxel`` um (1: the phantoms' fibre radius ~1 voxel; 2: the fine family's ~0.4; 4: unresolved, as white matter in 20-um HiP-CT); the image a phantom-like look: background +
  contrast x occupancy, blur, noise (``--look``: bright or dark fibres, inside the phantom families' ranges).
* **Crops** (96^3): centred where DiSCo's own 25-um bundle-count map holds 4+, 2-3 and 1 bundles (``--crops`` of each,
  seeded, inside the sphere).
* **Truth** at each point and window sigma: each connection's window occupancy and window-mean direction dyad (from the
  fine render, pooled); connections within ``--merge-deg`` of each other joined into one population (kissing and merging
  bundles are parallel: one population at that point).
* **Scores** (as ``synth_vs_truth.py``): single-population points -- angle to the true axis; crossing cores (two
  populations, the lighter >= half the heavier) -- angle to the nearer axis and both found (the model's two heaviest line
  populations each within 20 deg, one-to-one); the population count (mass >= ``--count-min``) against the truth's; the
  oriented mass against the true occupancy. Structure tensor: gradient sigma 1, integration sigma = the window.

Usage: python scripts/learn/disco_test.py OUT_STEM CKPT [--voxel 1 2] [--sigmas 2.5 4 6] [--crops 3 2 1]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.filters import gaussian
from shapetensor.learn import entity_infer as EI, entity_train as ET
from shapetensor.learn.model import PopulationNet, window_mean
from shapetensor.learn.axis_refine import refine_axes
sys.path.insert(0, str(Path(__file__).resolve().parent))
import disco_io as DIO

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--name", default="overnight pilot")
ap.add_argument("--data", default="/hdd/andrew/disco_data/DiSCo1"); ap.add_argument("--voxel", type=float, nargs="+", default=[1.0, 2.0, 4.0])
ap.add_argument("--sigmas", type=float, nargs="+", default=[2.5, 4.0, 6.0]); ap.add_argument("--show-sigma", type=float, default=4.0)
ap.add_argument("--crops", type=int, nargs=3, default=[3, 2, 1], help="crops at 4+, 2-3 and 1 bundles")
ap.add_argument("--n", type=int, default=96); ap.add_argument("--step", type=int, default=4); ap.add_argument("--look", default="bright", choices=["bright", "dark"])
ap.add_argument("--merge-deg", type=float, default=20.0); ap.add_argument("--count-min", type=float, default=0.03); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--spline", action="store_true", help="render along Catmull-Rom splines through the strands' points (disco_io)")
ap.add_argument("--readout", default="", help="readout options, e.g. 'saddle=0.9,lobe_merge_deg=15'"); ap.add_argument("--dirs", type=int, default=200)
a = ap.parse_args(); dev = "cuda"; UNIT = DIO.UNIT
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); sph = EI.sphere_dirs(a.dirs, dev)
LOOK = {"bright": {"bg": 0.15, "contrast": 0.6, "noise": 0.02, "blur": 0.5}, "dark": {"bg": 0.8, "contrast": -0.5, "noise": 0.02, "blur": 0.5}}[a.look]


DISCO = DIO.Disco(a.data, dev, spline=a.spline); NC = DISCO.NC


def sym(t6):
    T = torch.stack([torch.stack([t6[..., 0], t6[..., 3], t6[..., 4]], -1), torch.stack([t6[..., 3], t6[..., 1], t6[..., 5]], -1),
                     torch.stack([t6[..., 4], t6[..., 5], t6[..., 2]], -1)], -2)
    return T


def truth(occ, dyad, pts, sg):
    """Per point: the populations (connections within ``merge_deg`` joined), heaviest first -- ``mass (Q, P)``, ``axis
    (Q, P, 3)``; and the total occupancy ``(Q,)``."""
    live = occ.flatten(1).amax(1) > 0; oc, dd = occ[live], dyad[live]; K = oc.shape[0]; Q = pts.shape[0]
    m = window_mean(oc, pts[None].expand(K, -1, -1).contiguous(), torch.full((K, Q), sg, device=dev)).T          # (Q, K)
    w6 = window_mean(dd.flatten(0, 1), pts[None].expand(K * 6, -1, -1).contiguous(), torch.full((K * 6, Q), sg, device=dev)).T.view(Q, K, 6)
    T = sym(w6)                                                                                    # (Q, K, 3, 3), mass-weighted
    ax = torch.linalg.eigh(T / m.clamp_min(1e-9)[..., None, None])[1][..., -1]
    order = m.argsort(1, descending=True); cosm = math.cos(math.radians(a.merge_deg))
    pop = torch.full((Q, K), -1, dtype=torch.long, device=dev); pax = torch.zeros(Q, K, 3, device=dev); npop = torch.zeros(Q, dtype=torch.long, device=dev)
    ar = torch.arange(Q, device=dev)
    for j in range(K):                                                                             # greedy, heaviest first: join the first
        kj = order[:, j]; aj = ax[ar, kj]; mj = m[ar, kj]                                          # population within merge_deg
        c = (pax @ aj[:, :, None])[..., 0].abs(); c = torch.where(torch.arange(K, device=dev)[None] < npop[:, None], c, torch.full_like(c, -1))
        best = c.argmax(1); join = (c.gather(1, best[:, None])[:, 0] >= cosm) & (mj > 1e-4)
        new = ~join & (mj > 1e-4); slot = torch.where(join, best, npop)
        pop[ar, kj] = torch.where(join | new, slot, torch.full_like(slot, -1))
        pax[ar[new], npop[new]] = aj[new]; npop = npop + new.long()
    pm = torch.zeros(Q, K + 1, device=dev).scatter_add_(1, torch.where(pop >= 0, pop, K), m)[:, :K]
    PT = torch.zeros(Q, K + 1, 9, device=dev).scatter_add_(1, torch.where(pop >= 0, pop, K)[..., None].expand(-1, -1, 9), T.reshape(Q, K, 9))[:, :K].view(Q, K, 3, 3)
    pa = torch.linalg.eigh(PT / pm.clamp_min(1e-9)[..., None, None] + 1e-9 * torch.eye(3, device=dev))[1][..., -1]
    o2 = pm.argsort(1, descending=True)
    return pm.gather(1, o2), pa.gather(1, o2[..., None].expand(-1, -1, 3)), m.sum(1)


def structure_tensor(x, pts, rho):
    xs = gaussian(x.float()[None, None], 1.0)[0, 0]; gz, gy, gx = torch.gradient(xs)
    comps = [gaussian((u * v)[None, None], rho)[0, 0] for u, v in ((gz, gz), (gy, gy), (gx, gx), (gz, gy), (gz, gx), (gy, gx))]
    idx = pts.round().long(); G = torch.stack([c[idx[:, 0], idx[:, 1], idx[:, 2]] for c in comps], -1)
    ev, V = torch.linalg.eigh(sym(G))
    return (ev[:, 2] - ev[:, 0]) / ev.sum(-1).clamp_min(1e-9), V[:, :, 0]


def ang(u, v):
    u = torch.nn.functional.normalize(u, dim=-1); v = torch.nn.functional.normalize(v, dim=-1)
    return torch.rad2deg(torch.acos((u * v).sum(-1).abs().clamp(max=1)))


def model_at(x, pts, sg, feats):
    sig = torch.full((pts.shape[0],), sg, device=dev)
    with torch.no_grad():
        f = torch.cat([EI.query(net, feats, pts[j:j + 2048], sig[j:j + 2048], sph, vol_shape=x.shape)["f"].float() for j in range(0, pts.shape[0], 2048)])
        ro = EI.readout(net, x, pts, sig, sph, feats=feats, iso_k=ET.VOTE_KAPPA_MIN, **EI.parse_options(a.readout))
    mass = ro["mass"].float(); R = ro["R"].float(); sw = ro["sheet"].float()
    lm = mass * (1 - sw); o2 = lm.argsort(1, descending=True); ar = torch.arange(pts.shape[0], device=dev)
    ax = R[..., :, 2]
    p1 = ax[ar, o2[:, 0]]; p2 = ax[ar, o2[:, 1]] if mass.shape[1] > 1 else p1
    l1 = lm.gather(1, o2[:, :1])[:, 0]; l2 = lm.gather(1, o2[:, 1:2])[:, 0] if mass.shape[1] > 1 else torch.zeros_like(l1)
    dy = (lm[..., None, None] * ax[..., :, None] * ax[..., None, :]).sum(1); dom = torch.linalg.eigh(dy + 1e-9 * torch.eye(3, device=dev))[1][..., -1]
    o3 = o2[:, :3]; L0 = lm.gather(1, o3) >= a.count_min; L0[:, 0] = True                         # the line populations refined from the image
    Ar = refine_axes(x, pts, sg, ax.gather(1, o3[..., None].expand(-1, -1, 3)), L0)
    return {"peak": sph[f.argmax(1)], "p1": p1, "p2": p2, "p1r": Ar[:, 0], "p2r": Ar[:, 1] if Ar.shape[1] > 1 else Ar[:, 0], "l1": l1, "l2": l2, "count": (lm >= a.count_min).sum(1), "mass": mass.sum(1), "lmass": lm.sum(1),
            "smass": (mass * sw).sum(1), "bg": ro["bg"].float(), "iso": ro["iso"].float(), "dom": dom}


# ---- crops ------------------------------------------------------------------------------------------------------
centres = DIO.select_crops(a.data, a.crops, a.seed)
print("crop centres (DiSCo voxels x, y, z):", [(c.tolist(), l) for c, l in centres], flush=True)

ss_of = lambda v: max(2, int(round(v / 0.5)))                                                # fine cells of 0.5 um (at least 2 per voxel)
acc = {}; both = {}; both_ang = {}; cnt = {}; massr = {}; shows = {}
ABINS = ((20, 40), (40, 60), (60, 90.01))
for v in a.voxel:
    for ci, (c_, lab_) in enumerate(centres):
        centre_um = (c_[::-1].astype(np.float64)) * UNIT                                          # (z, y, x) um
        rd = DISCO.render(centre_um, v, a.n, ss_of(v)); occ, dyad = rd["occ"], rd["dyad"]
        x = DIO.image(occ.sum(0).clamp(max=1), LOOK, ci, lambda y, s_: gaussian(y[None, None], s_)[0, 0])
        with torch.no_grad():
            feats = EI.features(net, x)
        g1 = torch.arange(8, a.n - 8, a.step, device=dev, dtype=torch.float32)
        pts = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
        gu, gv = torch.meshgrid(torch.arange(a.n, device=dev, dtype=torch.float32), torch.arange(a.n, device=dev, dtype=torch.float32), indexing="ij")
        spts = torch.stack([torch.full_like(gu, a.n // 2), gu, gv], -1).view(-1, 3)
        for sg in a.sigmas:
            pm, pa, occ_t = truth(occ, dyad, pts, sg)
            m1 = pm[:, 0]; m2 = pm[:, 1] if pm.shape[1] > 1 else torch.zeros_like(m1)
            single = (m1 >= a.count_min) & (m2 < 0.2 * m1); core = (m2 >= a.count_min) & (m2 >= 0.5 * m1)
            ax1, ax2 = pa[:, 0], pa[:, 1] if pa.shape[1] > 1 else pa[:, 0]
            _, st = structure_tensor(x, pts, sg); mo = model_at(x, pts, sg, feats)
            key = (v, sg)
            for en, d1 in (("structure tensor", st), ("model: density peak", mo["peak"]), ("model: heaviest line population", mo["p1"]),
                           ("model + image refinement", mo["p1r"])):
                acc.setdefault(key + ("single", en), []).append(ang(d1, ax1)[single].cpu())
                acc.setdefault(key + ("core", en), []).append(torch.minimum(ang(d1, ax1), ang(d1, ax2))[core].cpu())
            okb = (((ang(mo["p1"], ax1) <= 20) & (ang(mo["p2"], ax2) <= 20)) | ((ang(mo["p1"], ax2) <= 20) & (ang(mo["p2"], ax1) <= 20))) & (mo["l2"] >= a.count_min)
            both.setdefault(key, []).append(okb[core].float().cpu())
            okr = (((ang(mo["p1r"], ax1) <= 20) & (ang(mo["p2r"], ax2) <= 20)) | ((ang(mo["p1r"], ax2) <= 20) & (ang(mo["p2r"], ax1) <= 20))) & (mo["l2"] >= a.count_min)
            both.setdefault(key + ("refined",), []).append(okr[core].float().cpu())
            acc.setdefault(key + ("core both axes", "model"), []).append(torch.maximum(torch.minimum(ang(mo["p1"], ax1), ang(mo["p1"], ax2)),
                                                                                       torch.minimum(ang(mo["p2"], ax1), ang(mo["p2"], ax2)))[core & (mo["l2"] >= a.count_min)].cpu())
            acc.setdefault(key + ("core both axes", "refined"), []).append(torch.maximum(torch.minimum(ang(mo["p1r"], ax1), ang(mo["p1r"], ax2)),
                                                                                         torch.minimum(ang(mo["p2r"], ax1), ang(mo["p2r"], ax2)))[core & (mo["l2"] >= a.count_min)].cpu())
            both_ang.setdefault(key, []).append(torch.stack([ang(ax1, ax2), okb.float(), torch.minimum(ang(st, ax1), ang(st, ax2)),
                                                             torch.minimum(ang(mo["p1"], ax1), ang(mo["p1"], ax2))], 1)[core].cpu())
            tc = (pm >= a.count_min).sum(1).clamp(max=3); mc = mo["count"].clamp(max=3)
            cnt.setdefault(key, []).append(torch.stack([tc, mc], 1).cpu())
            inn = occ_t > 0.02
            massr.setdefault(key, []).append(torch.stack([occ_t, mo["mass"], mo["bg"]], 1)[inn].cpu())
            print(f"{v:g} um, crop {ci} ({lab_}), sigma {sg:g}: single {int(single.sum())} core {int(core.sum())}; median angle single ST "
                  f"{float(ang(st, ax1)[single].median()) if single.any() else float('nan'):.0f} / model {float(ang(mo['p1'], ax1)[single].median()) if single.any() else float('nan'):.0f}; "
                  f"cores both found {float(okb[core].float().mean()) * 100 if core.any() else float('nan'):.0f} %; "
                  f"occupancy {float(occ_t[inn].mean()):.2f} vs model oriented mass {float(mo['mass'][inn].mean()):.2f}", flush=True)
            if sg == a.show_sigma:
                spm, spa, socc = truth(occ, dyad, spts, sg); smo = model_at(x, spts, sg, feats); coh, sst = structure_tensor(x, spts, sg)
                tdy = (spm[..., None, None] * spa[..., :, None] * spa[..., None, :]).sum(1); tdom = torch.linalg.eigh(tdy + 1e-9 * torch.eye(3, device=dev))[1][..., -1]
                shows[(v, ci)] = {"img": x[a.n // 2].cpu().numpy(), "tdom": tdom.cpu().numpy(), "tocc": socc.cpu().numpy(), "tcount": (spm >= a.count_min).sum(1).cpu().numpy(),
                                  "mdom": smo["dom"].cpu().numpy(), "mlm": smo["lmass"].cpu().numpy(), "msm": smo["smass"].cpu().numpy(), "mmass": smo["mass"].cpu().numpy(),
                                  "mcount": smo["count"].cpu().numpy(), "st": sst.cpu().numpy(), "coh": coh.cpu().numpy(), "label": lab_,
                                  "err": torch.where(spm[:, 0] >= a.count_min, ang(smo["p1"], spa[:, 0]), torch.full_like(socc, float("nan"))).cpu().numpy(),
                                  "err_st": torch.where(spm[:, 0] >= a.count_min, ang(sst, spa[:, 0]), torch.full_like(socc, float("nan"))).cpu().numpy()}
        del occ, dyad, feats; torch.cuda.empty_cache()


# ---- summary ----------------------------------------------------------------------------------------------------
def st_(x):
    x = x[torch.isfinite(x)]
    return {"n": int(x.numel())} if x.numel() < 5 else {"n": int(x.numel()), "median": float(x.median()), "within20": float((x <= 20).float().mean())}


summary = {}
for key in sorted({k[:2] for k in acc}):
    v, sg = key; s = {}
    for cls in ("single", "core"):
        for en in ("structure tensor", "model: density peak", "model: heaviest line population", "model + image refinement"):
            s[f"{cls}|{en}"] = st_(torch.cat(acc[key + (cls, en)]))
    for en in ("model", "refined"):                                                                # the worse of the two populations' angles
        s[f"core both axes|{en}"] = st_(torch.cat(acc[key + ("core both axes", en)]))
    b = torch.cat(both[key]); s["core both found"] = float(b.mean()) if b.numel() else float("nan")
    b = torch.cat(both[key + ("refined",)]); s["core both found (refined)"] = float(b.mean()) if b.numel() else float("nan")
    ba = torch.cat(both_ang[key])                                                                  # by the crossing angle (truth's two axes)
    for lo_, hi_ in ABINS:
        sel = (ba[:, 0] >= lo_) & (ba[:, 0] < hi_)
        s[f"core {lo_:g}-{min(hi_, 90):g} deg"] = {"n": int(sel.sum()), "both found": float(ba[sel, 1].mean()) if sel.any() else float("nan"),
                                                   "ST nearer median": float(ba[sel, 2].median()) if sel.any() else float("nan"),
                                                   "line nearer median": float(ba[sel, 3].median()) if sel.any() else float("nan")}
    c = torch.cat(cnt[key]); s["count"] = {f"truth {t}": {f"model {m}": float(((c[:, 0] == t) & (c[:, 1] == m)).sum() / max(int((c[:, 0] == t).sum()), 1)) for m in range(4)} for t in range(1, 4)}
    s["count exact"] = float((c[:, 0] == c[:, 1])[c[:, 0] >= 1].float().mean())
    mr = torch.cat(massr[key]); s["occupancy mean"] = float(mr[:, 0].mean()); s["oriented mass mean"] = float(mr[:, 1].mean())
    s["mass corr"] = float(np.corrcoef(mr[:, 0].numpy(), mr[:, 1].numpy())[0, 1]); s["background mean"] = float(mr[:, 2].mean())
    summary[f"{v:g} um|sigma {sg:g}"] = s
json.dump({"crops": [(c.tolist(), l) for c, l in centres], "look": LOOK, "summary": summary}, open(a.out + ".json", "w"), indent=1)
for k, s in summary.items():
    print(f"== {k}: single median ST {s['single|structure tensor'].get('median', np.nan):.0f} / peak {s['single|model: density peak'].get('median', np.nan):.0f} / line "
          f"{s['single|model: heaviest line population'].get('median', np.nan):.0f} deg (n {s['single|structure tensor']['n']}); cores ST {s['core|structure tensor'].get('median', np.nan):.0f} / line "
          f"{s['core|model: heaviest line population'].get('median', np.nan):.0f} / refined {s['core|model + image refinement'].get('median', np.nan):.0f} deg (n {s['core|structure tensor']['n']}), "
          f"both found {s['core both found'] * 100:.0f} % (refined {s['core both found (refined)'] * 100:.0f} %; worse of the two axes {s['core both axes|model'].get('median', np.nan):.1f} -> "
          f"{s['core both axes|refined'].get('median', np.nan):.1f} deg); single refined {s['single|model + image refinement'].get('median', np.nan):.1f}; count exact {s['count exact'] * 100:.0f} %; "
          f"occupancy {s['occupancy mean']:.2f} vs oriented mass {s['oriented mass mean']:.2f} (r {s['mass corr']:.2f}), background {s['background mean']:.2f}")

# ---- figures ----------------------------------------------------------------------------------------------------
def dec(axis, b):
    return np.clip(np.abs(axis[:, [2, 1, 0]]) * np.clip(b, 0, 1)[:, None], 0, 1)


n = a.n; rsh = lambda v_: v_.reshape(n, n, *v_.shape[1:])
cols = ["image", "truth: direction\n(heaviest-weighted; brightness occupancy)", "model: direction\n(lines; sheets grey)", "structure tensor: direction\n(brightness coherence)",
        "truth: populations", "model: line populations", "truth: occupancy", "model: oriented mass", "model error: heaviest line\nvs true heaviest (deg)", "structure tensor error (deg)"]
for v in a.voxel:
    keys = [k for k in shows if k[0] == v]
    fig, axs = plt.subplots(len(keys), len(cols), figsize=(2.15 * len(cols), 2.3 * len(keys)), squeeze=False, layout="constrained")
    for r_, k in enumerate(keys):
        S = shows[k]
        panels = [(S["img"], dict(cmap="gray", vmin=0, vmax=1)), (rsh(dec(S["tdom"], S["tocc"] / 0.4)), {}),
                  (rsh(np.clip(dec(S["mdom"], S["mlm"] / 0.4) + 0.6 * np.clip(S["msm"] / 0.4, 0, 1)[:, None], 0, 1)), {}),
                  (rsh(dec(S["st"], S["coh"] / 0.6)), {}), (rsh(S["tcount"]), dict(cmap="viridis", vmin=0, vmax=4)), (rsh(S["mcount"]), dict(cmap="viridis", vmin=0, vmax=4)),
                  (rsh(S["tocc"]), dict(cmap="magma", vmin=0, vmax=0.6)), (rsh(S["mmass"]), dict(cmap="magma", vmin=0, vmax=0.6)),
                  (rsh(S["err"]), dict(cmap="inferno", vmin=0, vmax=60)), (rsh(S["err_st"]), dict(cmap="inferno", vmin=0, vmax=60))]
        for j, (im, kw) in enumerate(panels):
            h = axs[r_, j].imshow(im, **kw); axs[r_, j].set_xticks([]); axs[r_, j].set_yticks([])
            if r_ == len(keys) - 1 and j in (4, 6, 8):
                fig.colorbar(h, ax=axs[:, j:j + 2].ravel().tolist(), location="bottom", shrink=0.6, pad=0.01)
        axs[r_, 0].set_ylabel(f"crop {k[1]}\n({S['label']} at 25 um)", fontsize=8)
        e = S["err"]; es = S["err_st"]
        axs[r_, 8].text(2, n - 4, f"median {np.nanmedian(e):.0f}", color="w", fontsize=7); axs[r_, 9].text(2, n - 4, f"median {np.nanmedian(es):.0f}", color="w", fontsize=7)
    for j, t_ in enumerate(cols):
        axs[0, j].set_title(t_, fontsize=8)
    fig.suptitle(f"{a.name} on DiSCo1 rendered as a greyscale volume ({a.look} fibres), {v:g} um voxels ({n * v:g} um crops), window sigma {a.show_sigma:g} voxels "
                 f"({a.show_sigma * v:g} um): middle slices, truth from the strands (connections within {a.merge_deg:g} deg joined)", fontsize=10)
    plt.savefig(f"{a.out}_slices_{v:g}um.png", dpi=72); plt.close(fig)

keys = list(summary); fig, axs = plt.subplots(1, 5, figsize=(25, 4.8), layout="constrained")
xl = [k.replace(" um|sigma ", " um\nsigma ") for k in keys]
ens = [("structure tensor", "#555555", None), ("model: density peak", "#2a78d6", "//"), ("model: heaviest line population", "#2a78d6", None),
       ("model + image refinement", "#eb6834", None)]
xs = np.arange(len(keys)); w = 0.2
for ax, cls, title in ((axs[0], "single", "single-population points:\nmedian angle to the true axis (deg)"), (axs[1], "core", "crossing cores:\nmedian angle to the nearer true axis (deg)")):
    for e_i, (en, col, hat) in enumerate(ens):
        ax.bar(xs + e_i * w, [summary[k][f"{cls}|{en}"].get("median", np.nan) for k in keys], w, color=col, hatch=hat, edgecolor="white", label=en)
    ax.set_xticks(xs + 1.5 * w); ax.set_xticklabels(xl, fontsize=7); ax.set_title(title, fontsize=10)
axs[0].legend(fontsize=8)
axs[2].bar(xs, [summary[k]["core both found"] * 100 for k in keys], 0.3, color="#2a78d6", label="both bundles found at cores")
axs[2].bar(xs + 0.3, [summary[k]["core both found (refined)"] * 100 for k in keys], 0.3, color="#eb6834", label="... after the image refinement")
axs[2].bar(xs + 0.6, [summary[k]["count exact"] * 100 for k in keys], 0.3, color="#1baf7a", label="population count exact")
axs[2].set_xticks(xs + 0.3); axs[2].set_xticklabels(xl, fontsize=7); axs[2].set_ylim(0, 100); axs[2].legend(fontsize=8)
axs[2].set_title("%% of points (cores: two populations, the lighter >= half;\ncount: line populations with mass >= %.2f, 3 = 3+)" % a.count_min, fontsize=10)
axs[3].bar(xs, [summary[k]["occupancy mean"] for k in keys], 0.4, color="#999999", label="true occupancy (window mean)")
axs[3].bar(xs + 0.4, [summary[k]["oriented mass mean"] for k in keys], 0.4, color="#2a78d6", label="model oriented mass")
for i, k in enumerate(keys):
    axs[3].text(xs[i] + 0.2, max(summary[k]["occupancy mean"], summary[k]["oriented mass mean"]) + 0.01, f"r {summary[k]['mass corr']:.2f}", ha="center", fontsize=7)
axs[3].set_xticks(xs + 0.2); axs[3].set_xticklabels(xl, fontsize=7); axs[3].legend(fontsize=8); axs[3].set_title("mass: mean over points with occupancy > 0.02\n(r: correlation per point)", fontsize=10)
ax = axs[4]; vs = [f"{v:g} um|sigma {a.show_sigma:g}" for v in a.voxel]; w = 0.8 / len(vs); pal = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
for i, k in enumerate(vs):
    ys = [summary[k][f"core {lo_:g}-{min(hi_, 90):g} deg"]["both found"] * 100 for lo_, hi_ in ABINS]
    ax.bar(np.arange(len(ABINS)) + i * w, ys, w, color=pal[i % 4], label=k.replace("|", ", "))
    for j, (lo_, hi_) in enumerate(ABINS):
        ax.text(j + i * w, ys[j] + 1, f"n {summary[k][f'core {lo_:g}-{min(hi_, 90):g} deg']['n']}", ha="center", fontsize=6)
ax.set_xticks(np.arange(len(ABINS)) + w); ax.set_xticklabels([f"{lo_:g}-{min(hi_, 90):g} deg" for lo_, hi_ in ABINS]); ax.set_ylim(0, 100); ax.legend(fontsize=8)
ax.set_title("crossing cores: both bundles found, by the crossing angle\n(truth's two axes; the phantoms cross at 30-90 deg)", fontsize=10)
fig.suptitle(f"{a.name} on DiSCo1 as a greyscale volume ({a.look} fibres; {len(centres)} crops of 96^3): against the strands' truth", fontsize=11)
plt.savefig(a.out + "_summary.png", dpi=75); print("saved figures", flush=True)
