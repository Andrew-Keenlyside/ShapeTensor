"""Single-fibre orientation on DiSCo (test only: DiSCo never enters training, the generator or model selection): at
resolutions where each axon is a resolved tube, the orientation read at points INSIDE a fibre against that fibre's own
tangent -- no window-mixed populations, no blend of a bundle's wandering fibres.

* **Render** (``disco_io``): each strand along a Catmull-Rom spline through its points (DiSCo's polylines turn ~21 deg
  every 15 um; the spline keeps the points and drops the kinks), at ``--voxel`` um -- 0.25: axon radius ~3.4 voxels
  (beyond the phantoms' 0.6-1.5), 0.5: ~1.7 (their top), 1: ~0.85 -- with the bright phantom-range look of
  ``disco_test.py`` and its crops (4, 4, 4, 3, 2, 1 bundles at 25 um).
* **Points**: voxels >= 8 from the crop's faces filled by one fibre (occupancy >= ``--fill``, its most common strand
  >= 95 % of it), up to ``--max-points`` per crop (seeded). **Truth**: that fibre's tangent there (the voxel's mean
  tangent dyad, from the spline's derivative).
* **Estimators** at window sigma ``--sigmas`` (voxels): the structure tensor (gradient sigma ``--grad``, integration
  rho = sigma); the model's density peak, its heaviest line population's axis and its lines' mass-weighted mean
  direction, with the density sampled at each of ``--dirs`` directions (200 is the readout's default: a random axis is
  3.9 deg on average from the nearest sample, 1.9 at 800, 1.2 at 2000).
* **Two truths**: the fibre's tangent at the point, and the window's mean direction (all fibres' tangent dyads weighted
  by the window over the occupancy) -- what a population's axis is trained to be. The window's coherence (largest
  eigenvalue of that mean dyad, 1 = all its fibres parallel) splits the points: >= ``--coherent`` the window holds
  one direction.

Usage: python scripts/learn/disco_fibres.py OUT_STEM CKPT [--voxel 0.25 0.5 1] [--sigmas 1.5 2 3 4] [--dirs 200 800 2000]
"""
import argparse, json, sys
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
from shapetensor.learn.axis_refine import refine_axes
from shapetensor.learn.model import PopulationNet, window_mean
sys.path.insert(0, str(Path(__file__).resolve().parent))
import disco_io as DIO

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--name", default="overnight pilot")
ap.add_argument("--data", default="/hdd/andrew/disco_data/DiSCo1"); ap.add_argument("--voxel", type=float, nargs="+", default=[0.25, 0.5, 1.0])
ap.add_argument("--sigmas", type=float, nargs="+", default=[1.5, 2.0, 3.0, 4.0]); ap.add_argument("--dirs", type=int, nargs="+", default=[200, 800, 2000])
ap.add_argument("--crops", type=int, nargs=3, default=[3, 2, 1]); ap.add_argument("--n", type=int, default=96); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fill", type=float, default=0.75); ap.add_argument("--max-points", type=int, default=2000); ap.add_argument("--grad", type=float, default=1.0)
ap.add_argument("--show", default="0.5:3", help="voxel:sigma of the slice figure"); ap.add_argument("--no-spline", action="store_true")
ap.add_argument("--coherent", type=float, default=0.95)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
SPH = {d: EI.sphere_dirs(d, dev) for d in a.dirs}
LOOK = {"bg": 0.15, "contrast": 0.6, "noise": 0.02, "blur": 0.5}
DISCO = DIO.Disco(a.data, dev, spline=not a.no_spline); centres = DIO.select_crops(a.data, a.crops, a.seed)
print("crop centres (DiSCo voxels x, y, z):", [(c.tolist(), l) for c, l in centres], flush=True)
show_v, show_s = (float(x) for x in a.show.split(":"))


def sym(t6):
    return torch.stack([torch.stack([t6[..., 0], t6[..., 3], t6[..., 4]], -1), torch.stack([t6[..., 3], t6[..., 1], t6[..., 5]], -1),
                        torch.stack([t6[..., 4], t6[..., 5], t6[..., 2]], -1)], -2)


def structure_tensor(x, pts, rho):
    xs = gaussian(x.float()[None, None], a.grad)[0, 0]; gz, gy, gx = torch.gradient(xs)
    comps = [gaussian((u * v)[None, None], rho)[0, 0] for u, v in ((gz, gz), (gy, gy), (gx, gx), (gz, gy), (gz, gx), (gy, gx))]
    idx = pts.round().long(); G = torch.stack([c[idx[:, 0], idx[:, 1], idx[:, 2]] for c in comps], -1)
    return torch.linalg.eigh(sym(G))[1][:, :, 0]


def ang(u, v):
    u = torch.nn.functional.normalize(u, dim=-1); v = torch.nn.functional.normalize(v, dim=-1)
    return torch.rad2deg(torch.acos((u * v).sum(-1).abs().clamp(max=1)))


def model_at(x, pts, sg, feats, nd):
    sph = SPH[nd]; sig = torch.full((pts.shape[0],), sg, device=dev)
    with torch.no_grad():
        f = torch.cat([EI.query(net, feats, pts[j:j + 1024], sig[j:j + 1024], sph, vol_shape=x.shape)["f"].float() for j in range(0, pts.shape[0], 1024)])
        ro = EI.readout(net, x, pts, sig, sph, feats=feats, iso_k=ET.VOTE_KAPPA_MIN, chunk=512)
    mass = ro["mass"].float(); R = ro["R"].float(); lm = mass * (1 - ro["sheet"].float()); ax = R[..., :, 2]
    top = lm.argmax(1); ar = torch.arange(pts.shape[0], device=dev)
    dy = (lm[..., None, None] * ax[..., :, None] * ax[..., None, :]).sum(1)
    out = {"model: density peak": sph[f.argmax(1)], "model: heaviest line": ax[ar, top],
           "model: lines' mean": torch.linalg.eigh(dy + 1e-9 * torch.eye(3, device=dev))[1][..., -1]}
    if nd == a.dirs[0]:                                                                            # the model's line populations refined from
        o3 = lm.argsort(1, descending=True)[:, :3]                                                 # the image (a structure tensor per population)
        L0 = lm.gather(1, o3) >= 0.03; L0[:, 0] = True
        out["model + image refinement: heaviest line"] = refine_axes(x, pts, sg, ax.gather(1, o3[..., None].expand(-1, -1, 3)), L0, grad_sigma=a.grad)[:, 0]
    return out


def stats(x):
    x = x[torch.isfinite(x)]
    return {"n": int(x.numel()), "median": float(x.median()), "p90": float(x.quantile(0.9)), "within2": float((x <= 2).float().mean()),
            "within5": float((x <= 5).float().mean()), "within10": float((x <= 10).float().mean())}


ss_of = lambda v: max(2, int(round(v / 0.25)))                                                     # fine cells <= 0.25 um (0.125 at 0.25 um)
acc = {}; slices = {}; npts = {}
for v in a.voxel:
    for ci, (c_, lab_) in enumerate(centres):
        rd = DISCO.render(c_[::-1].astype(np.float64) * DIO.UNIT, v, a.n, ss_of(v))
        occ_t = rd["occ"].sum(0); x = DIO.image(occ_t.clamp(max=1), LOOK, ci, lambda y, s_: gaussian(y[None, None], s_)[0, 0])
        T = sym(rd["dyad"].sum(0).movedim(0, -1))                                                   # (n, n, n, 3, 3)
        inner = torch.zeros_like(occ_t, dtype=torch.bool); inner[8:-8, 8:-8, 8:-8] = True
        ok = inner & (occ_t >= a.fill) & (rd["purity"] >= 0.95)
        cand = ok.nonzero(); g = torch.Generator(device="cpu").manual_seed(1000 * ci + int(v * 100))
        cand = cand[torch.randperm(len(cand), generator=g)[:a.max_points].to(dev)]; pts = cand.float(); npts[(v, ci)] = len(cand)
        tax = torch.linalg.eigh(T[cand[:, 0], cand[:, 1], cand[:, 2]])[1][..., -1]
        with torch.no_grad():
            feats = EI.features(net, x)
        line = f"{v:g} um, crop {ci} ({lab_}): {len(cand)} fibre points (of {int(ok.sum())});"
        D6 = rd["dyad"].sum(0)
        for sg in a.sigmas:
            Q = len(pts); wm = lambda m_: window_mean(m_, pts[None].expand(m_.shape[0], -1, -1).contiguous(), torch.full((m_.shape[0], Q), sg, device=dev)).T
            Tw = sym(wm(D6)) / wm(occ_t[None])[:, 0, None, None].clamp_min(1e-9)                     # the window's mean tangent dyad
            ev, V = torch.linalg.eigh(Tw); wax = V[..., -1]; coh = ev[:, -1] / ev.sum(-1).clamp_min(1e-9)
            acc.setdefault((v, sg, "_coherence"), []).append(coh.cpu())
            acc.setdefault((v, sg, "_tangent vs window mean"), []).append(ang(tax, wax).cpu())
            ests = {"structure tensor": structure_tensor(x, pts, sg)}
            for nd in a.dirs:
                for k_, d_ in model_at(x, pts, sg, feats, nd).items():
                    ests[f"{k_} ({nd} dirs)"] = d_
            for en, d_ in ests.items():
                acc.setdefault((v, sg, en), []).append(ang(d_, tax).cpu())
                acc.setdefault((v, sg, "window|" + en), []).append(ang(d_, wax).cpu())
            line += f" sigma {sg:g}: ST {float(acc[(v, sg, 'structure tensor')][-1].median()):.1f} / model line ({a.dirs[-1]}) {float(acc[(v, sg, f'model: heaviest line ({a.dirs[-1]} dirs)')][-1].median()):.1f};"
        print(line, flush=True)
        if v == show_v and ci < 4:                                                                 # the slice figure: the middle slice
            n = a.n; gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
            sp = torch.stack([torch.full_like(gu, n // 2), gu, gv], -1).view(-1, 3); si = sp.long()
            sok = ok[si[:, 0], si[:, 1], si[:, 2]]; st_ = T[si[:, 0], si[:, 1], si[:, 2]]
            sax = torch.linalg.eigh(st_ + 1e-9 * torch.eye(3, device=dev))[1][..., -1]
            sst = structure_tensor(x, sp, show_s); smo = model_at(x, sp, show_s, feats, a.dirs[-1])[f"model: heaviest line"]
            sre = model_at(x, sp, show_s, feats, a.dirs[0])["model + image refinement: heaviest line"]
            nan = torch.full((len(sp),), float("nan"), device=dev)
            slices[ci] = {"img": x[n // 2].cpu().numpy(), "occ": occ_t[n // 2].clamp(max=1).cpu().numpy(), "ok": sok.cpu().numpy(), "tax": sax.cpu().numpy(),
                          "st": sst.cpu().numpy(), "mo": smo.cpu().numpy(), "e_st": torch.where(sok, ang(sst, sax), nan).cpu().numpy(),
                          "e_mo": torch.where(sok, ang(smo, sax), nan).cpu().numpy(), "re": sre.cpu().numpy(),
                          "e_re": torch.where(sok, ang(sre, sax), nan).cpu().numpy(), "label": lab_}
        del rd, T, feats; torch.cuda.empty_cache()

summary = {}
for (v, sg, en) in acc:
    if en.startswith("_"):
        continue
    x_ = torch.cat(acc[(v, sg, en)]); coh_ = torch.cat(acc[(v, sg, "_coherence")])
    summary[f"{v:g} um|sigma {sg:g}|{en}"] = stats(x_)
    summary[f"{v:g} um|sigma {sg:g}|{en}|coherent window"] = stats(x_[coh_ >= a.coherent])
for v in a.voxel:
    for sg in a.sigmas:
        coh_ = torch.cat(acc[(v, sg, "_coherence")]); tw = torch.cat(acc[(v, sg, "_tangent vs window mean")])
        summary[f"{v:g} um|sigma {sg:g}|_windows"] = {"coherent share": float((coh_ >= a.coherent).float().mean()), "coherence median": float(coh_.median()),
                                                      "tangent vs window mean median": float(tw.median())}
json.dump({"crops": [(c.tolist(), l) for c, l in centres], "points": {f"{v:g} um|crop {ci}": k for (v, ci), k in npts.items()}, "summary": summary},
          open(a.out + ".json", "w"), indent=1)
ens = [k for k in dict.fromkeys(k[2] for k in acc) if not k.startswith("_") and not k.startswith("window|")]
for v in a.voxel:
    print(f"== {v:g} um, window sigma {a.sigmas}: coherent windows (>= {a.coherent}) " + " / ".join(f"{summary[f'{v:g} um|sigma {sg:g}|_windows']['coherent share'] * 100:.0f} %" for sg in a.sigmas)
          + "; the fibre's tangent vs the window's mean " + " / ".join(f"{summary[f'{v:g} um|sigma {sg:g}|_windows']['tangent vs window mean median']:.1f}" for sg in a.sigmas) + " deg")
    for tr_, pre in (("the fibre's tangent", ""), ("the fibre's tangent, coherent windows", "|coherent window"), ("the window's mean direction", "window|")):
        print(f"   median angle to {tr_} (within 5 deg):")
        for en in ens:
            k_ = (lambda sg: f"{v:g} um|sigma {sg:g}|window|{en}") if pre == "window|" else (lambda sg: f"{v:g} um|sigma {sg:g}|{en}{pre}")
            print(f"     {en:40s} " + "  ".join(f"{summary[k_(sg)].get('median', float('nan')):5.1f} ({summary[k_(sg)].get('within5', float('nan')) * 100:3.0f} %)" for sg in a.sigmas))

# ---- figures ----------------------------------------------------------------------------------------------------
pal = {200: "#9ec5f0", 800: "#4f95e0", 2000: "#1f5fae"}
fig, axs = plt.subplots(3, len(a.voxel), figsize=(5.2 * len(a.voxel), 11.5), squeeze=False, layout="constrained")
for j, v in enumerate(a.voxel):
    for r_, (pre, suf, yl) in enumerate((("", "", "median angle to the fibre's tangent (deg)"), ("", "|coherent window", "the same, coherent windows only"),
                                         ("window|", "", "median angle to the window's mean direction (deg)"))):
        ax = axs[r_, j]; met = "median"
        for en in ens:
            nd = int(en.split("(")[-1].split()[0]) if "dirs" in en else None
            col = "#555555" if nd is None else pal.get(nd, "#2a78d6")
            col = "#eb6834" if "refinement" in en else col
            ls = "-" if nd is None or "heaviest" in en else (":" if "peak" in en else "--")
            ys = [summary[f"{v:g} um|sigma {sg:g}|{pre}{en}{suf}"].get(met, np.nan) for sg in a.sigmas]
            ax.plot(a.sigmas, ys, ls, marker="o", ms=4, lw=2.2 if nd is None else 1.6, color=col, label=en)
        ax.set_xlabel("window sigma / structure-tensor rho (voxels)"); ax.set_ylabel(yl); ax.grid(color="#e4e3df", lw=0.6); ax.set_ylim(0, None)
        if r_ == 0:
            ax.set_title(f"{v:g} um voxels (axon radius ~{0.85 / v:.1f} voxels)", fontsize=10)
        if r_ == 1:
            ax.set_title("coherent windows: " + ", ".join(f"{summary[f'{v:g} um|sigma {sg:g}|_windows']['coherent share'] * 100:.0f} %" for sg in a.sigmas)
                         + f" of points (sigma {', '.join(f'{sg:g}' for sg in a.sigmas)})", fontsize=8)
axs[0, -1].legend(fontsize=7)
fig.suptitle(f"{a.name} on DiSCo1 at single-fibre resolution: orientation at points inside a fibre against its own tangent "
             f"(6 crops, {sum(npts.values()) // len(a.voxel)} points per resolution)\nsolid: heaviest line / structure tensor (grey) / model + image refinement (orange); dashed: lines' mean; dotted: density peak; "
             f"shade: directions sampled)", fontsize=10)
plt.savefig(a.out + "_accuracy.png", dpi=75); plt.close(fig)

if slices:
    def dec(axis, b):
        return np.clip(np.abs(axis[:, [2, 1, 0]]) * np.clip(b, 0, 1)[:, None], 0, 1)
    n = a.n; rsh = lambda u: u.reshape(n, n, *u.shape[1:])
    cols = ["image", "truth: fibre tangent\n(inside fibres)", f"structure tensor (rho {show_s:g})", f"model: heaviest line\n({a.dirs[-1]} dirs)",
            f"model + image refinement\n({a.dirs[0]} dirs)", "structure tensor error (deg)", "model error (deg)", "model + refinement error (deg)"]
    fig, axs = plt.subplots(len(slices), len(cols), figsize=(2.6 * len(cols), 2.7 * len(slices)), squeeze=False, layout="constrained")
    for r_, (ci, S) in enumerate(slices.items()):
        okm = S["ok"].astype(float)
        panels = [(S["img"], dict(cmap="gray", vmin=0, vmax=1)), (rsh(dec(S["tax"], okm)), {}), (rsh(dec(S["st"], S["occ"].ravel() + 0.25)), {}),
                  (rsh(dec(S["mo"], S["occ"].ravel() + 0.25)), {}), (rsh(dec(S["re"], S["occ"].ravel() + 0.25)), {}),
                  (rsh(S["e_st"]), dict(cmap="inferno", vmin=0, vmax=20)), (rsh(S["e_mo"]), dict(cmap="inferno", vmin=0, vmax=20)),
                  (rsh(S["e_re"]), dict(cmap="inferno", vmin=0, vmax=20))]
        for j, (im, kw) in enumerate(panels):
            h = axs[r_, j].imshow(im, **kw); axs[r_, j].set_xticks([]); axs[r_, j].set_yticks([])
            if j >= 5:
                axs[r_, j].set_facecolor("#20202a"); axs[r_, j].text(2, n - 4, f"median {np.nanmedian(im):.1f}", color="w", fontsize=7)
        axs[r_, 0].set_ylabel(f"crop {ci}\n({S['label']} at 25 um)", fontsize=8)
    fig.colorbar(h, ax=axs[:, 5:].ravel().tolist(), shrink=0.5, label="angle to the fibre's tangent (deg)")
    for j, t_ in enumerate(cols):
        axs[0, j].set_title(t_, fontsize=9)
    fig.suptitle(f"{a.name} on DiSCo1 at {show_v:g} um voxels ({a.n * show_v:g} um crops), window sigma {show_s:g} voxels: middle slices; "
                 f"directions coloured red x, green y, blue z; errors at fibre-interior voxels only", fontsize=10)
    plt.savefig(f"{a.out}_slices_{show_v:g}um.png", dpi=75); plt.close(fig)
print("saved figures", flush=True)
