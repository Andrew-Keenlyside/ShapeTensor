"""A phantom suite for entity-field checkpoints: arrangements built from the compositional operators with set parameters,
each with its ground truth -- the population region, the mass in a window, the orientation (a line's axis, a sheet's
normal) -- imaged simply (a contrast per population, a background, blur, noise) at two difficulties.

Phantoms: stacked planes; concentric cylindrical shells; a two-layer helical wall (+-30 deg); a laminar stack whose fibre
direction turns 90 deg across it; fibres fanning from a source; spindle blobs aligned to a smooth field; fibres crossing
a sheet; a bundle through a grey / white-like step (its contrast flips across the step); fibre populations crossing at
90 and 45 deg (``two_peaks``: the share of shared windows whose density has two peaks; ``cross_err_deg``: their error);
hollow tubes. Imaging: easy, hard (more blur and noise), inverted (easy, negated).

Per checkpoint, phantom and imaging: ``region_corr`` (the region map against the populations' occupancy), ``mass_corr``
and ``mass_ratio`` (the mass output against the occupancy windowed at the query's sigma), ``axis_deg`` (median angle
between the model ODF's second-moment axis -- the long axis for lines and spindles, the normal for sheets -- and the
truth's, where one population holds >= 80 % of the window) and ``vote_axis_deg`` (the same from the vote's Bingham frame at the
density's peak: its mode axis, or a girdle's normal). A figure: image, region truth, each checkpoint's region map.

Usage: python scripts/learn/phantom_suite_entities.py OUT_STEM CKPT [CKPT ...]
"""
import json, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from shapetensor.filters import gaussian
from shapetensor.learn import compose as C
from shapetensor.learn import entity_infer as EI
from shapetensor.learn.generator import sheet_tensor
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.scene import H, Group, Scene, curves_like

out_stem, ckpts = sys.argv[1], sys.argv[2:]
dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 96; VOX = 2.0; c0 = torch.full((3,), (N - 1) / 2, device=dev)
U = lambda *v: C._unit(torch.tensor(v, device=dev, dtype=torch.float32))


def lines(sc, field, seeds, length, rf, kind="line"):
    P = C.grow(field, seeds, length, H, sc.shape)
    n, S = P.shape[:2]; pid = int(sc.ids(1, 50.0)[0]); obj = sc.ids(n, rf)
    for o_ in obj.tolist():
        sc.parents[o_] = pid
    cv = curves_like(P, torch.full((n, S), float(rf), device=dev), sc.r, obj=obj, seg=torch.full((n,), pid, device=dev), contrast=60.0)
    return Group("phantom", kind, float(rf), 60.0, curves=[cv])


def sheets(sc, phi, T, t, mask=None):
    gz, gy, gx = torch.gradient(phi); gr = torch.stack([gz, gy, gx], -1); gn = gr.norm(dim=-1)
    u = phi / T; k = torch.floor(u); dist = (u - k - 0.5).abs() * T / gn.clamp_min(0.2)
    occ = ((t / 2 - dist) + 0.5).clamp(0, 1) * (gn > 0.2).float()
    if mask is not None:
        occ = occ * mask
    tens = sheet_tensor(gr / gn.clamp_min(1e-6)[..., None]).movedim(-1, 0).contiguous()
    desc = torch.zeros((5,) + (N,) * 3, device=dev); desc[0] = t / 2
    oid = int(sc.ids(1, t / 2)[0])
    return Group("phantom", "sheet", t / 2, 60.0, dense=[(occ, tens, desc, 60.0, torch.full((N,) * 3, oid, device=dev, dtype=torch.long))])


def grid():
    g = torch.stack(torch.meshgrid(*[torch.arange(N, device=dev, dtype=torch.float32)] * 3, indexing="ij"), -1)
    return g


def build(name, seed=7):
    sc = Scene((N,) * 3, seed, dev, VOX); r = sc.r; G = grid(); gs = []; step = None
    if name == "planes":
        n = U(0.3, 0.5, 0.81); gs.append(sheets(sc, (G - c0) @ n, 8.0, 3.0))
    elif name == "shells":
        a = U(0.2, 0.9, 0.3); v = G - c0; rho = (v - (v @ a)[..., None] * a).norm(dim=-1)
        gs.append(sheets(sc, rho, 7.0, 2.5, mask=(rho < 40).float()))
    elif name == "helical_wall":
        a = U(1.0, 0.2, 0.1)
        for hand, (r0, r1) in ((1.0, (10, 20)), (-1.0, (24, 34))):
            f = C.Cylinder(30.0, hand, c=c0, a=a)
            seeds = torch.cat([C.seeds_in("shell", 40, (N,) * 3, r, c=c0, d=a, R=rr) for rr in np.linspace(r0, r1, 4)])
            gs.append(lines(sc, f, seeds, 160.0, 1.0))
    elif name == "laminar_rotation":
        f = C.Laminar(c0, U(1.0, 0.0, 0.0), U(0.0, 1.0, 0.0), 0.0, math.radians(90.0) / 60.0)
        gs.append(lines(sc, f, C.seeds_in("box", 220, (N,) * 3, r), 140.0, 0.9))
    elif name == "fanning":
        f = C.Source(c0 - 70 * U(0.0, 0.3, 1.0), U(0.0, 0.3, 1.0), 0.3, 1.0)
        gs.append(lines(sc, f, C.seeds_in("disc", 160, (N,) * 3, r, c=c0, d=U(0.0, 0.3, 1.0), R=30.0), 160.0, 1.0))
    elif name == "spindles":
        f = C.Smooth(U(0.0, 1.0, 1.0), 0.3, 70.0, r)
        X = C.seeds_in("box", 600, (N,) * 3, r); keep = []
        for i in range(X.shape[0]):
            if len(keep) < 140 and (not keep or float((X[keep] - X[i]).norm(dim=1).min()) > 12.0):
                keep.append(i)
        Cc = X[keep]; d = f(Cc); q = C._perp(d); w = torch.linalg.cross(d, q)
        Rm = torch.stack([q, w, d], -1); ax = torch.tensor([2.2, 2.2, 6.5], device=dev).expand(len(keep), 3)
        gs.append(sc._solids(Cc, Rm @ torch.diag_embed(ax), "phantom", sc.ids(len(keep), 2.2)))
    elif name == "fibres_through_sheet":
        n = U(0.0, 0.2, 1.0); gs.append(sheets(sc, (G - c0) @ n + 4.0, 1e4, 4.0))
        d = U(0.9, 0.0, 0.45)
        gs.append(lines(sc, C.Smooth(d, 0.0, 50.0, r), C.seeds_in("disc", 120, (N,) * 3, r, c=c0, d=d, R=16.0), 160.0, 1.0))
    elif name in ("crossing_90", "crossing_45"):
        ang = math.radians(90.0 if name == "crossing_90" else 45.0)
        for k, t in enumerate((-ang / 2, ang / 2)):
            d = U(0.25, math.cos(t), math.sin(t))
            gs.append(lines(sc, C.Smooth(d, 0.0, 50.0, r), C.seeds_in("disc", 130, (N,) * 3, r, c=c0, d=d, R=26.0), 170.0, 1.0))
    elif name == "tubes":
        f = C.Smooth(U(0.3, 0.8, 0.5), 0.25, 80.0, r)
        g_ = lines(sc, f, C.seeds_in("box", 9, (N,) * 3, r), 170.0, 4.0)
        cv = g_.curves[0]; cv.wall = cv.rad * 0.3; cv.lumen_c = torch.full_like(cv.rad, -30.0); gs.append(g_)
    elif name == "bundle_through_step":
        a = U(math.cos(math.radians(60)), 0.0, math.sin(math.radians(60)))
        gs.append(sc.bundle(None, "single", axis=a, centre=c0, radius=18.0, curv=1e-4, fibre={"rf_um": 3.0, "jitter": 0.1, "disp": 0.1}))
        step = ((G[..., 0] - c0[0]) * 0.8 + (G[..., 2] - c0[2]) * 0.6) > 0
    sc.groups = gs
    ren = sc.render()
    occ = torch.stack([g_["occupancy"].float() for g_ in ren["groups"]]); occ = occ / occ.sum(0).clamp_min(1.0)[None]
    tens = torch.stack([g_["tensor"].float() for g_ in ren["groups"]])
    kinds = [g_.kind for g_ in gs]
    return occ, tens, kinds, step


def image(occ, step, hard, seed=3, invert=False):
    rng = torch.Generator(device=dev).manual_seed(seed)
    con = [-0.3, 0.25, -0.2][:occ.shape[0]]
    if step is not None:
        bgv = torch.where(step, torch.tensor(0.75, device=dev), torch.tensor(0.35, device=dev))
        img = bgv * (1 - occ.sum(0)) + 0.55 * occ.sum(0)
    else:
        img = 0.55 + sum(cc * o for cc, o in zip(con, occ))
    img = gaussian(img, 1.5 if hard else 1.0) + (0.08 if hard else 0.03) * torch.randn(img.shape, generator=rng, device=dev)
    return (1 - img).clamp(0, 1) if invert else img.clamp(0, 1)


def sym(t6):
    return torch.stack([torch.stack([t6[..., 0], t6[..., 3], t6[..., 4]], -1), torch.stack([t6[..., 3], t6[..., 1], t6[..., 5]], -1),
                        torch.stack([t6[..., 4], t6[..., 5], t6[..., 2]], -1)], -2)


PHANTOMS = ["planes", "shells", "helical_wall", "laminar_rotation", "fanning", "spindles", "fibres_through_sheet", "bundle_through_step",
            "crossing_90", "crossing_45", "tubes"]
models = []
for c in ckpts:
    ck = torch.load(c, map_location=dev, weights_only=False)
    m_ = PopulationNet.from_args(ck["args"]).to(dev).eval(); m_.load_state_dict(ck["model"])
    models.append((c.split("/")[-2] if c.endswith("model.pt") else c.split("/")[-1].replace(".pt", ""), m_))
dirs = EI.sphere_dirs(200, dev)
gp = torch.arange(14, N - 14, 5, device=dev, dtype=torch.float32)
pts = torch.stack(torch.meshgrid(gp, gp, gp, indexing="ij"), -1).reshape(-1, 3); SIG = 2.0
rows, figrows = [], []
for ph in PHANTOMS:
    occ, tens, kinds, step = build(ph)
    tot = occ.sum(0).clamp(0, 1)
    w = gaussian(occ, SIG)                                                                         # (K, Z, Y, X) windowed occupancy
    pi = pts.long()
    wk = w[:, pi[:, 0], pi[:, 1], pi[:, 2]].T                                                       # (Q, K)
    tk = gaussian(tens * occ[:, None], SIG)[:, :, pi[:, 0], pi[:, 1], pi[:, 2]].permute(2, 0, 1)    # (Q, K, 6)
    m_true = wk.sum(1); dom = wk.argmax(1); share = wk.max(1).values / m_true.clamp_min(1e-6)
    sel = (m_true > 0.15) & (share > 0.8)
    Tt = sym(tk[torch.arange(pts.shape[0], device=dev), dom]); ev, evec = torch.linalg.eigh(Tt)
    is_sheet = torch.tensor([kinds[int(k)] == "sheet" for k in dom.tolist()], device=dev)
    ax_true = torch.where(is_sheet[:, None], evec[..., 0], evec[..., 2])                           # a sheet's normal, a line's (blob's) long axis
    for hard in (False, True, "inverted"):
        x = image(occ, step, hard is True, invert=hard == "inverted")
        fr = {"phantom": ph, "hard": hard, "img": x[N // 2].cpu(), "truth": tot[N // 2].cpu(), "pred": []}
        for name, model in models:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                P = model(x[None, None], voxel=torch.tensor([math.log2(VOX)], device=dev))
                feats = EI.features(model, x, voxel_um=VOX)
                q = EI.query(model, feats, pts, torch.full((pts.shape[0],), SIG, device=dev), dirs)
            row = {"ckpt": name, "phantom": ph, "hard": hard}
            if "region_logit" in P:
                rp = torch.sigmoid(P["region_logit"].float()).reshape(N, N, N); mv = 12
                a_, b_ = rp[mv:-mv, mv:-mv, mv:-mv].flatten(), tot[mv:-mv, mv:-mv, mv:-mv].flatten()
                row["region_corr"] = round(float(torch.corrcoef(torch.stack([a_, b_]))[0, 1]), 3)
                fr["pred"].append((name, rp[N // 2].cpu()))
            if "m" in q:
                mm = q["m"].float()
                row["mass_corr"] = round(float(torch.corrcoef(torch.stack([mm, m_true]))[0, 1]), 3)
                on = m_true > 0.15
                row["mass_ratio"] = round(float((mm[on] / m_true[on]).median()), 3) if bool(on.any()) else None
            f = q["f"].float().clamp_min(0)                                                        # (Q, D) on the hemisphere
            M = (f[..., None, None] * dirs[:, :, None] * dirs[:, None, :]).sum(1) / f.sum(1).clamp_min(1e-9)[:, None, None]
            ev_m, evec_m = torch.linalg.eigh(M)
            ax_m = torch.where(is_sheet[:, None], evec_m[..., 0], evec_m[..., 2])
            ang = torch.rad2deg(torch.acos((ax_m * ax_true).sum(-1).abs().clamp(max=1)))
            row["axis_deg"] = round(float(ang[sel].median()), 1) if bool(sel.any()) else None
            Rk = q["R"].float()[torch.arange(pts.shape[0], device=dev), f.argmax(1)]                # the vote's frame at the density's peak:
            ax_v = torch.where(is_sheet[:, None], Rk[..., :, 0], Rk[..., :, 2])                    # the mode axis; a girdle's normal (kappa1)
            av = torch.rad2deg(torch.acos((ax_v * ax_true).sum(-1).abs().clamp(max=1)))
            row["vote_axis_deg"] = round(float(av[sel].median()), 1) if bool(sel.any()) else None
            row["n_eval"] = int(sel.sum())
            if ph.startswith("crossing"):                                                          # two peaks where both populations share
                both = (wk.min(1).values / m_true.clamp_min(1e-6) > 0.3) & (m_true > 0.15)
                if bool(both.any()):
                    nb = EI.dir_neighbours(dirs, 8)
                    fb = f[both]; ismax = (fb[:, :, None] >= fb[:, nb]).all(-1)                      # local maxima on the hemisphere grid
                    val = torch.where(ismax, fb, torch.zeros_like(fb)); top = val.topk(2, dim=1)
                    p1, p2 = dirs[top.indices[:, 0]], dirs[top.indices[:, 1]]
                    t1 = sym(tk[both, 0]); t2 = sym(tk[both, 1])
                    a1 = torch.linalg.eigh(t1)[1][..., 2]; a2 = torch.linalg.eigh(t2)[1][..., 2]
                    A = lambda u, v: torch.rad2deg(torch.acos((u * v).sum(-1).abs().clamp(max=1)))
                    err = torch.minimum(torch.maximum(A(p1, a1), A(p2, a2)), torch.maximum(A(p1, a2), A(p2, a1)))
                    two = (top.values[:, 1] > 0.3 * top.values[:, 0]) & (A(p1, p2) > 20)
                    row["two_peaks"] = round(float(two.float().mean()), 3); row["cross_err_deg"] = round(float(err[two].median()), 1) if bool(two.any()) else None
            rows.append(row); print(json.dumps(row), flush=True)
        figrows.append(fr)
json.dump(rows, open(out_stem + ".json", "w"), indent=1)
nc = 2 + len(models)
fig, axs = plt.subplots(len(figrows), nc, figsize=(2.6 * nc, 2.6 * len(figrows)), squeeze=False)
for i, fr in enumerate(figrows):
    axs[i, 0].imshow(fr["img"], cmap="gray", vmin=0, vmax=1); axs[i, 0].set_title(f"{fr['phantom']} ({'hard' if fr['hard'] is True else fr['hard'] or 'easy'})", fontsize=7)
    axs[i, 1].imshow(fr["truth"], cmap="magma", vmin=0, vmax=1); axs[i, 1].set_title("population region (truth)", fontsize=7)
    for j, (name, rp) in enumerate(fr["pred"]):
        axs[i, 2 + j].imshow(rp, cmap="magma", vmin=0, vmax=1)
        rr = [r_ for r_ in rows if r_["ckpt"] == name and r_["phantom"] == fr["phantom"] and r_["hard"] == fr["hard"]][0]
        axs[i, 2 + j].set_title(f"{name[:22]}\nreg {rr.get('region_corr')} ax {rr.get('axis_deg')}/{rr.get('vote_axis_deg')}", fontsize=6)
for a in axs.flat:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(out_stem + ".png", dpi=60); print("saved", out_stem + ".png")
