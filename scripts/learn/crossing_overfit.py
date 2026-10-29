"""A small, fast check of a decoder variant on crossings: from a checkpoint, train a few hundred steps on a fixed set of
generator line crossings (two line-like object populations crossing at >= 45 deg; the current targets, no augmentation),
then draw, on two of those crops (can it fit crossings at all?) and on two unseen ones (does it carry over?), a slice
through the crossing in the plane both lines run in: the image with the two populations outlined; the target mass of
population A and of B; the model's mass along A's direction, along B's, in other directions (oriented entities matching
neither) and in isotropic (background) entities -- one colour scale for every mass panel. The model's entities are its
own (:func:`~shapetensor.learn.entity_infer.cluster`); the truth only names which direction is A's and which is B's at
each point (an entity within 30 deg of a population's local axis, nearer it than the other's, is that population's).
Every ``--eval-every`` steps the loss on all seen and unseen crops (fixed directions) is recorded; the trained weights and
the curves are saved beside the figure (``OUT.pt``).

Usage: python scripts/learn/crossing_overfit.py OUT.png INIT_CKPT --variant control|dirsh|ray [--steps 400] [--ntrain 8]
       python scripts/learn/crossing_overfit.py OUT.png --plot-only            (redraw from OUT.pt)
       python scripts/learn/crossing_overfit.py OUT.png --curves A.pt B.pt ... (the loss curves of several runs)
"""
import argparse, json, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
import shapetensor.learn.scene as SS
from shapetensor.learn import entity_train as ET, lobe_train as LT, losses as L, entity_infer as EI, dataflow as DF
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("init", nargs="?"); ap.add_argument("--variant", default="control")
ap.add_argument("--steps", type=int, default=400); ap.add_argument("--ntrain", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--eval-every", type=int, default=50); ap.add_argument("--plot-only", action="store_true"); ap.add_argument("--curves", nargs="+")
a = ap.parse_args(); dev = "cuda"
stem = a.out.rsplit(".", 1)[0]

if a.curves:                                                                                      # the loss curves of several runs
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, p in enumerate(a.curves):
        c = torch.load(p, map_location="cpu", weights_only=False)
        for j, part in enumerate(("total", "density")):
            st = [r["step"] for r in c["curves"]]
            axs[j].plot(st, [r["seen"][part] for r in c["curves"]], "-o", color=f"C{i}", ms=3, label=f"{c['variant']} seen")
            axs[j].plot(st, [r["unseen"][part] for r in c["curves"]], "--s", color=f"C{i}", ms=3, label=f"{c['variant']} unseen")
    for j, part in enumerate(("total loss", "density loss (the per-direction shape)")):
        axs[j].set_title(part, fontsize=11); axs[j].set_xlabel("training step"); axs[j].grid(alpha=0.3)
    axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(a.out, dpi=90); print("saved", a.out); sys.exit()

TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.ENTITY.update({"sigma": (1.0, 6.0), "p_cross": 0.2}); TR.READ_KEEP = 1.0
TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; TR.set_background_entity(True)
gen = {"scene": {**LT.difficulty_gen(0.4, {"canvas": 128, "crops_per_scene": 1, "cross_p": 1.0, "p_centre": 1.0, "deep_tree": True}), "crops_per_scene": 1}}
if a.plot_only:
    ck = torch.load(stem + ".pt", map_location=dev, weights_only=False); a.variant = ck["variant"]
    net = PopulationNet.from_args(ck["args"]).to(dev); net.load_state_dict(ck["model"]); curves = ck["curves"]; args = ck["args"]
else:
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    args = {**ck["args"], "region_density": True, "dir_sh": 6 if a.variant == "dirsh" else 0, "ray": a.variant == "ray"}
    net = PopulationNet.from_args(args).to(dev); DF.load_partial(net, ck["model"], lambda m: None); curves = []


def tensor33(t6):
    """``(..., 6)`` (xx, yy, zz, xy, xz, yz) -> ``(..., 3, 3)``."""
    T = torch.zeros(*t6.shape[:-1], 3, 3, device=t6.device, dtype=torch.float32); t6 = t6.float()
    for (i, j), c in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
        T[..., i, j] = t6[..., c]; T[..., j, i] = t6[..., c]
    return T


def principal(T):
    """The share of the largest eigenvalue (1: a line, 1/2: a plane's girdle, 1/3: isotropic) and its axis."""
    ev, V = torch.linalg.eigh(T)
    return ev[..., -1] / ev.sum(-1).clamp_min(1e-12), V[..., -1]


def line_crossing(s, d):
    """The two most-overlapping object populations if both run as lines through their overlap and cross at >= 45 deg:
    ``(A, B, slice axis, slice index, angle)`` -- the slice normal to the axis both lines lie most in, through the most
    overlap; else None."""
    keep = [k for k, L_ in enumerate(s.layers) if L_.role == "shape"]
    if len(keep) < 2:
        return None
    occ = d["occupancy"][keep].float()
    ov = (occ.unsqueeze(1) * occ.unsqueeze(0)).sum((2, 3, 4)) * (1 - torch.eye(len(keep), device=dev))
    i, j = divmod(int(ov.argmax()), len(keep))
    core = (occ[i] > 0.3) & (occ[j] > 0.3)
    if int(core.sum()) < 60:
        return None
    A, B = keep[i], keep[j]; ax = []
    for k in (A, B):
        lin, u = principal(tensor33(d["tensors"][k].float()[:, core].mean(1)))
        if float(lin) < 0.6:
            return None
        ax.append(u)
    ang = math.degrees(math.acos(min(1.0, abs(float(ax[0] @ ax[1])))))
    if ang < 45:
        return None
    sa = int(torch.minimum(1 - ax[0] ** 2, 1 - ax[1] ** 2).argmax())                           # the plane both lines lie most in
    prof = core.float().sum(tuple(k for k in range(3) if k != sa))
    return A, B, sa, int(prof.argmax()), ang


def entity_layers(s, d, gn):
    """The layer of each target entity (from its node in the scene's tree; -1: none)."""
    ctx = d["entity_ctx"]; tree = ctx["tree"]; gl = list(ctx["group_layer"])
    gl_ = [k for k in gl if k >= 0]; dense = [k for k, L_ in enumerate(s.layers) if L_.role not in TR.NOT_ENTITY and k not in gl_]
    gn = gn.long(); lay = torch.full_like(gn, -1)
    grp = tree.group.to(gn.device)[gn.clamp(0, tree.n - 1)]
    glt = torch.tensor(gl + [-1], device=gn.device)
    lay = torch.where((gn >= 0) & (gn < tree.n), glt[grp.clamp(-1, len(gl) - 1)], lay)
    if dense:
        lay = torch.where(gn >= tree.n, torch.tensor(dense, device=gn.device)[(gn - tree.n).clamp(0, len(dense) - 1)], lay)
    return lay


def target_maps(c):
    """The slice's grid, image, outlines, the target mass of A and of B, and each one's local axis (carried a few voxels
    past its edge, so the model's mass just outside still counts as along it)."""
    s, d = c["s"], c["d"]; A, B, sa, si, ang = c["lc"]
    oth = [k for k in range(3) if k != sa]
    gu, gv = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    pts = torch.zeros(64 * 64, 3); pts[:, sa] = float(si); pts[:, oth[0]] = gu.flatten().float(); pts[:, oth[1]] = gv.flatten().float()
    pts = pts.to(dev); sig = torch.full((pts.shape[0],), 2.5, device=dev)
    t = TR.entity_queries(s, d, 64, c["seed"], dev, points=pts, sigma=sig)
    lay = entity_layers(s, d, t["e_gnode"]); w = t["e_mass"].float() * t["e_vis"].float() * (t["e_node"] >= 0)
    Tt = tensor33(t["e_T6"])
    out = {"pts": pts, "sig": sig, "image": d["image"].float().select(sa, si).cpu() / 255.0, "angle": ang,
           "occA": d["occupancy"][A].float().select(sa, si).cpu(), "occB": d["occupancy"][B].float().select(sa, si).cpu(), "axes": {}}
    for nm, k in (("A", A), ("B", B)):
        wk = w * (lay == k)
        out["t" + nm] = wk.sum(1).view(64, 64).cpu()
        Tk = (wk[..., None, None] * Tt).sum(1).view(64, 64, 9).cpu().numpy()
        Tk = np.stack([gaussian_filter(Tk[..., q], 3.0) for q in range(9)], -1)
        wb = gaussian_filter(out["t" + nm].numpy(), 3.0)
        _, u = principal(torch.as_tensor(Tk).view(-1, 3, 3))
        out["axes"][nm] = (u.to(dev), torch.as_tensor(wb.reshape(-1) > 0.02, device=dev))
    iso_l = torch.tensor([L_.role == "isotropic" for L_ in s.layers] + [False], device=dev)[lay]      # (index -1: no layer)
    out["tAB"] = torch.minimum(out["tA"], out["tB"])
    out["tO"] = (w * ((lay >= 0) & (lay != A) & (lay != B) & ~iso_l)).sum(1).view(64, 64).cpu()    # the other populations
    out["tI"] = (w * iso_l).sum(1).view(64, 64).cpu()                                              # the background
    return out


def crossings(seed0, n):
    out, seed = [], seed0
    while len(out) < n and seed < seed0 + 3000:
        SS._CACHE.clear()
        s, d = SS.draw_scene((64,) * 3, seed=seed, device=dev, strength=0.3, entity_ctx=True, **gen["scene"])
        seed += 1
        lc = line_crossing(s, d)
        if lc is None:
            continue
        SS._CACHE.clear()
        img, t = TR._make_pair(seed - 1, 64, dev, 0.3, gen)
        dif = (img.reshape(-1).float() - d["image"].reshape(-1).float()).abs()
        if float(dif.max()) > 2:                                                                   # (GPU rounding: 1 level)
            print(f"  seed {seed - 1}: the training pair differs from the drawn scene (max {float(dif.max()):.0f}, "
                  f"{float((dif > 2).float().mean()):.4f} of voxels) -- skipped", flush=True)
            continue
        c = {"pair": (img, t), "seed": seed - 1, "s": s, "d": d, "lc": lc}
        c["tm"] = target_maps(c)
        if int(((c["tm"]["tA"] > 0.25) & (c["tm"]["tB"] > 0.25)).sum()) < 40:                     # both visible where they cross
            continue
        out.append(c)
    print(f"  {len(out)} visible line crossings from {seed - seed0} scenes", flush=True)
    return out


train = crossings(5_100_000, a.ntrain); test = crossings(5_200_000, 4)
gd = torch.Generator(device=dev)


def eval_loss(cases):
    net.eval(); acc = {}
    with torch.no_grad():
        for i, c in enumerate(cases):
            x, T = TR.stack([c["pair"]], aug=False)
            gd.manual_seed(1000 + i); dirs = ET.sample_dirs(T, 32, g=gd)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P = net(x, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
            gd.manual_seed(2000 + i); tot, parts = ET.entity_loss(P, T, dirs, g=gd)
            for k, v in {"total": tot, **parts}.items():
                acc.setdefault(k, []).append(float(v))
    net.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


if not a.plot_only:
    print(f"{a.variant}: {len(train)} training crossings, {len(test)} unseen", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); g = torch.Generator(device=dev).manual_seed(0); gc = torch.Generator().manual_seed(0)
    net.train()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            r_ = {"step": step, "seen": eval_loss(train), "unseen": eval_loss(test)}; curves.append(r_)
            print(f"  step {step}: seen total {r_['seen']['total']:.3f} density {r_['seen']['density']:.3f} | "
                  f"unseen total {r_['unseen']['total']:.3f} density {r_['unseen']['density']:.3f}", flush=True)
        if step == a.steps:
            break
        for pg in opt.param_groups:
            pg["lr"] = a.lr * min(1.0, (step + 1) / 30)
        idx = torch.randint(0, len(train), (4,), generator=gc).tolist()
        x, T = TR.stack([train[i]["pair"] for i in idx], g=gc, aug=False)
        dirs = ET.sample_dirs(T, 32, g=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P = net(x, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
        tot, parts = ET.entity_loss(P, T, dirs, g=g)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    torch.save({"model": net.state_dict(), "args": args, "variant": a.variant, "curves": curves, "steps": a.steps}, stem + ".pt")
net.eval()
dirs = EI.sphere_dirs(200, dev)
COS30 = math.cos(math.radians(30))


def model_maps(c):
    """The model's mass at the slice's grid: along A, along B, in other directions, in isotropic entities."""
    d, tm = c["d"], c["tm"]; out = {}
    with torch.no_grad():
        f = EI.features(net, d["image"].float() / 255.0, voxel_um=d.get("voxel_um"))
        r = EI.query(net, f, tm["pts"], tm["sig"], dirs, vol_shape=d["image"].shape)
        e = EI.cluster(r, dirs)
    m = (e["mass"] * e["live"]).float(); iso = e["kappa_odf"][..., 0].float() < 2.0
    ax = e["R_odf"][..., :, 2].float()
    cA = (ax * tm["axes"]["A"][0][:, None]).sum(-1).abs() * tm["axes"]["A"][1][:, None]
    cB = (ax * tm["axes"]["B"][0][:, None]).sum(-1).abs() * tm["axes"]["B"][1][:, None]
    toA = ~iso & (cA >= COS30) & (cA >= cB); toB = ~iso & (cB >= COS30) & ~toA
    for nm, sel in (("mA", toA), ("mB", toB), ("mO", ~iso & ~toA & ~toB), ("mI", iso)):
        out[nm] = (m * sel).sum(1).view(64, 64).cpu()
    out["mAB"] = torch.minimum(out["mA"], out["mB"])
    return out


rows = [("seen (trained on)", c) for c in train[:2]] + [("unseen", c) for c in test[:2]]
cols = [("image", None, None), ("population A", "tA", "mA"), ("population B", "tB", "mB"), ("A and B together\n(the crossing: min of the two)", "tAB", "mAB"),
        ("other directions\n(other populations)", "tO", "mO"), ("isotropic\n(background)", "tI", "mI")]
outl = {1: "A", 2: "B", 3: "AB", 4: "AB", 5: "AB"}
cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_over("red")
fig, axs = plt.subplots(2 * len(rows), len(cols), figsize=(2.55 * len(cols) + 1.2, 2.45 * 2 * len(rows)), squeeze=False, layout="constrained")
for i, (tag, c) in enumerate(rows):
    p = {**c["tm"], **model_maps(c)}
    for h, who in enumerate(("target", "model")):
        r_ = 2 * i + h
        for j, (title, kt, km) in enumerate(cols):
            ax_ = axs[r_, j]; ax_.set_xticks([]); ax_.set_yticks([])
            if j == 0:
                ax_.imshow(p["image"], cmap="gray")
                ax_.set_ylabel(f"{tag} {i % 2 + 1}: {who.upper()}\nseed {c['seed']}, A x B {p['angle']:.0f} deg", fontsize=9,
                               color="k" if who == "target" else "firebrick")
            else:
                im = ax_.imshow(p[kt if who == "target" else km], cmap=cmap, vmin=0, vmax=1)
                ax_.text(2, 61, f"sum {float(p[kt if who == 'target' else km].sum()):.0f}", color="w", fontsize=8)
            for nm, occ, col in (("A", p["occA"], "orange"), ("B", p["occB"], "cyan")):
                if j == 0 or nm in outl[j]:
                    ax_.contour(occ, levels=[max(0.1, 0.4 * float(occ.max()))], colors=col, linewidths=1.0 if j == 0 else 0.6)
            if r_ == 0:
                ax_.set_title(title, fontsize=10)
fig.suptitle(f"{a.variant}: after {curves[-1]['step'] if curves else 0} steps on {len(train)} crossings.  Each crossing: TARGET row, then MODEL row.  "
             f"Model mass along A / B = its entities within 30 deg of A's / B's local direction.\nOutlines: A orange, B cyan (40% of each one's peak occupancy "
             f"in the slice). Mass per voxel (sigma 2.5 window); red = above 1.", fontsize=11)
fig.colorbar(im, ax=axs[:, 1:].ravel().tolist(), shrink=0.4, label="mass", extend="max")
plt.savefig(a.out, dpi=72, bbox_inches="tight"); print("saved", a.out)
