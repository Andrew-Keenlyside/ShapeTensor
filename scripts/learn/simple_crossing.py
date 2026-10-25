"""The simplest crossing: two straight bundles of parallel fibres crossing at ``--angle`` (A along x, B in the x-y plane),
each one population entity (nothing below it), their fibres in alternating z layers (interleaved, as crossing fibres
are), on a flat background with (almost) no noise; the rest of each voxel is the isotropic background entity. Fits a
checkpoint to this one volume (can the model represent a crossing at all?) and draws one fixed figure: a TARGET row and
a MODEL row -- the image (slab mean through the crossing), the mass of A, of B, of both together (min), in other
directions, isotropic; and the density over the in-plane directions at the crossing centre and on A's arm (the model's
raw density scaled to its mass, against the target's). The model's entities are its own
(:func:`~shapetensor.learn.entity_infer.cluster`); along A / B = within 30 deg of A's / B's direction.

Usage: python scripts/learn/simple_crossing.py OUT.png [--init CKPT | --scratch] [--steps 600] [--angle 90] [--noise 0.01]
       python scripts/learn/simple_crossing.py OUT.png --plot-only
       python scripts/learn/simple_crossing.py OUT.png --curves A.pt B.pt ...
"""
import argparse, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, losses as L, entity_infer as EI, dataflow as DF, bingham as BG
from shapetensor.learn.model import PopulationNet, window_mean

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--init"); ap.add_argument("--scratch", action="store_true")
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--angle", type=float, default=90.0); ap.add_argument("--noise", type=float, default=0.01); ap.add_argument("--n-points", type=int, default=512)
ap.add_argument("--plot-only", action="store_true"); ap.add_argument("--curves", nargs="+"); ap.add_argument("--tag", default="")
ap.add_argument("--kappa-log", action="store_true", help="the vote's concentrations in log space (model.EntityDecoder.kappa_log)")
ap.add_argument("--vote-objects", action="store_true", help="votes trained only where an oriented entity owns the direction (entity_train.VOTE_OBJECTS)")
a = ap.parse_args(); dev = "cuda"; stem = a.out.rsplit(".", 1)[0]
N, C, HALF, SPACING, RADIUS = 64, 32.0, 10.0, 4.0, 1.2

if a.curves:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i, p in enumerate(a.curves):
        c = torch.load(p, map_location="cpu", weights_only=False); st = [r["step"] for r in c["curves"]]
        for j, part in enumerate(("total", "density", "mass")):
            axs[j].plot(st, [r[part] for r in c["curves"]], "-o", ms=3, color=f"C{i}", label=c["tag"])
    for j, part in enumerate(("total loss", "density loss (per-direction shape)", "mass loss")):
        axs[j].set_title(part, fontsize=11); axs[j].set_xlabel("training step"); axs[j].grid(alpha=0.3); axs[j].set_yscale("log")
    axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(a.out, dpi=90); print("saved", a.out); sys.exit()

TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; TR.set_background_entity(True)

# ---- the phantom -----------------------------------------------------------------------------------------------
th = math.radians(a.angle)
AX = {"A": torch.tensor([0.0, 0.0, 1.0], device=dev), "B": torch.tensor([0.0, math.sin(th), math.cos(th)], device=dev)}   # (z, y, x)
zz, yy, xx = torch.meshgrid(*(torch.arange(N, dtype=torch.float32, device=dev),) * 3, indexing="ij")


def bundle(u, z0):
    """Occupancy of a bundle along unit ``u`` in the x-y plane: fibres on a ``SPACING`` grid across it (in-plane offset
    and z, z layers from ``z0``), within ``HALF`` of the centre line in both cross directions."""
    p = torch.stack([zz - C, yy - C, xx - C], -1)
    w = torch.stack([torch.zeros_like(u[0]), -u[2], u[1]])                                          # in-plane, across the bundle
    q1 = (p * w).sum(-1); q2 = p[..., 0]
    d1 = q1 - SPACING * torch.round(q1 / SPACING); d2 = q2 - z0 - SPACING * torch.round((q2 - z0) / SPACING)
    inside = (q1.abs() <= HALF) & (q2.abs() <= HALF)
    return ((RADIUS + 0.5 - torch.sqrt(d1 ** 2 + d2 ** 2)).clamp(0, 1) * inside).float()


occ = {"A": bundle(AX["A"], 0.0), "B": bundle(AX["B"], SPACING / 2)}                            # interleaved z layers
occ["bg"] = (1 - (occ["A"] + occ["B"]).clamp(max=1.0))
gn = torch.Generator(device=dev).manual_seed(0)
image = (0.1 + 0.8 * (occ["A"] + occ["B"]).clamp(max=1.0) + a.noise * torch.randn(N, N, N, generator=gn, device=dev)).clamp(0, 1)
image = (image * 255).round() / 255                                                                 # 8-bit, as the generator's
core = ((xx - C).abs() <= HALF) & ((yy - C).abs() <= HALF) & ((zz - C).abs() <= HALF)


def frame(u, conc):
    """``(T6, R, kappa)`` of a line along ``u`` with second moment ``conc`` on the axis (``None``: isotropic)."""
    T = torch.eye(3, device=dev) / 3 if u is None else conc * torch.outer(u, u) + (1 - conc) / 2 * (torch.eye(3, device=dev) - torch.outer(u, u))
    R, k = BG.fit(T.double()); T6 = torch.stack([T[0, 0], T[1, 1], T[2, 2], T[0, 1], T[0, 2], T[1, 2]])
    return T6.float(), R.float(), k.float()


ENT = [frame(AX["A"], 0.94), frame(AX["B"], 0.94), frame(None, 0)]                                 # nodes 1, 2, 3 under the root 0


def targets(pts, sig):
    """The entity targets at ``pts (Q, 3)``, ``sig (Q,)``: per query A, B, background (window means of their
    occupancy, all visible), padded to 8."""
    Q = pts.shape[0]; E = 8
    vol = torch.stack([occ["A"], occ["B"], occ["bg"]])
    m = window_mean(vol, pts[None].expand(3, -1, -1).contiguous(), sig[None].expand(3, -1).contiguous()).T   # (Q, 3)
    live = m > 1e-4
    node = torch.full((Q, E), -1, dtype=torch.int32, device=dev); node[:, :3] = torch.where(live, torch.arange(1, 4, device=dev, dtype=torch.int32), -1)
    mass = torch.zeros(Q, E, device=dev); mass[:, :3] = m * live
    T6 = torch.zeros(Q, E, 6, device=dev); R = torch.eye(3, device=dev).expand(Q, E, 3, 3).clone(); K = torch.zeros(Q, E, 2, device=dev)
    for j, (t6, r, k) in enumerate(ENT):
        T6[:, j], R[:, j], K[:, j] = t6, r, k
    anc = torch.full((1536, 16), -1, dtype=torch.int32, device=dev); anc[0, 0] = 0
    for i in (1, 2, 3):
        anc[i, 0], anc[i, 1] = i, 0
    return {"e_points": pts, "e_sigma": sig, "e_node": node.long(), "e_mass": mass, "e_vis": (node >= 0).float(), "e_T6": T6,
            "e_desc": torch.zeros(Q, E, 5, device=dev), "e_anc": anc.long(), "e_n": torch.tensor(4, device=dev), "e_R": R, "e_kappa": K,
            "e_gnode": node.long(), "region_t": occ["A"] + occ["B"] + occ["bg"], "iface_t": torch.zeros(N, N, N, device=dev),
            "voxel": torch.tensor(float("nan"), device=dev)}


def draw_queries(g, n):
    """Most on the fibres, a share in the crossing core, the rest anywhere (2 voxels from the faces); sigma 1-6."""
    pts = 2 + torch.rand(n, 3, generator=g, device=dev) * (N - 5)
    for frac, sel in ((0.5, (occ["A"] + occ["B"]) > 0.05), (0.25, core)):
        cand = torch.nonzero(sel).float(); k = int(frac * n); o = 0 if frac == 0.5 else int(0.5 * n)
        pts[o:o + k] = (cand[torch.randint(0, cand.shape[0], (k,), generator=g, device=dev)] + torch.rand(k, 3, generator=g, device=dev) - 0.5).clamp(2, N - 3)
    sig = torch.exp(math.log(1.0) + torch.rand(n, generator=g, device=dev) * math.log(6.0))
    return pts, sig


batch = lambda t: {k: v[None] for k, v in t.items()}
x = image[None, None]

# ---- the model ------------------------------------------------------------------------------------------------
ET.VOTE_OBJECTS = a.vote_objects
if a.plot_only:
    ck = torch.load(stem + ".pt", map_location=dev, weights_only=False)
    net = PopulationNet.from_args(ck["args"]).to(dev); net.load_state_dict(ck["model"]); curves, args, a.tag = ck["curves"], ck["args"], ck["tag"]
else:
    ref = torch.load(a.init or "/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/pilot_entities/eval_FINAL/EP14b_64_dec_iface_pl4.pt",
                     map_location=dev, weights_only=False)
    args = {**ref["args"], "region_density": True, "kappa_log": a.kappa_log, "vote_objects": a.vote_objects}
    net = PopulationNet.from_args(args).to(dev)
    if not a.scratch:
        DF.load_partial(net, ref["model"], lambda m: None)
        if a.kappa_log and not ref["args"].get("kappa_log"):
            net.ent.reset_kappa()
    curves = []; a.tag = a.tag or ("scratch" if a.scratch else "from EP14b")
    ge = torch.Generator(device=dev).manual_seed(99); Te = batch(targets(*draw_queries(ge, a.n_points)))
    ge.manual_seed(98); de = ET.sample_dirs(Te, 32, g=ge)

    def evaluate(step):
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            P = net(x, voxel=Te["voxel"], e_points=Te["e_points"], e_sigma=Te["e_sigma"], e_dirs=de)
        tot, parts = ET.entity_loss(P, Te, de, W={"desc": 0.0}, g=torch.Generator(device=dev).manual_seed(97))
        net.train(); r = {"step": step, "total": float(tot), **{k: float(v) for k, v in parts.items()}}; curves.append(r)
        print(f"  step {step}: " + " ".join(f"{k} {r[k]:.3f}" for k in ("total", "density", "mass", "vote", "region", "tree") if k in r), flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); g = torch.Generator(device=dev).manual_seed(0)
    net.train()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
        if step == a.steps:
            break
        for pg in opt.param_groups:
            pg["lr"] = a.lr * min(1.0, (step + 1) / 30)
        T = batch(targets(*draw_queries(g, a.n_points))); dirs = ET.sample_dirs(T, 32, g=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P = net(x, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
        tot, parts = ET.entity_loss(P, T, dirs, W={"desc": 0.0}, g=g)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    torch.save({"model": net.state_dict(), "args": args, "curves": curves, "tag": a.tag, "angle": a.angle, "noise": a.noise}, stem + ".pt")
net.eval()

# ---- the figure -----------------------------------------------------------------------------------------------
ZQ = C + SPACING / 4                                                                                # between an A layer and a B layer
gu, gv = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
pts = torch.stack([torch.full((N * N,), ZQ), gu.flatten().float(), gv.flatten().float()], 1).to(dev); sig = torch.full((N * N,), 2.5, device=dev)
t = targets(pts, sig); w = t["e_mass"]
tm = {"tA": w[:, 0], "tB": w[:, 1], "tO": torch.zeros_like(w[:, 0]), "tI": w[:, 2]}
sph = EI.sphere_dirs(200, dev)
with torch.no_grad():
    f = EI.features(net, image, voxel_um=None)
    r = EI.query(net, f, pts, sig, sph, vol_shape=image.shape)
    e = EI.cluster(r, sph)
    m = (e["mass"] * e["live"]).float(); iso = e["kappa_odf"][..., 0].float() < 2.0; axm = e["R_odf"][..., :, 2].float()
    cA = (axm @ AX["A"]).abs(); cB = (axm @ AX["B"]).abs(); c30 = math.cos(math.radians(30))
    toA = ~iso & (cA >= c30) & (cA >= cB); toB = ~iso & (cB >= c30) & ~toA
    mm = {"mA": (m * toA).sum(1), "mB": (m * toB).sum(1), "mO": (m * (~iso & ~toA & ~toB)).sum(1), "mI": (m * iso).sum(1)}
    sp = EI.split_mass(r, sph, e); ms = sp["mass"]                                                  # the lobe-fit split: the entities'
    isv = e["kappa"][..., 0].float() < 2.0; axv = e["R"][..., :, 2].float()                         # voted Binghams and a uniform floor
    vA = (axv @ AX["A"]).abs(); vB = (axv @ AX["B"]).abs()
    sA = ~isv & (vA >= c30) & (vA >= vB); sB = ~isv & (vB >= c30) & ~sA
    mm.update({"sA": (ms * sA).sum(1), "sB": (ms * sB).sum(1), "sO": (ms * (~isv & ~sA & ~sB)).sum(1), "sI": (ms * isv).sum(1) + sp["floor"]})
    # the density over in-plane directions at the centre and on A's arm: the model's raw density scaled to its mass
    phi = torch.linspace(0, 2 * math.pi, 181, device=dev)
    pl = torch.stack([torch.zeros_like(phi), torch.sin(phi), torch.cos(phi)], 1)
    probe = torch.tensor([[ZQ, C, C], [ZQ, C, 8.0]], device=dev); ps = torch.full((2,), 2.5, device=dev)
    rp = EI.query(net, f, probe, ps, pl, vol_shape=image.shape); rs = EI.query(net, f, probe, ps, sph, vol_shape=image.shape)
    tp = targets(probe, ps)
    rho = (tp["e_mass"][:, None] * BG.pdf(pl[None, :, None], tp["e_R"][:, None], tp["e_kappa"][:, None].clamp(max=ET.DENS_KAPPA_MAX))).sum(-1)
    mprobe = (e["mass"] * e["live"]).float().sum(1).view(N, N)
    centre_m = float(mprobe[int(C), int(C)]); arm_m = float(mprobe[int(C), 8])
    fpl = rp["f"].float() / rs["f"].float().mean(1, keepdim=True) * torch.tensor([centre_m, arm_m], device=dev)[:, None]
img_slab = image[int(C) - 2:int(C) + 4].mean(0).cpu()
maps = {k: v.view(N, N).cpu() for k, v in {**tm, **mm}.items()}
maps["tAB"] = torch.minimum(maps["tA"], maps["tB"]); maps["mAB"] = torch.minimum(maps["mA"], maps["mB"]); maps["sAB"] = torch.minimum(maps["sA"], maps["sB"])
cols = [("image (z slab)", None, None), ("population A\n(fibres along x)", "tA", "mA"), ("population B\n(fibres along y)", "tB", "mB"),
        ("A and B together\n(min of the two)", "tAB", "mAB"), ("other directions", "tO", "mO"), ("isotropic\n(background)", "tI", "mI")]
cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_over("red")
from matplotlib.colors import PowerNorm
norm = PowerNorm(0.5, vmin=0, vmax=1)                                                               # square root: 0.3 at the middle
ROWS = (("TARGET", "t", "k"), ("MODEL\nentities by direction", "m", "firebrick"), ("MODEL\nentities by lobe fit", "s", "darkblue"))
fig = plt.figure(figsize=(2.6 * 7 + 1.0, 2.8 * len(ROWS)), layout="constrained"); gs = fig.add_gridspec(len(ROWS), 7)
for h, (who, pre, colr) in enumerate(ROWS):
    for j, (title, kt, km) in enumerate(cols):
        ax_ = fig.add_subplot(gs[h, j]); ax_.set_xticks([]); ax_.set_yticks([])
        key = None if kt is None else pre + kt[1:]
        if j == 0:
            ax_.imshow(img_slab, cmap="gray", vmin=0, vmax=1)
            ax_.set_ylabel(who, fontsize=11, color=colr)
            if h == 0:
                ax_.plot([8], [C], "o", color="orange", ms=5); ax_.plot([C], [C], "o", color="red", ms=5)
        else:
            im = ax_.imshow(maps[key], cmap=cmap, norm=norm)
            ax_.text(2, 61, f"sum {float(maps[key].sum()):.0f}", color="w", fontsize=8)
        if h == 0:
            ax_.set_title(title, fontsize=10)
    ph = phi.cpu().numpy()
    if h == 2:                                                                                     # a population should not change through
        axq = fig.add_subplot(gs[h, 6]); xs = np.arange(N)                                         # the crossing: A along x, B along y
        axq.axvspan(C - HALF, C + HALF, color="0.9", lw=0)
        axq.plot(xs, maps["tA"][int(C)].numpy(), color="k", lw=2, label="target (A = B)")
        for key, col, nm in (("m", "firebrick", "by direction"), ("s", "darkblue", "by lobe fit")):
            axq.plot(xs, maps[key + "A"][int(C)].numpy(), color=col, lw=1.3, label=f"A, {nm}")
            axq.plot(xs, maps[key + "B"][:, int(C)].numpy(), color=col, lw=1.3, ls="--", label=f"B, {nm}")
        axq.set_ylim(0, max(1.0, float(maps["mA"].max()) * 1.05)); axq.set_xlabel("position along the bundle (grey: crossing)", fontsize=8)
        axq.set_title("each population's mass along\nits own bundle (should be flat)", fontsize=9); axq.legend(fontsize=6, loc="upper right"); axq.tick_params(labelsize=7)
    if h < 2:
        axp = fig.add_subplot(gs[h, 6], projection="polar")
        if h == 0:
            axp.plot(ph, rho[0].cpu().numpy(), color="red", lw=1.5, label="centre (crossing)"); axp.plot(ph, rho[1].cpu().numpy(), color="orange", lw=1.5, label="A's arm")
            axp.set_title("density over in-plane\ndirections (0 deg = x)", fontsize=10)
            axp.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.25, -0.25))
        else:
            axp.plot(ph, fpl[0].cpu().numpy(), color="red", lw=1.5); axp.plot(ph, fpl[1].cpu().numpy(), color="orange", lw=1.5)
            axp.set_title("model's raw density", fontsize=9)
        axp.set_yticklabels([]); axp.tick_params(labelsize=7)
fig.colorbar(im, ax=fig.axes, shrink=0.5, label="mass per voxel (square-root scale)", extend="max", location="right")
st = curves[-1]["step"] if curves else 0
with torch.no_grad():
    kv = EI.query(net, f, probe[1:], ps[1:], AX["A"][None], vol_shape=image.shape)["kappa"][0, 0].float()
kinfo = f"vote kappa on A's lobe (arm): {float(kv[0]):.1f} (target {float(ENT[0][2][0]):.1f})"
fig.suptitle(f"simple {a.angle:.0f} deg crossing, noise {a.noise} -- model {a.tag}, {st} steps on this volume.  Maps at the plane z = {ZQ:.0f} "
             f"(sigma 2.5); red dot: crossing centre, orange: A's arm; red = above the scale\n"
             f"kappa: {'log space' if args.get('kappa_log') else 'softplus x 10'}, votes {'objects only' if args.get('vote_objects') else 'every owner'} -- {kinfo}", fontsize=11)
plt.savefig(a.out, dpi=72); print("saved", a.out)
