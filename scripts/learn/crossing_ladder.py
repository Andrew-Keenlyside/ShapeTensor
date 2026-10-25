"""The crossing ladder, one variable at a time on the simple phantom (:mod:`~shapetensor.learn.simple_phantom`: two
straight interleaved fibre bundles, one population entity each, a flat background). Trains a checkpoint on phantoms
drawn each step with the crossing angle in ``--train-angles`` (and, as later rungs add them, the other variables), then
draws one compact figure, a row per held-out phantom: the image; the target and the model's mass of A and of B (the
model's own entities, their mass split by their lobes -- :func:`~shapetensor.learn.entity_infer.split_mass` -- and
named A / B by the nearer of the two directions within 30 deg); the model's mass in other directions and isotropic;
and each population's mass along its own bundle (flat through the crossing when right). Every ``--eval-every`` steps
the loss on each held-out phantom is recorded; weights and curves are saved beside the figure (``OUT.pt``).

Usage: python scripts/learn/crossing_ladder.py OUT.png [--init CKPT] [--steps 1500] [--train-angles 30 90] [--test-angles 90 60 45 30]
       python scripts/learn/crossing_ladder.py OUT.png --plot-only
"""
import argparse, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.ndimage import map_coordinates
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, losses as L, entity_infer as EI, dataflow as DF, simple_phantom as SP
from shapetensor.learn.model import PopulationNet

R0 = "/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/pilot_entities"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--init", default=f"{R0}/simple/s90_ep14b_kfix.pt")
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--train-angles", type=float, nargs=2, default=[30.0, 90.0]); ap.add_argument("--test-angles", type=float, nargs="+", default=[90.0, 60.0, 45.0, 30.0])
ap.add_argument("--noise", type=float, default=0.01); ap.add_argument("--n-points", type=int, default=512); ap.add_argument("--plot-only", action="store_true")
ap.add_argument("--tag", default="")
a = ap.parse_args(); dev = "cuda"; stem = a.out.rsplit(".", 1)[0]
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; TR.set_background_entity(True); ET.VOTE_OBJECTS = True
batch = lambda t: {k: v[None] for k, v in t.items()}
tests = [SP.make(angle=ang, noise=a.noise, seed=1000 + i, device=dev) for i, ang in enumerate(a.test_angles)]

if a.plot_only:
    ck = torch.load(stem + ".pt", map_location=dev, weights_only=False)
    net = PopulationNet.from_args(ck["args"]).to(dev); net.load_state_dict(ck["model"]); curves, args, a.tag, a.init = ck["curves"], ck["args"], ck["tag"], ck["init"]
else:
    ref = torch.load(a.init, map_location=dev, weights_only=False)
    args = {**ref["args"], "region_density": True, "kappa_log": True, "vote_objects": True}
    net = PopulationNet.from_args(args).to(dev); DF.load_partial(net, ref["model"], lambda m: None)
    if not ref["args"].get("kappa_log"):
        net.ent.reset_kappa()
    a.tag = a.tag or f"angles {a.train_angles[0]:.0f}-{a.train_angles[1]:.0f}"
    curves = []; ge = torch.Generator(device=dev)
    held = []
    for i, ph in enumerate(tests):
        ge.manual_seed(500 + i); T = batch(SP.targets(ph, *SP.draw_queries(ph, ge, a.n_points))); held.append((ph, T, ET.sample_dirs(T, 32, g=ge)))

    def evaluate(step):
        net.eval(); r = {"step": step}
        for ph, T, d in held:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                P = net(ph["image"][None, None], voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=d)
            tot, parts = ET.entity_loss(P, T, d, W={"desc": 0.0}, g=torch.Generator(device=dev).manual_seed(97))
            r[f"{ph['angle']:.0f}"] = {"total": float(tot), **{k: float(v) for k, v in parts.items()}}
        net.train(); curves.append(r)
        print(f"  step {step}: " + " | ".join(f"{k} deg density {v['density']:.3f} mass {v['mass']:.3f}" for k, v in r.items() if k != "step"), flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); g = torch.Generator(device=dev).manual_seed(0); gc = torch.Generator().manual_seed(0)
    net.train()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
        if step == a.steps:
            break
        for pg in opt.param_groups:
            pg["lr"] = a.lr * min(1.0, (step + 1) / 30)
        ang = a.train_angles[0] + float(torch.rand(1, generator=gc)) * (a.train_angles[1] - a.train_angles[0])
        ph = SP.make(angle=ang, noise=a.noise, seed=step, device=dev)
        T = batch(SP.targets(ph, *SP.draw_queries(ph, g, a.n_points))); dirs = ET.sample_dirs(T, 32, g=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P = net(ph["image"][None, None], voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
        tot, parts = ET.entity_loss(P, T, dirs, W={"desc": 0.0}, g=g)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    torch.save({"model": net.state_dict(), "args": args, "curves": curves, "tag": a.tag, "train_angles": a.train_angles, "init": a.init}, stem + ".pt")
net.eval()
sph = EI.sphere_dirs(200, dev); C30 = math.cos(math.radians(30))


def maps(ph):
    """Target and model maps on the plane between an A and a B layer (sigma 2.5)."""
    n, c = ph["n"], ph["c"]; zq = c + ph["spacing"] / 4
    gu, gv = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    pts = torch.stack([torch.full((n * n,), zq), gu.flatten().float(), gv.flatten().float()], 1).to(dev); sig = torch.full((n * n,), 2.5, device=dev)
    t = SP.targets(ph, pts, sig)
    out = {"tA": t["e_mass"][:, 0], "tB": t["e_mass"][:, 1], "tI": t["e_mass"][:, 2]}
    with torch.no_grad():
        f = EI.features(net, ph["image"], voxel_um=None); r = EI.query(net, f, pts, sig, sph, vol_shape=ph["image"].shape)
        e = EI.cluster(r, sph); sp = EI.split_mass(r, sph, e)
    ms = sp["mass"]; iso = e["kappa"][..., 0].float() < 2.0; ax = e["R"][..., :, 2].float()
    cA = (ax @ ph["axes"]["A"]).abs(); cB = (ax @ ph["axes"]["B"]).abs()
    toA = ~iso & (cA >= C30) & (cA >= cB); toB = ~iso & (cB >= C30) & (cB > cA)
    out.update({"mA": (ms * toA).sum(1), "mB": (ms * toB).sum(1), "mO": (ms * (~iso & ~toA & ~toB)).sum(1), "mI": (ms * iso).sum(1) + sp["floor"]})
    out = {k: v.view(n, n).float().cpu().numpy() for k, v in out.items()}
    out["image"] = ph["image"][int(c) - 2:int(c) + 4].mean(0).cpu().numpy()
    return out


def along(m, ph, u):
    """A map along the line through the centre in the in-plane direction ``u`` (z, y, x): (t, values)."""
    t = np.arange(-ph["c"] + 1, ph["c"] - 1); c = ph["c"]
    return t, map_coordinates(m, [c + t * float(u[1]), c + t * float(u[2])], order=1, mode="nearest")


cols = ["image", "target A", "model A", "target B", "model B", "model: other\ndirections", "model: isotropic\n(background)", "mass along each bundle\n(flat = right)"]
norm = PowerNorm(0.5, vmin=0, vmax=1); cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_over("red")
fig, axs = plt.subplots(len(tests), len(cols), figsize=(2.5 * len(cols) + 1.0, 2.5 * len(tests)), squeeze=False, layout="constrained")
for i, ph in enumerate(tests):
    M = maps(ph)
    axs[i, 0].imshow(M["image"], cmap="gray", vmin=0, vmax=1); axs[i, 0].set_ylabel(f"{ph['angle']:.0f} deg crossing", fontsize=11)
    for j, k in enumerate(("tA", "mA", "tB", "mB", "mO", "mI")):
        im = axs[i, 1 + j].imshow(M[k], cmap=cmap, norm=norm); axs[i, 1 + j].text(2, 61, f"sum {M[k].sum():.0f}", color="w", fontsize=8)
    for j in range(len(cols) - 1):
        axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
    q = axs[i, -1]; th = math.radians(ph["angle"]); w = ph["half"] / max(math.sin(th), 1e-3)
    q.axvspan(-w, w, color="0.9", lw=0)
    t_, v = along(M["tA"], ph, ph["axes"]["A"]); q.plot(t_, v, color="k", lw=2, label="target")
    t_, v = along(M["mA"], ph, ph["axes"]["A"]); q.plot(t_, v, color="darkorange", lw=1.4, label="model A along A")
    t_, v = along(M["mB"], ph, ph["axes"]["B"]); q.plot(t_, v, color="teal", lw=1.4, ls="--", label="model B along B")
    q.set_ylim(0, 0.6); q.tick_params(labelsize=7)
    if i == 0:
        q.legend(fontsize=6, loc="upper right")
    if i == len(tests) - 1:
        q.set_xlabel("position along the bundle (grey: crossing)", fontsize=8)
for j, c_ in enumerate(cols):
    axs[0, j].set_title(c_, fontsize=10)
fig.colorbar(im, ax=axs[:, 1:-1].ravel().tolist(), shrink=0.5, label="mass per voxel (square-root scale)", extend="max")
st = curves[-1]["step"] if curves else 0
fig.suptitle(f"crossing ladder -- trained on {a.tag}, {st} steps (from {a.init.split('/')[-1]}); "
             f"held-out phantoms, noise {a.noise}; model mass split by lobe fit", fontsize=11)
plt.savefig(a.out, dpi=72); print("saved", a.out)
