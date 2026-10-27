"""Screen model changes on crossings by fit and transfer (plan_full_generator.md §5.1, X1-X2): each variant starts from a
checkpoint (new layers zero-initialised, so the start is the checkpoint), is fine-tuned on K fixed crossings in random
3-D orientations, and is scored by the share of its density within 12 deg of each bundle's axis at the crossing cores --
on its own training crossings (fit) and on held-out ones (transfer) -- against the target's share and a uniform density's.

Crossings are Monte Carlo random fibre bundles (:mod:`~shapetensor.learn.phantom2` kind ``fibres``: interwoven, both
directions recoverable from local evidence, X0 26 Oct); ``--lattice`` uses the earlier interleaved lattices.

Variants are comma-separated overrides: model args (``dir_sh=4``: the per-direction outputs' harmonics of the point
code, X2; ``oriented=24``: oriented line energies as input channels, X1), ``lr=...``, ``steps=...``, ``stem=rand`` (the new
input channels' stem weights small random instead of zero), or ``oracle=24``: the oriented channels filled with the
**true** line density along each direction (from the labels: each population's occupancy x (u^T T u)^8, centred over the
directions) -- the decoder test: if the network cannot fit crossings even when told the directions, the bottleneck is the
decoder / loss / optimisation, not the trunk finding features. ``odirect=48``: the decoder's direct oriented path (per
query direction, the image's line energy along it); ``doracle=48``: that path fed the true line energies. ``anneal=2``: the density target smoothed on the sphere,
from a kernel concentration of 2 (~35 deg) to 200 over the run (``entity_train.DENS_KAPPA_SMOOTH``). ``base`` is the
checkpoint as is.

Usage: python scripts/learn/crossing_fit_screen.py OUT_STEM [--init famN.pt] [--variants base dir_sh=4 oriented=24] [--K 8 64] [--steps 400]
"""
import argparse, copy, json, math, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, losses as L, entity_infer as EI, phantom2 as P2, bingham as BG, dataflow as DF
from shapetensor.learn.model import PopulationNet

R0 = "/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/pilot_entities/simple"
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--init", default=f"{R0}/famN.pt")
ap.add_argument("--variants", nargs="+", default=["base", "dir_sh=4", "oriented=24", "oriented=24,dir_sh=4"])
ap.add_argument("--K", type=int, nargs="+", default=[8, 64]); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--held", type=int, default=16); ap.add_argument("--lattice", action="store_true")
ap.add_argument("--save", action="store_true", help="save each trained variant (OUT_STEM_<variant>_K<K>.pt: model, args)")
ORACLE = {"ph": None}
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; TR.set_background_entity(True)
ET.VOTE_OBJECTS = True; ET.OWN_OBJECTS = True; ET.DENS_SAME_NORM = True
batch = lambda ts: {k: torch.stack([t[k] for t in ts]) for k in ts[0]}
SPH = EI.sphere_dirs(1000, dev); CAP = 12.0; UNIFORM = 1 - math.cos(math.radians(CAP)); N = 64; C0 = N / 2.0


def crossing(seed):
    """A crossing of two bundles at 45-90 deg in a random orientation; returns the phantom and the axes."""
    g = np.random.default_rng(seed)
    A = g.normal(size=3); A /= np.linalg.norm(A); q = np.cross(A, g.normal(size=3)); q /= np.linalg.norm(q)
    ang = math.radians(g.uniform(45, 90)); B = math.cos(ang) * A + math.sin(ang) * q; nrm = np.cross(A, B); nrm /= np.linalg.norm(nrm)
    look = {"bg": 0.1, "contrast": [0.8, 0.8], "noise": 0.01}
    if a.lattice:
        e2 = np.cross(nrm, A)
        fib = lambda ax, z0: {"kind": "bundle", "normal": nrm.tolist(), "axis": ax.tolist(), "spacing": 4.0, "radius": 0.9, "half": 12.0, "conc": 0.94, "z0": z0}
        spec = [fib(A, 0.0), fib(B, 2.0)]
    else:
        mc = lambda ax: {"kind": "fibres", "axis": ax.tolist(), "normal": nrm.tolist(), "half": float(g.uniform(9, 13)), "radius": float(g.uniform(0.7, 1.2)),
                         "fvf": float(g.uniform(0.2, 0.35)), "wander": float(g.uniform(0.2, 0.8)), "conc": 0.94}
        spec = [mc(A), mc(B)]
    return P2.make(spec, seed=seed, look=look, device=dev), torch.tensor(np.stack([A, B]), dtype=torch.float32, device=dev)


def oracle_energy(ph, dirs):
    """``(1, D, n, n, n)``: per direction u, the populations' occupancy x (u^T T u)^8 (a line along u counts, others fade),
    centred over the directions and scaled (the oriented channels' range) -- the directions the labels hold."""
    T = ph["tens"]; occ = ph["occ"]
    E = 0
    for k in range(occ.shape[0]):
        t = T[k]; q = (dirs[:, 0, None, None, None] ** 2 * t[0] + dirs[:, 1, None, None, None] ** 2 * t[1] + dirs[:, 2, None, None, None] ** 2 * t[2]
                       + 2 * (dirs[:, 0, None, None, None] * dirs[:, 1, None, None, None] * t[3] + dirs[:, 0, None, None, None] * dirs[:, 2, None, None, None] * t[4]
                              + dirs[:, 1, None, None, None] * dirs[:, 2, None, None, None] * t[5]))
        E = E + occ[k][None] * q.clamp(0, 1) ** 8
    return (4 * (E - E.mean(0, keepdim=True)))[None]


def oracle_levels(ph, dirs):
    """``(1, 3, D, n, n, n)`` the true line energies along ``dirs`` for the decoder's direct path: each population's occupancy
    x (u^T T u)^8, relative to the mean over the directions, pooled at sigma 2, 4, 6 (as ``PopulationNet.direct_levels``)."""
    from shapetensor.filters import gaussian
    E = oracle_energy(ph, dirs)[0] / 4                                                 # centred; back to (E - mean)
    T = ph["tens"]; occ = ph["occ"]; raw = 0
    for k in range(occ.shape[0]):
        t = T[k]; d0, d1, d2 = (dirs[:, i, None, None, None] for i in range(3))
        q = d0 ** 2 * t[0] + d1 ** 2 * t[1] + d2 ** 2 * t[2] + 2 * (d0 * d1 * t[3] + d0 * d2 * t[4] + d1 * d2 * t[5])
        raw = raw + occ[k][None] * q.clamp(0, 1) ** 8
    R = (raw + 0.02) / (raw.mean(0, keepdim=True) + 0.02)
    return torch.stack([gaussian(R, s) for s in (2.0, 4.0, 6.0)], 0)[None]


def core_points(r=4.0):
    ax_ = torch.arange(-r, r + 1e-6, 1.0, device=dev)
    P = torch.stack(torch.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).view(-1, 3)
    return P[P.norm(dim=1) <= r] + C0


def cap_shares(f, axes):
    tot = f.clamp_min(0).sum(1).clamp_min(1e-12); inside = (SPH @ axes.T).abs() >= math.cos(math.radians(CAP))
    return [float(((f.clamp_min(0) * inside[:, k].float()).sum(1) / tot).mean()) for k in range(axes.shape[0])]


PTS = None
def share(net, ph, axes, target=False):
    global PTS
    PTS = core_points() if PTS is None else PTS
    sig = torch.full((PTS.shape[0],), 2.5, device=dev)
    if target:
        t = P2.targets(ph, PTS, sig)
        return float(np.mean(cap_shares(BG.mixture_pdf(SPH, t["e_R"], t["e_kappa"], t["e_mass"], t["e_node"] >= 0), axes)))
    ORACLE["ph"] = ph
    with torch.no_grad():
        f = EI.features(net, ph["image"]); r = EI.query(net, f, PTS, sig, SPH, vol_shape=ph["image"].shape)
    return float(np.mean(cap_shares(r["f"].float(), axes)))


t0 = time.time()
held = [crossing(5_000 + i) for i in range(a.held)]
pool = [crossing(10_000 + i) for i in range(max(a.K))]
target = float(np.mean([share(None, ph, ax, True) for ph, ax in held]))
print(f"{len(pool) + len(held)} crossings drawn ({time.time() - t0:.0f}s); target core share {target:.3f}, uniform {UNIFORM:.3f}", flush=True)
ref = torch.load(a.init, map_location=dev, weights_only=False)
res = []
for var in a.variants:
    over = {} if var == "base" else dict(kv.split("=") for kv in var.split(","))
    lr = float(over.pop("lr", a.lr)); steps = int(over.pop("steps", a.steps)); stem = over.pop("stem", "zero")
    oracle = int(over.pop("oracle", 0)); anneal = float(over.pop("anneal", 0)); doracle = int(over.pop("doracle", 0))
    if doracle:
        over["odirect"] = doracle
    if oracle:
        over["oriented"] = oracle; stem = "rand"
    over = {k: int(v) for k, v in over.items()}
    for K in a.K:
        net = PopulationNet.from_args({**ref["args"], **over}).to(dev); DF.load_partial(net, ref["model"], lambda m: None)
        if stem == "rand" and getattr(net, "oriented", 0):
            with torch.no_grad():
                net.stem.weight[:, -net.oriented:].normal_(0, 0.05, generator=torch.Generator(device=dev).manual_seed(1))
        if doracle:                                                              # the true line energies into the decoder's direct path
            net.direct_levels = lambda x, net=net: oracle_levels(ORACLE["ph"], net.ent.od_dirs)
        if oracle:                                                               # the true directions as the oriented channels
            net.oriented_energy = lambda x, net=net: oracle_energy(ORACLE["ph"], net.o_dirs).to(x.dtype)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4); g = torch.Generator(device=dev).manual_seed(0)
        net.eval(); tr0 = np.mean([share(net, ph, ax) for ph, ax in pool[:8]]); he0 = np.mean([share(net, ph, ax) for ph, ax in held]); net.train()
        t = time.time(); losses = []
        for step in range(steps):
            ph, _ = pool[step % K]; ORACLE["ph"] = ph
            if anneal:                                                           # the density target from broad to sharp (ks: anneal -> 200)
                ET.DENS_KAPPA_SMOOTH = anneal * (200.0 / anneal) ** (step / max(steps - 1, 1))
            T = batch([P2.targets(ph, *P2.draw_queries(ph, g, 512, sigma=(1.0, 6.0)))]); dirs = ET.sample_dirs(T, 32, g=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P = net(ph["image"][None, None], voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
            tot, parts = ET.entity_loss(P, T, dirs, W={"desc": 0.0}, g=g)
            opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            losses.append(float(parts["density"]))
        ET.DENS_KAPPA_SMOOTH = None
        net.eval(); tr1 = np.mean([share(net, ph, ax) for ph, ax in pool[:min(8, K)]]); he1 = np.mean([share(net, ph, ax) for ph, ax in held])
        row = {"variant": var, "K": K, "lr": lr, "steps": steps, "train_before": round(float(tr0), 4), "train_after": round(float(tr1), 4), "held_before": round(float(he0), 4),
               "held_after": round(float(he1), 4), "dens_loss_first50": round(float(np.mean(losses[:50])), 4), "dens_loss_last50": round(float(np.mean(losses[-50:])), 4),
               "seconds": round(time.time() - t)}
        res.append(row); print(json.dumps(row), flush=True)
        if a.save:
            torch.save({"model": net.state_dict(), "args": {**ref["args"], **over}, "variant": var, "K": K, "steps": steps},
                       f"{a.out}_{var.replace(',', '_').replace('=', '')}_K{K}.pt")
json.dump({"rows": res, "target": target, "uniform": UNIFORM, "init": a.init, "lattice": a.lattice, "steps": a.steps}, open(a.out + ".json", "w"), indent=1)

fig, axs = plt.subplots(1, len(a.K), figsize=(5.5 * len(a.K), 4.2), squeeze=False, layout="constrained")
for j, K in enumerate(a.K):
    rs = [r for r in res if r["K"] == K]; x = np.arange(len(rs)); ax_ = axs[0, j]
    ax_.bar(x - 0.2, [r["train_after"] for r in rs], 0.4, label="train crossings (fit)"); ax_.bar(x + 0.2, [r["held_after"] for r in rs], 0.4, label="held-out (transfer)")
    ax_.axhline(target, color="k", ls="--", label="target"); ax_.axhline(UNIFORM, color="grey", ls=":", label="uniform")
    ax_.set_xticks(x); ax_.set_xticklabels([r["variant"] for r in rs], rotation=20, fontsize=8); ax_.set_title(f"K = {K} crossings", fontsize=10)
    ax_.set_ylabel(f"core density share within {CAP:.0f} deg of an axis"); ax_.legend(fontsize=7)
plt.savefig(a.out + ".png", dpi=80); print("saved", a.out + ".png")
