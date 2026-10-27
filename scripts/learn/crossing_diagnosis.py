"""X0: why crossings fail in arbitrary orientation (plan_full_generator.md §5.1). Three checks on the composable phantoms'
crossings (:mod:`~shapetensor.learn.phantom2`: two interleaved fibre lattices, no overlap -- both directions are in
the image):

(b) orientation sweep -- one crossing rotated away from the voxel grid (0-45 deg about an oblique axis) and in random
    orientations, at 60 and 90 deg crossing angles: the share of each model's density at the crossing core within
    12 deg of each bundle's axis, against the target's share and the uniform share (does the failure follow the angle to
    the grid?);
(c) classical bound -- an oriented-filter line score and per-voxel Hessian line directions in the same cores
    (:mod:`~shapetensor.learn.inferability`): is each direction recoverable from local evidence alone, and what else
    does the core show (the score along the crossing plane's normal: the lattices stack into columns there)?;
(a) fit vs transfer -- a model fine-tuned on fixed sets of 1, 8 and 64 random-orientation crossings: the core shares on
    its own training crossings and on held-out ones (does it fail to fit, or fit and not transfer?).

Usage: python scripts/learn/crossing_diagnosis.py OUT_STEM [--models famN famW fam_C_final] [--fit-model famN] [--fit-steps 300]
"""
import argparse, copy, json, math, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, losses as L, entity_infer as EI, phantom2 as P2, bingham as BG, inferability as IF
from shapetensor.learn.model import PopulationNet

R0 = "/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/pilot_entities/simple"
ap = argparse.ArgumentParser(); ap.add_argument("out")
ap.add_argument("--models", nargs="+", default=["famN", "famW", "fam_C_final"])
ap.add_argument("--fit-model", default="famN"); ap.add_argument("--fit-steps", type=int, default=300); ap.add_argument("--fit-sets", type=int, nargs="+", default=[1, 8, 64])
ap.add_argument("--skip-fit", action="store_true"); ap.add_argument("--mc", action="store_true", help="Monte Carlo random-fibre crossings instead of the lattices")
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; TR.set_background_entity(True)
ET.VOTE_OBJECTS = True; ET.OWN_OBJECTS = True; ET.DENS_SAME_NORM = True
batch = lambda ts: {k: torch.stack([t[k] for t in ts]) for k in ts[0]}

SPH = EI.sphere_dirs(1000, dev)
CAP = 12.0
UNIFORM = 1 - math.cos(math.radians(CAP))                                        # the share of a uniform density within the cap
N = 64; C0 = N / 2.0


def rodrigues(w, deg):
    w = np.asarray(w, float); w = w / np.linalg.norm(w); t = math.radians(deg)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def cross_phantom(Q, ang, seed=0):
    """A two-bundle crossing (interleaved lattices) in the frame ``Q`` (columns: plane normal, bundle A, in-plane
    across), crossing angle ``ang`` degrees; returns the phantom and the two axes (z, y, x)."""
    nrm, e1, e2 = Q[:, 0], Q[:, 1], Q[:, 2]
    A = e1; B = math.cos(math.radians(ang)) * e1 + math.sin(math.radians(ang)) * e2
    if a.mc:                                                                     # Monte Carlo random fibres (interwoven, no lattice)
        mc = lambda ax: {"kind": "fibres", "axis": ax.tolist(), "normal": nrm.tolist(), "half": 12.0, "radius": 0.9, "fvf": 0.3, "wander": 0.5, "conc": 0.94}
        ph = P2.make([mc(A), mc(B)], seed=seed, look={"bg": 0.1, "contrast": [0.8, 0.8], "noise": 0.01}, device=dev)
        return ph, torch.tensor(np.stack([A, B]), dtype=torch.float32, device=dev)
    s0 = 4.0
    fib = lambda ax, z0: {"kind": "bundle", "normal": nrm.tolist(), "axis": ax.tolist(), "spacing": s0, "radius": 0.9, "half": 12.0, "conc": 0.94, "z0": z0}
    ph = P2.make([fib(A, 0.0), fib(B, s0 / 2)], seed=seed, look={"bg": 0.1, "contrast": [0.8, 0.8], "noise": 0.01}, device=dev)
    return ph, torch.tensor(np.stack([A, B]), dtype=torch.float32, device=dev)


def core_points(r=4.0):
    ax_ = torch.arange(-r, r + 1e-6, 1.0, device=dev)
    P = torch.stack(torch.meshgrid(ax_, ax_, ax_, indexing="ij"), -1).view(-1, 3)
    return P[P.norm(dim=1) <= r] + C0


def cap_shares(f, axes):
    """``f (Q, D)`` over SPH: the share of the density within CAP of each axis, mean over the points."""
    tot = f.clamp_min(0).sum(1).clamp_min(1e-12)
    c = (SPH @ axes.T).abs()                                                     # (D, 2)
    inside = c >= math.cos(math.radians(CAP))
    return [float(((f.clamp_min(0) * inside[:, k].float()).sum(1) / tot).mean()) for k in range(axes.shape[0])]


def shares(net, ph, axes):
    pts = core_points(); sig = torch.full((pts.shape[0],), 2.5, device=dev)
    t = P2.targets(ph, pts, sig)
    ft = BG.mixture_pdf(SPH, t["e_R"], t["e_kappa"], t["e_mass"], t["e_node"] >= 0)
    out = {"target": cap_shares(ft, axes)}
    if net is not None:
        with torch.no_grad():
            f = EI.features(net, ph["image"])
            r = EI.query(net, f, pts, sig, SPH, vol_shape=ph["image"].shape)
        out["model"] = cap_shares(r["f"].float(), axes)
    return out


def classical(ph, axes, r=6.0):
    """Two classical detectors in the core: the oriented-filter line score (Gaussian elongated 4 voxels along each
    direction: integrates along the fibres, robust to their deflections at an interwoven crossing) and the per-voxel
    Hessian at the fibre scale. Each one's two strongest peaks' errors; the line densities at A, B and the crossing plane's
    normal (x the median score / x uniform)."""
    zz = torch.stack(torch.meshgrid(*[torch.arange(N, device=dev, dtype=torch.float32)] * 3, indexing="ij"), -1)
    mask = (zz - C0).norm(dim=-1) <= r
    nrm = torch.nn.functional.normalize(torch.linalg.cross(axes[0], axes[1]), dim=0)
    sc = IF.orientation_scores(ph["image"], SPH, mask, sig_par=4.0, sig_perp=0.6); est, _ = IF.peak_axes(sc, SPH, k=2)
    at_s = lambda u: float(sc[(SPH @ u).abs().argmax()] / sc.median())
    d = IF.hessian_line_dirs(ph["image"], mask, SPH, sigma=0.8); est_h, _ = IF.peak_axes(d, SPH, k=2)
    at = lambda u: float(d[(SPH @ u).abs().argmax()]) * SPH.shape[0]
    return {"err": IF.axis_errors(est, axes), "score_A": at_s(axes[0]), "score_B": at_s(axes[1]), "score_normal": at_s(nrm),
            "err_hessian": IF.axis_errors(est_h, axes), "dens_A": at(axes[0]), "dens_B": at(axes[1]), "dens_normal": at(nrm)}


def load(name):
    ck = torch.load(name if name.endswith(".pt") else f"{R0}/{name}.pt", map_location=dev, weights_only=False)   # a name in R0, or a path
    net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); return net, ck


# ---- (b) orientation sweep and (c) the classical bound on the same phantoms
sweep_deg = [0, 5, 10, 15, 20, 30, 45]; w_axis = (1.0, 2.0, 3.0)
rng = np.random.default_rng(7)
rand_Q = []
for _ in range(6):
    q, _r = np.linalg.qr(rng.normal(size=(3, 3))); rand_Q.append(q * np.sign(np.diag(_r)))
cases = [(f"rot{d}", rodrigues(w_axis, d)) for d in sweep_deg] + [(f"rand{i}", q) for i, q in enumerate(rand_Q)]
nets = {m.rsplit('/', 1)[-1].replace('.pt', ''): load(m)[0] for m in a.models}
rows = []
for ang in (60, 90):
    for ci, (name, Q) in enumerate(cases):
        ph, axes = cross_phantom(Q, ang, seed=100 * ang + ci)                    # a fibre draw per case
        row = {"angle": ang, "case": name, "classical": classical(ph, axes)}
        row["target"] = shares(None, ph, axes)["target"]
        for m, net in nets.items():
            row[m] = shares(net, ph, axes)["model"]
        rows.append(row)
        print(f"{ang} {name:6s} target {np.round(row['target'], 3)} " + " ".join(f"{m} {np.round(row[m], 3)}" for m in nets) +
              f" | oriented filter err {np.round(row['classical']['err'], 1)} (score/median A {row['classical']['score_A']:.1f} B {row['classical']['score_B']:.1f} normal {row['classical']['score_normal']:.1f})"
              f" | Hessian err {np.round(row['classical']['err_hessian'], 1)}", flush=True)

# ---- (a) fit vs transfer
fit = {}
if not a.skip_fit:
    net0, ck0 = load(a.fit_model)
    def rand_cross(seed):
        g = np.random.default_rng(seed); q, _r = np.linalg.qr(g.normal(size=(3, 3))); q = q * np.sign(np.diag(_r))
        return cross_phantom(q, float(g.uniform(45, 90)), seed=seed)
    held = [rand_cross(5_000 + i) for i in range(8)]
    for K in a.fit_sets:
        train = [rand_cross(10_000 + i) for i in range(K)]
        net = copy.deepcopy(net0).train(); opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        g = torch.Generator(device=dev).manual_seed(0)
        def core_mean(phs):
            return float(np.mean([np.mean(shares(net, ph, ax)["model"]) for ph, ax in phs]))
        net.eval(); before = (core_mean(train[:8]), core_mean(held)); net.train()
        for step in range(a.fit_steps):
            ph, _ = train[step % K]
            T = batch([P2.targets(ph, *P2.draw_queries(ph, g, 512, sigma=(1.0, 6.0)))]); dirs = ET.sample_dirs(T, 32, g=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P = net(ph["image"][None, None], voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
            tot, parts = ET.entity_loss(P, T, dirs, W={"desc": 0.0}, g=g)
            opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval(); after = (core_mean(train[:8]), core_mean(held))
        tgt = float(np.mean([np.mean(shares(None, ph, ax)["target"]) for ph, ax in held]))
        fit[K] = {"train_before": before[0], "held_before": before[1], "train_after": after[0], "held_after": after[1], "target": tgt}
        print(f"fit K={K}: core share train {before[0]:.3f} -> {after[0]:.3f} | held-out {before[1]:.3f} -> {after[1]:.3f} | target {tgt:.3f} | uniform {UNIFORM:.3f}", flush=True)

json.dump({"rows": rows, "fit": fit, "uniform": UNIFORM, "cap_deg": CAP, "models": list(nets)}, open(a.out + ".json", "w"), indent=1)

# ---- figure
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
for j, ang in enumerate((60, 90)):
    ax_ = axs[j]; rs = [r for r in rows if r["angle"] == ang]; x = np.arange(len(rs))
    ax_.plot(x, [np.mean(r["target"]) for r in rs], "k-o", lw=2, label="target")
    for m in nets:
        ax_.plot(x, [np.mean(r[m]) for r in rs], "-o", ms=3, label=m)
    ax_.axhline(UNIFORM, color="grey", ls=":", label="uniform")
    ax2 = ax_.twinx(); ax2.bar(x, [max(r["classical"]["err"]) for r in rs], color="tab:red", alpha=0.18, width=0.6); ax2.set_ylabel("oriented-filter worse-axis error (deg)", color="tab:red", fontsize=8)
    ax2.set_ylim(0, 60)
    ax_.set_xticks(x); ax_.set_xticklabels([r["case"] for r in rs], rotation=60, fontsize=7)
    ax_.set_ylabel(f"core density share within {CAP:.0f} deg of an axis (mean of A, B)"); ax_.set_title(f"crossing {ang} deg: rotation from the grid / random", fontsize=10)
    ax_.legend(fontsize=7, loc="upper right"); ax_.grid(alpha=0.3)
if fit:
    Ks = sorted(fit); x = np.arange(len(Ks))
    for key, st in (("train_before", "o:"), ("train_after", "o-"), ("held_before", "s:"), ("held_after", "s-")):
        axs[2].plot(x, [fit[k][key] for k in Ks], st, label=key.replace("_", " "))
    axs[2].plot(x, [fit[k]["target"] for k in Ks], "k--", label="target (held-out)"); axs[2].axhline(UNIFORM, color="grey", ls=":")
    axs[2].set_xticks(x); axs[2].set_xticklabels([f"K={k}" for k in Ks]); axs[2].set_title(f"fit vs transfer ({a.fit_model}, {a.fit_steps} steps on K crossings)", fontsize=10)
    axs[2].legend(fontsize=7); axs[2].grid(alpha=0.3)
plt.savefig(a.out + ".png", dpi=80); print("saved", a.out + ".png")
