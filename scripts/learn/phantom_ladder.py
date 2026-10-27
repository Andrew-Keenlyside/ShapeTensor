"""The separation ladder on composable phantoms (:mod:`~shapetensor.learn.phantom2`): coherent and dispersed fibres
(straight, curved, fanning bundles, crossings of two or three), sheets, blobs (non-fibrous objects), empty volumes and
mixtures, in random 3-D orientations with a varied look. Trains on a mixture of families drawn per step and scores a
fixed held-out set (``--test-per-family`` per family) with the acceptance criteria (crossing_acceptance.evaluate_p2):
the figure (a row per family: the image on the slice through the structure; each population's target and model mass;
the isotropic objects' target and model mass; the model's unmatched oriented mass and its background) and the table.

Usage: python scripts/learn/phantom_ladder.py OUT.png [--init CKPT] [--steps 6000] [--families cross fan curve sheet blob mix empty]
       python scripts/learn/phantom_ladder.py OUT.png --plot-only
"""
import argparse, math, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, losses as L, entity_infer as EI, dataflow as DF, phantom2 as P2, crossing_acceptance as CA
from shapetensor.learn.model import PopulationNet

R0 = "/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/pilot_entities"
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine")
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--init", default=f"{R0}/simple/acc_angles_own_b3.pt")
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--lr-final", type=float, default=3e-6)
ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--families", nargs="+", default=[f for f in FAMS if f != "fine"])
ap.add_argument("--test-per-family", type=int, default=2); ap.add_argument("--n-points", type=int, default=512); ap.add_argument("--plot-only", action="store_true")
ap.add_argument("--bg-head", action="store_true"); ap.add_argument("--oprobe", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--floor", type=float, default=0.05); ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--sigma-lo", type=float, default=1.0, help="the smallest training query window (below 1: sub-voxel)")
ap.add_argument("--obj-head", action="store_true", help="an objects'-occupancy head (populations' mass predicted directly)")
ap.add_argument("--widths", type=int, nargs=3, default=None, help="the trunk's widths (a new model from scratch unless the init's match)")
ap.add_argument("--scratch", action="store_true", help="no weights from --init (its args only)")
ap.add_argument("--no-merge", action="store_true", help="readout: keep near-parallel same-type entities apart")
ap.add_argument("--sigmas", type=float, nargs="+", default=[2.5], help="the query windows to score at (each its own table)")
ap.add_argument("--odirect", type=int, default=0, help="the decoder's direct oriented path with this many bank directions (0: off; e.g. 48)")
ap.add_argument("--final", action="store_true", help="score the final test set (seeds never used to choose between variants) instead of the selection set")
ap.add_argument("--ckpt", default=None, help="with --plot-only: score this checkpoint (outputs named after OUT)")
ap.add_argument("--entities", default="lobes", choices=("votes", "lobes"), help="readout: entities from the votes or the density's lobes")
ap.add_argument("--axes", default="ring", choices=("votes", "density", "hybrid", "ring"), help="readout: entity frames/types from the votes or the density moment")
ap.add_argument("--saddle", type=float, default=0.8, help="readout: lobes joined across saddles this high (of the lower peak); <= 0: off")
ap.add_argument("--kappa", default="votes", choices=("votes", "moments", "votes_all"), help="readout: reported concentrations")
ap.add_argument("--mass", default="maps", choices=("maps", "density", "votes", "maps_votes"), help="readout: masses from the split head's maps (if any), the density, the mass votes, or votes' shares of the map")
ap.add_argument("--mass-vote", action="store_true", help="a per-direction vote for the owner's mass (the decoder)")
ap.add_argument("--split-head", action="store_true", help="a head for the objects' occupancy by kind (oriented / isotropic)")
ap.add_argument("--lobe-weight", type=float, default=None, help="entity_train.DENS_LOBE_WEIGHT (the density loss weighted toward lobes)")
ap.add_argument("--score-every", type=int, default=0, help="score the held-out set (criteria) every this many steps during training")
ap.add_argument("--block-norm", type=int, default=0, help="the trunk's residual blocks: 1 read per-voxel normalised features (pre-norm), 2 also add a normalised update")
ap.add_argument("--warmup", type=int, default=30, help="linear learning-rate warm-up steps")
ap.add_argument("--clip", type=float, default=5.0, help="gradient norm clip")
ap.add_argument("--h-max", type=float, default=50.0, help="divergence guard: the trunk's mean |feature| above this rolls back to the last snapshot at half the rate")
ap.add_argument("--resume", default=None, help="continue from this checkpoint (OUT_last.pt or a final .pt with its step; --steps is the total)")
ap.add_argument("--cpu-workers", type=int, default=0, help="make the batches (phantoms, targets) in this many CPU processes")
ap.add_argument("--prefetch", action="store_true", help="make the next step's phantoms and targets in a thread while the step trains")
ap.add_argument("--sample-oriented", action="store_true", help="entity_train.SAMPLE_ORIENTED: entity directions from the oriented entities only")
ap.add_argument("--lobe-mass", type=float, default=0.0, help="weight of the per-population lobe-mass loss (entity_train: W['lobe_mass'])")
ap.add_argument("--test-families", nargs="+", default=None, help="score only these families (short tests; default: all)")
ap.add_argument("--no-final", action="store_true", help="skip the final set at the end of training (short tests)")
ap.add_argument("--od-sigmas", type=float, nargs="+", default=None, help="the direct path's pooling levels (e.g. 2 4 6 8: a context level)")
a = ap.parse_args(); dev = "cuda"; stem = a.out.rsplit(".", 1)[0]
torch.backends.cudnn.benchmark = True                                                            # fixed training shapes (review 3)
if a.ckpt and not a.plot_only:
    ap.error("--ckpt only with --plot-only (training writes OUT's .pt)")
if a.test_families and not set(a.test_families) <= {"cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine"}:
    ap.error(f"--test-families: unknown {sorted(set(a.test_families) - {'cross', 'fan', 'curve', 'sheet', 'blob', 'mix', 'empty', 'fine'})}")
if a.od_sigmas and not a.odirect:
    ap.error("--od-sigmas needs --odirect")
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; TR.set_background_entity(True)
ET.VOTE_OBJECTS = True; ET.OWN_OBJECTS = True; ET.DENS_SAME_NORM = True
ET.SAMPLE_ORIENTED = bool(a.sample_oriented)
if a.lobe_weight is not None:
    ET.DENS_LOBE_WEIGHT = float(a.lobe_weight)
batch = lambda ts: {k: torch.stack([t[k] for t in ts]) for k in ts[0]}

def build_tests(S0, k):
    """Held-out phantoms: ``k`` per family from seeds ``S0`` (900000: the selection set; 1900000: the final set)."""
    out = []
    for fi, fam in enumerate(FAMS):
        if a.test_families and fam not in a.test_families:
            continue
        for i in range(k):
            spec, look = P2.random_spec(fam, torch.Generator().manual_seed(S0 + 100 * fi + i))
            out.append((fam, P2.make(spec, seed=S0 + 50_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1)))
    return out


tests = build_tests(1_900_000, max(a.test_per_family, 4)) if a.final else build_tests(900_000, a.test_per_family)

sph = EI.sphere_dirs(200, dev)


def slice_frame(ph):
    """The slice: through the centre, normal to the first population's plane normal (else z); offset between fibre layers."""
    n = ph["n"]; spec = ph["spec"]
    if spec and "normal" in spec[0]:
        nrm = torch.nn.functional.normalize(torch.tensor(spec[0]["normal"], dtype=torch.float32), dim=0)
        e1, e2 = P2._frame(nrm)
        if spec[0]["kind"] == "sheet":                                                             # sheets: the slice across the stack
            R = torch.stack([e1, nrm, e2], 1); off = 0.0
        else:
            R = torch.stack([nrm, e1, e2], 1); off = spec[0].get("spacing", 0.0) / 4
    else:
        R = torch.eye(3); off = 0.0
    return R.to(dev), off


def maps(ph, sigma=2.5):
    n, c = ph["n"], ph["n"] / 2.0; R, off = slice_frame(ph)
    gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
    loc = torch.stack([torch.full_like(gu, off), gu - c, gv - c], -1).view(-1, 3); pts = c + loc @ R.T
    rotated = not torch.allclose(R, torch.eye(3, device=dev))
    disc = ((((gu - c) ** 2 + (gv - c) ** 2) <= 28 ** 2) if rotated else torch.ones_like(gu, dtype=torch.bool)).cpu().numpy()
    sig = torch.full((n * n,), float(sigma), device=dev)
    t = P2.targets(ph, pts, sig); K = ph["occ"].shape[0]
    tgt = {"mass": t["e_mass"][:, :K].cpu().numpy(), "k": t["e_kappa"][:, :K].cpu().numpy(), "R": t["e_R"][:, :K].cpu().numpy(),
           "bg": t["e_mass"][:, K].cpu().numpy(), "kinds": ph["kinds"]}
    ro = EI.readout(net, ph["image"], pts, sig, sph, floor=a.floor if a.floor >= 0 else None, iso_k=ET.VOTE_KAPPA_MIN,
                    merge_deg=None if a.no_merge else 15.0, entities=a.entities, axes=a.axes, kappa=a.kappa, mass=a.mass,
                    saddle=a.saddle if a.saddle > 0 else None)
    mdl = {k: v.cpu().numpy() for k, v in ro.items()}
    face = torch.minimum(pts, (n - 1) - pts).amin(1).view(n, n).cpu().numpy()
    size = torch.tensor([n - 1.0] * 3, device=dev)
    smp = lambda q: torch.nn.functional.grid_sample(ph["image"][None, None], (q / size * 2 - 1).flip(-1).view(1, -1, 1, 1, 3), align_corners=True).view(n, n)
    img = torch.stack([smp(c + torch.stack([torch.full_like(gu, off + k), gu - c, gv - c], -1).view(-1, 3) @ R.T) for k in (-1, 0, 1)]).mean(0).cpu().numpy()
    return tgt, mdl, face, disc, img


def oracle(tgt):
    """The targets as a perfect model's output: oriented populations as entities, isotropic ones as isotropic-object mass."""
    iso = CA._types(tgt["k"][..., 0], tgt["k"][..., 1]) == 2
    return {"mass": np.where(iso, 0.0, tgt["mass"]), "k": tgt["k"], "R": tgt["R"], "iso": (tgt["mass"] * iso).sum(1), "bg": tgt["bg"], "bgm": tgt["bg"]}


def mass_budget(tgt, mdl, face, disc):
    """Where the model puts the populations' mass (over the scored points, as fractions of the target total): on the
    matched populations, in unmatched oriented entities ("other"), as isotropic objects where there are none
    ("iso spurious"), as isotropic objects in all (oriented phantoms: the populations read as isotropic), and the
    model's object total (oriented + isotropic) against the target's."""
    m = CA.match_p2(tgt, mdl); valid = (face.reshape(-1) >= CA.FACE_MARGIN) & disc.reshape(-1)
    T = tgt["mass"][valid]; tot = max(float(T.sum()), 1e-9)                                    # fractions of the target's own total (the
    D = max(float(T.sum()), 0.1 * int(valid.sum()))                                             # leakage criterion's normaliser D kept apart)
    out = {"target": tot, "matched": float(m["m"][valid].sum()) / tot, "other": float(m["other"][valid].sum()) / tot,
           "iso spurious": float(m["iso_spur"][valid].sum()) / tot, "iso": float(mdl["iso"][valid].sum()) / tot, "target sum": float(T.sum()), "D": D,
           "model total": float((mdl["mass"].sum(1) + mdl["iso"])[valid].sum()) / tot}
    for k in range(T.shape[1]):
        out[f"pop {k + 1}"] = float(m["m"][valid, k].sum()) / max(float(T[:, k].sum()), 1e-9)
    return out


def score_set(tests_, sigma, suffix, figure=True):
    """Every held-out phantom the ground truth passes, scored at ``sigma``: the table (OUT{suffix}_acceptance.png / .json)
    and, with ``figure``, the figure (a row per family)."""
    acc = []; rows = []; dropped = []; budget = []
    for fam, ph in tests_:
        tgt, mdl, face, disc, img = maps(ph, sigma); n = ph["n"]
        orc = CA.evaluate_p2(n, tgt, oracle(tgt), face, disc)
        lab = f"{fam} ({','.join(k[:5] for k in ph['kinds']) or 'none'})"
        if not orc["PASS"]:                                                                      # the truth fails the measurement: unscoreable
            dropped.append((lab, [k.split()[0] for k, v in orc.items() if k != "PASS" and not v[0]])); continue
        res = CA.evaluate_p2(n, tgt, mdl, face, disc); acc.append((lab, res)); budget.append((lab, mass_budget(tgt, mdl, face, disc)))
        if not any(r_[0] == fam for r_ in rows):
            rows.append((fam, ph, tgt, mdl, CA.match_p2(tgt, mdl), disc, img, res))
    npass = sum(r_["PASS"] for _, r_ in acc); st = curves[-1]["step"] if curves else 0
    print(f"[{suffix or 'selection'} sigma {sigma}] ground-truth check: {len(dropped)} of {len(tests_)} phantoms unscoreable: {dropped}", flush=True)
    if figure:
        cols = ["image", "target pop 1", "model pop 1", "target pop 2", "model pop 2", "model: other\ndirections", "model:\nisotropic objects", "target\nbackground", "model\nbackground"]
        norm = PowerNorm(0.5, vmin=0, vmax=1); cmap = matplotlib.colormaps["viridis"].copy(); cmap.set_over("red")
        fig, axs = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols) + 1.0, 2.4 * len(rows)), squeeze=False, layout="constrained")
        for i, (fam, ph, tgt, mdl, m, disc, img, res) in enumerate(rows):
            n = ph["n"]; K = tgt["mass"].shape[1]; sh = lambda v: np.where(disc, v.reshape(n, n), np.nan)
            axs[i, 0].imshow(np.where(disc, img, np.nan), cmap="gray", vmin=0, vmax=1)
            axs[i, 0].set_ylabel(f"{fam}: {', '.join(ph['kinds']) or 'empty'}\n{'PASS' if res['PASS'] else 'FAIL'}", fontsize=9, color="darkgreen" if res["PASS"] else "firebrick")
            panels = []
            for k in range(2):
                panels += [(sh(tgt["mass"][:, k]), f"sum {np.nansum(tgt['mass'][:, k] * disc.reshape(-1)):.0f}"), (sh(m["m"][:, k]), f"sum {np.nansum(m['m'][:, k] * disc.reshape(-1)):.0f}")] if k < K else [(None, ""), (None, "")]
            panels += [(sh(m["other"]), ""), (sh(mdl["iso"]), ""), (sh(tgt["bg"]), ""), (sh(mdl["bg"]), "")]
            for j, (v, lab) in enumerate(panels):
                ax_ = axs[i, 1 + j]
                if v is not None:
                    im = ax_.imshow(v, cmap=cmap, norm=norm); ax_.text(2, 61, lab, color="w", fontsize=7)
            for ax_ in axs[i]:
                ax_.set_xticks([]); ax_.set_yticks([])
        for j, t_ in enumerate(cols):
            axs[0, j].set_title(t_, fontsize=9)
        fig.colorbar(im, ax=axs[:, 1:].ravel().tolist(), shrink=0.5, label="mass per voxel (square-root scale)", extend="max")
        fig.suptitle(f"phantom families -- {a.tag}, {st} steps (from {a.init.split('/')[-1]}): {npass}/{len(acc)} held-out phantoms pass every criterion "
                     f"(rows: one per family; sigma {sigma}{', final set' if 'final' in suffix else ''}); background = {'background head' if args.get('bg_head') else 'the density floor'}", fontsize=10)
        plt.savefig(stem + suffix + ".png", dpi=70); print("saved", stem + suffix + ".png"); plt.close(fig)
    CA.table_figure(acc, stem + suffix + "_acceptance.png", f"phantom families -- {a.tag}, {st} steps, sigma {sigma}{' (final set)' if 'final' in suffix else ''}: "
                    f"{'PASS' if npass == len(acc) else 'FAIL'} ({npass}/{len(acc)})")
    CA.save_json(acc, stem + suffix + "_acceptance.json")
    import json
    with open(stem + suffix + "_budget.json", "w") as fh:
        json.dump(budget, fh, indent=1)
    for fam in FAMS:                                                                             # the mass budget per family (medians)
        b = [v for lab, v in budget if lab.startswith(fam + " ") and v["target sum"] > 1.0]
        if b:
            print(f"  budget {fam:6s}: " + " ".join(f"{k} {np.median([x[k] for x in b]):.2f}" for k in ("matched", "other", "iso spurious", "model total")), flush=True)
    print(f"acceptance [{suffix or 'selection'} sigma {sigma}]:", f"{npass}/{len(acc)} |",
          " ; ".join(f"{lab}: " + " ".join(k.split()[0] + ("+" if v[0] else "-") for k, v in r_.items() if k != "PASS") for lab, r_ in acc), flush=True)



if a.plot_only:
    ck = torch.load(a.ckpt or stem + ".pt", map_location=dev, weights_only=False)
    net = PopulationNet.from_args(ck["args"]).to(dev); net.load_state_dict(ck["model"]); curves, args, a.tag, a.init = ck["curves"], ck["args"], ck["tag"], ck["init"]
else:
    ref = torch.load(a.resume or a.init, map_location=dev, weights_only=False)                 # (resuming: the run's own args)
    args = {**ref["args"], "region_density": True, "kappa_log": True, "vote_objects": True, "own_objects": True, "dens_same_norm": True}
    args["bg_head"] = bool(args.get("bg_head")) or a.bg_head; args["oprobe"] = bool(args.get("oprobe")) or a.oprobe
    args["obj_head"] = bool(args.get("obj_head")) or a.obj_head
    args["split_head"] = bool(args.get("split_head")) or a.split_head
    args["mass_vote"] = bool(args.get("mass_vote")) or a.mass_vote
    args["block_norm"] = int(args.get("block_norm") or 0) or a.block_norm
    if a.odirect:
        args["odirect"] = int(a.odirect)
    if a.od_sigmas:
        args["od_sigmas"] = [float(x) for x in a.od_sigmas]
    if a.widths:
        args["widths"] = list(a.widths)
    net = PopulationNet.from_args(args).to(dev)
    if not a.scratch:
        DF.load_partial(net, ref["model"], lambda m: None)
        Wo = ref["model"].get("ent.od_net.0.weight"); Lo = None if Wo is None else Wo.shape[1] - 1
        if Lo is not None and getattr(net.ent, "odirect", 0) and net.ent.od_lv > Lo:              # more levels than the init: its weights on the
            with torch.no_grad():                                                                 # shared levels and on the mean (load_partial put
                w = torch.zeros_like(net.ent.od_net[0].weight); w[:, :Lo] = Wo[:, :Lo].to(w); w[:, -1] = Wo[:, -1].to(w)   # the mean's on the new level)
                net.ent.od_net[0].weight.copy_(w)
    if not ref["args"].get("kappa_log") and not a.scratch:
        net.ent.reset_kappa()
    a.tag = a.tag or f"families {' '.join(a.families)}"
    curves = []
    ge = torch.Generator(device=dev); held = []
    for i, (fam, ph) in enumerate(tests):
        ge.manual_seed(500 + i); T = batch([P2.targets(ph, *P2.draw_queries(ph, ge, a.n_points))]); held.append((fam, ph, T, ET.sample_dirs(T, 32, g=ge)))

    def save(step):
        torch.save({"model": net.state_dict(), "args": args, "curves": curves, "tag": a.tag, "init": a.init, "families": a.families, "step": step,
                    "opt": opt.state_dict(),
                    "flags": {k: getattr(a, k) for k in ("lr", "lr_final", "batch", "steps", "bg_head", "oprobe", "floor", "obj_head", "widths", "scratch",
                                                         "sigma_lo", "odirect", "split_head", "lobe_weight", "lobe_mass", "od_sigmas", "sample_oriented", "mass_vote", "warmup", "clip")}},
                   stem + (".pt" if step == a.steps else "_last.pt"))                               # (mid-run: OUT_last.pt, never the final name)

    def evaluate(step):
        net.eval(); r = {"step": step}
        for fam, ph, T, d in held:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                P = net(ph["image"][None, None], voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=d)
            tot, parts = ET.entity_loss(P, T, d, W={"desc": 0.0, "lobe_mass": a.lobe_mass}, g=torch.Generator(device=dev).manual_seed(97))
            r.setdefault(fam, []).append({"total": float(tot), **{k: float(v) for k, v in parts.items()}})
        net.train(); curves.append(r)
        print(f"  step {step}: " + " | ".join(f"{f} dens {np.mean([x['density'] for x in v]):.3f}" + (f" ori {np.mean([x['split_ori'] for x in v]):.3f}" if "split_ori" in v[0] else "")
                                              + (f" obj {np.mean([x['obj'] for x in v]):.3f}" if "obj" in v[0] else "")
                                              + (f" mv {np.mean([x['mvote'] for x in v if 'mvote' in x]):.3f}" if any("mvote" in x for x in v) else "") for f, v in r.items() if f != "step"), flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); g = torch.Generator(device=dev).manual_seed(0); gc = torch.Generator().manual_seed(0)
    gq = torch.Generator(device=dev).manual_seed(1) if a.prefetch else g                        # (the thread's own query draws)

    def make_batch(step):
        imgs, ts = [], []
        for b in range(a.batch):
            fam = a.families[int(torch.randint(0, len(a.families), (1,), generator=torch.Generator().manual_seed(7919 * step + b) if a.prefetch else gc))]
            spec, look = P2.random_spec(fam, torch.Generator().manual_seed(step * 31 + b))
            ph = P2.make(spec, seed=step * 31 + b, look=look, device=dev, ss=2 if fam == "fine" else 1)
            imgs.append(ph["image"]); ts.append(P2.targets(ph, *P2.draw_queries(ph, gq, a.n_points, sigma=(a.sigma_lo, 6.0))))
        return imgs, ts

    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(1) if a.prefetch else None; nxt = None
    procs, wdir = [], None
    start_step = int(torch.load(a.resume, map_location="cpu", weights_only=False).get("step", 0)) if a.resume else 0
    if a.cpu_workers:                                                                            # phantoms and targets made by CPU processes
        import subprocess, tempfile, atexit, shutil                                              # (~0.22 s of a 0.9 s step on the GPU), as files
        wdir = tempfile.mkdtemp(prefix="ladder_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        for w in range(a.cpu_workers):
            procs.append(subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "phantom_workers.py"), wdir,
                                           "--rank", str(w), "--n", str(a.cpu_workers), "--families", *a.families, "--batch", str(a.batch),
                                           "--n-points", str(a.n_points), "--sigma-lo", str(a.sigma_lo), "--start", str(start_step), "--stop", str(a.steps)], env=env))
        def _stop_workers():
            for pr in procs:
                pr.terminate()
            shutil.rmtree(wdir, ignore_errors=True)
        atexit.register(_stop_workers)

    def load_batch(step):
        f = os.path.join(wdir, f"{step:07d}.pt")
        while not os.path.exists(f):
            if any(pr.poll() not in (None, 0) for pr in procs):
                raise RuntimeError("a phantom worker died")
            time.sleep(0.02)
        d = torch.load(f, weights_only=False); os.remove(f)
        return [x.to(dev, non_blocking=True) for x in d["imgs"]], [{k: v.to(dev, non_blocking=True) for k, v in t_.items()} for t_ in d["ts"]]

    step0 = 0
    if a.resume:                                                                                 # continue a run from its checkpoint (weights,
        rk = torch.load(a.resume, map_location=dev, weights_only=False)                         # optimiser, curves; the schedule as if never
        net.load_state_dict(rk["model"]); curves[:] = rk.get("curves", [])                      # stopped)
        if "opt" in rk:
            opt.load_state_dict(rk["opt"])
        step0 = int(rk.get("step", 0)); print(f"resumed {a.resume} at step {step0}", flush=True)
    net.train(); t_last = time.time()
    import copy
    hmag = {}                                                                                    # the trunk's feature magnitudes (divergence guard)
    for i_, lvl in enumerate([net.l0, net.l1, net.l2] + ([net.l3] if getattr(net, "l3", None) is not None else []) + [net.mix]):
        lvl.register_forward_hook(lambda m, inp, out, i_=i_: hmag.__setitem__(i_, out.detach().float().abs().mean()))
    lr_scale = 1.0; snap = None; bad_run = 0; loss_ema = None; h_ref = None
    take = lambda st: (st, {k: v.detach().clone() for k, v in net.state_dict().items()}, copy.deepcopy(opt.state_dict()), h_ref)
    for step in range(step0, a.steps + 1):
        if step % 100 == 0 and step > step0:                                                     # progress (and speed) between evaluations
            print(f"    step {step}: {(time.time() - t_last) / 100:.2f} s/step, |h| {' '.join(f'{float(v):.3g}' for _, v in sorted(hmag.items()))}, "
                  f"loss ema {loss_ema if loss_ema is None else round(loss_ema, 3)}, lr x{lr_scale:g}", flush=True); t_last = time.time()
        if (step % a.eval_every == 0 and not (a.resume and step == step0)) or step == a.steps:
            evaluate(step)
            if 0 < step < a.steps and bad_run == 0:
                save(step)                                                                       # (a checkpoint at every evaluation)
                if a.score_every and step % a.score_every == 0:                                  # the criteria along the way: correctness
                    net.eval(); score_set(tests, 2.5, f"_s{step}", figure=False); net.train()    # at low training first
        if step == a.steps:
            break
        for pg in opt.param_groups:
            lr_ = a.lr * lr_scale * min(1.0, (step + 1) / a.warmup)
            pg["lr"] = a.lr_final + (lr_ - a.lr_final) * 0.5 * (1 + math.cos(math.pi * step / max(a.steps, 1))) if a.lr_final is not None else lr_
        if a.cpu_workers:                                                                        # the batch from the CPU workers (in order)
            imgs, ts = load_batch(step)
        elif a.prefetch:                                                                         # the next step's phantoms made in a thread
            imgs, ts = nxt.result() if nxt is not None else make_batch(step)                     # while this one trains (~a quarter of a step)
            nxt = pool.submit(make_batch, step + 1) if step + 1 < a.steps else None
        else:
            imgs, ts = make_batch(step)
        T = batch(ts); dirs = ET.sample_dirs(T, 32, g=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P = net(torch.stack(imgs)[:, None], voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
        tot, parts = ET.entity_loss(P, T, dirs, W={"desc": 0.0, "lobe_mass": a.lobe_mass}, g=g)
        hs = [float(v) for v in hmag.values()]; hm = max(hs) if hs else 0.0
        bad = (not torch.isfinite(tot)) or not all(math.isfinite(v) for v in hs) or hm > a.h_max or (h_ref is not None and hm > 10 * h_ref)
        if bad:                                                                                   # diverging: roll back, halve the rate (from-scratch
            bad_run += 1                                                                          # runs blew the trunk's features up to 1e16 by
            print(f"    step {step}: diverging (loss {float(tot):.3g}, |h| max {hm:.3g}, ref {h_ref}) -- "  # step 500-1000 at lr 3e-4: 28 Oct)
                  f"{'rolled back to step ' + str(snap[0]) if snap else 'no snapshot: skipped'}, lr x{lr_scale * 0.5:g}", flush=True)
            del P, tot, parts
            if snap is not None:
                net.load_state_dict(snap[1]); opt.load_state_dict(copy.deepcopy(snap[2])); h_ref = snap[3]   # (a copy: the optimiser's state
            lr_scale *= 0.5                                                                       # would otherwise alias the snapshot's)
            if lr_scale < 1e-3:
                raise RuntimeError("diverged repeatedly")
            continue
        bad_run = 0; loss_ema = float(tot) if loss_ema is None else 0.98 * loss_ema + 0.02 * float(tot)
        h_ref = hm if h_ref is None else 0.99 * h_ref + 0.01 * hm                                 # (the features' usual size, for a relative test)
        if snap is None or step - snap[0] >= 250:                                                 # a snapshot of weights this forward has checked
            snap = take(step)
        opt.zero_grad(set_to_none=True); tot.backward(); gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        if torch.isfinite(gn):                                                                    # (a non-finite gradient: the step skipped)
            opt.step()
    save(a.steps)
net.eval()
for si, sg in enumerate(a.sigmas):
    score_set(tests, sg, ("_final" if a.final else "") + ("" if sg == 2.5 else f"_s{sg}"), figure=(si == 0))
if not a.plot_only and not a.final and not a.no_final:                                           # the final set, once, at the end of training
    score_set(build_tests(1_900_000, 4), 2.5, "_final", figure=False)
