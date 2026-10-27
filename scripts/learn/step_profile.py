"""Where a phantom-ladder training step's time goes: phantom generation, targets, the bank, the trunk, the decoder,
the loss, backward and the optimiser -- CUDA-synchronised wall times averaged over steps.

Usage: python scripts/learn/step_profile.py CKPT_FOR_ARGS [--steps 20] [--batch 2] [--split-head] [--od-sigmas 2 4 6 8]
"""
import argparse, sys, time, collections
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, phantom2 as P2
from shapetensor.learn.model import PopulationNet, with_oe, with_scal

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--steps", type=int, default=20); ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--split-head", action="store_true"); ap.add_argument("--od-sigmas", type=float, nargs="+", default=None); ap.add_argument("--n-points", type=int, default=512)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; ET.VOTE_OBJECTS = True; ET.OWN_OBJECTS = True; ET.DENS_SAME_NORM = True; ET.SAMPLE_ORIENTED = True
ref = torch.load(a.ckpt, map_location=dev, weights_only=False)
args = {**ref["args"], "split_head": a.split_head or ref["args"].get("split_head", False)}
if a.od_sigmas:
    args["od_sigmas"] = a.od_sigmas
net = PopulationNet.from_args(args).to(dev).train(); opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
FAM = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine"); T_ = collections.defaultdict(float)
g = torch.Generator(device=dev).manual_seed(0)


def tick(key, t0):
    torch.cuda.synchronize(); t = time.time(); T_[key] += t - t0; return t


for step in range(a.steps + 3):
    if step == 3:
        T_.clear()
    torch.cuda.synchronize(); t = time.time(); imgs, ts = [], []
    for b in range(a.batch):
        fam = FAM[(step * a.batch + b) % len(FAM)]
        spec, look = P2.random_spec(fam, torch.Generator().manual_seed(step * 31 + b))
        ph = P2.make(spec, seed=step * 31 + b, look=look, device=dev, ss=2 if fam == "fine" else 1); t = tick("phantoms", t)
        ts.append(P2.targets(ph, *P2.draw_queries(ph, g, a.n_points, sigma=(0.5, 6.0)))); imgs.append(ph["image"]); t = tick("targets", t)
    T = {k: torch.stack([x[k] for x in ts]) for k in ts[0]}; dirs = ET.sample_dirs(T, 32, g=g); X = torch.stack(imgs)[:, None]; t = tick("targets", t)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h0, h1, h = net.features(X); h = net.condition(h, T["voxel"]); t = tick("trunk forward", t)
        fe = (h0, h1, h)
        if getattr(net.ent, "odirect", 0):
            fe = with_oe(fe, net.direct_levels(X)); t = tick("bank", t)
        out = {"total_sh": net.total(h), "fraction_logit": net.frac(h)[:, 0]}
        if net.has_points:
            out["points"] = torch.nn.functional.softplus(net.points(h))
        out["ent"] = net.ent(fe, T["e_points"].float(), T["e_sigma"].float(), dirs, None); t = tick("decoder forward", t)
        if net.region is not None:
            from shapetensor.learn.model import window_mean
            out["region_logit"] = net.region_map(h0, h1, h); out["region"] = net.region_value(out["region_logit"])
            if getattr(net, "mass_region", False):
                out["ent"]["m"] = window_mean(out["region"], T["e_points"].float(), T["e_sigma"].float())
        if getattr(net, "bgh", None) is not None:
            out["bg"] = net.bg_map(h0, h1, h)
        if getattr(net, "objh", None) is not None:
            out["obj"] = net.obj_map(h0, h1, h)
        if getattr(net, "splith", None) is not None:
            out["split"] = net.split_map(h0, h1, h)
        t = tick("heads", t)
    tot, parts = ET.entity_loss(out, T, dirs, W={"desc": 0.0}, g=g); t = tick("loss", t)
    opt.zero_grad(set_to_none=True); tot.backward(); t = tick("backward", t)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); t = tick("optimiser", t)
tot_t = sum(T_.values()) / a.steps
print(f"step {tot_t * 1e3:.0f} ms (batch {a.batch}): " + ", ".join(f"{k} {v / a.steps * 1e3:.0f}" for k, v in sorted(T_.items(), key=lambda kv: -kv[1])))
print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
