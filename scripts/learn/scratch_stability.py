"""From-scratch training stability on the phantom families: the ladder's training loop for a few hundred steps, printing
every N steps the loss parts, the trunk's feature magnitudes (mean |h| per level) and the gradient norm before clipping
-- to find which change makes a from-scratch run diverge.

Usage: python scripts/learn/scratch_stability.py ARGS_CKPT --name NAME [--steps 300] [--every 25] [--batch 2] [--odirect 256]
       [--od-sigmas 2 4 6 8] [--split-head] [--mass-vote] [--sample-oriented] [--lr 3e-4] [--no-obj-head]
"""
import argparse, math, sys
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, phantom2 as P2
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--name", default="run"); ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--every", type=int, default=25); ap.add_argument("--batch", type=int, default=2); ap.add_argument("--odirect", type=int, default=0)
ap.add_argument("--od-sigmas", type=float, nargs="+", default=None); ap.add_argument("--split-head", action="store_true"); ap.add_argument("--mass-vote", action="store_true")
ap.add_argument("--sample-oriented", action="store_true"); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--no-obj-head", action="store_true")
ap.add_argument("--clip", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; ET.VOTE_OBJECTS = True; ET.OWN_OBJECTS = True; ET.DENS_SAME_NORM = True; ET.SAMPLE_ORIENTED = a.sample_oriented
torch.backends.cudnn.benchmark = True
ref = torch.load(a.ckpt, map_location=dev, weights_only=False)
args = {**ref["args"], "region_density": True, "kappa_log": True, "vote_objects": True, "own_objects": True, "dens_same_norm": True, "bg_head": True,
        "obj_head": not a.no_obj_head, "split_head": a.split_head, "mass_vote": a.mass_vote, "odirect": a.odirect}
if a.od_sigmas:
    args["od_sigmas"] = a.od_sigmas
torch.manual_seed(a.seed)
net = PopulationNet.from_args(args).to(dev).train()
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4); g = torch.Generator(device=dev).manual_seed(0)
FAM = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine"); feats = {}
for i, lvl in enumerate((net.l0, net.l1, net.l2)):
    lvl.register_forward_hook(lambda m, inp, out, i=i: feats.__setitem__(i, float(out.float().abs().mean())))
for step in range(a.steps + 1):
    for pg in opt.param_groups:
        lr_ = a.lr * min(1.0, (step + 1) / 30); pg["lr"] = 3e-6 + (lr_ - 3e-6) * 0.5 * (1 + math.cos(math.pi * step / 12000))
    imgs, ts = [], []
    for b in range(a.batch):
        fam = FAM[int(torch.randint(0, len(FAM), (1,), generator=torch.Generator().manual_seed(7919 * step + b + 1000 * a.seed)))]
        spec, look = P2.random_spec(fam, torch.Generator().manual_seed(step * 31 + b)); ph = P2.make(spec, seed=step * 31 + b, look=look, device=dev, ss=2 if fam == "fine" else 1)
        imgs.append(ph["image"]); ts.append(P2.targets(ph, *P2.draw_queries(ph, g, 512, sigma=(0.5, 6.0))))
    T = {k: torch.stack([t[k] for t in ts]) for k in ts[0]}; dirs = ET.sample_dirs(T, 32, g=g)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        P = net(torch.stack(imgs)[:, None], voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
    tot, parts = ET.entity_loss(P, T, dirs, W={"desc": 0.0}, g=g)
    opt.zero_grad(set_to_none=True); tot.backward(); gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)); opt.step()
    if step % a.every == 0:
        top = sorted(((float(p.grad.norm()), n) for n, p in net.named_parameters() if p.grad is not None), reverse=True)[:2]
        print(f"{a.name} step {step}: loss {float(tot):.3f} dens {float(parts['density']):.3f} | |h| {feats.get(0, 0):.3g} {feats.get(1, 0):.3g} {feats.get(2, 0):.3g} | "
              f"grad {gn:.3g} (top: {', '.join(f'{n} {v:.2g}' for v, n in top)}) | " + " ".join(f"{k} {float(v):.2f}" for k, v in parts.items() if k in ("mass", "region", "bg", "obj", "split_ori", "mvote", "vote", "kappa", "desc")), flush=True)
