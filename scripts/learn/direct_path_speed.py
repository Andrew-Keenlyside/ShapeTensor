"""The direct oriented path's speed fixes against the original code on held-out phantoms: (1) the bank sampled once per
point (not once per point and direction), (2) built in chunks of directions, (3) the smoothed levels kept at every 2nd
voxel (OD_STRIDES). Reports the largest differences in the decoder's density and the times (bank build, the direct
path's code for a query, a training forward + backward) before and after.

Usage: python scripts/learn/direct_path_speed.py CKPT [--n 3]
"""
import argparse, sys, time
import torch
import torch.nn.functional as Fn
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_infer as EI, entity_train as ET, phantom2 as P2, model as M
from shapetensor.learn.model import PopulationNet, with_oe
from shapetensor.learn.sh import real_sh

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--n", type=int, default=3); a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); dec = net.ent


def old_levels(x):                                                                                # the original direct_levels
    from shapetensor.filters import gaussian
    E = net.oriented_energy(x[:, :1], sig_par=8.0, sig_perp=0.6, pool=0.0, dirs=dec.od_dirs, relative=False).float()
    out = []
    for s in (2.0, 4.0, 6.0):
        Ep = gaussian(E, s); out.append(Ep / (Ep.mean(1, keepdim=True) + 1e-3 * Ep.mean() + 1e-12))
    return torch.stack(out, 1).half()


def old_od_feats(feats, pts, u, b=0, chunk=65536):                                                 # the original od_feats (per pair)
    E = feats.oe[b:b + 1].float(); L, Dd = E.shape[1], E.shape[2]; Ef = E.reshape(1, L * Dd, *E.shape[-3:])
    size = torch.tensor(E.shape[-3:], device=pts.device, dtype=torch.float32); out = []
    for a0 in range(0, pts.shape[0], chunk):
        p_, u_ = pts[a0:a0 + chunk].float(), Fn.normalize(u[a0:a0 + chunk].float(), dim=-1); n = p_.shape[0]
        g = (p_ / (size - 1).clamp_min(1) * 2 - 1).flip(-1).view(1, -1, 1, 1, 3)
        v = Fn.grid_sample(Ef, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, :, 0, 0].T.view(n, L, Dd)
        w = torch.softmax(dec.od_k * (u_ @ dec.od_dirs.T).abs(), -1); e = (v * w[:, None]).sum(-1)
        x = torch.cat([torch.log(e.clamp_min(1e-3)), torch.log(v.mean(-1).clamp_min(1e-3)).mean(-1, keepdim=True)], -1)
        out.append(Fn.silu(dec.od_net(x.to(dec.od_net[0].weight.dtype))))
    return torch.cat(out)


def timed(fn, reps=3):
    fn(); torch.cuda.synchronize(); t = time.time()
    for _ in range(reps):
        r = fn()
    torch.cuda.synchronize(); return r, (time.time() - t) / reps


sph = EI.sphere_dirs(200, dev); D = sph.shape[0]; Y = real_sh(sph, dec.lmax); res = {}
for fam in ("cross", "fine", "curve", "sheet"):
    for i in range(a.n):
        spec, look = P2.random_spec(fam, torch.Generator().manual_seed(900_000 + 100 * i + 7))
        ph = P2.make(spec, seed=950_000 + 100 * i + 7, look=look, device=dev, ss=2 if fam == "fine" else 1)
        x = ph["image"][None, None]; n = ph["n"]
        g = torch.Generator(device=dev).manual_seed(i); p = 4 + torch.rand(1024, 3, device=dev, generator=g) * (n - 9)
        s = torch.rand(1024, device=dev, generator=g) * 4 + 1; ctx = torch.minimum(p, n - 1 - p).amin(-1)
        with torch.no_grad():
            f0 = EI.features(net, ph["image"]); enc = dec.encode(f0, p[None], s[None], ctx[None])[0]
            fo = with_oe(f0, old_levels(x))
            od_old, t_old = timed(lambda: old_od_feats(fo, p[:, None].expand(-1, D, -1).reshape(-1, 3), sph.repeat(p.shape[0], 1)))
            F0 = dec.decode(enc[:, None].expand(-1, D, -1), Y[None].expand(p.shape[0], -1, -1).to(enc.dtype), None, None, od_old.view(-1, D, od_old.shape[-1]))["f"].float()
            tb_old = timed(lambda: old_levels(x))[1]
            for st in ((1, 1, 1), (1, 2, 2), (2, 2, 2), (2, 4, 4)):
                fn = with_oe(f0, net.direct_levels(x, strides=st)); tb = timed(lambda: net.direct_levels(x, strides=st))[1]
                od, t = timed(lambda: dec.od_code(dec.od_sample(fn, p), dec.od_weights(sph)))
                F1 = dec.decode(enc[:, None].expand(-1, D, -1), Y[None].expand(p.shape[0], -1, -1).to(enc.dtype), None, None, od)["f"].float()
                d = (F1 - F0).abs() / F0.mean(1, keepdim=True).clamp_min(1e-9); agree = float((F1.argmax(1) == F0.argmax(1)).float().mean())
                res.setdefault(st, []).append((float(d.max()), float(d.mean()), agree, t / t_old, tb / tb_old))
                print(f"{fam} {i} n {n} strides {st}: density diff / mean max {float(d.max()):.4f} mean {float(d.mean()):.5f}, same peak {agree:.3f}; "
                      f"code {t * 1e3:.1f} vs {t_old * 1e3:.1f} ms; bank {tb * 1e3:.0f} vs {tb_old * 1e3:.0f} ms", flush=True)
for st, v in res.items():
    print(f"strides {st}: worst diff {max(x[0] for x in v):.4f}, mean diff {sum(x[1] for x in v) / len(v):.5f}, same peak min {min(x[2] for x in v):.3f}, "
          f"code time x{sum(x[3] for x in v) / len(v):.3f}, bank time x{sum(x[4] for x in v) / len(v):.2f}")

net.train(); imgs, ts = [], []; g = torch.Generator(device=dev).manual_seed(0)                    # a training step (the ladder's batch)
for b in range(2):
    spec, look = P2.random_spec("cross", torch.Generator().manual_seed(b)); ph = P2.make(spec, seed=b, look=look, device=dev)
    imgs.append(ph["image"]); ts.append(P2.targets(ph, *P2.draw_queries(ph, g, 512, sigma=(1.0, 6.0))))
T = {k: torch.stack([t[k] for t in ts]) for k in ts[0]}; dirs = ET.sample_dirs(T, 32, g=g); X = torch.stack(imgs)[:, None]


def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        P = net(X, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
    tot, _ = ET.entity_loss(P, T, dirs, W={"desc": 0.0}, g=torch.Generator(device=dev).manual_seed(1))
    tot.backward(); net.zero_grad(set_to_none=True)


for label in ("old bank build", "new"):
    if label != "new":
        net.direct_levels = old_levels
    torch.cuda.reset_peak_memory_stats(); _, t = timed(step)
    print(f"training step ({label}): {t * 1e3:.0f} ms, peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    if label != "new":
        del net.direct_levels
with torch.no_grad():                                                                             # the old per-pair code in training
    fe = with_oe(net.features(X)[:3], old_levels(X)); Q = T["e_points"].shape[1]; Dq = dirs.shape[2]
    _, t_pair = timed(lambda: old_od_feats(fe, T["e_points"][0][:, None].expand(-1, Dq, -1).reshape(-1, 3).float(), dirs[0].reshape(-1, 3)))
    _, t_pt = timed(lambda: dec.od_code(dec.od_sample(fe, T["e_points"][0].float()), dec.od_weights(dirs[0])))
print(f"training direct-path code per item: per pair {t_pair * 1e3:.1f} ms, per point {t_pt * 1e3:.1f} ms")
