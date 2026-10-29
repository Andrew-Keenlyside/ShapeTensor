"""Score orientation densities by cross-entropy against the non-background material's ODF (synthetic crops only).

The target at a query: the Bingham mixture of the entities other than the background, on a fixed hemisphere grid,
normalised -- no visibility mixing. Scores are gains in log-likelihood over the uniform density, in nats per query
(sum_u p_t(u) log(4 pi q(u)) on the grid), for:
  uniform (0); a classical reader (the structure tensor's -g (u^T M u - 1/3) over a uniform floor w, with (g, w)
  fitted on the first half of the crops); the oracle (the target itself); the oracle blurred (kappa' = kappa ks /
  (kappa + ks), ks = 4, over a 10 % floor); and a model with the ``dens_ce`` head if given.
Reported on the second half of the crops, by query kind: visible / invisible line (a classical reader within / beyond
its angle of the true axis), crossing cores, no line; plus target sanity (line entities' kappa2, crossing lobes), the
model's calibration (its sharpness against its axis error) and a train/inference parity check.

    PYTHONPATH=src python scripts/eval_density_ce.py N [ckpt] (env STRENGTH, LEVEL, PROFILE, SEED0, MEMFRAC)
"""
import math
import os
import sys

import torch

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.5)))
import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
TR.ENTITY["p_cross"] = 0.2
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False; G.TARGETS_FROM_GEOMETRY = os.environ.get("GEOM", "1") == "1"   # targets from construction only
from shapetensor.learn import scene as SC, entity_infer as EI, lobe_train as LT, bingham as BG, entity_train as ET
from shapetensor.learn.model import PopulationNet, fib_hemisphere, st_products, st_matrix, window_mean, CE_GRID

N = int(sys.argv[1]); ckp = sys.argv[2] if len(sys.argv) > 2 else None; dev = "cuda" if torch.cuda.is_available() else "cpu"
STR = float(os.environ.get("STRENGTH", 1.0)); LEV = float(os.environ.get("LEVEL", 1.0)); seed0 = int(os.environ.get("SEED0", 160_000_000))
base = dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA)
if os.environ.get("PROFILE"):
    base["profile"] = os.environ["PROFILE"]
gen = LT.difficulty_gen(LEV, base)
net = None
if ckp:
    ck = torch.load(ckp, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
grid = fib_hemisphere(CE_GRID, dev); Gn = grid.shape[0]; L4 = math.log(4 * math.pi)
GAM = [0.0, 2, 4, 8, 12, 16, 24, 32, 48]; WS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
rec = {k: [] for k in ("half", "kind", "oracle", "blur", "cls", "model", "w0", "sharp", "aerr")}; k2 = []; core_ang = []; core_two = []; parity = None


def gain(pt, logq):                                                                              # nats over uniform, per query
    return (pt * (logq + L4)).sum(-1)


for i in range(N):
    s, d = SC.draw_scene((64,) * 3, seed=seed0 + 8 * i, device=dev, strength=STR, entity_ctx=True, **gen)
    t = TR.entity_queries(s, d, 64, i, dev); x = d["image"].float() / 255
    m = (t["e_mass"].float() * t["e_vis"].float()).clamp_min(0) * (t["e_node"] >= 0)
    mo = m * (~t["e_bg"].bool()).float(); m_o = mo.sum(1)
    kap = t["e_kappa"].float(); R = t["e_R"].float()
    pt = BG.pdf(grid.view(1, Gn, 1, 3), R[:, None], kap[:, None].clamp(max=ET.DENS_KAPPA_MAX)); pt = (mo[:, None] * pt).sum(-1)
    ok = (m_o > ET.MASS_EPS) & (pt.sum(-1) > 0); pt = pt / pt.sum(-1, keepdim=True).clamp_min(1e-30)
    line = (kap[..., 1] >= 2.0) & (mo > 0); lm = mo * line; lq = lm.sum(1) > 0.02; v = t["e_ovis"].float()
    srt, o = lm.sort(1, descending=True); ar = torch.arange(lm.shape[0], device=dev)
    ax1 = R[ar, o[:, 0]][:, :, 2]; ax2 = R[ar, o[:, 1]][:, :, 2]
    core = lq & (srt[:, 1] >= 0.5 * srt[:, 0]) & (srt[:, 1] > 0.02) & (torch.linalg.cross(ax1, ax2).norm(dim=-1) > 0.3)
    kind = torch.full_like(v, 3, dtype=torch.long); kind[lq & (v > 0.5)] = 0; kind[lq & (v < 0.2)] = 2; kind[core] = 1   # crossing last
    k2.append(kap[..., 1][line & (lm > 0.02)].cpu())
    if bool(core.any()):                                                                         # target sanity at crossing cores
        pk = grid[pt[core].argmax(1)]
        a1, a2 = ax1[core], ax2[core]
        core_ang.append(torch.rad2deg(torch.acos(torch.maximum((pk * a1).sum(1).abs(), (pk * a2).sum(1).abs()).clamp(max=1))).cpu())
        near2 = (grid[None] * a2[:, None]).sum(-1).abs().argmax(1)
        core_two.append((pt[core].gather(1, near2[:, None])[:, 0] / pt[core].amax(1)).cpu())
    pts, sig = t["e_points"].float(), t["e_sigma"].float()
    j6 = window_mean(st_products(x[None, None])[0], pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T
    qd = torch.einsum("gi,nij,gj->ng", grid, st_matrix(j6), grid)
    cls = []
    for g_ in GAM:
        l_ = -g_ * (qd - 1 / 3); l_ = l_ - torch.logsumexp(l_, 1, keepdim=True) + math.log(Gn) - L4
        cls.append(torch.stack([gain(pt, torch.logaddexp(torch.full_like(l_, math.log(w) - L4), math.log(1 - w) + l_)) for w in WS], 1))
    kb = kap * 4.0 / (kap + 4.0)
    pb = BG.pdf(grid.view(1, Gn, 1, 3), R[:, None], kb[:, None]); pb = (mo[:, None] * pb).sum(-1); pb = pb / pb.sum(-1, keepdim=True).clamp_min(1e-30)
    blur = gain(pt, torch.log(0.1 / Gn + 0.9 * pb) + math.log(Gn) - L4)
    sel = ok
    for k_, val in (("half", torch.full_like(v, float(i >= N // 2))), ("kind", kind.float()), ("oracle", gain(pt, torch.log((pt * Gn).clamp_min(1e-30)) - L4)),
                    ("blur", blur)):
        rec[k_].append(val[sel].cpu())
    rec["cls"].append(torch.stack(cls, 1)[sel].cpu())
    if net is not None:
        with torch.no_grad():
            f = EI.features(net, x); r = EI.query(net, f, pts, sig, grid, vol_shape=x.shape)
            lq_ = r["logq"].float()
            if parity is None:                                                                   # the training forward on the same crop
                dirs_ = grid[None, None, :8].expand(1, pts.shape[0], 8, 3).contiguous()
                P = net(x[None, None], voxel=torch.full((1,), float("nan"), device=dev), e_points=pts[None], e_sigma=sig[None], e_dirs=dirs_)
                parity = float((P["ent"]["logq_g"][0].float() - r["logq_g"].float()).abs().max())
        pk = grid[lq_.argmax(1)]
        rec["model"].append(gain(pt, lq_)[sel].cpu()); rec["w0"].append(r["w0"].float()[sel].cpu())
        rec["sharp"].append((lq_.amax(1) - torch.logsumexp(lq_, 1) + math.log(Gn))[sel].cpu())
        rec["aerr"].append(torch.where(lq, torch.rad2deg(torch.acos((pk * ax1).sum(1).abs().clamp(max=1))), torch.full_like(v, float("nan")))[sel].cpu())
    SC._scene_cache().clear()
    if dev == "cuda":
        torch.cuda.empty_cache()

C = {k: torch.cat(v_) for k, v_ in rec.items() if v_}
fit, test = C["half"] == 0, C["half"] == 1
gw = C["cls"][fit].mean(0); bi = int(gw.argmax()); bg, bw = GAM[bi // len(WS)], WS[bi % len(WS)]
C["classical"] = C["cls"][:, bi // len(WS), bi % len(WS)]
names = ["visible line", "crossing core", "invisible line", "no line"]
kd = C["kind"][test]
print(f"strength {STR} level {LEV}{' profile ' + os.environ['PROFILE'] if os.environ.get('PROFILE') else ''}: {N} crops ({int(test.sum())} test queries with non-background mass; "
      + ", ".join(f"{n} {int((kd == j).sum())}" for j, n in enumerate(names)) + ")")
print(f"classical reader fitted on the first {N // 2} crops: gain g = {bg}, floor w = {bw}")
print(f"target sanity: line entities' kappa2 median {float(torch.cat(k2).median()):.1f}; crossing cores {sum(len(a) for a in core_ang)}: target peak within "
      f"{float(torch.cat(core_ang).median()) if core_ang else float('nan'):.1f} deg (median) of an axis, second lobe / peak median "
      f"{float(torch.cat(core_two).median()) if core_two else float('nan'):.2f}")
rows = [("uniform", torch.zeros_like(C["oracle"])), ("classical", C["classical"])] + ([("model", C["model"])] if "model" in C else []) + [("oracle blurred", C["blur"]), ("oracle", C["oracle"])]
print(f"{'gain over uniform (nats)':26s} {'all':>7s} " + " ".join(f"{n:>15s}" for n in names))
for nm, g_ in rows:
    gt = g_[test]
    print(f"{nm:26s} {float(gt.mean()):7.4f} " + " ".join(f"{float(gt[kd == j].mean()) if (kd == j).any() else float('nan'):15.4f}" for j in range(4)))
if "model" in C:
    a = C["aerr"][test]; sh = C["sharp"][test]; okk = ~torch.isnan(a)
    rk = lambda z: torch.argsort(torch.argsort(z)).float()
    rho = float(torch.corrcoef(torch.stack([rk(sh[okk]), rk(-a[okk])]))[0, 1]) if int(okk.sum()) > 2 else float("nan")
    print(f"model: floor w0 mean {float(C['w0'][test].mean()):.3f} (visible {float(C['w0'][test][kd == 0].mean()):.3f}, invisible {float(C['w0'][test][kd == 2].mean()):.3f}); "
          f"axis error at line queries median {float(a[okk].median()):.1f} deg; calibration Spearman(sharpness, -axis error) {rho:.3f}; "
          f"train/inference parity max |d log q| {parity:.2e}")
