"""Score a populations-head model against the pilot acceptance criteria (synthetic crops and phantoms only).

Synthetic (held-out crops of the universal generator, ``STRENGTHS``): at query points with window ``WINDOW`` voxels,
the targets are each query's populations from construction (share of the window's material + elliptical ODF), matched
to the model's slots exactly as in training. Metrics:
  A  median axis error at single-line queries (one line population with share >= 0.6), against the fitted structure
     tensor's line axis at the same window;
  B  crossings (two line populations, each share >= 0.2, >= 45 deg apart): share with both axes found within 15 deg by
     slots with existence > 0.5 (the structure tensor reports one axis: 0 %);
  C  existence F1 for populations holding >= 10 % of the window (against every slot on), and the share error against
     the trivial answer "one population holds everything";
  D  shape class (line / sheet / isotropic, Westin) of matched populations with share >= 0.2;
  F  predicted existence rate against the targets', and the spread (sd) of predicted line-ness.
Phantoms (`shapetensor.phantoms`): bundle axis error, crossing both-found at 90/60/45 deg, and the fibre-free texture's
dominant line-ness (D).

    PYTHONPATH=src python scripts/eval_populations.py CKPT [N]   (env STRENGTHS=0.3,1.0 WINDOW=3 OUT=metrics.json)
"""
import json
import math
import os
import sys

import numpy as np
import torch

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.6)))
import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
TR.ENTITY["p_cross"] = 0.2
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False; G.TARGETS_FROM_GEOMETRY = True
from shapetensor import phantoms as PH
from shapetensor.learn import scene as SC, entity_infer as EI, lobe_train as LT, entity_train as ET
from shapetensor.learn.model import PopulationNet, st_products, st_matrix, window_mean

ckp = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 32; dev = "cuda"
STR = [float(v) for v in os.environ.get("STRENGTHS", "0.3,1.0").split(",")]; WIN = float(os.environ.get("WINDOW", 3.0))
ck = torch.load(ckp, map_location=dev, weights_only=False); net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
ET.configure_from_args(ck["args"])                                                                 # (its targets: background apart, matching)
dirs8 = EI.sphere_dirs(8, dev)


def westin(M):                                                                                     # (..., 3, 3) -> line, sheet, iso
    ev = torch.linalg.eigvalsh(M.double()).flip(-1).float().clamp_min(0); l1 = ev[..., 0].clamp_min(1e-9)
    return torch.stack([(ev[..., 0] - ev[..., 1]) / l1, (ev[..., 1] - ev[..., 2]) / l1, ev[..., 2] / l1], -1)


def axis(M):
    return torch.linalg.eigh(M.double())[1][..., :, -1].float()


def ang(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(-1).abs().clamp(max=1)))


def predict(x, pts, sig):
    with torch.no_grad():
        f = EI.features(net, x); r = EI.query(net, f, pts, sig, dirs8, vol_shape=x.shape)
    return {k: r[k].float() for k in ("pop_exist", "pop_share", "pop_T")}


def st_axis(x, pts, sig):
    j6 = window_mean(st_products(x[None, None])[0], pts[None].expand(6, -1, -1).contiguous(), sig[None].expand(6, -1).contiguous()).T
    return torch.linalg.eigh(st_matrix(j6).double())[1][..., :, 0].float()


res = {"checkpoint": ckp, "step": ck.get("step"), "window": WIN}
for S in STR:
    base = dict(canvas=128, crops_per_scene=8, cross_p=0.5, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA)
    gen = LT.difficulty_gen(0.4 + 0.6 * S, base)
    acc = {k: [] for k in ("ax_m", "ax_st", "cross", "tp", "fp", "fn", "sh_err", "sh_triv", "cls_ok", "ex_pred", "ex_true", "lin")}
    for i in range(N):
        try:
            s, d = SC.draw_scene((64,) * 3, seed=280_000_000 + 8 * i, device=dev, strength=S, entity_ctx=True, **gen)
        except torch.OutOfMemoryError:
            SC._scene_cache().clear(); torch.cuda.empty_cache(); continue
        t = TR.entity_queries(s, d, 64, i, dev); pts, sig = t["e_points"].float(), torch.full_like(t["e_sigma"].float(), WIN)
        t = TR.entity_queries(s, d, 64, i, dev, points=pts, sigma=sig)
        x = d["image"].float() / 255
        P = predict(x, pts, sig)
        T1 = {k: (v[None] if torch.is_tensor(v) and v.dim() >= 1 and k.startswith("e_") and k not in ("e_n", "e_anc") else v) for k, v in t.items()}
        mt = ET.population_match({k: v[None] for k, v in P.items()}, T1)
        mt = {k: (v[0] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in mt.items()}
        has = mt["has"]; ex = P["pop_exist"]; K = ex.shape[-1]
        s_t, v_t, E_t = mt["s_m"], mt["v_m"].bool(), mt["E_m"]                                  # per slot: its matched target
        on = ex > 0.5; big_t = v_t & (s_t >= 0.1); none_t = ~v_t                                 # (shares 0.02-0.1: not scored)
        hm = has[:, None]
        acc["tp"].append(int((on & big_t & hm).sum())); acc["fp"].append(int((on & none_t & hm).sum())); acc["fn"].append(int((~on & big_t & hm).sum()))
        acc.setdefault("fp_triv", []).append(int((none_t & hm).sum())); acc.setdefault("tp_triv", []).append(int((big_t & hm).sum()))
        acc["sh_err"].append(((P["pop_share"][:, :K] - s_t).abs().sum(-1) + (P["pop_share"][:, K] - mt["other"]).abs())[has].cpu())
        smax = mt["s"].max(-1).values
        acc["sh_triv"].append(((1 - smax) + (mt["s"].sum(-1) - smax) + mt["other"])[has].cpu())
        acc["ex_pred"].append(on[has].float().mean(-1).cpu()); acc["ex_true"].append(mt["valid"][has].float().mean(-1).cpu())
        wt = westin(E_t); wp = westin(P["pop_T"]); cls_t, cls_p = wt.argmax(-1), wp.argmax(-1)
        big = v_t & (s_t >= 0.2) & has[:, None]
        acc["cls_ok"].append((cls_t == cls_p)[big].float().cpu())
        acc["lin"].append(wp[..., 0][on & has[:, None]].cpu())
        line_t = v_t & (cls_t == 0)
        single = has & ((line_t & (s_t >= 0.6)).sum(-1) == 1)
        if bool(single.any()):
            k_ = (line_t & (s_t >= 0.6)).float().argmax(-1)
            q = torch.nonzero(single)[:, 0]
            at = axis(E_t[q, k_[q]]); am = axis(P["pop_T"][q, k_[q]])
            acc["ax_m"].append(ang(am, at).cpu()); acc["ax_st"].append(ang(st_axis(x, pts[q], sig[q]), at).cpu())
        lines = line_t & (s_t >= 0.2)
        for q in torch.nonzero(has & (lines.sum(-1) >= 2))[:, 0].tolist():
            ks = torch.nonzero(lines[q])[:, 0][:2]; a1, a2 = axis(E_t[q, ks[0]]), axis(E_t[q, ks[1]])
            if float(ang(a1, a2)) < 45:
                continue
            ok = all(bool(on[q, k]) and float(ang(axis(P["pop_T"][q, k]), axis(E_t[q, k]))) < 15 for k in ks.tolist())
            acc["cross"].append(float(ok))
        SC._scene_cache().clear(); torch.cuda.empty_cache()
    cat = lambda k: torch.cat(acc[k]) if acc[k] and torch.is_tensor(acc[k][0]) else torch.tensor(acc[k], dtype=torch.float32)
    tp, fp, fn = sum(acc["tp"]), sum(acc["fp"]), sum(acc["fn"]); f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    tpt, fpt = sum(acc["tp_triv"]), sum(acc["fp_triv"]); f1_triv = 2 * tpt / max(2 * tpt + fpt, 1)        # every slot on
    r = {"A_axis_model_deg": float(cat("ax_m").median()) if acc["ax_m"] else float("nan"),
         "A_axis_st_deg": float(cat("ax_st").median()) if acc["ax_st"] else float("nan"), "A_n": int(sum(len(a) for a in acc["ax_m"])),
         "B_cross_found": float(np.mean(acc["cross"])) if acc["cross"] else float("nan"), "B_n": len(acc["cross"]),
         "C_exist_f1": f1, "C_exist_f1_trivial": f1_triv, "C_share_err": float(cat("sh_err").mean()), "C_share_err_trivial": float(cat("sh_triv").mean()),
         "D_shape_acc": float(cat("cls_ok").mean()) if acc["cls_ok"] else float("nan"),
         "F_exist_rate_pred": float(cat("ex_pred").mean()), "F_exist_rate_true": float(cat("ex_true").mean()),
         "F_lineness_sd": float(cat("lin").std()) if acc["lin"] else float("nan")}
    res[f"synthetic_s{S}"] = r
    print(f"strength {S}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)

# phantoms
ph = {}
CASES = [("bundle", {"angle": 60.0}), ("crossing", {"cross_deg": 90.0}), ("crossing", {"cross_deg": 60.0}), ("crossing", {"cross_deg": 45.0}), ("texture", {})]
for name, kw in CASES:
    for tx in (0.5, 1.0):
        img, tr, meta = PH.build(name, seed=0, texture=tx, device=dev, **kw)
        img = torch.as_tensor(img, device=dev).float() / 255; Z, Y, X = img.shape; c0 = [(n - 96) // 2 for n in (Z, Y, X)]
        x = img[c0[0]:c0[0] + 96, c0[1]:c0[1] + 96, c0[2]:c0[2] + 96].contiguous()
        g_ = torch.arange(12, 85, 6, device=dev).float(); pts = torch.stack(torch.meshgrid(g_, g_, g_, indexing="ij"), -1).view(-1, 3)
        sig = torch.full((pts.shape[0],), WIN + 1.0, device=dev)
        zi = (pts[:, 0] + c0[0]).long(); xi = (pts[:, 2] + c0[2]).long()
        d1 = torch.as_tensor(np.asarray(tr["dir1"]), device=dev).float()[zi, xi]; d2 = torch.as_tensor(np.asarray(tr["dir2"]), device=dev).float()[zi, xi]
        npop = torch.as_tensor(np.asarray(tr["npop"]), device=dev)[zi, xi]
        P = predict(x, pts, sig); on = P["pop_exist"] > 0.5; ax_ = axis(P["pop_T"]); w = westin(P["pop_T"])
        sh = P["pop_share"][:, :-1] * on; dom = sh.argmax(-1); ar = torch.arange(pts.shape[0], device=dev)
        key = f"{name}{'_' + str(int(kw.get('cross_deg', kw.get('angle', 0)))) if kw else ''}_tex{tx}"
        if name == "bundle":
            sel = npop == 1
            ph[key] = {"axis_model_deg": float(ang(ax_[ar, dom][sel], d1[sel]).median()), "axis_st_deg": float(ang(st_axis(x, pts[sel], sig[sel]), d1[sel]).median())}
        elif name == "crossing":
            sel = npop == 2; ok = []
            for q in torch.nonzero(sel)[:, 0].tolist():
                a_on = ax_[q][on[q]]
                f1_ = bool((ang(a_on, d1[q][None]) < 15).any()) if len(a_on) else False
                f2_ = bool((ang(a_on, d2[q][None]) < 15).any()) if len(a_on) else False
                ok.append(f1_ and f2_)
            ph[key] = {"both_found": float(np.mean(ok)) if ok else float("nan")}
        else:
            ph[key] = {"dominant_lineness_below_0.4": float((w[ar, dom, 0] < 0.4).float().mean())}
        print(key, ph[key], flush=True)
res["phantoms"] = ph
out = os.environ.get("OUT")
if out:
    json.dump(res, open(out, "w"), indent=1); print("saved", out)
