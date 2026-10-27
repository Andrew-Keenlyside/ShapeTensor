"""Readout variants at crossings of controlled angle -- can the readout alone separate narrower crossings?

A tuning set of its own (seeds from ``--seed0``, 3.1M: not the selection set's 0.9M nor the final set's 1.9M): the
generator's Monte Carlo crossings ("cross": resolved fibres; "fine": below the grid) with the second bundle turned to
each angle of ``--angles`` (the third population, when drawn, dropped), their first bundle alone (single), and the
generator's fans, curves and sheets (one population each: what a readout that splits more must not split).

1. **Profile** (``OUT.profile.json``): at crossing cores the decoder queried along the great circle through the two
   true axes (1-deg steps), against the target's density there: two maxima near the two axes or one, the valley's
   depth against the lower peak, the maxima's separation against the true angle. Whether the information is in the
   density at all decides what a readout can do.
2. **Variants** (``OUT.json``): :func:`entity_infer.entities_from_query` options scored on the cached decoder outputs:
   at cores (two line populations, the lighter >= half the heavier) both bundles found -- the two heaviest line
   populations each within 20 deg of a true axis, one-to-one ("found", as disco_test.py) and also each nearer its own
   axis than the other with the two at least half the true angle apart ("resolved": at a 20-deg crossing a blended
   pair is within 20 deg of both axes) -- by the per-point crossing angle, with the axes raw and refined from the image
   (:func:`axis_refine.refine_axes`); where the truth has one population: a second line population (>= ``--count-min``,
   and >= half the first), on single bundles, fans and curves; sheets: the sheet share of the oriented mass.

Selection rule (fixed before the full grid ran, 27 Oct): the variant with the highest "resolved" share at cores (raw
axes -- the refinement drifts on thin interleaved fibres, 4 -> 10-17 deg on a 60-deg cross, so it is reported, not
selected on), the mean over the four angle bins and both windows, subject to, against the default: a second line
population (>= half the first) on single bundles at most 2 points more, on fans and on curves at most 3 more each, the
sheets' sheet share at most 0.05 less; within 1 point, the variant with fewer changed options. Then guarded on the
selection set's criteria (readout_lab.py) and checked once on the final set (synth_vs_truth.py) and DiSCo1.

Usage: python scripts/learn/readout_xing.py OUT_STEM CKPT [--variants 'saddle=0.9,lobe_merge_deg=15' ...]
"""
import argparse, json, math, sys
import numpy as np
import torch
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_infer as EI, entity_train as ET, phantom2 as P2, bingham as BG
from shapetensor.learn.axis_refine import refine_axes
from shapetensor.learn.model import PopulationNet

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt")
ap.add_argument("--angles", type=float, nargs="+", default=[20, 25, 30, 35, 40, 50, 60, 75, 90]); ap.add_argument("--n", type=int, default=2, help="phantoms per angle and family")
ap.add_argument("--neg", type=int, default=4, help="single bundles (per family), fans, curves, sheets"); ap.add_argument("--families", nargs="+", default=["cross", "fine"])
ap.add_argument("--sigmas", type=float, nargs="+", default=[2.5, 4.0]); ap.add_argument("--step", type=int, default=4); ap.add_argument("--dirs", type=int, default=200)
ap.add_argument("--count-min", type=float, default=0.03); ap.add_argument("--seed0", type=int, default=3_100_000); ap.add_argument("--variants", nargs="+", default=[""])
ap.add_argument("--no-refine", action="store_true")
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"]); sph = EI.sphere_dirs(a.dirs, dev)
BINS = ((20, 30), (30, 40), (40, 60), (60, 91))
unit = lambda v: torch.nn.functional.normalize(torch.as_tensor(v, dtype=torch.float32), dim=-1)


def ang(u, v):
    return torch.rad2deg(torch.acos((unit(u) * unit(v)).sum(-1).abs().clamp(max=1)))


def turned(fam, deg, seed):
    """The generator's crossing with its second bundle at ``deg`` from the first (in their plane); the third dropped."""
    spec, look = P2.random_spec(fam, torch.Generator().manual_seed(seed))
    A = unit(spec[0]["axis"]); B0 = unit(spec[1]["axis"]); q = unit(B0 - (B0 @ A) * A); t = math.radians(deg)
    B = math.cos(t) * A + math.sin(t) * q; nrm = unit(torch.linalg.cross(A, B))
    for s, x in zip(spec, (A, B)):
        s["axis"], s["normal"] = x.tolist(), nrm.tolist()
    return spec[:2], {**look, "contrast": look["contrast"][:2]}, A, B


def scene_list():
    out = []; s = a.seed0
    for fam in a.families:
        for deg in a.angles:
            for i in range(a.n):
                spec, look, A, B = turned(fam, deg, s); out.append(dict(kind="cross", fam=fam, deg=deg, spec=spec, look=look, A=A, B=B, seed=s)); s += 1
        for i in range(a.neg):
            spec, look, A, _ = turned(fam, 60.0, s)
            out.append(dict(kind="single", fam=fam, spec=spec[:1], look={**look, "contrast": look["contrast"][:1]}, seed=s)); s += 1
    for fam in ("fan", "curve", "sheet"):
        for i in range(a.neg):
            spec, look = P2.random_spec(fam, torch.Generator().manual_seed(s)); out.append(dict(kind=fam, fam=fam, spec=spec, look=look, seed=s)); s += 1
    return out


# ---- cache: decoder outputs, truth, profiles ------------------------------------------------------------------
cache = []; prof = []
for sc in scene_list():
    ph = P2.make(sc["spec"], seed=sc["seed"] + 50_000, look=sc["look"], device=dev, ss=2 if sc["fam"] == "fine" else 1); n = ph["n"]
    g1 = torch.arange(6, n - 6, a.step, device=dev, dtype=torch.float32)
    pts = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
    with torch.no_grad():
        feats = EI.features(net, ph["image"])
    for sg in a.sigmas:
        sig = torch.full((pts.shape[0],), sg, device=dev)
        t = P2.targets(ph, pts, sig); K = ph["occ"].shape[0]; mT, kT, RT = t["e_mass"][:, :K], t["e_kappa"][:, :K], t["e_R"][:, :K]
        pop = mT >= a.count_min; line = pop & (kT[..., 1] >= 2.0)
        mL = torch.where(line, mT, torch.zeros_like(mT)); o = mL.argsort(1, descending=True); ar = torch.arange(len(pts), device=dev)
        m1 = mL[ar, o[:, 0]]; m2 = mL[ar, o[:, 1]] if K > 1 else torch.zeros_like(m1)
        ax1 = RT[ar, o[:, 0], :, 2]; ax2 = RT[ar, o[:, 1], :, 2] if K > 1 else ax1
        mP = torch.where(pop, mT, torch.zeros_like(mT)).sort(1, descending=True).values
        one = (mP[:, 0] > 0) & ((mP[:, 1] if K > 1 else torch.zeros_like(m1)) < 0.2 * mP[:, 0])     # one population (any type)
        core = (m2 > 0) & (m2 >= 0.5 * m1) if sc["kind"] == "cross" else torch.zeros_like(one)
        sheet = one & (kT[ar, torch.where(pop, mT, torch.zeros_like(mT)).argmax(1), 1] < 2.0)
        with torch.no_grad():
            r = EI.query(net, feats, pts, sig, sph, vol_shape=ph["image"].shape)
        r = {k: v.half().cpu() if v.is_floating_point() else v.cpu() for k, v in r.items()}
        cache.append(dict(sc=sc, sg=sg, img=ph["image"].half().cpu(), pts=pts.cpu(), r=r, core=core.cpu(), one=one.cpu(), sheet=sheet.cpu(),
                          ax1=ax1.cpu(), ax2=ax2.cpu(), xang=ang(ax1, ax2).cpu()))
        if sc["kind"] == "cross" and bool(core.any()):                                            # the density along the A-B great circle
            A, B = sc["A"].to(dev), sc["B"].to(dev); q = unit(B - (B @ A) * A)
            th = torch.deg2rad(torch.arange(180, device=dev, dtype=torch.float32)); gc = torch.cos(th)[:, None] * A + torch.sin(th)[:, None] * q
            pc = pts[core]; sc_ = sig[core]
            with torch.no_grad():
                fm = EI.query(net, feats, pc, sc_, gc, vol_shape=ph["image"].shape)["f"].float()   # (Qc, 180)
            ft = BG.mixture_pdf(gc, RT[core], kT[core], mT[core], line[core])                    # the target's lobes
            prof.append(dict(fam=sc["fam"], deg=sc["deg"], sg=sg, model=fm.cpu(), target=ft.cpu()))
    print(f"cached {sc['kind']} {sc['fam']} {sc.get('deg', '')}: {len(pts)} points", flush=True)


def peaks(f, deg):
    """Per profile ``f (Q, 180)`` (index = deg from A along the circle to B at ``deg``): two maxima (a local maximum
    within deg/2 of each axis, not on the window's edge), their separation, and the valley between them / lower peak."""
    Q = f.shape[0]; h = max(int(round(deg / 2)), 1); idx = torch.arange(180)
    win = lambda c: ((idx[None] - c) % 180 + 90) % 180 - 90                                       # signed offset from c
    res = []
    for c in (0, int(round(deg)) % 180):
        d = win(c); m = d.abs() < h; fv = torch.where(m, f, torch.full_like(f, -1e9)); j = fv.argmax(1)
        ok = (f[torch.arange(Q), j] >= f[torch.arange(Q), (j + 1) % 180]) & (f[torch.arange(Q), j] >= f[torch.arange(Q), (j - 1) % 180]) & (d[0, j].abs() < h - 1)
        res.append((j, ok, f[torch.arange(Q), j]))
    (ja, oka, fa), (jb, okb, fb) = res
    S = torch.arange(-90, 180); fS = f[:, S % 180]                                                # signed positions from A
    sa = (ja + 90) % 180 - 90; between = (S[None] >= sa[:, None]) & (S[None] <= jb[:, None])
    valley = torch.where(between, fS, torch.full_like(fS, 1e9)).min(1).values
    two = oka & okb & (sa != jb) & (valley < torch.minimum(fa, fb))
    return two, (jb - sa).float(), valley / torch.minimum(fa, fb).clamp_min(1e-12)


summary_p = {}
for src in ("model", "target"):
    for sg in a.sigmas:
        for deg in a.angles:
            rows = [p for p in prof if p["deg"] == deg and p["sg"] == sg]
            if not rows:
                continue
            f = torch.cat([p[src] for p in rows]); two, sep, vr = peaks(f, deg)
            summary_p[f"{src}|{sg}|{deg:g}"] = {"n": int(f.shape[0]), "two_peaks": float(two.float().mean()), "valley_ratio_median": float(vr.median()),
                                                "separation_median": float(sep[two].median()) if bool(two.any()) else float("nan")}
json.dump(summary_p, open(a.out + ".profile.json", "w"), indent=1)
print("== profile at cores along the A-B great circle: two maxima (%), valley / lower peak (median), maxima's separation (median, deg)")
for sg in a.sigmas:
    for deg in a.angles:
        if f"model|{sg}|{deg:g}" in summary_p:
            m_, t_ = summary_p[f"model|{sg}|{deg:g}"], summary_p[f"target|{sg}|{deg:g}"]
            print(f"  sigma {sg} {deg:>3g} deg (n {m_['n']}): model {m_['two_peaks'] * 100:3.0f} %, valley {m_['valley_ratio_median']:.2f}, sep {m_['separation_median']:.0f}"
                  f" | target {t_['two_peaks'] * 100:3.0f} %, valley {t_['valley_ratio_median']:.2f}, sep {t_['separation_median']:.0f}", flush=True)


# ---- variants ----------------------------------------------------------------------------------------------------
def parse(v):
    kw = {}
    for kv in filter(None, v.split(",")):
        k_, x = kv.split("="); kw[k_] = None if x == "None" else (x == "True" if x in ("True", "False") else (float(x) if x.replace(".", "").replace("-", "").isdigit() else x))
    return kw


def score(kw):
    acc = {}
    put = lambda k, x: acc.setdefault(k, []).append(x.float().cpu())
    for it in cache:
        sc, sg = it["sc"], it["sg"]; r = {k: v.to(dev).float() if v.is_floating_point() else v.to(dev) for k, v in it["r"].items()}
        P = r["f"].shape[0]; parts = [EI.entities_from_query({k: v[j:j + 1024] for k, v in r.items()}, sph, iso_k=ET.VOTE_KAPPA_MIN, **kw) for j in range(0, P, 1024)]
        C = max(x["mass"].shape[1] for x in parts)
        pad = lambda x: torch.cat([x, torch.zeros((x.shape[0], C - x.shape[1]) + tuple(x.shape[2:]), device=dev, dtype=x.dtype)], 1) if x.shape[1] < C else x
        ro = {k: torch.cat([pad(x[k]) for x in parts]) for k in ("mass", "R", "sheet")}
        mass, sw = ro["mass"].float(), ro["sheet"].float(); lm = mass * (1 - sw); o = lm.argsort(1, descending=True); ar = torch.arange(P, device=dev)
        ax = ro["R"].float()[..., :, 2]; l1 = lm[ar, o[:, 0]]; l2 = lm[ar, o[:, 1]] if C > 1 else torch.zeros_like(l1)
        p1 = ax[ar, o[:, 0]]; p2 = ax[ar, o[:, 1]] if C > 1 else p1
        two = (l2 >= a.count_min) & (l1 > 0)
        one, core = it["one"].to(dev), it["core"].to(dev)
        if sc["kind"] in ("single", "fan", "curve"):
            put(f"second|{sc['kind']}|{sg}", two[one]); put(f"second_half|{sc['kind']}|{sg}", (two & (l2 >= 0.5 * l1))[one])
        if sc["kind"] == "sheet":
            put(f"sheet_share|{sg}", ((mass * sw).sum(1) / mass.sum(1).clamp_min(1e-9))[it["sheet"].to(dev) & (mass.sum(1) > 0)])
        if sc["kind"] == "cross" and bool(core.any()):
            A1, A2 = it["ax1"].to(dev)[core], it["ax2"].to(dev)[core]; xa = it["xang"].to(dev)[core]
            sets = [("raw", p1[core], p2[core])]
            if not a.no_refine:
                pc = it["pts"].to(dev)[core]; o3 = o[core][:, :min(3, C)]; L0 = lm[core].gather(1, o3) >= a.count_min; L0[:, 0] = True
                Ar = refine_axes(it["img"].to(dev).float(), pc, sg, ax[core].gather(1, o3[..., None].expand(-1, -1, 3)), L0)
                sets.append(("refined", Ar[:, 0], Ar[:, 1] if Ar.shape[1] > 1 else Ar[:, 0]))
            for nm, q1, q2 in sets:
                d11, d12, d21, d22 = ang(q1, A1), ang(q1, A2), ang(q2, A1), ang(q2, A2); tw = two[core]
                found = tw & (((d11 <= 20) & (d22 <= 20)) | ((d12 <= 20) & (d21 <= 20)))
                res = tw & ((((d11 <= 20) & (d22 <= 20) & (d11 < d12) & (d22 < d21)) | ((d12 <= 20) & (d21 <= 20) & (d12 < d11) & (d21 < d22))) & (ang(q1, q2) >= 0.5 * xa))
                for lo_, hi_ in BINS:
                    b = (xa >= lo_) & (xa < hi_)
                    put(f"found|{nm}|{sg}|{lo_}-{hi_}", found[b]); put(f"resolved|{nm}|{sg}|{lo_}-{hi_}", res[b])
                    if nm == "raw":
                        put(f"near|{sg}|{lo_}-{hi_}", torch.minimum(d11, d12)[b]); put(f"count2|{sg}|{lo_}-{hi_}", tw[b])
    out = {}
    for k, v in acc.items():
        x = torch.cat(v); x = x[torch.isfinite(x)]
        out[k] = float(x.median()) if k.startswith(("near", "sheet_share")) else float(x.mean()) if x.numel() else float("nan")
    return out


res = {}
for v in a.variants:
    s = score(parse(v)); res[v or "default"] = s
    print(f"== {v or 'default'}", flush=True)
    for sg in a.sigmas:
        fr = " ".join(f"{lo_}-{hi_ if hi_ < 91 else 90}: {s.get(f'found|raw|{sg}|{lo_}-{hi_}', float('nan')) * 100:3.0f}/"
                      f"{s.get(f'resolved|raw|{sg}|{lo_}-{hi_}', float('nan')) * 100:3.0f}" for lo_, hi_ in BINS)
        neg = " ".join(f"{k_} {s.get(f'second|{k_}|{sg}', float('nan')) * 100:.0f}/{s.get(f'second_half|{k_}|{sg}', float('nan')) * 100:.0f}" for k_ in ("single", "fan", "curve"))
        print(f"   sigma {sg}: cores found/resolved % {fr} | second line % (>= {a.count_min}/>= half) {neg} | sheet share {s.get(f'sheet_share|{sg}', float('nan')):.2f}", flush=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
