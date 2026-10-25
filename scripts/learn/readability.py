"""How readable the generator's training crops are: material fraction, contrast-to-noise of each population against the
background, and the generator's own visibility over the material -- per difficulty level and family mix.

Usage: python scripts/learn/readability.py [N per setting] [size] [device]
"""
import sys, json, math, numpy as np, torch
sys.path.insert(0, "src")
from shapetensor.learn import lobe_train as LT
from shapetensor.learn.scene import Scene, draw_scene

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40; size = int(sys.argv[2]) if len(sys.argv) > 2 else 64; dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
torch.set_num_threads(8)


def measure(s, d):
    img = d["image"].float(); K = d["occupancy"].shape[0]
    vis = d["visibility"].float(); vis = vis.view(-1, 1, 1, 1).expand(K, *img.shape) if vis.dim() == 1 else vis
    keep = [k for k, L in enumerate(s.layers) if L.role == "shape"]
    if not keep:
        return {"material": 0.0, "cnr": 0.0, "vis": 0.0}
    occ = d["occupancy"][keep].float(); tot = occ.sum(0)
    m = slice(4, -4)
    occ, tot, img, vis_ = occ[:, m, m, m], tot[m, m, m], img[m, m, m], vis[keep][:, m, m, m]
    bg = tot < 0.05
    if int(bg.sum()) < 50:                                                       # no background in the crop: compare with the crop's lowest-occupancy decile
        bg = tot <= torch.quantile(tot.flatten()[::7], 0.1)
    sd = img[bg].std().clamp_min(1.0); mu = img[bg].mean()
    cnr = 0.0
    for k in range(occ.shape[0]):
        on = occ[k] > 0.5
        if int(on.sum()) >= 20:
            cnr = max(cnr, float((img[on].mean() - mu).abs() / sd))
    w = tot.clamp(0, 1)
    v = float((vis_.amax(0) * w).sum() / w.sum().clamp_min(1e-6)) if float(w.sum()) > 0 else 0.0
    return {"material": float(w.mean()), "cnr": cnr, "vis": v}


out = {}
fam_compose = {k: (1.0 if k == "compose" else 0.0) for k in Scene.FAMILIES}
for label, extra in [("current mix", {}), ("compose only", {"families": fam_compose})]:
    for lv in (0.3, 0.7, 1.0):
        gen = {**LT.difficulty_gen(lv, {"canvas": 2 * size, "crops_per_scene": 1, "cross_p": 0.3, "p_centre": 0.5})}
        if "families" in extra:
            gen["families"] = extra["families"]
        R = []
        for i in range(N):
            s, d = draw_scene((size,) * 3, seed=70000 + 97 * i, device=dev, strength=lv, **gen)
            R.append(measure(s, d))
        mat = np.array([q["material"] for q in R]); cnr = np.array([q["cnr"] for q in R]); vis = np.array([q["vis"] for q in R])
        unreadable = (mat < 0.01) | (vis < 0.2) | (cnr < 0.5)
        out[f"{label} @ level {lv}"] = {"material<1%": round(float((mat < 0.01).mean()), 2), "material median": round(float(np.median(mat)), 3),
                                        "CNR median": round(float(np.median(cnr)), 2), "CNR<0.5": round(float((cnr < 0.5).mean()), 2),
                                        "visibility median": round(float(np.median(vis)), 2), "visibility<0.2": round(float((vis < 0.2).mean()), 2),
                                        "unreadable (any)": round(float(unreadable.mean()), 2)}
        print(label, lv, out[f"{label} @ level {lv}"], flush=True)
json.dump(out, open("/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/readability.json", "w"), indent=1)
