"""A per-voxel orientation head on anatomix encoder block 2 (layer 15, 32 ch): multinomial logistic regression from
the standardised channel vector to the fibre angle class (12 classes in the x-z plane), trained on fibre voxels of
single-population phantoms. On crossing windows: the histogram of per-voxel predictions (weighted by confidence) ->
peaks -> per-population voxel sets -> a structure tensor per set. CPU only."""
import json, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as Fn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from shapetensor.phantoms import make
from shapetensor.features.anatomix import load_net
from shapetensor.tensor import gradients
torch.set_num_threads(16)
t0 = time.time()
tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32))
net = load_net("cpu")
LAYER, ANG = 15, np.arange(0, 180, 15.0)


def feats(img):
    x = torch.as_tensor(img).float()[None, None] / 255.0
    with torch.no_grad():
        f = net(x, layers=[LAYER], encode_only=True)[0][0].float()
    return Fn.interpolate(f[None], size=img.shape, mode="trilinear", align_corners=False)[0]


def single(angle, seed):
    rng = np.random.default_rng(seed)
    sp = make.Splat(g, tex, rng, "cpu"); make._add_population(sp, angle, 1.0); d = sp.render()
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng)
    return img, (d > 0.25).numpy()


core = (slice(36, 124), slice(12, 84), slice(36, 124))
Xtr, ytr, Xte, yte = [], [], [], []
rng0 = np.random.default_rng(0)
for k, a in enumerate(ANG):
    for split, seed in (("train", 200 + k), ("test", 300 + k)):
        img, m = single(a, seed)
        f = feats(img)[(slice(None),) + core]
        v = f[:, torch.as_tensor(m[core])].T.numpy()
        sel = rng0.choice(len(v), min(len(v), 6000), replace=False)
        if split == "train":
            Xtr.append(v[sel]); ytr.append(np.full(len(sel), k))
        else:
            Xte.append(v[sel]); yte.append(np.full(len(sel), k))
    print(f"[{time.time() - t0:4.0f}s] angle {a:3.0f} done", flush=True)
Xtr, ytr, Xte, yte = (np.concatenate(z) for z in (Xtr, ytr, Xte, yte))
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
pred = clf.predict(sc.transform(Xte))
acc = float((pred == yte).mean())
near = float((np.minimum(np.abs(pred - yte), 12 - np.abs(pred - yte)) <= 1).mean())
print(f"[{time.time() - t0:4.0f}s] per-voxel head: held-out accuracy {acc:.0%} (exact class), {near:.0%} within +-15 deg; chance {1 / 12:.0%}", flush=True)

res = {"heldout_acc": acc, "heldout_within15": near, "tests": {}}
tru_dir = lambda a: make._dir(a)
ang = lambda u, v: np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v))))))
for cross in (90.0, 60.0, 45.0, 30.0):
    a1, a2 = 90 + cross / 2, 90 - cross / 2
    rng = np.random.default_rng(7)
    sp1 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp1, a1, 1.0); d1 = sp1.render()
    sp2 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp2, a2, 1.0); d2 = sp2.render()
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d1 + d2, rng)
    f = feats(img)
    Z, Y, X = img.shape; cz, cy, cx, r = Z // 2, Y // 2, X // 2, 16
    win = (slice(cz - r, cz + r + 1), slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
    fw = f[(slice(None),) + win]
    V = fw.reshape(fw.shape[0], -1).T.numpy()
    P = clf.predict_proba(sc.transform(V))                            # (P, 12)
    conf = P.max(1)
    hist = (P * conf[:, None]).sum(0); hist = hist / hist.sum()       # confidence-weighted vote over classes
    peaks = [i for i in range(12) if hist[i] > 0.3 * hist.max() and hist[i] >= hist[(i - 1) % 12] and hist[i] >= hist[(i + 1) % 12]]
    peaks = sorted(peaks, key=lambda i: -hist[i])[:3]
    tru = [tru_dir(a1), tru_dir(a2)]
    G = gradients(fw, 1.0, 3.0, out_dtype=torch.float32).reshape(fw.shape[0], 3, -1).numpy()
    zz, yy, xx = np.meshgrid(*(np.arange(-r, r + 1),) * 3, indexing="ij"); w = np.exp(-0.5 * (zz ** 2 + yy ** 2 + xx ** 2) / 10.0 ** 2).ravel()
    pops = []
    for i in peaks:
        # this population's voxel set: voxels whose predicted class is within +-15 deg of the peak, weighted by confidence
        dist = np.minimum(np.abs(P.argmax(1) - i), 12 - np.abs(P.argmax(1) - i))
        wv = (dist <= 1) * conf * w
        T = np.einsum("cip,cjp,p->ij", G, G, wv); e3 = np.linalg.eigh(T)[1][:, 0]
        pops.append({"peak_angle": float(ANG[i]), "vote": float(hist[i]), "n_voxels": int((dist <= 1).sum()),
                     "err_nearest_truth": min(ang(e3, tru[0]), ang(e3, tru[1])), "err_truth1": ang(e3, tru[0]), "err_truth2": ang(e3, tru[1])})
    T = np.einsum("cip,cjp,p->ij", G, G, w); e3 = np.linalg.eigh(T)[1][:, 0]
    res["tests"][f"cross{cross:g}"] = {"true_angles": [a1, a2], "hist": hist.tolist(), "peaks": pops, "all": [ang(e3, tru[0]), ang(e3, tru[1])]}
    print(f"[{time.time() - t0:4.0f}s] crossing {cross:g} (true {a1:g}, {a2:g}): votes " + " ".join(f"{ANG[i]:.0f}:{hist[i]:.2f}" for i in np.argsort(-hist)[:4])
          + " | peaks " + "; ".join(f"{p['peak_angle']:.0f}deg ({p['n_voxels']} vox) -> {p['err_nearest_truth']:.0f}deg off" for p in pops)
          + f" | all voxels {ang(e3, tru[0]):.0f}/{ang(e3, tru[1]):.0f}", flush=True)
json.dump(res, open("/home/andrew/bridge_wmcov/shapetensor/results/encoder_head.json", "w"), indent=1)
print("ENCODER HEAD DONE", flush=True)
