"""Label-free per-population channel sets from anatomix's second encoder block (layer 15, 32 ch): calibrate each
channel's orientation tuning on single-population phantoms (fibres at 12 in-plane angles, x-z plane), then on a
crossing window unmix the channel energies into an orientation response (NNLS over the tuning matrix), take its
peaks as the populations, use each peak's tuned channels as that population's channel set, and run a tensor per set.
CPU only. Output: results/encoder_tuning.json and a tuning-curve figure."""
import json, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as Fn
from scipy.optimize import nnls
from shapetensor.phantoms import make
from shapetensor.features.anatomix import load_net
from shapetensor.filters import gaussian
torch.set_num_threads(16)
t0 = time.time()
tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
g = make.Grid(vox=0.02, core=(96, 32, 96), margin=(32, 32, 32))
net = load_net("cpu")
LAYER = 15


def feats(img):
    x = torch.as_tensor(img).float()[None, None] / 255.0
    with torch.no_grad():
        f = net(x, layers=[LAYER], encode_only=True)[0][0].float()
    return Fn.interpolate(f[None], size=img.shape, mode="trilinear", align_corners=False)[0]


def channel_energy(f, mask):
    c = (slice(36, f.shape[1] - 36), slice(12, f.shape[2] - 12), slice(36, f.shape[3] - 36))
    fc, m = f[(slice(None),) + c], torch.as_tensor(mask[c])
    return (fc[:, m] ** 2).mean(1).numpy()


# 1. tuning matrix: channel energy on fibre voxels of single populations at 12 angles
ANG = np.arange(0, 180, 15.0)
M = []
for k, a in enumerate(ANG):
    rng = np.random.default_rng(100 + k)
    sp = make.Splat(g, tex, rng, "cpu"); make._add_population(sp, a, 1.0); d = sp.render()
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d, rng)
    M.append(channel_energy(feats(img), (d > 0.25).numpy()))
    print(f"[{time.time() - t0:4.0f}s] tuning angle {a:3.0f}", flush=True)
M = np.array(M).T                                                    # (C, n_angles)
Mn = M / M.mean(1, keepdims=True)                                     # per-channel relative tuning
tuning_depth = (Mn.max(1) - Mn.min(1)) / (Mn.max(1) + Mn.min(1))
pref = ANG[Mn.argmax(1)]
print("tuning depth per channel (max-min)/(max+min):", np.round(tuning_depth, 2), flush=True)

# 2. crossing windows: unmix the channel energies -> orientation response -> peaks -> per-population channel sets
res = {"angles": ANG.tolist(), "tuning_depth": tuning_depth.tolist(), "preferred_angle": pref.tolist(), "tests": {}}
for cross in (90.0, 60.0, 45.0):
    a1, a2 = 90 + cross / 2, 90 - cross / 2
    rng = np.random.default_rng(7)
    sp1 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp1, a1, 1.0); d1 = sp1.render()
    sp2 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp2, a2, 1.0); d2 = sp2.render()
    Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
    img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d1 + d2, rng)
    f = feats(img)
    Z, Y, X = img.shape; cz, cy, cx, r = Z // 2, Y // 2, X // 2, 16
    win = (slice(None), slice(cz - r, cz + r + 1), slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
    fw = f[win]
    e = (fw ** 2).mean((1, 2, 3)).numpy()                            # channel energies in the window
    # orientation response: e / mean_c  ~  Mn @ a, a >= 0
    a_resp, _ = nnls(Mn, e / M.mean(1))
    # peaks of the (circular) response
    peaks = [i for i in range(len(ANG)) if a_resp[i] > 0.3 * a_resp.max() and a_resp[i] >= a_resp[(i - 1) % len(ANG)] and a_resp[i] >= a_resp[(i + 1) % len(ANG)]]
    peaks = sorted(peaks, key=lambda i: -a_resp[i])[:3]
    tru = [make._dir(a1), make._dir(a2)]
    ang = lambda u, v: np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v))))))
    zz, yy, xx = np.meshgrid(*(np.arange(-r, r + 1),) * 3, indexing="ij"); w = torch.as_tensor(np.exp(-0.5 * (zz ** 2 + yy ** 2 + xx ** 2) / 10.0 ** 2).astype(np.float32))
    Gw = torch.stack([gaussian(fw[k], 1.0, order=o, truncate=3.0) for k in range(fw.shape[0]) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]).reshape(fw.shape[0], 3, -1) * w.reshape(-1).sqrt()[None, None]
    pops = []
    for i in peaks:
        th = ANG[i]
        # the channel set of this population: channels whose tuning near th is above their mean (weighted by tuning)
        wts = np.clip(Mn[:, i] - 1.0, 0, None)
        if wts.sum() == 0:
            continue
        T = torch.einsum("c,cip,cjp->ij", torch.as_tensor(wts, dtype=torch.float32), Gw, Gw).numpy()
        e3 = np.linalg.eigh(T)[1][:, 0]
        pops.append({"peak_angle": float(th), "response": float(a_resp[i]), "n_channels": int((wts > 0).sum()),
                     "err_truth1": ang(e3, tru[0]), "err_truth2": ang(e3, tru[1])})
    Tall = torch.einsum("cip,cjp->ij", Gw, Gw).numpy(); e3a = np.linalg.eigh(Tall)[1][:, 0]
    res["tests"][f"cross{cross:g}"] = {"true_angles": [a1, a2], "response": a_resp.tolist(), "peaks": pops,
                                       "all_channels": [ang(e3a, tru[0]), ang(e3a, tru[1])]}
    print(f"[{time.time() - t0:4.0f}s] crossing {cross:g} (true {a1:g}, {a2:g}): peaks " +
          "; ".join(f"{p['peak_angle']:.0f}deg ({p['n_channels']} ch) -> {min(p['err_truth1'], p['err_truth2']):.0f}deg off nearest truth" for p in pops)
          + f" | all channels {ang(e3a, tru[0]):.0f}/{ang(e3a, tru[1]):.0f}", flush=True)
json.dump(res, open("/home/andrew/bridge_wmcov/shapetensor/results/encoder_tuning.json", "w"), indent=1)
# figure: tuning curves
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), dpi=130)
order = np.argsort(pref)
im = axs[0].imshow(Mn[order], aspect="auto", cmap="Blues", vmin=0.5, vmax=1.5, extent=(-7.5, 172.5, 32, 0))
axs[0].set_xlabel("fibre angle to the z = const plane (deg)"); axs[0].set_ylabel("encoder-2 channel (sorted by preferred angle)")
axs[0].set_title("a  orientation tuning of anatomix encoder block 2 (layer 15)\nrelative energy on single-population fibres", fontsize=9, loc="left")
fig.colorbar(im, ax=axs[0], fraction=0.04, pad=0.02, label="energy / channel mean")
for cross, col in (("cross90", "#2a78d6"), ("cross60", "#eb6834"), ("cross45", "#1baf7a")):
    t = res["tests"][cross]
    axs[1].plot(ANG, np.array(t["response"]) / max(max(t["response"]), 1e-9), color=col, lw=2, marker="o", ms=4, label=f"crossing {cross[5:]}°: truths at {t['true_angles'][0]:.0f}°, {t['true_angles'][1]:.0f}°")
    for a in t["true_angles"]:
        axs[1].axvline(a, color=col, lw=0.8, ls=":")
axs[1].set_xlabel("orientation (deg)"); axs[1].set_ylabel("unmixed response (NNLS), normalised"); axs[1].legend(fontsize=8, frameon=False)
axs[1].set_title("b  a crossing window's channel energies unmixed over the tuning matrix\n(dotted = the true directions)", fontsize=9, loc="left")
for s in ("top", "right"):
    axs[1].spines[s].set_visible(False)
fig.tight_layout()
out = Path("/home/andrew/scripts/ShapeTensors/figures/tca_field/encoder_tuning.png"); fig.savefig(out); print("wrote", out, flush=True)
print("ENCODER TUNING DONE", flush=True)
