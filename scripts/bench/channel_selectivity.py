"""Are any anatomix channels orientation-selective? Per-channel energy (and gradient energy) on bundle-1-only vs
bundle-2-only voxels of a 60-degree crossing, for the first encoder block (layer 8, 16 ch), the second (layer 15, 32 ch
at 1/2 res), the output head's input (layer 58, 48 ch) and the 16-D output; then a tensor on the channels that prefer
each bundle."""
import json, numpy as np, torch
import torch.nn.functional as Fn
from shapetensor.phantoms import make
from shapetensor.features.anatomix import load_net
from shapetensor.filters import gaussian
torch.set_num_threads(16)
tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
g = make.Grid(vox=0.02, core=(128, 32, 128), margin=(32, 32, 32))
rng = np.random.default_rng(5); a1, a2 = 120.0, 60.0
sp1 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp1, a1, 1.0); d1 = sp1.render()
sp2 = make.Splat(g, tex, rng, "cpu"); make._add_population(sp2, a2, 1.0); d2 = sp2.render()
Zm, Xm = np.meshgrid(g.coords(0), g.coords(2), indexing="ij")
img = make._finish(g, tex, np.full(Zm.shape, 164.0, np.float32), np.full(Zm.shape, tex.amp, np.float32), d1 + d2, rng)
m1, m2 = (d1 > 0.25).numpy(), (d2 > 0.25).numpy()
only1, only2 = m1 & ~m2, m2 & ~m1
net = load_net("cpu")
x = torch.as_tensor(img).float()[None, None] / 255.0
with torch.no_grad():
    taps = net(x, layers=[8, 15, 58], encode_only=True)
    out16 = net(x)[0]
tru = [make._dir(a1), make._dir(a2)]
ang = lambda u, v: np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v))))))
Z, Y, X = img.shape; c = (slice(40, Z - 40), slice(16, Y - 16), slice(40, X - 40))       # core away from faces
for name, f in (("layer 8 (encoder 1, 16 ch)", taps[0][0]), ("layer 15 (encoder 2, 32 ch, 1/2 res)", taps[1][0]), ("layer 58 (48 ch)", taps[2][0]), ("output (16 ch)", out16)):
    f = f.float()
    if f.shape[1:] != img.shape:
        f = Fn.interpolate(f[None], size=img.shape, mode="trilinear", align_corners=False)[0]
    f = f / f.norm(dim=0, keepdim=True).clamp_min(1e-6) if name.startswith("output") else f
    fc = f[(slice(None),) + c]; o1, o2 = torch.as_tensor(only1[c]), torch.as_tensor(only2[c])
    e1 = (fc[:, o1] ** 2).mean(1); e2 = (fc[:, o2] ** 2).mean(1)
    sel = ((e1 - e2) / (e1 + e2 + 1e-12)).numpy()
    # gradient energy per channel on each bundle
    G = torch.stack([gaussian(fc[k], 1.0, order=o, truncate=3.0) for k in range(fc.shape[0]) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]).reshape(fc.shape[0], 3, *fc.shape[1:])
    ge1 = (G[:, :, o1] ** 2).sum(1).mean(1); ge2 = (G[:, :, o2] ** 2).sum(1).mean(1)
    gsel = ((ge1 - ge2) / (ge1 + ge2 + 1e-12)).numpy()
    print(f"\n{name}: energy selectivity (E1-E2)/(E1+E2) per channel: max |s| {np.abs(sel).max():.2f}, n(|s|>0.2) {int((np.abs(sel) > 0.2).sum())}/{len(sel)}; "
          f"gradient-energy selectivity: max |s| {np.abs(gsel).max():.2f}, n(|s|>0.2) {int((np.abs(gsel) > 0.2).sum())}")
    # per-population channel sets from the selectivity, scored by a tensor on each set over a window at the centre
    r = 16; cz, cy, cx = Z // 2, Y // 2, X // 2
    zz, yy, xx = np.meshgrid(*(np.arange(-r, r + 1),) * 3, indexing="ij"); w = torch.as_tensor(np.exp(-0.5 * (zz ** 2 + yy ** 2 + xx ** 2) / 10.0 ** 2).astype(np.float32))
    Gw = torch.stack([gaussian(f[k, cz - r:cz + r + 1, cy - r:cy + r + 1, cx - r:cx + r + 1], 1.0, order=o, truncate=3.0) for k in range(f.shape[0]) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]).reshape(f.shape[0], 3, -1) * w.reshape(-1).sqrt()[None, None]
    for lab, chans in (("channels preferring bundle 1", np.flatnonzero(gsel > 0.1)), ("channels preferring bundle 2", np.flatnonzero(gsel < -0.1)), ("all channels", np.arange(f.shape[0]))):
        if len(chans) == 0:
            print(f"   {lab}: none"); continue
        T = torch.einsum("cip,cjp->ij", Gw[chans], Gw[chans]).numpy(); e3 = np.linalg.eigh(T)[1][:, 0]
        print(f"   {lab} ({len(chans)}): tensor e3 -> {ang(e3, tru[0]):.0f}° / {ang(e3, tru[1]):.0f}° from truth 1 / 2")
