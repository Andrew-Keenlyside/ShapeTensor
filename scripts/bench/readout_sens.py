"""Does the tiling / bf16 fuzz move the orientations? One-pass fp32 (TF32 off) vs tiled bf16 on a textured block."""
import numpy as np, torch
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
from shapetensor.features.anatomix import AnatomixEngine
from shapetensor.pca import PCA
from shapetensor.tensor import gradients, k_sites
from shapetensor.readout import common, single
from shapetensor.phantoms import make
g = make.Grid(vox=0.02, core=(128, 32, 128), margin=(64, 64, 64))     # block 256^3: one pass possible
tex = make.TextureParams(amp=100, noise_sd=95.7, noise_sigma=1.1, density=120, widths=((1.0,0.45),(2.0,0.3),(3.0,0.15),(6.0,0.1)), levels={"wm":164.0,"gm":126.0,"csf":60.0})
img, tr, meta = make.build("P3-15", 0, tex, g)
v = torch.as_tensor(img, device="cuda").float()
def feats(**kw):
    e = AnatomixEngine(layer="l58", phases=2, nyquist="phase", n_pcs=None, l2=True, out_dtype=torch.float32, **kw)
    return e(v)
F_ref = feats(tile=256, margin=64, amp=False)                       # one pass, fp32
pca = PCA.fit(F_ref, 16, seed=0)
variants = {"tiled64 bf16": feats(tile=128, margin=64, amp=True), "tiled64 fp32": feats(tile=128, margin=64, amp=False),
            "tiled32 bf16": feats(tile=96, margin=32, amp=True), "onepass bf16": feats(tile=256, margin=64, amp=True)}
def dirs(F, rho):
    G = gradients(pca(F), 1.0, 3.0, out_dtype=torch.float32)
    K, s = k_sites(G, rho, 3.0, max(1, int(rho // 4)), [g.y_mid], g.core_zx)
    Kt = common.unpack(torch.as_tensor(K.reshape(-1, K.shape[-1])), 48)
    return single.eigen(Kt)["dir"], single.tca(Kt, rank=3)["dir"]
for rho in (8.0, 16.0):
    e0, t0 = dirs(F_ref, rho)
    for name, F in variants.items():
        e1, t1 = dirs(F, rho)
        ae = common.angle_deg(e0, e1); at = common.angle_deg(t0, t1)
        print(f"rho {rho:g} {name:13s}: eigen angle median {float(ae.median()):.2f} p90 {float(ae.quantile(0.9)):.2f} max {float(ae.max()):.1f} | "
              f"tca median {float(at.median()):.2f} p90 {float(at.quantile(0.9)):.2f} max {float(at.max()):.1f}", flush=True)
