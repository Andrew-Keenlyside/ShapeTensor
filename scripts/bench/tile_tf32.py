import time, numpy as np, torch
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
from shapetensor.features.anatomix import AnatomixEngine
rng = np.random.default_rng(0); n = 256
v = 160 + 10 * rng.normal(size=(n, n, n)).astype(np.float32)
z, y, x = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
v += 12 * np.cos(0.6 * (x + 0.4 * z)) * (rng.random((n, n, n)) > 0.5)
v = torch.as_tensor(np.clip(v, 0, 255), device="cuda")
kw = dict(layer="l58", nyquist="none", n_pcs=None, l2=False, out_dtype=torch.float32, amp=False)
t0 = time.time(); f1 = AnatomixEngine(tile=320, margin=32, **kw)(v); torch.cuda.synchronize(); t1 = time.time() - t0
scale = float(f1.abs().amax())
print(f"[TF32 off, fp32] one pass 256^3: {t1:.1f}s; share of exact zeros in f: {float((f1 == 0).float().mean()):.3f}", flush=True)
for tile, margin in ((96, 16), (96, 32), (96, 48), (128, 64)):
    f2 = AnatomixEngine(tile=tile, margin=margin, **kw)(v)
    d = (f1 - f2).abs()[:, 16:-16, 16:-16, 16:-16] / scale
    print(f"tile {tile:3d} margin {margin:2d}: max {float(d.amax()):.2e}  share>1e-4 {float((d > 1e-4).float().mean()):.2e}  share>1e-3 {float((d > 1e-3).float().mean()):.2e}  share>1e-2 {float((d > 1e-2).float().mean()):.2e}", flush=True)
f3 = AnatomixEngine(tile=320, margin=32, **dict(kw, amp=True))(v)
d = (f1 - f3).abs()[:, 16:-16, 16:-16, 16:-16] / scale
print(f"bf16 one pass vs fp32/TF32-off: max {float(d.amax()):.2e} share>1e-2 {float((d > 1e-2).float().mean()):.2e} rms {float(d.pow(2).mean().sqrt()):.2e}", flush=True)
torch.backends.cudnn.allow_tf32 = True
f4 = AnatomixEngine(tile=320, margin=32, **kw)(v)
d = (f1 - f4).abs()[:, 16:-16, 16:-16, 16:-16] / scale
print(f"TF32 one pass vs fp32/TF32-off: max {float(d.amax()):.2e} share>1e-2 {float((d > 1e-2).float().mean()):.2e} rms {float(d.pow(2).mean().sqrt()):.2e}", flush=True)
