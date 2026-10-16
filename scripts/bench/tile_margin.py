import time, numpy as np, torch
from shapetensor.features.anatomix import AnatomixEngine, load_net
net = load_net("cuda")
norms = sorted({type(m).__name__ for m in net.modules() if "Norm" in type(m).__name__})
print("norm layers:", norms, flush=True)
rng = np.random.default_rng(0); n = 256
v = 160 + 10 * rng.normal(size=(n, n, n)).astype(np.float32)
z, y, x = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
v += 12 * np.cos(0.6 * (x + 0.4 * z)) * (rng.random((n, n, n)) > 0.5)
v = torch.as_tensor(np.clip(v, 0, 255), device="cuda")
kw = dict(layer="l58", nyquist="none", n_pcs=None, l2=False, out_dtype=torch.float32, amp=False)
f1 = AnatomixEngine(tile=320, margin=32, **kw)(v)
scale = float(f1.abs().amax())
for tile, margin in ((96, 32), (96, 48), (96, 64), (128, 64), (64, 96)):
    f2 = AnatomixEngine(tile=tile, margin=margin, **kw)(v)
    d = (f1 - f2).abs()[:, 16:-16, 16:-16, 16:-16] / scale
    q = [float((d > t).float().mean()) for t in (1e-3, 1e-2)]
    # where are the big differences? distance to the nearest tile-core boundary along each axis
    idx = torch.nonzero(d.amax(0) > 1e-2)
    loc = ""
    if len(idx):
        c = (idx + 16).cpu().numpy()
        db = [np.min(np.abs(c[:, k][:, None] - np.arange(0, n + 1, tile)[None]), 1) for k in range(3)]
        loc = f"  voxels >1e-2: {len(idx)}, median dist to a tile edge z/y/x = {np.median(db[0]):.0f}/{np.median(db[1]):.0f}/{np.median(db[2]):.0f}"
    print(f"tile {tile:3d} margin {margin:2d}: max {float(d.amax()):.2e}  share>1e-3 {q[0]:.2e}  share>1e-2 {q[1]:.2e}{loc}", flush=True)
