import sys, time, numpy as np, torch
sys.path.insert(0, "/home/andrew/scripts/scalable-reg")
from chunkreg.features.anatomix import load_backbone
net = load_backbone("anatomix", "cuda"); net.eval()
print("params", sum(p.numel() for p in net.parameters()) / 1e6, "M", flush=True)
for n in (256, 320, 352, 384, 448):
    x = torch.rand(1, 1, n, n, n, device="cuda")
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = net(x, layers=[58], encode_only=True)[0][0]
        torch.cuda.synchronize(); dt = time.time() - t
        print(f"{n}^3 layer58 {tuple(f.shape)} {dt:.2f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
        del f
    except Exception as e:
        print(n, "FAIL", str(e)[:160], flush=True)
    del x; torch.cuda.empty_cache()
n = 192
x = torch.rand(1, 1, n, n, n, device="cuda")
with torch.no_grad():
    f0 = net(x, layers=[58], encode_only=True)[0][0].float()
    x2 = x.clone(); x2[0, 0, n // 2, n // 2, n // 2] += 1.0
    f1 = net(x2, layers=[58], encode_only=True)[0][0].float()
d = (f1 - f0).abs().amax(0).cpu().numpy()
prof = d[n // 2, n // 2, :]
for thr in (1e-2, 1e-3, 1e-4):
    nz = np.flatnonzero(prof > thr * prof.max())
    print(f"layer58 RF along z > {thr:g} of max: [{nz.min()}, {nz.max()}] half-width {(nz.max()-nz.min())//2} vox", flush=True)
