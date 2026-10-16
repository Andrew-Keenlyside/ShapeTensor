"""Do the TCA pieces' channel signatures differ between the two bundles, and does projecting the features onto a
pair's signature subspace give a per-population channel set whose structure tensor recovers that bundle alone?"""
import json, numpy as np, torch
from shapetensor.phantoms import make
from shapetensor.features.anatomix import AnatomixEngine
from shapetensor.tensor import gradients
from shapetensor.readout import multi
torch.set_num_threads(16)
tex = json.load(open("/home/andrew/bridge_wmcov/shapetensor/results/tex_calibrated.json"))["texture"]; tex["widths"] = [tuple(w) for w in tex["widths"]]
tex = make.TextureParams(**{**tex, "dispersion_deg": 8.0, "pitch_deg": 4.0})
g = make.Grid(vox=0.02, core=(128, 32, 128), margin=(32, 32, 32))
img, tr, meta = make.build("P4-lines-c60-f1", 3, tex, g, device="cpu")
F = AnatomixEngine(layer="l58", device="cpu", amp=False, nyquist="none", tile=128, margin=32, n_pcs=8, l2=True, out_dtype=torch.float32)(torch.as_tensor(img).float())
G = gradients(F, 1.0, 3.0, out_dtype=torch.float32)
Z, Y, X = img.shape; cz, cy, cx, r, rho = Z // 2, Y // 2, X // 2, 16, 10.0
sl = (slice(cz - r, cz + r + 1), slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1))
zz, yy, xx = np.meshgrid(*(np.arange(-r, r + 1),) * 3, indexing="ij"); w = np.exp(-0.5 * (zz ** 2 + yy ** 2 + xx ** 2) / rho ** 2).astype(np.float32)
Gw = G[(slice(None),) + sl].numpy() * np.sqrt(w)[None]; nC = F.shape[0]
Xt = torch.as_tensor(np.ascontiguousarray(Gw.reshape(nC, 3, -1).transpose(2, 0, 1)))[None]; Xt = Xt / Xt.norm()
A, B, D, err = multi.cp_als(Xt, 4, None, starts=6, iters=400, seed=1)
fib, en, msine, best = multi.pair_geometric(D, multi.block_gram(A, B, D)); pairs = multi.matchings(4)[int(best[0])]
Bn, Dn, fib = B[0].numpy(), D[0].numpy(), fib[0].numpy()
tru = [tr["dir1"][cz, cx], tr["dir2"][cz, cx]]
ang = lambda u, v: np.degrees(np.arccos(min(1.0, abs(float(np.dot(u, v))))))
print("pairs", pairs, "pair fibres -> nearest truth:", [round(min(ang(fib[q], tru[0]), ang(fib[q], tru[1])), 1) for q in range(2)])
cos = np.abs(Bn.T @ Bn); np.set_printoptions(precision=2, suppress=True)
print("|cosine| between piece channel signatures (rows/cols = pieces 1-4):\n", cos)
# per-population channel sets: project the 8 features onto span(b_r, b_s) of each pair, then a plain tensor on that set
Gfull = G[(slice(None),) + sl].numpy().reshape(nC, 3, -1)            # (C, 3, P)
for k, (p, q) in enumerate(pairs):
    Q, _ = np.linalg.qr(np.stack([Bn[:, p], Bn[:, q]], 1))            # (C, 2) orthonormal basis of the pair's signatures
    Gk = np.einsum("ck,cip->kip", Q, Gfull) * np.sqrt(w).ravel()[None, None]
    T = np.einsum("kip,kjp->ij", Gk, Gk); e3 = np.linalg.eigh(T)[1][:, 0]
    # the complement: everything the pair's signatures do not span
    Qc = np.linalg.svd(np.eye(nC) - Q @ Q.T)[0][:, :nC - 2]
    Gc = np.einsum("ck,cip->kip", Qc, Gfull) * np.sqrt(w).ravel()[None, None]
    Tc = np.einsum("kip,kjp->ij", Gc, Gc); e3c = np.linalg.eigh(Tc)[1][:, 0]
    print(f"pair {k + 1} = pieces {p + 1}+{q + 1}: tensor on its 2-channel set -> {ang(e3, tru[0]):.0f}° / {ang(e3, tru[1]):.0f}° from truth 1 / 2; "
          f"on the complementary 6 channels -> {ang(e3c, tru[0]):.0f}° / {ang(e3c, tru[1]):.0f}°")
T = np.einsum("cip,cjp->ij", Gw.reshape(nC, 3, -1), Gw.reshape(nC, 3, -1)); e3 = np.linalg.eigh(T)[1][:, 0]
print(f"all 8 channels (the shape tensor): {ang(e3, tru[0]):.0f}° / {ang(e3, tru[1]):.0f}°")
