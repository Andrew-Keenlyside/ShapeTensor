"""Parity and recovery tests for the core: filters vs scipy, K vs the structure tensor, SH vs scipy, CP recovery.
Run with pytest, or ``python tests/test_core.py``."""
import math
import sys

import numpy as np
import torch
from scipy import ndimage as ndi

from shapetensor import filters, odf, tensor
from shapetensor.readout import common, multi, single

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_gaussian_matches_scipy():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(40, 33, 37)).astype(np.float32)
    x = torch.as_tensor(v, device=DEV)
    for sigma, order in ((1.0, (0, 0, 0)), (1.0, (0, 0, 1)), (2.5, (1, 0, 0)), (3.0, (0, 1, 0)), (1.5, (0, 2, 0)), (2.0, (1, 0, 1))):
        ref = ndi.gaussian_filter(v, sigma, order=list(order), mode="nearest", truncate=3.0)
        got = filters.gaussian(x, sigma, order=order, truncate=3.0).cpu().numpy()
        assert np.allclose(got, ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max()), (sigma, order, np.abs(got - ref).max())


def _bridge_structure_tensor(vol, sigma, rho, truncate):
    """BRIDGE's _structure_tensor on the numpy lane: S (6, Z, Y, X) order [zz, yy, xx, zy, zx, yx]."""
    Vx = ndi.gaussian_filter(vol, sigma, order=[0, 0, 1], mode="nearest", truncate=truncate)
    Vy = ndi.gaussian_filter(vol, sigma, order=[0, 1, 0], mode="nearest", truncate=truncate)
    Vz = ndi.gaussian_filter(vol, sigma, order=[1, 0, 0], mode="nearest", truncate=truncate)
    S = np.empty((6,) + vol.shape, np.float64)
    for i, (a, b) in enumerate(((Vz, Vz), (Vy, Vy), (Vx, Vx), (Vz, Vy), (Vz, Vx), (Vy, Vx))):
        S[i] = ndi.gaussian_filter((a * b).astype(np.float64), rho, mode="nearest", truncate=truncate)
    return S


def test_k_one_channel_is_structure_tensor():
    rng = np.random.default_rng(1)
    vol = (120 + 20 * rng.normal(size=(64, 40, 64))).astype(np.float32)
    sigma, rho, tr = 1.0, 3.0, 3.0
    S = _bridge_structure_tensor(vol, sigma, rho, tr)
    F = torch.as_tensor(vol, device=DEV)[None]
    G = tensor.gradients(F, sigma, tr, out_dtype=torch.float32)
    jy = 20
    K, sites = tensor.k_sites(G, rho, tr, reduce=1, y_rows=[jy], core_zx=((8, 56), (8, 56)))
    assert K.shape == (1, 48, 48, 6)
    Km = common.unpack(K.reshape(-1, 6), 3).cpu().numpy().reshape(48, 48, 3, 3)
    ref = np.empty((48, 48, 3, 3))
    idx = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}
    for (i, j), k in idx.items():
        ref[..., i, j] = ref[..., j, i] = S[k, 8:56, jy, 8:56]
    assert np.allclose(Km, ref, rtol=1e-4, atol=1e-6 * np.abs(ref).max()), np.abs(Km - ref).max() / np.abs(ref).max()
    # the eigen read-out equals BRIDGE's smallest eigenvector (axially)
    e = single.eigen(common.unpack(K.reshape(-1, 6), 3))["dir"].cpu().numpy()
    w, V = np.linalg.eigh(ref.reshape(-1, 3, 3))
    e_ref = V[:, :, 0]
    ang = np.degrees(np.arccos(np.clip(np.abs((e * e_ref).sum(1)), 0, 1)))
    assert np.median(ang) < 0.5, np.median(ang)
    # the dense form agrees with the row form
    Kd, _ = tensor.k_dense(G, rho, tr, reduce=1, core=((8, 56), (jy, jy + 1), (8, 56)))
    assert np.allclose(Kd[:, 0].cpu().numpy(), K[0].cpu().numpy(), rtol=1e-4, atol=1e-6 * np.abs(ref).max())


def test_k_reduce_approximates_full():
    rng = np.random.default_rng(2)
    vol = (120 + 20 * rng.normal(size=(64, 40, 64))).astype(np.float32)
    F = torch.as_tensor(vol, device=DEV)[None]
    G = tensor.gradients(F, 1.0, 3.0, out_dtype=torch.float32)
    K1, s1 = tensor.k_sites(G, 8.0, 3.0, reduce=1, y_rows=[20], core_zx=((16, 48), (16, 48)))
    K2, s2 = tensor.k_sites(G, 8.0, 3.0, reduce=2, y_rows=[20], core_zx=((16, 48), (16, 48)))
    # compare at the sites of the reduced grid (block centres: between full-res samples; average the 4 neighbours)
    a = K1[0].reshape(16, 2, 16, 2, 6).mean((1, 3)).cpu().numpy()
    b = K2[0].cpu().numpy()
    rel = np.abs(a - b).max() / np.abs(a).max()
    assert rel < 0.02, rel


def test_real_sh_matches_scipy():
    from scipy.special import sph_harm_y
    rng = np.random.default_rng(3)
    d = rng.normal(size=(50, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lmax = 8
    got = odf.real_sh(torch.as_tensor(d, dtype=torch.float32), lmax).numpy()
    theta = np.arccos(np.clip(d[:, 0], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 2])
    cols = []
    for l in range(0, lmax + 1, 2):
        for m in range(-l, l + 1):
            Y = sph_harm_y(l, abs(m), theta, phi)
            cols.append(np.sqrt(2) * (-1) ** m * Y.imag if m < 0 else (Y.real if m == 0 else np.sqrt(2) * (-1) ** m * Y.real))
    ref = np.stack(cols, -1)
    assert np.allclose(got, ref, atol=1e-4), np.abs(got - ref).max()


def test_frt_ring_peaks_at_fibre():
    """Gradient directions spread on the great circle perpendicular to u give an FRT-ODF peak at u."""
    rng = np.random.default_rng(4)
    u = np.array([0.3, 0.5, 0.8])
    u /= np.linalg.norm(u)
    a = np.cross(u, [1.0, 0, 0]); a /= np.linalg.norm(a); b = np.cross(u, a)
    t = rng.uniform(0, 2 * np.pi, 4000)
    g = np.outer(np.cos(t), a) + np.outer(np.sin(t), b) + 0.05 * rng.normal(size=(4000, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    lmax = 8
    coef = odf.real_sh(torch.as_tensor(g, dtype=torch.float32), lmax).sum(0, keepdim=True)
    f = odf.frt(coef, lmax)
    dirs, amps, cnt = odf.peaks(f, lmax, nmax=3, rel=0.5, min_sep_deg=25)
    assert int(cnt[0]) >= 1
    ang = math.degrees(math.acos(min(1.0, abs(float(dirs[0, 0].numpy() @ u)))))
    assert ang < 6, ang


def test_cp_recovers_planted_directions():
    torch.manual_seed(0)
    N, P, C, I, R = 6, 20, 8, 3, 3
    A = torch.randn(N, P, R); B = torch.randn(N, C, R)
    D = torch.randn(N, I, R); D = D / D.norm(dim=1, keepdim=True)
    Y = torch.einsum("npr,ncr,nir->npci", A, B, D) + 0.01 * torch.randn(N, P, C, I)
    Y = Y.to(DEV)
    Ah, Bh, Dh, err = multi.cp_als(Y, R, None, starts=4, iters=200)
    assert float(err.max()) < 1e-3, err
    # every planted direction is found (up to permutation and sign)
    for n in range(N):
        for r in range(R):
            cos = (Dh[n].T.cpu() @ D[n, :, r]).abs().max()
            assert float(cos) > math.cos(math.radians(2.0)), float(cos)


def test_pair_geometric_two_populations():
    """Four pieces, two spanning the plane perpendicular to u1 and two perpendicular to u2 -> u1, u2 recovered."""
    u1 = torch.tensor([1.0, 0.0, 0.0]); u2 = torch.tensor([0.0, 0.6, 0.8])
    a1, b1 = torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
    a2 = torch.tensor([1.0, 0.0, 0.0]); b2 = torch.cross(u2, a2, dim=0)
    D = torch.stack([a1, a2, b1, b2], 1)[None]                           # shuffled pieces
    W = torch.eye(4)[None]
    fib, en, msine, best = multi.pair_geometric(D, W)
    got = fib[0]
    for u in (u1, u2):
        assert float((got @ u).abs().max()) > 0.999


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
