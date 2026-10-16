"""The structure-tensor-compatible volume API: element layout and eigen conventions against scipy / numpy, the
excised variant, fixels, array kinds, and the chunked driver's equality with a one-pass computation."""
import numpy as np
import torch
from scipy import ndimage as ndi

from shapetensor import st
from shapetensor.odf import peaks, real_sh

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _reference_S(vol, sigma, rho, truncate):
    """structure_tensor.structure_tensor_3d as the package computes it: [xx, yy, zz, xy, xz, yz], x = last axis."""
    Vx = ndi.gaussian_filter(vol, sigma, order=(0, 0, 1), mode="nearest", truncate=truncate)
    Vy = ndi.gaussian_filter(vol, sigma, order=(0, 1, 0), mode="nearest", truncate=truncate)
    Vz = ndi.gaussian_filter(vol, sigma, order=(1, 0, 0), mode="nearest", truncate=truncate)
    S = np.empty((6,) + vol.shape, np.float64)
    for k, (a, b) in enumerate(((Vx, Vx), (Vy, Vy), (Vz, Vz), (Vx, Vy), (Vx, Vz), (Vy, Vz))):
        S[k] = ndi.gaussian_filter(a * b, rho, mode="nearest", truncate=truncate)
    return S


def test_structure_tensor_layout_matches_package():
    rng = np.random.default_rng(0)
    vol = (120 + 20 * rng.normal(size=(40, 36, 44))).astype(np.float32)
    ref = _reference_S(vol, 1.0, 3.0, 4.0)
    S = st.structure_tensor_3d(vol, 1.0, 3.0, device=DEV)
    assert isinstance(S, np.ndarray) and S.shape == (6, 40, 36, 44)
    assert np.allclose(S, ref, rtol=1e-4, atol=1e-6 * np.abs(ref).max())


def test_eig_conventions():
    rng = np.random.default_rng(1)
    vol = (120 + 20 * rng.normal(size=(24, 20, 28))).astype(np.float32)
    S = st.structure_tensor_3d(vol, 1.0, 2.0, device=DEV)
    val, vec = st.eig_special_3d(S, device=DEV)
    assert val.shape == (3, 24, 20, 28) and vec.shape == (3, 24, 20, 28)
    assert np.all(np.diff(val, axis=0) <= 1e-5)                              # descending
    M = np.stack([np.stack([S[0], S[3], S[4]]), np.stack([S[3], S[1], S[5]]), np.stack([S[4], S[5], S[2]])])  # (3,3,...)
    Mf = M.reshape(3, 3, -1).transpose(2, 0, 1)
    w, V = np.linalg.eigh(Mf)
    e_small = V[:, :, 0]                                                     # smallest eigenvalue's vector, (x, y, z)
    cos = np.abs((vec.reshape(3, -1).T * e_small).sum(1))
    assert np.median(cos) > 0.9999
    assert np.allclose(np.sort(w, 1)[:, ::-1], val.reshape(3, -1).T, rtol=1e-4, atol=1e-6 * np.abs(w).max())
    val_a, vec_a = st.eig_special_3d(S, eigenvalue_order="asc", device=DEV)
    assert np.allclose(val_a[0], val[2])
    _, vecs = st.eig_special_3d(S, full=True, device=DEV)
    assert vecs.shape == (3, 3, 24, 20, 28)
    assert np.allclose(np.abs((vecs[2] * vec).sum(0)), 1.0, atol=1e-4)       # full[2] = the smallest's (desc order)


def test_torch_in_torch_out_and_excised():
    rng = np.random.default_rng(2)
    vol = torch.as_tensor((120 + 20 * rng.normal(size=(32, 32, 32))).astype(np.float32), device=DEV)
    S = st.structure_tensor_3d(vol, 1.0, 2.0, device=DEV)
    assert isinstance(S, torch.Tensor) and S.device.type == torch.device(DEV).type
    Sx = st.excised_structure_tensor_3d(vol, 1.0, 2.0, factor=100.0, device=DEV)      # nothing excised -> equal
    assert torch.allclose(S, Sx, rtol=1e-4, atol=1e-6 * float(S.abs().max()))


def test_peaks_vectorised_matches_truth():
    rng = np.random.default_rng(4)
    u1 = np.array([1.0, 0, 0]); u2 = np.array([0, 0.6, 0.8])
    coefs = []
    for u in (u1, u2):
        a = np.cross(u, [0.3, 0.7, 0.2]); a /= np.linalg.norm(a); b = np.cross(u, a)
        t = rng.uniform(0, 2 * np.pi, 3000)
        g = np.outer(np.cos(t), a) + np.outer(np.sin(t), b) + 0.05 * rng.normal(size=(3000, 3))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        coefs.append(real_sh(torch.as_tensor(g, dtype=torch.float32, device=DEV), 8).sum(0))
    from shapetensor.odf import frt
    f = frt(torch.stack([coefs[0], coefs[1], coefs[0] + coefs[1]]), 8)
    d, a, c = peaks(f, 8, nmax=3, rel=0.5, min_sep_deg=25)
    assert int(c[0]) == 1 and int(c[1]) == 1 and int(c[2]) == 2
    dn = d.cpu().numpy()
    assert abs(dn[0, 0] @ u1) > 0.99 and abs(dn[1, 0] @ u2) > 0.99
    found = sorted([max(abs(dn[2, k] @ u1), abs(dn[2, k] @ u2)) for k in range(2)])
    assert found[0] > 0.98


def test_fixels_shape():
    rng = np.random.default_rng(5)
    vol = (150 - 30 * (np.sin(0.8 * np.arange(48))[None, None, :] > 0.3) + 5 * rng.normal(size=(48, 48, 48))).astype(np.float32)
    dirs, fracs, npop = st.fixels_3d(vol, 1.0, 8.0, reduce=2, device=DEV)
    assert dirs.shape == (3, 3, 48, 48, 48) and fracs.shape == (3, 48, 48, 48) and npop.shape == (48, 48, 48)
    assert (npop >= 1).mean() > 0.9                                          # stripes along z-y planes: one population
    d = dirs[0][:, 8:-8, 8:-8, 8:-8]                                         # (x, y, z) components; fibres run along y and z
    assert np.median(np.abs(d[0])) < 0.2


def test_run_volume_equals_one_pass(tmp_path):
    import zarr
    rng = np.random.default_rng(6)
    vol = (120 + 20 * rng.normal(size=(60, 48, 52))).astype(np.float32)
    zarr.save_array(str(tmp_path / "src.zarr"), vol)
    st.run_volume(str(tmp_path / "src.zarr"), str(tmp_path / "dst.zarr"), sigma=1.0, rho=2.0, chunk=(32, 32, 32),
                  outputs=("S", "eig", "excised"), device=DEV, log=lambda *a: None)
    g = zarr.open_group(str(tmp_path / "dst.zarr"), mode="r")
    S1 = st.structure_tensor_3d(vol, 1.0, 2.0, device=DEV)
    assert np.allclose(g["S"][:], S1, rtol=1e-4, atol=1e-6 * np.abs(S1).max())
    val, vec = st.eig_special_3d(S1, device=DEV)
    assert np.allclose(g["val"][:], val, rtol=1e-3, atol=1e-5 * np.abs(val).max())
    assert np.median(np.abs((g["vec"][:] * vec).sum(0))) > 0.999


def test_spacing_isotropic_identity_and_physical_directions():
    """spacing=(1, 1, 1) is the plain tensor. A fibre texture along (x, y, z) = (1, 0, 1) coarsened 2.5x along z:
    the spacing-aware tensor recovers the direction, the naive (voxel-unit) tensor tilts it towards the coarse
    axis (the analytic bias for this geometry is 23 deg)."""
    import math
    rng = np.random.default_rng(0)
    v = torch.as_tensor(rng.normal(size=(40, 48, 48)).astype(np.float32))
    S0 = st.structure_tensor_3d(v, 1.0, 4.0, device="cpu"); S1 = st.structure_tensor_3d(v, 1.0, 4.0, spacing=(1, 1, 1), device="cpu")
    assert torch.equal(S0, S1)
    z, y, x = np.meshgrid(np.arange(100), np.arange(96), np.arange(96), indexing="ij")
    tex = (np.sin(0.3 * (z - x) / math.sqrt(2)) + np.sin(0.3 * y) + 0.3 * rng.normal(size=z.shape)).astype(np.float32)
    va = torch.nn.functional.interpolate(torch.as_tensor(tex)[None, None], size=(40, 96, 96), mode="area")[0, 0]
    truth = torch.tensor([1.0, 0.0, 1.0]) / math.sqrt(2)
    def med_angle(S):
        _, vec = st.eig_special_3d(S, device="cpu")
        c = (vec[:, 8:-8, 16:-16, 16:-16].reshape(3, -1) * truth[:, None]).sum(0).abs().clamp(max=1)
        return float(torch.rad2deg(torch.arccos(c)).median())
    a_sp = med_angle(st.structure_tensor_3d(va, 1.0, 6.0, spacing=(2.5, 1, 1), device="cpu"))
    a_nv = med_angle(st.structure_tensor_3d(va, 1.0, 6.0, device="cpu"))
    assert a_sp < 6 and a_nv > 15, (a_sp, a_nv)
    assert st.to_isotropic(va, (2.5, 1, 1)).shape == (100, 96, 96)


if __name__ == "__main__":
    import tempfile, pathlib
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print("ok", name, flush=True)
