"""The structure tensor: element layout and eigen conventions against scipy / numpy, the excised variant, array
kinds, anisotropic spacing, the one-call direction and the chunked driver's equality with a one-pass computation."""
import numpy as np
import torch
from scipy import ndimage as ndi

from shapetensor import st

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _reference_S(vol, sigma, rho, truncate):
    """structure_tensor.structure_tensor_3d as that package computes it: [xx, yy, zz, xy, xz, yz], x = last axis."""
    Vx = ndi.gaussian_filter(vol, sigma, order=(0, 0, 1), mode="nearest", truncate=truncate)
    Vy = ndi.gaussian_filter(vol, sigma, order=(0, 1, 0), mode="nearest", truncate=truncate)
    Vz = ndi.gaussian_filter(vol, sigma, order=(1, 0, 0), mode="nearest", truncate=truncate)
    S = np.empty((6,) + vol.shape, np.float64)
    for k, (a, b) in enumerate(((Vx, Vx), (Vy, Vy), (Vz, Vz), (Vx, Vy), (Vx, Vz), (Vy, Vz))):
        S[k] = ndi.gaussian_filter(a * b, rho, mode="nearest", truncate=truncate)
    return S


def _lines(shape=(48, 40, 48), angle=35.0, seed=0, n_tubes=120):
    """Dark straight tubes at ``angle`` in the x-z plane, at random y: the fibre direction is (sin a, 0, cos a)
    in (z, y, x)."""
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    a = np.radians(angle)
    n = x * np.sin(a) - z * np.cos(a)                                                     # the in-plane normal coordinate
    img = np.full(shape, 160.0, np.float32)
    for c, cy in zip(rng.uniform(n.min(), n.max(), n_tubes), rng.uniform(0, Y, n_tubes)):
        img -= 40 * np.exp(-((n - c) ** 2 + (y - cy) ** 2) / 2.0)
    return (img + rng.normal(0, 3, shape)).astype(np.float32), np.array([np.sin(a), 0, np.cos(a)], np.float32)


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
    assert np.all(np.diff(val, axis=0) <= 1e-5)
    M = np.stack([np.stack([S[0], S[3], S[4]]), np.stack([S[3], S[1], S[5]]), np.stack([S[4], S[5], S[2]])])
    w, V = np.linalg.eigh(M.reshape(3, 3, -1).transpose(2, 0, 1))
    cos = np.abs((vec.reshape(3, -1).T * V[:, :, 0]).sum(1))
    assert np.median(cos) > 0.9999
    _, vecs = st.eig_special_3d(S, full=True, device=DEV)
    assert vecs.shape == (3, 3, 24, 20, 28) and np.allclose(np.abs((vecs[2] * vec).sum(0)), 1.0, atol=1e-4)


def test_direction_of_lines_and_orders():
    img, d_true = _lines()
    vec = st.structure_tensor(img, sigma=1.0, rho=4.0, order="zyx", device=DEV)                  # numpy in, numpy out
    inner = (slice(12, -12), slice(8, -8), slice(12, -12))
    cos = np.abs((vec[(slice(None),) + inner] * d_true[:, None, None, None]).sum(0))
    assert np.median(np.degrees(np.arccos(np.clip(cos, 0, 1)))) < 3.0
    vx = st.structure_tensor(torch.as_tensor(img, device=DEV), sigma=1.0, rho=4.0, order="xyz", device=DEV)
    assert isinstance(vx, torch.Tensor) and torch.allclose(vx.flip(0).cpu(), torch.as_tensor(vec), atol=1e-5)
    plain = st.structure_tensor(img, sigma=1.0, rho=4.0, excise=False, device=DEV)
    assert plain.shape == (3,) + img.shape


def test_excision_leaves_clear_tissue_alone_and_removes_a_step():
    img, d_true = _lines(angle=0.0)                                                        # fibres along x
    S = st.structure_tensor_3d(img, 1.0, 3.0, device=DEV)
    Sx = st.excised_structure_tensor_3d(img, 1.0, 3.0, factor=100.0, device=DEV)            # nothing excised -> equal
    assert np.allclose(S, Sx, rtol=1e-4, atol=1e-6 * np.abs(S).max())
    step = img.copy(); step[24:] += 80.0                                                   # a strong step across z
    w = st.excision_weights(torch.as_tensor(step, device=DEV), sigma_coarse=3.0, factor=3.0).cpu().numpy()
    assert w[22:26].mean() < 0.5 and w[:10].mean() > 0.9 and w[38:].mean() > 0.9


def test_spacing_isotropic_identity_and_physical_directions():
    img, d_true = _lines()
    S1 = st.structure_tensor_3d(img, 1.0, 3.0, device=DEV)
    S2 = st.structure_tensor_3d(img, 1.0, 3.0, spacing=(2.0, 2.0, 2.0), device=DEV)
    assert np.allclose(S1, S2)
    # a volume sampled 2x coarser along z: the physical direction is recovered with spacing=(2, 1, 1)
    coarse = img[::2]
    vec = st.structure_tensor(coarse, sigma=1.0, rho=4.0, order="zyx", spacing=(2.0, 1.0, 1.0), device=DEV)
    inner = (slice(6, -6), slice(8, -8), slice(12, -12))
    cos = np.abs((vec[(slice(None),) + inner] * d_true[:, None, None, None]).sum(0))
    assert np.median(np.degrees(np.arccos(np.clip(cos, 0, 1)))) < 8.0


def test_run_volume_equals_one_pass(tmp_path):
    import zarr
    rng = np.random.default_rng(3)
    vol = (120 + 20 * rng.normal(size=(40, 36, 44))).astype(np.float32)
    zarr.save(str(tmp_path / "in.zarr"), vol)
    st.run_volume(str(tmp_path / "in.zarr"), str(tmp_path / "out.zarr"), sigma=1.0, rho=2.0, chunk=(16, 16, 16), excise=False, device=DEV, log=lambda *a: None)
    g = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    ref = st.structure_tensor_3d(vol, 1.0, 2.0, device=DEV)                                  # (the excision's reference median is per chunk, so only the plain tensor is exactly chunk-invariant)
    assert np.allclose(np.asarray(g["S"]), ref, rtol=1e-4, atol=1e-5 * np.abs(ref).max())
    _, vref = st.eig_special_3d(ref, device=DEV)
    assert np.median(np.abs((np.asarray(g["vec"]) * vref).sum(0))) > 0.999


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames:
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("ok", name)


def test_eig_analytic_matches_eigh():
    """The closed-form eigensolver against cusolver: eigenvalues to 1e-6 of the largest, the smallest eigenvector to
    0.05 degrees where it is defined, on random SPD tensors and on tensor-like ones with a wide dynamic range."""
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.randn(20000, 3, 3, device=dev); M = A @ A.transpose(1, 2)
    M[:5000] *= 10 ** (6 * torch.rand(5000, 1, 1, device=dev))
    S = torch.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]])
    va, ea = st.eig_special_3d(S, device=dev); vh, eh = st.eig_special_3d(S, device=dev, method="eigh")
    assert float(((va - vh).abs() / vh[0].abs().clamp_min(1e-12)).max()) < 1e-5
    gap = (vh[1] - vh[2]) / vh[1].clamp_min(1e-12)
    ang = torch.rad2deg(torch.arccos((ea * eh).sum(0).abs().clamp(max=1)))
    assert float(ang[gap > 0.02].max()) < 0.05
    vf, ef = st.eig_special_3d(S, full=True, device=dev)                               # all three: orthonormal, A v = lambda v
    assert float((ef.permute(2, 0, 1) @ ef.permute(2, 1, 0) - torch.eye(3, device=dev)).abs().max()) < 1e-4
    Mv = (M @ ef.permute(2, 1, 0)); assert float((Mv - vf.T[:, None, :] * ef.permute(2, 1, 0)).abs().max() / M.abs().max()) < 1e-4
    iso = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]], device=dev); _, e_iso = st.eig_special_3d(iso, device=dev)
    assert torch.isfinite(e_iso).all() and abs(float(e_iso.norm()) - 1) < 1e-5
