"""The dev pipeline's components, on the CPU and without the networks: the shape tensor recovers known directions, the
dev-vit engine's tiling puts every voxel back where it belongs and its shift average cancels a patch-grid imprint,
the sliceTCA engine wrapper smooths, the promoted fibre-set functions match the prototypes exactly, the chunked driver
runs through its new hooks, and calibration is reproducible."""
import math

import numpy as np
import torch

from shapetensor import tensors

DEV = "cpu"
torch.manual_seed(0)


def _unit(v):
    v = torch.as_tensor(v, dtype=torch.float32); return v / v.norm()


def _angle(a, b):
    return math.degrees(math.acos(min(1.0, abs(float((a / a.norm()) @ (b / b.norm()))))))


def _features_along(d, shape, n=6, seed=0):
    """``n`` channels of sinusoids that vary only across ``d`` (z, y, x): features constant along the fibre."""
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij")
    p = torch.stack([zz, yy, xx])
    a = torch.linalg.qr(torch.stack([d, torch.randn(3, generator=g), torch.randn(3, generator=g)]).T)[0]
    e1, e2 = a[:, 1], a[:, 2]                                                                  # an orthonormal basis of the plane across d
    F = []
    for c in range(n):
        th = 2 * math.pi * float(torch.rand(1, generator=g))
        e = math.cos(th) * e1 + math.sin(th) * e2
        lam = 5.0 + 4.0 * float(torch.rand(1, generator=g))
        F.append(torch.sin(2 * math.pi * (p * e[:, None, None, None]).sum(0) / lam + 6.28 * float(torch.rand(1, generator=g))))
    return torch.stack(F)


def test_shape_tensor_recovers_direction():
    d = _unit([1.0, 0.5, 0.2]); shape = (40, 40, 40)
    F = _features_along(d, shape)
    P = tensors.feature_products(F, 1.0, 3.0)
    _, dk, gk = tensors.shape_tensor(P, torch.ones(shape), rho=5.0, truncate=3.0)
    c = dk[:, 20, 20, 20]
    assert _angle(c, d) < 3.0, _angle(c, d)
    assert float(gk[20, 20, 20]) > 0.5


def test_shape_tensor_per_population():
    d1, d2 = _unit([1.0, 0.0, 0.3]), _unit([0.2, 0.3, 1.0]); shape = (32, 32, 64)
    F1, F2 = _features_along(d1, shape, seed=1), _features_along(d2, shape, seed=2)
    left = torch.zeros(shape); left[..., :32] = 1.0
    F = F1 * left + F2 * (1 - left)
    P = tensors.feature_products(F, 1.0, 3.0)
    _, da, _ = tensors.shape_tensor(P, left, rho=4.0, truncate=3.0)
    _, db, _ = tensors.shape_tensor(P, 1 - left, rho=4.0, truncate=3.0)
    assert _angle(da[:, 16, 16, 14], d1) < 5.0 and _angle(db[:, 16, 16, 50], d2) < 5.0


class _StubNet:
    """In place of the ViT: the input repeated over 32 channels, plus (optionally) a pattern tied to the window's patch
    grid -- what the real decoder imprints."""

    def __init__(self, grid_amp=0.0):
        self.grid_amp = grid_amp

    def __call__(self, x):
        y = x.repeat(1, 32, 1, 1, 1).clone()
        if self.grid_amp:
            k = torch.arange(x.shape[-1], dtype=torch.float32)
            y += self.grid_amp * torch.cos(2 * math.pi * k / 8)[None, None, None, None, :]
        return y


def _engine(monkeypatch_net, **kw):
    from shapetensor.features import devvit
    devvit._NET["cpu"] = monkeypatch_net
    return devvit.DevVitEngine(device="cpu", l2=False, amp=False, out_dtype=torch.float32, **kw)


def test_devvit_tiling_puts_voxels_back():
    for shape in [(70, 50, 90), (20, 20, 20), (64, 64, 64)]:
        vol = torch.rand(shape) * 255
        eng = _engine(_StubNet())
        out = eng(vol)
        assert out.shape == (32,) + shape
        assert torch.allclose(out[0], vol / 255, atol=1e-5), shape


def test_devvit_shift_average_cancels_patch_grid():
    vol = torch.rand((40, 40, 72)) * 255
    one = _engine(_StubNet(grid_amp=0.3), shifts=((0, 0, 0),))(vol)
    two = _engine(_StubNet(grid_amp=0.3), shifts=((0, 0, 0), (4, 4, 4)))(vol)
    err1 = float((one[0] - vol / 255).abs().max()); err2 = float((two[0] - vol / 255).abs().max())
    assert err1 > 0.25 and err2 < 1e-4, (err1, err2)


def test_smoothed_engine_reduces_noise():
    from shapetensor.smoothing import SmoothedEngine
    shape = (32, 32, 32)
    clean = _features_along(_unit([0.0, 0.0, 1.0]), shape, n=8) * 0.5
    noisy = clean + 0.3 * torch.randn_like(clean)

    class Base:
        name, n_channels, margin = "stub", 8, 4

        def __call__(self, vol):
            return noisy

        def describe(self):
            return {"engine": self.name}
    eng = SmoothedEngine(Base(), ranks=(2, 2, 2), spacing=1, window=16)
    out = eng(torch.zeros(shape))
    assert out.shape == noisy.shape and eng.margin == 4 + 2
    assert float(((out - clean) ** 2).mean()) < 0.6 * float(((noisy - clean) ** 2).mean())


def _synthetic_sets():
    """Window populations of two crossing straight groups on a 5 x 1 x 5 window grid (stride 8), plus a short stray set."""
    wins, pops = {}, []
    d1, d2 = _unit([1.0, 0.0, 0.0]), _unit([0.0, 0.0, 1.0])
    for iz in range(5):
        for ix in range(5):
            key = (iz, 0, ix); wins[key] = {"centre": [16 + 8 * iz, 16, 16 + 8 * ix], "pops": []}
            if ix == 2:
                pops.append({"win": key, "dir": d1}); wins[key]["pops"].append(len(pops) - 1)
            if iz == 2:
                pops.append({"win": key, "dir": d2}); wins[key]["pops"].append(len(pops) - 1)
    pops.append({"win": (0, 0, 0), "dir": _unit([1.0, 1.0, 0.0])}); wins[(0, 0, 0)]["pops"].append(len(pops) - 1)
    labels = [0] * (len(pops) - 1) + [1]
    return pops, wins, labels


def test_fibre_set_functions_match_the_prototypes():
    from shapetensor import fibre_sets as FS
    from shapetensor.experimental import fibre_sets as EX
    pops, wins, labels = _synthetic_sets()
    assert FS.split_crossings(pops, wins, labels, stride=8.0) == EX.split_crossings(pops, wins, labels, stride=8.0)
    assert FS.prune_short(pops, wins, labels, 32.0) == EX.prune_short(pops, wins, labels, 32.0)
    assert FS.set_extents(pops, wins, labels) == EX.set_extents(pops, wins, labels)
    new, n = FS.split_crossings(pops, wins, labels, stride=8.0)
    assert n == 1                                                                      # the cross is two fibre sets
    kept, rep = FS.prune_short(pops, wins, new, 32.0)
    assert kept[-1] == -1 and rep["sets_dropped"] == 1                                 # the stray single window is not a fibre course


def test_run_volume_hooks(tmp_path):
    import zarr
    from shapetensor.pipeline import run_volume
    src = str(tmp_path / "vol.zarr"); vol = (np.random.default_rng(0).random((40, 40, 40)) * 255).astype(np.uint8)
    zarr.open_array(src, mode="w", shape=vol.shape, chunks=(20, 20, 20), dtype="uint8")[:] = vol
    xdir = torch.tensor([0.0, 0.0, 1.0])                                                 # (z, y, x): the x axis

    class Cal:
        tau, smooth, engine, orientation_bins_occupied, readout_overlap = 2.0, 1, "stub", 24, 1.0

        def features(self, blk):
            return blk[None]

    def cal_fn(v, tissue=None, **kw):
        return Cal()

    def apply_fn(cal, blk, tis=None, **kw):
        sh = tuple(blk.shape)
        T = torch.zeros((6,) + sh); T[1] = 1.0; T[2] = 1.0                               # [xx, yy, zz, ...]: smallest along x
        vec = xdir.view(3, 1, 1, 1).expand(3, *sh).clone()
        return {"weight": torch.full(sh, 2.0), "tensor": T, "vec": vec, "vec_gap": torch.ones(sh), "vecs": vec[None], "weights": torch.full((1,) + sh, 2.0),
                "n_populations": 1}
    for blend in (0, 4):
        dst = str(tmp_path / f"out{blend}.zarr")
        run_volume(src, dst, chunk=(20, 20, 20), nmax=2, blend=blend, device="cpu", calibrate_fn=cal_fn, apply_fn=apply_fn, engine_margin=4,
                   pipeline="test", log=lambda *a: None, rho=2.0, truncate=2.0)
        g = zarr.open_group(dst, mode="r")
        v = np.asarray(g["vec"][:], np.float32); w = np.asarray(g["weight"][:], np.float32)
        assert g.attrs["pipeline"] == "test"
        assert np.allclose(np.abs(v[0]), 1.0, atol=1e-2), blend                              # written in (x, y, z): the x component
        assert np.allclose(w, 1.0, atol=1e-2), blend                                         # weight / tau
        assert np.allclose(np.asarray(g["weights"][1]), 0.0)                                  # the second row: no population


def test_calibration_is_reproducible():
    from shapetensor.pipeline import calibrate
    shape = (48, 48, 48)
    d = _unit([0.3, 0.2, 1.0])
    F = _features_along(d, shape, n=8, seed=3)
    zz, yy, xx = torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij")
    v = 128 + 40 * torch.sin(2 * math.pi * (0.9 * yy - 0.2 * zz) / 6) + 5 * torch.randn(shape, generator=torch.Generator().manual_seed(1))
    kw = dict(features=F, sigma=1.0, rho=4.0, truncate=3.0, smooth=1, margin=6, n_sample=20_000, device=DEV)
    a, b = calibrate(v, None, seed=0, **kw), calibrate(v, None, seed=0, **kw)
    assert a.tau == b.tau and torch.equal(a.dictionary.D, b.dictionary.D)
