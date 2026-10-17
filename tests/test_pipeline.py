"""The pipeline's pieces without the network (NNLS against scipy, the density modes, the merge rules, the phantoms'
truth) and, when a GPU and the anatomix weights are available, the whole pipeline on a crossing phantom."""
import numpy as np
import pytest
import torch

from shapetensor import merge, separate, tensors
from shapetensor.phantoms import Grid, build, truth_field

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_nnls_matches_scipy():
    from scipy.optimize import nnls
    rng = np.random.default_rng(0)
    A = rng.normal(size=(32, 4)); Y = rng.normal(size=(50, 32))
    X = separate.nnls_batched(torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32), iters=400).numpy()
    ref = np.stack([nnls(A, y)[0] for y in Y])
    assert np.allclose(X, ref, atol=2e-3)


def test_hemisphere_and_modes():
    u = separate.hemisphere(160)
    assert torch.allclose(u.norm(dim=1), torch.ones(160), atol=1e-5) and bool((u[:, 0] >= 0).all())
    d1, d2 = u[10], u[-20]
    sep = torch.rad2deg(torch.arccos((d1 * d2).sum().abs()))
    dens = torch.exp(8 * ((u @ d1) ** 2 - 1)) + 0.6 * torch.exp(8 * ((u @ d2) ** 2 - 1))
    modes, h, n = separate.modes_from_density(dens, u, rel=0.3, min_sep_deg=min(25.0, float(sep) / 2), nmax=3, mass_floor=0.05)
    assert n == 2
    assert float((modes[0] * d1).sum().abs()) > 0.99 and float((modes[1] * d2).sum().abs()) > 0.95


def test_single_field_discounts_an_incoherent_layer():
    # layer 1: a line tensor along x (energy in y and z); layer 2: isotropic. The merge must give x whatever layer 2's activation.
    Z = (1, 1, 1)
    S1 = torch.zeros((6,) + Z); S1[1] = 1.0; S1[2] = 1.0                                  # yy, zz energy: no gradient along x
    S2 = torch.ones((6,) + Z) * torch.tensor([1.0, 1.0, 1.0, 0, 0, 0]).view(6, 1, 1, 1)
    S = torch.stack([S1, S2]); A = torch.tensor([0.2, 5.0]).view(2, 1, 1, 1)
    d, gap, w = merge.single_field(S.to(DEV), A.to(DEV))
    assert abs(float(d[2, 0, 0, 0])) > 0.999 and abs(float(w[0, 0, 0]) - 5.2) < 1e-5


def test_multi_field_orders_by_activation_everywhere():
    dirs = torch.zeros(2, 3, 2, 1, 1); dirs[0, 0] = 1.0; dirs[1, 2] = 1.0
    A = torch.tensor([[[[0.1]], [[0.9]]], [[[0.8]], [[0.2]]]])                               # (2, 2, 1, 1): voxel 0 favours pop 2, voxel 1 pop 1
    vecs, w = merge.multi_field(dirs, A)
    assert float(vecs[0, 2, 0, 0, 0]) == 1.0 and float(vecs[0, 0, 1, 0, 0]) == 1.0
    assert bool((w[0] >= w[1]).all())


def test_layer_tensor_direction_of_lines():
    from tests.test_st import _lines
    img, d_true = _lines()
    S, d, gap = tensors.layer_tensor(torch.as_tensor(img, device=DEV), 1.0, 4.0)
    inner = (slice(None), slice(12, -12), slice(8, -8), slice(12, -12))
    cos = (d[inner] * torch.as_tensor(d_true, device=DEV)[:, None, None, None]).sum(0).abs()
    assert float(torch.rad2deg(torch.arccos(cos.clamp(max=1))).median()) < 3.0 and float(gap[inner[1:]].median()) > 0.3


def test_phantoms_build_with_truth():
    g = Grid(core=(32, 16, 32), margin=(8, 8, 8))
    for name, kw in (("bundle", {"angle": 30.0}), ("crossing", {"cross_deg": 60.0}), ("interface", {"step": 15.0})):
        img, tr, meta = build(name, 0, g, device=DEV, **kw)
        assert img.shape == g.shape and img.dtype == np.uint8
        t1 = truth_field(tr, "dir1", g.shape[1])
        assert t1.shape == (3,) + g.shape and torch.isfinite(t1).all()
        if name == "crossing":
            assert int(tr["npop"].max()) == 2 and np.isfinite(tr["dir2"]).all()


@pytest.mark.skipif(DEV != "cuda", reason="the pipeline needs a GPU")
def test_pipeline_on_a_crossing():
    pytest.importorskip("anatomix")
    from shapetensor.pipeline import apply, calibrate
    from shapetensor.validate import ANGLES
    g = Grid(core=(64, 32, 64), margin=(32, 32, 32))
    cal_imgs = [torch.as_tensor(build("bundle", 100 + k, g, angle=float(a), device=DEV)[0], device=DEV).float() for k, a in enumerate(ANGLES[::2])]
    cal = calibrate(cal_imgs, margin=32, device=DEV)
    img, tr, _ = build("crossing", 0, g, cross_deg=90.0, device=DEV)
    out = apply(cal, torch.as_tensor(img, device=DEV).float(), device=DEV)
    N = out["modes"].shape[0]
    assert 1 <= N <= 3 and out["vec"].shape == (3,) + g.shape and out["vecs"].shape == (N, 3) + g.shape
    assert torch.allclose(out["vec"].norm(dim=0), torch.ones(g.shape, device=DEV), atol=1e-3)
    assert bool((out["weights"][:-1] >= out["weights"][1:]).all()) if N > 1 else True
    assert out["tissue_mask"].dtype == torch.bool and 0.0 < out["tau"] < 5.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
