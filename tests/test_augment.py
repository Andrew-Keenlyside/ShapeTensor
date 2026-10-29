"""Contrast routes: range, geometry kept, non-monotone curves keep their range, missing lines and blocks, draw-time routes."""
import numpy as np
import pytest
import torch

from shapetensor.learn.augment import apply_lut, contrast_route, missing_blocks, missing_lines, random_lut

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_routes_stay_in_range_and_differ():
    x = torch.rand(48, 48, 48, device=DEV)
    ys = [contrast_route(x, np.random.default_rng(s), 1.0) for s in range(6)]
    for y in ys:
        assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1 and torch.isfinite(y).all()
    assert len({round(float(y.mean()), 3) for y in ys}) > 3


def test_curves_keep_their_range():
    rng = np.random.default_rng(0)
    for mono in (True, False):
        for _ in range(20):
            xk, yk = random_lut(rng, mono)
            assert np.abs(np.diff(yk)).sum() >= 0.6
            if mono:
                assert (np.diff(yk) >= 0).all() or (np.diff(yk) <= 0).all()
    t = torch.linspace(0, 1, 101, device=DEV)
    assert torch.allclose(apply_lut(t, np.array([0.0, 1.0]), np.array([1.0, 0.0])), 1 - t, atol=1e-6)


def test_missing_lines_and_blocks_touch_little_of_the_block():
    x = torch.rand(64, 64, 64, device=DEV)
    rec = {}
    y = missing_lines(x, np.random.default_rng(1), rec); z = missing_blocks(x, np.random.default_rng(2), rec)
    for v in (y, z):
        frac = float((v != x).float().mean())
        assert 0 < frac < 0.05


def test_draw_time_routes_run_and_are_recorded():
    from shapetensor.learn.scene import draw_scene
    s, d = draw_scene((32, 32, 32), seed=3, device=DEV, strength=0.3, canvas=64, p_blend=1.0, p_aniso=1.0, p_scramble=1.0)
    assert torch.isfinite(d["image"].float()).all()
    assert len(d["record"]["routes"]) >= 3


# ---- acquisition routes (per-axis resolution, k-space, resampling, bit depth, compression, library)

def _ball(n=48):
    g = torch.stack(torch.meshgrid(*[torch.arange(n, device=DEV, dtype=torch.float32)] * 3, indexing="ij"))
    return ((g - n / 2).norm(dim=0) < n / 4).float() * 0.8 + 0.1


def test_per_axis_resolution_blurs_unevenly_and_keeps_the_mean():
    from shapetensor.learn.augment import per_axis_resolution
    x = _ball(); rec = {}
    y = per_axis_resolution(x, np.random.default_rng(3), rec)
    assert y.shape == x.shape and abs(float(y.mean() - x.mean())) < 0.02
    f = rec["resolution"][0]; assert max(f) >= 1.2


def test_kspace_motion_keeps_the_object_and_adds_ghosts():
    from shapetensor.learn.augment import kspace_motion
    x = _ball(); y = kspace_motion(x, np.random.default_rng(5), {})
    assert torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1] > 0.6
    assert float((y - x).abs().mean()) > 1e-3


def test_spikes_add_stripes():
    from shapetensor.learn.augment import kspace_spikes
    x = _ball(); y = kspace_spikes(x, np.random.default_rng(1), {})
    assert float((y - x).abs().mean()) > 1e-3 and torch.isfinite(y).all()


def test_resample_artefacts_and_fourier_resize():
    from shapetensor.learn.augment import fourier_resize, resample_artefacts
    x = _ball()
    assert torch.allclose(fourier_resize(x, x.shape), x, atol=1e-4)                               # same size: identity
    for s in range(6):
        y = resample_artefacts(x, np.random.default_rng(s), {}); assert y.shape == x.shape and torch.isfinite(y).all()


def test_quantise_and_compress_are_lossy_but_close():
    from shapetensor.learn.augment import compress, quantise
    x = torch.rand(40, 44, 48, device=DEV) * 0.5 + _ball(48)[:40, :44, :48] * 0.5
    q = quantise(x, np.random.default_rng(0), {}); assert len(torch.unique(q)) <= 64
    for s in range(6):
        rec = {}; y = compress(x, np.random.default_rng(s), rec)
        err = float((y - x).abs().mean())
        assert y.shape == x.shape and 0 < err < 0.15, (rec, err)


def test_library_route_uses_monai():
    pytest.importorskip("monai")
    from shapetensor.learn.augment import library_route
    x = _ball(); names = set()
    for s in range(24):
        rec = {}; y = library_route(x, np.random.default_rng(s), rec); names.add(rec.get("library"))
        assert y.shape == x.shape and torch.isfinite(y).all()
    assert len(names) >= 4


def test_full_routes_with_every_operation():
    from shapetensor.learn import augment as AU
    saved = dict(AU.P)
    try:
        for k in AU.P:
            AU.P[k] = 1.0
        AU.P["nonmono"] = 0.5
        y = AU.contrast_route(torch.rand(48, 48, 48, device=DEV), np.random.default_rng(9), 1.0)
        assert torch.isfinite(y).all() and float(y.min()) >= 0 and float(y.max()) <= 1
    finally:
        AU.P.update(saved)


def test_kspace_ghosting_keeps_the_object():
    from shapetensor.learn.augment import kspace_ghosting
    x = _ball(); y = kspace_ghosting(x, np.random.default_rng(2), {})
    assert torch.corrcoef(torch.stack([x.flatten(), y.flatten()]))[0, 1] > 0.7 and float((y - x).abs().mean()) > 1e-3


def test_curves_are_slope_and_fold_bounded():
    rng = np.random.default_rng(1)
    for _ in range(50):
        xk, yk = random_lut(rng, False)
        dy = np.diff(yk); assert (np.abs(dy) / np.maximum(np.diff(xk), 1e-6)).max() <= 5.0 + 1e-6
        assert int((np.sign(dy[1:]) * np.sign(dy[:-1]) < 0).sum()) <= 2


# ---- backgrounds of either polarity

def _with_background(polarity: str, noise: float):
    g = torch.Generator(device=DEV).manual_seed(0)
    x = 0.3 + 0.4 * torch.rand(64, 64, 64, device=DEV, generator=g)                                # textured tissue
    m = torch.zeros(64, 64, 64, dtype=torch.bool, device=DEV); m[:, :, :28] = True                  # 44 % background
    bgv = 0.02 if polarity == "dark" else 0.98
    x[m] = bgv + noise * torch.randn(int(m.sum()), device=DEV, generator=g)
    return x, m


@pytest.mark.parametrize("polarity", ["dark", "bright"])
@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_background_found_at_either_end(polarity, noise):
    from shapetensor.learn.realdata import background_mask, normalise
    x, m = _with_background(polarity, noise)
    b = background_mask(x)
    assert float((b & m).float().sum() / m.float().sum()) > 0.9 and float((b & ~m).float().mean()) < 0.05
    y, share = normalise(x)
    assert abs(share - 0.56) < 0.08
    t = y[~m]; assert float(t.quantile(0.99) - t.quantile(0.01)) > 0.8                            # the tissue fills the range


def test_no_background_in_pure_tissue():
    from shapetensor.learn.realdata import background_mask
    x = 0.2 + 0.6 * torch.rand(48, 48, 48, device=DEV)
    assert float(background_mask(x).float().mean()) < 0.05


def test_specimen_medium_is_dark_or_bright():
    from shapetensor.learn.scene import draw_scene
    kinds = set()
    for sd in range(12):
        s, d = draw_scene((32, 32, 32), seed=40 + 8 * sd, device=DEV, strength=0.0, canvas=64, p_boundary=1.0, p_warp=0.0, p_affine=0.0)
        if "boundary" in s.record:
            kinds.add(s.record["boundary"][1])
    assert {"dark", "bright"} <= kinds


# ---- sections, optics and missing planes (batch time, mild)

@pytest.mark.parametrize("name", ["missing_slices", "thin_slab", "sections", "ls_stripes", "axial_psf", "shot_noise", "saturate"])
def test_section_and_optics_routes_keep_shape_and_range(name):
    from shapetensor.learn import augment as AU
    fn = getattr(AU, name); x = _ball() * 0.8 + 0.1 * torch.rand(48, 48, 48, device=DEV)
    for s in range(4):
        rec = {}; y = fn(x, np.random.default_rng(s), rec)
        assert y.shape == x.shape and torch.isfinite(y).all() and float(y.min()) >= 0 and float(y.max()) <= 1 + 1e-6
        assert rec and float((y - x).abs().mean()) > 1e-4
        y2 = fn(x, np.random.default_rng(s), {}); assert torch.allclose(y, y2), name                  # reproducible from the numpy draw


def test_missing_slices_and_slabs_touch_whole_planes_only():
    from shapetensor.learn.augment import missing_slices, thin_slab
    x = torch.rand(40, 40, 40, device=DEV)
    for fn in (missing_slices, thin_slab):
        for s in range(6):
            rec = {}; y = fn(x, np.random.default_rng(s), rec); ax = list(rec.values())[0][0]
            changed = (y != x).movedim(ax, 0).flatten(1).any(1)
            assert 1 <= int(changed.sum()) <= 4 and float((y != x).float().mean()) < 0.12


def test_axial_psf_blurs_one_axis_most():
    from shapetensor.learn.augment import axial_psf
    x = torch.rand(48, 48, 48, device=DEV); rec = {}; y = axial_psf(x, np.random.default_rng(0), rec)
    ax = rec["axial_psf"][0]
    d = [float(torch.diff(y, dim=a).abs().mean()) for a in range(3)]
    assert d[ax] == min(d)
