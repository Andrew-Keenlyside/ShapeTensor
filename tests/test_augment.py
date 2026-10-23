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
