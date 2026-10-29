"""Cluster training pieces: exact faster targets, the pooled blur, warm starts with grown channels, the producer, checkpoints."""
import math
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as Fn

DEV = "cpu"


def test_separable_pool_equals_blur_then_average():
    from shapetensor.filters import gaussian
    from shapetensor.learn.targets import _pool
    x = torch.rand(5, 40, 36, 44)
    ref = Fn.avg_pool3d(gaussian(x, 5.0, truncate=3.0)[None], 4)[0]
    assert torch.allclose(_pool(x, 5.0, 4), ref, atol=1e-5)


def test_fraction_target_equals_the_slot_targets_fraction():
    from shapetensor.learn.scene import draw_scene
    from shapetensor.learn.targets import fraction_target, target
    s, d = draw_scene((48, 48, 48), seed=16, device=DEV, strength=0.3, canvas=64)
    assert torch.allclose(fraction_target(s, d), target(s, d)["fraction"].float(), atol=1e-5)


def test_fast_queries_are_close_to_exact():
    from shapetensor.learn.query import draw_queries, sample_density
    F = torch.rand(15, 64, 64, 64); F = torch.nn.functional.avg_pool3d(F[None], 5, 1, 2)[0]
    pts, sig = draw_queries((64, 64, 64), 64, 4, torch.Generator().manual_seed(0), DEV, sigma_range=(1.5, 10.0))
    a, b = sample_density(F, pts, sig, fast=False), sample_density(F, pts, sig, fast=True)
    assert float((a - b).norm() / a.norm()) < 0.05


def test_closed_form_density_matches_quadrature_for_lines_and_sheets():
    from shapetensor.learn.generator import line_tensor, sheet_tensor
    from shapetensor.learn.query import axial_density_sh
    from shapetensor.learn.sh import SHBasis
    d = Fn.normalize(torch.randn(50, 3), dim=1)
    S = torch.cat([line_tensor(d), sheet_tensor(d), 0.7 * line_tensor(d) + 0.3 * torch.tensor([1, 1, 1, 0, 0, 0.0]) / 3]).T   # (6, 150)
    c, ok = axial_density_sh(S, 20.0, 8)
    assert bool(ok.all())
    b = SHBasis(8, 600, DEV); gz, gy, gx = b.grid.unbind(1)
    quad = torch.stack([gz * gz, gy * gy, gx * gx, 2 * gz * gy, 2 * gz * gx, 2 * gy * gx], 1)
    e = torch.exp(20.0 * (quad @ S - 1)); e = e / e.mean(0, keepdim=True)
    assert float((c - b.proj @ e).norm() / (b.proj @ e).norm()) < 2e-3


def test_downsampling_blur_equals_the_depthwise_convolution():
    from shapetensor.learn.model import Down
    d = Down(6, 12); x = torch.randn(1, 6, 20, 24, 16)
    xm = Fn.max_pool3d(Fn.pad(x, (1,) * 6, mode="replicate"), 3, stride=1)
    ref = d.proj(Fn.conv3d(Fn.pad(xm, (1,) * 6, mode="replicate"), d.k.expand(6, 1, 4, 4, 4), stride=2, groups=6))
    assert torch.allclose(d(x), ref, atol=1e-5)


def test_warm_start_with_a_grown_trunk_computes_the_old_features():
    from shapetensor.learn.dataflow import load_partial
    from shapetensor.learn.model import PopulationNet
    old = PopulationNet(widths=(8, 16, 32), head="lobes", context_level=False)
    new = PopulationNet(widths=(8, 16, 32), head="lobes", context_level=True)
    load_partial(new, old.state_dict(), log=lambda m: None)
    x = torch.rand(1, 1, 48, 48, 48)
    assert torch.allclose(old.features(x)[2], new.features(x)[2], atol=1e-5)


def test_producer_makes_pairs_in_the_background():
    from shapetensor.learn.dataflow import Producer
    made = []
    def make(seed, size, device, strength, gen):
        made.append(seed); return torch.full((2, 2), float(seed)), {"t": torch.zeros(1)}
    p = Producer(make, 8, DEV, 100, {"strength": 1.0, "gen": None}, maxq=4)
    t0 = time.time(); got = []
    while len(got) < 6 and time.time() - t0 < 10:
        got += p.take(2)
    p.close()
    assert len(got) >= 6 and [int(i[0][0, 0]) for i in got[:3]] == [100, 101, 102]


def test_checkpointer_signals_and_timer():
    import os, signal
    from shapetensor.learn.dataflow import Checkpointer
    c = Checkpointer(minutes=1e-4)
    time.sleep(0.02); assert c.due()
    os.kill(os.getpid(), signal.SIGUSR1); time.sleep(0.05)
    assert c.stop and c.due()
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)
