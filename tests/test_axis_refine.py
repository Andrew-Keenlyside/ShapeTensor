import math

import torch

from shapetensor.filters import gaussian
from shapetensor.learn.axis_refine import refine_axes


def _rods(n, axes, spacing=6.0, radius=1.2):
    """Bright rods along each of ``axes`` on a square lattice (the lattices offset), ``(n, n, n)``."""
    g = torch.arange(n, dtype=torch.float32)
    X = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1) - n / 2
    img = torch.zeros(n, n, n)
    for i, a in enumerate(axes):
        a = torch.nn.functional.normalize(torch.tensor(a, dtype=torch.float32), dim=0)
        e1 = torch.nn.functional.normalize(torch.linalg.cross(a, torch.tensor([0.3, 0.5, 0.8])), dim=0); e2 = torch.linalg.cross(a, e1)
        u, v = X @ e1 + i * spacing / 2, X @ e2 + i * spacing / 2
        d = torch.sqrt((u - spacing * torch.round(u / spacing)) ** 2 + (v - spacing * torch.round(v / spacing)) ** 2)
        img = torch.maximum(img, (radius + 0.5 - d).clamp(0, 1))
    return img


def _deg(u, v):
    return math.degrees(math.acos(min(1.0, abs(float(torch.nn.functional.normalize(u, dim=0) @ torch.nn.functional.normalize(v, dim=0))))))


def _tilt(a, deg):
    a = torch.nn.functional.normalize(torch.tensor(a, dtype=torch.float32), dim=0)
    p = torch.nn.functional.normalize(torch.linalg.cross(a, torch.tensor([0.6, -0.2, 0.77])), dim=0)
    return math.cos(math.radians(deg)) * a + math.sin(math.radians(deg)) * p


def _st_axis(img, p, sigma):
    G = torch.stack(torch.gradient(gaussian(img[None, None], 1.0)[0, 0]), -1)
    S = torch.stack([gaussian((G[..., i] * G[..., j])[None, None], sigma)[0, 0][p] for i in range(3) for j in range(3)]).view(3, 3)
    return torch.linalg.eigh(S)[1][:, 0]


def test_a_seed_15_deg_off_a_single_bundle_comes_back_to_the_structure_tensors_axis():
    a = [0.2, 0.5, 0.84]; img = gaussian(_rods(40, [a], spacing=8.0, radius=1.8)[None, None], 0.7)[0, 0]
    pts = torch.tensor([[20.0, 20.0, 20.0], [16.0, 22.0, 19.0]])
    seed = _tilt(a, 15.0)[None, None].expand(2, 1, 3).clone()
    out = refine_axes(img, pts, 3.0, seed, torch.ones(2, 1, dtype=torch.bool), iters=4)
    for q in range(2):
        st = _st_axis(img, tuple(int(v) for v in pts[q]), 3.0)
        assert _deg(out[q, 0], st) < 0.5 and _deg(out[q, 0], torch.tensor(a)) < 3.0


def test_a_crossing_keeps_both_bundles_each_refined_to_its_own_axis():
    a, b = [0.0, 0.2, 0.98], [0.0, 0.98 * math.sin(math.radians(60)) + 0.2 * math.cos(math.radians(60)), 0.98 * math.cos(math.radians(60)) - 0.2 * math.sin(math.radians(60))]
    img = gaussian(_rods(48, [a, b], spacing=8.0, radius=1.8)[None, None], 0.7)[0, 0]
    pts = torch.tensor([[24.0, 24.0, 24.0], [21.0, 26.0, 23.0]])
    seed = torch.stack([_tilt(a, 10.0), _tilt(b, 10.0)])[None].expand(2, 2, 3).clone()
    out = refine_axes(img, pts, 3.0, seed, torch.ones(2, 2, dtype=torch.bool), iters=4)
    for q in range(2):                                                                             # (a structure tensor: one blended direction)
        assert _deg(out[q, 0], torch.tensor(a)) < 3.0 and _deg(out[q, 1], torch.tensor(b)) < 3.0


def test_dead_populations_keep_their_input_axis():
    img = _rods(32, [[0.0, 0.0, 1.0]]); pts = torch.tensor([[16.0, 16.0, 16.0]])
    seed = torch.tensor([[[0.0, 0.1, 1.0], [1.0, 0.0, 0.0]]])
    out = refine_axes(img, pts, 2.0, seed, torch.tensor([[True, False]]))
    assert torch.allclose(out[0, 1], seed[0, 1])
