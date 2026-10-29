import math

import torch

from shapetensor.learn.entity_infer import sharpen_density, sphere_dirs


def _lobes(dirs, axes, kappa=7.5, w=None):
    w = w or [1.0] * len(axes)
    f = sum(wi * torch.exp(-kappa * (1 - (dirs @ torch.nn.functional.normalize(torch.tensor(a), dim=0)) ** 2)) for wi, a in zip(w, axes))
    return (0.05 + f)[None]                                                                        # (1, D) on a floor


def _maxima(f, dirs, k=8):
    c = (dirs @ dirs.T).abs(); c.fill_diagonal_(-1); nb = c.topk(k, dim=1).indices
    return ((f[0][:, None] > f[0][nb]).all(1) & (f[0] > 0.2 * f[0].max())).nonzero()[:, 0]


def _rot(deg):
    t = math.radians(deg)
    return [math.cos(t), math.sin(t), 0.0]


def test_two_model_width_lobes_30_deg_apart_on_200_directions_become_two_maxima():
    dirs = sphere_dirs(200)                                                                        # (the readout's grid, ~10 deg)
    assert len(_maxima(sharpen_density(_lobes(dirs, [_rot(0), _rot(30)]), dirs, iters=300), dirs)) == 2
    assert len(_maxima(sharpen_density(_lobes(dirs, [_rot(0), _rot(25)]), dirs, iters=1000, kappa_to=None), dirs)) == 1   # its limit


def test_two_model_width_lobes_25_deg_apart_on_400_directions_become_two_maxima():
    dirs = sphere_dirs(400)
    f = _lobes(dirs, [_rot(0), _rot(25)])
    assert len(_maxima(f, dirs)) == 1                                                              # one lobe at the model's width
    m = _maxima(sharpen_density(f, dirs, iters=300, kappa_to=None), dirs)
    assert len(m) == 2
    got = sorted(math.degrees(math.atan2(float(dirs[i, 1]), float(dirs[i, 0]))) % 180 for i in m)
    assert abs(got[0] - 0) % 180 < 6 or abs(got[0] - 180) < 6
    assert abs(got[1] - 25) < 6


def test_a_single_lobe_stays_one_and_mass_is_kept():
    dirs = sphere_dirs(200)
    f = _lobes(dirs, [[0.3, 0.5, 0.8]])
    s = sharpen_density(f, dirs, iters=60)
    assert len(_maxima(s, dirs)) == 1
    h = (f - torch.quantile(f, 0.05, dim=1, keepdim=True)).clamp_min(0)
    assert abs(float(s.sum()) / float(h.sum()) - 1) < 0.02
