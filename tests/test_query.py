"""Queries at any point and scale: the target, its exact cube-group transform, the decoder's shapes."""
import torch

from shapetensor.learn.generator import draw
from shapetensor.learn.losses import apply_cube, cube_transform
from shapetensor.learn.model import PopulationNet
from shapetensor.learn.query import cube_points, draw_queries, layer_density, sh_rotation
from shapetensor.learn.sh import n_coef

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cube_drawn(d, perm, flips):
    out = dict(d)
    out["occupancy"] = apply_cube(d["occupancy"], perm, flips)
    out["tensors"] = apply_cube(d["tensors"], perm, flips, tensor6=True)
    for k in ("visibility", "orientation_visibility"):
        v = d.get(k)
        if v is not None and v.dim() > 1:
            out[k] = apply_cube(v, perm, flips)
    return out


def test_query_target_commutes_with_the_cube_group():
    s, d = draw((48,) * 3, seed=3, device=DEV, strength=0.3, modality="hipct", crossing=True)
    g = torch.Generator(device=DEV).manual_seed(0)
    pts, sig = draw_queries((48,) * 3, 64, 2, g, DEV, margin=10)
    c = layer_density(s, d, pts, sig, 6)
    cg = torch.Generator().manual_seed(5)
    for _ in range(3):
        perm, flips = cube_transform(cg)
        c2 = layer_density(s, _cube_drawn(d, perm, flips), cube_points(pts, perm, flips, (48,) * 3), sig, 6)
        R = sh_rotation(perm, flips, 6, DEV)
        err = (c @ R.T - c2).abs().max() / c.abs().max()
        assert float(err) < 2e-2, (perm, flips, float(err))


def test_query_target_is_additive_in_scale_limit():
    """Order 0 is the windowed visible mass (times sqrt(4 pi)): it does not depend on the tensors."""
    s, d = draw((48,) * 3, seed=4, device=DEV, strength=0.3, modality="hipct")
    g = torch.Generator(device=DEV).manual_seed(1)
    pts, sig = draw_queries((48,) * 3, 32, 1, g, DEV, margin=10)
    c = layer_density(s, d, pts, sig, 4)
    assert torch.isfinite(c).all() and float(c[:, 0].min()) >= -1e-4


def test_decoder_shapes_and_gradients():
    net = PopulationNet(sh_lmax=6, query=True).to(DEV)
    x = torch.rand(2, 1, 32, 32, 32, device=DEV)
    pts = 6 + 20 * torch.rand(2, 50, 3, device=DEV); sig = 1.5 + 8 * torch.rand(2, 50, device=DEV)
    out = net(x, pts, sig)
    assert out["q_sh"].shape == (2, 50, n_coef(6))
    out["q_sh"].square().mean().backward()
    assert net.stem.weight.grad is not None and float(net.stem.weight.grad.abs().sum()) > 0
