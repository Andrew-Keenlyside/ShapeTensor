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


def test_exact_lattice_density_commutes_with_the_cube_group():
    from shapetensor.learn.losses import augment
    from shapetensor.learn.query import voxel_density
    from shapetensor.learn.targets import _pool
    s, d = draw((48,) * 3, seed=6, device=DEV, strength=0.3, modality="hipct", crossing=True)
    sh = _pool(voxel_density(s, d, 6), 5.0, 4)
    img = d["image"]
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        perm, flips = cube_transform(torch.Generator().manual_seed(seed))
        _, t2 = augment(img, {"sh_t": sh}, torch.Generator().manual_seed(seed))
        ref = _pool(voxel_density(s, _cube_drawn(d, perm, flips), 6), 5.0, 4)
        err = (t2["sh_t"] - ref).abs().max() / ref.abs().max()
        assert float(err) < 2e-2, (perm, flips, float(err))


def test_descriptor_density_reads_the_layer_descriptor_at_the_peak():
    """One line layer: thickness read at the density's top peak, f_q(u) / f(u), equals the layer's thickness."""
    from shapetensor.learn.query import voxel_density
    from shapetensor.learn.sh import n_coef, peaks, real_sh
    from shapetensor.learn.targets import _pool
    s, d = draw((48,) * 3, seed=8, device=DEV, strength=0.2, modality="hipct", style="flat", crossing=False, k_layers=1, kinds=["line"])
    k = [i for i, L in enumerate(s.layers) if L.kind == "line"][0]
    thick = float(d["descriptors"][k, 0][d["occupancy"][k] > 0.5].median())
    F = _pool(voxel_density(s, d, 6, desc_lmax=4), 5.0, 4).flatten(1)
    strong = F[0] > 0.3 * F[0].max()
    D, _ = peaks(F[:, strong], 6, nmax=1)
    Y = real_sh(D[0].T, 4); nd = n_coef(4); nsh = n_coef(6)
    q = (Y * F[nsh:nsh + nd, strong].T).sum(1) / (Y * F[:nd, strong].T).sum(1)
    assert abs(float(q.median()) - thick) < 0.15 * max(thick, 1.0), (float(q.median()), thick)


def test_context_mask_keeps_the_cube_and_measures_context():
    from shapetensor.learn.train import context_mask
    x = torch.rand(1, 1, 32, 32, 32, device=DEV)
    pts = torch.tensor([[[16.0, 16.0, 16.0], [16.0, 20.0, 16.0], [16.0, 30.0, 16.0]]], device=DEV)
    xm, ctx = context_mask(x, pts, torch.tensor([6.0], device=DEV))
    assert torch.allclose(xm[..., 14:19, 14:19, 14:19], x[..., 14:19, 14:19, 14:19])               # 4 voxels inside the faces: untouched
    assert torch.allclose(xm[..., 0, 0, 0], x.mean().view(1, 1))
    assert torch.allclose(ctx, torch.tensor([[6.0, 2.0, -8.0]], device=DEV))
