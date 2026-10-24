"""Inference of the entity field: tiling without block artefacts, entities from the direction grid, links, hierarchy."""
import math

import torch

from shapetensor.learn import entity_infer as EI
from shapetensor.learn.model import PopulationNet, poincare

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _net():
    torch.manual_seed(0)
    return PopulationNet.from_args(dict(head="entities", widths=[8, 16, 32], sh_lmax=8, embed=4, context_level=True, voxel_cond=True, points=True)).to(DEV).eval()


def test_tiled_equals_whole():
    """The trunk has no normalisation and the tiles' halo covers the receptive field: tiled = whole volume."""
    net = _net(); g = torch.Generator(device=DEV).manual_seed(1)
    from shapetensor.filters import gaussian
    x = gaussian(torch.rand(112, 104, 96, generator=g, device=DEV), 1.5)
    x = (x - x.min()) / (x.max() - x.min())
    pts = torch.rand(300, 3, generator=g, device=DEV) * (torch.tensor(x.shape, device=DEV) - 1)
    sig = 1 + 3 * torch.rand(300, generator=g, device=DEV)
    dirs = EI.sphere_dirs(24, DEV)
    whole = EI.query(net, EI.features(net, x, 1.0), pts, sig, dirs, vol_shape=x.shape)
    tiled = EI.query_volume(net, x, pts, sig, dirs, voxel=1.0, core=32)
    for k in ("f", "z", "kappa", "desc"):
        err = float((whole[k] - tiled[k]).abs().max()); ref = float(whole[k].abs().max())
        assert err <= 1e-4 * max(ref, 1.0), (k, err, ref)


def test_sphere_dirs_cover_the_hemisphere():
    d = EI.sphere_dirs(300)
    assert torch.allclose(d.norm(dim=1), torch.ones(300)) and bool((d[:, 0] >= 0).all())
    probe = torch.nn.functional.normalize(torch.randn(2000, 3), dim=1)
    gap = torch.rad2deg(torch.acos((probe @ d.T).abs().amax(1).clamp(max=1))).max()
    assert float(gap) < 9.0                                                                          # every direction within ~8 deg of the grid


def _fake(dirs, lobes):
    """Decoder outputs at one point from lobes ``(axis, kappa, mass, z)``: density = sum of Watson-like lobes, each
    direction's embedding and vote those of the lobe dominating it."""
    D = dirs.shape[0]; f = torch.zeros(D); z = torch.zeros(D, 4); R = torch.eye(3).expand(D, 3, 3).clone(); K = torch.zeros(D, 2); best = torch.zeros(D)
    for ax, k, m, zz in lobes:
        ax = torch.nn.functional.normalize(torch.tensor(ax, dtype=torch.float32), dim=0)
        p = m * torch.exp(k * ((dirs @ ax) ** 2 - 1)) * k / 2
        a = torch.nn.functional.normalize(torch.linalg.cross(ax, torch.tensor([0.3, 0.5, 0.8])), dim=0); b = torch.linalg.cross(ax, a)
        Ri = torch.stack([a, b, ax], 1)
        win = p > best
        z[win] = torch.tensor(zz); R[win] = Ri; K[win] = torch.tensor([k, k]); best = torch.maximum(best, p); f = f + p
    return {"f": f[None], "z": z[None], "R": R[None], "kappa": K[None], "desc": torch.zeros(1, D, 5)}


def test_cluster_separates_crossings_and_parallel_neighbours():
    dirs = EI.sphere_dirs(400)
    za, zb = [0.5, 0, 0, 0], [-0.5, 0, 0, 0]
    # a crossing at 40 degrees: two entities, axes right
    r = _fake(dirs, [((1, 0, 0), 40.0, 1.0, za), ((math.cos(math.radians(40)), math.sin(math.radians(40)), 0), 40.0, 0.6, zb)])
    e = EI.cluster(r, dirs)
    assert int(e["live"][0].sum()) == 2
    ax = e["R"][0, :2, :, 2]
    assert float((ax[0] @ torch.tensor([1.0, 0, 0])).abs()) > math.cos(math.radians(3))
    # one lobe whose two halves carry different embeddings (parallel neighbours): two entities by embedding
    r2 = _fake(dirs, [((1, 0, 0), 30.0, 1.0, za)]); half = dirs[:, 1] > 0
    r2["z"][0, half] = torch.tensor(zb)
    assert int(EI.cluster(r2, dirs)["live"][0].sum()) == 2
    # two lobes with the same embedding (a branch in the window): one entity
    r3 = _fake(dirs, [((1, 0, 0), 40.0, 1.0, za), ((0, 1, 0), 40.0, 1.0, za)])
    assert int(EI.cluster(r3, dirs)["live"][0].sum()) == 1


def test_link_follows_a_line_and_keeps_neighbours_apart():
    n = 20
    pts = torch.cat([torch.stack([torch.arange(n).float(), torch.zeros(n), torch.zeros(n)], 1),
                     torch.stack([torch.arange(n).float(), torch.full((n,), 2.0), torch.zeros(n)], 1)])
    Q = pts.shape[0]; C = 2
    R = torch.eye(3).expand(Q, C, 3, 3).clone(); R[..., :, 2] = torch.tensor([1.0, 0, 0]); R[..., :, 0] = torch.tensor([0, 0, 1.0]); R[..., :, 1] = torch.tensor([0, 1.0, 0])
    z = torch.zeros(Q, C, 4); z[:n, 0, 0] = 0.6; z[n:, 0, 0] = -0.6
    live = torch.zeros(Q, C, dtype=torch.bool); live[:, 0] = True
    ids = EI.link(pts, {"mass": torch.ones(Q, C), "live": live, "z": z, "R": R}, radius=2.5)
    assert len(set(ids[:n, 0].tolist())) == 1 and len(set(ids[n:, 0].tolist())) == 1 and ids[0, 0] != ids[n, 0]
    assert bool((ids[:, 1] == -1).all())


def test_hierarchy_groups_by_tree_position():
    z = poincare(torch.tensor([[3.0, 0.2], [3.0, -0.2], [-3.0, 0.2], [-3.0, -0.2]]))
    from scipy.cluster.hierarchy import fcluster
    lab = fcluster(EI.hierarchy(z), 2, "maxclust")
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]


def test_follow_runs_on_a_model():
    net = _net(); x = torch.rand(64, 64, 64, device=DEV)
    out = EI.follow(net, x, (32.0, 32.0, 32.0), (1.0, 0.0, 0.0), voxel=1.0, max_steps=3, n_dirs=60, core=32, f_rel=0.0, f_abs=0.0)
    assert out["points"].shape[1] == 3 and out["points"].shape[0] == out["axes"].shape[0] >= 1
