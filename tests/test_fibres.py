"""Monte Carlo random fibres: packing without overlaps, the collision relaxation, the classical line-direction check."""
import math

import numpy as np
import torch

from shapetensor.learn import fibres as FB, inferability as IF
from shapetensor.learn.entity_infer import sphere_dirs
from shapetensor.learn.scene import Rng, Scene, curves_like

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_random_section_has_no_overlaps_and_reaches_its_fraction():
    C, R = FB.random_section(0.35, 1.0, 6.0, 12.0, np.random.default_rng(0))
    d = np.hypot(*(C[:, None] - C[None]).transpose(2, 0, 1)); np.fill_diagonal(d, np.inf)
    assert (d >= R[:, None] + R[None] + FB.GAP - 1e-9).all()
    assert (np.hypot(*C.T) + R <= 12.0 + 1e-9).all()
    assert abs((R ** 2).sum() / 12.0 ** 2 - 0.35) < 0.03


def _bundle(axis, offset, n, rad, seed, S=161):
    """``n`` straight fibres along ``axis`` through a disc of radius 5 around (32, 32, 32) + offset."""
    g = np.random.default_rng(seed); ax = np.asarray(axis, float); ax /= np.linalg.norm(ax)
    e1 = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
    s = np.linspace(-40, 40, S)
    P = []
    for _ in range(n):
        r, t = 5 * math.sqrt(g.uniform()), g.uniform(0, 2 * math.pi)
        c = np.array([32.0, 32, 32]) + offset + r * math.cos(t) * e1 + r * math.sin(t) * e2
        P.append(c + s[:, None] * ax)
    P = torch.tensor(np.stack(P), dtype=torch.float32, device=DEV)
    return curves_like(P, torch.full((n, S), rad, device=DEV), Rng(seed, DEV), obj=torch.arange(n, device=DEV), seg=torch.zeros(n, device=DEV), contrast=80.0)


def test_relaxation_removes_the_overlap_of_two_crossing_bundles():
    cur = [_bundle((0, 1, 0), 0.0, 40, 0.8, 1), _bundle((0, 0.5, math.sqrt(3) / 2), 0.0, 40, 0.8, 2)]
    before = FB.overlap_fraction(cur, (64, 64, 64), DEV)
    st = FB.relax(cur, box=((0, 0, 0), (63, 63, 63)))
    after = FB.overlap_fraction(cur, (64, 64, 64), DEV)
    assert before > 0.05 and after < 0.25 * before, (before, after, st)
    # the fibres stay smooth: no sample moves sideways by more than a few radii from its neighbours' mean
    for c in cur:
        lap = 0.5 * (c.P[:, :-2] + c.P[:, 2:]) - c.P[:, 1:-1]
        assert float(lap.norm(dim=-1).max()) < 0.5


def test_scene_flags_off_by_default_and_relaxed_scene_builds():
    assert FB.RELAX is False and FB.PACKING == "hex"
    FB.RELAX, FB.PACKING = True, "random"
    try:
        sc = Scene((64,) * 3, 3, DEV, voxel_um=2.0).build({"bundle": 1.0}, n_groups=(2, 2), cross_p=1.0, underlay_p=0.0)
        assert "relax" in sc.record or not any(g.name == "bundle" for g in sc.groups)
    finally:
        FB.RELAX, FB.PACKING = False, "hex"


def test_hessian_line_directions_find_a_single_rod_lattice_axis():
    n = 48; ax = torch.nn.functional.normalize(torch.tensor([0.3, 0.8, 0.5]), dim=0)
    X = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32)] * 3, indexing="ij"), -1) - n / 2
    perp = X - (X @ ax)[..., None] * ax
    e1 = torch.nn.functional.normalize(torch.linalg.cross(ax, torch.tensor([1.0, 0, 0])), dim=0); e2 = torch.linalg.cross(ax, e1)
    a, b = perp @ e1, perp @ e2
    d = torch.sqrt((a - 5 * torch.round(a / 5)) ** 2 + (b - 5 * torch.round(b / 5)) ** 2)
    img = (1.4 - d).clamp(0, 1).to(DEV)
    mask = (X.norm(dim=-1) <= 10).to(DEV); dirs = sphere_dirs(1000, DEV)
    dens = IF.hessian_line_dirs(img, mask, dirs, sigma=1.0)
    est, _ = IF.peak_axes(dens, dirs, k=1)
    assert IF.axis_errors(est, ax[None].to(DEV))[0] < 6.0


def test_cell_pairs_match_a_kd_tree():
    from scipy.spatial import cKDTree
    g = torch.Generator().manual_seed(0)
    A = (torch.rand(3000, 3, generator=g) * 40 - 5).to(DEV); B = (torch.rand(2000, 3, generator=g) * 40 - 5).to(DEV)
    for same, Xb in ((True, A), (False, B)):
        got = FB._cell_pairs(A, Xb, 2.3, same).cpu().numpy()
        if same:
            ref = cKDTree(A.cpu().numpy()).query_pairs(2.3, output_type="ndarray")
        else:
            ref = cKDTree(A.cpu().numpy()).sparse_distance_matrix(cKDTree(Xb.cpu().numpy()), 2.3, output_type="ndarray")
            ref = np.stack([ref["i"], ref["j"]], 1)
        assert {tuple(p) for p in got} == {tuple(p) for p in ref}
