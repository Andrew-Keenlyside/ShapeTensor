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
    # a branch in the window: two lobes of one entity -- the same embedding and the same vote (the entity's one Bingham,
    # moment matched to both arms) -> one entity, though the lobes lie apart on the sphere
    r3 = _fake(dirs, [((1, 0, 0), 40.0, 1.0, za), ((0, 1, 0), 40.0, 1.0, za)])
    r3["R"][:] = r3["R"][0, int(r3["f"][0].argmax())]; r3["kappa"][:] = torch.tensor([30.0, 0.5])
    assert int(EI.cluster(r3, dirs)["live"][0].sum()) == 1
    # two lobes voting for different entities with one embedding (siblings the embedding failed to tell apart): two
    r4 = _fake(dirs, [((1, 0, 0), 40.0, 1.0, za), ((0, 1, 0), 40.0, 1.0, za)])
    assert int(EI.cluster(r4, dirs)["live"][0].sum()) == 2


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
    out = EI.follow(net, x, (32.0, 32.0, 32.0), (1.0, 0.0, 0.0), voxel=1.0, max_steps=3, n_dirs=60, core=32, f_frac=0.0, f_abs=0.0)
    assert out["points"].shape[1] == 3 and out["points"].shape[0] == out["axes"].shape[0] >= 1


def test_sparse_directions_match_the_dense_query():
    net = _net(); g = torch.Generator(device=DEV).manual_seed(2)
    x = torch.rand(48, 48, 48, generator=g, device=DEV)
    pts = 8 + torch.rand(50, 3, generator=g, device=DEV) * 32; sig = torch.full((50,), 2.0, device=DEV)
    dirs = EI.sphere_dirs(40, DEV); f = EI.features(net, x, 1.0)
    dense = EI.query(net, f, pts, sig, dirs, vol_shape=x.shape)
    sparse = EI.query(net, f, pts, sig, dirs, vol_shape=x.shape, t_rel=0.5)
    ev = sparse["evaluated"]
    assert 0 < int(ev.sum()) < ev.numel()
    for k in ("f", "z", "kappa"):
        assert torch.allclose(dense[k][ev], sparse[k][ev], atol=1e-5)
    assert bool((sparse["f"][~ev] == 0).all())


def test_link_never_merges_a_crossing():
    """Two lines crossing at one point, with embeddings drifting so a chain of weak links could join them: the
    cannot-link rule keeps them two objects."""
    n = 21; t = torch.arange(n).float() - 10
    A = torch.stack([t, torch.zeros(n), torch.zeros(n)], 1); B = torch.stack([torch.zeros(n), t, torch.zeros(n)], 1)
    pts = torch.cat([A, B[B.abs().sum(1) > 0]])                                                    # the crossing point once, in A
    Q, C = pts.shape[0], 2
    R = torch.eye(3).expand(Q, C, 3, 3).clone(); z = torch.zeros(Q, C, 4); live = torch.zeros(Q, C, dtype=torch.bool)
    on_b = torch.zeros(Q, dtype=torch.bool); on_b[n:] = True
    R[~on_b, 0, :, 2] = torch.tensor([1.0, 0, 0]); live[~on_b, 0] = True; z[~on_b, 0, 0] = 0.3
    R[on_b, 0, :, 2] = torch.tensor([0, 1.0, 0]); live[on_b, 0] = True; z[on_b, 0, 0] = 0.0
    centre = 10; R[centre, 1, :, 2] = torch.tensor([0, 1.0, 0]); live[centre, 1] = True; z[centre, 1, 0] = 0.15   # B seen at the crossing
    ids = EI.link(pts, {"mass": torch.ones(Q, C), "live": live, "z": z, "R": R}, radius=1.5, z_link=0.7, ang_max=95.0)
    assert ids[centre, 0] != ids[centre, 1]
    assert len(set(ids[~on_b, 0].tolist())) == 1


def test_cluster_keeps_broad_entities_and_weighs_them():
    """A sharp lobe (kappa 200) and a broad one (kappa 8) of equal mass crossing at 60 degrees: both found, shares ~1/2
    -- the cut is relative to the mean density, and the mass comes from the voted Bingham (not the grid)."""
    from shapetensor.learn import bingham as BG
    dirs = EI.sphere_dirs(300)
    D = dirs.shape[0]
    def lobe(ax, k):
        ax = torch.nn.functional.normalize(torch.tensor(ax, dtype=torch.float32), dim=0)
        a = torch.nn.functional.normalize(torch.linalg.cross(ax, torch.tensor([0.3, 0.5, 0.8])), dim=0); b = torch.linalg.cross(ax, a)
        R = torch.stack([a, b, ax], 1); K = torch.tensor([k, k])
        return R, K, BG.pdf(dirs, R.expand(D, 3, 3), K.expand(D, 2))
    R1, K1, p1 = lobe((1, 0, 0), 200.0); R2, K2, p2 = lobe((0.5, math.sqrt(3) / 2, 0), 8.0)
    f = 0.5 * p1 + 0.5 * p2; own1 = 0.5 * p1 > 0.5 * p2
    z = torch.zeros(D, 4); z[own1, 0] = 0.6; z[~own1, 0] = -0.6
    R = torch.where(own1[:, None, None], R1, R2); K = torch.where(own1[:, None], K1, K2)
    r = {"f": f[None], "z": z[None], "R": R[None], "kappa": K[None], "desc": torch.zeros(1, D, 5)}
    e = EI.cluster(r, dirs)
    assert int(e["live"][0].sum()) == 2
    sh = sorted(e["share"][0][e["live"][0]].tolist())
    assert abs(sh[0] - 0.5) < 0.12 and abs(sh[1] - 0.5) < 0.12, sh
    assert abs(float(e["mass"][0][e["live"][0]].sum()) - 1.0) < 0.2


def test_link_does_not_chain_parallel_objects_through_their_parent():
    """Two parallel objects A, B (depth 2) and, at the points between them, their parent (depth 1) as the entity: an
    anchored embedding puts the parent one tree edge from each, so neither the threshold nor the depth rule links them."""
    n = 12
    pts = torch.cat([torch.stack([torch.arange(n).float(), torch.full((n,), y), torch.zeros(n)], 1) for y in (0.0, 1.0, 2.0)])
    Q, C = pts.shape[0], 2
    R = torch.eye(3).expand(Q, C, 3, 3).clone(); R[..., :, 2] = torch.tensor([1.0, 0, 0]); R[..., :, 0] = torch.tensor([0, 0, 1.0]); R[..., :, 1] = torch.tensor([0, 1.0, 0])
    at = lambda d, phi: math.tanh(d / 2) * torch.tensor([math.cos(phi), math.sin(phi), 0.0, 0.0])
    z = torch.zeros(Q, C, 4); z[:n, 0] = at(2.0, 0.3); z[n:2 * n, 0] = at(1.0, 0.0); z[2 * n:, 0] = at(2.0, -0.3)
    live = torch.zeros(Q, C, dtype=torch.bool); live[:, 0] = True
    ids = EI.link(pts, {"mass": torch.ones(Q, C), "live": live, "z": z, "R": R}, radius=1.5)
    assert len({int(ids[0, 0]), int(ids[n, 0]), int(ids[2 * n, 0])}) == 3


def test_mass_from_region_tiled_equals_whole():
    """A model whose mass is the region map's window mean: the tiled query's mass equals the whole volume's."""
    import torch
    from shapetensor.learn import entity_infer as EI
    from shapetensor.learn.model import PopulationNet
    torch.manual_seed(0)
    net = PopulationNet.from_args(dict(head="entities", widths=[8, 8, 8], sh_lmax=8, embed=4, context_level=True, voxel_cond=True, points=True,
                                       dir_lmax=4, mass_head=True, region_head=True, dens_sh=True, dens_mlp=True, mass_region=True)).eval()
    x = torch.rand(48, 40, 56)
    pts = torch.rand(30, 3) * (torch.tensor([47.0, 39.0, 55.0]))
    sig = 1 + torch.rand(30) * 4; dirs = EI.sphere_dirs(20)
    with torch.no_grad():
        whole = EI.query(net, EI.features(net, x), pts, sig, dirs, vol_shape=x.shape)
        tiled = EI.query_volume(net, x, pts, sig, dirs, core=16, halo=EI.halo_for(net))
    assert "m" in whole and torch.allclose(whole["m"], tiled["m"], atol=2e-3, rtol=1e-2)


def test_cluster_lobes_separates_a_crossing_density():
    """Entities from the density's lobes: two Watson lobes -> two entities on their axes; one lobe -> one."""
    import torch
    from shapetensor.learn import entity_infer as EI
    dirs = EI.sphere_dirs(200)
    U = lambda *v: torch.nn.functional.normalize(torch.tensor(v), dim=0)
    a1, a2 = U(0.25, 0.7, -0.7), U(0.25, 0.7, 0.7)
    f = torch.stack([torch.exp(20 * (dirs @ a1) ** 2 - 20) + 0.8 * torch.exp(20 * (dirs @ a2) ** 2 - 20), torch.exp(20 * (dirs @ a1) ** 2 - 20)])
    Q, D = f.shape
    r = {"f": f, "z": torch.zeros(Q, D, 4), "R": torch.eye(3).expand(Q, D, 3, 3).clone(), "kappa": torch.ones(Q, D, 2), "desc": torch.zeros(Q, D, 5)}
    e = EI.cluster_lobes(r, dirs)
    A = lambda u, v: float(torch.rad2deg(torch.acos((u @ v).abs().clamp(max=1))))
    assert int(e["live"][0].sum()) == 2 and int(e["live"][1].sum()) == 1
    ax = [e["R_odf"][0, i, :, 2] for i in torch.nonzero(e["live"][0])[:, 0].tolist()]
    assert min(A(ax[0], a1), A(ax[1], a1)) < 5 and min(A(ax[0], a2), A(ax[1], a2)) < 5


def test_split_mass_keeps_the_floor_under_a_lobe():
    """A sharp lobe (mass 0.3) on an isotropic floor (mass 0.7): split by the lobes, the entity keeps 0.3 and the floor
    0.7 (whole directions would give the lobe the floor beneath it); a crossing's two lobes keep their own masses."""
    from shapetensor.learn import bingham as BG
    dirs = EI.sphere_dirs(300)
    I = torch.eye(3)
    RA, RB = I.clone(), torch.stack([I[:, 0], I[:, 2], I[:, 1]], 1)                               # mode axes z (column 2) and y
    k = torch.tensor([17.0, 17.0])
    pA, pB = BG.pdf(dirs, RA, k), BG.pdf(dirs, RB, k)
    f = torch.stack([0.3 * pA + 0.7, 0.3 * pA + 0.3 * pB + 0.4])
    Q, D = f.shape
    R = torch.stack([RA, RB, I]).expand(Q, 3, 3, 3); K = torch.stack([k, k, torch.zeros(2)]).expand(Q, 3, 2)
    e = {"live": torch.tensor([[True, False, False], [True, True, False]]), "R": R, "kappa": K}
    r = {"f": f, "m": f.mean(1)}
    sp = EI.split_mass(r, dirs, e)
    assert abs(float(sp["mass"][0, 0]) - 0.3) < 0.03 and abs(float(sp["floor"][0]) - 0.7) < 0.03
    assert abs(float(sp["mass"][1, 0]) - 0.3) < 0.03 and abs(float(sp["mass"][1, 1]) - 0.3) < 0.03 and abs(float(sp["floor"][1]) - 0.4) < 0.03


def test_kappa_log_concentrations_are_alive_and_bounded():
    """``kappa_log``: concentrations in (0.006, 150) with gradient everywhere -- the softplus head saturates at 0."""
    torch.manual_seed(0)
    net = PopulationNet.from_args(dict(head="entities", widths=[8, 8, 8], sh_lmax=8, embed=4, context_level=True, voxel_cond=True, points=True,
                                       dir_lmax=4, mass_head=True, region_head=True, dens_sh=True, dens_mlp=True, mass_region=True, kappa_log=True))
    net.ent.reset_kappa()
    x = torch.rand(1, 1, 32, 32, 32); pts = torch.rand(1, 20, 3) * 31; sig = torch.full((1, 20), 2.0); dirs = torch.nn.functional.normalize(torch.randn(1, 20, 8, 3), dim=-1)
    P = net(x, e_points=pts, e_sigma=sig, e_dirs=dirs)
    k = P["ent"]["kappa"]
    assert float(k.min()) > 0.006 and float(k.max()) < 150 and 0.3 < float(k.median()) < 3.0
    k.sum().backward()
    assert float(net.ent.dir[-1].weight.grad.abs().sum()) > 0


def test_cluster_floor_keeps_the_background_out_of_the_entities():
    """A lobe on a uniform floor whose directions vote at random (the background's votes are not trained): with
    ``floor`` the floor is the background's mass and one entity of mass 0.3 forms on the lobe; without, the floor's
    directions join the lobe's entity, which takes the whole mass."""
    from shapetensor.learn import bingham as BG
    torch.manual_seed(0)
    dirs = EI.sphere_dirs(300); D = dirs.shape[0]
    RA = torch.eye(3); k = torch.tensor([17.0, 17.0]); pA = BG.pdf(dirs, RA, k)
    f = (0.3 * pA + 0.7)[None]
    lobe = (dirs @ RA[:, 2]).abs() > 0.9
    Rv = torch.linalg.qr(torch.randn(D, 3, 3))[0]; Rv = Rv * torch.sign(torch.linalg.det(Rv))[:, None, None]
    Rv[lobe] = RA; Kv = torch.where(lobe[:, None], k, torch.full((D, 2), 10.0))
    r = {"f": f, "m": torch.tensor([1.0]), "z": torch.zeros(1, D, 4), "R": Rv[None], "kappa": Kv[None], "desc": torch.zeros(1, D, 5)}
    e = EI.cluster(r, dirs, floor=0.05)
    live = torch.nonzero(e["live"][0])[:, 0]
    assert live.numel() == 1 and abs(float(e["floor_mass"][0]) - 0.7) < 0.05 and abs(float(e["mass"][0, live[0]]) - 0.3) < 0.05
    assert float((e["R"][0, live[0], :, 2] @ RA[:, 2]).abs()) > 0.99
    e0 = EI.cluster(r, dirs)
    assert float((e0["mass"][0] * e0["live"][0]).max()) > 0.9


def _tiny_args(**kw):
    return dict(head="entities", widths=[8, 8, 8], sh_lmax=8, embed=4, context_level=True, voxel_cond=True, points=True, dir_lmax=4,
                mass_head=True, region_head=True, dens_sh=True, dens_mlp=True, mass_region=True, lcn=3.0, **kw)


def test_oprobe_warm_start_computes_what_the_old_decoder_did():
    """A model with the oriented probe, loaded from one without (load_partial: the decoder's grown input zero, the
    probe's density path zero-initialised), gives the old outputs."""
    from shapetensor.learn import dataflow as DF
    torch.manual_seed(0)
    old = PopulationNet.from_args(_tiny_args()).eval()
    new = PopulationNet.from_args(_tiny_args(oprobe=True)).eval()
    DF.load_partial(new, old.state_dict(), lambda m: None)
    x = torch.rand(1, 1, 32, 32, 32); pts = 4 + torch.rand(1, 12, 3) * 23; sig = torch.full((1, 12), 2.0)
    dirs = torch.nn.functional.normalize(torch.randn(1, 12, 6, 3), dim=-1)
    with torch.no_grad():
        a = old(x, e_points=pts, e_sigma=sig, e_dirs=dirs)["ent"]; b = new(x, e_points=pts, e_sigma=sig, e_dirs=dirs)["ent"]
    assert torch.allclose(a["f"], b["f"], atol=1e-5) and torch.allclose(a["kappa"], b["kappa"], atol=1e-5)


def test_oprobe_code_turns_with_the_volume():
    """The oriented probe reads the material along u: rotating the volume by a quarter turn and the point and direction
    with it leaves its code (nearly) unchanged -- the rings' angular magnitudes do not depend on the roll about u."""
    from shapetensor.learn.model import with_scal
    torch.manual_seed(0)
    net = PopulationNet.from_args(_tiny_args(oprobe=True)).eval()
    n = 33; c = (n - 1) / 2
    z, y, x_ = torch.meshgrid(*(torch.arange(n, dtype=torch.float32),) * 3, indexing="ij")
    vol = torch.exp(-((y - c) ** 2 + (z - c) ** 2) / 8)[None, None] + 0.2 * torch.exp(-((x_ - 10) ** 2 + (y - 20) ** 2 + (z - 12) ** 2) / 20)[None, None]
    rot = lambda v: v.transpose(-1, -3).flip(-1)                                                    # a quarter turn about y: (z, y, x) -> (x, y, -z)
    S0, S1 = net.scalar_fields(vol), net.scalar_fields(rot(vol))
    p = torch.tensor([[c, c, 9.0], [12.0, 20.0, 14.0]]); u = torch.nn.functional.normalize(torch.tensor([[0.0, 0.3, 1.0], [0.6, 0.5, 0.6]]), dim=-1)
    mp = lambda q: torch.stack([q[:, 2], q[:, 1], (n - 1) - q[:, 0]], 1); mu = lambda q: torch.stack([q[:, 2], q[:, 1], -q[:, 0]], 1)
    with torch.no_grad():
        a = net.ent.oprobe_feats(with_scal((None, None, None), S0), p, u); b = net.ent.oprobe_feats(with_scal((None, None, None), S1), mp(p), mu(u))
    assert float((a - b).abs().max()) < 0.05 * float(a.abs().max())


def test_density_shape_loss_same_norm_is_zero_for_the_right_shape():
    """A prediction proportional to the target: with the target normalised over the same uniform directions the shape
    loss is 0 whichever directions were drawn; against its exact sphere mean it is not when the uniform draws miss the
    lobe (the bias that trained lobes low)."""
    from shapetensor.learn import entity_train as ET, bingham as BG
    torch.manual_seed(0)
    D, nu = 32, 16
    dirs = torch.nn.functional.normalize(torch.randn(1, 50, D, 3), dim=-1)
    ax = torch.tensor([0.0, 0.0, 1.0]); dirs[:, :, nu:] = torch.nn.functional.normalize(ax + 0.15 * torch.randn(1, 50, D - nu, 3), dim=-1)
    rho = 0.3 * BG.pdf(dirs, torch.eye(3), torch.tensor([17.0, 17.0])) + 0.7
    m_t = torch.ones(1, 50); f = 3.7 * rho
    old = ET.DENS_SAME_NORM
    try:
        ET.DENS_SAME_NORM = True; same = float(ET.density_shape_loss(f, rho, m_t, nu))
        ET.DENS_SAME_NORM = False; exact = float(ET.density_shape_loss(f, rho, m_t, nu))
    finally:
        ET.DENS_SAME_NORM = old
    assert same < 1e-5 and exact > 0.05
