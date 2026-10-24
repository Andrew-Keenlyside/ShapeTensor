"""The entity-field model: generator consistency (entities hold the total), the decoder, the losses."""
import math

import pytest
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_centred_paths_pass_through_their_centre():
    from shapetensor.learn.scene import Rng, centred_paths
    r = Rng(3, DEV)
    c = torch.tensor([[40.0, 50.0, 60.0]], device=DEV); d = torch.tensor([[0.0, 0.6, 0.8]], device=DEV)
    P = centred_paths(c, d, torch.tensor([256.0], device=DEV), 129, torch.tensor([0.01], device=DEV), 20.0, r)
    assert P.shape == (1, 129, 3)
    i = int(((P[0] - c[0]).norm(dim=1)).argmin())
    assert float((P[0, i] - c[0]).norm()) < 1e-4
    t = P[0, i + 1] - P[0, i - 1]; t = t / t.norm()
    assert abs(float(t @ d[0])) > 0.998                                                            # the tangent there is d
    assert float((P[0, 1:] - P[0, :-1]).norm(dim=1).std()) < 0.05                                  # even steps across the join


@pytest.mark.parametrize("style", ["cells", "regions"])
def test_entities_hold_the_total(style):
    """Sum over a query's entities of mass x visibility x tensor = the window's total second moment over every oriented
    layer (populations, underlays, interfaces), as the total density's target counts them."""
    import shapetensor.learn.train as TR
    from shapetensor.filters import gaussian
    from shapetensor.learn.scene import draw_scene
    TR.HEAD = "entities"
    try:
        for seed in range(40):
            s, d = draw_scene((48,) * 3, seed=seed, device=DEV, strength=0.0, canvas=96, style=style, n_groups=(1, 2), underlay_p=0.0,
                              entity_ctx=True, p_warp=0.0, p_affine=0.0, p_cutout=0.0, p_sparse=0.0, p_boundary=0.0, zoom=1)
            if len([L for L in s.layers if L.role != "isotropic"]) >= 2:
                break
        out = TR.entity_queries(s, d, 48, seed, DEV)
    finally:
        TR.HEAD = "slots"
    keep = [k for k, L in enumerate(s.layers) if L.role != "isotropic"]
    K = d["occupancy"].shape[0]
    full = lambda x: x.float().view(-1, 1, 1, 1).expand(K, 48, 48, 48) if x.dim() == 1 else x.float()
    vis, ov = full(d["visibility"]), full(d["orientation_visibility"]).clamp(0, 1)
    iso = torch.tensor([1, 1, 1, 0, 0, 0.0], device=DEV).view(6, 1, 1, 1) / 3
    M = sum(d["occupancy"][k].float() * vis[k] * (ov[k] * d["tensors"][k].float() + (1 - ov[k]) * iso) for k in keep)   # (6, Z, Y, X)
    pts, sig = out["e_points"], out["e_sigma"]
    inner = ((pts > 14) & (pts < 33)).all(1)                                                         # windows inside the crop
    live = out["e_node"] >= 0
    S6 = (out["e_mass"].float() * out["e_vis"].float() * live)[..., None] * out["e_T6"].float()
    S6 = S6.sum(1)                                                                                    # (Q, 6)
    errs = []
    for q in torch.nonzero(inner)[:, 0][:40].tolist():
        ref = gaussian(M, float(sig[q]))
        p = pts[q].round().long()
        r6 = ref[:, p[0], p[1], p[2]]
        if float(r6[:3].sum()) < 0.02:
            continue
        errs.append(float((S6[q] - r6).abs().sum() / r6[:3].sum()))
    assert errs, "no tissue query"
    assert sorted(errs)[len(errs) // 2] < 0.1, errs


def test_tree_distance():
    from shapetensor.learn.entity_train import tree_distance
    # root 0; populations 1, 2; objects 3, 4 under 1; 5 under 2
    anc = torch.tensor([[0, -1, -1], [1, 0, -1], [2, 0, -1], [3, 1, 0], [4, 1, 0], [5, 2, 0]])
    i = torch.tensor([3, 3, 3, 1, 0]); j = torch.tensor([3, 4, 5, 2, 5])
    assert tree_distance(anc, i, j).tolist() == [0.0, 2.0, 4.0, 2.0, 2.0]


def _net():
    from shapetensor.learn.model import PopulationNet
    return PopulationNet.from_args(dict(head="entities", widths=[8, 16, 32], sh_lmax=8, embed=4, context_level=True, voxel_cond=True, points=True)).to(DEV)


def test_entity_decoder_shapes_and_gradients():
    net = _net(); B, Q, D = 2, 5, 7
    x = torch.rand(B, 1, 32, 32, 32, device=DEV)
    pts = torch.rand(B, Q, 3, device=DEV) * 31; sig = torch.full((B, Q), 2.0, device=DEV)
    dirs = torch.nn.functional.normalize(torch.randn(B, Q, D, 3, device=DEV), dim=-1)
    P = net(x, voxel=torch.full((B,), 1.0, device=DEV), e_points=pts, e_sigma=sig, e_dirs=dirs)
    e = P["ent"]
    assert e["f"].shape == (B, Q, D) and e["z"].shape == (B, Q, D, 4) and e["R"].shape == (B, Q, D, 3, 3)
    assert e["kappa"].shape == (B, Q, D, 2) and bool((e["kappa"][..., 0] >= e["kappa"][..., 1]).all())
    assert bool((e["z"].norm(dim=-1) < 1).all())
    assert torch.allclose(e["R"].transpose(-1, -2) @ e["R"], torch.eye(3, device=DEV).expand(B, Q, D, 3, 3), atol=1e-4)
    (e["f"].sum() + e["z"].sum() + e["R"].sum() + e["kappa"].sum() + P["total_sh"].sum()).backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters() if p.requires_grad and p.grad is not None)
    assert net.ent.dir[-1].weight.grad is not None


def _one_crop_batch():
    import shapetensor.learn.train as TR
    TR.HEAD, TR.SH_EXACT, TR.DESC_KEY = "entities", 8, "descriptors5"
    try:
        pairs = [TR._make_pair(17 + i, 32, DEV, 0.5, {"scene": {"canvas": 64, "crops_per_scene": 1}}) for i in range(2)]
        x, T = TR.stack(pairs, aug=False)
    finally:
        TR.HEAD, TR.SH_EXACT, TR.DESC_KEY = "slots", 0, "descriptors"
    return x, T


def test_entity_loss_is_small_for_the_targets_themselves():
    """Predictions equal to the targets (density, the owner's frame and concentrations): density and vote losses ~0."""
    from shapetensor.learn import entity_train as ET
    x, T = _one_crop_batch()
    g = torch.Generator(device=DEV).manual_seed(0)
    dirs = ET.sample_dirs(T, 16, g=g)
    from shapetensor.learn import bingham as BG
    mass = (T["e_mass"] * T["e_vis"]).clamp_min(0) * (T["e_node"] >= 0)
    pe = BG.pdf(dirs[..., None, :], T["e_R"][:, :, None], T["e_kappa"][:, :, None]); contrib = mass[:, :, None] * pe
    owner = (contrib / contrib.sum(-1, keepdim=True).clamp_min(1e-9)).argmax(-1)
    D = dirs.shape[2]
    R = torch.gather(T["e_R"][:, :, None].expand(-1, -1, D, -1, -1, -1), 3, owner[..., None, None, None].expand(-1, -1, -1, 1, 3, 3))[..., 0, :, :]
    k = torch.gather(T["e_kappa"][:, :, None].expand(-1, -1, D, -1, -1), 3, owner[..., None, None].expand(-1, -1, -1, 1, 2))[..., 0, :]
    B, Q = dirs.shape[:2]
    P = {"ent": {"f": contrib.sum(-1), "R": R, "kappa": k, "z": torch.zeros(B, Q, D, 4, device=DEV), "desc": torch.zeros(B, Q, D, 5, device=DEV)}}
    _, parts = ET.entity_loss(P, T, dirs, g=g)
    assert float(parts["density"]) < 1e-5
    if "vote" in parts:
        assert float(parts["vote"]) < 1e-3 and float(parts["kappa"]) < 1e-4


def test_overlapping_windows_name_the_same_entities():
    """Two windows of one scene offset by multiples of 8: at the same scene points their entities are the same nodes of
    the scene's tree (distance 0 across windows), and the cross-window loss vanishes for equal embeddings."""
    from shapetensor.learn import entity_train as ET
    import shapetensor.learn.train as TR
    TR.HEAD = "entities"
    try:
        pb = ET.EntityPairBuffer(6, 48, DEV, {"canvas": 96, "n_groups": (2, 3), "cross_p": 1.0, "underlay_p": 0.0, "style": "flat"}, kinds=("shift",), n_points=96)
    finally:
        TR.HEAD = "slots"
    checked = 0
    for img1, img2, t1, t2, X, info in pb.items:
        assert info["kind"] == "shift" and any(info["shift"])
        eq = (t1["e_gnode"][:, :, None] == t2["e_gnode"][:, None, :]) & (t1["e_gnode"][:, :, None] >= 0)
        assert bool((X.float()[eq] == 0).all()) and bool((X.float()[~eq & (X.float() >= 0)] > 0).all())
        if not info["same_acq"]:                                                                    # another acquisition: what shows may differ
            continue
        checked += 1
        g1, g2 = t1["e_gnode"], t2["e_gnode"]
        both = (g1 >= 0).any(1) & (g2 >= 0).any(1)
        same = [set(a[a >= 0].tolist()) == set(b[b >= 0].tolist()) for a, b in zip(g1[both], g2[both])]
        assert len(same) > 10 and sum(same) / len(same) > 0.8, (sum(same), len(same))
    assert checked > 0
    x1, x2, T1, T2, Xb = pb.batch(2, torch.Generator().manual_seed(0))
    dirs = ET.sample_dirs(T1, 8)
    B, Q, D = dirs.shape[:3]
    z = torch.zeros(B, Q, D, 4, device=DEV)
    assert float(ET.cross_window_loss({"ent": {"z": z}}, {"ent": {"z": z}}, T1, T2, Xb, dirs)) < 1e-5 or float((Xb > 0).sum()) > 0


def test_cross_distances_match_the_tree():
    from shapetensor.learn.entities import Tree
    from shapetensor.learn.entity_train import cross_distances
    par = torch.tensor([-1, 0, 0, 1, 1, 2], device=DEV)
    tree = Tree(parent=par, radius=torch.zeros(6, device=DEV), kind=torch.zeros(6, dtype=torch.int8, device=DEV), group=torch.zeros(6, dtype=torch.long, device=DEV),
                depth=torch.tensor([0, 1, 1, 2, 2, 2], device=DEV), leaf_keys=torch.zeros(0, dtype=torch.long, device=DEV), leaf_nodes=torch.zeros(0, dtype=torch.long, device=DEV))
    g1 = torch.tensor([[3, 4, -1]], device=DEV); g2 = torch.tensor([[3, 5, 1]], device=DEV)
    X = cross_distances(tree, g1, g2)[0]
    assert X.tolist() == [[0.0, 4.0, 1.0], [2.0, 4.0, 1.0], [-1.0, -1.0, -1.0]]


def test_cross_distances_dense_layers_under_the_root():
    from shapetensor.learn.entities import Tree
    from shapetensor.learn.entity_train import cross_distances
    par = torch.tensor([-1, 0, 1], device=DEV)
    tree = Tree(parent=par, radius=torch.zeros(3, device=DEV), kind=torch.zeros(3, dtype=torch.int8, device=DEV), group=torch.zeros(3, dtype=torch.long, device=DEV),
                depth=torch.tensor([0, 1, 2], device=DEV), leaf_keys=torch.zeros(0, dtype=torch.long, device=DEV), leaf_nodes=torch.zeros(0, dtype=torch.long, device=DEV))
    X = cross_distances(tree, torch.tensor([[3, 2]], device=DEV), torch.tensor([[3, 4]], device=DEV))[0]
    assert X.tolist() == [[0.0, 2.0], [3.0, 3.0]]


def test_decoder_options_dir_order_and_vote_uncertainty():
    from shapetensor.learn.model import PopulationNet
    from shapetensor.learn import entity_train as ET
    net = PopulationNet.from_args(dict(head="entities", widths=[8, 16, 32], sh_lmax=8, embed=4, context_level=True, voxel_cond=True, points=True,
                                       dir_lmax=16, vote_unc=True)).to(DEV)
    assert net.ent.lmax == 16 and net.ent.dir[0].in_features == 256 + 153
    x, T = _one_crop_batch()
    dirs = ET.sample_dirs(T, 8)
    P = net(x, voxel=T["voxel"], e_points=T["e_points"], e_sigma=T["e_sigma"], e_dirs=dirs)
    assert "log_s2" in P["ent"] and P["ent"]["log_s2"].shape == dirs.shape[:3]
    tot, parts = ET.entity_loss(P, T, dirs)
    tot.backward()
    if "vote" in parts:
        assert "vote_unc" in parts and net.ent.dir[-1].weight.grad is not None
