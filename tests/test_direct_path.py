"""The direct oriented path's bank and sampling, the matrix-product Gaussian, the split targets and the direction sampler
(27-28 Oct): small volumes, CPU-fast."""
import math

import pytest
import torch

DEV = "cpu"


def _net(od_sigmas=(2.0, 4.0, 6.0), odirect=32):
    import shapetensor.learn.train as TR
    from shapetensor.learn.model import PopulationNet
    TR.HEAD = "entities"
    args = {"head": "entities", "widths": [8, 8, 8], "sh_lmax": 4, "embed": 4, "context_level": True, "voxel_cond": True,
            "dir_lmax": 4, "mass_head": True, "region_head": True, "dens_sh": True, "dens_mlp": True, "kappa_log": True,
            "bg_head": True, "obj_head": True, "split_head": True, "odirect": odirect, "od_sigmas": list(od_sigmas)}
    torch.manual_seed(0)
    return PopulationNet.from_args(args).to(DEV).eval()


def _reference_levels(net, x, sigmas):
    """The bank as first written (all directions at once, separable convolutions)."""
    from shapetensor.filters import gaussian
    E = net.oriented_energy(x[:, :1], sig_par=8.0, sig_perp=0.6, pool=0.0, dirs=net.ent.od_dirs, relative=False).float()
    out = []
    for s in sigmas:
        Ep = gaussian(E, s); out.append(Ep / (Ep.mean(1, keepdim=True) + 1e-3 * Ep.mean(dim=(1, 2, 3, 4), keepdim=True) + 1e-12))
    return out


def test_gaussian_mm_equals_gaussian():
    from shapetensor.filters import gaussian, gaussian_mm
    x = torch.rand(2, 3, 12, 9, 7) ** 3
    for s in (0.7, 2.0, 6.0):                                                                   # (6: the kernel longer than the axes)
        assert float((gaussian(x, s) - gaussian_mm(x, s)).abs().max()) < 1e-5


@pytest.mark.parametrize("sigmas", [(2.0, 4.0, 6.0), (2.0, 4.0, 6.0, 8.0)])
def test_levels_match_reference_and_chunking(sigmas):
    net = _net(sigmas); x = torch.rand(2, 1, 16, 16, 16)
    ref = _reference_levels(net, x, sigmas)
    for chunk in (7, 32):
        lv = net.direct_levels(x, chunk=chunk, strides=(1, 1, 1))
        assert len(lv) == len(sigmas) and lv.strides == (1,) * len(sigmas)
        for a, b in zip(ref, lv):
            assert float((a - b).abs().max() / a.abs().max()) < 1e-4


def test_levels_are_per_item():
    net = _net(); x = torch.rand(2, 1, 16, 16, 16); x[1] *= 0.01                                 # a low-contrast batch-mate
    both = net.direct_levels(x, strides=(1, 1, 1)); alone = net.direct_levels(x[:1], strides=(1, 1, 1))
    for a, b in zip(both, alone):
        assert float((a[:1] - b).abs().max()) < 1e-5


def test_levels_fp32_under_autocast():
    net = _net(); x = torch.rand(1, 1, 16, 16, 16)
    ref = net.direct_levels(x, strides=(1, 1, 1))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lv = net.direct_levels(x, strides=(1, 1, 1))
    assert all(t.dtype == torch.float32 for t in lv)
    assert max(float((a - b).abs().max()) for a, b in zip(ref, lv)) < 1e-5


def test_per_point_code_matches_per_pair_and_strides():
    from shapetensor.learn.model import with_oe
    net = _net(); dec = net.ent; x = torch.rand(1, 1, 16, 16, 16)
    feats = with_oe((None, None, None), net.direct_levels(x, strides=(1, 1, 1)))
    p = torch.rand(5, 3) * 13 + 1; u = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1)
    grid = dec.od_code(dec.od_sample(feats, p), dec.od_weights(u))                                # (5, 7, 32)
    pair = dec.od_feats(feats, p[:, None].expand(-1, 7, -1).reshape(-1, 3), u.repeat(5, 1)).view(5, 7, -1)
    assert float((grid - pair).abs().max()) < 1e-4
    qi, di = torch.tensor([0, 2, 4]), torch.tensor([6, 1, 3])
    assert float((dec.od_code(dec.od_sample(feats, p), dec.od_weights(u), pairs=(qi, di)) - grid[qi, di]).abs().max()) < 1e-6
    lv2 = net.direct_levels(x, strides=(2, 2, 2))                                                  # stride 2: exact at even voxels
    pe = torch.tensor([[2.0, 4.0, 8.0], [6.0, 10.0, 12.0]])
    v1 = dec.od_sample(feats, pe); v2 = dec.od_sample(with_oe((None, None, None), lv2), pe)
    assert float((v1 - v2).abs().max()) < 1e-4


def test_old_checkpoint_args_keep_three_levels():
    import shapetensor.learn.train as TR
    from shapetensor.learn.model import PopulationNet
    TR.HEAD = "entities"
    net = PopulationNet.from_args({"head": "entities", "widths": [8, 8, 8], "sh_lmax": 4, "embed": 4, "dir_lmax": 4, "odirect": 16})
    assert net.ent.od_sigmas == (2.0, 4.0, 6.0) and net.ent.od_net[0].in_features == 4


def test_split_targets_and_isotropic_directions():
    from shapetensor.learn import phantom2 as P2, entity_train as ET
    spec, look = P2.random_spec("mix", torch.Generator().manual_seed(1))
    ph = P2.make(spec, seed=1, look=look, device=DEV)
    t = P2._split_targets(ph); iso = [k in P2.ISO_KINDS for k in ph["kinds"]]
    assert torch.allclose(t["ori_t"] + t["iso_t"], ph["occ"].sum(0), atol=1e-5)
    assert torch.allclose(t["iso_t"], sum((ph["occ"][k] for k in range(len(iso)) if iso[k]), torch.zeros_like(t["iso_t"])), atol=1e-6)
    # isotropic entities: their sampled directions uniform (not clustered about the frame's m-axis)
    Q, E = 4000, 2
    T = {"e_mass": torch.tensor([[1.0, 0.0]]).expand(Q, E)[None], "e_vis": torch.ones(1, Q, E), "e_node": torch.zeros(1, Q, E, dtype=torch.long),
         "e_R": torch.eye(3).expand(1, Q, E, 3, 3), "e_kappa": torch.zeros(1, Q, E, 2)}
    d = ET.sample_dirs(T, 32, g=torch.Generator().manual_seed(0))[0, :, 16:]                     # the entity half
    assert abs(float((d[..., 2].abs() > math.cos(math.radians(30))).float().mean()) - (1 - math.cos(math.radians(30)))) < 0.03


def test_readout_options_validated():
    from shapetensor.learn import entity_infer as EI
    with pytest.raises(ValueError):
        EI.entities_from_query({"f": torch.ones(2, 10)}, torch.randn(10, 3), entities="lobe")


def test_lobe_saddle_merge_keeps_girdles_whole_and_crossings_apart():
    from shapetensor.learn import entity_infer as EI, bingham as BG
    dirs = EI.sphere_dirs(200, DEV)
    def entities(f, saddle):
        Q, D = f.shape
        r = {"f": f, "z": torch.zeros(Q, D, 4), "R": torch.eye(3).expand(Q, D, 3, 3).clone(), "kappa": torch.ones(Q, D, 2),
             "desc": torch.zeros(Q, D, 5)}
        e = EI.cluster_lobes_floor(r, dirs, floor=0.05, saddle=saddle)
        return int(e["live"].sum())
    R = torch.eye(3)[None]
    girdle = BG.pdf(dirs, R, torch.tensor([[12.0, 0.0]]))[None] + 0.3                              # a sheet: a great circle
    a = torch.tensor([0.0, 0.0, 1.0]); b = torch.tensor([0.0, math.sin(math.radians(60)), math.cos(math.radians(60))])
    Rb = torch.stack([torch.tensor([1.0, 0.0, 0.0]), torch.linalg.cross(b, torch.tensor([1.0, 0.0, 0.0])), b], 1)[None]
    cross = (BG.pdf(dirs, R, torch.tensor([[20.0, 20.0]])) + BG.pdf(dirs, Rb, torch.tensor([[20.0, 20.0]])))[None] + 0.3
    assert entities(girdle, None) > 1                                                               # the ridge's peaks split it ...
    assert entities(girdle, 0.8) == 1                                                               # ... joined across their saddles
    assert entities(cross, 0.8) == 2                                                                # a 60-deg crossing stays two
