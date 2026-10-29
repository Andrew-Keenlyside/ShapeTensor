"""The oriented-texture family (scene.Scene.texture): finite fields, occupancy in [0, 1], trace-1 line tensors that follow
the construction's orientation field; superposed fields are separate populations. CPU, synthetic only."""
import torch

from shapetensor.learn.scene import Scene


def _scene(seed):
    return Scene((48,) * 3, seed, "cpu", voxel_um=5.0)


def test_texture_populations_are_finite_line_fields():
    for seed in range(6):
        sc = _scene(seed); gs = sc.texture()
        assert gs, sc.record
        mode = sc.record["families"][-1]["texture"]["mode"]
        assert len(gs) == {"single": 1, "boundary": 2, "superposed2": 2, "superposed3": 3}[mode]
        for g in gs:
            occ, T, desc, con, om = g.dense[0]
            assert g.kind == "line" and torch.isfinite(occ).all() and torch.isfinite(T).all()
            assert float(occ.min()) >= 0 and float(occ.max()) <= 1
            on = occ > 0.05
            assert torch.allclose(T[:3].sum(0)[on], torch.ones(int(on.sum())), atol=1e-3)
            assert om.unique().numel() == 1                                                         # one object per field


def test_texture_tensor_follows_its_field():
    sc = _scene(3)
    f = sc._texture_field()
    torch.manual_seed(0)
    mask = torch.ones(sc.shape)
    g = sc._texture_population(f, mask, {})
    occ, T, *_ = g.dense[0]
    on = occ > 0.1
    Tm = torch.stack([torch.stack([T[0], T[3], T[4]], -1), torch.stack([T[3], T[1], T[5]], -1), torch.stack([T[4], T[5], T[2]], -1)], -2)[on]
    ax = torch.linalg.eigh(Tm)[1][..., 2]
    cos = (ax * f.movedim(0, -1)[on]).sum(-1).abs()
    assert float(cos.median()) > 0.95                                                               # within ~18 deg (dispersion <= 15)
