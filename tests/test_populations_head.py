"""The populations head (set prediction of elliptical ODFs) and its loss (CPU, seconds)."""
import torch

from shapetensor.learn import entity_train as ET
from shapetensor.learn.model import PopulationNet


def _targets(B=2, Q=5, E=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(B, Q, E, 3, generator=g); a = a / a.norm(dim=-1, keepdim=True)
    M = 0.9 * a[..., :, None] * a[..., None, :] + 0.1 * torch.eye(3) / 3
    t6 = torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)
    mass = torch.rand(B, Q, E, generator=g); mass[..., 4:] = 0.0                                   # at most 4 entities per query
    return {"e_mass": mass, "e_vis": torch.ones_like(mass), "e_node": torch.zeros(B, Q, E, dtype=torch.long), "e_T6": t6}


def test_head_outputs_are_valid_ellipses():
    torch.manual_seed(0)
    dec = PopulationNet.from_args({"head": "entities", "pop_k": 4}).ent
    with torch.no_grad():
        for m in dec.pop:
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(0, 0.2); m.bias.normal_(0, 0.2)
        out = dec.populations(torch.randn(3, 7, dec.pop[0].in_features))
    T = out["pop_T"]
    assert T.shape == (3, 7, 4, 3, 3) and out["pop_share"].shape == (3, 7, 5)
    assert torch.allclose(T.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(3, 7, 4), atol=1e-5)
    assert bool((torch.linalg.eigvalsh(T) > -1e-6).all())
    assert torch.allclose(out["pop_share"].sum(-1), torch.ones(3, 7), atol=1e-5)


def test_targets_shares_sum_to_at_most_one():
    s, E, valid, other, has = ET.population_targets(_targets(), 4)
    assert torch.allclose(s.sum(-1) + other, torch.ones_like(other), atol=1e-5)
    assert bool((s[~valid] == 0).all())


def test_perfect_prediction_and_slot_order_invariance():
    T = _targets()
    s, E, valid, other, has = ET.population_targets(T, 4)
    ent = {"pop_exist": valid.float().clamp(1e-4, 1 - 1e-4), "pop_share": torch.cat([s, other[..., None]], -1), "pop_T": E}
    l0 = ET.population_loss(ent, T)
    assert float(l0["pop_T"]) < 1e-8 and float(l0["pop_share"]) < 1e-6
    perm = torch.tensor([2, 0, 3, 1])                                                               # the same set, slots shuffled
    ent2 = {"pop_exist": ent["pop_exist"][..., perm], "pop_share": torch.cat([s[..., perm], other[..., None]], -1), "pop_T": E[..., perm, :, :]}
    l1 = ET.population_loss(ent2, T)
    assert abs(float(l1["pop"]) - float(l0["pop"])) < 1e-5


def test_direction_moment_readout_turns_with_the_directions():
    torch.manual_seed(0)
    dec = PopulationNet.from_args({"head": "entities", "pop_k": 4, "pop_dir": 48, "odirect": 64}).ent.eval()
    p = torch.randn(5, dec.pop[0].in_features); U = dec.pop_quadrature(1, "cpu")[0]; cu = torch.randn(5, U.shape[0], 32)
    with torch.no_grad():
        T0 = dec.populations(p, (U, cu))["pop_T"]
        assert torch.allclose(T0, torch.eye(3).expand_as(T0) / 3, atol=0.02)                      # uniform weights at the start: isotropic
        for m in dec.pop_u:
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(0, 0.5)
        T = dec.populations(p, (U, cu))["pop_T"]
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        TR_ = dec.populations(p, (U @ R.T, cu))["pop_T"]                                            # the same codes along turned directions
    assert torch.allclose(T.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(5, 4), atol=1e-5)
    assert bool((torch.linalg.eigvalsh(T) > -1e-6).all())
    assert torch.allclose(TR_, R @ T @ R.T, atol=1e-5)


def test_sequential_slots_turn_with_the_directions():
    torch.manual_seed(1)
    dec = PopulationNet.from_args({"head": "entities", "pop_k": 4, "pop_dir": 48, "odirect": 64, "pop_seq": True}).ent.eval()
    p = torch.randn(5, dec.pop[0].in_features); U = dec.pop_quadrature(1, "cpu")[0]; cu = torch.randn(5, U.shape[0], 32)
    with torch.no_grad():
        for m in dec.pop_u:
            if isinstance(m, torch.nn.Linear):
                m.weight.normal_(0, 0.5)
        T = dec.populations(p, (U, cu))["pop_T"]
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        TR_ = dec.populations(p, (U @ R.T, cu))["pop_T"]
        Tb = dec.populations(p[None].expand(2, -1, -1), (U[None].expand(2, -1, -1), cu[None].expand(2, -1, -1, -1)))["pop_T"]   # batched
    assert T.shape == (5, 4, 3, 3) and torch.allclose(T.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(5, 4), atol=1e-5)
    assert torch.allclose(TR_, R @ T @ R.T, atol=1e-5) and torch.allclose(Tb[0], T, atol=1e-5)


def test_sequential_mode2_shares_rotation_and_batching():
    torch.manual_seed(2)
    dec = PopulationNet.from_args({"head": "entities", "pop_k": 4, "pop_dir": 48, "odirect": 64, "pop_seq": 2}).ent.eval()
    p = torch.randn(5, dec.pop[0].in_features); U = dec.pop_quadrature(1, "cpu")[0]; cu = torch.randn(5, U.shape[0], 32)
    with torch.no_grad():
        for net in (dec.pop_u, dec.pop_s, dec.pop):
            for m in net:
                if isinstance(m, torch.nn.Linear):
                    m.weight.normal_(0, 0.3)
        out = dec.populations(p, (U, cu))
        R = torch.linalg.qr(torch.randn(3, 3))[0]
        TR_ = dec.populations(p, (U @ R.T, cu))["pop_T"]
        ob = dec.populations(p[None].expand(2, -1, -1), (U[None].expand(2, -1, -1), cu[None].expand(2, -1, -1, -1)))
    T, sh, bg = out["pop_T"], out["pop_share"], out["pop_bg"]
    assert T.shape == (5, 4, 3, 3) and sh.shape == (5, 5) and bg.shape == (5,)
    assert torch.allclose(sh.sum(-1) + bg, torch.ones(5), atol=1e-5) and bool((sh >= 0).all())
    assert torch.allclose(T.diagonal(dim1=-2, dim2=-1).sum(-1), torch.ones(5, 4), atol=1e-5)
    assert torch.allclose(TR_, R @ T @ R.T, atol=1e-5)
    assert torch.allclose(ob["pop_T"][1], T, atol=1e-5) and torch.allclose(ob["pop_bg"][0], bg, atol=1e-6)


def test_background_apart_from_the_slots(monkeypatch):
    monkeypatch.setattr(ET, "POP_BG", True)
    T = _targets(); bgf = torch.zeros_like(T["e_mass"], dtype=torch.bool); bgf[..., 0] = True; T["e_bg"] = bgf
    s, E, valid, other, has = ET.population_targets(T, 4); bg = ET.population_bg(T)
    m = T["e_mass"]; assert torch.allclose(bg, m[..., 0] / m.sum(-1), atol=1e-5)
    assert torch.allclose(s.sum(-1) + other + bg, torch.ones_like(bg), atol=1e-5)
    assert float(bg.min()) > 0 and float((s > 0).sum(-1).max()) <= 3                             # the background never a slot


def test_ordered_matching_swaps_only_near_equal_shares(monkeypatch):
    monkeypatch.setattr(ET, "POP_BG", True)
    T = _targets(B=1, Q=1); T["e_bg"] = torch.zeros_like(T["e_mass"], dtype=torch.bool)
    T["e_mass"] = torch.tensor([[[0.5, 0.2, 0.19, 0.05, 0.0, 0.0]]])                             # ranks: 0.5 | 0.2 ~ 0.19 | 0.05
    s, E, valid, other, has = ET.population_targets(T, 4)
    perm = torch.tensor([3, 2, 1, 0])                                                             # ellipses that fit the reversed order
    ent = {"pop_exist": torch.full((1, 1, 4), 0.5), "pop_share": torch.full((1, 1, 5), 0.2), "pop_T": E[..., perm, :, :]}
    m = ET.population_match(ent, T)["match"][0, 0].tolist()
    assert m[0] == 0 and m[3] == 3 and sorted(m[1:3]) == [1, 2]                                   # far shares keep their rank
    assert m[1:3] == [2, 1]                                                                       # near-equal ones follow the ellipses
