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
