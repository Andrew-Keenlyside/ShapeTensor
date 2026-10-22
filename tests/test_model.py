"""The population model, its loss and its training: an exact prediction costs nothing, the loss ignores slot order and
weighs every kind's orientation error equally per degree, the cube-group augmentation is exact on the lattice, a
chunked forward pass equals one pass, and a short training run goes end to end."""
import math

import pytest
import torch

from shapetensor.learn.generator import draw, line_tensor, sheet_tensor
from shapetensor.learn.losses import KIND_WEIGHT, WF, apply_cube, cube_transform, loss
from shapetensor.learn.model import Block, PopulationNet, receptive_field
from shapetensor.learn.targets import target
from shapetensor.lift import hemisphere

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _batch_target(t):
    keys = ("slots", "mass", "kind", "layer", "fraction", "classes", "size", "size_mask")
    return {k: t[k][None] for k in keys}


def _exact_pred(T):
    K = T["mass"].shape[1]
    slots = torch.where(T["mass"][:, :, None] > 0, T["slots"], torch.tensor([1, 1, 1, 0, 0, 0.0], device=DEV).view(1, 1, 6, 1, 1, 1) / 3)
    kind = 30.0 * torch.nn.functional.one_hot(T["kind"].clamp_min(0), 4).movedim(-1, 2).float()
    f = T["fraction"].clamp(1e-4, 1 - 1e-4)
    return {"slots": slots, "mass": T["mass"], "root_mass": T["mass"].sqrt(), "kind": kind,
            "embed": torch.nn.functional.normalize(torch.randn(*T["mass"].shape[:2], 8, *T["mass"].shape[2:], device=DEV), dim=2),
            "fraction_logit": torch.log(f / (1 - f)), "class_logits": torch.log(T["classes"] + 1e-8), "log_size": T["size"].clamp_min(0.3).log()}


@pytest.fixture(scope="module")
def sample_target():
    s, d = draw((64, 64, 64), seed=3, device=DEV, crossing=True)
    return s, d, _batch_target(target(s, d))


def test_exact_prediction_costs_nothing(sample_target):
    _, _, T = sample_target
    P = _exact_pred(T)
    _, parts = loss(P, T, hemisphere(160, DEV), margin=1)
    assert parts["shape"] < 1e-6 and parts["mass"] < 1e-6 and parts["kind"] < 1e-3 and parts["hellinger"] < 1e-6, parts


def test_loss_ignores_slot_order(sample_target):
    _, _, T = sample_target
    P = _exact_pred(T)
    P = {k: (v + 0.05 * torch.randn_like(v) if k in ("root_mass",) else v) for k, v in P.items()}
    K = P["mass"].shape[1]; perm = [(i * 5 + 2) % K for i in range(K)]                          # a permutation of the K slots
    Pp = {k: (v[:, perm] if k in ("slots", "mass", "root_mass", "kind", "embed") else v) for k, v in P.items()}
    a, _ = loss(P, T, hemisphere(160, DEV), margin=1); b, _ = loss(Pp, T, hemisphere(160, DEV), margin=1)
    assert abs(float(a) - float(b)) < 1e-4 * max(1.0, abs(float(a)))


def test_shape_cost_balanced_across_kinds():
    def rot(v, deg):
        t = math.radians(deg); return torch.tensor([v[0] * math.cos(t) - v[1] * math.sin(t), v[0] * math.sin(t) + v[1] * math.cos(t), v[2]])
    d0 = torch.tensor([1.0, 0.0, 0.0]); d1 = rot(d0, 10.0)
    c = lambda A, B, k: float(((A - B) ** 2 * WF).sum() * KIND_WEIGHT[k])
    line = c(line_tensor(d0), line_tensor(d1), 0); sheet = c(sheet_tensor(d0), sheet_tensor(d1), 1)
    assert 0.8 < line / sheet < 1.25, (line, sheet)


def test_cube_augmentation_is_exact(sample_target):
    s, d, T = sample_target
    g = torch.Generator().manual_seed(4)
    for _ in range(3):
        perm, flips = cube_transform(g)
        dr = {"occupancy": apply_cube(d["occupancy"], perm, flips), "tensors": apply_cube(d["tensors"], perm, flips, tensor6=True),
              "visibility": apply_cube(d["visibility"], perm, flips), "orientation_visibility": apply_cube(d["orientation_visibility"], perm, flips),
              "stretch": apply_cube(d["stretch"], perm, flips)}
        T2 = target(s, dr)
        for k in ("mass", "fraction", "classes"):
            assert torch.allclose(T2[k], apply_cube(T[k][0], perm, flips), atol=1e-4), k
        m = T2["mass"] > 0.01
        a = apply_cube(T["slots"][0], perm, flips, tensor6=True)
        assert torch.allclose(T2["slots"].movedim(1, -1)[m], a.movedim(1, -1)[m], atol=1e-3)


def test_chunked_forward_equals_one_pass():
    tf32 = torch.backends.cudnn.allow_tf32; torch.backends.cudnn.allow_tf32 = False             # exact arithmetic: TF32 differs by algorithm (~1e-4)
    torch.manual_seed(0)
    m = PopulationNet().to(DEV).eval()
    for b in m.modules():
        if isinstance(b, Block):
            b.g.data.fill_(0.5)
    rf = receptive_field(m); halo_cells = math.ceil((rf / 2) / 4)
    x = torch.rand(1, 1, 96, 96, 96, device=DEV)
    with torch.no_grad():
        full = m(x); part = m(x[:, :, :64, :64, :64])
    n = 16 - halo_cells
    for k in ("slots", "mass", "fraction_logit", "class_logits"):
        a = full[k][..., :n, :n, :n]; b = part[k][..., :n, :n, :n]
        assert torch.allclose(a, b, atol=1e-6), (k, float((a - b).abs().max()))
    torch.backends.cudnn.allow_tf32 = tf32


def test_lattice_cells_centred_on_the_target_cells():
    m = PopulationNet().to(DEV)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.Conv3d):
                mod.weight.fill_(1.0 / mod.weight[0].numel()); mod.bias.zero_()                      # symmetric kernels: the response is centred on the cell
            if isinstance(mod, Block):
                mod.g.fill_(1.0)
    x = torch.zeros(1, 1, 64, 64, 64, device=DEV, requires_grad=True)
    c = 8; m(x)["fraction_logit"][0, c, c, c].backward()
    w = x.grad[0, 0].abs(); pos = torch.arange(64, device=DEV, dtype=torch.float32)
    for ax in range(3):
        prof = w.sum(tuple(i for i in range(3) if i != ax))
        assert abs(float((prof * pos).sum() / prof.sum()) - (4 * c + 1.5)) < 0.05


def test_receptive_field_fits_the_window():
    rf = receptive_field(PopulationNet().to(DEV))
    assert 25 <= rf <= 48, rf


def test_training_runs_end_to_end(tmp_path):
    from shapetensor.learn.train import train
    train(str(tmp_path), steps=4, size=48, batch=2, buffer=3, refresh=1, val_n=1, val_every=4, margin=1, device=DEV)
    ck = torch.load(tmp_path / "model.pt", map_location=DEV)
    assert ck["step"] == 4 and (tmp_path / "log.jsonl").exists()
