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
    return {"slots": slots, "mass": T["mass"], "root_mass": T["mass"].sqrt(), "kind": kind, "presence_logit": torch.where(T["mass"] > 0.005, 30.0, -30.0),
            "embed": torch.nn.functional.normalize(torch.randn(*T["mass"].shape[:2], 8, *T["mass"].shape[2:], device=DEV), dim=2),
            "fraction_logit": torch.log(f / (1 - f)), "class_logits": torch.log(T["classes"] + 1e-8), "log_size": T["size"].clamp_min(0.3).log(),
            "mixture": (T["slots"] * T["mass"][:, :, None]).sum(1) / T["mass"].sum(1).clamp_min(1e-9)[:, None]}


@pytest.fixture(scope="module")
def sample_target():
    s, d = draw((64, 64, 64), seed=3, device=DEV, crossing=True)
    return s, d, _batch_target(target(s, d))


def test_exact_prediction_costs_nothing(sample_target):
    _, _, T = sample_target
    P = _exact_pred(T)
    _, parts = loss(P, T, hemisphere(160, DEV), margin=1)
    assert parts["shape"] < 1e-6 and parts["mass"] < 1e-6 and parts["presence"] < 1e-6 and parts["kind"] < 1e-3 and parts["hellinger"] < 1e-5, parts


def test_slots_are_rank_ordered(sample_target):
    """Slot k predicts the k-th heaviest population: exchanging predicted slots whose targets are empty changes
    nothing; exchanging the two heaviest populations' slots costs (where their masses do not tie)."""
    _, _, T = sample_target
    P = _exact_pred(T)
    w = {"identity": 0.0}
    base, _ = loss(P, T, hemisphere(160, DEV), margin=1, weights=w)
    K = P["mass"].shape[1]
    swap_empty = list(range(K)); swap_empty[-1], swap_empty[-2] = swap_empty[-2], swap_empty[-1]
    keys = ("slots", "mass", "root_mass", "presence_logit", "kind", "embed")
    both_empty = (T["mass"][:, -1] <= 0.005) & (T["mass"][:, -2] <= 0.005)                          # swap only where both targets are empty
    sw = lambda v: torch.where(both_empty.view(both_empty.shape[0], 1, *([1] * (v.dim() - 5)), *both_empty.shape[1:]) if v.dim() > 5 else both_empty[:, None], v[:, swap_empty], v)
    Pe = {k: (sw(v) if k in keys else v) for k, v in P.items()}
    b, _ = loss(Pe, T, hemisphere(160, DEV), margin=1, weights=w)
    assert abs(float(b) - float(base)) < 1e-3 * max(1.0, abs(float(base)))                       # the last two slots are empty almost everywhere
    swap_top = list(range(K)); swap_top[0], swap_top[1] = 1, 0
    Pt = {k: (v[:, swap_top] if k in keys else v) for k, v in P.items()}
    c, _ = loss(Pt, T, hemisphere(160, DEV), margin=1, weights=w)
    assert float(c) > float(base) + 1e-3


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
        dr["descriptors"] = apply_cube(d["descriptors"], perm, flips)
        T2 = target(s, dr)
        for k in ("fraction", "classes"):
            assert torch.allclose(T2[k], apply_cube(T[k][0], perm, flips), atol=1e-4), k
        A = apply_cube(T["mass"][0], perm, flips)
        assert torch.allclose(T2["mass"].sum(0), A.sum(0), atol=1e-4)                                # the cell's populations' total: exact
        assert float(((T2["mass"] - A).abs() > 1e-4).any(0).float().mean()) < 1e-3                  # slots: exact up to ties at the 20° merge threshold
        m = (T2["mass"] > 0.01) & ((T2["mass"] - A).abs() < 1e-4).all(0)[None]
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
    c = 8; w = torch.zeros(64, 64, 64, device=DEV); g = torch.Generator(device=DEV).manual_seed(0)
    for _ in range(16):                                                                            # random inputs: the max's choices unbiased (a tied input sends it all one way)
        x = torch.rand(1, 1, 64, 64, 64, device=DEV, generator=g, requires_grad=True)
        m(x)["fraction_logit"][0, c, c, c].backward(); w += x.grad[0, 0].abs()
    pos = torch.arange(64, device=DEV, dtype=torch.float32)
    for ax in range(3):
        prof = w.sum(tuple(i for i in range(3) if i != ax))
        assert abs(float((prof * pos).sum() / prof.sum()) - (4 * c + 1.5)) < 0.25


def test_receptive_field_fits_the_window():
    rf = receptive_field(PopulationNet().to(DEV))
    assert 25 <= rf <= 64, rf


def test_training_runs_end_to_end(tmp_path):
    from shapetensor.learn.train import train
    train(str(tmp_path), steps=4, size=48, batch=2, accum=2, buffer=3, refresh=1, val_n=1, val_every=4, margin=1, warmup=2, device=DEV)
    ck = torch.load(tmp_path / "model.pt", map_location=DEV)
    assert ck["step"] == 4 and (tmp_path / "log.jsonl").exists()


def test_shape_aware_matching():
    """With ``DISTINCT`` set, populations of distinct shape (a blob and a line) may take each other's slots; two lines
    keep their rank order. Without it, every present slot keeps its rank."""
    from shapetensor.learn import losses as L
    from shapetensor.learn.generator import eye6
    from shapetensor.learn.losses import match
    K = 6
    blob = eye6("cpu", ()) / 3; lx = line_tensor(torch.tensor([1.0, 0.0, 0.0])); ly = line_tensor(torch.tensor([0.0, 1.0, 0.0]))
    St = torch.zeros(1, K, 6, 1, 1, 1); mt = torch.zeros(1, K, 1, 1, 1); kt = torch.full((1, K, 1, 1, 1), -1)
    for k, (T_, m_, k_) in enumerate(((blob, 0.3, 2), (lx, 0.1, 0), (ly, 0.05, 0))):
        St[0, k, :, 0, 0, 0] = T_; mt[0, k] = m_; kt[0, k] = k_
    tgt = {"slots": St, "mass": mt, "kind": kt}

    def pred(order):                                                                       # predicted slot j holds target order[j]
        S = torch.zeros_like(St); rm = torch.zeros_like(mt); kl = torch.full((1, K, 4, 1, 1, 1), -10.0); pl = torch.full_like(mt, -10.0)
        for j, t in enumerate(order):
            S[0, j] = St[0, t]; rm[0, j] = mt[0, t].sqrt(); kl[0, j, max(int(kt[0, t]), 0)] = 10.0; pl[0, j] = 10.0
        for j in range(len(order), K):
            S[0, j] = blob[:, None, None, None]
        return {"slots": S, "root_mass": rm, "kind": kl, "presence_logit": pl}

    old = L.DISTINCT
    try:
        L.DISTINCT = None
        c_rank, _, _ = match(pred([1, 0, 2]), tgt)                                         # line before blob: the rank rule refuses
        assert float(c_rank) > 0.05
        L.DISTINCT = 0.2
        c, _, perm = match(pred([1, 0, 2]), tgt)                                           # distinct shapes exchange freely
        assert float(c) < 1e-3 and perm[0, :3, 0, 0, 0].tolist() == [1, 0, 2]
        c2, _, _ = match(pred([2, 0, 1]), tgt)                                             # the lighter line first: order among lines kept
        assert float(c2) > 0.05
    finally:
        L.DISTINCT = old
