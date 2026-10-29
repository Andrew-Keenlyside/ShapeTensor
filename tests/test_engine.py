"""The engine: every output from the network's output. With the target standing in for a perfect network (an oracle)
it must give one population on a bend; two at crossings down to 30° on the right axes; ignore empty slots whatever
mass they hold; keep a faint population's own mass; bridge a population the network lost, never extend a free end;
give the line and the sheet at a boundary; a smooth voxel field; and a volume in chunks must equal one pass."""
import math

import numpy as np
import pytest
import torch

from shapetensor.learn.engine import PopulationEngine
from shapetensor.learn.generator import Layer, Sample, _composite, arc_field, boundary_sample, companion_field, direction_field, render_layer
from shapetensor.learn.model import Block, PopulationNet
from shapetensor.learn.targets import clean, target

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def oracle(t):
    """Network-shaped outputs from a target (one block, no batch dim)."""
    K = t["mass"].shape[0]
    slots = torch.where(t["mass"][:, None] > 0, t["slots"], torch.tensor([1, 1, 1, 0, 0, 0.0], device=DEV).view(1, 6, 1, 1, 1) / 3)
    f = t["fraction"].clamp(1e-4, 1 - 1e-4)
    return {"slots": slots, "mass": t["mass"], "root_mass": t["mass"].sqrt(), "presence_logit": torch.where(t["mass"] > 0.005, 10.0, -10.0),
            "kind": 30.0 * torch.nn.functional.one_hot(t["kind"].clamp_min(0), 4).movedim(-1, 1).float(),
            "embed": torch.nn.functional.normalize(torch.randn(K, 8, *t["mass"].shape[1:], device=DEV), dim=1),
            "fraction_logit": torch.log(f / (1 - f)), "class_logits": torch.log(t["classes"] + 1e-8), "log_size": t["size"].clamp_min(0.3).log()}


def _sample(fields, shape=(64, 64, 64), density=1.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed); ones = torch.ones(shape, dtype=torch.bool, device=DEV); layers = []
    for f in fields:
        L = Layer("line", f, 1.0, 60.0, density, ones, 50.0); L.occupancy, L.tensor = render_layer(L, shape, g, DEV); layers.append(L)
    bg = torch.full(shape, 100.0, device=DEV); img = _composite(bg, bg.clone(), layers, DEV)
    return Sample(img, layers, bg, torch.zeros(shape, dtype=torch.long, device=DEV), "flat", 1.0, "general", {}, bg.clone())


@pytest.fixture(scope="module")
def eng():
    return PopulationEngine(PopulationNet(), device=DEV)


def _cells(f):
    return f[:, 1::4, 1::4, 1::4] * 0.5 + f[:, 2::4, 2::4, 2::4] * 0.5


def _angle(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1.0)))


def _inner(x, k=2):
    m = torch.zeros_like(x, dtype=torch.bool); m[k:-k, k:-k, k:-k] = True
    return m


def _radius_cells(R, seed, shape=(64, 64, 64)):
    """The curvature radius (voxels) of arc_field's circles at each lattice cell: the distance to its axis."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    from shapetensor.learn.generator import _unit
    a = _unit(g, DEV); e = _unit(g, DEV); e = e - (e @ a) * a; e = e / e.norm()
    c = torch.tensor([(n - 1) / 2 for n in shape], device=DEV) + R * e
    ci = torch.arange(16, device=DEV, dtype=torch.float32) * 4 + 1.5
    P = torch.stack(torch.meshgrid(ci, ci, ci, indexing="ij")) - c.view(3, 1, 1, 1)
    return (P - (P * a.view(3, 1, 1, 1)).sum(0, keepdim=True) * a.view(3, 1, 1, 1)).norm(dim=0)


def test_bend_is_one_population(eng):
    f = arc_field((64, 64, 64), 15.0, torch.Generator(device=DEV).manual_seed(1), DEV)
    s = _sample([f], density=2.0); t = target(s, clean(s)); pop = eng.populations(oracle(t), labels=False)
    on = (t["mass"][0] > 0.05) & _inner(t["mass"][0]) & (_radius_cells(15.0, 1) >= 8.0)          # a line at the window's scale: radius of curvature >= 8 voxels
    assert on.sum() > 50 and float((pop["N"][on] == 1).float().mean()) > 0.95
    err = _angle(pop["dirs_cells"][0], torch.nn.functional.normalize(_cells(f), dim=0))[on]
    assert float(err.median()) < 12.0, float(err.median())


@pytest.mark.parametrize("theta", [30.0, 45.0, 60.0, 90.0])
def test_crossing_two_populations(eng, theta):
    g = torch.Generator(device=DEV).manual_seed(2)
    d1 = direction_field((64, 64, 64), 30.0, g, DEV); d2 = companion_field(d1, theta, 30.0, g, DEV)
    s = _sample([d1, d2], density=0.8); t = target(s, clean(s)); pop = eng.populations(oracle(t), labels=False)
    both = (t["mass"][0] > 0.03) & (t["mass"][1] > 0.03) & _inner(t["mass"][0])
    assert float((pop["N"][both] == 2).float().mean()) > 0.9, float((pop["N"][both] == 2).float().mean())
    a0, a1 = pop["dirs_cells"][0], pop["dirs_cells"][1]
    c1, c2 = torch.nn.functional.normalize(_cells(d1), dim=0), torch.nn.functional.normalize(_cells(d2), dim=0)
    err = torch.minimum(torch.maximum(_angle(a0, c1), _angle(a1, c2)), torch.maximum(_angle(a0, c2), _angle(a1, c1)))
    sel = both & (pop["N"] == 2)
    assert float(err[sel].median()) < 10.0, float(err[sel].median())


def test_empty_slots_are_ignored_whatever_their_mass(eng):
    f = torch.tensor([0.0, 0, 1.0], device=DEV).view(3, 1, 1, 1).expand(3, 48, 48, 48).contiguous()
    s = _sample([f], shape=(48, 48, 48), density=1.0); t = target(s, clean(s)); net = oracle(t)
    empty = t["mass"] <= 0.005
    g = torch.Generator(device=DEV).manual_seed(3)
    junk = torch.nn.functional.normalize(torch.randn(net["slots"].shape[0], 3, *t["mass"].shape[1:], device=DEV, generator=g), dim=1)
    from shapetensor.learn.generator import outer6
    net["slots"] = torch.where(empty[:, None], outer6(junk.movedim(1, -1)).movedim(-1, 1), net["slots"])
    net["mass"] = torch.where(empty, torch.full_like(net["mass"], 0.03), net["mass"])         # a soft mass head never says zero
    net["presence_logit"] = torch.where(empty, torch.full_like(net["mass"], -3.0), net["presence_logit"])
    net["kind"] = torch.where(empty[:, None], 30.0 * torch.nn.functional.one_hot(torch.zeros_like(t["kind"]), 4).movedim(-1, 1).float(), net["kind"])
    pop = eng.populations(net, labels=False)
    on = (t["mass"][0] > 0.05) & _inner(t["mass"][0])
    assert float((pop["N"][on] == 1).float().mean()) > 0.95


def test_faint_population_keeps_its_mass(eng):
    f = torch.tensor([0.0, 0.6, 0.8], device=DEV).view(3, 1, 1, 1).expand(3, 48, 48, 48).contiguous()
    s = _sample([f], shape=(48, 48, 48), density=0.08); t = target(s, clean(s))
    pop = eng.populations(oracle(t), labels=False)
    sel = (t["mass"][0] > 0.012) & (t["mass"][0] < 0.05) & _inner(t["mass"][0])
    assert sel.sum() > 20
    assert float((pop["N"][sel] == 1).float().mean()) > 0.9
    assert float((pop["mass"][0][sel] / t["mass"][0][sel]).median()) < 1.6                     # absolute, not scaled up to a weight


def test_dropped_population_is_bridged_free_end_is_not(eng):
    shape = (64, 64, 64)
    d1 = torch.tensor([1.0, 0, 0], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    d2 = torch.tensor([0, 0, 1.0], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    s = _sample([d1, d2], shape=shape, density=0.8); t = target(s, clean(s))
    z1 = (t["layer"] == 1)                                                                     # the second population (along x)
    gap = torch.zeros_like(z1); gap[:, :, :, 7:9] = True                                       # lost by the network in two cells along its own axis
    net = oracle(t); lost = z1 & gap
    net["mass"] = torch.where(lost, torch.zeros_like(net["mass"]), net["mass"]); net["presence_logit"] = torch.where(lost, -10.0, net["presence_logit"])
    pop = eng.populations(net, labels=False)
    ex = torch.tensor([0, 0, 1.0], device=DEV).view(3, 1, 1, 1)
    near_x = torch.stack([_angle(pop["dirs_cells"][k], ex) < 15 for k in range(pop["dirs_cells"].shape[0])]) & (pop["mass"] > 0)
    inner = (slice(3, -3), slice(3, -3))
    assert float(near_x.any(0)[inner][..., 7:9].float().mean()) > 0.9                           # bridged in the gap
    end = z1 & (torch.arange(16, device=DEV).view(1, 1, 1, 16) >= 10)
    net2 = oracle(t); net2["mass"] = torch.where(end, torch.zeros_like(net2["mass"]), net2["mass"]); net2["presence_logit"] = torch.where(end, -10.0, net2["presence_logit"])
    pop2 = eng.populations(net2, labels=False)
    near_x2 = torch.stack([_angle(pop2["dirs_cells"][k], ex) < 15 for k in range(pop2["dirs_cells"].shape[0])]) & (pop2["mass"] > 0)
    assert float(near_x2.any(0)[inner][..., 12:].float().mean()) < 0.05                         # a free end is not extended


def test_boundary_line_and_sheet(eng):
    b = boundary_sample((48, 48, 48), step=60.0, contrast=30.0, seed=0, device=DEV, density=1.5)
    pop = eng.populations(oracle(target(b, clean(b))), labels=False)
    assert float((pop["N"][5:7] >= 1).float().mean()) > 0.9 and float(pop["sheet_mass"][5:7].mean()) > 0.1
    assert float(pop["sheet_mass"][0].max()) < 0.02
    n = pop["sheet_normal"][:, 5:7]; ez = torch.tensor([1.0, 0, 0], device=DEV).view(3, 1, 1, 1)
    assert float(_angle(n, ez).median()) < 10.0
    fld = eng.field(pop, (48, 48, 48), labels=False)
    assert float(fld["weights"][0, 20:28].mean()) > 0.1                                         # vectors kept across the step


def test_voxel_field_is_smooth(eng):
    g = torch.Generator(device=DEV).manual_seed(6)
    f = direction_field((64, 64, 64), 40.0, g, DEV)
    s = _sample([f], density=2.0); pop = eng.populations(oracle(target(s, clean(s))), labels=False)
    v = eng.field(pop, (64, 64, 64), labels=False)
    w = v["weights"][0]; vec = v["vecs"][0]
    sl = (slice(None), slice(12, -12), slice(12, -12), slice(12, -13))
    a = vec[sl]; b = vec[:, 12:-12, 12:-12, 13:-12]
    ok = (w[12:-12, 12:-12, 12:-13] > 0.05) & (w[12:-12, 12:-12, 13:-12] > 0.05)
    step = _angle(a, b)[ok]
    assert ok.sum() > 1000 and float(step.quantile(0.99)) < 4.0, float(step.quantile(0.99))     # no jumps at cell borders


def test_volume_in_chunks_equals_one_pass(tmp_path):
    torch.manual_seed(0); net = PopulationNet()
    for b in net.modules():
        if isinstance(b, Block):
            b.g.data.fill_(0.5)
    with torch.no_grad():
        net.slots.bias.view(net.K, net.per_slot)[:, 7] = 3.0                                    # presence on, so the post-processing has populations to carry
    eng = PopulationEngine(net, device=DEV, exact=True)
    rng = np.random.default_rng(0); vol = (rng.random((128, 128, 128)) * 255).astype(np.uint8)
    win = (0.0, 255.0)
    root = eng.run_volume(vol, str(tmp_path / "out.zarr"), chunk=(64, 64, 64), win=win, min_fraction=-1.0, log=lambda *a: None)
    one = eng.populations(eng.network(torch.as_tensor(vol), win), labels=False)
    assert int(one["N"].sum()) > 0
    np.testing.assert_allclose(np.asarray(root["fraction"][:]), one["fraction"].half().cpu().numpy(), atol=2e-3)
    np.testing.assert_allclose(np.asarray(root["mass"][:]), one["mass"].half().cpu().numpy(), atol=2e-3)
    assert (np.asarray(root["N"][:]) == one["N"].cpu().numpy()).mean() > 0.999
