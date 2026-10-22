"""The engine: every output from the network's output. With the target standing in for a perfect network (an oracle)
it must give one population on a bend, two at a crossing on the right axes, the line and the sheet at a boundary; a
volume run in chunks must equal one pass."""
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
    return {"slots": slots, "mass": t["mass"], "root_mass": t["mass"].sqrt(), "kind": 30.0 * torch.nn.functional.one_hot(t["kind"].clamp_min(0), 4).movedim(-1, 1).float(),
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
    return f[:, 2::4, 2::4, 2::4]


def _angle(a, b):
    return torch.rad2deg(torch.acos((a * b).sum(0).abs().clamp(max=1.0)))


def test_bend_is_one_population(eng):
    f = arc_field((64, 64, 64), 15.0, torch.Generator(device=DEV).manual_seed(1), DEV)
    s = _sample([f], density=2.0); pop = eng.populations(oracle(target(s, clean(s))), labels=False)
    on = pop["weight"] > 0.5
    assert on.sum() > 50 and float((pop["N"][on] == 1).float().mean()) > 0.95
    err = _angle(pop["dirs_cells"][0], _cells(f))[on]
    assert float(err.median()) < 12.0, float(err.median())


@pytest.mark.parametrize("theta", [45.0, 60.0, 90.0])
def test_crossing_two_populations(eng, theta):
    g = torch.Generator(device=DEV).manual_seed(2)
    d1 = direction_field((64, 64, 64), 30.0, g, DEV); d2 = companion_field(d1, theta, 30.0, g, DEV)
    s = _sample([d1, d2], density=0.8); t = target(s, clean(s)); pop = eng.populations(oracle(t), labels=False)
    both = (t["mass"][0] > 0.03) & (t["mass"][1] > 0.03)
    both[:2] = both[-2:] = False; both[:, :2] = both[:, -2:] = False; both[:, :, :2] = both[:, :, -2:] = False
    assert float((pop["N"][both] == 2).float().mean()) > 0.85
    a0, a1 = pop["dirs_cells"][0], pop["dirs_cells"][1]; c1, c2 = _cells(d1), _cells(d2)
    err = torch.minimum(torch.maximum(_angle(a0, c1), _angle(a1, c2)), torch.maximum(_angle(a0, c2), _angle(a1, c1)))
    sel = both & (pop["N"] == 2)
    assert float(err[sel].median()) < 10.0, float(err[sel].median())


def test_dropped_population_is_bridged_free_end_is_not(eng):
    shape = (64, 64, 64); g = torch.Generator(device=DEV).manual_seed(5)
    d1 = torch.tensor([1.0, 0, 0], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    d2 = torch.tensor([0, 0, 1.0], device=DEV).view(3, 1, 1, 1).expand(3, *shape).contiguous()
    s = _sample([d1, d2], shape=shape, density=0.8); t = target(s, clean(s))
    net = oracle(t)
    z1 = (t["layer"] == 1)                                                                     # the second population (along x)
    gap = torch.zeros_like(z1); gap[:, :, :, 7:9] = True                                       # lost by the network in two cells along its own axis
    net["mass"] = torch.where(z1 & gap, torch.zeros_like(net["mass"]), net["mass"]); net["root_mass"] = net["mass"].sqrt()
    pop = eng.populations(net, labels=False)
    ex = torch.tensor([0, 0, 1.0], device=DEV).view(3, 1, 1, 1)
    near_x = torch.stack([_angle(pop["dirs_cells"][k], ex) < 15 for k in range(pop["dirs_cells"].shape[0])]) & (pop["mass"] > 0)
    inner = (slice(3, -3), slice(3, -3))
    assert float(near_x.any(0)[inner][..., 7:9].float().mean()) > 0.9                           # bridged in the gap
    end = torch.where(z1 & (torch.arange(16, device=DEV).view(1, 1, 1, 16) >= 10), torch.zeros_like(net["mass"]), net["mass"])
    net2 = dict(net); net2["mass"] = torch.where(z1, end, t["mass"]); net2["root_mass"] = net2["mass"].sqrt()
    pop2 = eng.populations(net2, labels=False)
    near_x2 = torch.stack([_angle(pop2["dirs_cells"][k], ex) < 15 for k in range(pop2["dirs_cells"].shape[0])]) & (pop2["mass"] > 0)
    assert float(near_x2.any(0)[inner][..., 12:].float().mean()) < 0.05                         # a free end is not extended


def test_boundary_line_and_sheet(eng):
    b = boundary_sample((48, 48, 48), step=60.0, contrast=30.0, seed=0, device=DEV, density=1.5)
    pop = eng.populations(oracle(target(b, clean(b))), labels=False)
    at, far = (slice(None), slice(5, 7)), (slice(None), slice(0, 1))
    assert float((pop["N"][5:7] >= 1).float().mean()) > 0.9 and float(pop["sheet_mass"][5:7].mean()) > 0.1
    assert float(pop["sheet_mass"][0].max()) < 0.02
    n = pop["sheet_normal"][:, 5:7]; ez = torch.tensor([1.0, 0, 0], device=DEV).view(3, 1, 1, 1)
    assert float(_angle(n, ez).median()) < 10.0
    v = pop["voxel_vecs"] if "voxel_vecs" in pop else eng.field(pop, (48, 48, 48), labels=False)["vecs"]
    assert float(eng.field(pop, (48, 48, 48), labels=False)["weights"][0, 20:28].mean()) > 0.1     # vectors kept across the step


def test_volume_in_chunks_equals_one_pass(tmp_path):
    torch.manual_seed(0); net = PopulationNet()
    for b in net.modules():
        if isinstance(b, Block):
            b.g.data.fill_(0.5)
    eng = PopulationEngine(net, device=DEV, reach=2)
    rng = np.random.default_rng(0); vol = (rng.random((96, 96, 96)) * 255).astype(np.uint8)
    win = (0.0, 255.0)
    root = eng.run_volume(vol, str(tmp_path / "out.zarr"), chunk=(48, 48, 48), win=win, min_fraction=-1.0, log=lambda *a: None)
    one = eng.populations(eng.network(torch.as_tensor(vol), win), labels=False)
    np.testing.assert_allclose(np.asarray(root["fraction"][:]), one["fraction"].half().cpu().numpy(), atol=2e-3)
    np.testing.assert_allclose(np.asarray(root["mass"][:]), one["mass"].half().cpu().numpy(), atol=2e-3)
    assert (np.asarray(root["N"][:]) == one["N"].cpu().numpy()).mean() > 0.999
