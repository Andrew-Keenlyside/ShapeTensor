"""The synthesis for the population model: every voxel has a state, the fields are unit, every modality profile renders,
the boundary case carries both states at the step."""
import pytest
import torch

from shapetensor.learn.generator import MODALITIES, boundary_sample, nuisance, sample, target, westin
from shapetensor.lift import hemisphere

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SHAPE = (40, 40, 40)


def test_full_volume_states():
    for style in ("regions", "tissue", "cells"):
        s = sample(SHAPE, k_layers=2, seed=1, device=DEV, style=style, scale=0.7)
        occ = torch.stack([L.occupancy for L in s.layers])
        assert occ.sum(0).min() > 0.99, style                                                     # no voxel without a state
        assert s.layers[-2].kind == "sheet" and s.layers[-1].kind == "blob" and s.layers[-1].kappa == 0.0
        for L in s.layers:
            assert torch.allclose(L.field.norm(dim=0), torch.ones(SHAPE, device=DEV), atol=1e-4)


@pytest.mark.parametrize("modality", list(MODALITIES))
def test_modalities_render(modality):
    s = sample(SHAPE, k_layers=2, seed=2, device=DEV, scale=0.7)
    d = nuisance(s, seed=3, device=DEV, modality=modality)
    assert d["image"].shape == SHAPE and torch.isfinite(d["image"]).all() and d["record"]["modality"] == modality
    assert d["occupancy"].shape == (len(s.layers),) + SHAPE and d["fields"].shape == (len(s.layers), 3) + SHAPE
    assert torch.allclose(d["fields"].norm(dim=1), torch.ones_like(d["occupancy"]), atol=1e-3)


def test_target_windows_and_shapes():
    g = hemisphere(60, DEV)
    s = sample(SHAPE, k_layers=1, seed=4, device=DEV, scale=0.7); d = nuisance(s, seed=5, device=DEV, modality="mixed")
    t = target(s, d, g, step=4, sigma_w=(4.0, 8.0))
    assert t["odf"].shape == (2, 60, 10, 10, 10) and t["size"].shape == (10, 10, 10) and t["kappa"].shape == (10, 10, 10)
    assert (t["odf"] >= 0).all() and t["windows"] == (4.0, 8.0)
    w = westin(t["odf"][0], g)
    assert torch.allclose(w.sum(0), torch.ones_like(w[0]), atol=1e-4)


def test_boundary_has_both_states():
    g = hemisphere(160, DEV)
    b = boundary_sample((48, 48, 48), step=60.0, contrast=30.0, seed=0, device=DEV, density=1.0)
    d = {"image": b.image, "occupancy": torch.stack([L.occupancy for L in b.layers]), "fields": torch.stack([L.field for L in b.layers]), "record": {}}
    t = target(b, d, g, step=4, sigma_w=4.0); w = westin(t["odf"], g)
    at, away = w[:, 6], w[:, 1]                                                                # the lattice row at the step (z = 24) and one far from it
    assert t["classes"][3, 6].mean() > 0.3 and t["classes"][0, 6].mean() > 0.05                # the interface and the line share the step's cells
    assert at[1].mean() > 0.15 and away[0].mean() > 0.6 and away[1].mean() < 0.1              # planar at the step; the line alone elsewhere
    line = b.layers[0].field[:, 0, 0, 0]; i_line = int((g @ line).abs().argmax()); ring = (g @ line).abs() < 0.15
    o = t["odf"][:, 6].mean((1, 2))
    assert o[i_line] > o[ring].max()                                                           # the line's lobe stands above the interface's ring
    assert t["fraction"][1].mean() < 0.3 and t["fraction"][6].mean() > t["fraction"][1].mean()
    assert torch.allclose(t["classes"].sum(0), torch.ones_like(t["fraction"]), atol=1e-3)
