"""Acquisition at generation time (generator.nuisance): serial-section drift, lost sections and slabs, knife chatter,
light-sheet shadows, depth attenuation, the CTF, a low dose, the cryo-ET missing wedge, saturation; the new modality
profiles (em, fluorescence, cryoet) end to end; a lost slab asks for nothing in the targets."""
import math

import numpy as np
import pytest
import torch

from shapetensor.learn import generator as G

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEW = ("slabs", "drift", "chatter", "lsstripes", "depth", "lowdose", "wedge", "ctf", "saturate")


def _only(key: str):
    """A profile with only ``key`` on (cross-mixing off), registered under a temporary name."""
    prof = {k: 0.0 for k in G.MODALITIES["hipct"]}; prof[key] = 1.0
    return prof


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(G, "CROSS_MIX", 0.0); monkeypatch.setattr(G, "CROSS_KEY", {})
    monkeypatch.setattr(G, "SIGNAL_REDRAW", 0.0)                                                    # (the signal floor's half-strength redraw
    yield                                                                                           # may leave the one nuisance out)
    for k in list(G.MODALITIES):
        if k.startswith("_only_"):
            del G.MODALITIES[k]


@pytest.mark.parametrize("key", NEW)
def test_each_new_nuisance_runs_and_is_recorded(key, isolated):
    G.MODALITIES["_only_" + key] = _only(key)
    s = G.sample((40, 40, 40), seed=3, device=DEV)
    rec_key = {"missing": "missing_sections"}.get(key, key)
    d = G.nuisance(s, seed=5, device=DEV, strength=1.0, modality="_only_" + key, warp=False, fast=True)
    assert torch.isfinite(d["image"]).all() and float(d["image"].min()) >= 0 and float(d["image"].max()) <= 255
    assert rec_key in d["record"], (key, d["record"].keys())
    assert d["visibility"].min() >= 0 and d["visibility"].max() <= 1


def test_missing_sections_have_four_fills(isolated):
    G.MODALITIES["_only_missing"] = _only("missing"); modes = set()
    s = G.sample((40, 40, 40), seed=1, device=DEV)
    for seed in range(12):
        d = G.nuisance(s, seed=seed, device=DEV, strength=1.0, modality="_only_missing", warp=False, fast=True)
        modes |= set(d["record"]["missing_sections"][1])
    assert len(modes) >= 3


def test_a_lost_slab_asks_for_nothing_inside(isolated, monkeypatch):
    """A blank slab of sections (at most 12 % of the axis: 8+ sections need a 96-voxel block): the visibility inside
    it is far below the visibility outside."""
    G.MODALITIES["_only_slabs"] = _only("slabs")
    s = G.sample((96, 96, 96), seed=7, device=DEV)
    found = False
    for seed in range(40):
        d = G.nuisance(s, seed=seed, device=DEV, strength=1.0, modality="_only_slabs", warp=False, fast=True)
        slabs = [sl for sl in d["record"]["slabs"] if sl[2] in ("mean", "low", "high") and sl[1] >= 8]
        if not slabs:
            continue
        a0, w, _ = slabs[0]; ax = d["record"]["axis"]
        sh = torch.tensor([L.role == "shape" for L in s.layers], device=DEV)                     # the shapes (not the isotropic remainder, always seen)
        vis = (d["visibility"][sh] * d["occupancy"][sh]).sum(0).movedim(ax, 0)                  # visible shape volume per voxel
        occ = d["occupancy"][sh].sum(0).movedim(ax, 0)
        inside = vis[a0 + 2:a0 + w - 2].sum() / occ[a0 + 2:a0 + w - 2].sum().clamp_min(1e-6)
        out = torch.cat([vis[:max(a0 - 8, 0)], vis[a0 + w + 8:]]).sum() / torch.cat([occ[:max(a0 - 8, 0)], occ[a0 + w + 8:]]).sum().clamp_min(1e-6)
        assert float(inside) < 0.5 * float(out), (float(inside), float(out))
        found = True; break
    assert found, "no wide constant slab drawn in 40 seeds"


def test_the_missing_wedge_removes_its_fourier_region(isolated, monkeypatch):
    G.MODALITIES["_only_wedge"] = _only("wedge")
    s = G.sample((48, 48, 48), seed=2, device=DEV)
    for seed in range(20):
        d = G.nuisance(s, seed=seed, device=DEV, strength=1.0, modality="_only_wedge", warp=False, fast=True)
        if d["record"]["window"] > 0.003:                                                     # a little clipping blurs the zeros
            continue
        th = d["record"]["wedge"]; ax = d["record"]["axis"]
        img = d["image"].float().movedim(ax, 0); A, B, C = img.shape
        K = torch.fft.fftn(img - img.mean()).abs()
        k0 = torch.fft.fftfreq(A, device=DEV).view(-1, 1, 1).abs(); k1 = torch.fft.fftfreq(B, device=DEV).view(1, -1, 1).abs()
        deep = (k0 > math.tan(math.radians(th + 15)) * k1 + 0.05).expand(A, B, C)            # well inside the wedge
        kept = (k0 < math.tan(math.radians(th - 15)) * k1 - 0.05).expand(A, B, C)
        assert float(K[deep].mean()) < 0.25 * float(K[kept].mean())
        return
    pytest.skip("only clipped windows drawn")


@pytest.mark.parametrize("modality", ["em", "fluorescence", "cryoet"])
def test_new_modalities_end_to_end(modality):
    from shapetensor.learn.scene import draw_scene
    s, d = draw_scene((32, 32, 32), seed=11, device=DEV, strength=1.0, canvas=64, modality=modality)
    assert torch.isfinite(d["image"].float()).all() and d["record"]["modality"] == modality
    s, d = G.draw((32, 32, 32), seed=4, device=DEV, modality=modality)
    assert torch.isfinite(d["image"]).all()
