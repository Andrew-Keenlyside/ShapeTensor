"""Poor resolution in the training generator (27 Oct): an unresolved bundle's partial volume carries the texture of its
sub-voxel fibres (strands), so its direction -- and both directions where two cross -- shows inside it, not only at its
edges; tortuous bundle paths; a heavy PSF."""
import math

import pytest
import torch

from shapetensor.filters import gaussian
from shapetensor.learn import generator as G
from shapetensor.learn import scene as SC

DEV = "cuda" if torch.cuda.is_available() else "cpu"
A = (0.0, 0.0, 1.0)


def _render(monkeypatch, textured, axes, seed=0, radius=14.0, texture=None):
    monkeypatch.setattr(SC, "P_UNRESOLVED_TEXTURE", 1.0 if textured else 0.0); monkeypatch.setattr(SC, "P_TORTUOUS", 0.0)
    sc = SC.Scene((64, 64, 64), seed, DEV, 20.0); c = torch.tensor([32.0, 32.0, 32.0], device=DEV)
    sc.groups = [sc.bundle(None, "single", axis=torch.tensor(a, device=DEV), centre=c, radius=radius, curv=0.002,
                            fibre={"rf_um": 2.0, **({"texture": texture} if (textured and texture) else {})}) for a in axes]
    out = sc.render()["groups"]
    return torch.stack([o["occupancy"] for o in out]), torch.stack([o["tensor"] for o in out]), sc.record


def _st_axes(img, rho, seed=0):
    x = gaussian(img[None, None], 0.7)[0, 0] + 0.01 * torch.randn(img.shape, device=img.device, generator=torch.Generator(device=img.device).manual_seed(seed))
    Gr = torch.stack(torch.gradient(x), -1)
    S = torch.stack([gaussian((Gr[..., i] * Gr[..., j])[None, None], rho)[0, 0] for i in range(3) for j in range(3)], -1).view(*img.shape, 3, 3)
    return torch.linalg.eigh(S + 1e-9 * torch.eye(3, device=img.device))[1][..., 0]


def _deg(V, a):
    return torch.rad2deg(torch.acos((V @ torch.tensor(a, device=V.device)).abs().clamp(max=1)))


def _core(frac=0.5, R=14.0):
    g = torch.arange(64, device=DEV).float(); Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    return (((Z - 32) ** 2 + (Y - 32) ** 2).sqrt() < frac * R) & (X > 10) & (X < 54)


def test_a_textured_unresolved_bundle_shows_its_direction_inside_and_keeps_its_packing(monkeypatch):
    done = 0
    for seed in range(12):
        occ, T, rec = _render(monkeypatch, True, [A], seed)
        core = _core()
        if float(T[0, 2][core].mean()) < 0.93:                                            # (a fanning bundle: its directions spread)
            continue
        b = rec["families"][0]["bundle"]
        assert b["texture"] > 0 and abs(float(occ[0][core].mean()) - b["packing"]) < 0.12, (b, float(occ[0][core].mean()))
        assert float(_deg(_st_axes(occ[0], 2.0)[core], A).median()) < 20
        occ_s, _, _ = _render(monkeypatch, False, [A], seed)                               # the smooth tube: no direction inside
        assert float(_deg(_st_axes(occ_s[0], 2.0)[core], A).median()) > 35
        done += 1
        if done == 2:
            break
    assert done == 2


def test_two_textured_unresolved_bundles_show_both_directions_where_they_cross(monkeypatch):
    t = math.radians(60.0); B = (0.0, math.sin(t), math.cos(t))
    g = torch.arange(64, device=DEV).float(); Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
    P = torch.stack([Z, Y, X], -1) - 32
    dist = lambda a: (P - (P @ torch.tensor(a, device=DEV))[..., None] * torch.tensor(a, device=DEV)).norm(dim=-1)
    over = (dist(A) < 7) & (dist(B) < 7)
    shares = {}
    for textured in (True, False):                                                            # (both moderately textured: a strongly textured
        sh = []                                                                               # bundle masks a weakly textured one, as it would)
        for seed in range(3):
            occ, _, _ = _render(monkeypatch, textured, [A, B], seed=seed, texture=0.4)
            V = _st_axes(occ.sum(0), 1.5)[over]                                              # the voxel's structure tensor: one direction,
            sh.append(((_deg(V, A) < 20).float().mean(), (_deg(V, B) < 20).float().mean()))   # so a share of the overlap each
        shares[textured] = tuple(float(torch.stack(x).mean()) for x in zip(*sh))
    assert min(shares[True]) > 0.05, shares                                                  # both directions in the overlap's texture (the
    assert max(shares[False]) < 0.03, shares                                                 # denser bundle dominates); the smooth one shows neither


def test_tortuous_paths_turn_more_and_stay_wider_than_their_bundle(monkeypatch):
    turn = {}
    for tort in (0.0, 1.0):
        monkeypatch.setattr(SC, "P_TORTUOUS", tort); vals = []
        for seed in range(6):
            sc = SC.Scene((96, 96, 96), seed, DEV, 2.0)
            g_ = sc.bundle(None, "single", centre=torch.tensor([48.0, 48.0, 48.0], device=DEV), radius=6.0)
            b = sc.record["families"][-1]["bundle"]
            if tort:
                assert "tortuous" in b and b["tortuous"] * 6.0 <= 0.4 + 0.01               # curvature radius >= 2.5 bundle radii
            P = g_.centreline.P[0]; s = torch.cat([torch.zeros(1, device=DEV), (P[1:] - P[:-1]).norm(dim=-1).cumsum(0)])
            i = torch.searchsorted(s, s + 15.0).clamp(max=len(s) - 1); ok = (s + 15.0) <= s[-1]
            Tg = torch.nn.functional.normalize(torch.gradient(P, dim=0)[0], dim=-1)
            vals.append(torch.rad2deg(torch.acos((Tg[ok] * Tg[i[ok]]).sum(-1).clamp(-1, 1))).median())
        turn[tort] = float(torch.stack(vals).median())
    assert turn[1.0] > 1.5 * turn[0.0] and turn[1.0] > 15, turn                             # turns over 15 voxels (DiSCo ~20 deg)


def test_a_heavy_psf_widens_the_blur_and_the_targets_resolution_rule(monkeypatch):
    monkeypatch.setattr(G, "CROSS_MIX", 0.0); monkeypatch.setattr(G, "CROSS_KEY", {})
    prof = {k: 0.0 for k in G.MODALITIES["hipct"]}; G.MODALITIES["_only_heavy"] = dict(prof, heavy_psf=1.0); G.MODALITIES["_only_none"] = prof
    try:
        s = G.sample((40, 40, 40), seed=3, device=DEV)
        heavy = G.nuisance(s, seed=5, device=DEV, strength=1.0, modality="_only_heavy", warp=False, fast=True)["record"]
        plain = G.nuisance(s, seed=5, device=DEV, strength=1.0, modality="_only_none", warp=False, fast=True)["record"]
    finally:
        del G.MODALITIES["_only_heavy"], G.MODALITIES["_only_none"]
    assert 1.5 <= heavy["heavy_psf"] <= 3.5 and "heavy_psf" not in plain
    assert all(h >= min(1.5 * p, 6.0) - 0.01 for h, p in zip(heavy["psf"], plain["psf"]))
    from shapetensor.learn.train import acquisition_blur
    assert acquisition_blur(heavy) > acquisition_blur(plain)
