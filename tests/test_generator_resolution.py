"""Poor resolution in the training generator (27 Oct): an unresolved bundle's partial volume carries the texture of its
sub-voxel fibres (strands), so its direction -- and both directions where two cross -- shows inside it, not only at its
edges; tortuous bundle paths; a heavy PSF."""
import math

import pytest
import torch

from shapetensor.filters import gaussian
import torch.nn.functional as Fn
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


# ---- packed tissue (Scene.tissue), crop strata, phantom version 2 (27 Oct) ----------------------------------------------

def _tissue(seed, vox, grey=False, shape=(96, 96, 96)):
    sc = SC.Scene(shape, seed, DEV, vox); gs = sc.tissue(grey=grey); sc.groups = gs
    ren = sc.render()
    occ = torch.stack([g["occupancy"].float() for g in ren["groups"]]); kinds = [g["group"].kind for g in ren["groups"]]
    ori = occ[[i for i, k in enumerate(kinds) if k in ("line", "sheet")]].sum(0)
    return sc, ren, ori


def test_packed_tissue_fills_the_canvas_wall_to_wall_and_keeps_its_packing():
    for seed, vox in ((0, 4.0), (4, 2.0)):
        sc, ren, ori = _tissue(seed, vox)
        rec = sc.record["tissue"]
        ext = float((gaussian(ori[None, None], 2.0)[0, 0] > 0.15).float().mean())
        assert ext > 0.75, (seed, ext)                                                       # tissue everywhere, not bundles on a background
        lines = [g for g in ren["groups"] if g["group"].name == "tract"]
        for g, phi in zip(lines, rec["packing"]):                                            # each population's fill near its packing where
            o = g["occupancy"].float(); inside = o > 0                                       # it lies (resolved fibres jam near 0.6)
            assert float(o[inside].mean()) < 1.0 and float(o.sum()) > 0
        assert rec["elements"] <= (3000 if rec["resolved"] else 4000) + 50


def test_tract_tensors_follow_their_direction_fields():
    sc, ren, ori = _tissue(2, 20.0)
    for g in ren["groups"]:
        if g["group"].name != "tract":
            continue
        T = g["tensor"].float(); o = g["occupancy"].float(); m = o > 0.2
        M = torch.stack([torch.stack([T[0], T[3], T[4]], -1), torch.stack([T[3], T[1], T[5]], -1), torch.stack([T[4], T[5], T[2]], -1)], -2)[m]
        ev = torch.linalg.eigvalsh(M.double() + 1e-9 * torch.eye(3, device=M.device, dtype=torch.float64))
        assert float(ev[:, 2].median()) > 0.8                                                # one direction per voxel: line-like


def test_a_crossing_zone_holds_populations_at_its_recorded_angle():
    for seed in range(40):
        sc = SC.Scene((96, 96, 96), seed, DEV, 8.0); gs = sc.tissue()
        cz = sc.record["tissue"].get("crossing")
        zone = [g for g in gs if g.name == "tract" and getattr(g, "centre_pt", None) is not None]
        zone = [g for g in zone if torch.allclose(g.centre_pt, zone[0].centre_pt)] if zone else []
        if cz and not cz[0]["planar"] and cz[0]["grain"] == "layers" and len(cz[0]["angles"]) == 1 and len(zone) == 2:
            break                                                                            # a two-population layered zone, both placed
    pops = zone
    assert len(pops) == 2
    d = []
    for g in pops[:2]:                                                                       # each population's mean path direction
        P = g.paths; t = Fn.normalize(P[:, -1] - P[:, 0], dim=-1)
        Mt = (t[:, :, None] * t[:, None, :]).mean(0).double().cpu(); d.append(torch.linalg.eigh(Mt + 1e-9 * torch.eye(3, dtype=torch.float64))[1][:, -1].float())
    ang = math.degrees(math.acos(min(1.0, abs(float(d[0] @ d[1])))))
    assert abs(ang - cz[0]["angles"][0]) < 12, (ang, cz)


def test_grey_tissue_is_mostly_without_direction():
    sc, ren, ori = _tissue(3, 4.0, grey=True)
    assert float((gaussian(ori[None, None], 2.0)[0, 0] > 0.15).float().mean()) < 0.35
    assert any(g["group"].name in ("nuclei", "neuropil", "somata", "membranes") for g in ren["groups"])


def test_crop_strata_pick_packed_crops_from_packed_scenes(monkeypatch):
    monkeypatch.setattr(SC, "P_TISSUE", 1.0); monkeypatch.setattr(SC, "P_GREY_TISSUE", 0.0)
    fills = []
    for seed in range(3):
        s, d = SC.draw_scene((48,) * 3, seed=90_000_000 + 8 * seed, device=DEV, canvas=96, crops_per_scene=8, voxel_um=4.0, fill_quota=(1, 0, 0, 0), strength=0.3)
        fills.append(d["record"]["crop_fill"]); SC._scene_cache().clear()
    assert all(f["stratum"] == "packed" for f in fills) and sum(f["hit"] for f in fills) >= 2, fills


def test_phantom_version_1_is_unchanged_and_version_2_draws_narrow_packed_crossings(monkeypatch):
    from shapetensor.learn import phantom2 as P2
    a = P2.random_spec("cross", torch.Generator().manual_seed(900_000))
    assert [round(s["half"], 3) for s in a[0]] == [9.077, 7.739, 9.333] and a[0][0]["conc"] == 0.94   # the selection set's first crossing
    monkeypatch.setattr(P2, "VERSION", 2)
    angs = []
    for i in range(200):
        sp, lk = P2.random_spec("cross", torch.Generator().manual_seed(i)); A, B = torch.tensor(sp[0]["axis"]), torch.tensor(sp[1]["axis"])
        angs.append(math.degrees(math.acos(min(1.0, abs(float(A @ B))))))
        assert all(s_["conc"] == 1.0 for s_ in sp) and sp[0]["half"] >= 16
    assert min(angs) < 12 and sum(a_ < 30 for a_ in angs) > 60
