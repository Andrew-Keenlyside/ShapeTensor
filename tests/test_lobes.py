"""The lobe-mixture output: exact harmonics of Watson lobes, the losses' behaviour, crossings recovered by a direct fit,
the zoom-consistency formula on exact targets, the identity loss, the model's lobe head."""
import math

import pytest
import torch
import torch.nn.functional as Fn

from shapetensor.learn import lobes as LB
from shapetensor.learn.generator import line_tensor, sheet_tensor
from shapetensor.learn.sh import SHBasis, real_sh

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _exact(S6, kappa=20.0, lmax=8):
    b = SHBasis(lmax, 600, DEV); gz, gy, gx = b.grid.unbind(1)
    quad = torch.stack([gz * gz, gy * gy, gx * gx, 2 * gz * gy, 2 * gz * gx, 2 * gy * gx], 1)
    e = torch.exp(kappa * (quad @ S6.to(DEV) - 1)); e = e / e.mean()
    return b.proj @ e


def test_watson_lobes_equal_the_exact_line_and_sheet_densities():
    a = Fn.normalize(torch.tensor([0.3, -0.5, 0.8], device=DEV), dim=0)
    one = torch.ones(1, 1, device=DEV)
    pl = LB.lobes_to_sh(a.view(1, 1, 3), one, torch.full((1, 1), 20.0, device=DEV), 8)[0]
    ps = LB.lobes_to_sh(a.view(1, 1, 3), one, torch.full((1, 1), -10.0, device=DEV), 8)[0]
    for p, t in ((pl, _exact(line_tensor(a))), (ps, _exact(sheet_tensor(a)))):
        assert float((p - t).norm() / t.norm()) < 2e-3
    assert float(pl[0]) == pytest.approx(math.sqrt(4 * math.pi), rel=1e-5)


def test_polynomial_harmonics_match_and_are_smooth_at_the_pole():
    u = Fn.normalize(torch.randn(300, 3, device=DEV), dim=1)
    assert float((LB.sh_poly(u, 8) - real_sh(u, 8)).abs().max()) < 1e-4
    a = torch.tensor([[[1.0, 0.0, 0.0]]], device=DEV, requires_grad=True)
    LB.lobes_to_sh(a, torch.ones(1, 1, device=DEV), torch.full((1, 1), 20.0, device=DEV), 8).sum().backward()
    assert torch.isfinite(a.grad).all()


def test_direct_fit_recovers_both_axes_of_a_crossing():
    """Six free lobes fitted to the exact density of two crossing lines (60 degrees, masses 0.6 / 0.3) from 16 random
    starts at once, with the training loss (per-order squared error + direction cross-entropy): both axes found in
    most starts (the bounded concentration keeps girdles from intersecting into fake peaks; measured ~13/16)."""
    from shapetensor.learn.sh import bands
    torch.manual_seed(0); n, J = 16, 6
    a1 = torch.tensor([0.0, 0.0, 1.0], device=DEV); a2 = torch.tensor([0.0, math.sin(math.radians(60)), math.cos(math.radians(60))], device=DEV)
    tgt = (0.6 * _exact(line_tensor(a1)) + 0.3 * _exact(line_tensor(a2)))[None, :, None].expand(1, -1, n).contiguous()
    raw = torch.randn(1, J * 11, n, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for it in range(800):
        lob = LB.decode(raw, J, 5); f, _ = LB.render(lob, 8)
        loss = sum(((f[:, b] - tgt[:, b]) ** 2).sum() / (tgt[:, b] ** 2).sum() for b in bands(8)) / 5 + 0.5 * LB.direction_ce(f, tgt, 8, m_min=0.01) * n
        opt.zero_grad(); loss.backward(); opt.step()
    d, m = LB.lobe_peaks({k: v.detach() for k, v in lob.items()})
    ang = lambda u, v: math.degrees(math.acos(min(1.0, abs(float(u @ v)))))
    ok = 0
    for c in range(n):
        dd = [d[0, j, :, c] for j in range(J) if float(m[0, j, c]) > 0]
        ok += len(dd) >= 2 and min(ang(dd[0], a1) + ang(dd[1], a2), ang(dd[0], a2) + ang(dd[1], a1)) < 10
    assert ok >= 10, ok


def test_direction_ce_penalises_a_missing_lobe_more_than_squared_error():
    a1 = torch.tensor([0.0, 0.0, 1.0], device=DEV); a2 = torch.tensor([0.0, 1.0, 0.0], device=DEV)
    t = (0.7 * _exact(line_tensor(a1)) + 0.3 * _exact(line_tensor(a2)))[None, :, None]
    miss = (1.0 * _exact(line_tensor(a1)))[None, :, None]                                          # the second lobe absent, mass moved to the first
    broad = (_exact(line_tensor(a1), 6.0) * 0.7 + 0.3 * _exact(line_tensor(a2), 6.0))[None, :, None]   # both lobes, too broad
    se = lambda p: float(((p - t) ** 2).sum() / (t ** 2).sum())
    ce = lambda p: float(LB.direction_ce(p, t, 8, m_min=0.01))
    assert ce(miss) / ce(broad) > se(miss) / se(broad)


def test_zoom_consistency_is_exact_on_exact_targets():
    """The exact density of a scene at voxel v, windowed (sigma 5) and pooled to the stride-4 lattice, then blurred and
    pooled 2x by the consistency rule, equals the same scene rendered at voxel 2v and windowed there."""
    from shapetensor.learn.query import voxel_density
    from shapetensor.learn.scene import draw_pair
    from shapetensor.learn.targets import _pool
    (s1, d1), (s2, d2), info = draw_pair((64, 64, 64), seed=12, device=DEV, kind="zoom", strength=0.0, canvas=128, p_warp=0.0, physics="additive")
    for d in (d1, d2):                                                                              # visibility 1: geometry only
        d["visibility"] = torch.ones_like(d["visibility"]); d["orientation_visibility"] = torch.ones_like(d["orientation_visibility"])
    F1 = _pool(voxel_density(s1, d1, 4), 5.0, 4)[None]; F2 = _pool(voxel_density(s2, d2, 4), 5.0, 4)[None]
    pooled = LB.pool_sh(F1, math.sqrt(10 ** 2 - 5 ** 2) / 4, 2)
    oz, oy, ox = info["offset_cells"]; n = pooled.shape[2:]
    reg = F2[:, :, oz:oz + n[0], oy:oy + n[1], ox:ox + n[2]]
    c = lambda x: x[..., 2:-2, 2:-2, 2:-2]
    a, b = c(reg), c(pooled)[..., :a.shape[2], :a.shape[3], :a.shape[4]] if False else c(pooled)
    if float((b ** 2).sum()) < 1e-6:
        pytest.skip("no populations in the fine crop")
    rel = float(((a - b) ** 2).sum() / (b ** 2).sum())
    assert rel < 0.15, rel


def test_identity_loss_separates_objects():
    ids = torch.zeros(1, 8, 8, 8, dtype=torch.long, device=DEV); ids[:, 4:] = 1; ids[:, :, :2] = -1
    e = (0.01 * torch.randn(1, 4, 8, 8, 8, device=DEV)).requires_grad_(True)
    opt = torch.optim.Adam([e], lr=0.1)
    l0 = float(LB.identity_loss(e, ids))
    for _ in range(200):
        l = LB.identity_loss(e, ids); opt.zero_grad(); l.backward(); opt.step()
    assert float(l) < 0.05 * l0
    m0 = e[0][:, ids[0] == 0].mean(1); m1 = e[0][:, ids[0] == 1].mean(1)
    assert float((m0 - m1).norm()) > 2.0


def test_lobe_model_outputs_and_context():
    from shapetensor.learn.model import PopulationNet, receptive_field
    m = PopulationNet(widths=(8, 16, 32), head="lobes", J=4, context_level=True, voxel_cond=True, identity_dim=4, points=True).to(DEV)
    o = m(torch.rand(1, 1, 64, 64, 64, device=DEV), voxel=torch.tensor([float("nan")], device=DEV))
    assert o["lobes_raw"].shape == (1, 44, 16, 16, 16) and o["identity"].shape == (1, 4, 32, 32, 32) and o["points"].shape == (1, 2, 16, 16, 16)
    assert receptive_field(m) > 80
