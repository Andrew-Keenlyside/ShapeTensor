"""The readout's no-orientation output and the oriented-share presence test (CPU, synthetic features)."""
import torch

from shapetensor.separate import OrientationReadout, hemisphere


def _features(n_vox=(8, 8, 8), oriented=True, seed=0):
    """Features that encode a direction (its dyad's six entries, plus noise) or carry none (noise of the same size)."""
    g = torch.Generator().manual_seed(seed)
    d = torch.nn.functional.normalize(torch.randn(3, *n_vox, generator=g), dim=0)
    if oriented:
        f = torch.stack([d[0] * d[0], d[1] * d[1], d[2] * d[2], d[0] * d[1], d[0] * d[2], d[1] * d[2]]) * 3
        f = torch.cat([f, torch.zeros(2, *n_vox)]) + 0.05 * torch.randn(8, *n_vox, generator=g)
    else:
        f = 1.2 * torch.randn(8, *n_vox, generator=g)
    return f, d


def test_iso_output_separates_texture_and_leaves_directions_out():
    blocks = []
    for s in range(4):
        f, d = _features(seed=s)
        blocks.append((f, d, torch.ones(f.shape[1:])))
    iso = [_features(oriented=False, seed=10 + s)[0] for s in range(2)]
    r = OrientationReadout(smooth=1, n_dirs=60, steps=600, hidden=32).fit(blocks, n_sample=6000, prepped=True, iso=iso)
    assert r.iso and r.net[-1].out_features == 61
    f_or, _ = _features(seed=20); f_tex, _ = _features(oriented=False, seed=21)
    m_or, m_tex = r.iso_mass(f_or, prepped=True), r.iso_mass(f_tex, prepped=True)
    assert float(m_tex.mean()) > 0.7 and float(m_or.mean()) < 0.3, (float(m_tex.mean()), float(m_or.mean()))
    P = r.predict_dist(f_tex, prepped=True)
    assert P.shape[0] == 60
    assert torch.allclose(P.sum(0) + m_tex, torch.ones_like(m_tex), atol=1e-4)               # directions + no orientation = 1


def test_without_iso_the_readout_is_unchanged():
    f, d = _features(seed=1)
    r = OrientationReadout(smooth=1, n_dirs=60, steps=50, hidden=16).fit([(f, d, torch.ones(f.shape[1:]))], n_sample=2000, prepped=True)
    assert not r.iso and r.net[-1].out_features == 60
    assert torch.allclose(r.predict_dist(f, prepped=True).sum(0), torch.ones(f.shape[1:]), atol=1e-4)
    assert float(r.iso_mass(f, prepped=True).abs().max()) == 0.0


def test_presence_probe_keeps_mixtures_as_fibres():
    from shapetensor.separate import PresenceProbe
    blocks = []
    for s in range(4):
        f, d = _features(seed=s)
        blocks.append((f, d, torch.ones(f.shape[1:])))
    iso = [_features(oriented=False, seed=10 + s)[0] for s in range(2)]
    pr = PresenceProbe(smooth=1).fit(blocks, iso, n_sample=6000, prepped=True)
    assert pr.fit_report["accuracy"] > 0.9
    f1, _ = _features(seed=30); f2, _ = _features(seed=31); ftex, _ = _features(oriented=False, seed=32)
    p_mix = pr.predict(0.5 * (f1 + f2), prepped=True); p_tex = pr.predict(ftex, prepped=True)
    assert float(p_mix.mean()) > 0.8 and float(p_tex.mean()) < 0.3, (float(p_mix.mean()), float(p_tex.mean()))


def test_texture_model_puts_fibres_and_mixtures_far_from_texture():
    """Fibre features carry what texture lacks (here two channels at 3, as dev-vit's features separate fibres from
    texture); a mix of two fibre patterns keeps it, so it stays far from texture though unlike either fibre."""
    from shapetensor.separate import TextureModel
    g = torch.Generator().manual_seed(60)

    def fibres(seed):
        f, d = _features(seed=seed); f = f.clone(); f[6:] += 3.0; return f

    iso = [torch.randn(8, 8, 8, 8, generator=g) for _ in range(2)]
    tm = TextureModel(smooth=1).fit(iso, n_sample=4000, prepped=True)
    d_tex = tm.predict(torch.randn(8, 8, 8, 8, generator=g), prepped=True)
    f1, f2 = fibres(61), fibres(62)
    d_fib, d_mix = tm.predict(f1, prepped=True), tm.predict(0.5 * (f1 + f2), prepped=True)
    assert float(d_fib.median()) > float(d_tex.quantile(0.99)) and float(d_mix.median()) > float(d_tex.quantile(0.99))
