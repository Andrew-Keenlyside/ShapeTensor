"""Quick test of voxel-first ownership (notes/framing_review_2026-10-27.md, §7), no training: a trained model is read out
at the voxel scale instead of per direction at the query scale, and its voxel lobes are grouped into entities.

1. At every voxel near the scoring slice, the model's density at sigma 1 over a direction grid: up to two lobes (the
   strongest direction; a second at least 25 deg away holding at least half the first's height above the mean), each
   with its share of the voxel's material (the mass output). Voxels where no direction stands out are isotropic.
2. Voxel lobes grouped into entities by continuity: neighbouring voxels' (26-connected) lobes join when their directions
   agree within ``--join`` degrees -- the two lobes of a shared voxel at a crossing go to different entities. A classical
   stand-in for a learned per-voxel instance embedding.
3. Each oriented entity's mass and orientation tensor windowed at the scoring points exactly as the targets are
   (:func:`~shapetensor.learn.phantom2.targets`: window means of occupancy and occupancy-weighted tensor, Bingham fit),
   then scored with the acceptance criteria next to the model's current readout. Isotropic objects and the background
   are the model's own (the current readout's), so only the oriented ownership differs.

Usage: python scripts/learn/voxel_first_test.py OUT_STEM CKPT [--final] [--join 20]
"""
import argparse, json, math, sys, time
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, "src")
import shapetensor.learn.train as TR
from shapetensor.learn import entity_train as ET, losses as L, entity_infer as EI, phantom2 as P2, crossing_acceptance as CA, bingham as BG
from shapetensor.learn.model import PopulationNet, window_mean

ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("ckpt"); ap.add_argument("--final", action="store_true")
ap.add_argument("--join", type=float, default=20.0); ap.add_argument("--slab", type=float, default=10.0); ap.add_argument("--max-ent", type=int, default=10)
ap.add_argument("--families", nargs="+", default=None)
ap.add_argument("--merge", type=float, default=15.0); ap.add_argument("--speck", type=float, default=0.03)
a = ap.parse_args(); dev = "cuda"
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None
L.DESC_K = 5; L.DESC_NC = 0; TR.REGION_DENSITY = True; ET.REGION_DENSITY = True; TR.set_background_entity(True)
FAMS = ("cross", "fan", "curve", "sheet", "blob", "mix", "empty", "fine")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
net = PopulationNet.from_args(ck["args"]).to(dev).eval(); net.load_state_dict(ck["model"])
SPH = EI.sphere_dirs(300, dev); sph200 = EI.sphere_dirs(200, dev)


def build_tests(S0, k):                                                     # as phantom_ladder.build_tests
    out = []
    for fi, fam in enumerate(FAMS):
        for i in range(k):
            spec, look = P2.random_spec(fam, torch.Generator().manual_seed(S0 + 100 * fi + i))
            out.append((fam, P2.make(spec, seed=S0 + 50_000 + 100 * fi + i, look=look, device=dev, ss=2 if fam == "fine" else 1)))
    return out


def slice_frame(ph):                                                        # as phantom_ladder.slice_frame
    spec = ph["spec"]
    if spec and "normal" in spec[0]:
        nrm = torch.nn.functional.normalize(torch.tensor(spec[0]["normal"], dtype=torch.float32), dim=0); e1, e2 = P2._frame(nrm)
        if spec[0]["kind"] == "sheet":
            R = torch.stack([e1, nrm, e2], 1); off = 0.0
        else:
            R = torch.stack([nrm, e1, e2], 1); off = spec[0].get("spacing", 0.0) / 4
    else:
        R = torch.eye(3); off = 0.0
    return R.to(dev), off


def t6(t):
    return torch.stack([t[:, 0] ** 2, t[:, 1] ** 2, t[:, 2] ** 2, t[:, 0] * t[:, 1], t[:, 0] * t[:, 2], t[:, 1] * t[:, 2]], 0)


def voxel_readout(ph, pts, R, off, sig_out=2.5):
    """The voxel-first readout's model dict at the scoring points ``pts`` (and its number of entities)."""
    n = ph["n"]; c = n / 2.0
    zz = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32)] * 3, indexing="ij"), -1).view(-1, 3)
    dist = ((zz - c) @ R[:, 0] - off).abs()
    keep = (dist <= a.slab) & ((zz >= 1) & (zz <= n - 2)).all(1)
    V = zz[keep]; vi = torch.nonzero(keep)[:, 0]
    with torch.no_grad():
        f = EI.features(net, ph["image"]); r = EI.query(net, f, V, torch.ones(V.shape[0], device=dev), SPH, vol_shape=ph["image"].shape, chunk=2048)
    F = r["f"].float(); w = r["m"].float().clamp(0, 1) if "m" in r else F.mean(1)
    mu = F.mean(1); i1 = F.argmax(1); h1 = F.gather(1, i1[:, None])[:, 0]; t1 = SPH[i1]
    far = (SPH @ t1.T).abs().T < math.cos(math.radians(25.0))
    F2 = torch.where(far, F, torch.zeros_like(F)); i2 = F2.argmax(1); h2 = F2.gather(1, i2[:, None])[:, 0]; t2 = SPH[i2]
    oriented = h1 >= 1.5 * mu
    two = oriented & (h2 - mu >= 0.5 * (h1 - mu)) & (h2 >= 1.5 * mu)
    s1 = torch.where(two, (h1 - mu) / ((h1 - mu) + (h2 - mu)).clamp_min(1e-9), torch.ones_like(h1))
    # nodes: (voxel, lobe) -- lobe 0 for oriented voxels, lobe 1 where a second lobe exists
    nid0 = torch.full((V.shape[0],), -1, dtype=torch.long, device=dev); nid1 = nid0.clone()
    o_idx = torch.nonzero(oriented)[:, 0]; nid0[o_idx] = torch.arange(o_idx.numel(), device=dev)
    t_idx = torch.nonzero(two)[:, 0]; nid1[t_idx] = o_idx.numel() + torch.arange(t_idx.numel(), device=dev)
    nn_ = o_idx.numel() + t_idx.numel()
    ndir = torch.cat([t1[o_idx], t2[t_idx]]); nvox = torch.cat([o_idx, t_idx]); nw = torch.cat([(w * s1)[o_idx], (w * (1 - s1))[t_idx]])
    # neighbours: 26-connected voxels within the kept set
    lut = torch.full((n ** 3,), -1, dtype=torch.long, device=dev); lut[vi] = torch.arange(vi.numel(), device=dev)
    Vi = V.long(); rows, cols = [], []
    cj = math.cos(math.radians(a.join))
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                q = Vi + torch.tensor([dz, dy, dx], device=dev)
                ok = ((q >= 0) & (q < n)).all(1)
                j = torch.full((V.shape[0],), -1, dtype=torch.long, device=dev)
                j[ok] = lut[(q[ok, 0] * n + q[ok, 1]) * n + q[ok, 2]]
                for la, na in ((0, nid0), (1, nid1)):
                    for lb, nb in ((0, nid0), (1, nid1)):
                        i_ok = (na >= 0) & (j >= 0)
                        ii = torch.nonzero(i_ok)[:, 0]; jj = j[ii]
                        a_n = na[ii]; b_n = nb[jj]; m = b_n >= 0
                        a_n, b_n = a_n[m], b_n[m]
                        agree = (ndir[a_n] * ndir[b_n]).sum(1).abs() >= cj
                        rows.append(a_n[agree]); cols.append(b_n[agree])
    rr = torch.cat(rows).cpu().numpy(); cc = torch.cat(cols).cpu().numpy()
    G = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nn_, nn_))
    ncomp, lab = connected_components(G, directed=False)
    lab = torch.as_tensor(lab, device=dev)
    # pieces of one entity broken apart by noise: components whose principal axes agree within ``--merge`` deg join
    # (union-find over the components, heaviest first); specks below ``--speck`` of the oriented material are dropped
    Tc = torch.zeros(ncomp, 6, device=dev).index_add_(0, lab, nw[:, None] * t6(ndir).T)
    mass_c = torch.zeros(ncomp, device=dev).index_add_(0, lab, nw)
    T3c = torch.zeros(ncomp, 3, 3, device=dev)
    for (i_, j_), c_ in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
        T3c[:, i_, j_] = Tc[:, c_]; T3c[:, j_, i_] = Tc[:, c_]
    ax_c = torch.linalg.eigh(T3c)[1][..., -1]
    big = torch.argsort(mass_c, descending=True)
    big = big[mass_c[big] >= a.speck * float(mass_c.sum())]
    parent = {int(i): int(i) for i in big.tolist()}; cm = math.cos(math.radians(a.merge))
    heads = []
    for i in big.tolist():
        for h in heads:
            if abs(float(ax_c[i] @ ax_c[h])) >= cm:
                parent[i] = h; break
        else:
            heads.append(i)
    remap = torch.full((ncomp,), -1, dtype=torch.long, device=dev)
    for i, h in parent.items():
        remap[i] = h
    lab = remap[lab]; keep_n = lab >= 0
    ndir, nvox, nw, lab = ndir[keep_n], nvox[keep_n], nw[keep_n], lab[keep_n]
    mass_c = torch.zeros(ncomp, device=dev).index_add_(0, lab, nw)
    order = torch.argsort(mass_c, descending=True)[:a.max_ent]
    order = order[mass_c[order] > 0]
    occs, tens = [], []
    for ci in order.tolist():
        sel = lab == ci
        occ = torch.zeros(n ** 3, device=dev); occ.index_add_(0, vi[nvox[sel]], nw[sel])
        T = torch.zeros(6, n ** 3, device=dev); T.index_add_(1, vi[nvox[sel]], nw[sel][None] * t6(ndir[sel]))
        occs.append(occ.view(n, n, n)); tens.append(T.view(6, n, n, n))
    iso = torch.zeros(n ** 3, device=dev); iso.index_add_(0, vi[~oriented], w[~oriented]); iso = iso.view(n, n, n)
    Q = pts.shape[0]; E = len(occs); sig = torch.full((Q,), sig_out, device=dev)
    mass = np.zeros((Q, max(E, 1))); k = np.zeros((Q, max(E, 1), 2)); Rm = np.tile(np.eye(3), (Q, max(E, 1), 1, 1))
    if E:
        O = torch.stack(occs); m_ = window_mean(O, pts[None].expand(E, -1, -1).contiguous(), sig[None].expand(E, -1).contiguous()).T   # (Q, E)
        Tw = torch.stack(tens).view(E * 6, n, n, n)
        mt = window_mean(Tw, pts[None].expand(E * 6, -1, -1).contiguous(), sig[None].expand(E * 6, -1).contiguous()).T.view(Q, E, 6)
        mt = mt / m_[..., None].clamp_min(1e-6)
        T3 = torch.zeros(Q, E, 3, 3, device=dev)
        for (i_, j_), c_ in zip(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)), range(6)):
            T3[..., i_, j_] = mt[..., c_]; T3[..., j_, i_] = mt[..., c_]
        tr = T3.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-9)[..., None, None]; T3 = T3 / tr
        T3 = 0.94 * T3 + 0.06 * (torch.eye(3, device=dev) - T3) / 2                           # the targets' dispersion floor
        live = m_ > 1e-4
        Rf = torch.eye(3, device=dev).expand(Q, E, 3, 3).clone(); Kf = torch.zeros(Q, E, 2, device=dev)
        if bool(live.any()):
            r_, k_ = BG.fit(T3[live].double()); Rf[live], Kf[live] = r_.float(), k_.float()
        mass = (m_ * live).cpu().numpy(); k = Kf.cpu().numpy(); Rm = Rf.cpu().numpy()
    iso_q = window_mean(iso[None], pts[None], sig[None])[0].cpu().numpy()
    return {"mass": mass, "k": k, "R": Rm, "iso": iso_q}, E, int(two.sum()), int(oriented.sum())


def oracle(tgt):
    iso = CA._types(tgt["k"][..., 0], tgt["k"][..., 1]) == 2
    return {"mass": np.where(iso, 0.0, tgt["mass"]), "k": tgt["k"], "R": tgt["R"], "iso": (tgt["mass"] * iso).sum(1), "bg": tgt["bg"], "bgm": tgt["bg"]}


tests = build_tests(1_900_000, 4) if a.final else build_tests(900_000, 2)
if a.families:
    tests = [t for t in tests if t[0] in a.families]
res = {"current": [], "voxel-first": []}; t0 = time.time()
for fam, ph in tests:
    n, c = ph["n"], ph["n"] / 2.0; R, off = slice_frame(ph)
    gu, gv = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
    loc = torch.stack([torch.full_like(gu, off), gu - c, gv - c], -1).view(-1, 3); pts = c + loc @ R.T
    rotated = not torch.allclose(R, torch.eye(3, device=dev))
    disc = ((((gu - c) ** 2 + (gv - c) ** 2) <= 28 ** 2) if rotated else torch.ones_like(gu, dtype=torch.bool)).cpu().numpy()
    sig = torch.full((n * n,), 2.5, device=dev)
    t = P2.targets(ph, pts, sig); K = ph["occ"].shape[0]
    tgt = {"mass": t["e_mass"][:, :K].cpu().numpy(), "k": t["e_kappa"][:, :K].cpu().numpy(), "R": t["e_R"][:, :K].cpu().numpy(),
           "bg": t["e_mass"][:, K].cpu().numpy(), "kinds": ph["kinds"]}
    face = torch.minimum(pts, (n - 1) - pts).amin(1).view(n, n).cpu().numpy()
    if not CA.evaluate_p2(n, tgt, oracle(tgt), face, disc)["PASS"]:
        print(fam, "unscoreable (the ground truth fails)", flush=True); continue
    ro = EI.readout(net, ph["image"], pts, sig, sph200, floor=0.05, iso_k=ET.VOTE_KAPPA_MIN, merge_deg=15.0)
    cur = {k_: v.cpu().numpy() for k_, v in ro.items()}
    vox, E, n2, no = voxel_readout(ph, pts, R, off)
    vox["bg"], vox["bgm"] = cur["bg"], cur.get("bgm", cur["bg"])                                 # the background: the model's own head, both
    vox["iso"] = cur["iso"]                                                                       # isotropic objects: the model's own estimate
                                                                                                  # (only the oriented ownership is under test)
    lab = f"{fam} ({','.join(k_[:5] for k_ in ph['kinds']) or 'none'})"
    for name, mdl in (("current", cur), ("voxel-first", vox)):
        r_ = CA.evaluate_p2(n, tgt, mdl, face, disc); res[name].append((lab, r_))
    line = lambda r_: " ".join(k_.split()[0] + ("+" if v[0] else "-") for k_, v in r_.items() if k_ != "PASS")
    print(f"{lab:28s} current: {line(res['current'][-1][1])} | voxel-first ({E} entities, {n2}/{no} voxels with two lobes): {line(res['voxel-first'][-1][1])}  [{time.time() - t0:.0f}s]", flush=True)

for name, acc in res.items():
    crit = [k_ for k_ in acc[0][1] if k_ != "PASS"]
    print(f"{name}: {sum(r_['PASS'] for _, r_ in acc)}/{len(acc)} pass | " + " ".join(f"{k_.split()[0]} {sum(r_[k_][0] for _, r_ in acc)}" for k_ in crit), flush=True)
    fams = {}
    for lab_, r_ in acc:
        fams.setdefault(lab_.split(" ")[0], []).append(r_)
    for fam, rs in fams.items():
        print(f"    {fam:6s}", " ".join(f"{k_.split()[0]} {sum(r_[k_][0] for r_ in rs)}/{len(rs)}" for k_ in crit), flush=True)
    CA.save_json(acc, f"{a.out}_{name.replace('-', '_')}_acceptance.json")
