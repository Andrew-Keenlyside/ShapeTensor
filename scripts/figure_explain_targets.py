"""What the generator gives the model, built up in three rows from one synthetic scene with two crossing bundles (the
trainer's own targets; construction, no blur; projections along z so whole fibres show):

1. one fibre (the smallest object): its material; its own ODF in each voxel; its ODF over a query window (the target);
2. its population (the bundle): all its fibres; then what one query window holds at three collection scales --
   single fibres, collections of nearby fibres, the whole population -- each an ODF with its share of the window;
3. every population at that query (scale 16): their ODFs and shares; the overall ODF as their share-weighted overlay
   (the crossing shows); and, for contrast, one ellipse for the whole window (a crossing averages to a disc).

    PYTHONPATH=src python scripts/figure_explain_targets.py OUT.png [SEED0]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import shapetensor.learn.train as TR
TR.HEAD, TR.SH_EXACT, TR.DESC_KEY, TR.DESC_LMAX, TR.QUERY = "entities", 8, "descriptors5", 0, None; TR.set_background_entity(True)
import shapetensor.learn.generator as G
G.ORIENT_FROM_TEXTURE = True; G.ORIENT_VIS = False; G.TARGETS_FROM_GEOMETRY = True
from shapetensor.learn import scene as SC, lobe_train as LT
from odf_render import draw_tensors

out = sys.argv[1]; seed0 = int(sys.argv[2]) if len(sys.argv) > 2 else 270_000_000; dev = "cuda"
SC.MATRIX_P = float(os.environ.get("MATRIX", 1.0)); TR.ENTITY["u_max"] = 60_000; SIG = 3.0
FIBRE_NAMES = ("bundle", "tract", "fibre field", "fascicles", "fibres")
gen = LT.difficulty_gen(0.6, dict(canvas=128, crops_per_scene=8, cross_p=1.0, p_centre=0.5, deep_tree=True, fill_quota=SC.FILL_QUOTA))
M3 = lambda t: torch.stack([torch.stack([t[..., 0], t[..., 3], t[..., 4]], -1), torch.stack([t[..., 3], t[..., 1], t[..., 5]], -1),
                            torch.stack([t[..., 4], t[..., 5], t[..., 2]], -1)], -2)
T6 = lambda M: torch.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2], M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]], -1)


def eig(t6):
    ev, E = torch.linalg.eigh(M3(t6.double())); return ev.flip(-1).float(), E[..., :, 2].float()


def line_ness(t6):
    ev, _ = eig(t6); return float((ev[0] - ev[1]) / ev[0].clamp_min(1e-9))


def window(s, d, q, scale):
    """The trainer's targets at one query ``q (3,)`` (window SIG) and collection ``scale``: per entity its node, group,
    share of the window's material, ODF (6); and the background's share."""
    TR.ENTITY["scales"] = float(scale)
    t = TR.entity_queries(s, d, 64, 0, dev, points=q[None].float(), sigma=torch.tensor([SIG], device=dev))
    tree = d["entity_ctx"]["tree"]; gn = t["e_gnode"][0].long(); live = gn >= 0
    m = (t["e_mass"][0].float() * t["e_vis"][0].float()) * live; tot = float(t["e_tot"][0])
    grp = torch.where(live & (gn < tree.n), tree.group[gn.clamp(0, tree.n - 1)], torch.full_like(gn, -1))
    return {"node": gn, "grp": grp, "share": m / max(tot, 1e-9), "T6": t["e_T6"][0].float(), "bg": float(t["e_bgm"][0]) / max(tot, 1e-9),
            "isbg": t["e_bg"][0].bool()}


def find():
    """A scene (no warp, no zoom) and a query where two bundles cross: the two largest non-background entities at scale
    16 from different populations, both line-like, >= 40 deg apart, each >= 8 % of the window."""
    for k in range(80):
        seed = seed0 + 8 * k
        s, d = SC.draw_scene((64,) * 3, seed=seed, device=dev, strength=0.0, entity_ctx=True, **gen)
        ctx = d["entity_ctx"]
        if d.get("warp_grid") is not None or int(ctx["zoom"]) != 1:
            SC._scene_cache().clear(); continue
        gl = list(ctx["group_layer"]); roles = {g_: getattr(s.layers[gl[g_]], "mrole", "fore") for g_ in range(len(gl)) if 0 <= gl[g_] < len(s.layers)}
        fibrous = {g_ for g_ in roles if getattr(s.layers[gl[g_]], "name", "") in FIBRE_NAMES}         # bundles of fibres only
        many = {g_ for g_ in fibrous if int(torch.unique(ctx["entries"][2][ctx["entries"][1].long() == g_]).numel()) >= 10}
        g_ = torch.Generator(device=dev).manual_seed(seed); best = None
        if len(many) < 2:
            SC._scene_cache().clear(); continue
        for q in (10 + torch.rand(200, 3, generator=g_, device=dev) * 44):
            w = window(s, d, q, 16.0); ok = (w["node"] >= 0) & ~w["isbg"]
            idx = torch.nonzero(ok)[:, 0]; idx = idx[w["share"][idx].argsort(descending=True)][:2]
            if idx.numel() < 2 or int(w["grp"][idx[0]]) == int(w["grp"][idx[1]]):
                continue
            if any(int(w["grp"][i]) not in many for i in idx):
                continue
            (_, a0), (_, a1) = eig(w["T6"][idx[0]]), eig(w["T6"][idx[1]])
            ang = float(torch.rad2deg(torch.acos((a0 @ a1).abs().clamp(max=1))))
            sm = float(w["share"][idx].min())
            if ang >= 40 and sm >= 0.08 and min(line_ness(w["T6"][i]) for i in idx) > 0.4 and (best is None or sm > best[0]):
                best = (sm, q, idx, ang)
        if best is not None:
            return seed, s, d, best
        SC._scene_cache().clear(); torch.cuda.empty_cache()
    raise SystemExit("no crossing found")


seed, s, d, (sm, q, idx_cross, ang_cross) = find()
ctx = d["entity_ctx"]; tree = ctx["tree"]; gl = list(ctx["group_layer"])
name = {g_: getattr(s.layers[gl[g_]], "name", "?") for g_ in range(len(gl)) if 0 <= gl[g_] < len(s.layers)}
w16 = window(s, d, q, 16.0); gA, gB = int(w16["grp"][idx_cross[0]]), int(w16["grp"][idx_cross[1]])
print("seed", seed, "query", [round(float(v), 1) for v in q], "populations", name[gA], name[gB], "angle", round(ang_cross, 1))

# the construction records in the crop
vox, grp, obj, wt, v = ctx["entries"]
cs = torch.as_tensor(ctx["canvas_shape"], device=dev); org = torch.as_tensor([int(o) for o in ctx["origin"]], device=dev)
vox = vox.long(); P = torch.stack([vox // (cs[1] * cs[2]), (vox // cs[2]) % cs[1], vox % cs[2]], -1) - org
ins = ((P >= 0) & (P < 64)).all(-1); P, grp, obj, wt, v = P[ins], grp[ins].long(), obj[ins].long(), wt[ins].float(), v[ins, :6].float()
leaf = tree.leaves(grp, obj)


def proj(mask):
    """Material and volume-weighted tensor of the records in ``mask``, summed along z: (64, 64), (64, 64, 6)."""
    li = P[mask, 1] * 64 + P[mask, 2]
    return (torch.zeros(4096, device=dev).index_add_(0, li, wt[mask]).view(64, 64),
            torch.zeros(4096, 6, device=dev).index_add_(0, li, v[mask]).view(64, 64, 6))


# row 1: the fibre of population A with the most material near the query
near = ((P.float() - q).norm(dim=-1) < 3 * SIG) & (grp == gA)
lv, li_ = torch.unique(leaf[near], return_inverse=True)
lm = torch.zeros(len(lv), device=dev).index_add_(0, li_, wt[near])
def thickness(l_):
    """An object's measured cross-section (voxels): twice the root of its material's second-largest spread."""
    m_ = leaf == l_; x_ = P[m_].float(); w_ = wt[m_]; c_ = (x_ * w_[:, None]).sum(0) / w_.sum()
    C = ((x_ - c_).T * w_) @ (x_ - c_) / w_.sum(); ev_ = torch.linalg.eigvalsh(C.double()).flip(0)
    return 2 * float(ev_[1].clamp_min(0).sqrt()), 2 * float(ev_[0].clamp_min(0).sqrt())
cand = lv[lm.argsort(descending=True)][:40].tolist(); fib, best_m = cand[0], -1.0
for l_ in cand:                                                                                    # a thin, long fibre (not a bundle's tube)
    th_, ln_ = thickness(l_)
    if th_ < 3.0 and ln_ > 4 * th_ and float(lm[lv == l_].sum()) > best_m:
        fib, best_m = l_, float(lm[lv == l_].sum())
Wf, Vf = proj(leaf == fib)
pts_f = P[leaf == fib]; wf_ = wt[leaf == fib]
order = torch.argsort(pts_f[:, 2] * 4096 + pts_f[:, 1] * 64 + pts_f[:, 0])
qs_f = pts_f[order][:: max(1, len(order) // 9)][:9].float()                                       # query points along the fibre
fib_w = []
for qf in qs_f:
    w0 = window(s, d, qf, 0.0); hit = torch.nonzero(w0["node"] == fib)[:, 0]
    if hit.numel():
        fib_w.append((qf, w0["T6"][hit[0]], float(w0["share"][hit[0]])))

fig = plt.figure(figsize=(22, 17)); gs = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.12)
A = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]
img = (d["image"].float() / 255).cpu().numpy(); zq = int(q[0])


def mapax(ax, title):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 63.5); ax.set_ylim(63.5, -0.5); ax.set_title(title, fontsize=13)


def glyph_grid(ax, W, V, step, thr):
    yy, xx = np.meshgrid(np.arange(step // 2, 64, step), np.arange(step // 2, 64, step), indexing="ij")
    Wn, Vn = W.cpu().numpy(), V.cpu().numpy(); k = Wn[yy, xx] > thr
    if k.any():
        draw_tensors(ax, np.stack([xx[k], yy[k]], -1), Vn[yy, xx][k] / Wn[yy, xx][k][:, None], 0.95 * step, view="z")


def glyph_row(ax, items, title, size=1.7):
    """ODF glyphs side by side, each labelled with its share (``items``: (T6, share, label))."""
    for i, (t6, sh, lab) in enumerate(items):
        draw_tensors(ax, np.array([[i * 3.2, 0.0]]), t6[None].cpu().numpy(), size, view="z")
        ax.text(i * 3.2, -1.45, f"{lab}\nshare {sh:.2f}", ha="center", va="top", fontsize=10)
    ax.set_xlim(-1.7, max(len(items), 4) * 3.2 - 1.5); ax.set_ylim(-2.6, 1.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=13)


# ---- row 1: one fibre
ax = A[0][0]; ax.imshow(Wf.clamp(0, 1.5).cpu().numpy(), cmap="gray"); mapax(ax, f"1. one fibre (an object of {name[gA]}):\nits material (projected along z)")
ax = A[0][1]; ax.imshow(Wf.clamp(0, 1.5).cpu().numpy(), cmap="gray", alpha=0.35); glyph_grid(ax, Wf, Vf, 2, 0.05)
mapax(ax, "its own ODF in each voxel\n(line-ness ~1: it points one way there)")
ax = A[0][2]; ax.imshow(Wf.clamp(0, 1.5).cpu().numpy(), cmap="gray", alpha=0.35)
if fib_w:
    draw_tensors(ax, np.array([[float(p_[2]), float(p_[1])] for p_, _, _ in fib_w]), torch.stack([t_ for _, t_, _ in fib_w]).cpu().numpy(), 5.5, view="z")
mapax(ax, f"its ODF as a training target: over a window\n(sigma {SIG:g}) at points along it -- its waviness shows")
ax = A[0][3]
if fib_w:
    j = len(fib_w) // 2; ev, _ = eig(fib_w[j][1])
    draw_tensors(ax, np.array([[0.0, 0.0]]), fib_w[j][1][None].cpu().numpy(), 2.0, view="z")
    ax.text(0, -1.5, f"one of them, enlarged\neigenvalues {ev[0]:.2f} / {ev[1]:.2f} / {ev[2]:.2f}\nline-ness {line_ness(fib_w[j][1]):.2f}", ha="center", va="top", fontsize=12)
ax.set_xlim(-2, 2); ax.set_ylim(-2.8, 1.4); ax.set_aspect("equal"); ax.axis("off"); ax.set_title("a fibre's ODF: an ellipsoid (its second moment)", fontsize=13)

# ---- row 2: its population
pm = grp == gA; ids = torch.unique(leaf[pm]); rgb = torch.zeros(4096, 3, device=dev)
col = torch.tensor(np.random.default_rng(1).uniform(0.25, 1.0, (len(ids), 3)), device=dev, dtype=torch.float32)
li = P[pm, 1] * 64 + P[pm, 2]; ci = torch.searchsorted(ids, leaf[pm])
rgb.index_add_(0, li, wt[pm, None] * col[ci]); rgb = (rgb / rgb.max(0).values.clamp_min(1e-6)).view(64, 64, 3).cpu().numpy()
ax = A[1][0]; ax.imshow(np.clip(rgb * 1.6, 0, 1)); ax.plot(float(q[2]), float(q[1]), "s", mfc="none", mec="white", ms=16, mew=2)
mapax(ax, f"2. its population ({name[gA]}): {len(ids)} fibres\n(one colour each); square = the query window")
for c_, (sc_, lab) in enumerate(((0.0, "scale 0: single fibres"), (4.0, "scale 4: nearby fibres grouped"), (16.0, "scale 16: the whole population"))):
    wq = window(s, d, q, sc_); k = torch.nonzero((wq["grp"] == gA) & (wq["node"] >= 0))[:, 0]
    k = k[wq["share"][k].argsort(descending=True)][:4]
    items = [(wq["T6"][i], float(wq["share"][i]), f"#{n_ + 1}") for n_, i in enumerate(k.tolist())]
    tot_a = float(wq["share"][(wq["grp"] == gA) & (wq["node"] >= 0)].sum())
    glyph_row(A[1][1 + c_], items, f"{lab}\n(largest {len(items)}; this population {tot_a:.2f} of the window)")

# ---- row 3: every population at the query
ax = A[2][0]; ax.imshow(img[zq], cmap="gray"); ax.plot(float(q[2]), float(q[1]), "s", mfc="none", mec="cyan", ms=16, mew=2)
mapax(ax, f"3. the image (slice z = {zq}) and the query;\n{name[gA]} and {name[gB]} cross here ({ang_cross:.0f} deg)")
w = w16; k = torch.nonzero((w["node"] >= 0) & ~w["isbg"])[:, 0]; k = k[w["share"][k].argsort(descending=True)][:4]
items = [(w["T6"][i], float(w["share"][i]), name.get(int(w["grp"][i]), "?")) for i in k.tolist()]
other = max(1.0 - float(w["share"][k].sum()) - w["bg"], 0.0)
glyph_row(A[2][1], items + [(torch.tensor([1, 1, 1, 0, 0, 0.0], device=dev) / 3, w["bg"], "background")],
          f"the populations at the query (scale 16): what the\nhead is trained to give (+ other {other:.2f})")
ax = A[2][2]                                                                                 # the overall ODF: the share-weighted overlay
for i in k.tolist():
    draw_tensors(ax, np.array([[0.0, 0.0]]), w["T6"][i][None].cpu().numpy(), 2.4, view="z", scale=np.array([np.sqrt(float(w["share"][i]) / float(w["share"][k].max()))]))
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("the overall ODF there: every population's\nODF overlaid, sized by its share -- the crossing", fontsize=13)
ax = A[2][3]; Tm = (w["T6"][k] * w["share"][k, None]).sum(0) / w["share"][k].sum()
draw_tensors(ax, np.array([[0.0, 0.0]]), Tm[None].cpu().numpy(), 2.4, view="z")
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"for contrast: one ellipse for the whole window\n(line-ness {line_ness(Tm):.2f}: the crossing averages to a disc)", fontsize=13)
fig.savefig(out, dpi=72, bbox_inches="tight"); print("saved", out)
