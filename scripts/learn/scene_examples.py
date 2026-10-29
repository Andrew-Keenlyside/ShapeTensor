"""Examples of the object-scene generator (`learn/scene.py`): one figure per topic.

    python scripts/learn/scene_examples.py [--out DIR]

1. ``scene_families.png``: each family alone on a 96³ canvas, middle slice: the composited clean image, the same after
   an acquisition, the label axis (DEC x occupancy), object identities, a descriptor in µm (size) and the lumen fraction.
2. ``scene_collapse.png``: one vessel collapsing and narrowing along its length: cross-sections, a longitudinal slice,
   the descriptor profiles along it and its single identity.
3. ``scene_modalities.png``: one crop of one scene under the HiP-CT, CT, MRI and histology acquisition models.
4. ``scene_pairs.png``: training pairs (64³ crops) with their targets: the heaviest population's axis x mass, the
   population count and the identities.
5. ``scene_crossings.png``: two bundles crossing, fibres passing through each other vs interleaved fascicles (a plane
   containing the crossing's normal and one bundle's axis), a tract splitting in two, a kissing pair.
6. ``scene_physics.png``: one crop under each image formation (additive, X-ray, dark-field, histology, PLI, MRI), and a
   zoom pair (the crop and the region twice its size at half the resolution).
7. ``scene_new_families.png``: the added families -- plywood lamellae, fascicle hierarchies, trabeculae, tubules,
   epithelia, glands, dendrites, crystals (populations coloured).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shapetensor.learn.generator import eig3
from shapetensor.learn.scene import Canvas, Rng, Scene, curves_like, draw_scene, render_curves
from shapetensor.learn.targets import target
from shapetensor.viz import dec_rgb

DEV = "cuda"


def id_rgb(ids: np.ndarray) -> np.ndarray:
    rs = np.random.default_rng(3); pal = rs.uniform(0.25, 1.0, size=(65536, 3)); pal[0] = 0
    out = pal[np.clip(ids, 0, 65535)]; out[ids < 0] = 0
    return out


def axis_dec(T: torch.Tensor, occ: torch.Tensor) -> np.ndarray:
    _, V = eig3(T); v = V[2].cpu().numpy()
    return (dec_rgb(v) * np.clip(occ.cpu().numpy(), 0, 1)[..., None]).clip(0, 1)


def families(out: Path):
    names = [("bundle", 3.0), ("tree", 10.0), ("network", 1.5), ("helix", 2.0), ("sheet", 4.0), ("foam", 20.0), ("porous", 20.0), ("cells", 2.0), ("crystals", 8.0), ("radial", 3.0)]
    fig, axs = plt.subplots(len(names), 6, figsize=(21, 3.6 * len(names)), facecolor="white")
    for r_, (nm, vum) in enumerate(names):
        fam = {k: (1.0 if k == nm else 0.0) for k in Scene.FAMILIES}
        sc = None
        for seed in range(20):                                              # a seed where the family renders at this voxel size
            sc = Scene((96, 96, 96), 100 + seed, DEV, vum).build(fam, (1, 1), 1.0 if nm == "radial" else 0.0)
            if any(g.name in (nm, "membranes", "radial", "tangential") or g.name == nm for g in sc.groups):
                break
        ren = sc.render()
        tot = sum(g["occupancy"] for g in ren["groups"]); k = int(tot.sum((1, 2)).argmax())   # the busiest slice
        occ = torch.stack([g["occupancy"] for g in ren["groups"]]); T = torch.stack([g["tensor"] for g in ren["groups"]])
        hv = occ.argmax(0); Tm = torch.gather(T, 0, hv[None, None].expand(1, 6, *hv.shape))[0]
        omax = occ.max(0).values
        img = 100 + sum(g["occupancy"] * g["contrast"] for g in ren["groups"]) + ren["lumen"] * ren["lumen_c"]
        s, d = draw_scene((96, 96, 96), seed=8 * 50, device=DEV, strength=0.4, canvas=96, voxel_um=vum, families=fam, n_groups=(1, 1),
                          layout_p=1.0 if nm == "radial" else 0.0, modality="hipct", style="flat")
        D = torch.stack([g["desc"] for g in ren["groups"]])
        size_um = (torch.gather(D[:, 0], 0, hv[None])[0] * vum).cpu().numpy() * (omax.cpu().numpy() > 0.3)
        lum = torch.gather(D[:, 2], 0, hv[None])[0].cpu().numpy() * (omax.cpu().numpy() > 0.3)
        ax = axs[r_]
        ax[0].imshow(img[k].cpu().numpy(), cmap="gray"); ax[0].set_title(f"{nm} ({vum:g} µm voxels): clean, slice {k}", fontsize=9)
        ax[1].imshow(d["image"][k].cpu().numpy(), cmap="gray", vmin=0, vmax=255); ax[1].set_title("HiP-CT acquisition (another crop)", fontsize=9)
        ax[2].imshow(axis_dec(Tm[:, k], omax[k])); ax[2].set_title("label axis (DEC) x occupancy", fontsize=9)
        ax[3].imshow(id_rgb(ren["ids"][0, k].cpu().numpy())); ax[3].set_title(f"object identities ({int(torch.unique(ren['ids'][0]).numel()) - 1} objects)", fontsize=9)
        h = ax[4].imshow(size_um[k], cmap="viridis"); plt.colorbar(h, ax=ax[4], fraction=0.046); ax[4].set_title("size (µm): radius / half-thickness", fontsize=9)
        h = ax[5].imshow(lum[k], cmap="magma", vmin=0, vmax=1); plt.colorbar(h, ax=ax[5], fraction=0.046); ax[5].set_title("lumen / hollow fraction", fontsize=9)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        print(nm, sc.record["families"], flush=True)
    fig.suptitle("Object-scene generator: each family alone (middle slice of a 96³ canvas); every label from the rendered objects", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out / "scene_families.png", dpi=50); plt.close(fig)


def collapse(out: Path):
    rng = Rng(0, DEV); S = 260
    s = torch.arange(S, device=DEV, dtype=torch.float32)
    P = torch.stack([torch.full((S,), 40.0, device=DEV) + 6 * torch.sin(s / 40), torch.full((S,), 40.0, device=DEV), s * 0.5 - 2], -1)[None]
    rad = 9 - 3.5 * torch.exp(-0.5 * ((s - 70) / 12) ** 2)                 # a stenosis
    flat = 1 - 0.96 * torch.exp(-0.5 * ((s - 170) / 14) ** 2)              # a collapse: the lumen closes, the wall continues
    c = curves_like(P, rad[None], rng, obj=torch.tensor([5], device=DEV), seg=torch.tensor([0], device=DEV), contrast=90.0, flat=flat[None],
                    wall=torch.full((1, S), 1.8, device=DEV), lumen_c=torch.full((1, S), -40.0, device=DEV))
    cv = Canvas((80, 80, 128), DEV); render_curves(cv, c); g = cv.finish(); ids, _ = cv.identities()
    lum = cv.lumen.view(80, 80, 128).clamp(0, 1) * (cv.lumen_c.view(80, 80, 128) / cv.lumen.view(80, 80, 128).clamp_min(1e-9))
    img = (100 + g["occupancy"] * 90 + lum).cpu().numpy(); occ = g["occupancy"].cpu().numpy(); D = g["desc"].cpu().numpy()
    fig = plt.figure(figsize=(20, 9), facecolor="white"); gs = fig.add_gridspec(2, 6)
    for j, x in enumerate((15, 35, 60, 85, 100)):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(img[:, :, x], cmap="gray"); ax.set_title(f"cross-section at x = {x}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 5]); ax.imshow(id_rgb(ids[0, :, 40].cpu().numpy())); ax.set_title(f"identity (longitudinal): {torch.unique(ids[0][ids[0] >= 0]).tolist()}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :3]); ax.imshow(img[:, 40, :], cmap="gray", aspect="auto"); ax.set_title("longitudinal slice: a stenosis (x ~ 35) and a collapse (x ~ 85): the wall continues", fontsize=9)
    ax = fig.add_subplot(gs[1, 3:])
    xs = np.arange(128); prof = {k: [] for k in ("size", "wall", "lumen", "flat")}
    for x in xs:
        m = occ[:, :, x] > 0.5
        for k, i in (("size", 0), ("wall", 1), ("lumen", 2), ("flat", 3)):
            prof[k].append(D[i][:, :, x][m].mean() if m.any() else np.nan)
    ax.plot(xs, prof["size"], label="outer radius (vox)"); ax.plot(xs, prof["wall"], label="wall (vox)")
    ax2 = ax.twinx(); ax2.plot(xs, prof["lumen"], "k-", label="lumen fraction"); ax2.plot(xs, prof["flat"], "k--", label="flattening (minor/major)"); ax2.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax.set_xlabel("x (voxels along the vessel)")
    ax.set_title("descriptors along the vessel (the labels): one object throughout", fontsize=9)
    fig.suptitle("A vessel that narrows and collapses: one population, its lumen fraction falls to 0 while the wall continues", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "scene_collapse.png", dpi=55); plt.close(fig)


def modalities(out: Path):
    mods = ["hipct", "ct", "mri", "histology"]
    fig, axs = plt.subplots(2, 5, figsize=(20, 8.4), facecolor="white")
    seeds, sd = [], 8 * 30
    while len(seeds) < 2:                                                    # crops that show objects
        _, d0 = draw_scene((64, 64, 64), seed=sd, device=DEV, strength=0.0, modality="hipct", n_groups=(3, 4))
        if float((d0["ids"][0] >= 0).float().mean()) > 0.05:
            seeds.append(sd)
        sd += 8
    for row, seed in enumerate(seeds):
        for j, m in enumerate(mods):
            s, d = draw_scene((64, 64, 64), seed=seed, device=DEV, strength=0.5, modality=m, n_groups=(3, 4))
            axs[row, j].imshow(d["image"][32].cpu().numpy(), cmap="gray", vmin=0, vmax=255)
            extra = [k for k in ("tear", "fold", "local_tomography") if k in d["record"]]
            axs[row, j].set_title(f"{m} {extra if extra else ''}", fontsize=9)
        axs[row, 4].imshow(s.image[32].cpu().numpy(), cmap="gray"); axs[row, 4].set_title(f"clean ({d['voxel_um']:.1f} µm voxels)", fontsize=9)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("One crop of a scene under each acquisition model (strength 0.5)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / "scene_modalities.png", dpi=55); plt.close(fig)


def pairs(out: Path):
    from shapetensor.learn.engine import PopulationEngine
    fig, axs = plt.subplots(4, 5, figsize=(20, 16), facecolor="white")
    for row in range(4):
        seed = 1000 + 8 * row * 3
        s, d = draw_scene((64, 64, 64), seed=seed, device=DEV, strength=0.4, n_groups=(2, 5))
        t = target(s, d); k = 32; kc = 8
        m = t["mass"].float(); _, V = eig3(t["slots"][0].float().movedim(0, 0))
        img = d["image"][k].cpu().numpy()
        axs[row, 0].imshow(img, cmap="gray", vmin=0, vmax=255); axs[row, 0].set_title(f"{d['record']['modality']}, {d['voxel_um']:.1f} µm: {[list(f)[0] for f in d['record']['families']]}", fontsize=8)
        axs[row, 1].imshow((dec_rgb(V[2][:, kc].cpu().numpy()) * (m[0][:, kc] / 0.2).clamp(0, 1).cpu().numpy()[..., None]).clip(0, 1), interpolation="nearest"); axs[row, 1].set_title("target: heaviest slot axis x mass", fontsize=9)
        n = (m[:, :, kc] > 0.05).sum(0).cpu().numpy(); h = axs[row, 2].imshow(n, cmap="viridis", vmin=0, vmax=4, interpolation="nearest"); plt.colorbar(h, ax=axs[row, 2], fraction=0.046)
        axs[row, 2].set_title("target: populations per cell (mass > 0.05)", fontsize=9)
        axs[row, 3].imshow(id_rgb(d["ids"][0, k].cpu().numpy())); axs[row, 3].set_title(f"identities ({int(torch.unique(d['ids'][0]).numel()) - 1} objects in the crop)", fontsize=9)
        shp = [i for i, L in enumerate(s.layers) if L.role == "shape"]
        rec_ = (d["recoverability"][shp] * d["occupancy"][shp]).sum(0) / d["occupancy"][shp].sum(0).clamp_min(1e-6)
        h = axs[row, 4].imshow(rec_[k].cpu().numpy(), cmap="magma", vmin=0, vmax=1); plt.colorbar(h, ax=axs[row, 4], fraction=0.046)
        axs[row, 4].set_title("orientation recoverability of the shapes (confidence target)", fontsize=9)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Training pairs from object scenes (64³ crops of 128³ scenes) and their targets", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / "scene_pairs.png", dpi=50); plt.close(fig)


def _plane(vol: torch.Tensor, c: torch.Tensor, u: torch.Tensor, v: torch.Tensor, half: int = 40) -> np.ndarray:
    """A planar slice of ``vol (C, Z, Y, X)`` through ``c`` spanned by unit ``u`` (rows) and ``v`` (columns)."""
    t = torch.arange(-half, half + 1, device=vol.device, dtype=torch.float32)
    P = c + t[:, None, None] * u + t[None, :, None] * v                                                 # (n, n, 3)
    sz = torch.tensor(vol.shape[-3:], device=vol.device, dtype=torch.float32)
    g = (2 * P / (sz - 1) - 1).flip(-1)[None, None]
    return torch.nn.functional.grid_sample(vol[None].float(), g, align_corners=True)[0, :, 0].permute(1, 2, 0).cpu().numpy()


def crossings(out: Path):
    import shapetensor.learn.scene as S
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for j, inter in enumerate((False, True)):
        sc = Scene((96, 96, 96), 1, DEV, 2.0); c = torch.tensor([48.0, 48.0, 48.0], device=DEV)
        for _ in range(50):
            a = sc.bundle(None, "single", axis=torch.tensor([0.0, 0.0, 1.0], device=DEV), centre=c, radius=14.0, curv=1e-4)
            if a is not None and a.offsets is not None:
                break
        for _ in range(50):
            b = sc.bundle(a.centreline, "angled", curv=1e-4)
            if b is not None and not isinstance(b, list) and b.offsets is not None:
                break
        if inter:
            sc.interleave(a, b)
        sc.groups = [a, b]; ren = sc.render()
        o = torch.stack([ren["groups"][0]["occupancy"], ren["groups"][1]["occupancy"], torch.zeros_like(ren["groups"][0]["occupancy"])])
        P0, tA, dB = b.cross; n = torch.nn.functional.normalize(torch.linalg.cross(tA, dB), dim=0)
        both = int(((o[0] > 0.05) & (o[1] > 0.05)).sum())
        ax[j].imshow(np.clip(_plane(o, P0, n, tA), 0, 1)); ax[j].set_title(f"{'interleaved' if inter else 'passing through'}: {both} shared voxels\n(rows: crossing normal, cols: red bundle axis)")
    fam = {k: (1.0 if k == "bundle" else 0.0) for k in Scene.FAMILIES}
    for sd in range(400):
        sc = Scene((96, 96, 96), sd, DEV, 3.0).build(fam, (1, 1), 0.0)
        if "split" in sc.record and len(sc.groups) == 2:
            break
    ren = sc.render(); o = torch.stack([g["occupancy"] for g in ren["groups"]]); mip = o.amax(1)
    ax[2].imshow(np.clip(torch.stack([mip[0], mip[1], torch.zeros_like(mip[0])], -1).cpu().numpy(), 0, 1)); ax[2].set_title(f"a tract splitting ({sc.record['split'][0]} deg): two populations, MIP")
    sc = Scene((96, 96, 96), 7, DEV, 4.0); c = torch.tensor([48.0, 48.0, 48.0], device=DEV)
    a = sc.bundle(None, "single", axis=torch.tensor([0.0, 0.0, 1.0], device=DEV), centre=c, radius=10.0, curv=1e-4)
    b = sc.bundle(a.centreline, "kissing", curv=1e-4)
    if isinstance(b, list):
        b = b[0]
    cl = a.centreline; b.curves = [sc.deflect(cv, extra=[(cl.P[0, ::4], cl.rad[0, ::4])]) for cv in b.curves]
    sc.groups = [a, b]; ren = sc.render(); o = torch.stack([g["occupancy"] for g in ren["groups"]]); mip = o.amax(2)
    ax[3].imshow(np.clip(torch.stack([mip[0], mip[1], torch.zeros_like(mip[0])], -1).cpu().numpy(), 0, 1)); ax[3].set_title("kissing: the newer bundle yields (MIP)")
    for a_ in ax:
        a_.axis("off")
    plt.tight_layout(); fig.savefig(out / "scene_crossings.png", dpi=60); plt.close(fig)


def physics(out: Path):
    from shapetensor.learn.scene import draw_pair
    modes = ("additive", "xray", "darkfield", "histology", "pli", "mri")
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
    for j, ph in enumerate(modes):
        s, d = draw_scene((96, 96, 96), seed=8 * 1234, device=DEV, strength=0.3, physics=ph, p_warp=0.0, canvas=192, modality="mixed")
        a_ = ax.flat[j]; a_.imshow(d["image"][48].float().cpu().numpy(), cmap="gray"); a_.set_title(f"{ph}"); a_.axis("off")
    (s1, d1), (s2, d2), info = draw_pair((96, 96, 96), seed=8 * 77, device=DEV, kind="zoom", strength=0.3, canvas=192, p_warp=0.0)
    ax.flat[6].imshow(d1["image"][48].float().cpu().numpy(), cmap="gray"); ax.flat[6].set_title(f"zoom pair: fine ({d1['voxel_um']:.1f} um)")
    ax.flat[7].imshow(d2["image"][48 // 2 + 4 * info["offset_cells"][0]].float().cpu().numpy(), cmap="gray"); ax.flat[7].set_title(f"coarse ({d2['voxel_um']:.1f} um), 2x the extent")
    for a_ in ax.flat:
        a_.axis("off")
    plt.tight_layout(); fig.savefig(out / "scene_physics.png", dpi=55); plt.close(fig)


def new_families(out: Path):
    names = ["plywood", "fascicles", "trabecular", "tubules", "epithelium", "gland", "dendrite", "crystals"]
    vox = {"dendrite": 1.0, "fascicles": 3.0, "gland": 4.0}
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
    cols = torch.tensor([[1, 0.2, 0.2], [0.2, 1, 0.2], [0.2, 0.6, 1], [1, 1, 0.2]], device=DEV)
    for a_, name in zip(ax.flat, names):
        fam = {k: (1.0 if k == name else 0.0) for k in Scene.FAMILIES}
        for sd in range(5, 30):
            sc = Scene((128, 128, 128), sd, DEV, vox.get(name, 4.0)).build(fam, (1, 1), 0.0)
            if sc.groups:
                ren = sc.render()
                o = torch.stack([g["occupancy"] for g in ren["groups"]])
                if float(o.sum()) > 200:
                    break
        k = int(o.sum((0, 2, 3)).argmax())
        rgb = (o[:, k, :, :, None] * cols[:o.shape[0], None, None, :]).sum(0).clamp(0, 1)
        a_.imshow(rgb.cpu().numpy()); a_.set_title(f"{name}: " + ", ".join(sorted({g['group'].name for g in ren['groups']}))); a_.axis("off")
    plt.tight_layout(); fig.savefig(out / "scene_new_families.png", dpi=55); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/andrew/bridge_wmcov/shapetensor/results/dev_pipeline/scene_generator")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("families", families), ("collapse", collapse), ("modalities", modalities), ("pairs", pairs), ("crossings", crossings),
                     ("physics", physics), ("new_families", new_families)):
        if a.only and name not in a.only:
            continue
        fn(out); print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
