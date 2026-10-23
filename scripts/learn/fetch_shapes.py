"""Build the real-shape library the scene generator can draw from (instead of only procedural shapes).

    python scripts/learn/fetch_shapes.py [--out DIR] [--neurons 240] [--no-tracts] [--no-surfaces]

* **neurons** -- reconstructed morphologies from NeuroMorpho.org (standardised SWC: node, type, x, y, z, radius,
  parent; um), across species and cell classes (pyramidal and other principal cells, interneurons, Purkinje cells,
  glia), saved as ``neurons/<name>.npz`` (nodes, radii, parents, types);
* **tracts** -- the HCP-1065 population tractography atlas (Yeh 2022, CC BY-SA 4.0): ~80 white-matter tracts as
  streamlines (MNI, mm), saved as ``tracts/<tract>.npz`` (streamlines resampled to 0.5 mm, float16, offsets);
* **surfaces** -- the BigBrain cortical layer surfaces (Wagstyl et al. 2020; local GIFTIs): pial (0) to white (6),
  left and right, saved as ``surfaces/layer<k>_<hemi>.npz`` (vertices um, faces, vertex normals).
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

OUT = Path("/home/andrew/bridge_wmcov/shapetensor/shape_library")
API = "https://neuromorpho.org/api/neuron/select"
QUERIES = [("species:human", "cell_type:pyramidal"), ("species:human", "cell_type:interneuron"), ("species:mouse", "cell_type:pyramidal"),
           ("species:mouse", "cell_type:interneuron"), ("species:rat", "cell_type:Purkinje"), ("species:rat", "cell_type:pyramidal"),
           ("cell_type:Glia", None), ("species:mouse", "cell_type:Purkinje"), ("species:human", "cell_type:principal cell"), ("species:monkey", "cell_type:pyramidal")]
TRACTS = "https://github.com/data-others/atlas/releases/download/hcp1065/hcp1065_avg_tracts_trk.zip"
GII = Path("/hdd/andrew/Data/BigBrain2015/Layer_Segmentation_PLoSBiology2020_gii")


def get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "shapetensor-shape-library/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_swc(text: str):
    rows = [l.split() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    a = np.array([[float(v) for v in r[:7]] for r in rows if len(r) >= 7])
    ids = a[:, 0].astype(np.int64); lut = {int(i): k for k, i in enumerate(ids)}
    par = np.array([lut.get(int(p), -1) for p in a[:, 6]], dtype=np.int64)
    return a[:, 2:5].astype(np.float32), a[:, 5].astype(np.float32), par, a[:, 1].astype(np.int8)


def neurons(out: Path, n_total: int):
    od = out / "neurons"; od.mkdir(parents=True, exist_ok=True)
    per = max(1, n_total // len(QUERIES)); got = len(list(od.glob("*.npz")))
    for q, fq in QUERIES:
        try:
            d = json.loads(get(f"{API}?q={urllib.parse.quote(q)}" + (f"&fq={urllib.parse.quote(fq)}" if fq else "") + f"&size={per}&page=0"))
        except Exception as e:
            print("query failed", q, e, flush=True); continue
        for x in d.get("_embedded", {}).get("neuronResources", []):
            name = x["neuron_name"]; f = od / f"{name}.npz"
            if f.exists():
                continue
            url = f"https://neuromorpho.org/dableFiles/{x['archive'].lower().replace(' ', '%20')}/CNG%20version/{urllib.parse.quote(name)}.CNG.swc"
            try:
                P, R, par, typ = parse_swc(get(url).decode("latin-1"))
                if P.shape[0] < 20:
                    continue
                np.savez_compressed(f, nodes=P, radii=R, parents=par, types=typ, species=x.get("species", ""), cell_type=",".join(x.get("cell_type", []) or []))
                got += 1
            except Exception as e:
                print("skip", name, str(e)[:80], flush=True)
            time.sleep(0.2)
        print(f"neurons: {got} after '{q} | {fq}'", flush=True)


def tracts(out: Path):
    import nibabel as nib
    from nibabel.streamlines import load as sl_load
    od = out / "tracts"; od.mkdir(parents=True, exist_ok=True)
    if len(list(od.glob("*.npz"))) > 40:
        print("tracts: present"); return
    zf = out / "hcp1065_avg_tracts_trk.zip"
    if not zf.exists():
        print("downloading the tract atlas (588 MB)...", flush=True)
        tmp = zf.with_suffix(".part"); tmp.write_bytes(get(TRACTS, timeout=1800)); tmp.rename(zf)
    with zipfile.ZipFile(zf) as z:
        names = [n for n in z.namelist() if n.endswith(".trk") or n.endswith(".trk.gz")]
        for n in names:
            stem = Path(n).name.replace(".trk.gz", "").replace(".trk", "")
            f = od / f"{stem}.npz"
            if f.exists():
                continue
            data = z.read(n)
            if n.endswith(".gz"):
                import gzip
                data = gzip.decompress(data)
            t = sl_load(io.BytesIO(data))
            sls, offs = [], [0]
            for s in t.streamlines:
                s = np.asarray(s, dtype=np.float32)
                if s.shape[0] < 4:
                    continue
                seg = np.linalg.norm(np.diff(s, axis=0), axis=1); cum = np.concatenate([[0], np.cumsum(seg)])
                if cum[-1] < 2:
                    continue
                q = np.arange(0, cum[-1], 0.5)
                rs = np.stack([np.interp(q, cum, s[:, k]) for k in range(3)], 1)
                sls.append(rs.astype(np.float16)); offs.append(offs[-1] + rs.shape[0])
            if sls:
                np.savez_compressed(f, points=np.concatenate(sls), offsets=np.array(offs, dtype=np.int64))
                print(f"tract {stem}: {len(sls)} streamlines", flush=True)


def surfaces(out: Path):
    import nibabel as nib
    od = out / "surfaces"; od.mkdir(parents=True, exist_ok=True)
    for g in sorted(GII.glob("layer*_*.gii*")):
        k = g.name.split("_")[0]; hemi = g.name.split("_")[1]
        f = od / f"{k}_{hemi}.npz"
        if f.exists():
            continue
        img = nib.load(str(g))
        V = img.darrays[0].data.astype(np.float32); F = img.darrays[1].data.astype(np.int32)
        if V.shape[1] != 3:
            V, F = img.darrays[1].data.astype(np.float32), img.darrays[0].data.astype(np.int32)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        N = np.zeros_like(V)
        for j in range(3):
            np.add.at(N, F[:, j], fn)
        N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-9)
        scale = 1000.0 if np.abs(V).max() < 1000 else 1.0                                       # mm -> um
        np.savez_compressed(f, vertices=V * scale, faces=F, normals=N.astype(np.float32))
        print(f"surface {k}_{hemi}: {V.shape[0]} vertices", flush=True)


def main():
    import urllib.parse  # noqa: F401
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(OUT)); ap.add_argument("--neurons", type=int, default=240)
    ap.add_argument("--no-tracts", action="store_true"); ap.add_argument("--no-surfaces", action="store_true"); ap.add_argument("--no-neurons", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.no_surfaces:
        surfaces(out)
    if not a.no_neurons:
        neurons(out, a.neurons)
    if not a.no_tracts:
        tracts(out)
    print("done", flush=True)


if __name__ == "__main__":
    import urllib.parse
    main()
