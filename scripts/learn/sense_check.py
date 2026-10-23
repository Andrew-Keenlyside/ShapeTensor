"""A multimodal sense-check set of real volumes, and where the generator and the models fail on it.

    python scripts/learn/sense_check.py fetch --catalogue sources.json --out DIR [--crops 4] [--size 96]
    python scripts/learn/sense_check.py analyse --data DIR --models A.pt ... [--out DIR]

``fetch`` reads a few small crops from each catalogue entry (OME-Zarr / Zarr v2-v3 / N5 / Neuroglancer precomputed
over HTTPS, GCS or S3 through tensorstore, or a small single file: NIfTI, TIFF, HDF5, MRC), keeping crops with
enough structure (background of either polarity rejected), and saves them as uint8/uint16 with their metadata
(organism, tissue, modality, voxel size).

``analyse``:
* **generator coverage** -- the label-free image statistics of each real crop (domain_gap.stats, identical
  normalisation) against a pool of synthetic crops from the full training mixture: a novelty score (Mahalanobis
  distance to the synthetic distribution, shrunk covariance, in units of the synthetic crops' own spread) and the
  statistics the generator never reaches (|z| > 3 against the synthetic pool), per source;
* **model outputs** -- each model's density on each crop (lobes rendered, SH heads read), the top direction's
  colour and strength beside the image and the structure tensor's, as a montage per source, and the share of cells
  with a calibrated peak (does the model see structure there at all).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import urllib.parse
from pathlib import Path

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------------------------------------------

def _kvstore(url: str):
    if url.startswith("/") or url.startswith("file://"):
        return {"driver": "file", "path": url.replace("file://", "")}
    if url.startswith("gs://"):
        b, _, p = url[5:].partition("/"); return {"driver": "gcs", "bucket": b, "path": p}
    if url.startswith("s3://"):
        b, _, p = url[5:].partition("/"); return {"driver": "s3", "bucket": b, "path": p, "aws_region": os.environ.get("AWS_REGION", "us-east-1")}
    return url if url.endswith("/") else url + "/"


def open_remote(e: dict):
    """A tensorstore handle for a catalogue entry (format: zarr / zarr3 / n5 / precomputed)."""
    import tensorstore as ts
    fmt = e["format"].lower(); url = e["url"].rstrip("/"); path = (e.get("array_path") or "").strip("/")
    full = url + ("/" + path if path else "")
    if "precomputed" in fmt:
        spec = {"driver": "neuroglancer_precomputed", "kvstore": _kvstore(url)}
        if e.get("scale_index") is not None:
            spec["scale_index"] = int(e["scale_index"])
        if e.get("scale_key"):
            spec["scale_metadata"] = {"key": e["scale_key"]}
    elif "n5" in fmt:
        spec = {"driver": "n5", "kvstore": _kvstore(full)}
    elif "zarr3" in fmt or "zarr v3" in fmt or "v3" in fmt:
        spec = {"driver": "zarr3", "kvstore": _kvstore(full)}
    else:
        spec = {"driver": "zarr", "kvstore": _kvstore(full)}
    if not isinstance(spec["kvstore"], dict) and spec["kvstore"].startswith("http"):
        spec["kvstore"] = {"driver": "http", "base_url": spec["kvstore"]}
    return ts.open(spec, read=True).result()


def _spatial_view(arr, e: dict):
    """Index away non-spatial axes (channel / time: index from the entry or 0), leaving three spatial axes with a
    zero-based domain; returns the view and their order (``axes`` in the entry, e.g. "zyx", "xyz", "cxyz"; precomputed
    volumes are x, y, z, channel)."""
    import tensorstore as ts
    nd = arr.rank
    axes = e.get("axes") or ("xyzc" if "precomputed" in e["format"].lower() else ("zyx" if nd == 3 else None))
    if axes is None:                                                                        # leading non-spatial axes (OME: t, c)
        axes = "t" * (nd - 4) + "czyx" if nd >= 4 else "zyx"
    idx = []
    for a in axes:
        idx.append(int(e.get("time", 0)) if a == "t" else int(e.get("channel", 0)) if a == "c" else slice(None))
    v = arr[tuple(idx)]
    v = v.translate_to[(0,) * v.rank]                                                       # zero-based (precomputed has a voxel offset)
    return v, tuple(a for a in axes if a in "xyz")


def _probe_precomputed(e: dict, size: int, rng, n: int = 64):
    """Crop origins on tissue for a sparse precomputed volume: the coarsest scale read whole, its tissue (not an end
    mode of the histogram) sampled, mapped to the target scale. Zero-based origins in the target's (x, y, z)."""
    import tensorstore as ts
    import urllib.request
    from shapetensor.learn.realdata import background_mask
    url = e["url"].rstrip("/")
    info = json.loads(urllib.request.urlopen(url + "/info", timeout=60).read())
    sc = info["scales"]; ti = next((i for i, s_ in enumerate(sc) if s_["key"] == e.get("scale_key")), int(e.get("scale_index", 0)))
    ci = int(np.argmin([np.prod(s_["size"], dtype=float) for s_ in sc]))                    # the smallest scale
    if np.prod(sc[ci]["size"], dtype=float) > 1e9:
        print(f"  probe skipped: coarsest scale {sc[ci]['size']} too large"); return []
    coarse = ts.open({"driver": "neuroglancer_precomputed", "kvstore": {"driver": "http", "base_url": url}, "scale_index": ci}, read=True).result()
    c = np.asarray(coarse[..., 0].translate_to[0, 0, 0].read().result()).astype(np.float32)
    if e.get("probe_mode") == "bright":                                                         # sparse fluorescence: the brightest voxels
        tissue = c > np.quantile(c, 0.995)
    else:
        tissue = ~background_mask(torch.tensor(c)).numpy() & (c > 0)
    pts = np.argwhere(tissue)
    if pts.shape[0] == 0:
        return []
    rc, rt = np.array(sc[ci]["resolution"], float), np.array(sc[ti]["resolution"], float)
    out = []
    for p_ in pts[rng.integers(0, pts.shape[0], n)]:
        tgt = (p_ * rc / rt).astype(int) - size // 2
        out.append([max(0, min(int(t), int(sz) - size)) for t, sz in zip(tgt, sc[ti]["size"])])
    return out


def _decode_bytes(data: bytes, name: str) -> np.ndarray:
    """One slice (or stack) from bytes by its extension: TIFF, BMP / PNG / JPEG (grey), DICOM."""
    import io
    n = name.lower()
    if n.endswith((".tif", ".tiff")):
        import tifffile
        return tifffile.imread(io.BytesIO(data))
    if n.endswith((".bmp", ".png", ".jpg", ".jpeg")):
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        return np.asarray(im.convert("L") if im.mode not in ("L", "I;16", "I") else im)
    if n.endswith(".dcm") or "." not in Path(n).name:
        import tempfile
        import SimpleITK as sitk
        with tempfile.NamedTemporaryFile(suffix=".dcm") as tf:
            tf.write(data); tf.flush()
            return sitk.GetArrayFromImage(sitk.ReadImage(tf.name))[0]
    raise ValueError(f"cannot decode {name}")


def _http_range(url: str, a: int, b: int) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": "shapetensor-sense-check"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _cached_download(e: dict):
    """The whole remote file in the cache when it is at most ``max_mb`` (default 1500 MB) -- many range reads of a zip
    run into rate limits (Zenodo) -- else None (read it by range). Streamed, written under a temporary name first."""
    import shutil
    import urllib.request
    cache = Path(os.environ.get("SENSE_CACHE", "/home/andrew/bridge_wmcov/shapetensor/sense_check/cache")); cache.mkdir(parents=True, exist_ok=True)
    f = cache / (e.get("filename") or Path(e["url"].split("?")[0]).name)
    if f.exists():
        return str(f)
    req = urllib.request.Request(e["url"], headers={"User-Agent": "shapetensor-sense-check"})
    with urllib.request.urlopen(req, timeout=600) as r:
        n = int(r.headers.get("Content-Length", 0) or 0)
        if not n or n > int(e.get("max_mb", 1500)) * 2 ** 20:
            return None
        tmp = f.with_suffix(f.suffix + ".part")
        with open(tmp, "wb") as w:
            shutil.copyfileobj(r, w, 2 ** 22)
    tmp.rename(f)
    return str(f)


class RawAccessor:
    """A headerless raw volume (x fastest; numpy z, y, x) read by HTTP range, one request per slice of a crop:
    ``raw_shape`` [Z, Y, X], ``dtype``, ``offset`` (bytes before the data)."""

    def __init__(self, e: dict):
        self.url = e["url"]; self.dt = np.dtype(e.get("dtype", "uint8")); self.off = int(e.get("offset", 0))
        if e.get("tiff_offset"):                                                                # a contiguous TIFF: offset, shape, type from its header
            import fsspec
            import tifffile
            with tifffile.TiffFile(fsspec.open(self.url).open()) as t:
                sr = t.series[0]; self.off = int(sr.dataoffset); shp = tuple(sr.shape); self.dt = np.dtype(sr.dtype)
        else:
            shp = tuple(int(v) for v in e["raw_shape"])
        if len(shp) == 4:                                                                      # C, Z, Y, X: one channel
            self.off += int(e.get("channel", 0)) * int(np.prod(shp[1:])) * self.dt.itemsize; shp = shp[1:]
        self.shape = shp

    def get(self, o, ext):
        Z, Y, X = self.shape; b = self.dt.itemsize; out = []
        for z in range(o[0], o[0] + ext[0]):
            a = self.off + ((z * Y + o[1]) * X) * b
            raw = _http_range(self.url, a, a + ext[1] * X * b)
            out.append(np.frombuffer(raw, self.dt).reshape(ext[1], X)[:, o[2]:o[2] + ext[2]])
        return np.stack(out)


class SliceAccessor:
    """A stack of one file per slice: numbered URLs (``url_pattern`` with ``{i}`` and a format such as ``{i:04d}``,
    ``index_range`` [first, last]) or members of a remote zip (``url`` + ``member_filter``); only the slices a crop
    needs are fetched (and cached in memory)."""

    def __init__(self, e: dict):
        self.e = e; self.cache = {}
        if e["format"] == "zip_slices":
            import zipfile
            self.zip = zipfile.ZipFile(_cached_download(e) or __import__("fsspec").open(e["url"]).open())
            filt = e.get("member_filter", ""); ex = tuple(e.get("extensions", (".tif", ".tiff", ".bmp", ".png", ".dcm")))
            self.names = sorted(n for n in self.zip.namelist() if filt in n and n.lower().endswith(ex) and "__MACOSX" not in n)
            w = int(e.get("window", 0))
            if w and len(self.names) > w:                                                     # one slab of slices for all crops
                c = int(float(e.get("centre", 0.5)) * len(self.names)); a = max(0, min(len(self.names) - w, c - w // 2))
                self.names = self.names[a:a + w]
            self.n = len(self.names)
        else:
            self.i0, self.i1 = [int(v) for v in e["index_range"]]; self.n = self.i1 - self.i0 + 1
        first = self._slice(self.n // 2)
        self.shape = (self.n,) + tuple(first.shape[-2:])

    def _slice(self, k: int) -> np.ndarray:
        if k not in self.cache:
            if self.e["format"] == "zip_slices":
                nm = self.names[k]; arr = _decode_bytes(self.zip.read(nm), nm)
            else:
                import urllib.request
                url = self.e["url_pattern"].format(i=self.i0 + k)
                req = urllib.request.Request(url, headers={"User-Agent": "shapetensor-sense-check"})
                arr = _decode_bytes(urllib.request.urlopen(req, timeout=180).read(), url.split("?")[0])
            while arr.ndim > 2:
                arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[k] = arr
        return self.cache[k]

    def get(self, o, ext):
        return np.stack([self._slice(z)[o[1]:o[1] + ext[1], o[2]:o[2] + ext[2]] for z in range(o[0], o[0] + ext[0])])


class TiffZarrAccessor:
    """A remote tiled (pyramidal) TIFF read tile by tile through tifffile's zarr view: ``level``; z, y, x."""

    def __init__(self, e: dict):
        import fsspec
        import tifffile
        import zarr
        self.t = tifffile.TiffFile(fsspec.open(e["url"]).open())
        sr = self.t.series[0]; lv = sr.levels[int(e.get("level", 0))]
        self.arr = zarr.open(lv.aszarr(), mode="r"); self.shape = tuple(self.arr.shape[-3:])

    def get(self, o, ext):
        sl = tuple(slice(a, a + n) for a, n in zip(o, ext))
        return np.asarray(self.arr[(0,) * (self.arr.ndim - 3) + sl])


class ZarrPyAccessor:
    """A remote Zarr v2 array through zarr-python (stores tensorstore rejects, e.g. a string fill value); ``axes`` as
    for tensorstore views, ``channel`` / ``time`` indexed away."""

    def __init__(self, e: dict):
        import zarr
        url = e["url"].rstrip("/") + "/" + (e.get("array_path") or "").strip("/")
        self.arr = zarr.open_array(url, mode="r", zarr_format=2)
        ax = e.get("axes") or "zyx"; self.lead = tuple(int(e.get("time", 0)) if a == "t" else int(e.get("channel", 0)) for a in ax if a in "tc")
        self.axes = tuple(a for a in ax if a in "xyz"); self.shape = tuple(self.arr.shape[len(self.lead):])

    def get(self, o, ext):
        return np.asarray(self.arr[self.lead + tuple(slice(a, a + n) for a, n in zip(o, ext))])


def _structure(x: np.ndarray) -> float:
    """Structure against noise: the spread of a smoothed crop (sigma 1.5) over the spread of what smoothing removed,
    on the robust 0-1 range. Pure noise ~0.1-0.2; cryo-ET ~0.1-0.6; resolved tissue 1-10."""
    from scipy.ndimage import gaussian_filter
    x = x.astype(np.float32); lo, hi = np.percentile(x, [0.5, 99.5]); x = (x - lo) / max(hi - lo, 1e-6)
    g = gaussian_filter(x, 1.5)
    return float(g.std() / (np.std(x - g) + 1e-6))


def _extent(e: dict, shp_zyx, size: int):
    """The read extent (z, y, x) for an output crop of ``size``^3 and its isotropic voxel: with ``voxel_zyx`` given
    (anisotropic data) the target voxel is the geometric mean of the axes' (or ``target_um``), shrunk until the
    extent fits the volume; else ``size`` on every axis at the native voxel."""
    v = e.get("voxel_zyx")
    if not v:
        return (size,) * 3, e.get("voxel_um")
    v = [float(a) for a in v]; t = float(e.get("target_um") or np.prod(v) ** (1 / 3))
    t = min(t, min(s_ * a / size for s_, a in zip(shp_zyx, v)))
    return tuple(max(1, int(round(size * t / a))) for a in v), round(t, 4)


def _to_size(x: np.ndarray, size: int) -> np.ndarray:
    """``x (z, y, x)`` resampled to ``size``^3: area averaging on the axes that shrink, then linear on those that grow."""
    if x.shape == (size,) * 3:
        return x
    import torch.nn.functional as F
    t = torch.as_tensor(x.astype(np.float32))[None, None]
    t = F.interpolate(t, size=tuple(min(a, size) for a in x.shape), mode="area")
    t = F.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)
    return t[0, 0].numpy()


def _probe_view(e: dict, view, axes, size: int, rng, n: int = 64):
    """Crop origins on signal for a sparse store: the coarse level ``probe_path`` read whole, voxels picked by
    ``probe_mode`` ("bright": the top 0.5 % -- sparse fluorescence; "tissue": not background), mapped to the target by
    the shape ratio. Origins in the view's axis order."""
    from shapetensor.learn.realdata import background_mask
    pe = dict(e, array_path=e["probe_path"]); arr = open_remote(pe); cv, cax = _spatial_view(arr, pe)
    c = np.asarray(cv.read().result()).astype(np.float32)
    if (e.get("probe_mode") or "bright") == "bright":
        m = c > np.quantile(c, 0.995)
    else:
        m = ~background_mask(torch.tensor(c)).numpy() & (c > 0)
    pts = np.argwhere(m)
    if pts.shape[0] == 0:
        return []
    ratio = np.array(view.shape, float) / np.array(c.shape, float); out = []
    for p_ in pts[rng.integers(0, pts.shape[0], n)]:
        tgt = (p_ * ratio).astype(int) - size // 2
        out.append([max(0, min(int(t), int(sz) - size)) for t, sz in zip(tgt, view.shape)])
    return out


def fetch_entry(e: dict, out: Path, crops: int = 4, size: int = 96, tries: int = 12, seed: int = 0) -> int:
    """``crops`` crops of ``size``^3 (``size`` per entry overrides) from one catalogue entry, kept when at least
    ``min_tissue`` (default 0.5) is not background; anisotropic entries (``voxel_zyx``) read over a per-axis extent and
    resampled to isotropic voxels."""
    from shapetensor.learn.realdata import tissue_share
    rng = np.random.default_rng(seed); size = int(e.get("size", size)); min_tissue = float(e.get("min_tissue", 0.5))
    lazy = None
    try:
        fmt = e["format"].lower()
        if fmt == "raw":
            lazy = RawAccessor(e)
        elif fmt in ("slices", "zip_slices"):
            lazy = SliceAccessor(e)
        elif fmt == "tiff_zarr":
            lazy = TiffZarrAccessor(e)
        elif fmt == "zarr_py":
            lazy = ZarrPyAccessor(e)
        arr = open_remote(e) if (lazy is None and fmt not in ("file", "nifti", "tiff", "hdf5", "mrc", "nrrd", "dicom", "zip_tiff")) else None
    except Exception as ex:
        print(f"  open failed: {ex!r}"[:300]); return 0
    if lazy is not None:
        axes = getattr(lazy, "axes", ("z", "y", "x")); shp = lazy.shape; get = lazy.get
        if fmt in ("raw", "slices", "zip_slices"):
            tries = min(tries, 4)                                                            # each try downloads slices
    elif arr is None:
        vol = _read_file(e)
        if vol is None:
            return 0
        get = lambda o, ext: vol[o[0]:o[0] + ext[0], o[1]:o[1] + ext[1], o[2]:o[2] + ext[2]]
        shp = vol.shape; axes = ("z", "y", "x")
    else:
        view, axes = _spatial_view(arr, e); shp = view.shape
        get = lambda o, ext: np.asarray(view[tuple(slice(a, a + n) for a, n in zip(o, ext))].read().result())
    shp_zyx = tuple(shp[axes.index(a)] for a in "zyx")
    ext_zyx, vox = _extent(e, shp_zyx, size)
    ext = tuple(ext_zyx["zyx".index(a)] for a in axes)                                       # in storage order
    if any(n > s_ for n, s_ in zip(ext, shp)):
        print(f"  too small {shp}"); return 0
    probes = []
    try:
        if e.get("probe") and "precomputed" in fmt:
            probes = _probe_precomputed(e, size, rng)
        elif e.get("probe_path") and arr is not None:
            probes = _probe_view(e, view, axes, max(ext), rng)
    except Exception as ex:
        print(f"  probe failed: {ex!r}"[:200])
    reg = e.get("region")                                                                   # fractions of the shape, per spatial axis in storage order
    got = 0; cands = []; want = int(e.get("candidates", 2)) * crops
    for t in range(tries * crops):
        if len(cands) >= want:
            break
        if probes:
            o = probes[t % len(probes)]
        else:
            if reg:
                lo = [int(r_[0] * s_) for r_, s_ in zip(reg, shp)]; hi = [max(lo_ + 1, int(r_[1] * s_) - n) for lo_, r_, s_, n in zip(lo, reg, shp, ext)]
            else:
                lo = [int(0.1 * s_) for s_ in shp]; hi = [max(lo_ + 1, int(0.9 * s_) - n) for lo_, s_, n in zip(lo, shp, ext)]
            o = [int(rng.integers(a, b)) if b > a else 0 for a, b in zip(lo, hi)]
        o = [max(0, min(v, s_ - n)) for v, s_, n in zip(o, shp, ext)]
        try:
            x = get(o, ext)
        except Exception as ex:
            print(f"  read failed: {ex!r}"[:200]); return got
        x = np.squeeze(x)
        if x.shape != ext or float(np.std(x)) == 0 or np.unique(x[::2, ::2, ::2]).size < 24:      # empty, saturated or a label volume
            continue
        if axes != ("z", "y", "x"):                                                          # to z, y, x
            x = np.ascontiguousarray(x.transpose([axes.index(a) for a in "zyx"]))
        if ext_zyx != (size,) * 3:
            x = _to_size(x, size)
        if tissue_share(x.astype(np.float32)) < min_tissue:
            continue
        cands.append((_structure(x), x, o))
    meta = dict(e, voxel_um=vox, native_voxel_um=e.get("voxel_um")) if e.get("voxel_zyx") else e
    for sc, x, o in sorted(cands, key=lambda c: -c[0])[:crops]:                                # the better-structured half: no crop of empty medium
        np.savez_compressed(out / f"{e['name']}_{got}.npz", image=x, origin=np.array(o), meta=json.dumps(dict(meta, structure=round(float(sc), 3))))
        got += 1
    print(f"  {got} crops from shape {shp} ({x.dtype if got else '-'}){f', read {ext_zyx} -> {size}^3 at {vox} um' if e.get('voxel_zyx') else ''}")
    return got


def _read_file(e: dict):
    """A small single file, cached: NIfTI, TIFF, HDF5 (dataset ``array_path``), MRC, NRRD; a DICOM series under a public
    S3 prefix (``format: dicom``); consecutive TIFF slices inside a remote zip (``format: zip_tiff``)."""
    import urllib.request
    cache = Path(os.environ.get("SENSE_CACHE", "/home/andrew/bridge_wmcov/shapetensor/sense_check/cache")); cache.mkdir(parents=True, exist_ok=True)
    fmt = e["format"].lower()
    if fmt == "dicom":
        return _read_dicom_prefix(e, cache)
    if fmt == "zip_tiff":
        return _read_zip_tiffs(e)
    f = cache / (e.get("filename") or Path(e["url"].split("?")[0]).name)
    if not f.exists():
        req = urllib.request.Request(e["url"], headers={"User-Agent": "shapetensor-sense-check"})
        with urllib.request.urlopen(req, timeout=600) as r:
            n = int(r.headers.get("Content-Length", 0) or 0)
            if n > int(e.get("max_mb", 800)) * 2 ** 20:
                print(f"  file too large ({n / 2 ** 20:.0f} MB)"); return None
            f.write_bytes(r.read())
    name = f.name.lower()
    if name.endswith((".bz2", ".x-bzip2")):                                                    # a compressed file: decompress once beside it
        import bz2
        g = f.with_name(e.get("inner_name") or f.name.rsplit(".", 1)[0])
        if not g.exists():
            g.write_bytes(bz2.decompress(f.read_bytes()))
        f, name = g, g.name.lower()
    v = _decode_file(f, name, e)
    if v is not None and e.get("channel_axis") is not None:
        v = np.take(v, int(e.get("channel", 0)), axis=int(e["channel_axis"]))
    return v


def _decode_file(f: Path, name: str, e: dict):
    if name.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        v = np.asanyarray(nib.load(str(f)).dataobj)
        while v.ndim > 3:
            v = v[..., 0]
        return np.ascontiguousarray(v.transpose(2, 1, 0))
    if name.endswith((".tif", ".tiff")):
        import tifffile
        v = tifffile.imread(str(f))
        while v.ndim > 3 and e.get("channel_axis") is None:
            v = v[0]
        return v
    if name.endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(f, "r") as h:
            v = h[e.get("array_path") or list(h.keys())[0]][...]
        while v.ndim > 3:
            v = v[0]
        return v
    if name.endswith((".nrrd", ".mha", ".mhd")):
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(f)))                                  # z, y, x
    if name.endswith((".mrc", ".map", ".rec")):
        import mrcfile
        with mrcfile.open(str(f), permissive=True) as m:
            return np.asarray(m.data)
    print(f"  unknown file type {name}"); return None


def _read_dicom_prefix(e: dict, cache: Path):
    """All .dcm objects under a public S3 prefix (``url`` = https://bucket.s3.amazonaws.com/?list-type=2&prefix=P/ or
    s3://bucket/P/), cached, read as one series (sorted by position) with SimpleITK; z, y, x."""
    import re
    import urllib.request
    import SimpleITK as sitk
    url = e["url"]
    if url.startswith("s3://"):
        b, _, pfx = url[5:].partition("/"); base = f"https://{b}.s3.amazonaws.com"
    else:
        base = url.split("?")[0].rstrip("/"); pfx = re.search(r"prefix=([^&]+)", url).group(1)
    d = cache / ("dicom_" + pfx.strip("/").replace("/", "_")); d.mkdir(parents=True, exist_ok=True)
    keys, token = [], None
    while True:
        q = f"{base}/?list-type=2&prefix={pfx}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        xml = urllib.request.urlopen(q, timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            break
        token = m.group(1)
    keys = [k for k in keys if k.endswith(".dcm")][: int(e.get("max_files", 600))]
    for k in keys:
        f = d / Path(k).name
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(f"{base}/{k}", timeout=120).read())
    r = sitk.ImageSeriesReader(); names = r.GetGDCMSeriesFileNames(str(d))
    r.SetFileNames(names); img = r.Execute()
    return sitk.GetArrayFromImage(img)


def _read_zip_tiffs(e: dict):
    """A remote zip's ``member`` (one TIFF stack, read alone), or ``n_slices`` consecutive TIFF slices from a remote zip (HTTP range reads of its members), stacked; ``start``:
    the first slice's index among the sorted members (default: the middle)."""
    import io
    import fsspec
    import tifffile
    import zipfile
    loc = None if e.get("member") else _cached_download(e)                                     # one member: read it alone
    z = zipfile.ZipFile(loc or fsspec.open(e["url"]).open())
    if e.get("member"):
        v = tifffile.imread(io.BytesIO(z.read(e["member"])))
        if e.get("channel_axis") is not None:
            v = np.take(v, int(e.get("channel", 0)), axis=int(e["channel_axis"]))
        while v.ndim > 3:
            v = v[0]
        return v
    names = sorted(n for n in z.namelist() if n.lower().endswith((".tif", ".tiff")) and "__MACOSX" not in n)
    if len(names) == 1:                                                                        # one multi-page TIFF
        v = tifffile.imread(z.open(names[0]))
        while v.ndim > 3:
            v = v[..., 0]
        return v
    n = int(e.get("n_slices", 96)); s0 = int(e.get("start", max(0, len(names) // 2 - n // 2)))
    sl = [tifffile.imread(z.open(nm)) for nm in names[s0:s0 + n]]
    v = np.stack(sl)
    while v.ndim > 3:
        v = v[..., 0]
    return v


def fetch(catalogue: str, out: str, crops: int, size: int, skip_done: bool = False):
    od = Path(out); od.mkdir(parents=True, exist_ok=True)
    cat = json.load(open(catalogue)); done = {}
    for e in cat:
        print(f"{e['name']}: {e.get('organism', '')} {e.get('tissue', '')} {e.get('modality', '')} {e.get('voxel_um', '')} um", flush=True)
        if skip_done and (od / f"{e['name']}_0.npz").exists():
            done[e["name"]] = len(list(od.glob(f"{e['name']}_[0-9]*.npz"))); print("  already fetched"); continue
        try:
            done[e["name"]] = fetch_entry(e, od, int(e.get("crops", crops)), size)
        except Exception as ex:                                                                # one source never stops the rest
            print(f"  failed: {ex!r}"[:300], flush=True); done[e["name"]] = 0
    json.dump(done, open(od / f"fetched_{Path(catalogue).stem}.json", "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------------------------------------------

def load_set(d: str):
    items = []
    for f in sorted(Path(d).glob("*.npz")):
        z = np.load(f, allow_pickle=False); items.append((json.loads(str(z["meta"])), z["image"]))
    return items


def synthetic_pool(n: int, size: int = 64):
    from shapetensor.learn.augment import contrast_route
    from shapetensor.learn.lobe_train import difficulty_gen
    from shapetensor.learn.scene import draw_scene
    gen = difficulty_gen(1.0, {"canvas": 2 * size, "crops_per_scene": 4}); out = []
    for i in range(n):
        rng = np.random.default_rng(64_000_000 + i)
        _, d = draw_scene((size,) * 3, seed=64_000_000 + i, device=DEV, strength=float(rng.uniform(0.2, 1.0)), **gen)
        x = d["image"].float() / 255
        out.append(contrast_route(x, rng, 1.0) if rng.random() < 0.5 else x)
    return out


def analyse(data: str, models: list, out: str, n_syn: int = 384):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import domain_gap as DG
    from shapetensor.learn.realdata import normalise
    DG.DEV = DEV
    od = Path(out); od.mkdir(parents=True, exist_ok=True)
    items = load_set(data)
    N = lambda x: normalise(torch.as_tensor(x, dtype=torch.float32, device=DEV))[0]
    c0 = lambda x: x[16:80, 16:80, 16:80] if x.shape[0] >= 96 else x                              # the statistics at 64^3, as for the synthetic pool
    R = np.stack([DG.stats(N(c0(x))) for _, x in items])
    S = np.stack([DG.stats(N(x)) for x in synthetic_pool(n_syn)])
    mu, sd = S.mean(0), S.std(0) + 1e-6
    Ss = (S - mu) / sd; C = np.cov(Ss, rowvar=False); C = 0.8 * C + 0.2 * np.eye(C.shape[0])          # shrunk
    Ci = np.linalg.inv(C)
    maha = lambda v: float(np.sqrt(((v - mu) / sd) @ Ci @ ((v - mu) / sd)))
    syn_maha = np.array([maha(v) for v in S]); p95 = float(np.quantile(syn_maha, 0.95))
    W = np.linalg.cholesky(Ci); Sw = Ss @ W                                                       # whitened by the shrunk covariance
    dss = np.linalg.norm(Sw[:, None] - Sw[None], axis=-1); np.fill_diagonal(dss, np.inf)
    nn_syn = float(np.median(dss.min(1)))                                                         # a synthetic crop's typical nearest synthetic neighbour
    knn = lambda v: float(np.linalg.norm(((v - mu) / sd) @ W - Sw, axis=-1).min() / nn_syn)
    rows = {}
    for (meta, _), v in zip(items, R):
        key = meta["name"]; z = (v - mu) / sd
        r = rows.setdefault(key, {"meta": {k: meta.get(k) for k in ("organism", "tissue", "modality", "voxel_um")}, "novelty": [], "nn": [], "out_of_range": {}})
        r["novelty"].append(maha(v) / p95)                                                          # > 1: beyond 95 % of the synthetic crops
        r["nn"].append(knn(v))                                                                      # > 1: farther from any synthetic crop than they are from each other
        for i in np.nonzero(np.abs(z) > 3)[0]:
            r["out_of_range"][DG.NAMES[i]] = r["out_of_range"].get(DG.NAMES[i], 0) + 1
    for r in rows.values():
        r["novelty"] = float(np.mean(r["novelty"])); r["nn"] = float(np.mean(r["nn"]))
    order = sorted(rows, key=lambda k: -rows[k]["nn"])
    print(f"{'source':38s} {'modality':16s} {'voxel um':>9s} {'novelty':>8s} {'nn':>6s}  statistics the generator never reaches")
    for k in order:
        r = rows[k]; m = r["meta"]
        print(f"{k[:38]:38s} {str(m['modality'])[:16]:16s} {str(m['voxel_um'])[:9]:>9s} {r['novelty']:8.2f} {r['nn']:6.2f}  {', '.join(sorted(r['out_of_range'], key=lambda s: -r['out_of_range'][s])[:6])}")
    json.dump({"rows": rows, "synthetic_p95_mahalanobis": p95, "synthetic_nn_median": nn_syn}, open(od / "coverage.json", "w"), indent=1)
    if models:
        montage(items, models, od)


def _top(m, x, vu):
    """A model's top calibrated direction per cell ``(3, ...)``, its presence and the SH field, for ``x (Z, Y, X)``."""
    from shapetensor.learn.lobe_train import predict_sh, sh_peaks
    try:
        v = torch.tensor([math.log2(float(vu))], device=DEV) if vu else None
    except (TypeError, ValueError):
        v = None
    f = predict_sh(m, x[None, None], voxel=v)[0][0]
    D, V = sh_peaks(f, 8 if f.shape[0] >= 45 else 4); cs = f.shape[1:]
    return D[0].reshape(3, *cs), (V[0] > 0).reshape(cs), f


def _agree(d1, ok1, d2, ok2):
    """Mean |cos| between two direction fields where both have a peak (nan when they never do together)."""
    both = ok1 & ok2
    return float((d1 * d2).sum(0).abs()[both].mean()) if bool(both.any()) else float("nan")


@torch.no_grad()
def model_checks(m, xt: torch.Tensor, vu, coh, sd) -> dict:
    """Label-free checks of one model on one crop (0-1): the share of cells with a peak; agreement of the top direction
    with the structure tensor's least-gradient direction where the tissue is coherent (coherence > 0.3); the same
    direction after a contrast flip (1 - x)^0.7, a non-monotone intensity curve, and added noise (sigma 0.08)."""
    from shapetensor.learn.augment import apply_lut, random_lut
    d0, ok0, f0 = _top(m, xt, vu)
    out = {"peak_share": float(ok0.float().mean())}
    if sd.shape[1:] == d0.shape[1:]:
        c = coh > 0.3
        out["st_agree"] = _agree(d0, ok0 & c, sd, c)
    g = torch.Generator(device=xt.device).manual_seed(0)
    views = {"flip": (1 - xt).clamp(0, 1) ** 0.7, "curve": apply_lut(xt, *random_lut(np.random.default_rng(3), False)),
             "noise": (xt + 0.08 * torch.randn(xt.shape, generator=g, device=xt.device)).clamp(0, 1)}
    for k, y in views.items():
        d1, ok1, _ = _top(m, y.float(), vu)
        out[f"{k}_agree"] = _agree(d0, ok0, d1, ok1)
    # rotation: z and x swapped -- the direction field must follow (cells transposed, components dz <-> dx)
    d1, ok1, _ = _top(m, xt.transpose(0, 2).contiguous(), vu)
    out["rot_agree"] = _agree(d0, ok0, d1.flip(0).transpose(1, 3), ok1.transpose(0, 2))
    out["aniso"] = float(((f0[1:].norm(dim=0) / f0.norm(dim=0).clamp_min(1e-9)).clamp(0, 1)).mean())
    return out


def montage(items, models, od: Path):
    """Per category: each source's first crop, its structure tensor and each model's top direction (colour, scaled by
    the anisotropy), one page per category; and the label-free checks over all crops (``model_checks.json``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from real_validation import load_model, st_field
    from sense_check_montage import CATEGORIES, category
    from shapetensor.learn.realdata import normalise
    ms = {Path(p).parent.name: load_model(p) for p in models}
    checks, first = {}, {}
    for meta, x in items:
        n = meta["name"]; xt, _ = normalise(torch.as_tensor(x, dtype=torch.float32, device=DEV))
        coh, sd = st_field(xt)
        for mn, m in ms.items():
            r = model_checks(m, xt, meta.get("voxel_um"), coh, sd)
            for k, v in r.items():
                checks.setdefault(mn, {}).setdefault(n, {}).setdefault(k, []).append(v)
        first.setdefault(n, (meta, xt, coh, sd))
    for mn in checks:
        for n in checks[mn]:
            checks[mn][n] = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in checks[mn][n].items()}
    json.dump(checks, open(od / "model_checks.json", "w"), indent=1)
    keys = ("peak_share", "aniso", "st_agree", "flip_agree", "curve_agree", "noise_agree", "rot_agree")
    for mn in checks:
        print(f"\n{mn}: {'source':36s} " + " ".join(f"{k:>11s}" for k in keys))
        for n, r in sorted(checks[mn].items(), key=lambda t: t[1].get("flip_agree", 0)):
            print(f"{'':{len(mn) + 2}s}{n[:36]:36s} " + " ".join(f"{r.get(k, float('nan')):11.2f}" for k in keys))
    cols = 2 + len(ms)
    for cat in CATEGORIES:
        names = [n for n in first if category(first[n][0]) == cat]
        if not names:
            continue
        fig, ax = plt.subplots(len(names), cols, figsize=(2.6 * cols, 2.7 * len(names)), squeeze=False)
        for i, n in enumerate(names):
            meta, xt, coh, sd = first[n]; mid = xt.shape[0] // 2
            ax[i, 0].imshow(xt[mid].cpu().numpy(), cmap="gray"); ax[i, 0].set_title(f"{n[:28]}\n{str(meta.get('modality', ''))[:20]} {meta.get('voxel_um', '')} um", fontsize=6)
            rgb = (sd.abs() * coh[None].clamp(0, 1)).permute(1, 2, 3, 0)[sd.shape[1] // 2].cpu().numpy()
            ax[i, 1].imshow(np.kron(rgb, np.ones((4, 4, 1))).clip(0, 1)); ax[i, 1].set_title("structure tensor", fontsize=6)
            for j, (mn, m) in enumerate(ms.items()):
                d, ok, f = _top(m, xt, meta.get("voxel_um")); cs = f.shape[1:]
                strength = (f[1:].norm(dim=0) / f.norm(dim=0).clamp_min(1e-9)).clamp(0, 1)
                rgb = (d.abs() * strength[None] * ok[None]).permute(1, 2, 3, 0)[cs[0] // 2].cpu().numpy()
                r = checks[mn][n]
                ax[i, 2 + j].imshow(np.kron(rgb, np.ones((4, 4, 1))).clip(0, 1))
                ax[i, 2 + j].set_title(f"{mn[:18]}\npeaks {r['peak_share']:.2f} st {r.get('st_agree', float('nan')):.2f}\nflip {r['flip_agree']:.2f} rot {r['rot_agree']:.2f}", fontsize=6)
            for a_ in ax[i]:
                a_.axis("off")
        plt.tight_layout(); fig.savefig(od / f"montage_{cat.replace(' ', '_')}.png", dpi=55); plt.close(fig)
    print("wrote", od / "montage_*.png")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--catalogue", required=True); f.add_argument("--out", required=True)
    f.add_argument("--crops", type=int, default=4); f.add_argument("--size", type=int, default=96)
    f.add_argument("--skip-done", action="store_true", help="skip sources with crops already in --out")
    a_ = sub.add_parser("analyse"); a_.add_argument("--data", required=True); a_.add_argument("--models", nargs="*", default=[]); a_.add_argument("--out", required=True)
    a_.add_argument("--n-syn", type=int, default=384)
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.catalogue, a.out, a.crops, a.size, a.skip_done)
    else:
        analyse(a.data, a.models, a.out, a.n_syn)


if __name__ == "__main__":
    main()
