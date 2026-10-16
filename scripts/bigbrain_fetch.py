"""Fetch a block of the BigBrain 2015 release (neuroglancer precomputed, uint8, raw chunks of 64^3) over HTTP.
    python bb_fetch.py <scale_key> x0 x1 y0 y1 z0 z1 out.npy   (voxel ranges at that scale; (z, y, x) array written)
"""
import sys, io, gzip, json, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.error
import numpy as np

BASE = "https://neuroglancer.humanbrainproject.eu/precomputed/BigBrainRelease.2015/8bit"
INFO = json.loads(urllib.request.urlopen(BASE + "/info", timeout=60).read())
SCALES = {s["key"]: s for s in INFO["scales"]}

def fetch_chunk(key, x0, x1, y0, y1, z0, z1, tries=4):
    url = f"{BASE}/{key}/{x0}-{x1}_{y0}-{y1}_{z0}-{z1}"
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                buf = r.read()
            if buf[:2] == b"\x1f\x8b":
                buf = gzip.decompress(buf)
            a = np.frombuffer(buf, np.uint8)
            return a.reshape(z1 - z0, y1 - y0, x1 - x0)          # Fortran order on disk: x fastest -> (z, y, x) C-order
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        except Exception as e:
            pass
        time.sleep(2 * (t + 1))
    raise RuntimeError(f"failed {url}")

def fetch_block(key, x0, x1, y0, y1, z0, z1, workers=12):
    s = SCALES[key]; cs = s["chunk_sizes"][0]; sx, sy, sz = s["size"]
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    jobs = []
    for cz in range(z0 // cs[2] * cs[2], z1, cs[2]):
        for cy in range(y0 // cs[1] * cs[1], y1, cs[1]):
            for cx in range(x0 // cs[0] * cs[0], x1, cs[0]):
                jobs.append((cx, min(cx + cs[0], sx), cy, min(cy + cs[1], sy), cz, min(cz + cs[2], sz)))
    t0 = time.time()
    def work(j):
        cx0, cx1, cy0, cy1, cz0, cz1 = j
        a = fetch_chunk(key, cx0, cx1, cy0, cy1, cz0, cz1)
        ox0, ox1 = max(cx0, x0), min(cx1, x1); oy0, oy1 = max(cy0, y0), min(cy1, y1); oz0, oz1 = max(cz0, z0), min(cz1, z1)
        out[oz0 - z0:oz1 - z0, oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = a[oz0 - cz0:oz1 - cz0, oy0 - cy0:oy1 - cy0, ox0 - cx0:ox1 - cx0]
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(work, jobs)):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} chunks, {time.time() - t0:.0f} s", flush=True)
    print(f"fetched {len(jobs)} chunks of {key} in {time.time() - t0:.0f} s", flush=True)
    return out

if __name__ == "__main__":
    key = sys.argv[1]; x0, x1, y0, y1, z0, z1 = map(int, sys.argv[2:8]); out = sys.argv[8]
    a = fetch_block(key, x0, x1, y0, y1, z0, z1)
    np.save(out, a); print("wrote", out, a.shape, a.dtype, "min/median/max", a.min(), np.median(a), a.max())
