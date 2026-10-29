"""Collections of objects at a scale (entities._collections): objects alone at scale 0; one population only; nested
across scales (every collection at a scale lies inside one at any coarser scale); larger scales, fewer collections."""
import torch

from shapetensor.learn.entities import _collections


def _scene(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = torch.rand(n, 3, generator=g) * 128; grp = torch.randint(0, 3, (n,), generator=g)
    return torch.arange(n), grp, cent


def test_scale_zero_is_the_objects():
    leaf, grp, cent = _scene()
    out = _collections(leaf, grp, torch.zeros(len(leaf)), cent, torch.zeros(3), len(leaf))
    assert torch.equal(out, leaf)


def test_collections_nest_and_keep_populations_apart():
    leaf, grp, cent = _scene(); off = torch.tensor([17.0, 3.0, 40.0]); prev = None; counts = []
    for s in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0):
        c = _collections(leaf, grp, torch.full((len(leaf),), s), cent, off, len(leaf))
        for u in torch.unique(c):                                                       # one population per collection
            assert torch.unique(grp[c == u]).numel() == 1
        if prev is not None:                                                            # nested: a finer collection never splits
            for u in torch.unique(prev):
                assert torch.unique(c[prev == u]).numel() == 1
        counts.append(int(torch.unique(c).numel())); prev = c
    assert counts == sorted(counts, reverse=True) and counts[-1] <= 3 * 8                # coarser: fewer (at most 2^3 cells x 3 populations)
