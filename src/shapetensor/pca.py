"""Per-volume PCA of a feature field, to compress any engine's channels to ``n_pcs`` before the tensor pass.

The basis is fitted on a random sample of voxels (the whole block, or a mask) and applied to every voxel. The
projection is linear, so it commutes with the grid-phase average and with the derivative filters: PCA of the
features then gradients equals PCA applied to the gradients.
"""
from __future__ import annotations

import torch


class PCA:
    """``mean (C,)`` and ``basis (n, C)`` (rows = components, descending variance), with ``explained (n,)``."""

    def __init__(self, mean: torch.Tensor, basis: torch.Tensor, explained: torch.Tensor):
        self.mean, self.basis, self.explained = mean, basis, explained

    @property
    def n_components(self) -> int:
        return int(self.basis.shape[0])

    @classmethod
    def fit(cls, F: torch.Tensor, n_components: int, n_sample: int = 200_000, mask: torch.Tensor | None = None,
            seed: int = 0) -> "PCA":
        """Fit on ``F (C, ...)``: a sample of ``n_sample`` voxels (from ``mask`` if given)."""
        C = F.shape[0]
        X = F.reshape(C, -1)
        if mask is not None:
            idx = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
        else:
            idx = torch.arange(X.shape[1], device=F.device)
        g = torch.Generator(device="cpu").manual_seed(seed)
        if idx.numel() > n_sample:
            sel = torch.randperm(idx.numel(), generator=g)[:n_sample].to(F.device)
            idx = idx[sel]
        S = X[:, idx].double()
        mean = S.mean(1)
        S = S - mean[:, None]
        cov = (S @ S.T) / max(1, S.shape[1] - 1)
        w, V = torch.linalg.eigh(cov)
        order = torch.argsort(w, descending=True)
        w, V = w[order].clamp_min(0), V[:, order]
        n = min(int(n_components), C)
        basis = V[:, :n].T
        # a sign convention, so bases are reproducible across fits
        s = torch.sign(basis[torch.arange(n), basis.abs().argmax(1)])
        basis = basis * s[:, None]
        return cls(mean.float(), basis.float(), (w / w.sum()).float()[:n])

    def __call__(self, F: torch.Tensor, chunk: int = 1 << 24) -> torch.Tensor:
        """Project ``F (C, ...)`` -> ``(n, ...)`` in ``F``'s dtype, computed in float32 in chunks of ``chunk``
        voxels so a large field is never held whole in float32."""
        C = F.shape[0]
        X = F.reshape(C, -1)
        out = torch.empty((self.n_components, X.shape[1]), dtype=F.dtype, device=F.device)
        for a in range(0, X.shape[1], chunk):
            part = X[:, a:a + chunk].float() - self.mean[:, None]
            out[:, a:a + chunk] = (self.basis @ part).to(F.dtype)
        return out.reshape((self.n_components,) + tuple(F.shape[1:]))

    def state(self) -> dict:
        return {"mean": self.mean.cpu().numpy(), "basis": self.basis.cpu().numpy(), "explained": self.explained.cpu().numpy()}
