"""sliceTCA denoising of the shape features (prototype; not in the docs).

After Pellegrino, Stein & Cayco-Gajic (Nature Neuroscience 2024): a tensor is approximated by a sum of *slice*
components, each a vector along one mode times a full matrix (the slice) over the other two. Kept at low slice rank,
the reconstruction holds the structure the data shares along each mode and drops what is idiosyncratic to single
entries -- their Fig. 4 denoises population recordings this way.

Here the tensor is built per window of the feature field: ``positions × channels × lags`` -- every position's
neighbourhood of the encoder-2 features, the lags a ``patch``³ grid of offsets ``spacing`` voxels apart. The three
component kinds are

* position-slicing: a spatial map × a channel-by-lag pattern (a local structure with its own shape signature),
* channel-slicing: a channel signature × a position-by-lag pattern (one signature whose spatial layout is free),
* lag-slicing: a lag pattern × a position-by-channel pattern (one neighbourhood shape, e.g. elongated along a fibre).

Each kind's term is a low-rank matrix in its mode's unfolding, so the fit is backfitting: each kind in turn is the
best rank-``r`` approximation (truncated SVD) of its unfolding of what the other kinds leave. The denoised feature of a
voxel is the average of its reconstructed values over every patch that covers it.

Parameters: ``ranks`` (r_position, r_channel, r_lag) -- how many components of each kind; ``spacing`` -- the scale:
the lag step in voxels (the neighbourhood spans ``(patch - 1) * spacing + 1`` voxels); ``window`` -- the region one
decomposition covers (locality); ``pos_stride`` -- the spacing of the patch centres (keep it coprime with ``spacing``
so every voxel is covered).
"""
# Promoted to shapetensor.smoothing (2026-10-20); the prototype names stay importable from here.
from ..smoothing import _lags, denoise_features, slice_tca  # noqa: F401
