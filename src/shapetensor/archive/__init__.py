"""Archived alternatives, kept importable but not part of the pipeline or its documentation:

* :mod:`.odf` / :mod:`.fixels` — several directions per site from constrained spherical deconvolution of the
  gradient-direction ODF (``fixels_3d``);
* :mod:`.tensor` / :mod:`.tca` — the channel-resolved tensor ``K = G_ρ ∗ (g gᵀ)`` of all feature-channel gradients
  and its tensor component analysis (CP decomposition) into populations;
* :mod:`.field` — the site-grid orientation-field store of that route.

They were the first forms of stages 3 and 4 and were set aside for the activation layers (see ``notes/``).
"""
