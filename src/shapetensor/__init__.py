"""shapetensor: fibre orientation fields from tensor component analysis (CP decomposition) of the gradients of
multichannel shape features, with the intensity structure tensor as the special case of one channel.

Everything here is a PROTOTYPE under validation on phantoms (docs/TCA_FIELD_REVIEW.md, docs/TCA_FIELD_PLAN.md).

Pipeline:  image block -> features (engine) -> channel-resolved tensor K on a site grid -> read-outs -> field.
"""

__version__ = "0.1.0"

DATA_ROOT = "/home/andrew/bridge_wmcov/shapetensor"
