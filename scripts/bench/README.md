# Bench scripts of the 2026-10-16 validation session

Driver and analysis scripts used with the `shapetensor` package (see `docs/TCA_FIELD_PLAN.md` / `TCA_FIELD_RESULTS.md`).
They were run from a session scratchpad; paths to the data root (`/home/andrew/bridge_wmcov/shapetensor/`) are absolute.

- engine numerics: `anatomix_cost.py`, `tile_margin.py`, `tile_tf32.py`, `readout_sens.py`
- phantom calibration to the real crop: `real_stats.py`, `real_cl.py`, `calib_texture*.py`, `calib_dispersion.py` -> `tex_calibrated.json`
- suite drivers: `run_pilot.py`, `run_variant.py`, `run_full.py SPEC.json [ENGINE_KWARGS.json]`, specs `spec_*.json`, `after_suite.sh`
- summaries: `compare_tags.py`, `grid_summary.py`; CSD threshold calibration `calib_csd.py` -> `csd_calibrated.json`
- orientation inside anatomix: `channel_selectivity.py`, `encoder_tuning.py`, `encoder_head.py`, `per_pop_channels.py`
- `logs/`: the calibration and test logs quoted in the docs
