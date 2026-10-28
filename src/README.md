# Source utilities

`evaluate_release.py` independently recalculates stored metrics, label links, role separation and LOPO scores from the released numeric files. `annotation.py`, `audio.py` and `utils/` contain auxiliary preprocessing and analysis utilities. The original restricted raw-media GPU training pipeline is **not** part of this archive. Usage: `python -m src.evaluate_release --root .`.
