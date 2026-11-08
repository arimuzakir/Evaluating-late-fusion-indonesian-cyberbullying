# Figures

These illustrations are generated deterministically from the frozen **v1.1-GoldAdjudicated** and **V4 revision** result tables by `python scripts/analysis/make_figures.py`, without refitting any model.

- `comment_benchmark_macro_f1.png`: Standalone comment benchmark held-out Macro F1 (`n = 1,226`, seed 42).
- `video_three_seed_stability.png`: Aligned video-window Macro F1 means and SD across three training seeds (`42, 2026, 2027`).
- `video_confusion_matrices_seed42.png`: Paired seed-42 confusion matrices for Extracted Text and Equal VAT (`n = 3,951`).
- `focal_paired_difference_seed42.png`: Focal seed-42 paired Macro F1 contrast and 100,000-replicate clustered 95% bootstrap CI (`214` source groups).
- `revision_20261108/reliability_*.png`: Publication-formatted (`>10pt`, 300 DPI, borderless) window-weighted and source-weighted reliability curves for seed-42 primary and Table 6 probability-control models. Archived V4 PNGs in `results/statistics/revision_20261108/probability/` remain untouched.
