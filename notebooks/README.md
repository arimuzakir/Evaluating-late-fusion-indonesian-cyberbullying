# Numerical reproduction notebooks

These six read-only notebooks provide a guided path from the frozen `v1.1-GoldAdjudicated` labels, saved out-of-fold predictions, and additive V4 reviewer revision tables to the manuscript tables and statistical diagnostics. Run from the repository root or the `notebooks/` folder using Python 3 and the dependencies in `requirements.txt`. They do not require withheld videos or neural checkpoints and **do not** retrain primary models.

1. `01_corpus_and_splits.ipynb` — population accounting (`13,307` master records), 5-role split sizes (`ROLE_COUNTS.csv`), and 450 alignment exclusions (`EXCLUSION_REASON_PLATFORM_CLASS.csv`).
2. `02_comment_benchmark.ipynb` — held-out standalone comment scores (`n = 1,226`) and seed-42 comparisons.
3. `03_video_models_and_errors.ipynb` — 21 primary video models, confusion matrices, paired error transitions, and 48-run neural-head 2×3×2 sensitivity summary (`HEAD_SENSITIVITY_POOLED.csv`).
4. `04_statistics_and_coverage.ipynb` — 100,000-replicate focal clustered inference, retrospective MDE (`0.02105`), and OCR/STT availability states (`685` no-text fallback windows).
5. `05_input_audit_and_calibration.ipynb` — 100-window SRSWOR audit verification, `46/9/0` keyframes, conservative Bonferroni-hypergeometric finite-population bounds, and seed-42 probability controls (`sigmoid_equal_vat`, `convex_vat`, window vs source weighting).
6. `06_frozen_feature_generalization.ipynb` — LOPO clustered intervals, five frozen-feature linear probe partitions, and the two V4 full main-pipeline source-group allocations (`1042` and `2042` with effective outer seed `2044`).

All notebooks examine archived numerical tables only; no original social-media text is displayed. For full automated recalculation of all 27 baseline checks and 35 V4 revision tables, run `python scripts/validation/verify_release.py --root .` and `python scripts/validation/verify_revision.py --root .`.
