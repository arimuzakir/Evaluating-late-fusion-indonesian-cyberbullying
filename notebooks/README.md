# Numerical reproduction notebooks

These notebooks provide a guided, read-only path from archived labels and saved predictions to the main tables and statistical diagnostics. Run from the repository root or the `notebooks/` folder using Python 3 and the dependencies in `requirements.txt`. They do not require the original videos or encoder checkpoints and **cannot** retrain the full primary pipeline.

1. `01_corpus_and_splits.ipynb` — population accounting and split roles.
2. `02_comment_benchmark.ipynb` — held-out comment scores and seed-42 comparisons.
3. `03_video_models_and_errors.ipynb` — 21 video models, confusion matrices and paired errors.
4. `04_statistics_and_coverage.ipynb` — 100,000-replicate saved inference and OCR/STT availability states.
5. `05_input_audit_and_calibration.ipynb` — audit fidelity, keyframes, Brier and ECE.
6. `06_frozen_feature_generalization.ipynb` — LOPO intervals and five source-group repartition probes on frozen features.

All notebooks contain **existing analyses only**. Published bootstrap or permutation distributions are not regenerated in these notebooks because their retained summary tables provide the reported inference. The repository validation scripts perform additional reconstruction checks. No original social-media text is displayed.
