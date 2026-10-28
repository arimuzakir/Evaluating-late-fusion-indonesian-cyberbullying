# Numerical environment and frozen-feature verification

The v1.1.1 frozen-feature repeated-partition tables agree exactly with numerical recomputation using a single BLAS thread. Multi-threaded linear algebra in some environments changes the ranking of near-equal floating-point predictions and yielded tiny ROC-AUC/AP differences in three text-probe rows, up to ROC-AUC 0.0005965401 and AP 0.0001426912. It leaves model predictions equal within recorded numerical precision, confusion matrices, Macro F1, and primary focal effects unchanged.

The `scripts/analysis/repeated_group_probe.py` runner now calls `threadpoolctl.threadpool_limits(limits=1)` internally for deterministic reconstruction. The archived `REPEATED_SPLIT_METRICS.csv` and `REPEATED_SPLIT_SUMMARY.csv` **have not been changed**, and their source checksums match v1.1.1. This is a numerical reproducibility fix, not a new experimental run or a performance correction.

To verify: `python scripts/validation/verify_release.py --root . --full`.
