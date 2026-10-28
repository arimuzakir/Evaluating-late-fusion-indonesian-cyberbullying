# Changelog

## v1.1.2 — final numerical archive (assembled 2026-10-28)

- Retains the **original v1.1.1** GoldAdjudicated labels, predictions, primary statistics, and frozen-feature repartition result tables with **no numerical score changes**.
- Fixes cross-environment verification of supplementary frozen-feature ROC-AUC and average precision by enforcing a single BLAS thread in `repeated_group_probe.py`. Different multi-threaded BLAS execution previously generated small changes in three text-probe ranking metrics (up to 0.000597 ROC-AUC and 0.000143 AP), even though predictions, group roles, Macro F1 and main findings were unchanged. The archived results match deterministic one-thread recomputation.
- Adds six read-only Jupyter notebooks for population, text, video, inference, audit and transfer results.
- Records researcher-specified initial project release date **2026-10-06** and analysis reference date **2026-10-22**, distinguished from archive assembly date **2026-10-28**. These dates do not establish that calculations produced after 22 October existed earlier.
- Rebuilds the public SHA-256 inventory and reruns full numerical verification.

## v1.1.1 — previous archive

- Pseudonymised numerical view, source-disjoint predictions and inference records, and item-level keyframe reconciliation.
