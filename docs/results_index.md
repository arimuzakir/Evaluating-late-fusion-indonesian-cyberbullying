# Result and source index

| Analysis | Direct artifact | How it is assessed |
|:--|:--|:--|
| Master cohort, versioned labels (`FROZEN`) | `data/labels/v1_1_GoldAdjudicated/POPULATION_COUNTS.csv` and `annotation/LABEL_LINEAGE_ALL_13307.csv` | `python -m src.evaluate_release --root .` |
| Five human votes (`FROZEN`) | `data/annotation_sources/anotasi_mentah_5_anotator.csv` | Vote agreement and consistency checks |
| Comment benchmark (`FROZEN`) | `results/comments/benchmark_v1.1/TEXT_TEST_PREDICTIONS.csv`, `TEXT_TEST_METRICS.csv` | Held-out row-level metrics, 1,226 rows per model |
| Aligned video model comparisons (`FROZEN`) | `results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv`, `VIDEO_FOLD_METRICS.csv` | 21 models × 3 seeds × 3,951 windows |
| Encoder fitting and test roles (`FROZEN` + `V4`) | `results/video/gold_adjudicated_v1.1/ROLE_MANIFEST.csv`, `results/video/revision_20261108/roles/ROLE_COUNTS.csv`, `MODEL_ROLE_RULES.csv` | Group/role isolation and exact 5-role fold counts |
| No-text fallback (`V4`) | `results/statistics/revision_20261108/fallback/NO_TEXT_FALLBACK.csv`, `TEXT_FALLBACK_DECISIONS.csv`, `CALIBRATION_CONTROL_FALLBACK.csv` | Exact confusion counts on 685 no-OCR/no-STT windows |
| Three-seed video stability (`FROZEN`) | `results/video/gold_adjudicated_v1.1/THREE_SEED_SUMMARY.csv` | Means and sample SD independently recomputed |
| Neural-head 2×3×2 sensitivity & runtime reconciliation (`V4`) | `results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_POOLED.csv`, `HEAD_BASELINE_RECONCILIATION.csv` | 48 seed-42 runs recomputed from `HEAD_SENSITIVITY_PREDICTIONS.csv` |
| Focal paired inference & retrospective MDE (`FROZEN` + `V4`) | `results/statistics/FOCAL_FOUR_100000.csv`, `docs/provenance/revision_20261108/RETROSPECTIVE_MDE.json` | 100,000 clustered bootstrap/permutation repeats and retrospective MDE (`0.02105`) |
| Coverage, 450 exclusions and source characteristics (`FROZEN` + `V4`) | `results/statistics/EXCLUSION_BY_PLATFORM.csv`, `results/statistics/revision_20261108/exclusions/EXCLUSION_REASON_PLATFORM_CLASS.csv`, `EXCLUSION_SOURCE_CHARACTERISTICS.csv` | Recomputed directly from `MANIFEST_VIDEO_4401.csv` and `ROLE_MANIFEST.csv` |
| Learned-fusion & seed-42 probability controls (`FROZEN` + `V4`) | `results/learned_fusion/TEST_COMPARISON_SEED42.csv`, `results/statistics/revision_20261108/probability/CALIBRATION_CONTROL_TEST_PREDICTIONS.csv`, `TABLE6_PROBABILITY_METRICS.csv` | Window-weighted (primary) vs source-weighted (sensitivity) metrics and Brier intervals (seed 42 only for `sigmoid_equal_vat` and `convex_vat`) |
| Input fidelity, SRSWOR audit design, and finite-population bounds (`FROZEN` + `V4`) | `results/input_audit/`, `results/input_audit/revision_20261108/AUDIT_SAMPLING_REPLAY.json`, `AUDIT_FIDELITY_DESIGN_BOUNDS.csv`, `AUDIT_ERROR_TRANSITIONS.csv` | Blinded 100-window SRSWOR replay, 46/9/0 keyframe check, Bonferroni-hypergeometric bounds, `NA` subgroup CIs |
| Two main-pipeline source allocations & frozen-feature probes (`FROZEN` + `V4`) | `results/video/revision_20261108/source_partitions/REPARTITION_EFFECTS.csv`, `REPARTITION_PREDICTIONS.csv`, `results/generalization/REPEATED_SPLIT_SUMMARY.csv`, `LOPO_CLUSTER_BOOTSTRAP_CI.csv` | Full main-pipeline partitions `1042` and `2042` (effective outer seed `2044`), plus 5 frozen-feature probes and LOPO |
| Manuscript artifact manifest & revision ledger (`V4`) | `docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv`, `docs/provenance/revision_20261108/SOURCE_REVISION_LEDGER.csv` | `python scripts/validation/verify_revision.py --root .` |
| Numerical verification (`FROZEN` + `V4`) | `results/reproduction/REPRODUCTION_CHECKS.json`, `configs/revision_20261108/RESULTS_SOURCE_FREEZE.json` | 27 baseline checks + 381 source-freeze checks + 35 CPU V4 table recomputations |

The `results/figures/` directory contains **publication-formatted illustrations regenerated deterministically from the frozen v1.1 and V4 result tables** via `python scripts/analysis/make_figures.py`.
