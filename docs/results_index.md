# Result and source index

| Analysis | Direct artifact | How it is assessed |
|:--|:--|:--|
| Master cohort, versioned labels | `data/labels/v1_1_GoldAdjudicated/POPULATION_COUNTS.csv` and `annotation/LABEL_LINEAGE_ALL_13307.csv` | `python -m src.evaluate_release --root .` |
| Five human votes | `data/annotation_sources/anotasi_mentah_5_anotator.csv` | Vote agreement and consistency checks |
| Comment benchmark | `results/comments/benchmark_v1.1/TEXT_TEST_PREDICTIONS.csv` | Held-out row-level metrics, 1,226 rows per model |
| Aligned video model comparisons | `results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv` | 21 models × 3 seeds × 3,951 windows |
| Encoder fitting and test roles | `results/video/gold_adjudicated_v1.1/ROLE_MANIFEST.csv` | Group and role-isolation checks |
| Three-seed video stability | `results/video/gold_adjudicated_v1.1/THREE_SEED_SUMMARY.csv` | Means and sample SD independently recomputed |
| Focal paired inference | `results/statistics/FOCAL_FOUR_100000.csv` | 100,000 clustered bootstrap/permutation repeats |
| Coverage, exclusions and label sensitivity | `results/statistics/EXCLUSION_BY_PLATFORM.csv`, `LABEL_AND_PLATFORM_SENSITIVITY.csv` | Descriptive, conditional on valid rows |
| Learned-fusion diagnostics | `results/learned_fusion/TEST_COMPARISON_SEED42.csv`, `ROLE_SIZES.csv` | Scores, fitting-role diagnostics, calibration |
| Input fidelity and keyframe | `results/input_audit/` | Blinded 100-window audit, item-level keyframe summary and archived aggregate provenance |
| Transfer and frozen-feature partition probes | `results/generalization/` | LOPO 20,000 clustered draws, 5 partition-probe seeds |
| Numerical verification | `results/reproduction/REPRODUCTION_CHECKS.json` | 27 verification outcomes |

The figure directory contains **regenerated illustrations from the final saved result tables**. Previously superseded manuscript figure panels were not retained.
