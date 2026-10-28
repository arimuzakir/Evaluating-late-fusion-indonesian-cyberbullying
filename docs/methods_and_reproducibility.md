# Methods and reproducibility

## Cohorts and evaluation

The frozen v1.1 label inventory contains 13,307 observations, of which 8,906 are standalone text comments and 4,401 are ten-second video windows. The video alignment rule retains 3,951 windows from 214 linked source groups. The remaining 450 do not have aligned input views and no model accuracy is imputed for them. The selected high-agreement standalone-comment subset contains 8,196 cases, with 1,226 held out for testing. These tracks are independent evaluations and are not combined into a single pooled performance metric.

The video main experiment uses four outer folds, five mutually disjoint roles (`fit`, `early_stop`, `meta_fit`, `calibration`, `test`) within each outer fold, and training seeds 42, 2026 and 2027. Source-group roles and exact OOF prediction rows are archived under `results/video/gold_adjudicated_v1.1/`. Text has a neutral probability 0.5 when both OCR and STT are unavailable, including 685 aligned windows.

21 configurations include visual and acoustic branches, extracted text, arithmetic late-fusion averages, L2 logistic stacking, availability-aware controls, feature MLP and the local HEALNet adaptation, plus frozen SigLIP2 probes. A comparison to these local implementations does not establish superiority over original published architectures.

## Primary inference

The focal seed-42 comparison is Equal VAT minus extracted video text on exactly the same 3,951 held-out video windows, resampling/exchanging all rows in each of 214 source groups together. The 100,000-replicate bootstrap interval is [−0.0178766, 0.0115901] for Δ Macro F1 = −0.0033623, with paired permutation p = 0.6607534. This was a retrospectively selected four-contrast reporting family. Do not infer equivalence from this result. Exact comparisons and their Holm corrections are in `results/statistics/FOCAL_FOUR_100000.csv`.

## Diagnostic controls

Calibration and fusion-control parameters, meta-fit sizes and reliability bins are preserved in `results/learned_fusion/`. In the seed-42 aligned evaluation, Brier Score is 0.282976 for extracted text and 0.236166 for Equal VAT; these descriptive values do not imply better thresholded Macro F1. Paired error transitions show 308 text errors fixed and 324 newly introduced by Equal VAT. The 100-window independent extraction audit documents OCR 43/52 faithful and STT 51/64 faithful among available outputs, with pre-consensus aggression Cohen kappa 0.8998. These associations do not establish that extraction errors caused the observed fusion difference.

The 450 excluded video windows have a positive proportion of 280/450 and the retained windows 2,326/3,951. Poststratification only adjusts platform proportions within retained windows and cannot establish absence of selection bias.

## Numerical verification

`python -m src.evaluate_release --root .` checks label consistency, role separation, group-safe splits and independently recomputes 63 video metric rows, 21 three-seed summaries, seven comment result rows and 20 LOPO scores. `python scripts/validation/verify_release.py --root . --full` checks the full byte inventory and repeats stored auxiliary diagnostics.

The reference statistical and frozen-feature calculations are conditional on already fitted models. Without original media/encoder checkpoints, they do not imply independent end-to-end training reproduction.
