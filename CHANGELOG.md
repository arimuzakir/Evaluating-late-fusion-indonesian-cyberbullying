# Changelog

## v1.2.0 — additive V4 reviewer revision archive (assembled 2026-11-08)

- **Preserves all 279 primary numerical source files (`INASS_RESULTS_FREEZE_20261108_V4`) byte-for-byte** from `v1.1-GoldAdjudicated` (`v1.1.2`), including labels, five human votes, source groups, role manifests, out-of-fold predictions, thresholds, primary Table 1–6 metrics, 100,000-replicate focal clustered inference, LOPO intervals, and five frozen-feature partition probes.
- **Integrates 89 public V4 revision artifacts additively** across `results/video/revision_20261108/`, `results/input_audit/revision_20261108/`, `results/statistics/revision_20261108/`, and `docs/provenance/revision_20261108/`:
  - **Neural-head sensitivity (`R1.2`)**: 48-run 2×3×2 factorial sensitivity (`weight_decay ∈ {1e-4, 1e-3}`, `dropout ∈ {0.1, 0.2, 0.3}`, `head_lr ∈ {5e-4, 1e-3}`) at seed 42 across `visual_linear`, `audio_mlp`, `feature_mlp`, and `healnet`, plus runtime reconciliation documenting why primary Table 3 (`torch.set_float32_matmul_precision('high')`) is retained while sensitivity runs share a single consistent runtime.
  - **Two additional main-pipeline source-group allocations (`R2.5`)**: Full 4-fold, 5-role pipeline execution at training seed 42 under partition `1042` (`effective_outer_seed = 1042`, `Δ(Equal VAT − Text) = +0.01530`, 95% CI `[−0.00297, +0.03356]`) and partition `2042` (`search_start_seed = 2042`, `effective_outer_seed = 2044`, `Δ(Equal VAT − Text) = −0.00534`, 95% CI `[−0.02353, +0.01264]`), updating prior documentation that only frozen-feature linear probes had been repartitioned.
  - **Role counts, threshold protocol, and no-text fallback (`R1.3`)**: Explicit per-fold/per-seed 5-role window counts (`ROLE_COUNTS.csv`), 23-model role-usage rules (`MODEL_ROLE_RULES.csv`), and no-text fallback confusion matrices across the 685 aligned windows lacking both OCR and STT (`NO_TEXT_FALLBACK.csv`, `TEXT_FALLBACK_DECISIONS.csv`, `CALIBRATION_CONTROL_FALLBACK.csv`).
  - **Seed-42 probability controls and weighting sensitivity (`R1.4`, `R2.4`)**: Out-of-fold seed-42 predictions and clustered Brier/log-loss/ECE diagnostics for `sigmoid_equal_vat` and `convex_vat` (available for seed 42 only), alongside window-weighted (primary estimand) and equal-source-weighted (sensitivity estimand) reliability bins and Table 6 metrics, with explicit disclosure of the revision-time estimand reporting decision.
  - **100-window audit sampling verification and finite-population bounds (`R1.5`, `R2.6`)**: Executable SRSWOR sampling replay (`26102026` positive, `26102027` negative), conservative Bonferroni-hypergeometric finite-population bounds over the 3,951-window cohort (`OCR: [0.23342, 0.99459]`; `STT: [0.24302, 0.99042]`), model-assisted parametric bootstrap (`B = 2,000`), and subgroup error transitions with `NA` confidence intervals.
  - **450 alignment exclusions and retrospective MDE (`R1.6`, `R2.4`)**: Explicit exclusion reason × platform × label breakdown (`EXCLUSION_REASON_PLATFORM_CLASS.csv`), source-group retention profile (`EXCLUSION_SOURCE_CHARACTERISTICS.csv`), and retrospective minimum detectable effect (`SE = 0.0075125`, `MDE = 0.0210469` at 80% power, two-sided `α = 0.05`).
- **Adds CPU reproduction, archived code reference, and manuscript artifact mapping**:
  - `scripts/analysis/reproduce_revision.py` deterministically reconstructs all 35 CPU-derivable V4 tables and JSON summaries from public repository inputs.
  - `scripts/validation/verify_revision.py` enforces the 279-file source freeze, 89-file V4 integrity check, 35-table CPU recomputation check, and reviewer invariants `R1.1`–`R2.6`.
  - `src/reference_archived/` and `docs/provenance/revision_20261108/ARCHIVED_CODE_LINEAGE.json` provide path-sanitized historical training/sensitivity/repartition reference code with explicit three-level reproduction boundaries.
  - `docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv` links manuscript tables and figures to input files, SHA-256 hashes, scripts, parameters, and executable commands.

## v1.1.2 — final numerical archive (assembled 2026-10-28)

- Retains the **original v1.1.1** GoldAdjudicated labels, predictions, primary statistics, and frozen-feature repartition result tables with **no numerical score changes**.
- Fixes cross-environment verification of supplementary frozen-feature ROC-AUC and average precision by enforcing a single BLAS thread in `repeated_group_probe.py`.
- Adds six read-only Jupyter notebooks for population, text, video, inference, audit and transfer results.
- Records researcher-specified initial project release date **2026-10-06** and analysis reference date **2026-10-22**, distinguished from archive assembly date **2026-10-28**.
- Rebuilds the public SHA-256 inventory and reruns full numerical verification.

## v1.1.1 — previous archive

- Pseudonymised numerical view, source-disjoint predictions and inference records, and item-level keyframe reconciliation.
