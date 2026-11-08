# Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

Versioned scientific artefacts for two **distinct experimental tracks**: a standalone Indonesian comment-classification benchmark, and a controlled visual–acoustic–extracted-text late-fusion comparison on source-disjoint video windows. This repository is an empirical evaluation, **not** a claim of a newly invented fusion architecture or demonstrated multimodal superiority.

**Project release date:** 6 October 2026 (researcher-designated). **Analysis reference date:** 22 October 2026 (researcher-designated). **Baseline archive assembly:** 28 October 2026 (`v1.1.2`). **Additive reviewer revision assembly:** 8 November 2026 (`v1.2.0`, source freeze `INASS_RESULTS_FREEZE_20261108_V4`). Primary numerical sources (`v1.1-GoldAdjudicated`, 279 protected files) are preserved byte-for-byte; additive V4 sensitivity, calibration, audit, exclusion, neural-head, and two-partition main-pipeline results are documented in [CHANGELOG.md](CHANGELOG.md), [RELEASE_INFO.json](RELEASE_INFO.json), and [`docs/provenance/revision_20261108/SOURCE_REVISION_LEDGER.csv`](docs/provenance/revision_20261108/SOURCE_REVISION_LEDGER.csv). **Operational label:** content-level *bullying-indicative aggression* (in a standalone comment or a ten-second video window), not evidence that an interpersonal episode satisfies all criteria of cyberbullying.

## Main findings and revision sensitivity summary

| Dataset / experiment | Observations | Outcome |
|:--|--:|:--|
| Master label inventory | 13,307 | 8,906 comment records + 4,401 video windows; 7,252 positive and 6,055 negative labels |
| Selected standalone comments | 8,196 | Held-out test = 1,226; transformer probability ensemble Macro F1 = **0.9663** at seed 42 |
| Aligned video windows (primary `v1.1`) | 3,951 | 214 linked source groups, four source-disjoint outer folds, five mutually disjoint roles, 21 model configurations, three training seeds (`42, 2026, 2027`) |
| Video extracted text (primary) | 3,951 | Seed-42 window-weighted Macro F1 = **0.57034** (source-weighted **0.58777**); 3-seed mean **0.5591 ± 0.0111** |
| Video equal-weight VAT (primary) | 3,951 | Seed-42 window-weighted Macro F1 = **0.56698** (source-weighted **0.58290**); 3-seed mean **0.5590 ± 0.0084** |
| Focal paired contrast (primary) | 214 source groups | Window-weighted Δ(Equal VAT − extracted text) = **−0.003362**, 95% clustered CI **[−0.017877, 0.011590]**, permutation p **0.6608**; retrospective MDE (80% power, α=0.05) = **0.02105** |
| Main-pipeline source repartitions (V4) | 3,951 × 2 partitions | Training seed 42 under outer partition `1042` (effective outer seed `1042`): Δ = **+0.01530** (`[−0.00297, +0.03356]`); outer partition `2042` (effective outer seed `2044`): Δ = **−0.00534** (`[−0.02353, +0.01264]`) |
| Seed-42 probability controls (V4) | 3,951 (seed 42 only) | `sigmoid_equal_vat` Macro F1 = **0.56213** (Brier **0.23370**); `convex_vat` Macro F1 = **0.57845** (Brier **0.23676**) |
| Neural-head 2×3×2 sensitivity (V4) | 48 fits (seed 42) | `visual_linear` **0.4648–0.4910**, `audio_mlp` **0.4980–0.5177**, `feature_mlp` **0.5333–0.5608**, `healnet` **0.5327–0.5568** |

The comment and video results belong to **different populations and selection protocols**. Comparing their Macro F1 values is not a controlled ablation of modality choice. Neither the primary video evaluation nor the two additional main-pipeline source-group allocations establishes a reliable classification advantage or formal statistical equivalence from multimodal late fusion.

## Project structure

```text
.
├── .github/workflows/verify-release.yml     Continuous numerical & V4 revision checks
├── configs/                                 Recorded model, audio, environment, and V4 freeze manifests
├── data/
│   ├── annotation_sources/                  Five recorded binary human votes
│   ├── labels/v1_1_GoldAdjudicated/         Versioned labels, lineage and source-group manifests
│   ├── manifests/comments/                  Comment split and text-hash isolation
│   └── features/                            Frozen SigLIP 2 and waveform16 features
├── notebooks/                               Six read-only Jupyter notebooks for archived numerical analyses
├── src/
│   ├── evaluate_release.py                  Primary 27-check numerical verifier
│   └── reference_archived/                  Sanitized historical training/sensitivity/repartition reference code
├── scripts/
│   ├── preprocessing/                       Reference official API ingestion, de-identification and 10s/3s windowing
│   ├── validation/                          verify_release.py, verify_revision.py, reconcile_keyframes.py, build_checksums.py
│   └── analysis/                            reproduce_revision.py, make_figures.py, lopo_cluster_intervals.py, repeated_group_probe.py
├── results/
│   ├── comments/benchmark_v1.1/             Held-out comment predictions and scores
│   ├── video/
│   │   ├── gold_adjudicated_v1.1/           Primary 3-seed, 4-fold video OOF, roles, thresholds (FROZEN)
│   │   └── revision_20261108/               48-run 2x3x2 head sensitivity, 2 source partitions, 5-role rules
│   ├── statistics/
│   │   ├── *.csv, *.json                    Primary focal inference, exclusions, sensitivities (FROZEN)
│   │   └── revision_20261108/               No-text fallback, exclusions, probability/weighting controls
│   ├── input_audit/
│   │   ├── *.csv, *.json                    Blinded 100-window OCR/STT audit and 46/9/0 keyframe trace (FROZEN)
│   │   └── revision_20261108/               SRSWOR sampling replay, Bonferroni-hypergeometric bounds, error transitions
│   ├── learned_fusion/                      Primary calibration, coefficients and learned-fusion controls (FROZEN)
│   ├── generalization/                      Frozen-feature LOPO and 5-seed grouped-partition probes (FROZEN)
│   ├── figures/                             Publication figures regenerated from frozen & V4 result tables
│   └── reproduction/                        Independent saved-prediction verification results
├── docs/                                    Methods, provenance, artifact manifest, limitations and data index
└── CITATION.cff, LICENSE, requirements.txt, environment.yml
```

All paths are organised by **scientific function**. No manuscript drafts, editorial response letters, private coordinator workbooks, or original raw media are included in the public repository.

## Quick start and three-level reproduction boundaries

This repository distinguishes three levels of reproducibility (detailed in [`docs/methods_and_reproducibility.md`](docs/methods_and_reproducibility.md), [`docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv`](docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv), and [`src/reference_archived/README.md`](src/reference_archived/README.md)):

1. **Level 1 — CPU reproduction from archived predictions, roles, and audit records (fully executable from repo root)**:
   ```bash
   python -m pip install -r requirements.txt
   python scripts/validation/verify_release.py --root .
   python scripts/validation/verify_revision.py --root .
   python scripts/analysis/reproduce_revision.py --root . --output /tmp/reproduced_v4
   python scripts/analysis/make_figures.py --root .
   ```
   Optional full suite (including 20,000-replicate clustered LOPO intervals and 5-seed frozen-feature linear probes):
   ```bash
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   python scripts/validation/verify_release.py --root . --full
   ```
2. **Level 2 — Saved-feature / checkpoint inference**: Frozen `SigLIP 2` and `waveform16` probes run on CPU via `scripts/analysis/repeated_group_probe.py` and `scripts/analysis/lopo_cluster_intervals.py`. Fine-tuned text/audio/visual neural checkpoint inference requires unreleased model weights and unredacted OCR/STT text.
3. **Level 3 — Full end-to-end training from scratch**: Reference training, neural-sensitivity, and main-pipeline repartition logic is archived in [`src/reference_archived/model_and_fusion_reference.py`](src/reference_archived/model_and_fusion_reference.py) (`docs/provenance/revision_20261108/ARCHIVED_CODE_LINEAGE.json`), but cannot be executed from the public repository because raw social-media videos, audio streams, and unredacted text are withheld for privacy and platform-policy compliance.

## Explore with Jupyter notebooks

Start with [`notebooks/README.md`](notebooks/README.md). Six read-only `.ipynb` notebooks trace the corpus, comment benchmark, video error transitions, focal statistics and extraction-state results, input audit and calibration (including V4 finite-population bounds and seed-42 probability controls), and generalization diagnostics (including the two V4 main-pipeline source allocations).

## Locate the evidence

- [Manuscript table/figure-to-code artifact manifest](docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv)
- [Source revision ledger (`INITIAL_RUN` → `CPU_V4`)](docs/provenance/revision_20261108/SOURCE_REVISION_LEDGER.csv)
- [Comment predictions and scores](results/comments/benchmark_v1.1/TEXT_TEST_METRICS.csv)
- [Video per-window and per-seed predictions (frozen `v1.1`)](results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv)
- [Two additional main-pipeline source allocations (`1042` and `2042`/`2044`)](results/video/revision_20261108/source_partitions/REPARTITION_EFFECTS.csv)
- [Neural-head 2×3×2 sensitivity and runtime reconciliation](results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_POOLED.csv)
- [Seed-42 probability controls and window vs source weighting](results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv)
- [100-window SRSWOR audit replay and Bonferroni-hypergeometric bounds](results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_BOUNDS.csv)
- [450 alignment exclusions and source characteristics](results/statistics/revision_20261108/exclusions/EXCLUSION_REASON_PLATFORM_CLASS.csv)
- [Signed administrative data provenance and protection record (`010/DRPM/Univ-BD/IX/2026`)](docs/UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf)

## Important boundaries

1. **Redacted public numerical views**: Raw comments, original OCR/STT strings, identifying links, and freeform audit notes are replaced with `[WITHHELD]`. Labels, human votes, group assignments, and probability predictions remain intact (`docs/source_redaction_manifest.json`).
2. **Three-level reproduction boundary**: Public scripts verify and recompute all metrics, bootstrap/permutation intervals, finite-population bounds, reliability tables, and frozen-feature probes from archived numerical inputs, but do not retrain primary neural encoders from withheld raw media.
3. **Audit sampling and finite-population bounds vs external validation**: The 100-window blinded audit (`55` positive, `45` negative; SRSWOR seeds `26102026` and `26102027`) yields item-level keyframe scores **46/9/0** (scores 2/1/0; earlier unverified coordinator aggregate `44/8/3` documented separately) and observed extraction fidelity of `43/52` (`82.30%` weighted) for OCR and `51/64` (`78.53%` weighted) for STT. Conservative 95% Bonferroni-hypergeometric bounds (`[0.2334, 0.9946]` for OCR; `[0.2430, 0.9904]` for STT) bound fidelity across the **finite 3,951-window cohort** under worst-case unaudited completions and **must not** be interpreted as validation on new external sources. Subgroup error-transition confidence intervals in `AUDIT_ERROR_TRANSITIONS.csv` remain `NA` due to small audited subgroup sample sizes (`n = 9 to 51`), and the specific failure mechanism of multimodal fusion remains unidentified.
4. **Source-partition sensitivity scope**: In addition to the five frozen-feature linear probe partitions (`REPEATED_SPLIT_SUMMARY.csv`), two full main-pipeline outer source-group allocations (`1042` with effective outer seed `1042`, and `2042` with effective outer seed `2044`, both using training seed `42`) are archived in `results/video/revision_20261108/source_partitions/`. Cross-platform LOPO remains a frozen-feature probe.
5. **Seed-42 probability controls and estimand selection timing**: The two calibration controls (`sigmoid_equal_vat` and `convex_vat`) are available for **seed 42 only**. Window-weighted evaluation is designated as the primary reporting estimand and equal-source-weighted evaluation as a cluster-size sensitivity analysis; this estimand hierarchy was formalized during revision rather than preregistered.
6. **Exploratory provenance and administrative vs ethics documentation**: Historical exploratory development included inspection of earlier held-out runs (`docs/provenance/revision_20261108/PROVENANCE_TIMELINE.csv`), so retrospective inference must not be presented as prospective preregistration. The signed Universitas Bina Darma record (`010/DRPM/Univ-BD/IX/2026`, dated 23 September 2026) is an **administrative data provenance and protection record**, not an institutional ethics approval or ethics-exemption determination; formal institutional ethics determination remains an open item (`OPEN`).

## Citation

Use [`CITATION.cff`](CITATION.cff) for the research-software archive.
