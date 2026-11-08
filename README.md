# Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

Versioned scientific artefacts for two **distinct experimental tracks**: a standalone Indonesian comment-classification benchmark, and a controlled visual–acoustic–extracted-text late-fusion comparison on source-disjoint video windows. This repository is an empirical evaluation, **not** a claim of a newly invented fusion architecture or demonstrated multimodal superiority.

**Project release date:** 6 October 2026 (researcher-designated). **Analysis reference date:** 22 October 2026 (researcher-designated). **Baseline archive assembly:** 28 October 2026 (`v1.1.2`). **Additive reviewer revision assembly:** 8 November 2026 (`v1.2.0`, source freeze `INASS_RESULTS_FREEZE_20261108_V4`). Primary numerical sources (`v1.1-GoldAdjudicated`, 279 protected files) are preserved byte-for-byte; additive V4 sensitivity, calibration, audit, exclusion, neural-head, and two-partition main-pipeline results are documented in [CHANGELOG.md](CHANGELOG.md), [RELEASE_INFO.json](RELEASE_INFO.json), and [`docs/provenance/revision_20261108/SOURCE_REVISION_LEDGER.csv`](docs/provenance/revision_20261108/SOURCE_REVISION_LEDGER.csv). **Operational label:** content-level *bullying-indicative aggression* (in a standalone comment or a ten-second video window), not evidence that an interpersonal episode satisfies all criteria of cyberbullying.

## Main findings and revision sensitivity summary

| Dataset / experiment | Observations | Outcome |
|:--|--:|:--|
| Master label inventory | 13,307 | 8,906 comment records + 4,401 video windows; 7,252 positive and 6,055 negative labels |
| Selected standalone comments | 8,196 | Held-out test = 1,226 (`865` source groups); transformer probability ensemble Macro F1 = **0.966271** at seed 42 (`TEXT_TEST_METRICS.csv`) |
| Aligned video windows (primary `v1.1`) | 3,951 | 214 linked source groups, four source-disjoint outer folds, five mutually disjoint roles, 21 model configurations, three training seeds (`42, 2026, 2027`); 450 duplicate/content-match alignment exclusions (`EXCLUSION_REASON_PLATFORM_CLASS.csv`) |
| Video extracted text (`text`, primary) | 3,951 | Seed-42 pooled OOF Macro F1 = **0.5703396012843986** (window Brier **0.2829761456553238**, source Brier **0.2512264663532779**); 3-seed mean **0.559139 ± 0.011066** (`TABLE3_RECOMPUTED.csv`, `THREE_SEED_RECOMPUTED.csv`) |
| Video equal-weight VAT (`equal_vat`, primary) | 3,951 | Seed-42 pooled OOF Macro F1 = **0.5669772833605011** (window Brier **0.23616631966321168**, source Brier **0.20519627702685368**); 3-seed mean **0.558972 ± 0.008434** (`TABLE3_RECOMPUTED.csv`, `THREE_SEED_RECOMPUTED.csv`) |
| Focal paired contrast (`R1.7`, primary) | 214 source groups | Pooled OOF Δ(Equal VAT − extracted text) = **−0.0033623179238975**, 95% clustered CI **[−0.0178765858028778, +0.0115901332544028]**, permutation `p = 0.6607533924660753`, `p_holm_four = 0.6607533924660753`; unadjusted retrospective normal MDE (`SE = 0.0075124837897468`, 80% power, two-sided α=0.05) = **0.021046863536667928** (`FOCAL_FOUR_100000.csv`, `ROUNDING_AND_MDE.json`) |
| Focal source repartitions (`R2.5`, V4) | 3,951 × 2 partitions | Focal `text` vs `equal_vat` contrast at `training_seed = 42` (`REPARTITION_EFFECTS.csv`): partition `1042` (`effective_outer_seed = 1042`): `text_f1 = 0.5322441613114784`, `equal_vat_f1 = 0.547547707084874`, Δ = **+0.015303545773395544**, 95% CI **[−0.0013907197934134488, +0.032653003054556234]**; partition `2042` (`effective_outer_seed = 2044`): `text_f1 = 0.5469555003724585`, `equal_vat_f1 = 0.541611762016599`, Δ = **−0.005343738355859551**, 95% CI **[−0.02085725995059201, +0.01140696636205858]** |
| Seed-42 probability controls (`R2.6`, V4) | 3,951 (seed 42 only) | `sigmoid_equal_vat` (monotone branch sigmoids fitted on `meta_fit`, threshold on `calibration`): Macro F1 = **0.5019314267553432**, window Brier = **0.2491781368632344** (source Brier **0.214898353943736**); `convex_vat` (simplex weights fitted on `meta_fit`, threshold on `calibration`): Macro F1 = **0.5209103528213086**, window Brier = **0.2492549371758121** (source Brier **0.218641645807787**) (`TABLE6_PROBABILITY_METRICS.csv`) |
| Neural-head 2×3×2 sensitivity (`R1.3`, V4) | 12 configs / 48 fold fits (seed 42) | Two neural heads (`concat_mlp` and `healnet` / `healnet_adapted_2024`) across epochs `{15, 30, 60}` and weight decay `{0.01, 0.1}` across 4 folds (`HEAD_SENSITIVITY_POOLED.csv`): `concat_mlp` pooled Macro F1 = **0.4876617530360183–0.5054619174213288**; `healnet` pooled Macro F1 = **0.48995107844566277–0.5094701632707828** |

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
│   │   └── revision_20261108/               48-fit 2x3x2 head sensitivity, 2 source partitions, 5-role rules
│   ├── statistics/
│   │   ├── *.csv, *.json                    Primary focal inference, exclusions, sensitivities (FROZEN)
│   │   └── revision_20261108/               No-text fallback, exclusions, probability/weighting controls
│   ├── input_audit/
│   │   ├── *.csv, *.json                    Blinded 100-window OCR/STT audit and 46/9/0 keyframe trace (FROZEN)
│   │   └── revision_20261108/               20-stratum SRSWOR replay, Bonferroni-hypergeometric bounds, error transitions
│   ├── learned_fusion/                      Primary calibration, coefficients and learned-fusion controls (FROZEN)
│   ├── generalization/                      Frozen-feature LOPO and 5-seed grouped-partition probes (FROZEN)
│   ├── figures/                             Author schematic diagrams (Fig 1-2) and regenerated empirical figures
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
3. **Level 3 — Full end-to-end training from scratch**: Reference training, neural-sensitivity (`concat_mlp` and `healnet`), and main-pipeline focal repartition logic is archived in [`src/reference_archived/model_and_fusion_reference.py`](src/reference_archived/model_and_fusion_reference.py) (`docs/provenance/revision_20261108/ARCHIVED_CODE_LINEAGE.json`), but cannot be executed from the public repository because raw social-media videos, audio streams, and unredacted text are withheld for privacy and platform-policy compliance.

## Explore with Jupyter notebooks

Start with [`notebooks/README.md`](notebooks/README.md). Six read-only `.ipynb` notebooks trace the corpus, comment benchmark, video error transitions, focal statistics and extraction-state results, input audit and calibration (including V4 finite-population bounds and seed-42 probability controls), and generalization diagnostics (including the two V4 main-pipeline source allocations).

## Locate the evidence

- [Manuscript table/figure-to-code artifact manifest (`R1.6`)](docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv)
- [Source revision ledger (`INITIAL_RUN` → `CPU_V4`, `R2.2`)](docs/provenance/revision_20261108/SOURCE_REVISION_LEDGER.csv)
- [Comment predictions, scores, and multiplicity family (`R1.7`)](results/comments/benchmark_v1.1/TEXT_TEST_METRICS.csv)
- [Video per-window and per-seed predictions (frozen `v1.1`, `R1.8`)](results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv)
- [Two additional main-pipeline source allocations (`1042` and `2042`/`2044`, `R2.5`)](results/video/revision_20261108/source_partitions/REPARTITION_EFFECTS.csv)
- [Neural-head 2×3×2 sensitivity and runtime reconciliation (`R1.3`)](results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_POOLED.csv)
- [Seed-42 probability controls and window vs source weighting (`R2.6`)](results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv)
- [100-window 20-stratum SRSWOR audit replay and Bonferroni-hypergeometric bounds (`R1.5`)](results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_BOUNDS.csv)
- [450 alignment exclusions and source characteristics (`R2.4`)](results/statistics/revision_20261108/exclusions/EXCLUSION_REASON_PLATFORM_CLASS.csv)
- [Signed administrative data provenance and protection record (`010/DRPM/Univ-BD/IX/2026`, `R2.4`)](docs/UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf)

## Important boundaries

1. **Redacted public numerical views**: Raw comments, original OCR/STT strings, identifying links, and freeform audit notes are replaced with `[WITHHELD]`. Labels, human votes, group assignments, and probability predictions remain intact (`docs/source_redaction_manifest.json`).
2. **Three-level reproduction boundary (`R1.6`)**: Public scripts verify and recompute all metrics, bootstrap/permutation intervals, finite-population bounds, reliability tables, and frozen-feature probes from archived numerical inputs (`atol = 1e-12`), while `Fig_1.png` and `Fig_2.png` are author-exported Draw.io schematic diagrams verified against the underlying cohort and role tables, and primary neural encoders are not retrained from withheld raw media.
3. **Audit sampling and finite-population bounds vs external validation (`R1.5`, `R2.3`)**: The 100-window blinded audit (`68` source groups; `55` consensus-positive, `45` consensus-negative) was drawn by stratified SRSWOR across **20 `availability × platform` strata** (`4` OCR/STT availability states × `5` platforms; sample selection seed `26102026`, blinded audit presentation order seed `26102027`). It yields item-level keyframe scores **46/9/0** (scores 2/1/0; earlier unverified coordinator aggregate `44/8/3` documented separately) and observed extraction fidelity among available outputs (`2,035` OCR-available and `2,506` STT-available windows in the 3,951-window population) of **`43/52`** (design-weighted **`0.8263523263523264`**) for OCR and **`51/64`** (design-weighted **`0.7943872080743589`**) for STT. Conservative 95% Bonferroni-hypergeometric bounds (`[0.2334152334152334, 0.9945945945945946]` for OCR; `[0.2430167597765363, 0.9904229848363927]` for STT) bound fidelity across the **fixed finite 3,951-window cohort** and **must not** be interpreted as validation on new external sources. Subgroup error-transition confidence intervals in `AUDIT_ERROR_TRANSITIONS.csv` remain **`NA`**, and the causal failure mechanism of multimodal fusion remains unresolved.
4. **Source-partition sensitivity scope (`R2.5`)**: In addition to the five frozen-feature linear probe partitions (`REPEATED_SPLIT_SUMMARY.csv`), two complete 5-role main-pipeline source-group allocations (`1042` with effective outer seed `1042`, and `2042` with effective outer seed `2044`, both at `training_seed = 42`) evaluate the focal **`text` vs `equal_vat`** contrast in `results/video/revision_20261108/source_partitions/`. Cross-platform LOPO remains a frozen-feature probe.
5. **Seed-42 probability controls and estimand selection timing (`R2.6`)**: The two calibration controls (`sigmoid_equal_vat` and `convex_vat`) are available for **seed 42 only**. Window-weighted evaluation is designated as the primary reporting estimand and equal-source-weighted evaluation as a cluster-size sensitivity analysis (`PROBABILITY_ESTIMAND.json`); this estimand hierarchy was formalized during revision rather than preregistered.
6. **Exploratory provenance and administrative vs ethics documentation (`R2.2`, `R2.4`)**: Historical exploratory development included inspection of earlier held-out runs (`docs/provenance/revision_20261108/RESEARCH_CHRONOLOGY.csv`), so retrospective inference must not be presented as prospective preregistration. The signed Universitas Bina Darma record (`010/DRPM/Univ-BD/IX/2026`, dated 23 September 2026) is an **administrative data provenance and protection record**, not an institutional ethics approval or ethics-exemption determination; formal institutional ethics determination remains an open item (`OPEN`).

## Citation

Use [`CITATION.cff`](CITATION.cff) for the research-software archive.
