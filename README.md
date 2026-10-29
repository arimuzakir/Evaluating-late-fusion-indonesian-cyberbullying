# Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

Versioned scientific artefacts for two **distinct experimental tracks**: a standalone Indonesian comment-classification benchmark, and a controlled visual–acoustic–extracted-text late-fusion comparison on source-disjoint video windows. This repository is an empirical evaluation, **not** a claim of a newly invented fusion architecture or demonstrated multimodal superiority.

**Project release date:** 6 October 2026 (researcher-designated). **Analysis reference date:** 22 October 2026 (researcher-designated). **Final archive assembly:** 28 October 2026 (v1.1.2). These dates serve different purposes. Analyses finalized after 22 October and the numerical-environment reproducibility fix are documented in [CHANGELOG.md](CHANGELOG.md) and [RELEASE_INFO.json](RELEASE_INFO.json). This archive does not represent that every published calculation existed on 22 October. **Operational label:** content-level *bullying-indicative aggression* (in a standalone comment or a ten-second video window), not evidence that the entire interpersonal episode satisfies all criteria of cyberbullying.

## Main findings

| Dataset / experiment | Observations | Outcome |
|:--|--:|:--|
| Master label inventory | 13,307 | 8,906 comment records + 4,401 video windows; 7,252 positive and 6,055 negative labels |
| Selected standalone comments | 8,196 | Held-out test = 1,226; transformer probability ensemble Macro F1 = **0.9663** at seed 42 |
| Aligned video windows | 3,951 | 214 linked source groups, four source-disjoint outer folds, 21 model configurations, three training seeds |
| Video extracted text | 3,951 | Seed-42 Macro F1 = **0.57034**; mean across three seeds **0.5591 ± 0.0111** |
| Video equal-weight VAT | 3,951 | Seed-42 Macro F1 = **0.56698**; mean across three seeds **0.5590 ± 0.0084** |
| Focal paired contrast | 214 source groups | Δ(Equal VAT − extracted text) = **−0.003362**, 95% clustered CI **[−0.017877, 0.011590]**, permutation p **0.6608** |

The comment and video results belong to **different populations and selection protocols**. Comparing their Macro F1 values is not a controlled ablation of modality choice. The video contrast does not establish a classification advantage from late fusion or equivalence between the methods.

## Project structure

```text
.
├── .github/workflows/verify-release.yml     Continuous numerical checks
├── configs/                                 Recorded model, audio, and environment settings
├── data/
│   ├── annotation_sources/                  Five recorded binary human votes
│   ├── labels/v1_1_GoldAdjudicated/         Versioned labels, lineage and source-group manifests
│   ├── manifests/comments/                  Comment split and text-hash isolation
│   └── features/                            Frozen SigLIP 2 and waveform16 features
├── notebooks/                               Six Jupyter notebooks for archived numerical analyses
├── src/                                     Numerical metric and source-consistency checks
├── scripts/
│   ├── preprocessing/                       Reference official API ingestion, de-identification and 10s/3s windowing specification
│   ├── validation/                          Hash, annotation and saved-result verification
│   └── analysis/                            LOPO intervals and frozen-feature partition probes
├── results/
│   ├── comments/benchmark_v1.1/             Held-out comment predictions and scores
│   ├── video/gold_adjudicated_v1.1/         Three-seed, 4-fold video OOF, roles, thresholds
│   ├── statistics/                           Focal inference, data exclusions, sensitivities
│   ├── input_audit/                          Blinded OCR/STT assessment and keyframe trace
│   ├── learned_fusion/                      Calibration, coefficients and learned-fusion controls
│   ├── generalization/                      Frozen-feature LOPO and grouped-partition probes
│   ├── figures/                             Figures regenerated from final result tables
│   └── reproduction/                        Independent saved-prediction verification results
├── docs/                                    Methods, provenance, limitations and data index
└── CITATION.cff, LICENSE, requirements.txt, environment.yml
```

All paths are organised by **scientific function**. No manuscript drafts, editorial responses, historical superseded score tables or original raw media are part of this release.

Before releasing, review [data governance](docs/data_release.md). This numerical archive is not a grant of redistribution rights over source content.

## Quick start

Python 3.11+ with dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python scripts/validation/verify_release.py --root .
```

This checks the versioned SHA-256 inventory, recomputes **27 saved-data/prediction checks**, and independently reconstructs the keyframe distribution from recorded item-level ratings and adjudication entries. Optional larger numerical diagnostics:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/validation/verify_release.py --root . --full
```

Individual routines:

```bash
python -m src.evaluate_release --root .
python scripts/validation/reconcile_keyframes.py --root .
python scripts/analysis/lopo_cluster_intervals.py --root .
python scripts/analysis/repeated_group_probe.py --root .
```

`--full` also reproduces the previously completed 20,000-replicate clustered LOPO confidence intervals and five grouped-partition *frozen-feature* probes. These are **not** retrained end-to-end primary models. GitHub Actions runs basic and full checks on push and pull requests.

## Explore with Jupyter notebooks

Start with [`notebooks/README.md`](notebooks/README.md). The six `.ipynb` notebooks trace the corpus, comment benchmark, video error transitions, focal statistics and extraction-state results, input audit and calibration, and frozen-feature generalization. They use **archived numerical data only** and do not rerun full model training or invent new experiments.

```bash
python -m pip install -r requirements.txt
python -m pip install jupyterlab ipykernel
jupyter lab
```

## Locate the evidence

- [Comment predictions and scores](results/comments/benchmark_v1.1/TEXT_TEST_METRICS.csv)
- [Video per-window and per-seed predictions](results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv)
- [Source-disjoint role assignments](results/video/gold_adjudicated_v1.1/ROLE_MANIFEST.csv)
- [Three-seed video metrics](results/video/gold_adjudicated_v1.1/THREE_SEED_SUMMARY.csv)
- [Focal grouped inference, 100,000 repetitions](results/statistics/FOCAL_FOUR_100000.csv)
- [Learned-fusion probability and calibration diagnostics](results/learned_fusion/TEST_COMPARISON_SEED42.csv)
- [OCR and speech extraction audit](results/input_audit/AUDIT_OCR_STT_SUMMARY.json)
- [Frozen-feature cross-platform and partition diagnostics](results/generalization/REPEATED_SPLIT_SUMMARY.csv)
- [Input and experiment provenance](docs/methods_and_reproducibility.md)
- [Data access and privacy scope](docs/data_release.md)
- [Signed institutional data provenance and protection record (`010/DRPM/Univ-BD/IX/2026`)](docs/UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf)
- [Study limitations](docs/limitations.md)

## Important boundaries

1. The published CSVs are **public redacted numerical views**. Raw comments, original OCR/STT strings, potentially identifying links and freeform audit notes have been withheld. Labels, rater scores, group assignments and probability predictions remain intact. SHA-256 of original source versions and public copies are recorded separately in [`docs/source_redaction_manifest.json`](docs/source_redaction_manifest.json). Thus, the 27 numeric checks operate on the redacted public copies and validate their retained numeric fields, but do not independently certify the withheld original text.
2. End-to-end retraining requires restricted original video/audio media, original text and model checkpoints. The public release supports saved-prediction reproducibility and frozen-feature probe recomputation, **not** original full training reproduction.
3. The 100-window blinded audit includes 55 consensus-positive windows. **Recorded item-level keyframe scores are reproducibly 46/9/0** (scores 2/1/0). An earlier coordinator aggregate differs and is retained as historical provenance, but is not used as the numerical result because no supporting item-level corrections are available. This audit does not establish population-wide keyframe coverage. See [input audit](docs/input_audit.md).
4. Three main training seeds share the same source partitions. Five additional group partitions are on *different frozen-feature linear probes*. Neither result measures variation from repartitioning and retraining the main encoders. Cross-platform LOPO is likewise a frozen-feature probe.
5. These data do not provide a new independent external test cohort. Historical exploratory analysis included examination of earlier held-out outcomes, so retrospective inference must not be presented as prospective preregistration.
6. Administrative data provenance, de-identification, and release boundaries are documented in the signed Universitas Bina Darma record (`010/DRPM/Univ-BD/IX/2026`, dated 23 September 2026; see [data governance](docs/data_release.md) and [signed PDF record](docs/UBD_Data_Provenance_and_Protection_Record_Signed_20260923.pdf)). This administrative record is not an institutional ethics approval, an ethics-exemption determination, or a grant of rights over original third-party social-media content.

## Version history and release dates

See [RELEASE_INFO.json](RELEASE_INFO.json) and [CHANGELOG.md](CHANGELOG.md). Initial project release and analysis reference dates were supplied by the researcher; GitHub tag creation timestamps are controlled by GitHub and cannot be backdated by this ZIP.

## Citation

Use [`CITATION.cff`](CITATION.cff) for the research-software archive. The manuscript bibliographic reference should be updated only after publication.
