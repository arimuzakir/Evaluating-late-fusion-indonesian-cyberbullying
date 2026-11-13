# Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection

Versioned code, anonymized labels, stored predictions, and reproducible CPU verification suites for the manuscript *"Evaluating Late Fusion under Fragmented Multimodal Evidence for Indonesian Cyberbullying Detection"*.

This repository documents two **distinct empirical tracks** using the frozen primary release **`GoldAdjudicated v1.1`**: (1) a standalone Indonesian social-media comment benchmark (`n = 8,196` selected comments; held-out test `n = 1,226` across `865` source groups), and (2) a controlled comparison of unimodal and multimodal late-fusion configurations on temporally aligned 10-second short-form video windows (`n = 3,951` aligned windows across `214` linked source groups and four source-disjoint outer folds; `450` segmented windows excluded due to alignment duplication/content-match rules). The operational target is content-level *bullying-indicative aggression*, not clinical or interpersonal verification of a complete cyberbullying episode.

## Summary of Primary Findings

| Evaluation track | Cohort | Headline result (`GoldAdjudicated v1.1`, seed 42) |
| :--- | :--- | :--- |
| **Standalone comments** | `8,196` selected (`1,226` test) | Equal text ensemble Macro F1 = `0.9663` (Accuracy `0.9666`, ROC-AUC `0.9932`); `IndoBERT-base-p2` Macro F1 = `0.9604` |
| **Aligned video (main)** | `3,951` windows (`214` groups) | Extracted text Macro F1 = `0.5703` (`0.5591 ± 0.0111` across 3 seeds); Equal VAT Macro F1 = `0.5670` (`0.5590 ± 0.0084` across 3 seeds) |
| **Focal paired contrast** | `214` source groups (`100,000` replicates) | Equal VAT − Extracted text Δ Macro F1 = `−0.0034`, 95% CI `[−0.0179, +0.0116]`, Holm-adjusted `p = 0.6608` |
| **Probability quality** | `3,951` windows (seed 42) | Equal VAT lowers window Brier (`0.2362`, ECE `0.0252`) vs Extracted text (`0.2830`, ECE `0.1950`); source-weighted rankings differ (`configs/revision_20261108/PROBABILITY_ESTIMAND.json`) |
| **Additional sensitivities** | CPU V4 & S7 suites | Internal 5-role source partitions (`1042`, `2042`/`2044`), 12 neural-head fits (`concat_mlp`, `healnet`), 20-stratum audit bounds, and S7 pointwise reliability intervals confirm bounded negative classification evidence without external validation |

Because the comment benchmark and video evaluation differ in sampling unit, agreement filtering, and text provenance, their Macro F1 scores are **not** directly comparable as an ablation of modality choice.

## Repository Structure

```text
├── .github/workflows/verify-release.yml   Automated CI validation workflow
├── configs/                               Model, audio, environment, and source-freeze configs
├── data/                                  Human votes, Gold v1.1 manifests, frozen SigLIP 2 & waveform16 features
├── docs/                                  Methods, audit, controlled access, manifest, SHA-256 inventory
├── notebooks/                             Six read-only notebooks covering Tables 1–6 and sensitivities
├── results/
│   ├── comments/benchmark_v1.1/           Held-out comment predictions, metrics, and Holm tests
│   ├── figures/                           Author workflow diagrams (Fig_1–2) & manuscript composites (Fig_3–6)
│   ├── generalization/                    Cross-platform LOPO & 5-seed frozen-feature probe summaries
│   ├── input_audit/                       100-window blinded audit ratings, keyframe trace & 20-stratum bounds
│   ├── learned_fusion/                    Logistic stacking & seed-42 calibration control splits/weights
│   ├── reproduction/                      Baseline release check summaries
│   ├── statistics/                        Focal 100k contrasts, exclusions & S7 reliability pointwise CIs
│   └── video/                             Frozen 3-seed video OOF predictions, head grid & 2 source partitions
├── scripts/                               Preprocessing, validation, and CPU table/figure reproduction scripts
└── src/                                   Canonical release checker, inference utilities & archived references
```

## Quick Start (CPU Verification & Artifact Reproduction)

All Level 1 numerical checks, statistical intervals, manuscript tables (`Table 1–6`), manuscript composite figures (`Fig_3–6`), and S7 reliability curves reproduce on CPU from the repository root without restricted raw media or GPU training:

```bash
python -m pip install -r requirements.txt
python scripts/validation/verify_release.py --root .
python scripts/validation/verify_revision.py --root .
python scripts/validation/verify_manuscript_artifacts.py --root .
python scripts/analysis/reproduce_revision.py --root . --output /tmp/repro_v4 --replicates 10000
python scripts/analysis/build_manuscript_artifacts.py --root . --output /tmp/manuscript_artifacts
python results/statistics/revision_20261113/reliability/reproduce_reliability.py --output /tmp/reliability_ci
```

Full suite including 20,000-replicate LOPO clustered intervals and 5-seed frozen-feature linear probes:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/validation/verify_release.py --root . --full
```

## Reproducibility Boundaries & Governance

- **Level 1 (Public, CPU-only):** Full recomputation of manuscript Tables 1–6, composite Figures 3–6, V4 sensitivity tables (`atol = 1e-12`), and S7 reliability intervals (`reproduce_reliability.py`). Workflow diagrams `results/figures/Fig_1.png` and `results/figures/Fig_2.png` are manual author exports verified against tabular counts.
- **Level 2 (Frozen-feature & Checkpoint Tier):** `SigLIP 2` + `waveform16` probes run on CPU (`scripts/analysis/repeated_group_probe.py`, `scripts/analysis/lopo_cluster_intervals.py`). Re-evaluating fine-tuned neural encoders requires restricted media/checkpoints.
- **Level 3 (End-to-End Training from Scratch):** Historical model/fusion reference code is archived in [`src/reference_archived/model_and_fusion_reference.py`](src/reference_archived/model_and_fusion_reference.py); raw social-media videos, audio, and unredacted PII are withheld. See [`docs/controlled_access.md`](docs/controlled_access.md) for the `PROPOSED` institutional access procedure and DRPM letters `010/DRPM/Univ-BD/IX/2026` / `021/KET-DRPM/Univ-BD/XI/2026`.

## Key Documentation

- [Manuscript Table/Figure & Supplementary Manifest](docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv)
- [Methods & Reproducibility Protocol](docs/methods_and_reproducibility.md)
- [Input Audit & Finite-Population Bounds](docs/input_audit.md)
- [Modality Feature Interactions (S2)](docs/feature_interactions.md)
- [Data Release & Provenance](docs/data_release.md)
- [Controlled Access & Institutional Scope (`PROPOSED`)](docs/controlled_access.md)
- [Limitations & Claim Scope](docs/limitations.md)
- [Results & Path Index](docs/results_index.md)

## Citation & License

Released under the [MIT License](LICENSE). Cite this repository via [`CITATION.cff`](CITATION.cff).
