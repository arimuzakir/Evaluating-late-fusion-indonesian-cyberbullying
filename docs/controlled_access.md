# Controlled Access, Ethics Scope, and Technical Prerequisites

**Status:** `PROPOSED` institutional request protocol (mediated by the corresponding author; no automated self-service portal is deployed).

## 1. Public vs. Controlled-Access Assets

To protect child-safety, privacy, and third-party platform terms of service while enabling full numerical verification, the study separates public verification artifacts from restricted source materials.

| Asset tier | Location | Contents | Access status |
| --- | --- | --- | --- |
| **Public verification archive** | This GitHub repository | Anonymized metadata (`src_<16hex>` group IDs), `GoldAdjudicated v1.1` labels, deterministic outer/inner role splits, saved test and out-of-fold (OOF) predictions, operating thresholds, audit scoring sheets, V4 revision tables, pointwise reliability tables, reproduction/validation scripts, and SHA-256 manifests. | Publicly available |
| **Restricted primary & intermediate artifacts** | Managed offline / institutional archive (not in Git) | Raw short-form videos and audio tracks, sampled keyframes, unredacted social media text/comments, original platform URLs/handles, pre-redaction OCR/STT text streams, raw feature tensors/embeddings, and trained neural checkpoints. | Controlled access (`PROPOSED` upon verified request) |

## 2. Institutional Governance and Ethics Scope

Institutional documentation referenced by the manuscript comprises two letters issued by the Directorate of Research and Community Service (Direktorat Riset dan Pengabdian kepada Masyarakat, DRPM), Universitas Bina Darma:

- `010/DRPM/Univ-BD/IX/2026` (23 September 2026) — initial institutional statement on the research dataset and academic governance framework.
- `021/KET-DRPM/Univ-BD/XI/2026` (6 November 2026) — institutional clarification and governance statement citing Universitas Bina Darma Rector's Decrees `0171/SK/Univ-BD/I/2011` and `0172/SK/Univ-BD/I/2011` (2 January 2011) and the University's Code of Ethics for Academic Personnel.

**Explicit scope limitation:** Neither letter claims formal clearance from an independent Institutional Review Board (IRB) / research ethics committee nor a formal waiver of informed consent. The study did not conduct human-subject interventions or direct contact with minors, and the public release excludes raw audiovisual media and personally identifying text.

## 3. Proposed Request Procedure

Qualified researchers seeking restricted materials for independent verification or non-commercial academic audit should contact the corresponding author:

- **Corresponding author:** Ari Muzakir (`arimuzakir@binadarma.ac.id`)
- **Affiliation:** Universitas Bina Darma, Palembang, Indonesia

A complete request under this `PROPOSED` procedure should include:

1. Applicant name, academic title/role, verifiable institutional email, and department/faculty.
2. Specific artifacts requested (e.g., frozen feature tensors, checkpoints, or redacted raw text/media subset) and research purpose.
3. Written non-redistribution and child-safety undertaking: no attempt to re-identify anonymized sources (`src_<16hex>`) or depicted individuals, no public posting of raw media/text, and secure local storage followed by deletion once verification concludes.
4. Where required by the requester's home institution, documentation of local ethics/IRB review for secondary analysis.

## 4. Technical Prerequisites Across Reproducibility Levels

The repository distinguishes three reproducibility tiers (see [`docs/methods_and_reproducibility.md`](methods_and_reproducibility.md) for exact CPU commands):

- **Level 1 — Numerical, Statistical, Reliability, Table & Figure Reproduction (Public, CPU-only):** Fully executable from this repository using `scripts/analysis/reproduce_revision.py`, `scripts/analysis/build_manuscript_artifacts.py`, `results/statistics/revision_20261113/reliability/reproduce_reliability.py`, and `scripts/validation/verify_*.py` (`atol = 1e-12`). No restricted files or GPU hardware are needed.
- **Level 2 — Checkpoint Inference Over Aligned Features/Media (Controlled Access):** Requires obtaining trained model checkpoints and aligned feature/media inputs via controlled access, plus compatible Python/PyTorch/Transformers packages matching `docs/provenance/revision_20261108/TRAINING_ENVIRONMENT_COMPARISON.csv`.
- **Level 3 — End-to-End Training from Scratch (Not Publicly Self-Contained):** Public release contains reference architectures and preprocessing scripts (`src/reference_archived/model_and_fusion_reference.py`, `scripts/preprocessing/official_api_ingestion_and_windowing_reference.py`), not standalone raw media/feature dumps or runnable end-to-end GPU training pipelines. Re-training head configurations on L4 vs. different PyTorch/CUDA stacks produced non-identical sensitivity replay values (documented in `docs/provenance/revision_20261108/TRAINING_ENVIRONMENT_COMPARISON.csv` and `results/video/revision_20261108/head_sensitivity/HEAD_BASELINE_RECONCILIATION.csv`), so primary manuscript inference remains anchored on frozen `GoldAdjudicated v1.1` outputs.
