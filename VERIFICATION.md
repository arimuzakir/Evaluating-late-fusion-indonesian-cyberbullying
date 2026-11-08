# Release verification

**Archive:** v1.2.0 additive V4 reviewer revision release, assembled 8 November 2026 (preserving primary `v1.1-GoldAdjudicated` numerical sources from 28 October 2026 byte-for-byte).

## Automated checks

- **Source-freeze gate (`INASS_RESULTS_FREEZE_20261108_V4`)**: **279/279 protected baseline files PASS**, **89/89 public V4 files PASS**, **2/2 private audit files excluded from public git PASS**.
- **Public SHA-256 inventory (`docs/RELEASE_SHA256.json`)**: **433 files PASS** across all tracked public files (excluding the manifest itself).
- **Primary numeric checks (`src.evaluate_release`)**: **27/27 PASS**.
- **Keyframe item-level reaggregation (`reconcile_keyframes.py`)**: **PASS** with 46 score-2, 9 score-1, and 0 score-0 among 55 audited positive windows (`44/8/3` historical coordinator aggregate documented separately).
- **V4 CPU revision reproduction (`verify_revision.py`)**: **35/35 CPU-derivable tables and JSON summaries recomputed from public inputs PASS**, plus **all R1.1–R2.6 structural and statistical invariants PASS**.
- **LOPO clustered bootstrap (`lopo_cluster_intervals.py`)**: **PASS** (`20,000` clustered replicates).
- **Frozen-feature five-partition probe (`repeated_group_probe.py`)**: **PASS** with single-thread BLAS determinism.
- **Notebook validation and execution**: **6/6 notebooks validated and executed** in Python 3.13 with documented dependencies.

## Verification commands

Run from the repository root:

```bash
# 1. Fast baseline + V4 source-freeze & 27 primary numeric checks
python scripts/validation/verify_release.py --root .

# 2. V4 source-freeze gate + 35-table CPU revision recomputation + R1.1-R2.6 invariants
python scripts/validation/verify_revision.py --root .

# 3. Full suite including 20,000-replicate LOPO and 5-seed frozen-feature probe recomputation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/validation/verify_release.py --root . --full
```

All automated checks examine the publicly archived numerical files. They do not verify restricted raw social-media videos, authenticate third-party source ownership, or establish institutional ethics clearance. Administrative data provenance is recorded in `010/DRPM/Univ-BD/IX/2026` (which is not an ethics approval or exemption determination; formal institutional ethics determination remains open).
