# Release verification

**Archive:** v1.2.1 manuscript artifact, S7 reliability, and V4 revision synchronization (13 November 2026), preserving primary `v1.1-GoldAdjudicated` numerical sources (28 October 2026) and V4 reviewer revision files (8 November 2026) byte-for-byte.

## Automated checks

- **Source-freeze gate (`INASS_RESULTS_FREEZE_20261108_V4`)**: **279/279 protected baseline files PASS**, **89/89 public V4 files PASS**, **2/2 private audit files excluded from public git PASS**.
- **Public SHA-256 inventory (`docs/RELEASE_SHA256.json`)**: **454 files PASS** across all tracked public files (excluding the manifest itself), verified against the actual public repository file count.
- **Primary numeric checks (`src.evaluate_release`)**: **27/27 PASS**.
- **Keyframe item-level reaggregation (`reconcile_keyframes.py`)**: **PASS** with 46 score-2 (`captures focal event`), 9 score-1 (`partial event or context`), and 0 score-0 (`focal event in clip not captured by frame`) among 55 consensus-positive audited windows from independent raters `R1` and `R2` (`44/8/3` historical coordinator aggregate documented separately).
- **V4 CPU revision reproduction (`verify_revision.py`)**: **35/35 CPU-derivable tables and JSON summaries recomputed from public inputs PASS at `atol = 1e-12`** (including exact `1e-12` bootstrap CI verification under both the public `master_id`-sorted first-appearance group ordering rule and lexicographic `src_<16hex>` pseudonym ordering for `REPARTITION_EFFECTS.csv`), plus structural/statistical invariants, `Fig_1.png`/`Fig_2.png` hashes, composite `Fig_3.png`–`Fig_6.png`, and structured documentation-vs-artifact consistency checks (including negative mutation self-tests).
- **Final manuscript & S7 verification (`verify_manuscript_artifacts.py`)**: **Tables 1–6** dynamically recomputed against frozen inputs (`atol = 1e-12`), **Figures 1–6** verified, **Supplementary S7** pointwise reliability (`reproduce_reliability.py`, 100 rows = 68 CIs + 12 sparse + 20 empty) reproduced at `atol = 1e-12`, and all **21** manifest rows in `docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv` hash-verified.
- **LOPO clustered bootstrap (`lopo_cluster_intervals.py`)**: **PASS** (`20,000` clustered replicates).
- **Frozen-feature five-partition probe (`repeated_group_probe.py`)**: **PASS** with single-thread BLAS determinism.
- **Notebook validation and execution**: **6/6 notebooks validated and executed** with documented dependencies.

## Verification commands

Run from the repository root:

```bash
# 1. Fast baseline + V4 source-freeze & 27 primary numeric checks
python scripts/validation/verify_release.py --root .

# 2. V4 source-freeze gate + 35-table CPU revision recomputation (atol=1e-12) + doc consistency
python scripts/validation/verify_revision.py --root .

# 3. Manuscript Table 1-6, Fig 1-6, Supplementary S1-S8, and S7 reliability verification
python scripts/validation/verify_manuscript_artifacts.py --root .

# 4. Full suite including 20,000-replicate LOPO and 5-seed frozen-feature probe recomputation
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/validation/verify_release.py --root . --full
```

All automated checks examine the publicly archived numerical files. They do not verify restricted raw social-media videos, authenticate third-party source ownership, or establish institutional IRB/ethics clearance. Institutional statements `010/DRPM/Univ-BD/IX/2026` (23 September 2026) and `021/KET-DRPM/Univ-BD/XI/2026` (6 November 2026, citing Decrees `0171/SK/Univ-BD/I/2011` and `0172/SK/Univ-BD/I/2011`) document institutional governance without claiming dedicated research ethics committee approval or an informed-consent waiver (see [`docs/controlled_access.md`](docs/controlled_access.md)).
