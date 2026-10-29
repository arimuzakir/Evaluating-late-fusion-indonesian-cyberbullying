# Release verification

**Archive:** v1.1.2 final numerical release, assembled 28 October 2026.

## Automated checks

- SHA-256 inventory: **326 files PASS** (the manifest excludes itself).
- Main numeric checks: **27/27 PASS**.
- Keyframe item-level reaggregation: **PASS** with 46 score-2, 9 score-1 and 0 score-0 among 55 audited positive windows. Historical aggregate remains documented separately.
- LOPO clustered bootstrap: **PASS**.
- Frozen-feature five-partition probe: **PASS**, after enforcing deterministic BLAS computation.
- Notebook validation: **6/6 notebooks executed** in Python 3.13 with the documented dependencies.

Run:

```bash
python scripts/validation/verify_release.py --root . --full
```

All checks examine the publicly archived numerical files. They do not verify the original withheld media, authenticate source ownership, or establish institutional ethics clearance. The release dates recorded in `RELEASE_INFO.json` were supplied by the researcher and are distinct from assembly dates.
