# OCR, STT and keyframe audit

The blinded input audit contains 100 video windows. Pre-consensus aggression agreement between the two raters was 95% with Cohen's kappa 0.8998. Among cases with available outputs, 43/52 OCR cases and 51/64 STT cases received the recorded accurate/faithful score. These fractions describe the audited subset, not corpus-wide error rates.

The numerical public views retain rater scores and recorded adjudication values. Freeform remarks that could identify individuals have been withheld. Source and public-view hashes are distinguished in [`source_redaction_manifest.json`](source_redaction_manifest.json).

## Keyframe score summary from archived item-level data

The auditable reporting unit is the **100-window audit**, of which **55 windows** have consensus-positive aggression labels. For each item, the script reads both independent scores and applies a recorded adjudication value if available. A non-disputed item retains its agreed score. Every disputed core field must have a recorded adjudication entry or the script fails.

| Score assigned to keyframe evidence | Number of consensus-positive windows |
|:--|--:|
| 2 | 46 |
| 1 | 9 |
| 0 | 0 |
| **Total** | **55** |

This **46/9/0** distribution is the reproducible, descriptive **item-level summary** used for reporting within this audit. It is not a new human reassessment and does not establish coverage or accuracy on the entire 3,951-window evaluation cohort. It should not be treated as evidence of model superiority.

### Provenance of an earlier aggregate

An earlier coordinator summary recorded **44/8/3** (scores 2/1/0) for the same claimed 55 positive windows. That summary has no corresponding per-item correction records in the material supplied for this release, so it **cannot replace the recorded item-level scores**. The historical aggregate is preserved in `results/statistics/CURRENT_AUDIT_RECORD_CHECK.json` and is compared with the derived counts in `results/input_audit/KEYFRAME_RECONCILIATION_STATUS.json`. This is a documented difference in source summaries, not an amendment to an annotator's judgment. No new human endorsement of either aggregate is asserted.

Reproduce the item-level calculation without modifying the source forms:

```bash
python scripts/validation/reconcile_keyframes.py --root . --output /tmp/keyframe-audit
```

The result is descriptive. Recorded relationships between OCR/STT fidelity and model decisions do not independently establish that extraction errors caused fusion errors.
