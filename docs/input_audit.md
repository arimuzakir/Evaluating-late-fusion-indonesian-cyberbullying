# OCR, STT and keyframe audit

The blinded input audit contains 100 video windows (`55` consensus-positive and `45` consensus-negative). Pre-consensus aggression agreement between the two independent raters (`A06`, `A07`) was 95% with Cohen's kappa 0.8998. Among audited cases with available outputs, **43/52 OCR cases** (`82.69%` unweighted; `82.30%` stratified finite-population weighted) and **51/64 STT cases** (`79.69%` unweighted; `78.53%` stratified finite-population weighted) received the accurate/faithful score (`2`).

The numerical public views retain rater scores and recorded adjudication values. Freeform narratives that could identify individuals have been withheld (`[WITHHELD]`). Source and public-view hashes are distinguished in [`source_redaction_manifest.json`](source_redaction_manifest.json).

## 1. Stratified SRSWOR sampling design and replay verification (`R1.5`)

The 100 audited windows were drawn by stratified simple random sampling without replacement (SRSWOR) from the 3,951 aligned video windows sorted in canonical `segment_id` order (`results/input_audit/revision_20261108/AUDIT_SAMPLING_REPLAY.json` and `AUDIT_SAMPLING_ALLOCATION.csv`):
- **Positive stratum (`N_1 = 2,326`)**: `n_1 = 55` windows sampled with `numpy.random.default_rng(26102026).choice(2326, size=55, replace=False)`.
- **Negative stratum (`N_0 = 1,625`)**: `n_0 = 45` windows sampled with `numpy.random.default_rng(26102027).choice(1625, size=45, replace=False)`.

Both sampling draws are verified deterministically (`replay_exact_match = True` across all 100 `segment_id`s) by `python scripts/validation/verify_revision.py --root .`.

## 2. Finite-population fidelity bounds vs model-assisted sensitivity (`R1.5`, `R2.6`)

Because OCR and STT availability is observed on all 3,951 cohort windows (`1,781` OCR-present: `1,069` positive, `712` negative; `2,869` STT-present: `1,666` positive, `1,203` negative), the V4 revision archives two distinct uncertainty summaries (`results/input_audit/revision_20261108/`):

1. **Conservative Bonferroni-hypergeometric finite-population bounds (`AUDIT_FIDELITY_DESIGN_BOUNDS.csv` and `AUDIT_FIDELITY_INTERVALS.csv`)**:
   Exact one-sided hypergeometric inversion (`α / 4 = 0.0125` per tail across the two strata) combined with worst-case completion (0% vs 100% fidelity) over unaudited windows in the 3,951-window cohort:
   - **OCR faithful (`43/52`)**: Stratified Horvitz-Thompson estimate = **0.82297**, conservative 95% finite-population bound = **[0.23342, 0.99459]**.
   - **STT faithful (`51/64`)**: Stratified Horvitz-Thompson estimate = **0.78529**, conservative 95% finite-population bound = **[0.24302, 0.99042]**.
   > **Important boundary:** These bounds constrain the true extraction fidelity fraction **within the finite 3,951-window evaluation cohort** under worst-case unaudited completions. They **must not** be interpreted as external validation on new social-media sources or unseen platforms.
2. **Model-assisted cluster bootstrap sensitivity (`AUDIT_FIDELITY_MODEL_ASSISTED.csv`)**:
   Parametric source-cluster bootstrap (`B = 2,000`, `seed = 6112026`) under a homogeneous stratum-superpopulation assumption (`OCR: [0.71147, 0.91632]`; `STT: [0.68080, 0.87809]`). Reported strictly as a model-assisted sensitivity analysis.

## 3. Subgroup error transitions and `NA` confidence intervals (`R1.5`, `R2.6`)

Paired seed-42 error transitions between `text` and `equal_vat` across audited OCR/STT fidelity and keyframe strata are archived in `results/input_audit/revision_20261108/AUDIT_ERROR_TRANSITIONS.csv`. Because audited subgroup sample sizes are small (`n = 9 to 51` windows across `7 to 39` source groups), subgroup confidence intervals are explicitly recorded as **`NA`** (`small_n_no_reliable_cluster_ci`). Descriptive audit associations do not identify the causal mechanism of fusion failure.

## 4. Keyframe score summary from archived item-level data

Among the **55 consensus-positive** audited windows, item-level reconciliation from the two independent raters (`A06`, `A07`) and recorded adjudications (`ADJ_KF`) yields:

| Score assigned to keyframe evidence | Number of consensus-positive windows |
|:--|--:|
| 2 (`clear visual aggression`) | 46 |
| 1 (`ambiguous / context-dependent`) | 9 |
| 0 (`absent from midpoint frame`) | 0 |
| **Total** | **55** |

An earlier coordinator summary recorded **44/8/3** (scores 2/1/0) without supporting item-level correction records (`results/statistics/CURRENT_AUDIT_RECORD_CHECK.json` and `results/input_audit/KEYFRAME_RECONCILIATION_STATUS.json`). The repository reports the reproducible **46/9/0** item-level distribution while retaining the historical `44/8/3` aggregate provenance note.

Reproduce the item-level keyframe check and the V4 audit tables:

```bash
python scripts/validation/reconcile_keyframes.py --root . --output /tmp/keyframe-audit
python scripts/analysis/reproduce_revision.py --root . --output /tmp/reproduced_v4
```
