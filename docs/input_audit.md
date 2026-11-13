# OCR, STT and keyframe audit

The blinded input audit contains 100 video windows (`68` unique source groups; `55` consensus-positive and `45` consensus-negative). Pre-consensus window-aggression agreement (`B_agresi_window`) between the two independent raters (`R1` in `results/input_audit/Form_Audit_R1_scored.csv` and `R2` in `results/input_audit/Form_Audit_R2_scored.csv`) was `95%` (`n = 100`, `5` pre-consensus disagreements adjudicated in `results/input_audit/Form_Adjudikasi.csv`) with unweighted Cohen's kappa **`0.8997995991983968`** (`results/input_audit/revision_20261108/AUDIT_AGREEMENT.csv`). Among audited windows with available branch outputs, **`43/52` OCR cases** (`82.6923076923077%` unweighted; **`0.8263523263523264`** design-weighted) and **`51/64` STT cases** (`79.6875%` unweighted; **`0.7943872080743589`** design-weighted) received the faithful score (`accuracy == 2`). Target label disagreement between consensus `B_agresi_window` and the archived evaluation label `label_final` was **`0/100`** (`disagreements = 0` in `results/input_audit/revision_20261108/AUDIT_TARGET_DISAGREEMENT.json`; archived evaluation labels were not altered).

The numerical public views retain rater scores (`penilai_id = R1` and `penilai_id = R2`) and recorded adjudication values (`Form_Adjudikasi.csv`). Freeform narratives and adjudicator identifiers that could identify individuals have been withheld (`[WITHHELD]`). Source and public-view hashes are distinguished in [`source_redaction_manifest.json`](source_redaction_manifest.json).

## 1. Twenty-stratum (`availability × platform`) SRSWOR sampling design and replay verification (`R1.5`)

As documented in `results/input_audit/revision_20261108/AUDIT_SAMPLING_REPLAY.json`, `results/input_audit/revision_20261108/AUDIT_SAMPLING_ALLOCATION.csv`, and `results/input_audit/revision_20261108/AUDIT_DESIGN_METHOD.json`, the 100 audited windows were drawn from the `3,951` aligned video windows (`CB_MASTER_08907..CB_MASTER_13307`, sorted by `master_id`) **without using labels or model predictions**:
- **Stratification (`20` strata = `4` OCR/STT availability states × `5` platforms)**:
  - Availability state quotas (`n_cell`): `(ocr_ok=0, stt_ok=0) → 17` (from `685` windows), `(ocr_ok=0, stt_ok=1) → 31` (from `1,231` windows), `(ocr_ok=1, stt_ok=0) → 19` (from `760` windows), and `(ocr_ok=1, stt_ok=1) → 33` (from `1,275` windows).
  - Within each availability state, `n_cell` is allocated across the 5 platforms (`Facebook`, `Instagram`, `TikTok`, `X/Twitter`, `YouTube`) by proportional allocation with **largest-remainder rounding** (`N_h` and `n_h` in `results/input_audit/revision_20261108/AUDIT_SAMPLING_ALLOCATION.csv`, with inverse-probability design weights `design_weight = N_h / n_h`).
- **Two distinct RNG seeds (`NumPy default_rng PCG64`)**:
  - **`seed_sample = 26102026`**: Sequentially permutes each of the 20 `(ocr_ok, stt_ok, platform)` blocks (`rng.permutation(len(block))[:quota[platform]]`) to select the 100-window SRSWOR sample.
  - **`seed_blind_order = 26102027`**: Permutes the 100 selected windows (`rng.permutation(100)`) to assign the blinded audit presentation order `AU001..AU100` (`matched_sample_and_audit_order = true`).

Both the 20-stratum sample selection and the `AU001..AU100` presentation order are reproduced deterministically by `python scripts/analysis/reproduce_revision.py --root . --output /tmp/reproduced_v4` and verified by `python scripts/validation/verify_revision.py --root .`.

## 2. Finite-population fidelity bounds vs shared-source exponential multiplier sensitivity (`R1.5`)

In the fixed 3,951-window aligned population, OCR output is available (`ocr_ok = 1`) in **`2,035`** windows (`760 + 1,275`, audited `n = 52` across `10` strata) and STT output is available (`stt_ok = 1`) in **`2,506`** windows (`1,231 + 1,275`, audited `n = 64` across `10` strata). The V4 revision reports two clearly separated uncertainty summaries (`results/input_audit/revision_20261108/AUDIT_FIDELITY_INTERVALS.csv` and `results/input_audit/revision_20261108/AUDIT_DESIGN_METHOD.json`):

1. **Primary conservative finite-population Bonferroni-hypergeometric bounds (`results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_BOUNDS.csv` and `results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_CELLS.csv`)**:
   Within each stratum `h` where a branch is available (`10` OCR strata + `10` STT strata = `20` branch-stratum evaluations), both tails of `Hypergeom(N_h, K_h, n_h)` are inverted at **`cell_alpha = 0.05 / 20 = 0.0025`** (`alpha / 2 = 0.00125` per tail) to obtain integer bounds `[success_count_low, success_count_high]` on the number of faithful outputs `K_h` in that stratum. Summing the integer count bounds across the 10 available strata and dividing by the known available population total (`2,035` for OCR; `2,506` for STT) yields simultaneous (`>= 95%`) finite-population bounds that remain valid even in singleton strata (`n_h = 1`) and under arbitrary fixed within-source clustering:
   - **OCR faithful (`43/52`, `population_available = 2,035`)**: Design-weighted proportion = **`0.8263523263523264`**, conservative 95% finite-population bound = **`[0.2334152334152334, 0.9945945945945946]`** (`sum(success_count_low) = 475`, `sum(success_count_high) = 2,024`).
   - **STT faithful (`51/64`, `population_available = 2,506`)**: Design-weighted proportion = **`0.7943872080743589`**, conservative 95% finite-population bound = **`[0.2430167597765363, 0.9904229848363927]`** (`sum(success_count_low) = 609`, `sum(success_count_high) = 2,482`).
   > **Important boundary:** These bounds constrain the faithful extraction fraction **within the fixed 3,951-window aligned cohort** conditional on the reproduced SRSWOR design and fixed adjudicated ratings. They **must not** be interpreted as validation on newly collected external social-media sources.

2. **Secondary exploratory source-dependence sensitivity (`results/input_audit/revision_20261108/AUDIT_FIDELITY_MODEL_ASSISTED.csv`)**:
   Draws a shared `Exp(1)` multiplier per unique source group (`68` groups, `B = 2,000` replicates, `seed = 6112026`), multiplies `design_weight`, and recalibrates perturbed weights within cells so cell weight totals match the design totals:
   - **`original_20_cells` (`20` `availability × platform` cells)**: OCR 95% interval = **`[0.7418091717475274, 0.8993919698110915]`**; STT 95% interval = **`[0.6956329285188523, 0.8753662604232116]`**.
   - **`availability_4_cells_sensitivity` (`4` `ocr_ok × stt_ok` cells, collapsing platforms)**: OCR 95% interval = **`[0.712582867008305, 0.9139040869850982]`**; STT 95% interval = **`[0.6822614246331797, 0.8833876204406421]`**.
   Reported strictly as secondary exploratory model-assisted sensitivity intervals (`singleton` strata are not design-identifiable under `original_20_cells`).

## 3. Subgroup error transitions and `NA` confidence intervals (`R2.3`)

Paired seed-42 error transitions between `text` and `equal_vat` across audited ratings (`ocr_akurasi`, `stt_akurasi`, `ocr_relevansi`, `stt_relevansi`, and `B_keyframe_event`) are archived in `results/input_audit/revision_20261108/AUDIT_ERROR_TRANSITIONS.csv` and `results/input_audit/revision_20261108/AUDIT_UNCERTAINTY_NOTES.json`. Because discordant source groups per rating level are sparse (`0 to 7` discordant groups across `n = 1 to 55` windows), subgroup confidence intervals (`ci_low`, `ci_high`) are explicitly recorded as **`NA`** (`ci_status = not_estimated_sparse_transitions_and_unvalidated_design_variance`). These descriptive subgroup counts do not identify the causal mechanism of multimodal fusion failure.

## 4. Rubric definitions and keyframe score summary (`AUDIT_RATING_DEFINITIONS.json`)

Recorded rubric definitions from `Panduan_Audit_Blinded.docx` (technical protocol 26 October 2026, SHA-256 `eb52a00c431111415d70e505333bfa19e1d1c43ca9847053b7fd62423a970bfb`) are archived in `results/input_audit/revision_20261108/AUDIT_RATING_DEFINITIONS.json`. For `B_keyframe_event` (`keyframe`), the recorded category definitions and item-level reconciliation from the two independent raters (`R1`, `R2`) and recorded adjudications (`Form_Adjudikasi.csv`) across the **55 consensus-positive** audited windows (`45` consensus-negative windows are `NA`: `no aggression event in clip`; `U`: `unratable`) yield:

| Score (`B_keyframe_event`) | Recorded definition in `AUDIT_RATING_DEFINITIONS.json` | Number of consensus-positive windows |
|:--|--:|--:|
| `2` | `captures focal event` | 46 |
| `1` | `partial event or context` | 9 |
| `0` | `focal event in clip not captured by frame` | 0 |
| **Total** | **Consensus-positive audited windows** | **55** |

An earlier coordinator summary recorded **44/8/3** (scores 2/1/0) without supporting item-level correction records (`results/statistics/CURRENT_AUDIT_RECORD_CHECK.json` and `results/input_audit/KEYFRAME_RECONCILIATION_STATUS.json`). The repository reports the reproducible **46/9/0** item-level distribution while retaining the historical `44/8/3` aggregate provenance note.

Reproduce the item-level keyframe check and the V4 audit tables:

```bash
python scripts/validation/reconcile_keyframes.py --root . --output /tmp/keyframe-audit
python scripts/analysis/reproduce_revision.py --root . --output /tmp/reproduced_v4
```
