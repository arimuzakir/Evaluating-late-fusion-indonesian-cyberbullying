"""Independent numerical, source-freeze, asset, and structured documentation verification for the V4 reviewer revision.

Recalculates all CPU-reproducible V4 revision tables into a temporary directory via
`scripts/analysis/reproduce_revision.py` and verifies:
1. Byte-level preservation of all 279 frozen GoldAdjudicated v1.1 repository files
   and all 89 public integrated V4 files (`configs/revision_20261108/RESULTS_SOURCE_FREEZE.json`),
   plus exclusion of private author callback files (`KEEP_PRIVATE`).
2. Preservation and SHA-256 verification of final author diagram exports `results/figures/Fig_1.png`
   and `results/figures/Fig_2.png`, intentional absence of `.drawio` files in the public repository,
   and exact match between `docs/RELEASE_SHA256.json` file count and actual public repository contents.
3. Exact numerical agreement (`atol = 1e-12`, with zero relaxed tolerances) between freshly
   recomputed tables and archived V4 revision tables across all 35 CPU-reproducible files,
   plus explicit verification of both the public `master_id`-sorted first-appearance group ordering
   and lexicographic `src_<16hex>` pseudonym ordering for the partition bootstrap CIs (`R2.5`).
4. Cohort, label, group, and role invariants across primary, neural-head sensitivity,
   two-partition source allocation (1042 and 2042 -> effective outer seed 2044), and
   seed-42 probability control predictions (`R1.1` - `R1.9`, `R2.1` - `R2.6`).
5. Structured documentation-vs-artifact consistency checks by model, seed, estimand, category,
   and configuration across `README.md`, `CHANGELOG.md`, `VERIFICATION.md`, `docs/*.md`,
   `results/figures/README.md`, `src/reference_archived/`, and `MANUSCRIPT_ARTIFACT_MANIFEST.csv`,
   plus negative mutation self-tests confirming that altering summary numbers or removing
   final PNG figures causes verification to fail.
"""
from pathlib import Path
import argparse, hashlib, json, re, sys, tempfile
import numpy as np
import pandas as pd

FREEZE_MANIFEST = 'configs/revision_20261108/RESULTS_SOURCE_FREEZE.json'

EXPECTED_FIGURES = {
    'results/figures/Fig_1.png': '5a2145c99127310215df17553f24f2219658355e883e609269a41a742cfa069e',
    'results/figures/Fig_2.png': 'eeae9d409efe2b264068562c5e35f3d65260920236f2386e719ebfef13f693c1',
}

EXPECTED_COMPOSITE_FIGURES = [
    'results/figures/manuscript/Fig_3.png',
    'results/figures/manuscript/Fig_4.png',
    'results/figures/manuscript/Fig_5.png',
    'results/figures/manuscript/Fig_6.png',
]

RECOMPUTED_TO_REPO = {
    'ROLE_COUNTS.csv': 'results/video/revision_20261108/roles/ROLE_COUNTS.csv',
    'ALL_THRESHOLDS.csv': 'results/video/revision_20261108/roles/ALL_THRESHOLDS.csv',
    'MODEL_ROLE_RULES.csv': 'results/video/revision_20261108/roles/MODEL_ROLE_RULES.csv',
    'TABLE3_RECOMPUTED.csv': 'results/statistics/revision_20261108/main_result_checks/TABLE3_RECOMPUTED.csv',
    'THREE_SEED_RECOMPUTED.csv': 'results/statistics/revision_20261108/main_result_checks/THREE_SEED_RECOMPUTED.csv',
    'NO_TEXT_FALLBACK.csv': 'results/statistics/revision_20261108/fallback/NO_TEXT_FALLBACK.csv',
    'TEXT_FALLBACK_DECISIONS.csv': 'results/statistics/revision_20261108/fallback/TEXT_FALLBACK_DECISIONS.csv',
    'ROUNDING_AND_MDE.json': 'docs/provenance/revision_20261108/ROUNDING_AND_MDE.json',
    'CALIBRATION_CONTROL_TEST_PREDICTIONS.csv': 'results/statistics/revision_20261108/probability/CALIBRATION_CONTROL_TEST_PREDICTIONS.csv',
    'CALIBRATION_CONTROL_THRESHOLDS.csv': 'results/statistics/revision_20261108/fallback/CALIBRATION_CONTROL_THRESHOLDS.csv',
    'CALIBRATION_CONTROL_FALLBACK.csv': 'results/statistics/revision_20261108/fallback/CALIBRATION_CONTROL_FALLBACK.csv',
    'CALIBRATION_CONTROL_BRIER_INTERVALS.csv': 'results/statistics/revision_20261108/probability/CALIBRATION_CONTROL_BRIER_INTERVALS.csv',
    'BRIER_PAIRED_INTERVALS.csv': 'results/statistics/revision_20261108/probability/BRIER_PAIRED_INTERVALS.csv',
    'TABLE6_PROBABILITY_METRICS.csv': 'results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv',
    'CALIBRATION_CONTROL_RELIABILITY_BINS.csv': 'results/statistics/revision_20261108/probability/CALIBRATION_CONTROL_RELIABILITY_BINS.csv',
    'RELIABILITY_BINS.csv': 'results/statistics/revision_20261108/probability/RELIABILITY_BINS.csv',
    'HEAD_SENSITIVITY_POOLED.csv': 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_POOLED.csv',
    'HEAD_BASELINE_RECONCILIATION.csv': 'results/video/revision_20261108/head_sensitivity/HEAD_BASELINE_RECONCILIATION.csv',
    'REPARTITION_EFFECTS.csv': 'results/video/revision_20261108/source_partitions/REPARTITION_EFFECTS.csv',
    'PARTITION_SEED_LEDGER.csv': 'results/video/revision_20261108/source_partitions/PARTITION_SEED_LEDGER.csv',
    'AUDIT_SAMPLING_ALLOCATION.csv': 'results/input_audit/revision_20261108/AUDIT_SAMPLING_ALLOCATION.csv',
    'AUDIT_AGREEMENT.csv': 'results/input_audit/revision_20261108/AUDIT_AGREEMENT.csv',
    'CONTINGENCY_B_agresi_window.csv': 'results/input_audit/revision_20261108/CONTINGENCY_B_agresi_window.csv',
    'CONTINGENCY_B_keyframe_event.csv': 'results/input_audit/revision_20261108/CONTINGENCY_B_keyframe_event.csv',
    'CONTINGENCY_ocr_akurasi.csv': 'results/input_audit/revision_20261108/CONTINGENCY_ocr_akurasi.csv',
    'CONTINGENCY_ocr_relevansi.csv': 'results/input_audit/revision_20261108/CONTINGENCY_ocr_relevansi.csv',
    'CONTINGENCY_stt_akurasi.csv': 'results/input_audit/revision_20261108/CONTINGENCY_stt_akurasi.csv',
    'CONTINGENCY_stt_relevansi.csv': 'results/input_audit/revision_20261108/CONTINGENCY_stt_relevansi.csv',
    'AUDIT_FIDELITY_MODEL_ASSISTED.csv': 'results/input_audit/revision_20261108/AUDIT_FIDELITY_MODEL_ASSISTED.csv',
    'AUDIT_FIDELITY_DESIGN_CELLS.csv': 'results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_CELLS.csv',
    'AUDIT_FIDELITY_DESIGN_BOUNDS.csv': 'results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_BOUNDS.csv',
    'AUDIT_FIDELITY_INTERVALS.csv': 'results/input_audit/revision_20261108/AUDIT_FIDELITY_INTERVALS.csv',
    'AUDIT_NUMERIC_ROWS.csv': 'results/input_audit/revision_20261108/AUDIT_NUMERIC_ROWS.csv',
    'AUDIT_ERROR_TRANSITIONS.csv': 'results/input_audit/revision_20261108/AUDIT_ERROR_TRANSITIONS.csv',
    'COMMENT_MULTIPLICITY_FAMILY.csv': 'docs/provenance/revision_20261108/COMMENT_MULTIPLICITY_FAMILY.csv',
}

DOC_FILES = [
    'README.md',
    'CHANGELOG.md',
    'VERIFICATION.md',
    'docs/README.md',
    'docs/methods_and_reproducibility.md',
    'docs/input_audit.md',
    'docs/limitations.md',
    'docs/numerical_environment.md',
    'docs/results_index.md',
    'docs/data_release.md',
    'docs/controlled_access.md',
    'docs/feature_interactions.md',
    'results/figures/README.md',
    'notebooks/README.md',
    'src/README.md',
    'src/reference_archived/README.md',
    'src/reference_archived/model_and_fusion_reference.py',
    'docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv',
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def compare_csv(repo_file, repro_file, atol=1e-12):
    a = pd.read_csv(repo_file)
    b = pd.read_csv(repro_file)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise AssertionError(f'Shape/column mismatch for {repo_file.name}: {a.shape} vs {b.shape}')
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            if not np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=atol, equal_nan=True):
                max_diff = float(np.nanmax(np.abs(a[col].to_numpy(float) - b[col].to_numpy(float))))
                raise AssertionError(f'Numeric mismatch in {repo_file.name}.{col}: max_diff={max_diff} > atol={atol}')
        else:
            if not a[col].fillna('').astype(str).equals(b[col].fillna('').astype(str)):
                raise AssertionError(f'String mismatch in {repo_file.name}.{col}')


def _assert_line_association(content, doc_name, anchor, required_tokens):
    matching_lines = [line for line in content.splitlines() if anchor in line]
    if not matching_lines:
        raise AssertionError(f'Anchor {anchor!r} not found in {doc_name}')
    for line in matching_lines:
        if all(str(tok) in line for tok in required_tokens):
            return
    missing = [str(tok) for tok in required_tokens if not any(str(tok) in line for line in matching_lines)]
    raise AssertionError(
        f'Structured association check failed in {doc_name} for anchor {anchor!r}: '
        f'missing or mismatched tokens {missing} on matching line(s)'
    )


def verify_figures_and_checksum_inventory(root, simulated_missing_figure=None):
    root = Path(root).resolve()
    # 1. Author decision: no .drawio files in public git repo
    drawio_files = [str(p.relative_to(root)) for p in root.rglob('*.drawio') if '.git' not in p.parts]
    if drawio_files:
        raise AssertionError(f'Editable .drawio files must not be committed to the public repository: {drawio_files}')

    # 2. Final Fig_1.png and Fig_2.png existence and SHA-256 verification, plus composite Fig_3..Fig_6
    for rel_fig, exp_hash in EXPECTED_FIGURES.items():
        p = root / rel_fig
        if simulated_missing_figure == rel_fig or not p.is_file():
            raise AssertionError(f'Required final figure missing: {rel_fig}')
        actual_hash = sha256(p)
        if actual_hash != exp_hash:
            raise AssertionError(f'SHA-256 mismatch for {rel_fig}: {actual_hash} != {exp_hash}')

    for rel_fig in EXPECTED_COMPOSITE_FIGURES:
        p = root / rel_fig
        if not p.is_file() or p.stat().st_size < 10000:
            raise AssertionError(f'Required composite manuscript figure missing or empty: {rel_fig}')

    # 3. Checksum file count computed dynamically from actual public repo contents (aligned with build_checksums.py)
    skip_dirs = {'.git', '__pycache__', '.venv', '.pytest_cache', '.ipynb_checkpoints', 'private'}
    public_files = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if p.name == '.DS_Store' or p.name.startswith('Icon\r'):
            continue
        if any(part in skip_dirs for part in rel.parts):
            continue
        if rel.as_posix().startswith('docs/internal/'):
            continue
        if rel.as_posix() == 'docs/RELEASE_SHA256.json':
            continue
        public_files.append(rel.as_posix())
    release_sha_path = root / 'docs/RELEASE_SHA256.json'
    release_sha = json.loads(release_sha_path.read_text(encoding='utf-8'))
    recorded_files = release_sha.get('files', {})
    if len(recorded_files) != len(public_files):
        raise AssertionError(
            f'RELEASE_SHA256.json count ({len(recorded_files)}) != actual public file count ({len(public_files)})'
        )
    return len(public_files)


def verify_documentation_consistency(root, texts_override=None, simulated_missing_figure=None):
    root = Path(root).resolve()
    actual_public_count = verify_figures_and_checksum_inventory(root, simulated_missing_figure=simulated_missing_figure)

    texts = {rel: (root / rel).read_text(encoding='utf-8') for rel in DOC_FILES}
    if texts_override:
        texts.update(texts_override)

    # 1. Forbidden / superseded strings that contradict V4 source artifacts
    forbidden = [
        '0.56213', '0.57845', '0.23370', '0.23676', '0.04928',
        '0.559139', '0.011066', '0.558972', '0.008434',
        '0.2512264663532779', '0.20519627702685368',
        '0.48315726365359944', '0.24975172285051612', '0.21479619426367523',
        '0.09545243198743931', '0.07153506150983952', '0.2564143928576902', '0.20735809107968097',
        'visual_linear', 'audio_mlp', 'feature_mlp',
        '0.4648', '0.5608', '0.5568', 'A100',
        '15, 30, 45', '15,30,45', '0.001, 0.01', '0.001,0.01',
        '0.556713', '0.572011', '0.583194', '0.577858',
        '0.00297', '0.03356', '0.02353', '0.01264', '0.002200',
        'partition `2041`', 'partitions (`2041`', '2041, 2042',
        '82.30%', '78.53%', '0.82297', '0.78529',
        '1,781', '2,869', '0.71147', '0.68080',
        'choice(2326', 'choice(1625',
        'p > τ*', 'missing_video_file', 'missing_keyframe_image',
        '0.58777', '0.58290', '0.0247206', 'p_holm_four = 1.0',
        'RETROSPECTIVE_MDE.json', 'PROVENANCE_TIMELINE.csv',
        '12_INASS_Reviewer_Revisions_CPU.ipynb',
        '5c5e5078f147551621d2d1d2fa68740e59f2c7eb9424667262da75d7226d18a9',
        'docs/provenance/revision_20261108/FUSION_HEAD_CONFIGURATION.json',
        'docs/provenance/revision_20261108/PROBABILITY_ESTIMAND.json',
        'docs/provenance/revision_20261108/AUDIT_DESIGN_METHOD.json',
        'docs/provenance/revision_20261108/AUDIT_TARGET_DISAGREEMENT.json',
        'docs/provenance/revision_20261108/AUDIT_UNCERTAINTY_NOTES.json',
        'MV_000001', 'MV_004386',
        'A06', 'A07', 'ADJ_KF',
        '6.6.113+',
        'mdeberta_v3',
        '**187**', '**220**', '**35**',
        '`187` `frame_identity_match`', '`220` `frame_identity_match+stt_content_match`', '`35` `ocr_content_match`',
    ]
    for rel, content in texts.items():
        for bad in forbidden:
            if bad in content:
                raise AssertionError(f'Contradictory/superseded string {bad!r} found in {rel}')

    # 2. Verify VERIFICATION.md checksum count matches actual public file count
    _assert_line_association(
        texts['VERIFICATION.md'],
        'VERIFICATION.md',
        'docs/RELEASE_SHA256.json',
        [f'**{actual_public_count} files PASS**'],
    )

    # 3. Concise top-level README cleanliness & primary summary verification + 3-seed check in methods docs
    if re.search(r'\bR[12]\.\d+\b', texts['README.md']) or 'AWAIT_FINAL_MANUSCRIPT_MAPPING' in texts['README.md']:
        raise AssertionError('Top-level README.md must remain clean without internal reviewer numbering (R1.x/R2.x) or AWAIT placeholders')

    three_seed = pd.read_csv(
        root / 'results/statistics/revision_20261108/main_result_checks/THREE_SEED_RECOMPUTED.csv',
        dtype=str,
    ).set_index('model')
    _assert_line_association(
        texts['README.md'],
        'README.md',
        '**Standalone comments**',
        ['0.9663', '0.9666', '0.9932', '0.9604'],
    )
    _assert_line_association(
        texts['README.md'],
        'README.md',
        '**Aligned video (main)**',
        ['0.5703', '0.5591', '0.0111', '0.5670', '0.5590', '0.0084'],
    )
    _assert_line_association(
        texts['README.md'],
        'README.md',
        '**Focal paired contrast**',
        ['0.0034', '0.0179', '0.0116', '0.6608'],
    )
    _assert_line_association(
        texts['README.md'],
        'README.md',
        '**Probability quality**',
        ['0.2362', '0.0252', '0.2830', '0.1950'],
    )
    _assert_line_association(
        texts['docs/methods_and_reproducibility.md'],
        'docs/methods_and_reproducibility.md',
        'THREE_SEED_RECOMPUTED.csv` records `text`',
        [
            three_seed.loc['text', 'mean'],
            three_seed.loc['text', 'std'],
            three_seed.loc['equal_vat', 'mean'],
            three_seed.loc['equal_vat', 'std'],
        ],
    )

    # 4. Item 2B: Structured Table 6 metrics, ECE, and ranks by model from TABLE6_PROBABILITY_METRICS.csv
    t6 = pd.read_csv(
        root / 'results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv',
        dtype=str,
    ).set_index('model')
    for model_name in ['text', 'equal_vat', 'vat_lr', 'sigmoid_equal_vat', 'convex_vat']:
        row = t6.loc[model_name]
        _assert_line_association(
            texts['docs/methods_and_reproducibility.md'],
            'docs/methods_and_reproducibility.md',
            f'window_macro_f1 = {row["window_macro_f1"]}',
            [
                f'window_brier = {row["window_brier"]}',
                f'window_ece = {row["window_ece"]}',
                f'window_brier_rank = {row["window_brier_rank"]}',
                f'source_brier = {row["source_brier"]}',
                f'source_ece = {row["source_ece"]}',
                f'source_brier_rank = {row["source_brier_rank"]}',
            ],
        )

    # Also verify CALIBRATION_CONTROL_BRIER_INTERVALS.csv paired deltas & CIs by model, reference, and weighting
    cbrier = pd.read_csv(
        root / 'results/statistics/revision_20261108/probability/CALIBRATION_CONTROL_BRIER_INTERVALS.csv',
        dtype=str,
    )
    for model_name in ['sigmoid_equal_vat', 'convex_vat']:
        sub = cbrier[cbrier['model'] == model_name]
        req_tokens = []
        for r in sub.itertuples():
            req_tokens.extend([r.delta, r.ci_low, r.ci_high])
        _assert_line_association(
            texts['docs/methods_and_reproducibility.md'],
            'docs/methods_and_reproducibility.md',
            f'delta = {sub.iloc[0]["delta"]}' if False else sub.iloc[0]['delta'],
            req_tokens,
        )

    # 5. Item 2C: Structured neural-head min/max value-to-configuration mapping from HEAD_SENSITIVITY_POOLED.csv
    hp = pd.read_csv(root / 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_POOLED.csv', dtype=str)
    hp['val'] = hp['pooled_macro_f1'].map(float)
    for prefix in ['concat_mlp', 'healnet']:
        sub = hp[hp['model'].str.startswith(prefix)]
        min_row = sub.loc[sub['val'].idxmin()]
        max_row = sub.loc[sub['val'].idxmax()]
        for doc_rel in ['docs/methods_and_reproducibility.md', 'docs/limitations.md', 'docs/numerical_environment.md']:
            _assert_line_association(
                texts[doc_rel],
                doc_rel,
                min_row['pooled_macro_f1'],
                [min_row['model'], max_row['pooled_macro_f1'], max_row['model']],
            )
            # Ensure min value appears directly before min config name and max value before max config name
            pattern = re.escape(min_row['pooled_macro_f1']) + r'[^;\n]*?' + re.escape(min_row['model']) + r'.*?' + re.escape(max_row['pooled_macro_f1']) + r'[^;\n]*?' + re.escape(max_row['model'])
            if not re.search(pattern, texts[doc_rel]):
                raise AssertionError(
                    f'Min/max configuration pairing mismatch for {prefix} in {doc_rel}: '
                    f'expected {min_row["pooled_macro_f1"]} -> {min_row["model"]} and {max_row["pooled_macro_f1"]} -> {max_row["model"]}'
                )

    # 6. Item 3: Structured exclusion reason aggregation and category-to-count pairing from EXCLUSION_REASON_PLATFORM_CLASS.csv
    excl_df = pd.read_csv(root / 'results/statistics/revision_20261108/exclusions/EXCLUSION_REASON_PLATFORM_CLASS.csv')
    excl_grouped = excl_df[excl_df['excluded'] == True].groupby('reason_pattern')['n'].sum().to_dict()
    expected_excl = {
        'frame_identity_match': 197,
        'frame_identity_match+stt_content_match': 200,
        'ocr_content_match': 45,
        'frame_identity_match+ocr_content_match': 5,
        'frame_identity_match+ocr_content_match+stt_content_match': 3,
    }
    if excl_grouped != expected_excl or sum(excl_grouped.values()) != 450:
        raise AssertionError(f'Exclusion category mismatch: {excl_grouped} != {expected_excl}')
    for reason, count in expected_excl.items():
        _assert_line_association(
            texts['docs/methods_and_reproducibility.md'],
            'docs/methods_and_reproducibility.md',
            f'- `{reason}`',
            [f'**{count}** windows'],
        )
    for doc_rel in ['docs/limitations.md', 'docs/data_release.md']:
        _assert_line_association(
            texts[doc_rel],
            doc_rel,
            '`197` `frame_identity_match`',
            [
                '`200` `frame_identity_match+stt_content_match`',
                '`45` `ocr_content_match`',
                '`5` `frame_identity_match+ocr_content_match`',
                '`3` `frame_identity_match+ocr_content_match+stt_content_match`',
            ],
        )

    # 7. Item 4: Structured hypothesis family verification from FOCAL_FOUR_100000.csv and COMMENT_MULTIPLICITY_FAMILY.csv
    focal_df = pd.read_csv(root / 'results/statistics/FOCAL_FOUR_100000.csv', dtype=str)
    for row in focal_df.itertuples():
        _assert_line_association(
            texts['docs/methods_and_reproducibility.md'],
            'docs/methods_and_reproducibility.md',
            f'`{row.candidate}` vs `{row.baseline}`',
            [
                f'delta_macro_f1 = {row.delta_macro_f1}',
                row.ci_low,
                row.ci_high,
                f'permutation_p = {row.permutation_p}',
                f'p_holm_four = {row.p_holm_four}',
            ],
        )
    cmf_df = pd.read_csv(root / 'docs/provenance/revision_20261108/COMMENT_MULTIPLICITY_FAMILY.csv', dtype=str)
    for row in cmf_df.itertuples():
        _assert_line_association(
            texts['docs/methods_and_reproducibility.md'],
            'docs/methods_and_reproducibility.md',
            f'`{row.candidate}` vs `{row.baseline}`',
            [
                f'delta_macro_f1 = {row.delta_macro_f1}',
                row.ci_low,
                row.ci_high,
                f'group_permutation_p = {row.group_permutation_p}',
                f'group_permutation_p_holm = {row.group_permutation_p_holm}',
            ],
        )

    # 8. Item 5: Partition effects and CB_MASTER_... bootstrap provenance
    rep = pd.read_csv(root / 'results/video/revision_20261108/source_partitions/REPARTITION_EFFECTS.csv', dtype=str).set_index('partition_seed')
    for pseed, eff_seed in [('1042', '1042'), ('2042', '2044')]:
        r = rep.loc[pseed]
        _assert_line_association(
            texts['docs/methods_and_reproducibility.md'],
            'docs/methods_and_reproducibility.md',
            f'**Partition `{pseed}`**',
            [
                f'effective_outer_seed = {eff_seed}',
                f'text_f1 = {r["text_f1"]}',
                f'equal_vat_f1 = {r["equal_vat_f1"]}',
                f'delta = {r["delta"]}',
                r['ci_low'],
                r['ci_high'],
            ],
        )
    _assert_line_association(
        texts['docs/methods_and_reproducibility.md'],
        'docs/methods_and_reproducibility.md',
        'Bootstrap group-ordering rule and provenance',
        ['CB_MASTER_08907..CB_MASTER_13307', "pd.unique(d.sort_values('master_id').group_id)", 'public deterministic ordering rule'],
    )

    # 9. Item 6: Audit rater IDs (R1/R2), keyframe definitions (AUDIT_RATING_DEFINITIONS.json), environment versions, and repo paths
    r1_df = pd.read_csv(root / 'results/input_audit/Form_Audit_R1_scored.csv', dtype=str)
    r2_df = pd.read_csv(root / 'results/input_audit/Form_Audit_R2_scored.csv', dtype=str)
    assert set(r1_df['penilai_id']) == {'R1'} and set(r2_df['penilai_id']) == {'R2'}
    _assert_line_association(
        texts['docs/input_audit.md'],
        'docs/input_audit.md',
        'Pre-consensus window-aggression agreement',
        ['`R1`', '`R2`', '0.8997995991983968', '0/100', 'disagreements = 0'],
    )

    rating_defs = json.loads((root / 'results/input_audit/revision_20261108/AUDIT_RATING_DEFINITIONS.json').read_text(encoding='utf-8'))
    kf_defs = rating_defs['keyframe']
    for score_str, exp_count in [('2', 46), ('1', 9), ('0', 0)]:
        _assert_line_association(
            texts['docs/input_audit.md'],
            'docs/input_audit.md',
            f'| `{score_str}` |',
            [f'`{kf_defs[score_str]}`', f'| {exp_count} |'],
        )

    bounds = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_BOUNDS.csv', dtype=str).set_index('branch')
    cells_sum = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_CELLS.csv').groupby('branch')[['success_count_low', 'success_count_high']].sum()
    for branch, label_str in [('ocr', '**OCR faithful'), ('stt', '**STT faithful')]:
        b_row = bounds.loc[branch]
        c_row = cells_sum.loc[branch]
        _assert_line_association(
            texts['docs/input_audit.md'],
            'docs/input_audit.md',
            label_str,
            [
                f'{b_row["faithful"]}/{b_row["n"]}',
                b_row['weighted_proportion'],
                b_row['ci_low'],
                b_row['ci_high'],
                f'sum(success_count_low) = {int(c_row["success_count_low"])}',
                f'sum(success_count_high) = {int(c_row["success_count_high"]):,}',
            ],
        )

    env_cmp = pd.read_csv(root / 'docs/provenance/revision_20261108/TRAINING_ENVIRONMENT_COMPARISON.csv', dtype=str).set_index('parameter')
    for doc_rel in ['docs/methods_and_reproducibility.md', 'docs/numerical_environment.md']:
        for param in ['numpy', 'torch', 'torchvision', 'platform']:
            _assert_line_association(
                texts[doc_rel],
                doc_rel,
                env_cmp.loc[param, 'original'],
                [env_cmp.loc[param, 'sensitivity']],
            )

    # Verify all backticked relative/repo paths in Markdown docs actually exist in the repository
    path_pat = re.compile(r'`([A-Za-z0-9_./-]+\.(?:csv|json|py|png|pdf|npy|npz|ipynb|md|yml|txt|cff))`')
    for rel, content in texts.items():
        if not rel.endswith('.md'):
            continue
        doc_dir = (root / rel).parent
        for match in path_pat.finditer(content):
            ref = match.group(1)
            if '/' in ref and '*' not in ref and not ref.startswith('/tmp/'):
                cand_rel = (doc_dir / ref).resolve()
                cand_root = (root / ref).resolve()
                if not cand_rel.is_file() and not cand_root.is_file():
                    raise AssertionError(f'Non-existent repository file path `{ref}` referenced in {rel}')

    # 10. Verify MANUSCRIPT_ARTIFACT_MANIFEST.csv paths, hashes, and command roles
    manifest = pd.read_csv(root / 'docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv')
    valid_roles = {
        'RECOMPUTE_ARTIFACT',
        'VERIFY_EXISTING_ARTIFACT',
        'MANUAL_DIAGRAM_EXPORT_VERIFIED_AGAINST_TABLES',
    }
    assert 'command_role' in manifest.columns, 'MANUSCRIPT_ARTIFACT_MANIFEST.csv must include command_role'
    assert set(manifest['command_role']).issubset(valid_roles), f"Invalid command_role values: {set(manifest['command_role']) - valid_roles}"
    for row in manifest.itertuples():
        out_p = root / row.output_path
        assert out_p.is_file() and sha256(out_p) == row.output_sha256, f'Output hash mismatch in manifest for {row.artifact_id}: {row.output_path}'
        for pair in str(row.input_sha256).split(';'):
            pair = pair.strip()
            rel_in, exp_hash = pair.rsplit(':', 1)
            in_p = root / rel_in.strip()
            assert in_p.is_file() and sha256(in_p) == exp_hash.strip(), f'Input hash mismatch in manifest for {row.artifact_id}: {rel_in}'
        if row.command_role == 'MANUAL_DIAGRAM_EXPORT_VERIFIED_AGAINST_TABLES':
            assert row.script_sha256 == 'MANUAL_AUTHOR_DIAGRAM_EXPORT', f'Diagram row {row.artifact_id} must mark script_sha256 as MANUAL_AUTHOR_DIAGRAM_EXPORT'
        else:
            sp = root / row.script_path
            assert sp.is_file() and sha256(sp) == row.script_sha256, f'Script hash mismatch in manifest for {row.artifact_id}: {row.script_path}'


def verify_negative_self_tests(root):
    root = Path(root).resolve()
    base_readme = (root / 'README.md').read_text(encoding='utf-8')
    base_methods = (root / 'docs/methods_and_reproducibility.md').read_text(encoding='utf-8')
    base_audit = (root / 'docs/input_audit.md').read_text(encoding='utf-8')

    mutations = [
        ('Headline video score mutation in README.md', {'README.md': base_readme.replace('0.5703', '0.57845', 1)}, None),
        ('Table 6 rank swap in methods_and_reproducibility.md', {'docs/methods_and_reproducibility.md': base_methods.replace('source_brier_rank = 4', 'source_brier_rank = 1', 1)}, None),
        ('Neural min/max config swap in methods_and_reproducibility.md', {'docs/methods_and_reproducibility.md': base_methods.replace('concat_mlp_e30_wd0.1', 'concat_mlp_e15_wd0.01', 1)}, None),
        ('Exclusion count mutation in methods_and_reproducibility.md', {'docs/methods_and_reproducibility.md': base_methods.replace('**197** windows', '**187** windows', 1)}, None),
        ('Hypothesis family mutation in methods_and_reproducibility.md', {'docs/methods_and_reproducibility.md': base_methods.replace('`equal_vat` vs `siglip2_vta_probe`', '`vat_coverage` vs `vat_lr`', 1)}, None),
        ('Keyframe definition mutation in input_audit.md', {'docs/input_audit.md': base_audit.replace('`captures focal event`', '`clear visual aggression`', 1)}, None),
        ('Missing Fig_1.png simulation', None, 'results/figures/Fig_1.png'),
        ('Missing Fig_2.png simulation', None, 'results/figures/Fig_2.png'),
    ]
    for label, override, missing_fig in mutations:
        failed = False
        try:
            verify_documentation_consistency(root, texts_override=override, simulated_missing_figure=missing_fig)
        except AssertionError:
            failed = True
        if not failed:
            raise AssertionError(f'Negative self-test did not fail as required: {label}')


def run(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root))

    # 1. Source freeze gate (279 protected repo files + 89 public V4 files + 2 private exclusions)
    freeze = json.loads((root / FREEZE_MANIFEST).read_text())
    protected = freeze['protected_repo_files']
    for rel, info in protected.items():
        p = root / rel
        if not p.is_file() or sha256(p) != info['sha256']:
            raise AssertionError(f'Protected frozen file modified or missing: {rel}')
    v4_plan = pd.read_csv(root / 'configs/revision_20261108/V4_INTEGRATION_PLAN.csv')
    public_v4_count = 0
    for row in v4_plan.itertuples():
        dest = root / row.destination
        if row.action == 'KEEP_PRIVATE':
            if dest.exists():
                raise AssertionError(f'Private file must not exist in public repo: {row.destination}')
        else:
            if not dest.is_file() or sha256(dest) != row.source_sha256:
                raise AssertionError(f'Integrated V4 file modified or missing: {row.destination}')
            public_v4_count += 1
    print(f'PASS source-freeze gate: {len(protected)} protected files and {public_v4_count} public V4 files verified')

    # 2. Recompute all 35 CPU revision artifacts and compare against archived V4 files at strict atol=1e-12
    from scripts.analysis.reproduce_revision import run as repro_run, paired_ci
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        repro_run(root, td_path)
        for name, rel_target in RECOMPUTED_TO_REPO.items():
            repo_p = root / rel_target
            repro_p = td_path / name
            if name.endswith('.json'):
                if json.loads(repo_p.read_text()) != json.loads(repro_p.read_text()):
                    raise AssertionError(f'JSON mismatch in {name}')
            else:
                compare_csv(repo_p, repro_p, atol=1e-12)
    print(f'PASS CPU revision reproduction: {len(RECOMPUTED_TO_REPO)}/35 tables and JSON summaries verified at strict atol=1e-12')

    # 3. Verify both public master_id-sorted first-appearance and lexicographic src_<16hex> bootstrap CIs for R2.5
    rp = pd.read_csv(root / 'results/video/revision_20261108/source_partitions/REPARTITION_PREDICTIONS.csv', dtype={'master_id': str, 'group_id': str})
    rep_eff = pd.read_csv(root / 'results/video/revision_20261108/source_partitions/REPARTITION_EFFECTS.csv').set_index('partition_seed')
    expected_lex_ci = {
        1042: (-0.0015131384766076195, 0.032329495427405756),
        2042: (-0.02062388609164806, 0.010964716587871032),
    }
    for split_seed in [1042, 2042]:
        d_sub = rp[rp.partition_seed == split_seed]
        ci_can = paired_ci(d_sub, 'equal_vat_prediction', 'text_prediction', group_order='canonical_master_id')
        ci_lex = paired_ci(d_sub, 'equal_vat_prediction', 'text_prediction', group_order='lexicographic_pseudonym')
        assert abs(ci_can['ci_low'] - rep_eff.loc[split_seed, 'ci_low']) < 1e-12
        assert abs(ci_can['ci_high'] - rep_eff.loc[split_seed, 'ci_high']) < 1e-12
        assert abs(ci_lex['ci_low'] - expected_lex_ci[split_seed][0]) < 1e-12
        assert abs(ci_lex['ci_high'] - expected_lex_ci[split_seed][1]) < 1e-12

    # 4. Structural and domain-specific invariant checks across reviewer requirements (R1.1 - R2.6)
    oof = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv', dtype={'master_id': str, 'group_id': str})
    z42 = oof[oof.seed == 42].set_index('master_id')
    assert len(z42) == 3951 and z42.group_id.nunique() == 214
    assert z42.label_final.value_counts().to_dict() == {1: 2326, 0: 1625}

    # R1.3: 12 neural head configurations (48 fold fits: concat_mlp & healnet, epochs 15/30/60, wd 0.01/0.1)
    hp = pd.read_csv(root / 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_PREDICTIONS.csv', dtype={'master_id': str})
    hp_prog = pd.read_csv(root / 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_PROGRESS.csv')
    assert len(hp) == 3951 * 12 and hp.model.nunique() == 12 and len(hp_prog) == 48
    assert set(hp_prog.epochs) == {15, 30, 60} and set(hp_prog.weight_decay) == {0.01, 0.1}

    # R2.5: 2 completed source allocations (1042 and 2042 with effective outer seed 2044)
    ledger = pd.read_csv(root / 'results/video/revision_20261108/source_partitions/PARTITION_SEED_LEDGER.csv')
    assert set(ledger.partition_id) == {1042, 2042}
    assert set(ledger.loc[ledger.partition_id == 1042, 'effective_outer_seed']) == {1042}
    assert set(ledger.loc[ledger.partition_id == 2042, 'effective_outer_seed']) == {2044}
    assert set(ledger.training_seed) == {42}
    for pseed in [1042, 2042]:
        proles = pd.read_csv(root / f'results/video/revision_20261108/source_partitions/PARTITION_{pseed}_ROLES.csv', dtype={'master_id': str, 'group_id': str})
        assert len(proles) == 3951 * 4
        for _, grp in proles.groupby('outer_fold'):
            assert set(grp.role) == {'fit', 'early_stop', 'meta_fit', 'calibration', 'test'}
            assert grp.groupby('group_id').role.nunique().max() == 1

    # R1.4 & R2.6: Fallback matrices and seed-42 probability controls
    fb = pd.read_csv(root / 'results/statistics/revision_20261108/fallback/NO_TEXT_FALLBACK.csv')
    assert len(fb) == 21 * 3 * 5
    cfb = pd.read_csv(root / 'results/statistics/revision_20261108/fallback/CALIBRATION_CONTROL_FALLBACK.csv')
    assert len(cfb) == 2 * 5 and set(cfb.model) == {'sigmoid_equal_vat', 'convex_vat'}
    t6 = pd.read_csv(root / 'results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv')
    assert len(t6) == 5 and set(t6.model) == {'text', 'equal_vat', 'vat_lr', 'sigmoid_equal_vat', 'convex_vat'}
    cbrier = pd.read_csv(root / 'results/statistics/revision_20261108/probability/CALIBRATION_CONTROL_BRIER_INTERVALS.csv')
    assert len(cbrier) == 8

    # R1.5 & R2.3: Audit bounds and NA subgroup error-transition intervals
    bounds = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_BOUNDS.csv').set_index('branch')
    assert int(bounds.loc['ocr', 'faithful']) == 43 and int(bounds.loc['ocr', 'n']) == 52 and int(bounds.loc['ocr', 'population_available']) == 2035
    assert int(bounds.loc['stt', 'faithful']) == 51 and int(bounds.loc['stt', 'n']) == 64 and int(bounds.loc['stt', 'population_available']) == 2506
    err_tr = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_ERROR_TRANSITIONS.csv')
    assert err_tr['ci_low'].isna().all() and err_tr['ci_high'].isna().all()

    # R2.4: Exclusion accounting across 4,401 segmented windows
    excl_reason = pd.read_csv(root / 'results/statistics/revision_20261108/exclusions/EXCLUSION_REASON_PLATFORM_CLASS.csv')
    assert int(excl_reason['n'].sum()) == 4401
    assert int(excl_reason.loc[excl_reason.excluded, 'n'].sum()) == 450
    assert int(excl_reason.loc[~excl_reason.excluded, 'n'].sum()) == 3951

    # 5. Structured documentation-vs-artifact consistency check + negative self-tests (Item 7)
    verify_documentation_consistency(root)
    verify_negative_self_tests(root)
    print('PASS all V4 revision structural, statistical, figure-hash, and structured documentation-consistency invariants (including 8/8 negative mutation self-tests)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify V4 reviewer revision integration, numerical reproduction, and documentation consistency.')
    parser.add_argument('--root', default='.', help='Repository root path')
    args = parser.parse_args()
    run(args.root)
