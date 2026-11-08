"""Independent numerical, source-freeze, and documentation-consistency verification for the V4 reviewer revision.

Recalculates all CPU-reproducible V4 revision tables into a temporary directory via
`scripts/analysis/reproduce_revision.py` and verifies:
1. Byte-level preservation of all 279 frozen GoldAdjudicated v1.1 repository files
   and all 89 public integrated V4 files (`configs/revision_20261108/RESULTS_SOURCE_FREEZE.json`),
   plus exclusion of private author callback files (`KEEP_PRIVATE`).
2. Cohort, label, group, and role invariants across primary, neural-head sensitivity,
   two-partition source allocation (1042 and 2042 -> effective outer seed 2044), and
   seed-42 probability control predictions (`R1.1` - `R1.9`, `R2.1` - `R2.6`).
3. Exact numerical agreement (`atol = 1e-12`, with zero relaxed tolerances) between freshly
   recomputed tables and archived V4 revision tables across all 35 CPU-reproducible files,
   plus explicit verification of both canonical pre-pseudonymisation (`master_id`) group ordering
   and lexicographic `src_<16hex>` pseudonym ordering for the partition bootstrap CIs (`Item 8`).
4. Automated documentation-vs-artifact consistency checks across `README.md`, `CHANGELOG.md`,
   `VERIFICATION.md`, `docs/*.md`, `src/reference_archived/`, and `MANUSCRIPT_ARTIFACT_MANIFEST.csv`
   against the authoritative CSV/JSON artifacts (`Item 1` - `Item 9`).
"""
from pathlib import Path
import argparse, hashlib, json, sys, tempfile
import numpy as np
import pandas as pd

FREEZE_MANIFEST = 'configs/revision_20261108/RESULTS_SOURCE_FREEZE.json'

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


def verify_documentation_consistency(root):
    root = Path(root).resolve()
    doc_files = [
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
        'notebooks/README.md',
        'src/README.md',
        'src/reference_archived/README.md',
        'src/reference_archived/model_and_fusion_reference.py',
        'docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv',
    ]
    texts = {rel: (root / rel).read_text(encoding='utf-8') for rel in doc_files}

    # 1. Forbidden / superseded strings that contradict V4 source artifacts
    forbidden = [
        '0.56213', '0.57845', '0.23370', '0.23676', '0.04928',
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
        '0.58777', '0.58290', '0.0247206', 'p_holm = 1.0',
        'RETROSPECTIVE_MDE.json', 'PROVENANCE_TIMELINE.csv',
        '12_INASS_Reviewer_Revisions_CPU.ipynb',
        '5c5e5078f147551621d2d1d2fa68740e59f2c7eb9424667262da75d7226d18a9',
        'docs/provenance/revision_20261108/FUSION_HEAD_CONFIGURATION.json',
    ]
    for rel, content in texts.items():
        for bad in forbidden:
            if bad in content:
                raise AssertionError(f'Contradictory/superseded string {bad!r} found in {rel}')

    # 2. Required artifact-derived values in README and methods/audit/limitations docs
    t6 = pd.read_csv(root / 'results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv', dtype=str).set_index('model')
    sig_f1 = t6.loc['sigmoid_equal_vat', 'window_macro_f1']
    sig_brier = t6.loc['sigmoid_equal_vat', 'window_brier']
    conv_f1 = t6.loc['convex_vat', 'window_macro_f1']
    conv_brier = t6.loc['convex_vat', 'window_brier']
    for token in [sig_f1, sig_brier, conv_f1, conv_brier]:
        assert token in texts['README.md'], f'Missing Table 6 control metric {token} in README.md'
        assert token in texts['docs/methods_and_reproducibility.md'], f'Missing Table 6 control metric {token} in methods_and_reproducibility.md'

    hp = pd.read_csv(root / 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_POOLED.csv', dtype=str)
    hp['val'] = hp['pooled_macro_f1'].map(float)
    mlp_sub = hp[hp.model.str.startswith('concat_mlp')]
    hn_sub = hp[hp.model.str.startswith('healnet')]
    mlp_min = mlp_sub.loc[mlp_sub['val'].idxmin(), 'pooled_macro_f1']
    mlp_max = mlp_sub.loc[mlp_sub['val'].idxmax(), 'pooled_macro_f1']
    hn_min = hn_sub.loc[hn_sub['val'].idxmin(), 'pooled_macro_f1']
    hn_max = hn_sub.loc[hn_sub['val'].idxmax(), 'pooled_macro_f1']
    for token in ['concat_mlp', 'healnet', mlp_min, mlp_max, hn_min, hn_max, 'NVIDIA L4']:
        assert token in texts['docs/methods_and_reproducibility.md'], f'Missing R1.3 token {token} in methods_and_reproducibility.md'

    rep = pd.read_csv(root / 'results/video/revision_20261108/source_partitions/REPARTITION_EFFECTS.csv', dtype=str).set_index('partition_seed')
    for pseed in ['1042', '2042']:
        for col in ['text_f1', 'equal_vat_f1']:
            val_str = rep.loc[pseed, col]
            assert val_str in texts['README.md'] and val_str in texts['docs/methods_and_reproducibility.md'], f'Missing R2.5 {pseed} {col} in docs'

    bounds = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_FIDELITY_DESIGN_BOUNDS.csv', dtype=str).set_index('branch')
    for branch in ['ocr', 'stt']:
        w_prop = bounds.loc[branch, 'weighted_proportion']
        ci_lo = bounds.loc[branch, 'ci_low']
        ci_hi = bounds.loc[branch, 'ci_high']
        for token in [w_prop, ci_lo, ci_hi]:
            assert token in texts['docs/input_audit.md'], f'Missing R1.5 bound {token} in input_audit.md'
    for token in ['2,035', '2,506', '26102026', '26102027', '0.7418091717475274', '0.6956329285188523']:
        assert token in texts['docs/input_audit.md'], f'Missing R1.5 design detail {token} in input_audit.md'

    for token in ['frame_identity_match', 'ocr_content_match', 'stt_content_match', 'probability >= threshold', '0.6607533924660753', '0.021046863536667928']:
        assert token in texts['docs/methods_and_reproducibility.md'], f'Missing R1.4/R1.7/R2.4 token {token} in methods_and_reproducibility.md'

    lineage = json.loads((root / 'docs/provenance/revision_20261108/ARCHIVED_CODE_LINEAGE.json').read_text())
    gold_cells_hash = next(x['sha256'] for x in lineage['files'] if x['file'] == 'archived_sources/GOLD_CODE_CELLS.txt')
    nb12_hash = next(x['source_notebook_sha256'] for x in lineage['files'] if x.get('source_notebook') == 'inputs/12_Reviewer_Requested_Only_Colab_V4_EXECUTED.ipynb')
    for expected_hash in [gold_cells_hash, nb12_hash]:
        assert expected_hash in texts['src/reference_archived/README.md'], f'Missing lineage hash {expected_hash} in src/reference_archived/README.md'
        assert expected_hash in texts['src/reference_archived/model_and_fusion_reference.py'], f'Missing lineage hash {expected_hash} in model_and_fusion_reference.py'

    # 3. Verify MANUSCRIPT_ARTIFACT_MANIFEST.csv paths, hashes, and command roles
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

    # 3. Verify both canonical pre-pseudonymisation and lexicographic src_<16hex> bootstrap CIs for R2.5 (Item 8)
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

    # 5. Documentation-vs-artifact consistency check (Item 9)
    verify_documentation_consistency(root)
    print('PASS all V4 revision structural, statistical, and documentation-consistency invariants')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify V4 reviewer revision integration, numerical reproduction, and documentation consistency.')
    parser.add_argument('--root', default='.', help='Repository root path')
    args = parser.parse_args()
    run(args.root)
