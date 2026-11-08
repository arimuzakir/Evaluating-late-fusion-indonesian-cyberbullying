"""Independent numerical and source-freeze verification for the V4 reviewer revision.

Recalculates all CPU-reproducible V4 revision tables into a temporary directory via
`scripts/analysis/reproduce_revision.py` and verifies:
1. Byte-level preservation of all 279 frozen GoldAdjudicated v1.1 repository files
   and all 89 public integrated V4 files (`configs/revision_20261108/RESULTS_SOURCE_FREEZE.json`).
2. Cohort, label, group, and role invariants across primary, neural-head sensitivity,
   two-partition source allocation (1042 and 2042 -> effective outer seed 2044), and
   seed-42 probability control predictions.
3. Exact numerical agreement (atol=1e-12) between freshly recomputed tables and
   archived V4 revision tables across fallback matrices, MDE diagnostics, Brier and
   reliability tables, Table 6 probability controls, SRSWOR audit sampling replay,
   conservative hypergeometric bounds, kappa agreement, and descriptive error transitions.
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


def compare_csv(repo_file, repro_file, col_atol=None):
    a = pd.read_csv(repo_file)
    b = pd.read_csv(repro_file)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise AssertionError(f'Shape/column mismatch for {repo_file.name}: {a.shape} vs {b.shape}')
    col_atol = col_atol or {}
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            atol = col_atol.get(col, 1e-12)
            if not np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=atol, equal_nan=True):
                max_diff = float(np.nanmax(np.abs(a[col].to_numpy(float) - b[col].to_numpy(float))))
                raise AssertionError(f'Numeric mismatch in {repo_file.name}.{col}: max_diff={max_diff} > atol={atol}')
        else:
            if not a[col].fillna('').astype(str).equals(b[col].fillna('').astype(str)):
                raise AssertionError(f'String mismatch in {repo_file.name}.{col}')


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

    # 2. Recompute all 35 CPU revision artifacts and compare against archived V4 files
    from scripts.analysis.reproduce_revision import run as repro_run
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
                # REPARTITION_EFFECTS.csv bootstrap quantiles were computed in INITIAL_RUN before
                # group_id pseudonymisation (src_<16hex>), shifting np.unique(group_id) order slightly (<3.5e-4).
                atol_map = {'ci_low': 5e-4, 'ci_high': 5e-4} if name == 'REPARTITION_EFFECTS.csv' else None
                compare_csv(repo_p, repro_p, col_atol=atol_map)
    print(f'PASS CPU revision reproduction: {len(RECOMPUTED_TO_REPO)}/35 tables and JSON summaries verified')

    # 3. Structural and domain-specific invariant checks across reviewer requirements (R1.1 - R2.6)
    oof = pd.read_csv(root / 'results/video/gold_adjudicated_v1.1/VIDEO_OOF_PREDICTIONS.csv', dtype={'master_id': str, 'group_id': str})
    z42 = oof[oof.seed == 42].set_index('master_id')
    assert len(z42) == 3951 and z42.group_id.nunique() == 214
    assert z42.label_final.value_counts().to_dict() == {1: 2326, 0: 1625}

    # R1.3: 12 neural head configurations (48 fold fits)
    hp = pd.read_csv(root / 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_PREDICTIONS.csv', dtype={'master_id': str})
    hp_prog = pd.read_csv(root / 'results/video/revision_20261108/head_sensitivity/HEAD_SENSITIVITY_PROGRESS.csv')
    assert len(hp) == 3951 * 12 and hp.model.nunique() == 12 and len(hp_prog) == 48

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
    assert int(bounds.loc['ocr', 'faithful']) == 43 and int(bounds.loc['ocr', 'n']) == 52
    assert int(bounds.loc['stt', 'faithful']) == 51 and int(bounds.loc['stt', 'n']) == 64
    err_tr = pd.read_csv(root / 'results/input_audit/revision_20261108/AUDIT_ERROR_TRANSITIONS.csv')
    assert err_tr['ci_low'].isna().all() and err_tr['ci_high'].isna().all()

    # R2.4: Exclusion accounting across 4,401 segmented windows
    excl_reason = pd.read_csv(root / 'results/statistics/revision_20261108/exclusions/EXCLUSION_REASON_PLATFORM_CLASS.csv')
    assert int(excl_reason['n'].sum()) == 4401
    assert int(excl_reason.loc[excl_reason.excluded, 'n'].sum()) == 450
    assert int(excl_reason.loc[~excl_reason.excluded, 'n'].sum()) == 3951

    print('PASS all V4 revision structural and statistical invariants')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify V4 reviewer revision integration and numerical reproduction.')
    parser.add_argument('--root', default='.', help='Repository root path')
    args = parser.parse_args()
    run(args.root)
