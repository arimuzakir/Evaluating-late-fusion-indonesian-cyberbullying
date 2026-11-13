"""Independent verification of final manuscript Tables 1-6, Figures 1-6, S1-S8, and S7 reliability.

Verifies without modifying any frozen primary files:
1. Dynamic recomputation of manuscript Tables 1-6 via `scripts/analysis/build_manuscript_artifacts.py`
   and exact numerical agreement (`atol = 1e-12`) against frozen public CSV/JSON tables.
2. Presence, PNG validity, and SHA-256 verification of `results/figures/Fig_1.png`,
   `results/figures/Fig_2.png`, and `results/figures/manuscript/Fig_3.png..Fig_6.png`,
   plus absence of `.drawio` source files in the public repository.
3. Temporary-directory reproduction of `results/statistics/revision_20261113/reliability/reproduce_reliability.py`
   and exact match against `RELIABILITY_POINTWISE_CI.csv` (`atol = 1e-12`) and `VALIDATION.json`.
4. Presence of S2/S8 provenance artifacts, `docs/controlled_access.md`, and full Table 1-6 / Fig. 1-6 /
   S1-S8 coverage with valid SHA-256 hashes in `docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv`.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


EXPECTED_MANUAL_FIGURES = {
    'results/figures/Fig_1.png': '5a2145c99127310215df17553f24f2219658355e883e609269a41a742cfa069e',
    'results/figures/Fig_2.png': 'eeae9d409efe2b264068562c5e35f3d65260920236f2386e719ebfef13f693c1',
}

EXPECTED_TABLE_ROWS = {
    'Table_1': 6,
    'Table_2': 5,
    'Table_3': 22,
    'Table_4': 5,
    'Table_5': 3,
    'Table_6': 5,
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _compare_frames(a, b, label, atol=1e-12):
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        raise AssertionError(f'Shape/column mismatch for {label}: {a.shape} vs {b.shape}')
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            if not np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), rtol=0, atol=atol, equal_nan=True):
                raise AssertionError(f'Numeric mismatch in {label}.{col}')
        else:
            if not a[col].fillna('').astype(str).equals(b[col].fillna('').astype(str)):
                raise AssertionError(f'String mismatch in {label}.{col}')


def verify_tables(root):
    sys.path.insert(0, str(root))
    from scripts.analysis.build_manuscript_artifacts import build_all_tables

    tables = build_all_tables(root)
    for name, exp_rows in EXPECTED_TABLE_ROWS.items():
        if name not in tables or len(tables[name]) != exp_rows:
            raise AssertionError(f'{name} expected {exp_rows} rows, got {len(tables.get(name, []))}')

    # Compare each actual export against a fresh builder result, not a differently shaped source summary.
    for name, table in tables.items():
        _compare_frames(table, pd.read_csv(root / 'results/manuscript' / (name + '.csv')), name, atol=1e-12)

    # Cross-check Table 3 against TABLE3_RECOMPUTED.csv + STRATIFIED_RANDOM_SUMMARY.json
    t3 = tables['Table_3'].set_index('model')
    t3_all = pd.read_csv(root / 'results/statistics/revision_20261108/main_result_checks/TABLE3_RECOMPUTED.csv')
    t3_ref = t3_all[t3_all['seed'] == 42].set_index('model')
    for model_key, row in t3_ref.iterrows():
        if abs(float(t3.loc[model_key, 'macro_f1_selected']) - float(row['pooled_macro_f1'])) > 1e-12:
            raise AssertionError(f'Table 3 selected Macro F1 mismatch for {model_key}')
        if abs(float(t3.loc[model_key, 'macro_f1_fixed_050']) - float(row['fixed_05_macro_f1'])) > 1e-12:
            raise AssertionError(f'Table 3 fixed-0.50 Macro F1 mismatch for {model_key}')

    # Cross-check Table 6 against TABLE6_PROBABILITY_METRICS.csv
    t6 = tables['Table_6'].set_index('model')
    t6_ref = pd.read_csv(root / 'results/statistics/revision_20261108/probability/TABLE6_PROBABILITY_METRICS.csv').set_index('model')
    for model_key in t6_ref.index:
        for col in ['window_macro_f1', 'window_brier', 'window_ece', 'source_brier', 'source_ece']:
            if abs(float(t6.loc[model_key, col]) - float(t6_ref.loc[model_key, col])) > 1e-12:
                raise AssertionError(f'Table 6 mismatch for {model_key}.{col}')

    print('PASS manuscript Tables 1-6 verified against frozen numerical sources (atol=1e-12)')
    return tables


def verify_figures(root):
    drawio = [str(p.relative_to(root)) for p in root.rglob('*.drawio') if '.git' not in p.parts]
    if drawio:
        raise AssertionError(f'Editable .drawio files must not be present in public repo: {drawio}')

    for rel, exp_sha in EXPECTED_MANUAL_FIGURES.items():
        p = root / rel
        if not p.is_file() or sha256(p) != exp_sha:
            raise AssertionError(f'Missing or modified manual author diagram: {rel}')

    for i in range(3, 7):
        fig_p = root / f'results/figures/manuscript/Fig_{i}.png'
        if not fig_p.is_file() or fig_p.stat().st_size < 10000:
            raise AssertionError(f'Missing or empty composite manuscript figure: {fig_p}')
        with fig_p.open('rb') as f:
            sig = f.read(8)
        if sig != b'\x89PNG\r\n\x1a\n':
            raise AssertionError(f'Invalid PNG signature for {fig_p}')

    print('PASS manuscript Figures 1-6 verified (manual Fig_1-2 hashes + composite Fig_3-6 PNGs)')


def verify_s7_reliability(root):
    rel_dir = root / 'results/statistics/revision_20261113/reliability'
    script = rel_dir / 'reproduce_reliability.py'
    if not script.is_file():
        raise AssertionError('Missing S7 reproduce_reliability.py')

    val_info = json.loads((rel_dir / 'VALIDATION.json').read_text(encoding='utf-8'))
    expected_top = {
        'rows': 100,
        'groups': 214,
        'windows': 3951,
        'direct_resampling_checks': 6,
        'point_estimates_match_existing': True,
    }
    for k, v in expected_top.items():
        if val_info.get(k) != v:
            raise AssertionError(f'S7 VALIDATION.json mismatch for {k}: {val_info.get(k)} != {v}')
    ci_counts = val_info.get('ci_status_counts', {})
    for k, v in {'pointwise_ci': 68, 'empty': 20, 'sparse_no_ci': 12}.items():
        if ci_counts.get(k) != v:
            raise AssertionError(f'S7 VALIDATION.json ci_status_counts mismatch for {k}: {ci_counts.get(k)} != {v}')

    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(script), '--output', td]
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(f'reproduce_reliability.py failed:\n{proc.stderr}\n{proc.stdout}')
        ref_csv = pd.read_csv(rel_dir / 'RELIABILITY_POINTWISE_CI.csv')
        rep_csv = pd.read_csv(Path(td) / 'RELIABILITY_POINTWISE_CI.csv')
        _compare_frames(ref_csv, rep_csv, 'RELIABILITY_POINTWISE_CI.csv', atol=1e-12)

    print('PASS S7 reliability reproduction verified (100 rows, 68 intervals, 12 sparse, 20 empty, atol=1e-12)')


def verify_governance_and_manifest(root):
    ca_path = root / 'docs/controlled_access.md'
    if not ca_path.is_file():
        raise AssertionError('Missing docs/controlled_access.md')
    ca_text = ca_path.read_text(encoding='utf-8')
    for tok in ['PROPOSED', 'arimuzakir@binadarma.ac.id', '010/DRPM/Univ-BD/IX/2026', '021/KET-DRPM/Univ-BD/XI/2026']:
        if tok not in ca_text:
            raise AssertionError(f'docs/controlled_access.md missing required token: {tok}')

    for req_rel in [
        'docs/feature_interactions.md',
        'docs/provenance/revision_20261108/CONTROL_PROTOCOL_LOCK_VERIFIED.json',
        'docs/provenance/revision_20261108/RUN_LOCK.json',
        'docs/provenance/revision_20261108/RESEARCH_CHRONOLOGY.csv',
    ]:
        if not (root / req_rel).is_file():
            raise AssertionError(f'Missing required S2/S8 artifact: {req_rel}')

    manifest_path = root / 'docs/reproduction/MANUSCRIPT_ARTIFACT_MANIFEST.csv'
    manifest_text = manifest_path.read_text(encoding='utf-8')
    if 'AWAIT_FINAL_MANUSCRIPT_MAPPING' in manifest_text:
        raise AssertionError('MANUSCRIPT_ARTIFACT_MANIFEST.csv still contains AWAIT_FINAL_MANUSCRIPT_MAPPING')

    manifest = pd.read_csv(manifest_path)
    required_ids = {
        'MS_TABLE_1_CORPUS_POPULATION',
        'MS_TABLE_2_COMMENT_BENCHMARK',
        'MS_TABLE_3_VIDEO_MAIN_21_MODELS',
        'MS_TABLE_4_LOPO_FROZEN_VTA',
        'MS_TABLE_5_BLINDED_AUDIT_AGREEMENT',
        'MS_TABLE_6_PROBABILITY_QUALITY',
        'MS_FIG_1_CORPUS_POPULATION_FLOW',
        'MS_FIG_2_MODALITY_FUSION_PROTOCOL',
        'MS_FIG_3_COMMENT_BENCHMARK_AND_PLATFORMS',
        'MS_FIG_4_VIDEO_CONFUSION_MATRICES',
        'MS_FIG_5_FOCAL_CONTRASTS_AND_SEEDS',
        'MS_FIG_6_AVAILABILITY_AND_THRESHOLD_EFFECTS',
        'SUPP_S1_THRESHOLDS_AND_ROLES',
        'SUPP_S2_FEATURE_AND_MODEL_INVENTORY',
        'SUPP_S3_AUDIT_BOUNDS_AND_TRANSITIONS',
        'SUPP_S6_HEAD_SENSITIVITY',
        'SUPP_S1_NO_TEXT_FALLBACK',
        'SUPP_S4_EXCLUSIONS_4401',
        'SUPP_S5_INFERENCE_AND_ROUNDING',
        'SUPP_S6_TWO_SOURCE_PARTITIONS',
        'SUPP_S7_RELIABILITY_POINTWISE_CI',
        'SUPP_S8_CHRONOLOGY_AND_LOCKS',
    }
    missing_ids = required_ids - set(manifest['artifact_id'])
    if missing_ids:
        raise AssertionError(f'MANUSCRIPT_ARTIFACT_MANIFEST.csv missing required rows: {sorted(missing_ids)}')

    for row in manifest.itertuples():
        out_p = root / row.output_path
        if not out_p.is_file() or sha256(out_p) != row.output_sha256:
            raise AssertionError(f'Output SHA mismatch for {row.artifact_id}: {row.output_path}')
        for pair in str(row.input_sha256).split(';'):
            rel_in, exp_h = pair.strip().rsplit(':', 1)
            in_p = root / rel_in.strip()
            if not in_p.is_file() or sha256(in_p) != exp_h.strip():
                raise AssertionError(f'Input SHA mismatch for {row.artifact_id}: {rel_in}')
        if row.command_role == 'MANUAL_DIAGRAM_EXPORT_VERIFIED_AGAINST_TABLES':
            if row.script_sha256 != 'MANUAL_AUTHOR_DIAGRAM_EXPORT':
                raise AssertionError(f'Expected MANUAL_AUTHOR_DIAGRAM_EXPORT for {row.artifact_id}')
        else:
            sp = root / row.script_path
            if not sp.is_file() or sha256(sp) != row.script_sha256:
                raise AssertionError(f'Script SHA mismatch for {row.artifact_id}: {row.script_path}')

    print(f'PASS manifest & governance check ({len(manifest)} artifact rows verified)')


def run(root):
    root = Path(root).resolve()
    verify_tables(root)
    verify_figures(root)
    verify_s7_reliability(root)
    verify_governance_and_manifest(root)
    print('PASS all final manuscript Table 1-6, Fig. 1-6, and Supplementary S1-S8 artifact checks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify manuscript Tables 1-6, Figures 1-6, S1-S8, and reliability CI reproduction.')
    parser.add_argument('--root', default='.', help='Repository root path')
    args = parser.parse_args()
    run(args.root)
