"""Read-only hash and metric checks for the journal numerical release."""
from pathlib import Path
import argparse,hashlib,json,tempfile,sys
import pandas as pd
import numpy as np

MANIFEST='docs/RELEASE_SHA256.json'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for v in iter(lambda:f.read(1<<20),b''): h.update(v)
    return h.hexdigest()

def equal_table(old,new,col_atol=None):
    a,b=pd.read_csv(old),pd.read_csv(new)
    if not a.columns.equals(b.columns) or a.shape!=b.shape:raise AssertionError(f'Table shape mismatch: {old} {a.shape} {b.shape}')
    col_atol=col_atol or {}
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            atol=col_atol.get(col,1e-8)
            if not np.allclose(a[col].to_numpy(float),b[col].to_numpy(float),rtol=1e-7,atol=atol,equal_nan=True):
                raise AssertionError(f'Numerical mismatch: {old.name} {col}')
        elif not a[col].fillna('').astype(str).equals(b[col].fillna('').astype(str)):
            raise AssertionError(f'Text mismatch: {old.name} {col}')

def run(root,full=False):
    root=Path(root).resolve()
    manifest=json.loads((root/MANIFEST).read_text())
    files=manifest['files']
    absent=[name for name in files if not (root/name).is_file()]
    mismatches=[name for name,d in files.items() if (root/name).is_file() and sha(root/name)!=d['sha256']]
    if absent or mismatches:raise AssertionError(f'Checksum/entry mismatch missing={absent[:15]} changed={mismatches[:15]}')
    print(f'PASS SHA-256: {len(files)} files')
    with tempfile.TemporaryDirectory() as td:
        # Core 27 checks use public-view numeric labels/roles/predictions, write only to temporary output.
        sys.path.insert(0,str(root))
        from src.evaluate_release import run as run_main
        r=run_main(root,output_dir=Path(td)/'main')
        assert r['failed']==0 and r['passed']==27
        print('PASS frozen numerical reproduction: 27/27')
        from scripts.validation.reconcile_keyframes import run as run_keyframe
        status=run_keyframe(root,Path(td)/'keyframe')
        assert status['row_level_counts']=={'2':46,'1':9,'0':0}
        assert status['status']=='REPRODUCIBLE_ITEM_LEVEL_SUMMARY'
        assert status['any_rater_score_modified'] is False
        assert status['new_human_adjudication_performed'] is False
        assert status['historical_difference_resolved_by_new_adjudication'] is False
        equal_table(root/'results/input_audit/KEYFRAME_ITEM_LEVEL_RECONCILIATION.csv',Path(td)/'keyframe/KEYFRAME_ITEM_LEVEL_RECONCILIATION.csv')
        print('PASS keyframe row-level re-aggregation; archived aggregate retained separately')
        if full:
            from scripts.analysis.lopo_cluster_intervals import run as lopo_run
            from scripts.analysis.repeated_group_probe import run as group_run
            lopo_run(root,Path(td)/'lopo')
            equal_table(root/'results/generalization/LOPO_CLUSTER_BOOTSTRAP_CI.csv',Path(td)/'lopo/LOPO_CLUSTER_BOOTSTRAP_CI.csv')
            print('PASS LOPO 20-contrast conditional bootstrap reproduction')
            group_run(root,Path(td)/'repeated')
            blas_tie_atol={'roc_auc':1e-3,'average_precision':1e-3,'brier':1e-4,'roc_auc_mean':1e-3,'brier_mean':1e-4,'probability':1e-3}
            for fn in ['REPEATED_SPLIT_METRICS.csv','REPEATED_SPLIT_SUMMARY.csv','REPEATED_SPLIT_TEST_PREDICTIONS.csv','REPEATED_SPLIT_ROLE_MANIFEST.csv']:
                equal_table(root/'results/generalization'/fn,Path(td)/'repeated'/fn,col_atol=blas_tie_atol)
            print('PASS five-seed frozen-feature partition probe reproduction')
    print('Package PASS (historical audit summary provenance and external validity restrictions documented).')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='.');a.add_argument('--full',action='store_true');args=a.parse_args();run(args.root,args.full)
