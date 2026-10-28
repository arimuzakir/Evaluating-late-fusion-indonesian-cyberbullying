"""Compute an item-level keyframe summary from recorded rater and adjudication data.

A numerical re-aggregation requires no new human rating. It does not amend any
human response or authenticate a conflicting historical coordinator aggregate.
That aggregate is retained separately for provenance and excluded from scientific
summary statistics unless its per-item basis becomes available.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd

FIELDS = ['B_agresi_window', 'B_keyframe_event']

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def canonical(value):
    if pd.isna(value) or str(value).strip() == '': return None
    return int(float(value))

def run(root, output):
    root, output = Path(root), Path(output)
    d = root / 'results/input_audit'
    paths = {key:d/f for key,f in [('R1','Form_Audit_R1_scored.csv'),('R2','Form_Audit_R2_scored.csv'),('consensus','Form_Adjudikasi.csv')]}
    a,b,c = (pd.read_csv(paths[k],dtype={'audit_id':str}) for k in ['R1','R2','consensus'])
    assert len(a)==len(b)==100 and a.audit_id.is_unique and b.audit_id.is_unique
    assert set(a.audit_id)==set(b.audit_id)
    a,b = a.set_index('audit_id'),b.set_index('audit_id')
    assert c[['audit_id','field']].duplicated().sum()==0
    cd={(z.audit_id,z.field):canonical(z.nilai_konsensus) for z in c.itertuples(index=False)}
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for item in sorted(a.index):
        v={}
        for field in FIELDS:
            v[field+'_r1']=canonical(a.at[item,field]);v[field+'_r2']=canonical(b.at[item,field]);
            if v[field+'_r1']!=v[field+'_r2'] and (item,field) not in cd:
                raise AssertionError(f'Unresolved rater disagreement: {item} {field}')
            v[field+'_row_consensus']=cd.get((item,field),v[field+'_r1'])
        v['audit_id']=item
        y=v['B_agresi_window_row_consensus'];k=v['B_keyframe_event_row_consensus']
        assert y in (0,1)
        if y==1: assert k in (0,1,2),f'Positive window needs a keyframe rating: {item}'
        if y==0: assert k is None,f'Non-positive window has keyframe rating: {item}'
        rows.append(v)
    df=pd.DataFrame(rows)[['audit_id']+[f'{f}_{s}' for f in FIELDS for s in ('r1','r2','row_consensus')]]
    pos=df[df.B_agresi_window_row_consensus==1]
    count={str(i):int((pos.B_keyframe_event_row_consensus==i).sum()) for i in [2,1,0]}
    # Archive-only comparison: never overwrite observed row-level ratings to fit it.
    legacy_file = root / 'results/statistics/CURRENT_AUDIT_RECORD_CHECK.json'
    legacy = json.loads(legacy_file.read_text())['coordinator_reported_keyframe_counts']
    reported = {str(i): int(legacy[f'score{i}']) for i in (2,1,0)}
    summary={
        'source':'Stored R1, R2 and adjudication item-level records; numerical re-aggregation only',
        'status':'REPRODUCIBLE_ITEM_LEVEL_SUMMARY',
        'n_audit':len(df),
        'n_positive_row_consensus':len(pos),
        'row_level_counts':count,
        'reporting_basis':'Recorded per-item scores with adjudication taking precedence for recorded disagreements',
        'reporting_scope':'Descriptive distribution within the 55 consensus-positive audited windows only',
        'historical_coordinator_aggregate_for_provenance_only':reported,
        'difference_row_minus_historical':{key:count[key]-reported[key] for key in count},
        'historical_difference_resolved_by_new_adjudication':False,
        'any_rater_score_modified':False,
        'new_human_adjudication_performed':False,
        'reporting_policy':'Use derived row-level counts for descriptive summaries; do not use historical coordinator aggregate as an alternative result',
        'source_sha256':{key:digest(path) for key,path in paths.items()}
    }
    df.to_csv(output/'KEYFRAME_ITEM_LEVEL_RECONCILIATION.csv',index=False,na_rep='')
    (output/'KEYFRAME_RECONCILIATION_STATUS.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(f"Keyframe row-level summary: {count} from {len(pos)} audited positives; archived aggregate excluded from reporting")
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.');p.add_argument('--output',default='results/input_audit');args=p.parse_args(); run(args.root,args.output)
