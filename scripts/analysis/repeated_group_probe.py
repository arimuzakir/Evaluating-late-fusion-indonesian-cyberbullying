"""Supplementary, explicitly NEW source-group split sensitivity on frozen features.

This does not retrain the original supervised encoders or reproduce the main
21-configuration study. It tests partition dependence of separate lightweight
linear probes and equal-branch probability averaging using label-safe features.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, brier_score_loss

BASE='data/labels/v1_1_GoldAdjudicated/manifests/VIDEO_PRIMARY_MANIFEST.csv'
VIS='data/features/siglip2/siglip2_frozen_features.npz'
AUDIO='data/features/waveform16_v1/audio_waveform16_aligned_3951.npy'
IDS='data/features/waveform16_v1/ALIGNED_ROW_ID_MANIFEST.csv'
SEEDS=[42, 2026, 2027, 1301, 1302]
GRID=np.round(np.arange(0.10,0.90001,0.025),8)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def metrics(y,p,t):
    yhat=(p>=t).astype(int)
    return {'macro_f1':f1_score(y,yhat,average='macro',labels=[0,1],zero_division=0),
            'accuracy':accuracy_score(y,yhat),'roc_auc':roc_auc_score(y,p),
            'average_precision':average_precision_score(y,p),
            'brier':brier_score_loss(y,p),
            'macro_f1_fixed_05':f1_score(y,p>=.5,average='macro',labels=[0,1],zero_division=0)}

def cutoff(y,p):
    vals=[(float(f1_score(y,p>=t,average='macro',labels=[0,1],zero_division=0)),float(t)) for t in GRID]
    best=max(x[0] for x in vals)
    tied=[t for score,t in vals if abs(score-best)<1e-12]
    return sorted(tied,key=lambda t:(abs(t-.5),t))[0]

def _run_single_thread(root,output,seeds=SEEDS):
    root,output=Path(root).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(root/BASE)
    frozen=np.load(root/VIS,allow_pickle=False)
    audio=np.load(root/AUDIO,allow_pickle=False)
    am=pd.read_csv(root/IDS)
    if not (np.array_equal(meta.master_id.values,frozen['ids']) and np.array_equal(meta.master_id.values,am.master_id.values)):
        raise ValueError('Feature IDs and label IDs misaligned')
    if meta.master_id.duplicated().any() or not np.isfinite(audio).all():raise ValueError('Invalid matrix')
    if meta.groupby('image_sha256').group_id.nunique().max()!=1: raise ValueError('Cross-group duplicate image detected')
    x={'visual':np.asarray(frozen['visual'],dtype=np.float64),
       'text':np.asarray(frozen['text'],dtype=np.float64),
       'audio':np.asarray(audio,dtype=np.float64)}
    if any((v.shape[0]!=len(meta) or not np.isfinite(v).all()) for v in x.values()):raise ValueError('Invalid feature values')
    y=meta.label_final.values.astype(int);groups=meta.group_id.values
    rows=[]; preds=[]; roles=[]
    for seed in seeds:
        outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        pool,test=next(outer.split(x['visual'],y,groups))
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed+100000)
        a,b=next(inner.split(x['visual'][pool],y[pool],groups[pool]));fit=pool[a];cal=pool[b]
        sets={'fit':fit,'calibration':cal,'test':test}
        gs=[set(groups[ix]) for ix in sets.values()]
        assert not(gs[0]&gs[1] or gs[0]&gs[2] or gs[1]&gs[2])
        for role,ix in sets.items():
            roles.extend([{'seed':seed,'master_id':meta.master_id.iat[i],'group_id':groups[i],'role':role,
                           'label_final':int(y[i]),'platform':meta.platform.iat[i]} for i in ix])
        # Normalize each source to the same total weight on fitting rows.
        nper=pd.Series(groups[fit]).value_counts()
        weights=np.array([1.0/nper[g] for g in groups[fit]])
        weights=weights/weights.mean()
        pcal={};ptest={}
        for name,feat in x.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(
                C=1.0,solver='liblinear',max_iter=1000,random_state=seed))
            model.fit(feat[fit],y[fit],logisticregression__sample_weight=weights)
            pcal[name]=model.predict_proba(feat[cal])[:,1]
            ptest[name]=model.predict_proba(feat[test])[:,1]
        pcal['equal_frozen_vat']=(pcal['visual']+pcal['audio']+pcal['text'])/3
        ptest['equal_frozen_vat']=(ptest['visual']+ptest['audio']+ptest['text'])/3
        for model,prob in ptest.items():
            thr=cutoff(y[cal],pcal[model]); vals=metrics(y[test],prob,thr)
            rows.append({'seed':seed,'model':model,'fit_n':len(fit),'calibration_n':len(cal),'test_n':len(test),
                         'fit_groups':len(gs[0]),'calibration_groups':len(gs[1]),'test_groups':len(gs[2]),
                         'threshold':thr,**vals})
            preds.extend([{'seed':seed,'model':model,'master_id':meta.master_id.iat[i],'group_id':groups[i],
                           'platform':meta.platform.iat[i],'label_final':int(y[i]),'probability':float(p),
                           'threshold':thr,'prediction':int(p>=thr)} for i,p in zip(test,prob)])
        print(f'Repeated split seed={seed}: n_test={len(test)}, groups_test={len(gs[2])}')
    df=pd.DataFrame(rows);pd.DataFrame(preds).to_csv(output/'REPEATED_SPLIT_TEST_PREDICTIONS.csv',index=False)
    pd.DataFrame(roles).to_csv(output/'REPEATED_SPLIT_ROLE_MANIFEST.csv',index=False)
    df.to_csv(output/'REPEATED_SPLIT_METRICS.csv',index=False)
    summary=df.groupby('model').agg(seeds=('seed','nunique'),macro_f1_mean=('macro_f1','mean'),
                                    macro_f1_sd=('macro_f1','std'),roc_auc_mean=('roc_auc','mean'),
                                    brier_mean=('brier','mean'),test_n_mean=('test_n','mean')).reset_index()
    summary.to_csv(output/'REPEATED_SPLIT_SUMMARY.csv',index=False)
    protocol={
        'status':'COMPLETED_SUPPLEMENTARY_FROZEN_FEATURE_PROBE',
        'comparison_scope':'New lightweight frozen-feature sensitivity ONLY. Not end-to-end retraining or replication of primary supervised fusion; not a contemporaneous task-matched baseline.',
        'population':'Aligned 3951 video windows; v1.1-GoldAdjudicated',
        'outer_split':'First fold of StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); held-out groups never fitted',
        'calibration_split':'First fold of StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=seed+100000), applied only to outer training pool',
        'seeds':list(seeds),'models':['visual','text','audio','equal_frozen_vat'],
        'features':{'visual':'SigLIP2 frozen visual 768D','text':'SigLIP2 frozen text 768D','audio':'Waveform16 16D (15 acoustic features + availability)'},
        'model_training':'StandardScaler fitted on fit role only + LogisticRegression(C=1,liblinear,random_state=seed); inverse-source-size fit weights; no pretrained encoder updates',
        'equal_frozen_vat':'Arithmetic mean of three separately trained frozen-feature branch probabilities',
        'threshold':'Calibration role only: Macro F1 maximum, threshold grid 0.10 to 0.90 by 0.025; ties closest to 0.5 then smaller',
        'caveats':['Repeated source splits overlap and are not independent external validation cohorts',
                   'Repeated partition SD is descriptive, not a confidence interval or hypothesis test',
                   'Frozen pretrained feature transfer does not establish end-to-end model generalisation',
                   'No hyperparameter optimisation on outer test',
                   'This analysis does not address unfiltered standalone-comment model performance'],
        'source_sha256':{f:sha(root/f) for f in [BASE,VIS,AUDIO,IDS]}
    }
    (output/'REPEATED_SPLIT_PROTOCOL.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
    return summary

def run(root,output,seeds=SEEDS):
    """Use one BLAS thread so equal-scored probability ties are reproducible.

    Different BLAS thread counts changed ~3 ROC-AUC/AP values at around
    1e-4 to 6e-4, without changing predictions or Macro F1. This wrapper
    reproduces the archived numerical environment consistently.
    """
    with threadpool_limits(limits=1):
        return _run_single_thread(root,output,seeds)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',default='results/generalization');p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);a=p.parse_args()
    print(run(a.root,a.output,a.seeds).to_string(index=False))
