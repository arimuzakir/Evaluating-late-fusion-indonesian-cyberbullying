"""Recalculate LOPO scores and descriptive cluster bootstrap intervals on saved predictions.

This analysis is conditional on the released fitted frozen-feature probes.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score

P='results/video/gold_adjudicated_v1.1/'
MODELS=['siglip2_visual_probe','siglip2_vt_probe','siglip2_vta_probe','siglip2_vta_coverage_probe']

def f1_from(c):
    tn,fp,fn,tp=[c[...,i] for i in range(4)]
    p=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)!=0)
    n=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn,dtype=float),where=(2*tn+fp+fn)!=0)
    return (p+n)/2

def run(root,output,reps=20000,seed=27062026):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(root/(P+'LOPO_FROZEN_FEATURE_PROBE_PREDICTIONS.csv'))
    ref=pd.read_csv(root/(P+'LOPO_FROZEN_FEATURE_PROBE_METRICS.csv'))
    role=pd.read_csv(root/(P+'LOPO_ROLE_MANIFEST.csv'))
    main=pd.read_csv(root/'data/labels/v1_1_GoldAdjudicated/manifests/VIDEO_PRIMARY_MANIFEST.csv').set_index('master_id')
    for fold,x in role.groupby('held_out_platform'):
        if x.groupby('group_id').role.nunique().max()!=1:raise AssertionError('LOPO group leakage for '+fold)
    assert pred.groupby(['master_id','model']).size().max()==1
    assert pred.master_id.map(main.label_final).eq(pred.label_final).all()
    assert pred.master_id.map(main.group_id).eq(pred.group_id).all()
    rng=np.random.default_rng(seed)
    out=[]
    for platform in sorted(pred.platform.unique()):
        base=pred[(pred.platform==platform)&(pred.model==MODELS[0])].copy().sort_values('master_id')
        groups=sorted(base.group_id.unique());n=len(groups)
        if n<2:raise ValueError('Insufficient clusters '+platform)
        draws=rng.integers(0,n,size=(reps,n))
        boot={};points={}
        for model in MODELS:
            d=pred[(pred.platform==platform)&(pred.model==model)].sort_values('master_id')
            assert np.array_equal(d.master_id.values,base.master_id.values)
            assert np.array_equal(d.label_final.values,base.label_final.values)
            tab=[]
            for group in groups:
                g=d[d.group_id==group]
                y,p=g.label_final.to_numpy(),g.prediction.to_numpy()
                tab.append([int(((y==0)&(p==0)).sum()),int(((y==0)&(p==1)).sum()),
                            int(((y==1)&(p==0)).sum()),int(((y==1)&(p==1)).sum())])
            tab=np.array(tab,dtype=int)
            val=float(f1_score(d.label_final,d.prediction,average='macro'))
            refval=ref[(ref.held_out_platform==platform)&(ref.model==model)].macro_f1
            if len(refval)!=1 or abs(refval.iloc[0]-val)>1e-10:raise AssertionError('LOPO metric mismatch')
            sampled=tab[draws].sum(axis=1);distribution=f1_from(sampled)
            boot[model]=distribution;points[model]=val
            out.append({'held_out_platform':platform,'model':model,'n_windows':len(d),
                        'n_groups':n,'positive_n':int((d.label_final==1).sum()),
                        'macro_f1':val,'cluster_ci_low':float(np.quantile(distribution,.025)),
                        'cluster_ci_high':float(np.quantile(distribution,.975)),
                        'delta_vs_visual':np.nan,'delta_cluster_ci_low':np.nan,'delta_cluster_ci_high':np.nan,
                        'bootstrap_replicates':reps})
        for row in out:
            if row['held_out_platform']!=platform:continue
            model=row['model'];diff=boot[model]-boot[MODELS[0]]
            row['delta_vs_visual']=points[model]-points[MODELS[0]]
            row['delta_cluster_ci_low']=float(np.quantile(diff,.025))
            row['delta_cluster_ci_high']=float(np.quantile(diff,.975))
    pd.DataFrame(out).to_csv(output/'LOPO_CLUSTER_BOOTSTRAP_CI.csv',index=False)
    info={'status':'VERIFIED_EXISTING_LOPO_PREDICTIONS_WITH_NEW_DESCRIPTIVE_CLUSTER_INTERVALS',
          'reps':reps,'seed':seed,'models':MODELS,'platforms':sorted(pred.platform.unique()),
          'n_records':len(pred),'number_of_models':len(MODELS),
          'description':'Resamples source group IDs within each held-out platform. No model refitted. Paired differences use the same group draws. Conditional uncertainty only.',
          'limitations':['Existing LOPO probes use frozen pretrained features, not end-to-end encoder retraining',
                         'Source group can span platforms; the released LOPO role manifest has source purge checks',
                         'Within-platform source-group resampling is not repeated external independent validation',
                         'Subgroup CI does not account for model selection, training or data acquisition uncertainty'],
          'source_sha256':{x:hashlib.sha256((root/(P+x)).read_bytes()).hexdigest() for x in ['LOPO_FROZEN_FEATURE_PROBE_PREDICTIONS.csv','LOPO_FROZEN_FEATURE_PROBE_METRICS.csv','LOPO_ROLE_MANIFEST.csv']}}
    (output/'LOPO_INTERVAL_PROTOCOL.json').write_text(json.dumps(info,indent=2)+'\n')
    return pd.DataFrame(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',default='results/generalization');p.add_argument('--reps',type=int,default=20000); a=p.parse_args();
    print(run(a.root,a.output,a.reps).to_string(index=False))
