"""Reproduce V6 seed42 and run two protocol-locked probability controls on CPU.

Inputs are immutable V6 predictions, role manifests, branch caches and models.
No original source file is changed. See docs/CONTROL_PROTOCOL_LOCK.json.
"""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, platform, warnings
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, log_loss
from joblib.numpy_pickle import NumpyUnpickler

BRANCHES = ['visual', 'audio', 'text']
BASELINES = BRANCHES + ['equal_vat', 'vat_lr']
CONTROLS = ['sigmoid_equal_vat', 'convex_vat']
GRID = np.round(.1 + .025 * np.arange(33), 3)
ALLOWED = {('sklearn.pipeline','Pipeline'),
 ('sklearn.preprocessing._data','StandardScaler'),
 ('sklearn.linear_model._logistic','LogisticRegression'),
 ('joblib.numpy_pickle','NumpyArrayWrapper'), ('numpy','ndarray'),
 ('numpy','dtype'), ('numpy.core.multiarray','scalar'),
 ('numpy._core.multiarray','scalar')}

class RestrictedModelReader(NumpyUnpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise ValueError(f'Unsupported model class {module}.{name}')
        return super().find_class(module, name)

def load_model(path):
    with path.open('rb') as stream:
        return RestrictedModelReader(str(path), stream,
                                     ensure_native_byte_order=True).load()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def group_weights(groups):
    counts = Counter(groups)
    w = np.array([1/counts[g] for g in groups], dtype=float)
    return w/w.mean()

def ece(y, p, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w)
    ix = np.minimum((np.asarray(p)*10).astype(int), 9)
    return float(sum(abs(np.sum(w[ix==b]*(p[ix==b]-y[ix==b])))
                     for b in range(10))/w.sum())

def threshold(y, p):
    scores = np.array([f1_score(y, p>=t, average='macro', zero_division=0)
                       for t in GRID])
    best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    i = min(best, key=lambda j: (abs(GRID[j]-.5), GRID[j]))
    return float(GRID[i])

def metrics(y, p, groups, pred=None):
    w = group_weights(groups)
    out = {'n':len(y), 'groups':len(set(groups)),
      'prevalence':float(y.mean()), 'group_weighted_prevalence':float(np.average(y,weights=w)),
      'mean_probability':float(p.mean()), 'group_weighted_mean_probability':float(np.average(p,weights=w)),
      'brier':float(np.mean((p-y)**2)), 'ece10':ece(y,p),
      'group_weighted_brier':float(np.average((p-y)**2,weights=w)),
      'group_weighted_ece10':ece(y,p,w),
      'log_loss':float(log_loss(y,p,labels=[0,1])),
      'roc_auc':float(roc_auc_score(y,p)),
      'average_precision':float(average_precision_score(y,p)),
      'macro_f1_fixed05':float(f1_score(y,p>=.5,average='macro'))}
    for q, v in zip(['min','q05','q25','median','q75','q95','max'],
                    np.quantile(p,[0,.05,.25,.5,.75,.95,1])):
        out['score_'+q] = float(v)
    if pred is not None:
        out.update(macro_f1=float(f1_score(y,pred,average='macro')),
          accuracy=float(np.mean(y==pred)),
          false_positives=int(np.sum((y==0)&(pred==1))),
          false_negatives=int(np.sum((y==1)&(pred==0))))
    return out

def fit_sigmoid(p, y, groups):
    w = group_weights(groups); w = w/w.sum()
    def objective(v):
        z = v[0]*p + v[1]
        loss = np.sum(w*(np.logaddexp(0,z)-y*z))
        delta = w*(expit(z)-y)
        return float(loss), np.array([np.sum(delta*p),np.sum(delta)])
    initial = np.array([1.,logit(np.clip(np.average(y,weights=w),1e-6,1-1e-6))-.5])
    opt = minimize(objective,initial,jac=True,method='L-BFGS-B',
                   bounds=[(0,None),(None,None)],
                   options={'maxiter':2000,'ftol':1e-14,'gtol':1e-9})
    assert opt.success, opt.message
    a,b = map(float,opt.x)
    grad = objective(opt.x)[1]
    residual = max(abs(grad[1]),abs(grad[0]) if a>1e-7 else max(0,-grad[0]))
    assert residual<1e-5
    return a,b,float(opt.fun),float(residual)

def fit_convex(x, y, groups):
    w = group_weights(groups); w = w/w.sum(); uniform=np.full(3,1/3)
    def objective(v):
        delta=x@v-y
        return float(np.sum(w*delta**2)), 2*x.T@(w*delta)
    opt=minimize(objective,uniform,jac=True,method='SLSQP',bounds=[(0,1)]*3,
       constraints=[{'type':'eq','fun':lambda v:v.sum()-1,
                     'jac':lambda v:np.ones(3)}],
       options={'ftol':1e-12,'maxiter':2000})
    assert opt.success,opt.message
    v=opt.x
    assert np.min(v)>-1e-9 and abs(v.sum()-1)<1e-9
    assert objective(v)[0]<=objective(uniform)[0]+1e-10
    grad=objective(v)[1]; active=v>1e-7; lam=-float(grad[active].mean())
    residual=max(float(np.max(np.abs(grad[active]+lam))),
                 max(0.,float(-np.min(grad[~active]+lam))) if (~active).any() else 0.)
    assert residual<1e-5
    return v,float(opt.fun),float(objective(uniform)[0]),residual

def run(data_root,cache_root,output,protocol):
    root=Path(data_root);cache=Path(cache_root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    result=root/'results/video/gold_adjudicated_v1.1'
    paths={'oof':result/'VIDEO_OOF_PREDICTIONS.csv',
           'roles':result/'ROLE_MANIFEST.csv','evaluation':result/'EVALUATION_MANIFEST.csv',
           'audio':root/'data/features/waveform16_v1/audio_waveform16_aligned_3951.npy',
           'audio_rows':root/'data/features/waveform16_v1/ALIGNED_ROW_ID_MANIFEST.csv',
           'protocol':Path(protocol)}
    locked=json.loads(paths['protocol'].read_text())
    assert sha(paths['oof'])==locked['oof_sha256']
    d=pd.read_csv(paths['evaluation']);rr=pd.read_csv(paths['roles'])
    oof=pd.read_csv(paths['oof']).query('seed==42').set_index('master_id')
    assert len(d)==len(oof)==3951 and d.group_id.nunique()==214
    assert d.master_id.is_unique and oof.index.is_unique
    assert set(d.master_id)==set(oof.index)
    yy=d.label_final.to_numpy(int);groups=d.group_id.to_numpy()
    expected=oof.loc[d.master_id]
    assert np.array_equal(expected.label_final.to_numpy(),yy)
    assert np.array_equal(expected.group_id.to_numpy(),groups)
    ar=pd.read_csv(paths['audio_rows']);audio=np.load(paths['audio'],allow_pickle=False)
    assert np.array_equal(ar.master_id.to_numpy(),d.master_id.to_numpy())
    assert audio.shape==(3951,16) and np.isfinite(audio).all()
    role_rows=[];coef_rows=[];metric_rows=[];cal_rows=[];pool_rows=[]
    threshold_rows=[];prob_rows=[];check_rows=[];bin_rows=[];class_rows=[]
    models=BASELINES+CONTROLS+['sigmoid_visual','sigmoid_audio','sigmoid_text']
    for fold in range(4):
        roles=rr.query('outer_fold==@fold').set_index('master_id')
        assert roles.index.is_unique and set(roles.index)==set(d.master_id)
        roles=roles.loc[d.master_id].role.to_numpy()
        names=['fit','early_stop','meta_fit','calibration','test']
        assert set(roles)==set(names)
        group_sets={r:set(groups[roles==r]) for r in names}
        assert sum(len(s) for s in group_sets.values())==len(set(groups))
        assert np.array_equal(roles=='test',d.outer_fold.to_numpy()==fold)
        for role in names:
            ix=roles==role;w=group_weights(groups[ix])
            role_rows.append({'fold':fold,'role':role,'windows':int(ix.sum()),
               'groups':len(group_sets[role]),'negative':int(np.sum(yy[ix]==0)),
               'positive':int(np.sum(yy[ix]==1)), 'positive_fraction':float(yy[ix].mean()),
               'group_weighted_positive_fraction':float(np.average(yy[ix],weights=w))})
        p={};test=roles=='test';meta=roles=='meta_fit';cal=roles=='calibration'
        for branch in ['visual','text']:
            path=cache/f'fold_{fold}/{branch}_branch.npz';paths[f'{fold}_{branch}']=path
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['ids'],d.master_id.to_numpy())
                p[branch]=z['probability'].astype(float)
        present=d.text_present.to_numpy()==1
        assert np.array_equal(present,(d.ocr_ok.to_numpy()==1)|(d.stt_ok.to_numpy()==1))
        p['text'][~present]=.5
        audio_path=cache/f'fold_{fold}/audio_model.joblib';paths[f'{fold}_audio_model']=audio_path
        model_path=cache/f'fold_{fold}/vat_lr.joblib';paths[f'{fold}_vat_lr']=model_path
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',sklearn.exceptions.InconsistentVersionWarning)
            aud=load_model(audio_path);stack=load_model(model_path)
        p['audio']=aud.predict_proba(audio)[:,1]
        x=np.column_stack([p[b] for b in BRANCHES])
        p['equal_vat']=x.mean(axis=1);p['vat_lr']=stack.predict_proba(x)[:,1]
        scaler,lr=stack[0],stack[-1]
        assert lr.n_features_in_==3 and float(lr.C)==.2
        coef=lr.coef_[0];raw=coef/scaler.scale_
        coef_rows.append({'fold':fold, **{f'coef_standardized_{b}':float(v) for b,v in zip(BRANCHES,coef)},
            'intercept_standardized':float(lr.intercept_[0]),
            **{f'coef_raw_{b}':float(v) for b,v in zip(BRANCHES,raw)},
            'intercept_raw':float(lr.intercept_[0]-np.dot(raw,scaler.mean_)),
            'C':float(lr.C),'penalty':'L2','fit_weighting':'inverse group size',
            **{f'scaler_mean_{b}':float(v) for b,v in zip(BRANCHES,scaler.mean_)},
            **{f'scaler_scale_{b}':float(v) for b,v in zip(BRANCHES,scaler.scale_)}})
        # Verify all baseline predictions BEFORE running either new control.
        for name in BASELINES:
            stored=oof.loc[d.loc[test,'master_id']]
            err=float(np.max(np.abs(p[name][test]-stored[name+'_probability'].to_numpy())))
            assert err<1e-6,(fold,name,err)
            t=threshold(yy[cal],p[name][cal]);stored_t=stored[name+'_threshold'].unique()
            assert len(stored_t)==1 and abs(t-stored_t[0])<1e-10,(fold,name,t,stored_t)
            assert np.array_equal((p[name][test]>=t).astype(int),stored[name+'_prediction'])
            check_rows.append({'fold':fold,'model':name,'max_test_probability_abs_difference':err,
                               'threshold_reproduced':True,'decisions_reproduced':True})
        for branch in BRANCHES:
            fitting=meta&present if branch=='text' else meta
            a,b,loss,residual=fit_sigmoid(p[branch][fitting],yy[fitting],groups[fitting])
            pp=expit(a*p[branch]+b)
            if branch=='text':pp[~present]=.5
            p['sigmoid_'+branch]=pp
            cal_rows.append({'fold':fold,'branch':branch,'slope':a,'intercept':b,
               'fit_n':int(fitting.sum()),'fit_groups':len(set(groups[fitting])),
               'weighted_fit_log_loss':loss,'optimizer_kkt_residual':residual,
               'text_fallback_preserved':bool(np.all(pp[~present]==.5)) if branch=='text' else True})
        p['sigmoid_equal_vat']=np.column_stack([p['sigmoid_'+b] for b in BRANCHES]).mean(axis=1)
        v,loss,uniform_loss,residual=fit_convex(x[meta],yy[meta],groups[meta])
        p['convex_vat']=x@v
        pool_rows.append({'fold':fold,**{f'weight_{b}':float(z) for b,z in zip(BRANCHES,v)},
             'weighted_meta_brier':loss,'uniform_weighted_meta_brier':uniform_loss,
             'fit_gain':uniform_loss-loss,'optimizer_kkt_residual':residual,
             'uniform_feasible':True,'objective_no_worse_than_uniform':True})
        chosen={name:threshold(yy[cal],p[name][cal]) for name in models}
        for name in models:
            assert np.isfinite(p[name]).all() and np.min(p[name])>=0 and np.max(p[name])<=1
            threshold_rows.append({'fold':fold,'model':name,'threshold':chosen[name],
                 'source_role':'calibration','macro_f1_on_selection_role':float(f1_score(yy[cal],p[name][cal]>=chosen[name],average='macro'))})
        for role in ['meta_fit','calibration','test']:
            ix=roles==role
            frame=d.loc[ix,['master_id','group_id','label_final','ocr_ok','stt_ok']].copy()
            frame['seed']=42;frame['fold_model']=fold;frame['role']=role
            for name in models:
                # Exact stored probabilities for the test baseline table; reconstructed values match <=1e-6.
                pp=oof.loc[frame.master_id,name+'_probability'].to_numpy() if role=='test' and name in BASELINES else p[name][ix]
                pred=(pp>=chosen[name]).astype(int)
                frame[name+'_probability']=pp;frame[name+'_prediction']=pred
                metric_rows.append({'fold':fold,'role':role,'model':name,'threshold':chosen[name],
                     **metrics(yy[ix],pp,groups[ix],pred)})
                for label in [0,1]:
                    val=pp[yy[ix]==label]
                    class_rows.append({'fold':fold,'role':role,'model':name,'true_class':label,
                         'n':len(val),'mean':float(val.mean()),'sd':float(val.std(ddof=1)),
                         'q05':float(np.quantile(val,.05)),'median':float(np.median(val)),
                         'q95':float(np.quantile(val,.95))})
                bid=np.minimum((pp*10).astype(int),9)
                for b in range(10):
                    mask=bid==b
                    bin_rows.append({'fold':fold,'role':role,'model':name,'bin':b,'n':int(mask.sum()),
                          'mean_probability':float(pp[mask].mean()) if mask.any() else None,
                          'positive_fraction':float(yy[ix][mask].mean()) if mask.any() else None})
            prob_rows.append(frame)
    probabilities=pd.concat(prob_rows,ignore_index=True)
    test=probabilities.query("role=='test'")
    assert len(test)==3951 and test.master_id.is_unique and test.group_id.nunique()==214
    per_fold=pd.DataFrame(metric_rows);summary=[]
    for name in models:
        fm=per_fold.query("role=='test' and model==@name")
        summary.append({'seed':42,'model':name,
          **metrics(test.label_final.to_numpy(),test[name+'_probability'].to_numpy(),
                    test.group_id.to_numpy(),test[name+'_prediction'].to_numpy()),
          'within_fold_ece10':float(np.average(fm.ece10,weights=fm.n))})
    tables={'ROLE_SIZES':role_rows,'VAT_LR_COEFFICIENTS':coef_rows,
      'PROBABILITY_DIAGNOSTICS_BY_ROLE':metric_rows,
      'SIGMOID_CALIBRATOR_PARAMETERS':cal_rows,'CONVEX_POOL_WEIGHTS':pool_rows,
      'CONTROL_THRESHOLDS':threshold_rows,'BASELINE_REPRODUCTION_CHECKS':check_rows,
      'SCORE_DISTRIBUTIONS_BY_CLASS':class_rows,'RELIABILITY_BINS_BY_ROLE':bin_rows,
      'TEST_COMPARISON_SEED42':summary}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    probabilities.to_csv(out/'ROLE_PROBABILITIES_SEED42.csv',index=False)
    execution={'protocol_sha256':sha(paths['protocol']),
      'source_sha256':{key:sha(path) for key,path in paths.items()},
      'script_sha256':sha(__file__),'environment':{'python':platform.python_version(),
      'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
      'baseline_checks':len(check_rows),'max_baseline_probability_abs_difference':max(r['max_test_probability_abs_difference'] for r in check_rows),
      'all_baseline_decisions_and_thresholds_reproduced':True,
      'role_group_overlap':0,'n_test':3951,'n_source_groups':214,
      'calibration_and_combination_fit_role':'meta_fit','threshold_selection_role':'calibration',
      'test_used_for_fit_or_selection':False,
      'new_analyses':'post hoc descriptive seed42; no new significance or causal claims',
      'public_export':'Only pseudonymous row IDs and probabilities plus aggregates; no raw text, source URLs or media.'}
    (out/'EXECUTION_CHECKS.json').write_text(json.dumps(execution,indent=2))
    print(pd.DataFrame(summary).query('model in @BASELINES or model in @CONTROLS')[
        ['model','macro_f1','brier','ece10','within_fold_ece10','roc_auc','macro_f1_fixed05']].to_string(index=False))
    print(pd.DataFrame(pool_rows).to_string(index=False))
    print('Reproduction checks passed:',len(check_rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',required=True);parser.add_argument('--cache-root',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--protocol',required=True)
    args=parser.parse_args()
    run(args.data_root,args.cache_root,args.output,args.protocol)
