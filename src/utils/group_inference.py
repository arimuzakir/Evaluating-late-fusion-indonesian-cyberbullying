"""Source-group paired inference used in the canonical evaluation."""
import numpy as np
REPLICATES=10000

def f1_counts(c):
    c=np.asarray(c,float);tn,fp,fn,tp=np.moveaxis(c,-1,0)
    return .5*(np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)>0)+
      np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0))

def group_counts(y,p,g):
    keys,ix=np.unique(g,return_inverse=True);counts=np.zeros((len(keys),4),int)
    np.add.at(counts,(ix,2*np.asarray(y,int)+np.asarray(p,int)),1);return counts

def paired_inference(d,a,b,reps=REPLICATES,seed=24102026):
    ca=group_counts(d.label_final,d[a+'_prediction'],d.group_id.to_numpy(str))
    cb=group_counts(d.label_final,d[b+'_prediction'],d.group_id.to_numpy(str))
    rng=np.random.default_rng(seed);g=len(ca);observed=float(f1_counts(ca.sum(0))-f1_counts(cb.sum(0)))
    boot=[];null=[]
    for start in range(0,reps,250):
        n=min(250,reps-start);w=rng.multinomial(g,np.full(g,1/g),size=n)
        boot.extend((f1_counts(w@ca)-f1_counts(w@cb)).tolist())
        swaps=rng.integers(0,2,size=(n,g));delta=swaps@(cb-ca)
        null.extend((f1_counts(ca.sum(0)+delta)-f1_counts(cb.sum(0)-delta)).tolist())
    lo,hi=np.quantile(boot,[.025,.975])
    return {'candidate':a,'baseline':b,'delta_macro_f1':observed,'ci_low':float(lo),'ci_high':float(hi),
      'group_permutation_p':float((1+np.sum(np.abs(null)>=abs(observed)-1e-12))/(reps+1)),
      'source_groups':g,'n':len(d),'replicates':reps,'scope':'conditional on fitted models; exploratory internal corpus'}

def holm(values):
    values=np.asarray(values,float);order=np.argsort(values);adj=np.minimum(1,np.maximum.accumulate((len(values)-np.arange(len(values)))*values[order]))
    result=np.empty_like(values);result[order]=adj;return result
