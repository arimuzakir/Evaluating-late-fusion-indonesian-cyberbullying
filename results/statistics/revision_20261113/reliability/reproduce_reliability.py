"""Reproduce supplementary reliability intervals from archived predictions.
No training, label editing, or manuscript modifications. See METODE_RELIABILITY.md.
"""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
here=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--predictions',type=Path,default=here/'input/CALIBRATION_CONTROL_TEST_PREDICTIONS.csv')
parser.add_argument('--reference',type=Path,default=here/'input/BASE_REFERENCE_SEED42.csv')
parser.add_argument('--existing-bins',type=Path,default=here/'input/RELIABILITY_BINS.csv')
parser.add_argument('--output',type=Path,default=here/'reproduced')
args=parser.parse_args()
pred_path,base_path,bins_path,output_dir=args.predictions,args.reference,args.existing_bins,args.output
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=pd.read_csv(pred_path).sort_values('master_id').reset_index(drop=True)
orig=pd.read_csv(base_path);orig=orig[orig.seed.eq(42)].sort_values('master_id').reset_index(drop=True)
assert np.array_equal(z.master_id,orig.master_id) and np.array_equal(z.label_final,orig.label_final)
mapping=pd.DataFrame({'control':z.group_id,'original':orig.group_id}).drop_duplicates()
assert mapping.control.nunique()==len(mapping)==mapping.original.nunique()==214
for m in ['text','equal_vat','vat_lr']:np.testing.assert_allclose(z[m+'_probability'],orig[m+'_probability'],atol=1e-12,rtol=0)
models=['text','equal_vat','vat_lr','sigmoid_equal_vat','convex_vat']
model_labels={'text':'Extracted text','equal_vat':'Equal VAT','vat_lr':'VAT-LR','sigmoid_equal_vat':'Sigmoid-calibrated VAT','convex_vat':'Convex probability pool'}
protocol={'scope':'Supplementary retrospective reliability analysis; primary predictions and Gold v1.1 unchanged; conditional on stored fitted models; no refitting',
 'training_seed':42,'bootstrap_seed':20261112,'replicates':10000,'resampling':'214 complete source groups with replacement, same multiplicity vector for every model and weighting',
 'group_order':'first appearance after ascending master_id','models':models,'edges':[round(i/10,1) for i in range(11)],
 'interval':'pointwise percentile 2.5th and 97.5th percentiles of observed positive fraction and mean predicted probability',
 'source_weight':'each window receives 1 / full-cohort number of windows in its source group, before bootstrap multiplicities; not bin-wise equalization',
 'window_weight':'each eligible window receives unit weight before bootstrap multiplicities',
 'bin_membership':'fixed stored probability; lower-inclusive upper-exclusive except final bin includes 1',
 'sparse_rule':'suppress CI when original bin n < 20 or original source groups < 5',
 'empty_replicates':'report count; omit empty-bin replicates from quantiles only if at least 95% replicates nonempty, otherwise suppress CI',
 'other_limitations':'not simultaneous bands; not external validation; sparse point estimates retained and marked; label adjudication may require regeneration',
 'input_file':pred_path.name,'input_sha256':sha(pred_path),'reference_sha256':sha(base_path),'existing_bins_sha256':sha(bins_path)}
rel=output_dir; rel.mkdir(parents=True,exist_ok=True)
(rel/'RELIABILITY_PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
groups,gi=np.unique(z.group_id,return_inverse=True)
# Use announced first-appearance ordering.
groups=pd.unique(z.group_id);gi=pd.Categorical(z.group_id,categories=groups).codes
G=len(groups); N=len(z);assert G==214 and N==3951
group_n=np.bincount(gi,minlength=G)
rng=np.random.default_rng(protocol['bootstrap_seed'])
mult=rng.multinomial(G,np.repeat(1/G,G),size=protocol['replicates'])
y=z.label_final.to_numpy(float);edges=np.linspace(0,1,11)
rows=[]
for m in models:
 p=z[m+'_probability'].to_numpy(float)
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 bi=np.minimum((p*10).astype(int),9)
 for weighting in ['window','source']:
  wt=np.ones(N) if weighting=='window' else 1/group_n[gi]
  sums=[]
  for values in [wt,wt*y,wt*p]:
   a=np.zeros((G,10));np.add.at(a,(gi,bi),values);sums.append(a)
  bd,bn,bp=[mult@a for a in sums]
  for b in range(10):
   mask=bi==b;n=int(mask.sum());ng=int(np.unique(gi[mask]).size)
   den=sums[0][:,b].sum();point_y=float(sums[1][:,b].sum()/den) if den else np.nan;point_p=float(sums[2][:,b].sum()/den) if den else np.nan
   valid=bd[:,b]>0;nv=int(valid.sum());sparse=n<20 or ng<5
   status='empty' if n==0 else 'sparse_no_ci' if sparse else 'too_many_empty_replicates' if nv<.95*len(mult) else 'pointwise_ci'
   iy=ip=[np.nan,np.nan]
   if status=='pointwise_ci':
    iy=np.quantile(bn[valid,b]/bd[valid,b],[.025,.975]);ip=np.quantile(bp[valid,b]/bd[valid,b],[.025,.975])
   rows.append(dict(model=m,weighting=weighting,bin=b,left=edges[b],right=edges[b+1],right_inclusive=b==9,n=n,groups=ng,mean_probability=point_p,observed_positive=point_y,mean_probability_ci_low=ip[0],mean_probability_ci_high=ip[1],observed_positive_ci_low=iy[0],observed_positive_ci_high=iy[1],valid_replicates=nv,empty_replicates=len(mult)-nv,sparse=sparse,ci_status=status))
r=pd.DataFrame(rows);r.to_csv(rel/'RELIABILITY_POINTWISE_CI.csv',index=False)
assert len(r)==100 and all(r.groupby(['model','weighting']).n.sum()==3951)
# Point estimates must match the existing public table under identical weighting.
prev=pd.read_csv(bins_path)
if 'seed' in prev:prev=prev[prev.seed.eq(42)]
prev=prev[prev.model.isin(models)]
joined=r.merge(prev,on=['model','weighting','bin'],suffixes=('_new','_old'),validate='one_to_one')
assert len(joined)==100
max_diff={}
for col in ['n','groups','mean_probability','observed_positive']:
 np.testing.assert_allclose(joined[col+'_new'],joined[col+'_old'],atol=1e-12,rtol=0,equal_nan=True)
 max_diff[col]=float(np.nanmax(abs(joined[col+'_new']-joined[col+'_old'])))
# Independent direct resampling check for one common bin and both weights.
for weighting in ['window','source']:
 m='equal_vat';b=5;p=z[m+'_probability'].to_numpy(float); mask=np.minimum((p*10).astype(int),9)==b
 wt=np.ones(N) if weighting=='window' else 1/group_n[gi]
 a=np.zeros(G);d=np.zeros(G);np.add.at(a,gi[mask],wt[mask]*y[mask]);np.add.at(d,gi[mask],wt[mask])
 for rep in [0,17,999]:
  expanded=np.repeat(np.arange(G),mult[rep]); ids=np.concatenate([np.where((gi==g)&mask)[0] for g in expanded])
  direct=np.sum(wt[ids]*y[ids])/np.sum(wt[ids]); aggregate=(mult[rep]@a)/(mult[rep]@d)
  assert abs(direct-aggregate)<1e-12
check={'rows':len(r),'groups':G,'windows':N,'point_estimates_match_existing':True,'maximum_absolute_differences':max_diff,'ci_status_counts':r.ci_status.value_counts().to_dict(),'group_mapping_bijection_verified':True,'three_shared_probability_columns_verified':True,'direct_resampling_checks':6,'source_files_unchanged':sha(pred_path)==protocol['input_sha256']}
(rel/'VALIDATION.json').write_text(json.dumps(check,indent=2))

plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':10,'font.family':'DejaVu Sans','pdf.fonttype':42})
colors={'text':'#165B9A','equal_vat':'#C56626','vat_lr':'#3B7563','sigmoid_equal_vat':'#72599A','convex_vat':'#6E6760'}
def draw(ax,ms,weighting):
 ax.plot([0,1],[0,1],color='#9B9B9B',ls='--',lw=1,label='Perfect reliability')
 for m in ms:
  a=r[(r.model==m)&(r.weighting==weighting)&r.n.gt(0)]
  valid=a.ci_status.eq('pointwise_ci');a1=a[valid];a0=a[~valid]
  ax.plot(a.mean_probability,a.observed_positive,color=colors[m],lw=1.2,alpha=.75)
  # Draw percentile interval as endpoint lines: percentile interval need not contain the point estimate.
  ax.vlines(a1.mean_probability,a1.observed_positive_ci_low,a1.observed_positive_ci_high,color=colors[m],lw=1.25)
  ax.plot(a1.mean_probability,a1.observed_positive,'o',color=colors[m],ms=5,label=model_labels[m])
  if len(a0):ax.plot(a0.mean_probability,a0.observed_positive,'o',mfc='white',mec=colors[m],ms=5)
 ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='Mean predicted probability',ylabel='Observed positive fraction')
 ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 ax.set_title('Window-weighted' if weighting=='window' else 'Equal total weight per source')
fig,axs=plt.subplots(2,1,figsize=(7.2,10),layout='constrained')
for ax,wgt in zip(axs,['window','source']):draw(ax,['text','equal_vat'],wgt);ax.legend(loc='upper left')
fig.savefig(rel/'Reliability_Text_EqualVAT.png',dpi=300);fig.savefig(rel/'Reliability_Text_EqualVAT.pdf');plt.close(fig)
fig,axs=plt.subplots(5,2,figsize=(12,20),layout='constrained')
for i,m in enumerate(models):
 for j,wgt in enumerate(['window','source']):
  draw(axs[i,j],[m],wgt);axs[i,j].set_title(model_labels[m]+' | '+('Window' if wgt=='window' else 'Source'))
fig.savefig(rel/'Reliability_All_Five_Models.png',dpi=200);fig.savefig(rel/'Reliability_All_Five_Models.pdf');plt.close(fig)
print('Reliability validation:',check,flush=True)
