"""Archived reference implementation of the v1.1-GoldAdjudicated model engine,
neural fusion head sensitivity grid (R1.3), and two-partition encoder retraining (R2.5).

PROVENANCE & SCOPE NOTE:
- Recovered from archived notebook `12_INASS_Reviewer_Revisions_CPU.ipynb`
  (embedding `v1_1_gold_adjudicated_Model.py`, SHA-256 `5c5e5078f147551621d2d1d2fa68740e59f2c7eb9424667262da75d7226d18a9`).
- Interactive Google Colab `drive.mount` and `files.upload` calls have been removed.
- End-to-end execution of `train_transformer`, `train_visual`, `sensitivity`, and `repartition`
  requires restricted unredacted text, extracted video keyframes, or cached branch embeddings
  that are withheld from the public repository for privacy and platform-policy compliance.
- For 100% public CPU reproduction of all reported metrics, bootstrap intervals, audit bounds,
  and revision tables from saved predictions, run:
    `python scripts/analysis/reproduce_revision.py --root . --output /tmp/revision_repro`
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import os, sys, json, hashlib, platform, zipfile, shutil
import numpy as np
import pandas as pd

BASE_REL = 'results/v6_reviewer_experiments_20261020_v1'
AUDIO_REL = 'results/audio_waveform16_v1_20261024'
POLICY_REL = 'results/audio_waveform16_quality_policy_20261024_v1'
EVAL_REL = 'results/audio_waveform16_reevaluation_20261024_v1'
MASTER_EXPECTED = 'cd3349fe5c2c824a1d9269135938d3a2de56a4321c685d880a097b71a147dafb'
FROZEN_AUDIO_HASHES = {
 'AUDIO_EXTRACTION_QUALITY.csv':'10d6c33e82b896c53b0e62b4d1223f135a7b8e67910536c4fea35cf51966bc06',
 'ROW_ID_MANIFEST.csv':'09d6e2259d437ce6b1ee0492c0dc27600e50502d95c3820bc021f3cb44200190',
 'ALIGNED_ROW_ID_MANIFEST.csv':'08262e1d205058ede758689a37528b4c5937679bfc21ca0c304ffc36cddbece0',
 'audio_waveform16_all_4401.npy':'5b40a43259daa2de26dd2b072194a21cd71baf09777a2636bf74009d4943c5a5',
 'audio_waveform16_aligned_3951.npy':'e35b8e453bb83c43280ac8affa892788e72f83580da76db18d2bbdebc0660c7d',
 'LOCK.json':'6bbfbfaa74c7d2ac0eec605199b767357b0277bbe7c1e27b41a30484d9a2949e',
 'EXTRACTION_COMPLETE_NEEDS_MODEL_REEVALUATION.json':'e7fd08f78424b988b941003b0fa29cbb6e3094c8efe191d47038a0c8f353d0ae'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()

def atomic_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8');tmp.replace(path)

def atomic_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial');frame.to_csv(tmp,index=False);tmp.replace(path)

def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(path)

def locked_json(path,value):
    path=Path(path)
    if path.exists():
        assert json.loads(path.read_text())==json.loads(json.dumps(value,default=str)), 'Konfigurasi/input berubah. Gunakan folder versi baru: '+str(path)
    else: atomic_json(value,path)

def environment():
    import importlib.metadata
    d={'python':sys.version,'platform':platform.platform()}
    for key in ['numpy','pandas','scikit-learn','torch','einops']:
        try:d[key]=importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:pass
    return d

def project_root(explicit_root=None):
    """Resolve project root from explicit argument or environment variable without interactive Drive mounts."""
    root = Path(explicit_root or os.environ.get('CYBERBULLY_PROJECT_ROOT', '.')).resolve()
    return root

def validate_roles(v,r,fold):
    rr=r[r.outer_fold==fold]
    assert len(rr)==len(v) and rr.master_id.is_unique and set(rr.master_id)==set(v.master_id)
    rr=rr.set_index('master_id').loc[v.master_id].reset_index()
    assert np.array_equal(rr.group_id.to_numpy(str),v.group_id.to_numpy(str))
    expected={'fit','early_stop','meta_fit','calibration','test'}
    assert set(rr.role)==expected
    for key in ['group_id','image_sha256']:
        assert pd.DataFrame({'key':v[key].to_numpy(),'role':rr.role.to_numpy()}).groupby('key').role.nunique().max()==1, key+' melintasi role'
    out={role:np.flatnonzero(rr.role.to_numpy()==role) for role in ['fit','early_stop','meta_fit','calibration','test']}
    assert np.array_equal(np.sort(out['test']),np.flatnonzero(v.outer_fold.to_numpy()==fold))
    for role,ix in out.items(): assert len(ix)>0 and v.iloc[ix].label_final.nunique()==2,role
    return out

def load_audio_inputs(root):
    root=Path(root);audio=root/AUDIO_REL;base=root/BASE_REL
    for name,digest in FROZEN_AUDIO_HASHES.items():
        p=audio/name
        assert p.is_file(), 'Input tidak ditemukan: '+str(p)
        assert sha256(p)==digest, 'Versi audio berbeda dari audit 24 Oktober: '+str(p)
    lock=json.loads((audio/'LOCK.json').read_text())
    for relative,digest in lock['input_sha256'].items():
        assert sha256(root/relative)==digest, 'Input ekstraksi berubah: '+relative
    v=pd.read_csv(base/'VIDEO_PRIMARY_MANIFEST.csv',dtype={'master_id':str,'group_id':str,'image_sha256':str})
    r=pd.read_csv(base/'VIDEO_ROLE_MANIFEST.csv',dtype={'master_id':str,'group_id':str})
    rows=pd.read_csv(audio/'ROW_ID_MANIFEST.csv',dtype={'master_id':str,'video_id':str})
    aligned=pd.read_csv(audio/'ALIGNED_ROW_ID_MANIFEST.csv',dtype={'master_id':str,'group_id':str})
    q=pd.read_csv(audio/'AUDIO_EXTRACTION_QUALITY.csv',dtype={'master_id':str,'video_id':str})
    all_a=np.load(audio/'audio_waveform16_all_4401.npy',allow_pickle=False)
    a=np.load(audio/'audio_waveform16_aligned_3951.npy',allow_pickle=False)
    assert len(v)==3951 and len(rows)==len(q)==4401
    assert v.master_id.is_unique and rows.master_id.is_unique and q.master_id.is_unique
    assert list(q.master_id)==list(rows.master_id) and list(q.video_id)==list(rows.video_id)
    assert list(aligned.master_id)==list(v.master_id)
    assert np.array_equal(aligned.group_id.to_numpy(str),v.group_id.to_numpy(str))
    assert np.array_equal(aligned.outer_fold,v.outer_fold)
    assert all_a.shape==(4401,16) and a.shape==(3951,16) and np.isfinite(all_a).all() and np.isfinite(a).all()
    pos=pd.Series(np.arange(len(rows)),index=rows.master_id).loc[v.master_id].to_numpy()
    assert np.array_equal(a,all_a[pos])
    assert set(q.status)=={'OK','TRUNCATED_AT_SOURCE_END','SOURCE_HAS_NO_AUDIO_STREAM'}
    assert (q.status.value_counts().to_dict()=={'OK':4281,'TRUNCATED_AT_SOURCE_END':94,'SOURCE_HAS_NO_AUDIO_STREAM':26})
    assert np.isfinite(q[['decoded_seconds','requested_seconds']].to_numpy()).all()
    assert (q.requested_seconds>0).all() and (q.decoded_seconds>=0).all()
    assert np.allclose(q.requested_seconds,rows.end_sec-rows.start_sec,atol=1e-6)
    missing=q.status.eq('SOURCE_HAS_NO_AUDIO_STREAM').to_numpy()
    assert np.array_equal(all_a[:,15]==0,missing) and np.all(all_a[missing]==0)
    assert np.all(all_a[~missing,15]==1) and np.all(q.loc[~missing,'decoded_seconds']>=.25)
    q=q.merge(rows[['master_id','start_sec','end_sec']],on='master_id',validate='one_to_one',sort=False)
    q['shortfall_seconds']=np.maximum(q.requested_seconds-q.decoded_seconds,0)
    q['decoded_fraction']=np.minimum(q.decoded_seconds/q.requested_seconds,1)
    q['in_primary_evaluation']=q.master_id.isin(v.master_id)
    qa=q.set_index('master_id').loc[v.master_id].reset_index()
    assert qa.status.value_counts().to_dict()=={'OK':3841,'TRUNCATED_AT_SOURCE_END':84,'SOURCE_HAS_NO_AUDIO_STREAM':26}
    role={fold:validate_roles(v,r,fold) for fold in [0,1,2,3]}
    return v,r,rows,q,qa,a,role

def result_zip(folder,name):
    folder=Path(folder);dest=folder/name;tmp=dest.with_name(name+'.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p not in [dest,tmp] and not p.name.endswith('.partial') and p.name!='resume.pt':z.write(p,p.relative_to(folder))
    tmp.replace(dest);return dest

import random, math, gc, importlib.util
import joblib, sklearn
from sklearn.metrics import (accuracy_score,f1_score,roc_auc_score,average_precision_score,
    confusion_matrix,brier_score_loss,precision_recall_fscore_support,roc_curve,precision_recall_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn

SEEDS=[42,2026,2027]
FOLDS=[0,1,2,3]
GRID=[round(.10+.025*i,3) for i in range(33)]
PRIMARY_SEED=42
REPLICATES=10000
HEAD_CONFIG={'head_epochs':30,'head_batch':64,'head_lr':.001}
ARCHIVE_RESULT_HASHES={'LOCKED_CONFIG.json':'acbe9851dfacf73e1aa477c1b107be197c8be4f71562c94809a9714bfda40a67',
 'VIDEO_OOF_PREDICTIONS.csv':'0f92849ada3f785cd280f7e0da41cda07a44be50d7c223a14e03d7c127c4c1a8'}
NAMES=['majority','text','visual','audio','equal_va','equal_vat','va_lr','vat_lr',
 'vat_flags','vat_coverage','coverage_only','coverage_shuffled_control','concat_mlp',
 'healnet_adapted_2024','siglip2_visual_probe','siglip2_vt_probe','siglip2_vta_probe',
 'equal_tv','equal_ta','tv_lr','ta_lr']
UNCHANGED=['majority','text','visual','coverage_only','siglip2_visual_probe','siglip2_vt_probe']
AUDIO_DEPENDENT=['audio','equal_va','equal_vat','va_lr','vat_lr','vat_flags','vat_coverage',
 'coverage_shuffled_control','concat_mlp','healnet_adapted_2024','siglip2_vta_probe']
PRIMARY_CONTRASTS=[('vat_coverage',b) for b in ['visual','vat_lr','vat_flags','coverage_only',
 'coverage_shuffled_control','healnet_adapted_2024','siglip2_vta_probe']]
EXPLORATORY_CONTRASTS=[('equal_vat','text'),('equal_va','visual'),('equal_ta','text'),
 ('ta_lr','text'),('equal_tv','text'),('tv_lr','text')]

def group_weights(groups):
    counts=Counter(groups);w=np.array([1/counts[g] for g in groups],float)
    return w*(len(w)/w.sum())

def fit_lr(x,y,groups,C=.2):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=42))
    model.fit(x,y,logisticregression__sample_weight=group_weights(np.asarray(groups)))
    return model

def choose_threshold(y,p):
    values=np.array([f1_score(y,np.asarray(p)>=t,average='macro',labels=[0,1],zero_division=0) for t in GRID])
    best=np.flatnonzero(np.isclose(values,values.max(),atol=1e-12,rtol=0))
    return min((GRID[i] for i in best),key=lambda t:(abs(t-.5),t))

def metrics(y,p,pred):
    y=np.asarray(y,int);p=np.asarray(p,float);pred=np.asarray(pred,int)
    assert len(y)==len(p)==len(pred) and len(y)>0
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
      'macro_f1':float(f1.mean()),'precision_macro':float(precision.mean()),'recall_macro':float(recall.mean()),
      'recall_nonbullying':float(recall[0]),'recall_bullying':float(recall[1]),
      'support_nonbullying':int(support[0]),'support_bullying':int(support[1]),
      'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
      'average_precision':float(average_precision_score(y,p)) if y.sum()>0 else None,
      'brier_score':float(brier_score_loss(y,p)),
      'macro_f1_fixed_050':float(f1_score(y,p>=.5,average='macro',labels=[0,1],zero_division=0)),
      'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def coverage_features(v,pv,pa,pt,variant):
    if variant=='va':return np.column_stack([pv,pa])
    base=[pv,pa,pt]
    if variant=='vat':return np.column_stack(base)
    o=v.ocr_ok.to_numpy(float);s=v.stt_ok.to_numpy(float)
    ol=np.log1p(v.ocr_length.to_numpy(float));sl=np.log1p(v.stt_length.to_numpy(float))
    if variant=='coverage_only':return np.column_stack([o,s,ol,sl])
    base.extend([o,s])
    if variant=='coverage':base.extend([ol,sl,pt*o,pt*s,pv*o,pa*s])
    return np.column_stack(base)

def branch_data(path,ids,embedding_dim):
    with np.load(path,allow_pickle=False) as z:
        assert np.array_equal(z['ids'].astype(str),np.asarray(ids,str)), 'Urutan/ID branch berbeda: '+str(path)
        p=z['probability'].astype(float).copy();e=z['embedding'].astype(np.float32).copy()
    assert p.shape==(len(ids),) and e.shape==(len(ids),embedding_dim)
    assert np.isfinite(p).all() and np.isfinite(e).all() and ((p>=0)&(p<=1)).all()
    return p,e

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
       'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def atomic_torch(value,path):
    path=Path(path);tmp=path.with_name(path.name+'.partial');torch.save(value,tmp);tmp.replace(path)

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

class FeatureFusion(nn.Module):
    def __init__(self,dimensions,kind):
        super().__init__();self.kind=kind
        if kind=='healnet':
            self.core=HealNet(n_modalities=3,channel_dims=list(dimensions),num_spatial_axes=[1,1,1],out_dims=2,
              depth=2,l_c=16,l_d=64,x_heads=4,l_heads=4,cross_dim_head=16,latent_dim_head=16,
              attn_dropout=.1,ff_dropout=.1,fourier_encode_data=False,snn=False)
        else:self.core=nn.Sequential(nn.Linear(sum(dimensions),128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,inputs):
        return self.core([x.unsqueeze(1) for x in inputs]) if self.kind=='healnet' else self.core(torch.cat(inputs,dim=1))

def train_feature_fusion(v,features,meta,seed,folder,device,kind='concat_mlp',config=None):
    config=HEAD_CONFIG if config is None else config
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'outputs.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['ids'].astype(str),v.master_id.to_numpy(str));p=z['probability'].copy()
        assert p.shape==(len(v),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        return p
    seed_all(seed);scalers=[StandardScaler().fit(x[meta]) for x in features]
    arrays=[sc.transform(x).astype(np.float32) for sc,x in zip(scalers,features)]
    arrays[2][v.text_present.to_numpy()==0]=0
    model=FeatureFusion([x.shape[1] for x in arrays],kind).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['head_lr'],weight_decay=config.get('weight_decay',.01))
    labels=v.label_final.to_numpy(int);weights=group_weights(v.iloc[meta].group_id.to_numpy())
    checkpoint=folder/'resume.pt';history=[];start=0
    if checkpoint.exists():
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
        start=cp['epoch'];history=cp['history'];restore_rng(cp['rng'])
        for state in optimizer.state.values():
            for k,value in state.items():
                if isinstance(value,torch.Tensor):state[k]=value.to(device)
        del cp
    for epoch in range(start,config['head_epochs']):
        model.train();order=np.random.default_rng(seed+epoch).permutation(len(meta));total=0
        for offset in range(0,len(order),config['head_batch']):
            ix=order[offset:offset+config['head_batch']];pos=meta[ix]
            inputs=[torch.from_numpy(x[pos]).to(device) for x in arrays]
            y=torch.from_numpy(labels[pos]).to(device);w=torch.from_numpy(weights[ix]).float().to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(inputs)
            assert logits.shape==(len(pos),2) and torch.isfinite(logits).all()
            loss=(nn.functional.cross_entropy(logits,y,reduction='none')*w).mean();loss.backward()
            if kind=='healnet' and epoch==0 and offset==0:
                for modality in range(3):
                    prefix=f'core.layers.0.{2*modality}.'
                    grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    assert grads and any(float(g.abs().sum())>0 for g in grads),f'HEALNet: modalitas {modality} tanpa gradient'
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())
        history.append({'epoch':epoch+1,'meta_fit_loss':total/max(math.ceil(len(meta)/config['head_batch']),1)})
        atomic_torch({'model':cpu_state(model),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'history':history,'rng':rng_state()},checkpoint)
    model.eval();prob=[]
    with torch.inference_mode():
        for startpos in range(0,len(v),256):
            x=[torch.from_numpy(z[startpos:startpos+256]).to(device) for z in arrays]
            prob.append(model(x).float().softmax(-1)[:,1].cpu().numpy())
    p=np.concatenate(prob)
    assert np.isfinite(p).all()
    atomic_torch(cpu_state(model),folder/'model.pt');joblib.dump(scalers,folder/'scalers.joblib')
    atomic_json({'kind':kind,'feature_dims':[x.shape[1] for x in arrays],'seed':seed,'training_role':'meta_fit only',
      'config':config,'history':history,'source_audio':'waveform16_v1'},folder/'MODEL_MANIFEST.json')
    atomic_npz(path,ids=v.master_id.to_numpy(str),probability=p)
    checkpoint.unlink(missing_ok=True);del model,optimizer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return p

def fold_probabilities(v,a,roles,pt,pv,te,ve,siglip,seed,fold,folder,device,head_config=None):
    fit=roles['fit'];meta=roles['meta_fit'];y=v.label_final.to_numpy(int);groups=v.group_id.to_numpy(str)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    audio=fit_lr(a[fit],y[fit],groups[fit],C=1.);joblib.dump(audio,folder/'audio_model.joblib')
    pa=audio.predict_proba(a)[:,1]
    p={'majority':np.full(len(v),y[fit].mean()),'text':pt,'visual':pv,'audio':pa,
       'equal_va':(pv+pa)/2,'equal_vat':(pv+pa+pt)/3,'equal_tv':(pt+pv)/2,'equal_ta':(pt+pa)/2}
    fitted={}
    for name,variant in [('va_lr','va'),('vat_lr','vat'),('vat_flags','flags'),('vat_coverage','coverage'),('coverage_only','coverage_only')]:
        x=coverage_features(v,pv,pa,pt,variant);model=fit_lr(x[meta],y[meta],groups[meta])
        p[name]=model.predict_proba(x)[:,1];fitted[name]=(model,variant);joblib.dump(model,folder/(name+'.joblib'))
    for name,x in [('tv_lr',np.column_stack([pv,pt])),('ta_lr',np.column_stack([pa,pt]))]:
        model=fit_lr(x[meta],y[meta],groups[meta]);p[name]=model.predict_proba(x)[:,1]
        joblib.dump(model,folder/(name+'.joblib'))
    shuffled=v.copy();columns=['ocr_ok','stt_ok','ocr_length','stt_length']
    for rid,(role,ix) in enumerate(roles.items()):
        for pid,platform_name in enumerate(sorted(v.iloc[ix].platform.unique())):
            pos=ix[v.iloc[ix].platform.to_numpy()==platform_name]
            perm=np.random.default_rng(seed+1000*fold+100*rid+pid).permutation(pos)
            shuffled.loc[pos,columns]=v.loc[perm,columns].to_numpy()
    x=coverage_features(shuffled,pv,pa,pt,'coverage');model=fit_lr(x[meta],y[meta],groups[meta])
    p['coverage_shuffled_control']=model.predict_proba(x)[:,1];joblib.dump(model,folder/'coverage_shuffled_control.joblib')
    p['concat_mlp']=train_feature_fusion(v,[ve,a,te],meta,seed,folder/'concat_mlp',device,config=head_config)
    p['healnet_adapted_2024']=train_feature_fusion(v,[ve,a,te],meta,seed,folder/'healnet_adapted_2024',device,kind='healnet',config=head_config)
    for name,x in [('siglip2_visual_probe',siglip['visual']),
       ('siglip2_vt_probe',np.column_stack([siglip['visual'],siglip['text']])),
       ('siglip2_vta_probe',np.column_stack([siglip['visual'],siglip['text'],a]))]:
        model=fit_lr(x[fit],y[fit],groups[fit]);p[name]=model.predict_proba(x)[:,1];joblib.dump(model,folder/(name+'.joblib'))
    assert set(p)==set(NAMES)
    return p,fitted,pa

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

def absolute_ci(d,name,reps=REPLICATES,seed=42):
    c=group_counts(d.label_final,d[name+'_prediction'],d.group_id.to_numpy(str));g=len(c)
    rng=np.random.default_rng(seed);scores=[]
    for start in range(0,reps,250):
        w=rng.multinomial(g,np.full(g,1/g),size=min(250,reps-start));scores.extend(f1_counts(w@c).tolist())
    lo,hi=np.quantile(scores,[.025,.975]);return {'ci_low':float(lo),'ci_high':float(hi),'source_groups':g,'replicates':reps}

def validate_completed_fold(d,v,test,seed,thresholds,folder):
    assert d.master_id.is_unique and list(d.master_id)==list(v.iloc[test].master_id)
    assert np.array_equal(d.label_final.to_numpy(int),v.iloc[test].label_final.to_numpy(int))
    assert np.array_equal(d.group_id.to_numpy(str),v.iloc[test].group_id.to_numpy(str))
    assert (d.seed==seed).all()
    for name in NAMES:
        p=d[name+'_probability'].to_numpy();t=d[name+'_threshold'].to_numpy();pred=d[name+'_prediction'].to_numpy()
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert np.allclose(t,thresholds[name],atol=1e-12) and np.array_equal(pred,(p>=t).astype(int))
    complete=json.loads((Path(folder)/'FOLD_COMPLETE.json').read_text())
    assert complete['predictions_sha256']==sha256(Path(folder)/'TEST_PREDICTIONS.csv')

def present(value):
    return not pd.isna(value) and str(value).strip().casefold() not in ('','-','nan','none','null','n/a')

EMOJI_MAP={'🤡':' badut ','🗿':' batu ','🤮':' muntah ','💩':' kotoran ','💀':' tengkorak ',
 '👎':' jempol bawah ','🖕':' jari tengah ','🤬':' mengumpat ','😡':' marah ','😭':' menangis ',
 '🤣':' tertawa ','😂':' tertawa ','😹':' tertawa ','🔥':' api ','🙏':' terima kasih '}

SLANG={'yg':'yang','dgn':'dengan','bgt':'banget','ga':'tidak','gak':'tidak','gk':'tidak',
 'klo':'kalau','kalo':'kalau','tp':'tapi','tpi':'tapi','udh':'sudah','udah':'sudah',
 'blm':'belum','blom':'belum','bkn':'bukan','krn':'karena','emg':'memang','jg':'juga','kls':'kelas'}

def preprocess_text(value):
    if not present(value):return ''
    text=html.unescape(str(value))
    for key,val in EMOJI_MAP.items():text=text.replace(key,val)
    text=emoji.demojize(text,language='en')
    text=re.sub(r':([a-zA-Z0-9_]+):',lambda m:' '+m.group(1).replace('_',' ')+' ',text)
    text=re.sub(r'https?://\S+|www\.\S+',' HTTPURL ',text)
    text=re.sub(r'@\w+',' @USER ',text)
    text=re.sub(r'#(?=\w)','',text).lower()
    text=re.sub(r'(.)\1{2,}',r'\1\1',text)
    for key,val in SLANG.items():text=re.sub(r'\b'+re.escape(key)+r'\b',val,text)
    return re.sub(r'\s+',' ',text).strip()

LABEL_ZIP_SHA256='be61fe7e8db88dccb41ac3e37eb5e013f2d9ef4a80efb4675f54d36b2bb49ac4'
LABEL_MANIFEST_SHA256='c992ba3078eccd7bf2ba91f228cc3cfcae9221ac080de5fe9c88d26c723d5465'
EXPECTED_SIGLIP_SHA256='9ad7e603e42864796e70f11240ce468e4acd75728e33483a21b3c5e5cc0af1e1'
EXPECTED_LOPO_ROLE_SHA256='85e18edeb62637aa0a92dd824df429499ca2dec9ef7837823ab1f109227ffb42'

import re,html,time,contextlib,tempfile
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
from transformers import AutoTokenizer,AutoModelForSequenceClassification,get_cosine_schedule_with_warmup
import transformers,torchvision,emoji

LABEL_VERSION='v1.1-GoldAdjudicated'
LABEL_REL='results/label_versions/v1_1_GoldAdjudicated'
RUN_REL='results/v6_gold_adjudicated_v1_1_20261024'
PREVIOUS_WAVEFORM_REL='results/audio_waveform16_reevaluation_20261024_v1'
EXPECTED_PATCH={'CB_MASTER_11039':(0,1),'CB_MASTER_11091':(0,1),'CB_MASTER_11325':(1,0),'CB_MASTER_09804':(0,1)}
TEXT_MODEL_ID='indobenchmark/indobert-base-p2'
TEXT_MODEL_COMMIT='94b4e0a82081fa57f227fcc2024d1ea89b57ac1f'
CONFIG={'text_batch':16,'visual_batch':32,'accumulation':2,'video_text_epochs':4,
 'visual_epochs':5,'video_max_length':192,'save_best_weights':True}
CONFIG.update(HEAD_CONFIG)

def env_v11():
    out=environment()
    out.update({'transformers':transformers.__version__,'torchvision':torchvision.__version__,
      'emoji':emoji.__version__,'cuda_runtime':torch.version.cuda,
      'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    return out

def ensure_label_release(root,allow_upload=True):
    root=Path(root);target=root/LABEL_REL
    manifest=target/'FILE_MANIFEST.json'
    if not manifest.exists():
        candidates=[root/'results/label_versions/CYBERBULLY_v1_1_GoldAdjudicated.zip',
          root/'CYBERBULLY_v1_1_GoldAdjudicated.zip',root/'CYBERBULLY_v1_1_GoldAdjudicated.zip']
        archives=[p for p in candidates if p.is_file()]
        # Interactive Colab upload removed for repository portability.
        assert archives,'Ekstrak paket label ke '+str(target)+' atau unggah ZIP label melalui panel Files Colab.'
        archive=archives[0]
        assert sha256(archive)==LABEL_ZIP_SHA256,'ZIP label bukan versi yang disiapkan. Gunakan paket v1.1 yang disertakan.'
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for member in z.infolist():
                rel=Path(member.filename)
                assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='v1_1_GoldAdjudicated'
                if member.is_dir():continue
                dest=root/'results/label_versions'/rel
                data=z.read(member)
                if dest.exists():assert dest.read_bytes()==data,'Isi folder label berbeda: '+str(dest)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    tmp=dest.with_name(dest.name+'.partial');tmp.write_bytes(data);tmp.replace(dest)
    assert sha256(manifest)==LABEL_MANIFEST_SHA256,'Manifest rilis label berbeda dari paket yang dikunci.'
    lock=json.loads(manifest.read_text());assert lock['version']==LABEL_VERSION
    for name,digest in lock['files'].items():
        p=target/name;assert p.is_file() and sha256(p)==digest,'File rilis label berubah atau belum lengkap: '+name
    return target

def validate_patch(old,new,label_column='label_final'):
    assert old.master_id.is_unique and new.master_id.is_unique and list(old.master_id)==list(new.master_id)
    assert set(old.columns)==set(new.columns)
    ignore=[label_column]+(['label_teks'] if 'label_teks' in old.columns else [])
    for col in old.columns:
        if col not in ignore:
            assert old[col].fillna('').astype(str).tolist()==new[col].fillna('').astype(str).tolist(),'Non-label berubah: '+col
    changed=old[label_column].to_numpy(int)!=new[label_column].to_numpy(int)
    expected=set(old.master_id)&set(EXPECTED_PATCH)
    assert set(old.loc[changed,'master_id'])==expected
    for mid in expected:
        i=old.index[old.master_id==mid][0]
        assert (int(old.at[i,label_column]),int(new.at[i,label_column]))==EXPECTED_PATCH[mid]

def prepare_v11(root,label_dir,device):
    root=Path(root);base=root/BASE_REL;run=root/RUN_REL

    oldv,oldroles,rows,quality,qa,a,roles=load_audio_inputs(root)
    oldmaster_path=root/'data/dataset_master_13307_labeled.csv'
    assert sha256(oldmaster_path)==MASTER_EXPECTED,'Sumber lama berubah. Jangan campur input root yang sudah ditimpa dengan rilis versi terpisah.'
    def csv(p):return pd.read_csv(p,dtype={'master_id':str,'group_id':str,'image_sha256':str})
    master=csv(label_dir/'data/dataset_master_13307_labeled.csv')
    oldmaster=csv(oldmaster_path);validate_patch(oldmaster,master)
    v=csv(label_dir/'manifests/VIDEO_PRIMARY_MANIFEST.csv');validate_patch(oldv,v)
    r=csv(label_dir/'manifests/VIDEO_ROLE_MANIFEST.csv')
    assert sha256(label_dir/'manifests/VIDEO_ROLE_MANIFEST.csv')==sha256(base/'VIDEO_ROLE_MANIFEST.csv')
    roles={f:validate_roles(v,r,f) for f in FOLDS}
    seg=csv(label_dir/'data/multimodal/segmentation_10s_stride3_v2/multimodal_segments.csv')
    meta=csv(label_dir/'features/multimodal_video_clean_metadata.csv')
    oldseg=csv(root/'data/multimodal/segmentation_10s_stride3_v2/multimodal_segments.csv')
    oldmeta=csv(root/'features/multimodal_video_clean_metadata.csv')
    validate_patch(oldseg,seg);validate_patch(oldmeta,meta)
    assert len(master)==13307 and len(seg)==len(meta)==4401 and len(v)==3951
    for frame in [master,seg,meta]:
        assert frame.master_id.is_unique and np.array_equal(frame.set_index('master_id').loc[v.master_id,'label_final'].to_numpy(int),v.label_final.to_numpy(int))
    assert v.label_final.value_counts().to_dict()=={1:2326,0:1625}
    v=v.merge(seg[['master_id','video_id','segment_id','ocr_text','stt_transcript']].rename(columns={'segment_id':'segment_id_source'}),on='master_id',validate='one_to_one',sort=False)
    v=v.merge(master[['master_id','raw_text_or_ocr','stt_audio_transcript']],on='master_id',validate='one_to_one',sort=False)
    assert list(v.master_id)==list(oldv.master_id)
    normalize=lambda x:re.sub(r'\s+',' ',str(x)).strip() if present(x) else ''
    assert (v.segment_id==v.segment_id_source).all()
    assert v.frame_name.map(lambda x:Path(x).stem).eq(v.segment_id).all()
    assert v.ocr_text.map(normalize).eq(v.raw_text_or_ocr.map(normalize)).all()
    assert v.stt_transcript.map(normalize).eq(v.stt_audio_transcript.map(normalize)).all()
    for name,source in [('ocr','ocr_text'),('stt','stt_transcript')]:
        flag=v[source].map(present).astype(int)
        length=v[source].map(lambda x:len(str(x)) if present(x) else 0).clip(upper=512)
        assert np.array_equal(flag,v[name+'_ok']) and np.array_equal(length,v[name+'_length'])
    v['text_present']=((v.ocr_ok==1)|(v.stt_ok==1)).astype(int)
    v['text_clean']=[preprocess_text(' '.join(str(x) for x in (o,s) if present(x))) for o,s in zip(v.ocr_text,v.stt_transcript)]
    v.loc[v.text_present==0,'text_clean']=''
    revisions_path=base/'MODEL_REVISIONS.json';revisions=json.loads(revisions_path.read_text())
    assert revisions[TEXT_MODEL_ID]==TEXT_MODEL_COMMIT,'Pretrained revision berbeda dari run asal.'
    original_config=json.loads((base/'LOCKED_CONFIG.json').read_text())['config']
    for key,value in CONFIG.items():
        if key in original_config:assert original_config[key]==value,(key,original_config[key],value)
    assert original_config['video_seeds']==SEEDS and original_config['threshold_grid']==GRID
    sigpath=base/'siglip2_frozen_features.npz'
    assert sha256(sigpath)==EXPECTED_SIGLIP_SHA256,'Fitur SigLIP2 generik berbeda dari input yang diaudit.'
    with np.load(sigpath,allow_pickle=False) as z:
        assert np.array_equal(z['ids'].astype(str),v.master_id.to_numpy(str))
        sig={k:z[k].astype(np.float32).copy() for k in ['visual','text']}
    for x in sig.values():assert len(x)==len(v) and x.ndim==2 and np.isfinite(x).all()
    assert np.all(sig['text'][v.text_present.to_numpy()==0]==0)
    old_lopo=root/PREVIOUS_WAVEFORM_REL/'LOPO_ROLE_MANIFEST.csv'
    assert sha256(old_lopo)==EXPECTED_LOPO_ROLE_SHA256,'Gunakan role LOPO dari evaluasi waveform16 yang sudah selesai.'
    lopo=pd.read_csv(old_lopo,dtype={'master_id':str,'group_id':str})
    inputs=[oldmaster_path,base/'VIDEO_PRIMARY_MANIFEST.csv',base/'VIDEO_ROLE_MANIFEST.csv',base/'LOCKED_CONFIG.json',revisions_path,sigpath,old_lopo]
    inputs += [root/AUDIO_REL/name for name in FROZEN_AUDIO_HASHES]
    inputs += [label_dir/name for name in json.loads((label_dir/'FILE_MANIFEST.json').read_text())['files']]
    inputs.append(label_dir/'FILE_MANIFEST.json')
    protocol={'protocol_version':'v6_gold_adjudicated_v1_1_20261024','label_version':LABEL_VERSION,
      'seeds':SEEDS,'folds':FOLDS,'primary_seed':PRIMARY_SEED,'models':NAMES,'configuration':CONFIG,
      'threshold_grid':GRID,'threshold_selection':'calibration role only; closest .5 then lower on ties',
      'text_checkpoint':TEXT_MODEL_ID,'text_revision':TEXT_MODEL_COMMIT,'visual_weights':'EfficientNet_B0_Weights.IMAGENET1K_V1',
      'supervised_branches':'train from generic pretrained weights under v1.1 labels in every seed/fold; old supervised NPZ/model cache never loaded',
      'encoder_selection':'best early_stop-role macro F1 at .5; earliest epoch on tie',
      'fusion_training':'meta_fit only, fixed 30 epochs for neural heads; audio/probes fit role only',
      'audio_representation':'waveform16_v1, quality policy unchanged; 3841 OK,84 partial,26 absent in aligned cohort',
      'parent_video_context':'metadata only; not a window-label override and not used for fitting',
      'primary_7_contrasts':PRIMARY_CONTRASTS,'exploratory_6_contrasts':EXPLORATORY_CONTRASTS,
      'multiplicity':'Holm within each family; all inference remains exploratory internal-corpus analysis',
      'inference_replicates':REPLICATES,'lopo_role_source':'previous waveform16 LOPO assignments frozen before this label update',
      'no_best_seed_selection':True,'environment':env_v11(),'device':str(device),
      'implementation_sha256':IMPLEMENTATION_SHA256,'healnet_sha256':HEALNET_SHA256,
      'input_sha256':{str(p.relative_to(root)):sha256(p) for p in inputs}}
    run.mkdir(parents=True,exist_ok=True);locked_json(run/'RUN_LOCK.json',protocol)
    atomic_csv(v,run/'EVALUATION_MANIFEST.csv');atomic_csv(r,run/'ROLE_MANIFEST.csv')
    atomic_csv(lopo,run/'LOPO_ROLE_MANIFEST.csv')
    for name in ['WINDOW_LABEL_CHANGELOG_4.csv','AFFECTED_ROLES_16.csv','IAA_SCOPE.json','RESEARCHER_DECLARATION_SCOPE.json']:
        dest=run/'annotation'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(label_dir/'annotation'/name,dest)
    atomic_json({'status':'PASS','n':len(v),'changed_labels':4,'label_version':LABEL_VERSION,
      'roles_unchanged':True,'comment_labels_unchanged':True,'legacy_supervised_branch_cache_used':False,
      'generic_siglip_and_waveform_reused':True,'environment':env_v11()},run/'PREFLIGHT.json')
    return v,a,qa,roles,sig,lopo,run

def stage_frames(root,v):
    local=Path(os.environ.get('CYBERBULLY_FRAME_CACHE', str(Path(root)/'cache/frames')));local.mkdir(parents=True,exist_ok=True)
    archive=Path(root)/'results/V6_FINAL_GPU_RUNS/canonical_efficientnet_b0_v2/FRAME_CACHE_3996.zip'
    expected=set(v.frame_name)
    if archive.is_file():
        with zipfile.ZipFile(archive) as z:
            assert expected.issubset(z.namelist())
            for name in tqdm(sorted(expected),desc='Stage keyframes from frozen cache'):
                assert Path(name).name==name
                dest=local/name
                if not dest.is_file() or dest.stat().st_size!=z.getinfo(name).file_size:
                    tmp=dest.with_name(dest.name+'.partial')
                    with z.open(name) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst)
                    tmp.replace(dest)
    else:
        source=Path(root)/'data/multimodal/segmentation_10s_stride3_v2/frames'
        for name in tqdm(sorted(expected),desc='Stage keyframes from Drive'):
            assert Path(name).name==name;src=source/name;assert src.is_file(),str(src)
            dest=local/name
            if not dest.exists() or dest.stat().st_size!=src.stat().st_size:shutil.copyfile(src,dest)
    for name,digest in tqdm(v[['frame_name','image_sha256']].drop_duplicates().itertuples(index=False,name=None),desc='Verify keyframe hashes'):
        assert sha256(local/name)==digest,'Keyframe berubah: '+name
        with Image.open(local/name) as im:im.verify()
    return local

def model_revision(model_id):
    assert model_id==TEXT_MODEL_ID
    return TEXT_MODEL_COMMIT

def cleanup():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def amp_context():
    return torch.autocast('cuda',dtype=AMP_DTYPE) if DEVICE.type=='cuda' else contextlib.nullcontext()

def branch_finished(folder,ids):
    folder=Path(folder);marker=folder/'BRANCH_COMPLETE.json'
    if not marker.exists():return False
    meta=json.loads(marker.read_text())
    for name,digest in meta['files'].items():assert sha256(folder/name)==digest,'Cache branch rusak: '+str(folder/name)
    with np.load(folder/'outputs.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'].astype(str),np.asarray(ids,str))
        assert np.isfinite(z['probability']).all() and np.isfinite(z['embedding']).all()
    return True

def commit_branch(folder):
    folder=Path(folder)
    files={str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file() and p.name not in ['resume.pt','BRANCH_COMPLETE.json'] and not p.name.endswith('.partial')}
    assert {'outputs.npz','MODEL_MANIFEST.json','best_model.pt'}.issubset(files)
    atomic_json({'label_version':LABEL_VERSION,'files':files},folder/'BRANCH_COMPLETE.json')

class TokenDataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_length):
        self.enc=tokenizer(list(texts),max_length=max_length,truncation=True,padding='max_length',return_tensors='pt')
        self.labels=torch.tensor(np.asarray(labels,int),dtype=torch.long)
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):return {**{k:v[i] for k,v in self.enc.items()},'labels':self.labels[i]}

def data_loader(ds,batch,shuffle=False,seed=42):
    generator=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=(DEVICE.type=='cuda'),generator=generator)

def training_loss(logits,y):return nn.functional.cross_entropy(logits,y,label_smoothing=.05)

@torch.inference_mode()
def predict_text_model(model,tokenizer,texts,max_length,batch=32,return_features=True):
    ds=TokenDataset(texts,np.zeros(len(texts),int),tokenizer,max_length)
    ps=[];features=[];model.eval()
    for data in data_loader(ds,batch):
        data={k:v.to(DEVICE,non_blocking=True) for k,v in data.items() if k!='labels'}
        with amp_context():
            out=model(**data,output_hidden_states=return_features,return_dict=True)
        ps.append(out.logits.float().softmax(-1)[:,1].cpu().numpy())
        if return_features:features.append(out.hidden_states[-1][:,0,:].float().cpu().numpy())
    return np.concatenate(ps),np.concatenate(features) if features else None

def train_transformer(model_id,frame,fit_index,val_index,pred_index,seed,folder,max_length,epochs):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'outputs.npz'
    if branch_finished(folder,frame.iloc[pred_index].master_id):
        with np.load(result_path,allow_pickle=False) as z:
            assert np.array_equal(z['ids'],frame.iloc[pred_index].master_id.to_numpy(str))
            return {k:z[k] for k in z.files}
    seed_all(seed)
    revision=model_revision(model_id)
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision)
    model=AutoModelForSequenceClassification.from_pretrained(model_id,revision=revision,num_labels=2,attn_implementation='sdpa').to(DEVICE)
    tokenized=TokenDataset(frame.iloc[fit_index].text_clean,frame.iloc[fit_index].label_final,tokenizer,max_length)
    steps=math.ceil(math.ceil(len(tokenized)/CONFIG['text_batch'])/CONFIG['accumulation'])*epochs
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,int(.1*steps)),steps)
    scaler=torch.amp.GradScaler('cuda',enabled=(DEVICE.type=='cuda' and not USE_BF16))
    checkpoint=folder/'resume.pt';best_path=folder/'best_model.pt'
    start=0;best=-float('inf');history=[]
    if checkpoint.is_file():

        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        scaler.load_state_dict(cp['scaler']);start=cp['epoch'];best=cp['best'];history=cp['history'];restore_rng(cp['rng'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(DEVICE)
        del cp
        print('Resume transformer:',folder.name,'epoch',start)
    for epoch in range(start,epochs):
        loader=data_loader(tokenized,CONFIG['text_batch'],shuffle=True,seed=seed+epoch)
        model.train();optimizer.zero_grad(set_to_none=True);total=0
        for step,batch in enumerate(tqdm(loader,desc=f'{folder.name} epoch {epoch+1}/{epochs}',leave=False)):
            y=batch.pop('labels').to(DEVICE);batch={k:v.to(DEVICE,non_blocking=True) for k,v in batch.items()}
            with amp_context():
                logits=model(**batch).logits
                window_start=(step//CONFIG['accumulation'])*CONFIG['accumulation']
                window_length=min(CONFIG['accumulation'],len(loader)-window_start)
                loss=training_loss(logits,y)/window_length
            scaler.scale(loss).backward();total+=float(loss.item())*window_length
            if (step+1)%CONFIG['accumulation']==0 or step+1==len(loader):
                scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();scheduler.step();optimizer.zero_grad(set_to_none=True)
        pv,_=predict_text_model(model,tokenizer,frame.iloc[val_index].text_clean,max_length,return_features=False)
        score=f1_score(frame.iloc[val_index].label_final,pv>=.5,average='macro',zero_division=0)
        improved=score>best+1e-12
        if improved:best=float(score);atomic_torch(cpu_state(model),best_path)
        history.append({'epoch':epoch+1,'fit_loss':total/max(len(loader),1),'early_stop_macro_f1_at_050':float(score)})
        atomic_json(history,folder/'TRAINING_HISTORY.json')
        atomic_torch({'model':cpu_state(model),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'epoch':epoch+1,'best':best,'history':history,'rng':rng_state()},checkpoint)
        print(folder.name,'epoch',epoch+1,'validation F1',round(float(score),4),flush=True)
    model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=True))
    p,embedding=predict_text_model(model,tokenizer,frame.iloc[pred_index].text_clean,max_length)
    result={'ids':frame.iloc[pred_index].master_id.to_numpy(str),'probability':p,'embedding':embedding.astype(np.float32)}
    atomic_npz(result_path,**result)
    atomic_json({'model_id':model_id,'resolved_commit':getattr(model.config,'_commit_hash',None),'seed':seed,
        'epochs':epochs,'max_length':max_length,'selection':'best early-stop-role macro F1 at fixed 0.5; earliest tie',
        'fit_n':len(fit_index),'early_stop_n':len(val_index)},folder/'MODEL_MANIFEST.json')
    tokenizer.save_pretrained(folder/'tokenizer')
    commit_branch(folder)
    checkpoint.unlink(missing_ok=True)
    if not CONFIG['save_best_weights']:best_path.unlink(missing_ok=True)
    del model,optimizer,scheduler,tokenized;cleanup();return result

TRAIN_TRANSFORM=T.Compose([T.Resize((256,256)),T.RandomResizedCrop(224,scale=(.85,1.0)),
 T.ColorJitter(brightness=.08,contrast=.08),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

EVAL_TRANSFORM=T.Compose([T.Resize((256,256)),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

class FrameDataset(Dataset):
    def __init__(self,index,transform):self.index=np.asarray(index,int);self.transform=transform
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        row=VIDEO.iloc[self.index[i]]
        with Image.open(FRAME_DIR/row.frame_name) as im:x=self.transform(im.convert('RGB'))
        return x,int(row.label_final)

@torch.inference_mode()
def predict_visual(model,index):
    model.eval();ps=[];emb=[]
    for x,y in data_loader(FrameDataset(index,EVAL_TRANSFORM),64):
        with amp_context():
            features=model.avgpool(model.features(x.to(DEVICE))).flatten(1)
            logits=model.classifier(features)
        ps.append(logits.float().softmax(-1)[:,1].cpu().numpy());emb.append(features.float().cpu().numpy())
    return np.concatenate(ps),np.concatenate(emb)

def train_visual(fit_index,val_index,seed,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);out=folder/'outputs.npz'
    if branch_finished(folder,VIDEO.master_id):
        with np.load(out,allow_pickle=False) as z:
            assert np.array_equal(z['ids'],VIDEO.master_id.to_numpy(str))
            return {k:z[k] for k in z.files}
    seed_all(seed);model=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():p.requires_grad=False
    for k in (7,8):
        for p in model.features[k].parameters():p.requires_grad=True
    model.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(model.classifier[1].in_features,2));model.to(DEVICE)
    optimizer=torch.optim.AdamW([{'params':list(model.features[7].parameters())+list(model.features[8].parameters()),'lr':2e-5},
        {'params':model.classifier.parameters(),'lr':3e-4}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=(DEVICE.type=='cuda' and not USE_BF16));start=0;best=-float('inf');history=[]
    cp_path=folder/'resume.pt';best_path=folder/'best_model.pt'
    if cp_path.is_file():
        cp=torch.load(cp_path,map_location='cpu',weights_only=False);model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
        scaler.load_state_dict(cp['scaler']);start=cp['epoch'];best=cp['best'];history=cp['history'];restore_rng(cp['rng'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to(DEVICE)
        del cp
    for epoch in range(start,CONFIG['visual_epochs']):
        loader=data_loader(FrameDataset(fit_index,TRAIN_TRANSFORM),CONFIG['visual_batch'],True,seed+epoch)
        model.train()
        for layer in model.modules():
            if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()
        total=0
        for x,y in tqdm(loader,desc=f'{folder.name} visual {epoch+1}',leave=False):
            optimizer.zero_grad(set_to_none=True)
            with amp_context():loss=training_loss(model(x.to(DEVICE)),y.to(DEVICE))
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer);scaler.update();total+=float(loss)
        pv,_=predict_visual(model,val_index);score=f1_score(VIDEO.iloc[val_index].label_final,pv>=.5,average='macro',zero_division=0)
        if score>best+1e-12:best=float(score);atomic_torch(cpu_state(model),best_path)
        history.append({'epoch':epoch+1,'fit_loss':total/max(len(loader),1),'early_stop_macro_f1_at_050':float(score)})
        atomic_json(history,folder/'TRAINING_HISTORY.json')
        atomic_torch({'model':cpu_state(model),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
         'epoch':epoch+1,'best':best,'history':history,'rng':rng_state()},cp_path)
        print(folder.name,'visual epoch',epoch+1,'validation F1',round(float(score),4),flush=True)
    model.load_state_dict(torch.load(best_path,map_location='cpu',weights_only=True))
    p,e=predict_visual(model,np.arange(len(VIDEO)))
    result={'ids':VIDEO.master_id.to_numpy(str),'probability':p,'embedding':e.astype(np.float32)}
    atomic_npz(out,**result)
    atomic_json({'architecture':'torchvision EfficientNet-B0','trainable_blocks':[7,8],'classifier':[.2,1280,2],
      'seed':seed,'epochs':CONFIG['visual_epochs'],'selection':'early-stop role fixed-0.5 macro F1'},folder/'MODEL_MANIFEST.json')
    commit_branch(folder)
    cp_path.unlink(missing_ok=True)
    if not CONFIG['save_best_weights']:best_path.unlink(missing_ok=True)
    del model,optimizer;cleanup();return result

def choose_group_split(frame,index,fraction,seed,require_platforms=False,candidates=240):
    index=np.asarray(index,int);y=frame.label_final.to_numpy(int);g=frame.group_id.to_numpy(str)
    available=set(frame.iloc[index].platform);target=y[index].mean();best=None

    base_props=frame.iloc[index].platform.value_counts(normalize=True)
    for frac in sorted(set([fraction,max(.05,fraction-.04),min(.45,fraction+.04)])):
        splitter=GroupShuffleSplit(n_splits=candidates//3,test_size=frac,random_state=seed)
        for k,(a,b) in enumerate(splitter.split(index,y[index],g[index])):
            fit,hold=index[a],index[b]
            if len(np.unique(y[fit]))<2 or len(np.unique(y[hold]))<2:continue
            if min(np.bincount(y[hold],minlength=2))<3:continue
            if require_platforms and (set(frame.iloc[fit].platform)!=available or set(frame.iloc[hold].platform)!=available):continue
            props=frame.iloc[hold].platform.value_counts(normalize=True).reindex(base_props.index,fill_value=0)
            penalty=4*abs(len(hold)/len(index)-fraction)+abs(y[hold].mean()-target)+float(abs(props-base_props).mean())
            item=(penalty,fit,hold,{'seed':seed,'fraction':frac,'candidate':k})
            if best is None or item[0]<best[0]:best=item
    if best is None:raise RuntimeError('Tidak ditemukan split kelompok yang memenuhi kriteria. Jangan mengganti ke random row split.')
    assert not (set(g[best[1]])&set(g[best[2]]))
    return best[1],best[2],best[3]


# --- Revision sensitivity and repartition training functions ---
#@title Fungsi pelatihan
def load_training_inputs(eng):
    gold=ROOT/'results/v6_gold_adjudicated_v1_1_20261024'
    v=read(gold,'EVALUATION_MANIFEST.csv');r=read(gold,'ROLE_MANIFEST.csv')
    assert len(v)==3951 and v.master_id.is_unique and v.group_id.nunique()==214
    assert v.label_final.value_counts().to_dict()=={1:2326,0:1625}
    for c in ['text_clean','ocr_text','stt_transcript','frame_name','image_sha256','text_present']:
        assert c in v,c
    assert not v[['text_clean','ocr_text','stt_transcript']].fillna('').astype(str).eq('[WITHHELD]').any().any(),'Input training harus asli terbatas.'
    v['text_clean']=v.text_clean.fillna('')
    for f in range(4):eng['validate_roles'](v,r,f)
    folder=ROOT/'results/audio_waveform16_v1_20261024'
    audio_path=locate(folder,'audio_waveform16_aligned_3951.npy')
    assert digest(audio_path)==eng['FROZEN_AUDIO_HASHES'][audio_path.name]
    a=np.load(audio_path,allow_pickle=False)
    ids=read(folder,'ALIGNED_ROW_ID_MANIFEST.csv')
    assert list(ids.master_id)==list(v.master_id) and a.shape==(3951,16) and np.isfinite(a).all()

    p=OOF[OOF.seed==42].set_index('master_id').loc[v.master_id]
    assert np.array_equal(p.label_final,v.label_final) and np.array_equal(p.outer_fold,v.outer_fold)
    eng['VIDEO']=v;eng['PROJECT_ROOT']=ROOT
    return v,r,a

def sensitivity(eng,v,r,a):
    results=[];predictions=[];gold=ROOT/'results/v6_gold_adjudicated_v1_1_20261024'
    for fold in range(4):
        roles=eng['validate_roles'](v,r,fold)
        original=gold/'video/seed_42'/f'fold_{fold}'

        for branch in ['text_branch','visual_branch']:
            marker_path=locate(original/branch,'BRANCH_COMPLETE.json')
            marker=json.loads(marker_path.read_text())
            assert marker['label_version']=='v1.1-GoldAdjudicated'
            assert digest(original/branch/'outputs.npz')==marker['files']['outputs.npz'],'Cached encoder outputs changed'
        tp,te=eng['branch_data'](locate(original/'text_branch','outputs.npz'),v.master_id.to_numpy(str),768)
        vp,ve=eng['branch_data'](locate(original/'visual_branch','outputs.npz'),v.master_id.to_numpy(str),1280)
        missing=v.text_present.to_numpy()==0;te[missing]=0;tp[missing]=.5
        for kind in ['concat_mlp','healnet']:
            for epochs in HEAD_EPOCHS:
                for wd in HEAD_WEIGHT_DECAYS:
                    name=f'{kind}_e{epochs}_wd{wd:g}'
                    dest=CACHE/'sensitivity'/f'fold_{fold}'/name
                    config={'head_epochs':epochs,'head_batch':64,'head_lr':.001,'weight_decay':wd}
                    eng['locked_json'](dest/'TRIAL_LOCK.json',dict(config=config,input_hashes=dict(INPUTS),fold=fold,seed=42,role_hash=digest(gold/'ROLE_MANIFEST.csv')))
                    print('Sensitivity',fold,name,flush=True)
                    probs=eng['train_feature_fusion'](v,[ve,a,te],roles['meta_fit'],42,dest,eng['DEVICE'],kind=kind,config=config)
                    cal=roles['calibration'];test=roles['test'];t=threshold(v.iloc[cal].label_final,probs[cal])
                    z=v.iloc[test][['master_id','group_id','label_final']].copy();z['prediction']=(probs[test]>=t).astype(int)
                    z['fold']=fold;z['model']=name;z['probability']=probs[test];z['threshold']=t;predictions.append(z)
                    results.append(dict(fold=fold,model=name,epochs=epochs,weight_decay=wd,seed=42,threshold=t,meta_groups=v.iloc[roles['meta_fit']].group_id.nunique(),**eng['metrics'](z.label_final,probs[test],z.prediction)))
                    save('HEAD_SENSITIVITY_PROGRESS.csv',pd.DataFrame(results),'R1.3')
    predictions=pd.concat(predictions,ignore_index=True);save('HEAD_SENSITIVITY_PREDICTIONS.csv',predictions,'R1.3')
    rows=[]
    for name,z in predictions.groupby('model'):
        assert len(z)==3951 and z.master_id.is_unique
        rows.append(dict(model=name,pooled_macro_f1=decisions(z.label_final,z.prediction),scope='all grid entries reported; no outer-test best model selected'))
    save('HEAD_SENSITIVITY_POOLED.csv',pd.DataFrame(rows),'R1.3')
    record('R1.3','COMPUTED','Fixed grid neural-head sensitivity; encoders unchanged; no best-test selection')

def allocations(v,eng,seed):
    from sklearn.model_selection import StratifiedGroupKFold
    frame=v.copy();chosen=None
    for s in range(seed,seed+64):
        folds=np.full(len(v),-1,int)
        for f,(_,ix) in enumerate(StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=s).split(v,v.label_final,v.group_id)):folds[ix]=f
        if all(len(v[folds==f])>=300 and v[folds==f].label_final.nunique()==2 and v[folds==f].platform.nunique()==5 for f in range(4)):
            chosen=s;frame.outer_fold=folds;break
    if chosen is None:raise ValueError('No valid source allocation; do not substitute random-window splits.')

    assert frame.groupby('image_sha256').outer_fold.nunique().max()==1
    parts={};rows=[];rules={}
    for f in range(4):
        outer=np.flatnonzero(frame.outer_fold.to_numpy()!=f);test=np.flatnonzero(frame.outer_fold.to_numpy()==f)
        pool,cal,cr=eng['choose_group_split'](frame,outer,.15,seed+500+f)
        base,meta,mr=eng['choose_group_split'](frame,pool,.15/.85,seed+600+f)
        fit,early,er=eng['choose_group_split'](frame,base,.10/.70,seed+700+f)
        part=dict(fit=fit,early_stop=early,meta_fit=meta,calibration=cal,test=test);parts[f]=part
        rules[f]=dict(outer_seed=chosen,calibration=cr,meta_fit=mr,early_stop=er)
        for role,ix in part.items():
            for j in ix:rows.append(dict(master_id=frame.iloc[j].master_id,group_id=frame.iloc[j].group_id,outer_fold=f,role=role))
    manifest=pd.DataFrame(rows)
    for f in range(4):eng['validate_roles'](frame,manifest,f)
    return frame,parts,manifest,rules

def repartition(eng,v,a):
    if not eng['torch'].cuda.is_available():raise RuntimeError('Repeated encoder training requires GPU runtime L4/T4.')
    eng['VIDEO']=v;eng['FRAME_DIR']=eng['stage_frames'](ROOT,v)
    allpred=[];effects=[]
    for split_seed in PARTITION_SEEDS:
        frame,parts,manifest,rules=allocations(v,eng,split_seed);eng['VIDEO']=frame
        base=CACHE/'repartitions'/f'partition_{split_seed}';base.mkdir(parents=True,exist_ok=True)
        save(f'PARTITION_{split_seed}_ROLES.csv',manifest,'R2.5');save(f'PARTITION_{split_seed}_ALLOCATION_RULES.json',rules,'R2.5')
        for fold in range(4):
            role=parts[fold];dest=base/f'fold_{fold}'
            eng['locked_json'](dest/'FOLD_LOCK.json',dict(training_seed=42,partition_seed=split_seed,configuration=eng['CONFIG'],rules=rules,raw_v_sha256=digest(ROOT/'results/v6_gold_adjudicated_v1_1_20261024/EVALUATION_MANIFEST.csv'),manifest_sha256=hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()))
            test=role['test'];saved=dest/'TEST_PREDICTIONS.csv'
            marker=dest/'PREDICTIONS_COMPLETE.json'
            if marker.is_file():
                assert digest(saved)==json.loads(marker.read_text())['sha256'],'Prediksi resume berubah'
                z=pd.read_csv(saved,dtype={'master_id':str,'group_id':str})
                assert list(z.master_id)==list(frame.iloc[test].master_id)
                assert np.array_equal(z.label_final,frame.iloc[test].label_final)
                assert len(z)==len(test) and np.isfinite(z[['text_probability','equal_vat_probability']]).all().all()
                for model in ['text','equal_vat']:
                    assert np.array_equal(z[model+'_prediction'],(z[model+'_probability']>=z[model+'_threshold']).astype(int))
                allpred.append(z);print('Resume partition/fold',split_seed,fold);continue
            fit=role['fit'];early=role['early_stop'];cal=role['calibration']
            tf=fit[frame.iloc[fit].text_present.to_numpy()==1];te=early[frame.iloc[early].text_present.to_numpy()==1]
            assert frame.iloc[tf].label_final.nunique()==frame.iloc[te].label_final.nunique()==2
            print('Train primary contrast partition/fold',split_seed,fold,flush=True)
            tx=eng['train_transformer'](eng['TEXT_MODEL_ID'],frame,tf,te,np.arange(len(frame)),42,dest/'text_branch',192,4)
            vi=eng['train_visual'](fit,early,42,dest/'visual_branch')
            pt=tx['probability'].astype(float);pt[frame.text_present.to_numpy()==0]=.5
            pa=eng['fit_lr'](a[fit],frame.iloc[fit].label_final,frame.iloc[fit].group_id.to_numpy(),C=1.).predict_proba(a)[:,1]
            pv=vi['probability'];vat=(pt+pv+pa)/3
            z=frame.iloc[test][['master_id','group_id','label_final']].copy();z['partition_seed']=split_seed;z['outer_fold']=fold
            for name,p in [('text',pt),('equal_vat',vat)]:
                t=threshold(frame.iloc[cal].label_final,p[cal]);z[name+'_probability']=p[test];z[name+'_prediction']=(p[test]>=t).astype(int);z[name+'_threshold']=t
            eng['atomic_csv'](z,saved);eng['atomic_json']({'sha256':digest(saved)},marker);allpred.append(z)
            save('REPARTITION_PROGRESS.csv',pd.concat(allpred,ignore_index=True),'R2.5')
            del tx,vi;eng['cleanup']()
        d=pd.concat([z for z in allpred if (z.partition_seed==split_seed).all()],ignore_index=True)
        assert len(d)==3951 and d.master_id.is_unique
        effects.append(dict(partition_seed=split_seed,training_seed=42,text_f1=decisions(d.label_final,d.text_prediction),equal_vat_f1=decisions(d.label_final,d.equal_vat_prediction),**paired_ci(d,'equal_vat_prediction','text_prediction',BOOTSTRAP_REPS)))
        save('REPARTITION_EFFECTS.csv',pd.DataFrame(effects),'R2.5')
    save('REPARTITION_PREDICTIONS.csv',pd.concat(allpred,ignore_index=True),'R2.5')

    record('R2.5','COMPUTED','Two complete source allocations; primary contrast only; training seed fixed; exploratory reused cohort')
