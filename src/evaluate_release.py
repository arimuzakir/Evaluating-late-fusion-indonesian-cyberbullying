"""Recompute metrics from frozen predictions. No fitting or GPU is used."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,cohen_kappa_score

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p)

def run(root, output_dir=None):
    ROOT=Path(root).resolve()
    LABEL=ROOT/'data/labels/v1_1_GoldAdjudicated'
    VIDEO=ROOT/'results/video/gold_adjudicated_v1.1'
    COMMENTS=ROOT/'results/comments/benchmark_v1.1'
    checks=[]
    def check(name,ok,details):checks.append({'check':name,'pass':bool(ok),'details':details})
    lm=json.loads((LABEL/'FILE_MANIFEST.json').read_text())
    bad=[rel for rel,digest in lm['files'].items() if sha(LABEL/rel)!=digest]
    check('Label package hashes',not bad,{'payload_files':len(lm['files']),'mismatch':bad,'manifest_sha256':sha(LABEL/'FILE_MANIFEST.json')})
    listing=read(ROOT/'docs/reproduction/LABEL_RELEASE_FILE_LIST.csv')
    bad=[r.destination_path for r in listing.itertuples() if sha(ROOT/r.destination_path)!=r.sha256_after]
    check('22-file frozen label inventory',len(listing)==22 and not bad,{'entries':len(listing),'mismatch':bad})

    master=read(LABEL/'data/dataset_master_13307_labeled.csv')
    raw=read(ROOT/'data/annotation_sources/anotasi_mentah_5_anotator.csv')
    meta=read(LABEL/'features/multimodal_video_clean_metadata.csv')
    seg=read(LABEL/'data/multimodal/segmentation_10s_stride3_v2/multimodal_segments.csv')
    v=read(LABEL/'manifests/VIDEO_PRIMARY_MANIFEST.csv')
    roles=read(VIDEO/'ROLE_MANIFEST.csv')
    lineage=read(LABEL/'annotation/LABEL_LINEAGE_ALL_13307.csv')
    change=read(LABEL/'annotation/WINDOW_LABEL_CHANGELOG_4.csv')
    sample=read(LABEL/'annotation/WINDOW_AUDIT_REFERENCE_153.csv')
    weak=read(LABEL/'annotation/HISTORICAL_194_VIDEO_CONTEXT_LINKAGE.csv')
    parents=read(LABEL/'annotation/PARENT_VIDEO_CONTEXT_94.csv')
    counts={}
    for name,df in [('master',master),('comments',master.query("modality_type=='text_corpus'")),('video',meta),('aligned',v)]:
        counts[name]={'n':len(df),'class_counts':df.label_final.value_counts().to_dict()}
    check('Population counts',len(master)==13307 and len(meta)==4401 and len(v)==3951 and meta.video_id.nunique()==233 and v.group_id.nunique()==214,counts)
    for name,df in [('meta',meta),('segments',seg),('aligned',v),('ocr',read(LABEL/'data/dataset_ocr_video_labeled.csv'))]:
        match=df.set_index('master_id').label_final.eq(master.set_index('master_id').loc[df.master_id,'label_final'])
        check('Labels match master: '+name,df.master_id.is_unique and match.all(),{'n':len(df),'mismatch':int((~match).sum())})
    cols=[f'anotator_{i}' for i in range(1,6)]
    check('Raw votes unchanged',master.set_index('master_id')[cols].equals(raw.set_index('master_id')[cols]),{'cells':len(master)*5})
    votes=master[cols].to_numpy(int);majority=(votes.sum(1)>=3).astype(int)
    overrides=master.loc[majority!=master.label_final,'master_id'].tolist()
    check('Exactly four majority overrides',set(overrides)==set(change.master_id),{'overrides':overrides})
    check('Lineage labels',np.array_equal(lineage.label_final_v1_1,master.set_index('master_id').loc[lineage.master_id].label_final),{'n':len(lineage),'changed':int((lineage.label_final_original!=lineage.label_final_v1_1).sum())})
    weakm=master.set_index('master_id').loc[weak.master_id]
    check('Historical 194/94 scope',len(weak)==194 and len(parents)==94 and int((weakm.label_final==0).sum())==179,{'rows':len(weak),'parent_rows':len(parents),'negative':int((weakm.label_final==0).sum())})
    k=cohen_kappa_score(sample.window_label_expert1,sample.window_label_expert2)
    alignment=bool(sample.reference_label_v1_1.eq(sample.adjudicated_window_label).all())
    agreements=int(sample.window_label_expert1.eq(sample.window_label_expert2).sum())
    check('153-window agreement',len(sample)==153 and agreements==145 and np.isclose(k,.8954203691045797) and alignment,{'n':len(sample),'kappa':k,'agreements':agreements,'reference_alignment':alignment})
    for fold,rr in roles.groupby('outer_fold'):
        check('Role isolation fold '+str(fold),rr.groupby('group_id').role.nunique().max()==1 and rr.master_id.nunique()==len(v),{'rows':len(rr),'group_max_roles':int(rr.groupby('group_id').role.nunique().max()),'roles':rr.role.value_counts().to_dict()})
    check('Outer fold source groups isolated',v.groupby('group_id').outer_fold.nunique().max()==1,{'groups':v.group_id.nunique(),'videos':meta.set_index('master_id').loc[v.master_id].video_id.nunique()})
    check('Image hashes disjoint across folds',v.groupby('image_sha256').outer_fold.nunique().max()==1,{'unique_image_hashes':v.image_sha256.nunique()})

    metrics=read(VIDEO/'METRICS_BY_SEED.csv');oof=read(VIDEO/'VIDEO_OOF_PREDICTIONS.csv');summary=read(VIDEO/'THREE_SEED_SUMMARY.csv')
    differences=[]
    for row in metrics.itertuples():
        d=oof[oof.seed==row.seed];truth=d.label_final.to_numpy();prob=d[row.model+'_probability'].to_numpy();pred=d[row.model+'_prediction'].to_numpy();threshold=d[row.model+'_threshold'].to_numpy()
        values={'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,average='macro'),'roc_auc':roc_auc_score(truth,prob),'average_precision':average_precision_score(truth,prob),'macro_f1_fixed_050':f1_score(truth,prob>=.5,average='macro')}
        values.update(zip(['tn','fp','fn','tp'],confusion_matrix(truth,pred,labels=[0,1]).ravel()))
        for name,value in values.items():
            if abs(value-getattr(row,name))>1e-8:differences.append((row.seed,row.model,name,float(value),float(getattr(row,name))))
        if not np.array_equal(pred,(prob>=threshold).astype(int)):differences.append((row.seed,row.model,'threshold_predictions'))
    check('All 63 video metric rows recomputed',len(metrics)==63 and not differences,{'rows':len(metrics),'models':metrics.model.nunique(),'seeds':sorted(metrics.seed.unique().tolist()),'oof_rows':len(oof),'differences':differences})
    diff=[]
    for row in summary.itertuples():
        mm=metrics[metrics.model==row.model]
        for name in ['macro_f1','accuracy','roc_auc','average_precision','brier_score']:
            for kind,value in [('mean',mm[name].mean()),('std',mm[name].std(ddof=1))]:
                if abs(value-getattr(row,name+'_'+kind))>1e-9:diff.append((row.model,name,kind))
    check('All 21 three-seed summaries recomputed',not diff,{'rows':len(summary),'differences':diff})
    check('OOF labels match master',oof.master_id.map(master.set_index('master_id').label_final).eq(oof.label_final).all(),{'rows':len(oof)})
    cp=read(COMMENTS/'TEXT_TEST_PREDICTIONS.csv');cm=read(COMMENTS/'TEXT_TEST_METRICS.csv');diff=[]
    for r in cm.itertuples():
        d=cp[(cp.model==r.model)&(cp.seed==r.seed)]
        for name,val in [('macro_f1',f1_score(d.label_final,d.prediction,average='macro')),('accuracy',accuracy_score(d.label_final,d.prediction)),('roc_auc',roc_auc_score(d.label_final,d.probability)),('average_precision',average_precision_score(d.label_final,d.probability))]:
            if abs(val-getattr(r,name))>1e-9:diff.append((r.model,r.seed,name))
    check('All seven comment result rows recomputed',not diff,{'rows':len(cm),'predictions':len(cp),'n_per_run':sorted(cp.groupby(['model','seed']).size().unique().tolist()),'differences':diff})
    split=read(ROOT/'data/manifests/comments/TEXT_SPLIT_MANIFEST.csv')
    check('Comment source/text isolation',split.groupby('group_id').split.nunique().max()==1 and split.groupby('text_sha256').split.nunique().max()==1,{'records':len(split),'split_counts':split.split.value_counts().to_dict()})
    check('Comment held-out labels and IDs',set(cp.master_id)==set(split.loc[split.split=='test','master_id']) and cp.label_final.eq(cp.master_id.map(master.set_index('master_id').label_final)).all(),{'unique_test_ids':cp.master_id.nunique()})

    lp=read(VIDEO/'LOPO_FROZEN_FEATURE_PROBE_PREDICTIONS.csv');lmets=read(VIDEO/'LOPO_FROZEN_FEATURE_PROBE_METRICS.csv');diff=[]
    for row in lmets.itertuples():
        d=lp[(lp.platform==row.held_out_platform)&(lp.model==row.model)]
        for name,val in [('macro_f1',f1_score(d.label_final,d.prediction,average='macro')),('accuracy',accuracy_score(d.label_final,d.prediction)),('roc_auc',roc_auc_score(d.label_final,d.probability)),('average_precision',average_precision_score(d.label_final,d.probability))]:
            if abs(val-getattr(row,name))>1e-9:diff.append((row.held_out_platform,row.model,name))
    check('All 20 frozen-feature LOPO metric rows recomputed',not diff,{'rows':len(lmets),'predictions':len(lp),'differences':diff})
    lroles=read(VIDEO/'LOPO_ROLE_MANIFEST.csv')
    check('LOPO role source isolation',all(rr.groupby('group_id').role.nunique().max()==1 for _,rr in lroles.groupby('held_out_platform')),{'held_out_platforms':lroles.held_out_platform.nunique()})

    a=np.load(ROOT/'data/features/waveform16_v1/audio_waveform16_aligned_3951.npy',allow_pickle=False)
    qa=read(ROOT/'data/features/waveform16_v1/AUDIO_EXTRACTION_QUALITY.csv').set_index('master_id').loc[v.master_id]
    check('Aligned audio shape/status',a.shape==(3951,16) and np.isfinite(a).all(),{'shape':list(a.shape),'status_counts':qa.status.value_counts().to_dict(),'zero_absent_rows':int(np.all(a==0,axis=1).sum())})


    output=Path(output_dir) if output_dir else ROOT/'results/reproduction'
    output.mkdir(parents=True,exist_ok=True)
    report={'scope':'stored predictions and released label records; no training or independent human validation',
            'checks':checks,'passed':sum(c['pass'] for c in checks),'failed':sum(not c['pass'] for c in checks)}
    (output/'REPRODUCTION_CHECKS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=lambda x:int(x) if isinstance(x,np.integer) else float(x))+'\n')
    pd.DataFrame([{'check':c['check'],'status':'PASS' if c['pass'] else 'FAIL'} for c in checks]).to_csv(output/'REPRODUCTION_CHECKS.csv',index=False)
    print(f"Numerical checks: {report['passed']}/{len(checks)} passed")
    if report['failed']:
        raise AssertionError([c for c in checks if not c['pass']])
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output-dir')
    a=p.parse_args();run(a.root,a.output_dir)
