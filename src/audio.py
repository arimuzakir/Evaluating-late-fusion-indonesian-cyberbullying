from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd

def verify(root):
    root=Path(root);folder=root/'data/features/waveform16_v1'
    hashes=json.loads((root/'configs/audio_release_hashes.json').read_text())
    for name,digest in hashes.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    master=pd.read_csv(root/'data/labels/v1_1_GoldAdjudicated/manifests/VIDEO_PRIMARY_MANIFEST.csv')
    rows=pd.read_csv(folder/'ROW_ID_MANIFEST.csv');aligned=pd.read_csv(folder/'ALIGNED_ROW_ID_MANIFEST.csv')
    quality=pd.read_csv(folder/'AUDIO_EXTRACTION_QUALITY.csv').set_index('master_id')
    all_a=np.load(folder/'audio_waveform16_all_4401.npy',allow_pickle=False)
    a=np.load(folder/'audio_waveform16_aligned_3951.npy',allow_pickle=False)
    assert all_a.shape==(4401,16) and a.shape==(3951,16)
    assert np.isfinite(all_a).all() and np.isfinite(a).all()
    assert list(master.master_id)==list(aligned.master_id)
    index=pd.Series(np.arange(len(rows)),index=rows.master_id).loc[master.master_id].to_numpy()
    assert np.array_equal(a,all_a[index])
    missing=quality.loc[rows.master_id].status.eq('SOURCE_HAS_NO_AUDIO_STREAM').to_numpy()
    assert np.array_equal(all_a[:,15]==0,missing) and np.all(all_a[missing]==0)
    q=quality.loc[master.master_id]
    report={'representation':'waveform16_v1','all_rows':4401,'aligned_rows':3951,'features':16,
       'aligned_status_counts':{k:int(v) for k,v in q.status.value_counts().items()},
       'feature_bytes_verified':True,'row_alignment_verified':True,
       'historical_lock_policy':'original LOCK.json retained; current layout validated through audio_release_hashes.json and IDs',
       'labels_modified':False,'features_modified':False}
    out=root/'results/reproduction';out.mkdir(parents=True,exist_ok=True)
    (out/'AUDIO_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report

def rebuild(root,media_root=None,output_dir=None):
    root=Path(root);cfg=json.loads((root/'configs/project_paths.json').read_text())
    media_root=media_root or cfg.get('external_media_root')
    if not media_root:raise ValueError('Provide media_root pointing to authorised source MP4 files.')
    media=Path(media_root).expanduser().resolve()
    label=root/'data/labels/v1_1_GoldAdjudicated'
    meta=pd.read_csv(label/'features/multimodal_video_clean_metadata.csv')
    paths={vid:media/(vid+'.mp4') for vid in sorted(meta.video_id.unique())}
    absent=[str(p) for p in paths.values() if not p.is_file()]
    if absent:raise FileNotFoundError(f'{len(absent)} required source videos missing; first: {absent[0]}')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):raise RuntimeError('FFmpeg and ffprobe required')
    if output_dir is None:raise ValueError('Choose a new output directory; frozen feature files are never overwritten.')
    out=Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be new or empty')
    if (root/'data/features').resolve() in out.parents:raise ValueError('Use a new directory under results/rebuilds')
    from src.utils.waveform import extract_video_windows,SCHEMA,CFG,digest
    stored=pd.read_csv(root/'data/features/waveform16_v1/SOURCE_VIDEO_SHA256.csv')
    expected=dict(zip(stored.video_id,stored.source_sha256)) if 'source_sha256' in stored else dict(zip(stored.video_id,stored.sha256))
    for vid,p in paths.items():assert digest(p)==expected[vid],f'Source bytes changed: {vid}'
    # All mandatory inputs are checked before the first output directory is created.
    out.mkdir(parents=True,exist_ok=True);X=np.zeros((len(meta),16),np.float32);qualities=[]
    for vid,p in paths.items():
        ix=np.flatnonzero(meta.video_id.eq(vid).to_numpy());features,q=extract_video_windows(p,meta.iloc[ix])
        X[ix]=features;qualities.append(q)
    np.save(out/'audio_waveform16_all_4401.npy',X,allow_pickle=False)
    v=pd.read_csv(label/'manifests/VIDEO_PRIMARY_MANIFEST.csv')
    pos=pd.Series(np.arange(len(meta)),index=meta.master_id).loc[v.master_id].to_numpy()
    np.save(out/'audio_waveform16_aligned_3951.npy',X[pos],allow_pickle=False)
    pd.concat(qualities,ignore_index=True).to_csv(out/'AUDIO_EXTRACTION_QUALITY.csv',index=False)
    (out/'EXTRACTION_CONFIG.json').write_text(json.dumps({'config':CFG,'schema':SCHEMA},indent=2)+'\n')
    print('Rebuilt audio in',str(out));return out
