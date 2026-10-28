import numpy as np,pandas as pd,json,hashlib,subprocess,wave,shutil,tempfile
from pathlib import Path

SCHEMA=[
 ('log10_rms_mean','mean(log10(sqrt(mean(frame^2))+1e-10))','log10 amplitude'),
 ('log10_rms_std','population std of log10 RMS','log10 amplitude'),
 ('absolute_peak','max(abs(decoded waveform))','amplitude [-1,1]'),
 ('absolute_mean','mean(abs(decoded waveform))','amplitude [-1,1]'),
 ('zero_crossing_mean','mean fraction of adjacent samples with differing signbit per frame','fraction'),
 ('zero_crossing_std','population std of frame zero crossing fraction','fraction'),
 ('spectral_centroid_mean','mean power-weighted frequency / Nyquist','normalized frequency'),
 ('spectral_centroid_std','population std of normalized centroid','normalized frequency'),
 ('spectral_bandwidth_mean','mean sqrt(power-weighted squared distance to centroid) / Nyquist','normalized frequency'),
 ('spectral_bandwidth_std','population std of normalized bandwidth','normalized frequency'),
 ('spectral_rolloff85_mean','mean first frequency reaching 85 percent cumulative frame power / Nyquist','normalized frequency'),
 ('spectral_rolloff85_std','population std of normalized rolloff','normalized frequency'),
 ('spectral_flatness_mean','mean geometric/arithmetic frame power ratio; zero for zero-power frame','ratio'),
 ('spectral_flatness_std','population std of frame spectral flatness','ratio'),
 ('spectral_flux_mean','mean squared L2 change between successive normalized power spectra','squared ratio'),
 ('audio_present','1 if a decoded source audio stream supplies at least 0.25 second for window; else 0','binary')]

CFG={'representation':'waveform16_v1','sample_rate':16000,'pcm':'signed little-endian 16-bit mono',
 'frame_length':400,'hop_length':160,'fft_length':512,'window':'numpy.hanning symmetric',
 'trailing_frame':'zero pad to next complete frame','epsilon':1e-10,'rolloff_fraction':.85,
 'normalization':'PCM / 32768; no loudness normalization; standardization later fits training role only',
 'missing_audio':'all 16 zeros with explicit audio_present=0; media errors are not silently converted to missing',
 'minimum_decoded_seconds':.25,'source_video_directory':'data/multimodal/video'}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def jsonout(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))

def waveform_features(y):
    y=np.asarray(y,np.float64)
    assert y.ndim==1 and np.isfinite(y).all() and len(y)>=400
    size=CFG['frame_length'];hop=CFG['hop_length'];nfft=CFG['fft_length'];sr=CFG['sample_rate'];eps=CFG['epsilon']
    number=1+max(0,int(np.ceil((len(y)-size)/hop)))
    padded=np.pad(y,(0,max(0,(number-1)*hop+size-len(y))))
    frames=np.lib.stride_tricks.sliding_window_view(padded,size)[::hop][:number]
    rms=np.log10(np.sqrt(np.mean(frames**2,axis=1))+eps)
    zcr=np.mean(np.signbit(frames[:,1:])!=np.signbit(frames[:,:-1]),axis=1)
    power=np.abs(np.fft.rfft(frames*np.hanning(size),n=nfft,axis=1))**2
    total=power.sum(1);valid=total>eps;freq=np.fft.rfftfreq(nfft,1/sr)/(sr/2)
    norm=np.divide(power,total[:,None],out=np.zeros_like(power),where=valid[:,None])
    centroid=(norm*freq).sum(1)
    bandwidth=np.sqrt(np.maximum(0,(norm*(freq[None,:]-centroid[:,None])**2).sum(1)))
    rolloff=freq[np.argmax(np.cumsum(norm,axis=1)>=.85,axis=1)];rolloff[~valid]=0
    flat=np.zeros(len(power));flat[valid]=np.exp(np.mean(np.log(power[valid]+eps),axis=1))/(power[valid].mean(1)+eps)
    flux=np.mean(np.sum(np.diff(norm,axis=0)**2,axis=1)) if len(norm)>1 else 0.
    result=np.array([rms.mean(),rms.std(),np.abs(y).max(),np.abs(y).mean(),zcr.mean(),zcr.std(),
      centroid.mean(),centroid.std(),bandwidth.mean(),bandwidth.std(),rolloff.mean(),rolloff.std(),flat.mean(),flat.std(),flux,1.],np.float32)
    assert result.shape==(16,) and np.isfinite(result).all();return result

def probe_audio(path):
    cp=subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=index','-of','json',str(path)],capture_output=True,text=True)
    if cp.returncode:raise RuntimeError('Cannot probe source '+str(path)+': '+cp.stderr[-500:])
    return bool(json.loads(cp.stdout).get('streams'))

def decode_audio(source,destination):
    cp=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-map','0:a:0','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(destination)],capture_output=True,text=True)
    if cp.returncode:raise RuntimeError('Cannot decode source '+str(source)+': '+cp.stderr[-500:])

def extract_video_windows(video_path,rows):
    # rows contain ONLY IDs/timestamps; no labels, rater votes or predictions enter extraction.
    source_hash=digest(video_path);has=probe_audio(video_path);features=[];quality=[]
    if not has:
        for row in rows.itertuples():
            features.append(np.zeros(16,np.float32));quality.append({'master_id':row.master_id,'video_id':row.video_id,
              'status':'SOURCE_HAS_NO_AUDIO_STREAM','decoded_seconds':0.,'requested_seconds':float(row.end_sec-row.start_sec),'source_sha256':source_hash})
        return np.stack(features),pd.DataFrame(quality)
    with tempfile.TemporaryDirectory(prefix='cyber_audio_decode_') as temp:
        wav=Path(temp)/'decoded.wav';decode_audio(video_path,wav)
        with wave.open(str(wav),'rb') as w:
            assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2
            n=w.getnframes()
            for row in rows.itertuples():
                start=int(round(float(row.start_sec)*16000));end=int(round(float(row.end_sec)*16000))
                assert 0<=start<end
                if start>=n:raise RuntimeError('Window starts after decoded source ends: '+row.master_id)
                w.setpos(start);y=np.frombuffer(w.readframes(min(end,n)-start),dtype='<i2').astype(np.float64)/32768.
                seconds=len(y)/16000
                if seconds<.25:raise RuntimeError('Too little decoded audio; inspect timing: '+row.master_id)
                features.append(waveform_features(y));quality.append({'master_id':row.master_id,'video_id':row.video_id,
                  'status':'OK' if end<=n else 'TRUNCATED_AT_SOURCE_END','decoded_seconds':seconds,
                  'requested_seconds':float(row.end_sec-row.start_sec),'source_sha256':source_hash})
    return np.stack(features),pd.DataFrame(quality)
