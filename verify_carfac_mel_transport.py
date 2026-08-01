#!/usr/bin/env python3
import hashlib,json,subprocess,sys,wave
from pathlib import Path
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration,WhisperProcessor

OUT=Path("outputs");MODEL=Path("work/whisper_tiny_en");AUDIO=OUT/"whisper_sample_1272-128104-0000.wav"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();norm=lambda s:"".join(c.lower() if c.isalnum() else " " for c in s).split()
m=json.loads((OUT/"carfac_mel_transport_manifest.json").read_text());s=m["selected"]
assert m["language"]=="CARFAC-MEL-TEMPORAL-TRANSPORT-1" and s["radius"]==2 and s["offsets"]==[-2,-1,0,1,2]
weights_path=OUT/s["weights_path"];assert sha(weights_path)==s["weights_sha256"] and weights_path.stat().st_size==128320
W=np.fromfile(weights_path,dtype="<f4").reshape(s["matrix_shape"]);assert W.shape==(401,80) and W.size==32080
input_path=OUT/s["input_features_path"];assert sha(input_path)==s["input_sha256"] and input_path.stat().st_size==80*3000*4
output_path=OUT/s["output_features_path"];assert sha(output_path)==s["output_sha256"] and output_path.stat().st_size==80*3000*4
stored=np.fromfile(output_path,dtype="<f4").reshape(80,3000)
assert np.isfinite(stored).all()
g=json.loads((OUT/"hierarchical_audio_filter_probabilistic_graph_calibrated.json").read_text())
assert g["transport_resource"]["weights_sha256"]==s["weights_sha256"] and "CARFACMelTemporalTransport" in g["templates"]
assert ["CARFACMelTransport","WhisperNeuralSuffix","MODEL_INTERFACE_SAMPLE_FITTED"] in g["outer_graph"]["edges"]

processor=WhisperProcessor.from_pretrained(MODEL,local_files_only=True);model=WhisperForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32).eval()
with wave.open(str(AUDIO),"rb") as w:pcm=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)/32768
source=processor(pcm,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
carfac_root=Path("work/google_carfac");assert subprocess.check_output(["git","-C",str(carfac_root),"rev-parse","HEAD"],text=True).strip()==m["carfac_commit"]
sys.path.insert(0,str(carfac_root/"python/src"));from carfac.np import carfac
cfp=carfac.carfac_init(carfac.design_carfac(fs=16000,car_params=carfac.CarParams(erb_per_step=.4),ihc_style="two_cap"))
nap,_,_,_,_=carfac.run_segment(cfp,pcm);nap=nap[:,:,0]
groups=list(reversed([x.tolist() for x in np.array_split(np.arange(81),80)]));energy=np.zeros((80,3000),np.float32)
for frame in range(int(np.ceil(len(pcm)/160))):
    chunk=nap[frame*160:min((frame+1)*160,len(pcm))]
    for band,members in enumerate(groups):energy[band,frame]=np.mean(np.square(chunk[:,members],dtype=np.float64))
x=np.log10(np.maximum(energy,1e-10));x=np.maximum(x,x.max()-8);features=((x+4)/4).T.astype(np.float32)
stored_input=np.fromfile(input_path,dtype="<f4").reshape(80,3000).T;assert np.array_equal(features,stored_input)
index=np.arange(3000);A=np.concatenate([*(features[np.clip(index+offset,0,2999)] for offset in s["offsets"]),np.ones((3000,1),np.float32)],axis=1)
replayed=(A@W).T.astype(np.float32);replay_error=float(np.max(np.abs(replayed-stored)));assert replay_error==0
with torch.no_grad():ids=model.generate(torch.from_numpy(replayed.copy()).unsqueeze(0),attention_mask=source.attention_mask,max_new_tokens=64)
text=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
assert ids[0].tolist()==s["token_ids"] and norm(text)==norm(m["target_transcript"]) and text==m["target_transcript"]
assert s["exact_target"] and s["word_error_rate"]==0
print(f"CARFAC_MEL_TRANSPORT_EXACT_TARGET_OK params={W.size} bytes={W.nbytes} replay_error={replay_error} transcript={text}")
