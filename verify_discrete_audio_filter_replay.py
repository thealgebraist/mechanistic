#!/usr/bin/env python3
"""Replay concrete nested detector instances from serialized coefficient bits."""
import json,wave
from pathlib import Path
import numpy as np
import torch
from transformers import WhisperProcessor

OUT=Path("outputs")
g=json.loads((OUT/"hierarchical_audio_filter_probabilistic_graph.json").read_text())
q=json.loads((OUT/"audio_frequency_quotient_dags.json").read_text())
blob=(OUT/g["coefficient_blob"]["path"]).read_bytes()
def block(name):
    b=next(x for x in g["coefficient_blob"]["blocks"] if x["name"]==name)
    return np.frombuffer(blob,dtype="<f4",count=b["bytes"]//4,offset=b["offset"]).reshape(b["shape"])

with wave.open(str(OUT/"whisper_sample_1272-128104-0000.wav"),"rb") as w:
    pcm=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)/32768
padded=np.zeros(480000,dtype=np.float32);padded[:len(pcm)]=pcm
hann=block("whisper.hann_window");cos=block("whisper.dft_cos");minus_sin=block("whisper.dft_minus_sin");mel=block("whisper.mel_weights")
stft=torch.stft(torch.from_numpy(padded),400,160,window=torch.from_numpy(hann.copy()),return_complex=True)
assert tuple(stft.shape)==(201,3001)
power=(stft.abs()**2).numpy()
centered=np.pad(padded,(200,200),mode="reflect")
max_power_rel=0.;max_mel_rel=0.
for frame_index in [0,1,17,127,384,585,3000]:
    samples=centered[frame_index*160:frame_index*160+400]*hann
    real=np.sum(cos*samples[None,:],axis=1,dtype=np.float32)
    imag=np.sum(minus_sin*samples[None,:],axis=1,dtype=np.float32)
    direct=real*real+imag*imag;reference=power[:,frame_index]
    max_power_rel=max(max_power_rel,float(np.linalg.norm(direct-reference)/max(np.linalg.norm(reference),1e-30)))
    direct_mel=np.sum(mel*direct[None,:],axis=1,dtype=np.float32)
    reference_mel=mel@reference
    max_mel_rel=max(max_mel_rel,float(np.linalg.norm(direct_mel-reference_mel)/max(np.linalg.norm(reference_mel),1e-30)))
assert max_power_rel<5e-6 and max_mel_rel<5e-6

# Reconstruct every stored Mel Fin16 state from the actual processor output.
processor=WhisperProcessor.from_pretrained("work/whisper_tiny_en",local_files_only=True)
features=processor(pcm,sampling_rate=16000,return_tensors="pt").input_features[0].numpy()
method=next(m for m in q["methods"] if m["id"]=="mel-triangular")
classes=np.digitize(features[:,:q["active_frames"]],np.asarray(method["quantizer_thresholds"])).astype(np.uint8)
flat=classes.T.reshape(-1);packed=(flat[0::2]|(flat[1::2]<<4)).astype(np.uint8).tobytes()
states=(OUT/q["packed_state_blob"]["path"]).read_bytes();b=method["packed_state_block"]
assert packed==states[b["offset"]:b["offset"]+b["bytes"]]
print(f"DISCRETE_AUDIO_FILTER_REPLAY_OK frames=7 power_rel={max_power_rel:.3g} mel_rel={max_mel_rel:.3g} packed_states={len(packed)}")
