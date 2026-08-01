#!/usr/bin/env python3
import hashlib,json,wave
from pathlib import Path
import numpy as np
import torch

OUT=Path("outputs");trace=json.loads((OUT/"actual_audio_filter_cirquent_trace.json").read_text())
graph=json.loads((OUT/trace["hierarchical_graph"]["path"]).read_text());quotient=json.loads((OUT/"audio_frequency_quotient_dags.json").read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert trace["language"]=="ACTUAL-AUDIO-FILTER-CIRQUENT-TRACE-1"
assert sha(OUT/trace["audio"]["path"])==trace["audio"]["sha256"]
assert sha(OUT/trace["hierarchical_graph"]["path"])==trace["hierarchical_graph"]["sha256"]
assert len(trace["whisper_path"]["pcm_q15_400"])==400 and len(trace["whisper_path"]["dft_power_f32_bits"])==201
assert len(trace["whisper_path"]["mel_energy_f32_bits"])==len(trace["whisper_path"]["class80"])==80
assert len(trace["carfac_path"]["nap_frame_energy81_f32_bits"])==81 and len(trace["carfac_path"]["class80"])==80
assert abs(sum(trace["whisper_path"]["mass80"])-1)<2e-6 and abs(sum(trace["carfac_path"]["mass80"])-1)<2e-6
assert trace["whisper_path"]["direct_dft_relative_error"]<5e-6 and trace["whisper_path"]["direct_mel_relative_error"]<5e-6

frame=trace["selection"]["frame"]
states=(OUT/quotient["packed_state_blob"]["path"]).read_bytes()
for method_id,key in [("mel-triangular","whisper_path"),("carfac-cochlea","carfac_path")]:
    method=next(m for m in quotient["methods"] if m["id"]==method_id);b=method["packed_state_block"]
    packed=states[b["offset"]+frame*40:b["offset"]+(frame+1)*40];classes=[]
    for x in packed:classes.extend([x&15,x>>4])
    assert classes==trace[key]["class80"]

# Independently check that the selection rule still chooses this frame.
with wave.open(str(OUT/trace["audio"]["path"]),"rb") as w:pcm=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)/32768
padded=np.zeros(480000,np.float32);padded[:len(pcm)]=pcm
blob=(OUT/graph["coefficient_blob"]["path"]).read_bytes()
def coeff(name):
    b=next(x for x in graph["coefficient_blob"]["blocks"] if x["name"]==name)
    return np.frombuffer(blob,dtype="<f4",count=b["bytes"]//4,offset=b["offset"]).reshape(b["shape"])
stft=torch.stft(torch.from_numpy(padded),400,160,window=torch.from_numpy(coeff("whisper.hann_window").copy()),return_complex=True)
energy=coeff("whisper.mel_weights")@((stft[:,:-1].abs()**2).numpy())
assert int(np.argmax(energy[:,:quotient["active_frames"]].sum(axis=0)))==frame
print(f"ACTUAL_AUDIO_FILTER_CIRQUENT_OK frame={frame} mel_top={trace['whisper_path']['top_mel'][0]['band']} carfac_top={trace['carfac_path']['top_places'][0]['node']}")
