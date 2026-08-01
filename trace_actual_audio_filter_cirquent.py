#!/usr/bin/env python3
"""Trace one real LibriSpeech frame through the hierarchical filter cirquent."""
from __future__ import annotations
import hashlib,html,json,math,subprocess,sys,wave
from pathlib import Path
import numpy as np
import torch
from transformers import WhisperProcessor

OUT=Path("outputs");AUDIO=OUT/"whisper_sample_1272-128104-0000.wav";FS=16000;N=400;HOP=160
CARFAC=Path("work/google_carfac");CARFAC_COMMIT="c74663cc7d05713ae2f2308765eb040530a81c7f"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
graph_path=OUT/"hierarchical_audio_filter_probabilistic_graph.json"
graph=json.loads(graph_path.read_text());quotient=json.loads((OUT/"audio_frequency_quotient_dags.json").read_text())
blob=(OUT/graph["coefficient_blob"]["path"]).read_bytes()
def coeff(name):
    b=next(x for x in graph["coefficient_blob"]["blocks"] if x["name"]==name)
    return np.frombuffer(blob,dtype="<f4",count=b["bytes"]//4,offset=b["offset"]).reshape(b["shape"])
def bits(a):return [f"0x{x:08x}" for x in np.asarray(a,dtype=np.float32).reshape(-1).view(np.uint32)]
def unpack_frame(method_id,frame):
    method=next(m for m in quotient["methods"] if m["id"]==method_id);b=method["packed_state_block"]
    raw=(OUT/quotient["packed_state_blob"]["path"]).read_bytes()[b["offset"]+frame*40:b["offset"]+(frame+1)*40]
    result=[]
    for x in raw:result.extend([x&15,x>>4])
    return result

with wave.open(str(AUDIO),"rb") as w:
    assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(FS,1,2)
    pcm_q15=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2")
pcm=pcm_q15.astype(np.float32)/32768
padded=np.zeros(30*FS,dtype=np.float32);padded[:len(pcm)]=pcm
hann=coeff("whisper.hann_window");dft_cos=coeff("whisper.dft_cos");dft_sin=coeff("whisper.dft_minus_sin");mel=coeff("whisper.mel_weights")
stft=torch.stft(torch.from_numpy(padded),N,HOP,window=torch.from_numpy(hann.copy()),return_complex=True)
assert tuple(stft.shape)==(201,3001)
power=(stft[:,:-1].abs()**2).numpy();mel_energy=mel@power
active_frames=int(quotient["active_frames"])
frame=int(np.argmax(mel_energy[:,:active_frames].sum(axis=0)))
time_seconds=frame*HOP/FS
centered=np.pad(padded,(N//2,N//2),mode="reflect")
frame_samples=centered[frame*HOP:frame*HOP+N]
windowed=frame_samples*hann
real=np.sum(dft_cos*windowed[None,:],axis=1,dtype=np.float32)
imag=np.sum(dft_sin*windowed[None,:],axis=1,dtype=np.float32)
direct_power=real*real+imag*imag
power_ref=power[:,frame]
power_rel=float(np.linalg.norm(direct_power-power_ref)/np.linalg.norm(power_ref))
direct_mel=np.sum(mel*direct_power[None,:],axis=1,dtype=np.float32)
mel_ref=mel_energy[:,frame]
mel_rel=float(np.linalg.norm(direct_mel-mel_ref)/np.linalg.norm(mel_ref))
assert power_rel<5e-6 and mel_rel<5e-6

processor=WhisperProcessor.from_pretrained("work/whisper_tiny_en",local_files_only=True)
source=processor(pcm,sampling_rate=FS,return_tensors="pt")
features=source.input_features[0].numpy();mel_method=next(m for m in quotient["methods"] if m["id"]=="mel-triangular")
mel_classes=np.digitize(features[:,frame],np.asarray(mel_method["quantizer_thresholds"])).astype(np.uint8)
assert mel_classes.tolist()==unpack_frame("mel-triangular",frame)
mel_mass=mel_ref/max(float(mel_ref.sum()),1e-30)
top_dft=np.argsort(direct_power)[-8:][::-1]
top_mel=np.argsort(mel_mass)[-8:][::-1]
mel_nodes=mel_method["nodes"]

assert subprocess.check_output(["git","-C",str(CARFAC),"rev-parse","HEAD"],text=True).strip()==CARFAC_COMMIT
sys.path.insert(0,str(CARFAC/"python/src"))
from carfac.np import carfac
cfp=carfac.carfac_init(carfac.design_carfac(fs=FS,car_params=carfac.CarParams(erb_per_step=.4),ihc_style="two_cap"))
nap,_,bm,ohc,agc=carfac.run_segment(cfp,pcm)
nap=nap[:,:,0];bm=bm[:,:,0];ohc=ohc[:,:,0];agc=agc[:,:,0]
groups=list(reversed([g.tolist() for g in np.array_split(np.arange(81),80)]))
car_energy=np.zeros((80,3000),dtype=np.float32)
for f in range(active_frames):
    chunk=nap[f*HOP:min((f+1)*HOP,len(pcm))]
    for band,g in enumerate(groups):car_energy[band,f]=np.mean(np.square(chunk[:,g],dtype=np.float64))
def whisper_log(e):
    x=np.log10(np.maximum(e,1e-10));x=np.maximum(x,x.max()-8.0);return ((x+4.0)/4.0).astype(np.float32)
car_features=whisper_log(car_energy);car_method=next(m for m in quotient["methods"] if m["id"]=="carfac-cochlea")
car_classes=np.digitize(car_features[:,frame],np.asarray(car_method["quantizer_thresholds"])).astype(np.uint8)
assert car_classes.tolist()==unpack_frame("carfac-cochlea",frame)
car_mass=car_energy[:,frame]/max(float(car_energy[:,frame].sum()),1e-30)
top_car=np.argsort(car_mass)[-8:][::-1]
sample_index=min(frame*HOP+HOP//2,len(pcm)-1)
strongest_group=int(top_car[0]);members=groups[strongest_group]
section=max(members,key=lambda s:float(np.mean(np.square(nap[frame*HOP:min((frame+1)*HOP,len(pcm)),s]))))

trace={
 "language":"ACTUAL-AUDIO-FILTER-CIRQUENT-TRACE-1","audio":{"path":AUDIO.name,"sha256":sha(AUDIO),"samples":len(pcm),"seconds":len(pcm)/FS,"sample_rate":FS},
 "hierarchical_graph":{"path":graph_path.name,"sha256":sha(graph_path)},
 "selection":{"rule":"argmax active-frame sum of actual pre-log Whisper Mel energy","frame":frame,"nominal_time_seconds":time_seconds,"center_sample":sample_index,"frame_rms":float(np.sqrt(np.mean(frame_samples**2))),"frame_peak_q15":int(np.max(np.abs(np.rint(frame_samples*32768).astype(np.int32))))},
 "whisper_path":{"pcm_q15_400":np.rint(frame_samples*32768).astype(np.int32).tolist(),"windowed_f32_bits":bits(windowed),"dft_power_f32_bits":bits(direct_power),"mel_energy_f32_bits":bits(mel_ref),"logmel_f32_bits":bits(features[:,frame]),"mass80":mel_mass.tolist(),"class80":mel_classes.tolist(),
   "top_dft":[{"bin":int(i),"frequency_hz":float(i*FS/N),"power":float(direct_power[i])} for i in top_dft],
   "top_mel":[{"band":int(i),"low_hz":mel_nodes[i]["low_hz"],"high_hz":mel_nodes[i]["high_hz"],"mass":float(mel_mass[i]),"class":int(mel_classes[i])} for i in top_mel],
   "direct_dft_relative_error":power_rel,"direct_mel_relative_error":mel_rel,"transcript":quotient["reference_transcript"],"token_ids":mel_method["token_ids"]},
 "carfac_path":{"nap_frame_energy81_f32_bits":bits(np.mean(np.square(nap[frame*HOP:min((frame+1)*HOP,len(pcm))],dtype=np.float64),axis=0)),"quotient_energy80_f32_bits":bits(car_energy[:,frame]),"mass80":car_mass.tolist(),"class80":car_classes.tolist(),
   "top_places":[{"node":int(i),"members":groups[i],"center_hz":car_method["nodes"][i]["center_hz"],"mass":float(car_mass[i]),"class":int(car_classes[i])} for i in top_car],
   "sample_state":{"sample_index":sample_index,"section":int(section),"pole_frequency_hz":float(cfp.pole_freqs[section]),"pcm_q15":int(pcm_q15[sample_index]),"basilar_motion_f32_bits":bits([bm[sample_index,section]])[0],"nap_f32_bits":bits([nap[sample_index,section]])[0],"ohc_za_state_f32_bits":bits([ohc[sample_index,section]])[0],"agc_zb_state_f32_bits":bits([agc[sample_index,section]])[0]},
   "transcript":car_method["transcript"],"word_error_rate":car_method["word_error_rate_vs_reference"]},
 "proof_boundary":"all displayed frontend values and Fin16 states are replayable for this finite recording; this is not a universal CAR-FAC-to-Whisper equivalence proof"
}
trace_path=OUT/"actual_audio_filter_cirquent_trace.json";trace_path.write_text(json.dumps(trace,indent=2)+"\n")

def rows(items,kind):
    if kind=="dft":return [f"{x['frequency_hz']:.0f} Hz  {x['power']:.3e}" for x in items[:5]]
    return [f"node {x.get('band',x.get('node')):02d}  p={x['mass']:.3f}  q={x['class']:02d}" for x in items[:5]]
W,H=1500,910;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
 '<title id="t">Actual LibriSpeech frame through the filter cirquent</title>',f'<desc id="d">Frame {frame} at {time_seconds:.3f} seconds traced through PCM, DFT, Mel, CAR-FAC, quotient probability, and Whisper outputs.</desc>',
 '<rect width="100%" height="100%" fill="#fbfcfe"/><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#526274"/></marker></defs>',
 '<style>text{font-family:Menlo,monospace;fill:#17212b}.title{font-size:20px;font-weight:bold}.h{font-size:14px;font-weight:bold}.s{font-size:11px;fill:#526274}.box{fill:#edf4ff;stroke:#285c9e;stroke-width:1.5}.prob{fill:#ecf8ef;stroke:#287a42;stroke-width:1.5}.state{fill:#fff4e9;stroke:#a45a24;stroke-width:1.5}.edge{stroke:#526274;stroke-width:1.6;marker-end:url(#a)}</style>',
 '<text x="28" y="34" class="title">One actual speech frame executing the filter cirquent</text>',f'<text x="28" y="58" class="s">LibriSpeech frame {frame} · t={time_seconds:.3f} s · center sample {sample_index} · RMS={trace["selection"]["frame_rms"]:.5f} · selected by maximum Mel energy</text>']
def box(x,y,w,h,title,lines,klass="box"):
    parts.extend([f'<rect class="{klass}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>',f'<text x="{x+14}" y="{y+24}" class="h">{html.escape(title)}</text>'])
    for j,line in enumerate(lines):parts.append(f'<text x="{x+14}" y="{y+47+j*18}" class="s">{html.escape(line)}</text>')
def edge(x1,y1,x2,y2,label=""):
    parts.append(f'<line class="edge" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    if label:parts.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-7}" text-anchor="middle" class="s">{html.escape(label)}</text>')
box(30,120,190,125,"PCM frame",["400 reflected Q15 samples",f"peak={trace['selection']['frame_peak_q15']}",f"first 3: {trace['whisper_path']['pcm_q15_400'][:3]}"])
box(270,105,220,160,"Window + DFT",rows(trace["whisper_path"]["top_dft"],"dft"))
box(540,105,220,160,"Mel detectors",rows(trace["whisper_path"]["top_mel"],"mel"))
box(810,120,220,125,"Mel mass / Fin16",[f"sum p={sum(mel_mass):.9f}",f"top class={int(mel_classes[top_mel[0]])}",f"replay err={mel_rel:.2e}"],"prob")
box(1090,105,380,160,"Whisper token law",["Mr. Quilter is the apostle", "of the middle classes, and we are", "glad to welcome his gospel.",f"tokens={len(mel_method['token_ids'])} · WER=0"],"prob")
edge(220,182,270,182);edge(490,182,540,182);edge(760,182,810,182);edge(1030,182,1090,182,"MODEL_INTERFACE")
box(30,420,190,125,"Same PCM stream",[f"sample[{sample_index}]={int(pcm_q15[sample_index])}","sample-by-sample update","81 cochlear sections"])
box(270,380,250,210,"CAR-FAC state",[f"section={section}",f"pole={cfp.pole_freqs[section]:.1f} Hz",f"BM={trace['carfac_path']['sample_state']['basilar_motion_f32_bits']}",f"NAP={trace['carfac_path']['sample_state']['nap_f32_bits']}",f"OHC={trace['carfac_path']['sample_state']['ohc_za_state_f32_bits']}",f"AGC={trace['carfac_path']['sample_state']['agc_zb_state_f32_bits']}"] ,"state")
box(570,395,220,180,"Cochlear places",rows(trace["carfac_path"]["top_places"],"car"))
box(840,420,220,125,"Place mass / Fin16",[f"sum p={sum(car_mass):.9f}",f"top class={int(car_classes[top_car[0]])}","81 sections -> 80 nodes"],"prob")
box(1120,395,350,180,"Uncalibrated Whisper output",["I'm gonna go to the next one.",f"WER={car_method['word_error_rate_vs_reference']:.3f}","candidate interface is unproved","failure is retained as evidence"],"state")
edge(220,482,270,482);edge(520,482,570,482);edge(790,482,840,482);edge(1060,482,1120,482,"CANDIDATE")
parts.extend([f'<text x="30" y="690" class="h">Finite replay certificate</text>',f'<text x="30" y="720" class="s">direct DFT vs torch FFT relative error: {power_rel:.3e}</text>',f'<text x="30" y="743" class="s">direct Mel vs model Mel relative error: {mel_rel:.3e}</text>',f'<text x="30" y="766" class="s">Mel and CAR-FAC Fin16 vectors exactly match packed frame {frame} in the 117,200-byte state blob.</text>',f'<text x="30" y="820" class="s">Blue: deterministic filter graph · Green: probability/quotient · Orange: recurrent biological state or failed candidate interface</text>',f'<text x="30" y="870" class="s">Scope: this recording and selected frame. No universal alternate-frontend equivalence is claimed.</text>','</svg>'])
(OUT/"actual_audio_filter_cirquent_trace.svg").write_text("\n".join(parts)+"\n")

(OUT/"ACTUAL_AUDIO_FILTER_CIRQUENT_TEST.md").write_text(f"""# Actual audio filter-cirquent test

The input is the real 5.855-second, 16 kHz LibriSpeech waveform `whisper_sample_1272-128104-0000.wav`. Frame **{frame}** at nominal time **{time_seconds:.3f} s** was selected deterministically as the active frame with the largest sum of pre-log Whisper Mel energy.

## Whisper path

- Frame RMS: `{trace['selection']['frame_rms']:.8f}`; Q15 peak: `{trace['selection']['frame_peak_q15']}`.
- Direct serialized-coefficient DFT versus PyTorch FFT relative error: `{power_rel:.9g}`.
- Direct serialized-coefficient Mel energy versus model Mel energy relative error: `{mel_rel:.9g}`.
- The complete 80-channel `Fin 16` state exactly matches the packed state blob.
- Highest-mass Mel node: `{int(top_mel[0])}`, interval `{mel_nodes[top_mel[0]]['low_hz']:.1f}–{mel_nodes[top_mel[0]]['high_hz']:.1f} Hz`, mass `{mel_mass[top_mel[0]]:.6f}`, class `{int(mel_classes[top_mel[0]])}`.
- Whole-recording transcript: “{quotient['reference_transcript']}”

## CAR-FAC path

- The same PCM stream was executed through all 81 official CAR-FAC sections with IHC and closed-loop AGC.
- At sample `{sample_index}`, the strongest selected place uses section `{section}` with pole frequency `{cfp.pole_freqs[section]:.2f} Hz.
- The 81 section energies were quotiented into 80 explicit place nodes; the complete selected-frame `Fin 16` vector exactly matches the packed state blob.
- Highest-mass cochlear node: `{int(top_car[0])}`, mass `{car_mass[top_car[0]]:.6f}`, class `{int(car_classes[top_car[0]])}`.
- Uncalibrated alternate transcript: “{car_method['transcript']}” (`WER={car_method['word_error_rate_vs_reference']:.3f}`).

This is a concrete finite execution certificate. It demonstrates that nested nodes, coefficient-resource edges, state-delay edges, quotient edges, and probability nodes can carry real values. It does not prove that the CAR-FAC interface is semantically equivalent to Whisper's Mel interface.
""")
print(json.dumps({"certificate":"ACTUAL_AUDIO_FILTER_CIRQUENT_TRACED","frame":frame,"time_seconds":time_seconds,"dft_relative_error":power_rel,"mel_relative_error":mel_rel,"mel_top":int(top_mel[0]),"carfac_top":int(top_car[0])},indent=2))
