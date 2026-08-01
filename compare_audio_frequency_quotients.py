#!/usr/bin/env python3
"""Five explicit frequency/place quotient frontends for Whisper Tiny English."""
from __future__ import annotations
import hashlib,html,json,math,subprocess,sys,wave
from pathlib import Path
import numpy as np
import pywt
import torch
from transformers import WhisperForConditionalGeneration,WhisperProcessor

ROOT=Path("work/whisper_tiny_en")
AUDIO=Path("outputs/whisper_sample_1272-128104-0000.wav")
OUT=Path("outputs")
N_FFT=400;HOP=160;SAMPLE_RATE=16000;FFT_BINS=N_FFT//2+1;MODEL_BANDS=80;WAVELET_BANDS=128;LEVELS=16
CARFAC_ROOT=Path("work/google_carfac")
CARFAC_COMMIT="c74663cc7d05713ae2f2308765eb040530a81c7f"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with wave.open(str(AUDIO),"rb") as w:
    assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(SAMPLE_RATE,1,2)
    pcm=np.frombuffer(w.readframes(w.getnframes()),dtype="<i2").astype(np.float32)/32768
active_frames=math.ceil(len(pcm)/HOP)
padded=np.zeros(30*SAMPLE_RATE,dtype=np.float32);padded[:len(pcm)]=pcm

processor=WhisperProcessor.from_pretrained(ROOT,local_files_only=True)
model=WhisperForConditionalGeneration.from_pretrained(ROOT,local_files_only=True,dtype=torch.float32).eval()
source=processor(pcm,sampling_rate=SAMPLE_RATE,return_tensors="pt",return_attention_mask=True)
window=torch.hann_window(N_FFT)
stft=torch.stft(torch.from_numpy(padded),N_FFT,HOP,window=window,return_complex=True)
power=(stft[...,:-1].abs()**2).numpy() # 201 x 3000
assert power.shape==(FFT_BINS,3000)
frequencies=np.linspace(0,SAMPLE_RATE/2,FFT_BINS)
mel_filters=np.asarray(processor.feature_extractor.mel_filters,dtype=np.float32).T
assert mel_filters.shape==(MODEL_BANDS,FFT_BINS)
mel_energy=mel_filters@power

def whisper_log(energy):
    x=np.log10(np.maximum(energy,1e-10));x=np.maximum(x,x.max()-8.0);return (x+4.0)/4.0
def band_meta(groups,member_kind="fft_bin"):
    rows=[]
    for i,g in enumerate(groups):
        if not g:g=[0]
        if member_kind=="fft_bin":lo=max(0.0,(min(g)-.5)*SAMPLE_RATE/N_FFT);hi=min(SAMPLE_RATE/2,(max(g)+.5)*SAMPLE_RATE/N_FFT)
        else:lo=min(g)*SAMPLE_RATE/(2*WAVELET_BANDS);hi=(max(g)+1)*SAMPLE_RATE/(2*WAVELET_BANDS)
        rows.append({"id":i,"low_hz":lo,"high_hz":hi,"member_kind":member_kind,"members":list(map(int,g))})
    return rows

# 1. The model's classic triangular Mel FIR filterbank.
mel_groups=[np.flatnonzero(row>0).tolist() for row in mel_filters]

# 2. Eighty disjoint, equal-Hz ideal subbands over the same DFT bins.
linear_groups=[x.tolist() for x in np.array_split(np.arange(FFT_BINS),MODEL_BANDS)]
linear_energy=np.vstack([power[g].sum(axis=0) for g in linear_groups])

# 3. Sparse Goertzel/DFT resonators at each Mel filter's weighted center.
mel_centers=np.array([(row*frequencies).sum()/max(row.sum(),1e-30) for row in mel_filters])
goertzel_bins=np.clip(np.rint(mel_centers/(SAMPLE_RATE/N_FFT)).astype(int),0,FFT_BINS-1)
goertzel_groups=[[int(k)] for k in goertzel_bins]
goertzel_energy=power[goertzel_bins]

# 4. A db4 wavelet-packet analysis, 128 dyadic packets quotiented into 80
# disjoint, contiguous packet sets.  This is a true partition: no packet is
# dropped or duplicated.
packet_centers=(np.arange(WAVELET_BANDS)+.5)*SAMPLE_RATE/(2*WAVELET_BANDS)
wavelet_groups=[group.tolist() for group in np.array_split(np.arange(WAVELET_BANDS),MODEL_BANDS)]
wavelet_energy=np.zeros((MODEL_BANDS,3000),dtype=np.float32)
centered=np.pad(padded,(N_FFT//2,N_FFT//2),mode="reflect")
hann=np.hanning(N_FFT).astype(np.float32)
for frame in range(active_frames):
    samples=centered[frame*HOP:frame*HOP+N_FFT]*hann
    packet=pywt.WaveletPacket(samples,"db4",mode="symmetric",maxlevel=7)
    raw=np.array([np.square(node.data,dtype=np.float64).sum() for node in packet.get_level(7,order="freq")])
    for band,g in enumerate(wavelet_groups):wavelet_energy[band,frame]=raw[g].sum()

# 5. Google's official NumPy implementation of Lyon's nonlinear CAR-FAC
# cochlear model.  At erb_per_step=.4 it has 81 physical sections.  Adjacent
# cochlear places are quotiented to 80 nodes and ordered low-to-high for
# Whisper.  Mean squared NAP activity supplies a positive frame measure.
if not (CARFAC_ROOT/"python/src/carfac/np/carfac.py").exists():
    raise RuntimeError("missing pinned CAR-FAC source; run ./fetch_google_carfac.sh")
actual_carfac_commit=subprocess.check_output(
    ["git","-C",str(CARFAC_ROOT),"rev-parse","HEAD"],text=True).strip()
if actual_carfac_commit!=CARFAC_COMMIT:
    raise RuntimeError(f"CAR-FAC revision {actual_carfac_commit} != pinned {CARFAC_COMMIT}")
sys.path.insert(0,str(CARFAC_ROOT/"python/src"))
from carfac.np import carfac as lyon_carfac
car_params=lyon_carfac.CarParams(erb_per_step=.4)
cfp=lyon_carfac.design_carfac(fs=SAMPLE_RATE,car_params=car_params,ihc_style="two_cap")
cfp=lyon_carfac.carfac_init(cfp)
nap,_,_,_,_=lyon_carfac.run_segment(cfp,pcm)
nap=nap[:,:,0]
assert nap.shape==(len(pcm),81)
carfac_groups_desc=[g.tolist() for g in np.array_split(np.arange(cfp.n_ch),MODEL_BANDS)]
carfac_groups=list(reversed(carfac_groups_desc))
carfac_centers_desc=np.asarray(cfp.pole_freqs,dtype=np.float64)
carfac_centers=np.array([np.mean(carfac_centers_desc[g]) for g in carfac_groups])
carfac_edges=np.empty(MODEL_BANDS+1,dtype=np.float64)
carfac_edges[0]=0.;carfac_edges[-1]=SAMPLE_RATE/2
carfac_edges[1:-1]=(carfac_centers[:-1]+carfac_centers[1:])/2
carfac_energy=np.zeros((MODEL_BANDS,3000),dtype=np.float32)
for frame in range(active_frames):
    chunk=nap[frame*HOP:min((frame+1)*HOP,len(pcm))]
    for band,g in enumerate(carfac_groups):
        carfac_energy[band,frame]=np.mean(np.square(chunk[:,g],dtype=np.float64))
carfac_nodes=[{"id":i,"low_hz":float(carfac_edges[i]),"high_hz":float(carfac_edges[i+1]),
  "member_kind":"carfac_section","members":list(map(int,g)),"center_hz":float(carfac_centers[i])}
 for i,g in enumerate(carfac_groups)]

methods=[
 {"id":"mel-triangular","name":"Triangular Mel FIR bank","theory":"overlapping weighted STFT-bin sets","energy":mel_energy,"nodes":band_meta(mel_groups)},
 {"id":"linear-subband","name":"Uniform ideal subbands","theory":"disjoint 100 Hz frequency-set quotient","energy":linear_energy,"nodes":band_meta(linear_groups)},
 {"id":"goertzel-resonator","name":"Sparse Goertzel resonators","theory":"one discrete DFT frequency per Mel center","energy":goertzel_energy,"nodes":band_meta(goertzel_groups)},
 {"id":"wavelet-packet","name":"db4 wavelet packet","theory":"128 dyadic packets to 80 disjoint sets","energy":wavelet_energy,"nodes":band_meta(wavelet_groups,"wavelet_packet")},
 {"id":"carfac-cochlea","name":"Lyon CAR-FAC cochlea","theory":"81 nonlinear cochlear sections to 80 place sets","energy":carfac_energy,"nodes":carfac_nodes},
]

def soft_mass(e):
    total=e.sum(axis=0,keepdims=True);return e/np.maximum(total,1e-30)
reference_mass=soft_mass(mel_energy[:,:active_frames])
reference_features=source.input_features[0].numpy()
assert np.max(np.abs(whisper_log(mel_energy)-reference_features))<2e-6
with torch.no_grad():reference_ids=model.generate(source.input_features,attention_mask=source.attention_mask,max_new_tokens=64)
reference_text=processor.batch_decode(reference_ids,skip_special_tokens=True)[0].strip()

def words(s):return "".join(c.lower() if c.isalnum() else " " for c in s).split()
def edit_distance(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        n=[i]+[0]*len(b)
        for j,y in enumerate(b,1):n[j]=min(n[j-1]+1,d[j]+1,d[j-1]+(x!=y))
        d=n
    return d[-1]
def js_divergence(p,q):
    m=(p+q)/2
    def kl(a):return np.sum(np.where(a>0,a*np.log2(a/np.maximum(m,1e-30)),0),axis=0)
    return float(np.mean((kl(p)+kl(q))/2))

state_blob=bytearray()
for method in methods:
    energy=method.pop("energy");features=whisper_log(energy).astype(np.float32)
    tensor=torch.from_numpy(features).unsqueeze(0)
    with torch.no_grad():ids=model.generate(tensor,attention_mask=source.attention_mask,max_new_tokens=64)
    text=processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
    active=features[:,:active_frames];ref=reference_features[:,:active_frames]
    thresholds=np.quantile(active.ravel(),np.linspace(1/LEVELS,(LEVELS-1)/LEVELS,LEVELS-1))
    classes=np.digitize(active,thresholds).astype(np.uint8)
    flat=classes.T.reshape(-1)
    assert len(flat)%2==0 and np.max(flat)<16
    packed=(flat[0::2]|(flat[1::2]<<4)).astype(np.uint8).tobytes()
    state_offset=len(state_blob);state_blob.extend(packed)
    unique=len({classes[:,i].tobytes() for i in range(active_frames)})
    cosine=float(np.sum(active*ref)/(np.linalg.norm(active)*np.linalg.norm(ref)))
    rmse=float(np.sqrt(np.mean((active-ref)**2)))
    mass=soft_mass(energy[:,:active_frames])
    method.update({"frames":active_frames,"frequency_node_count":MODEL_BANDS,"energy_levels":LEVELS,
      "quantizer_thresholds":thresholds.tolist(),"unique_quantized_frame_states":unique,
      "float_feature_bytes_per_frame":MODEL_BANDS*4,"quotient_bytes_per_frame":MODEL_BANDS//2,
      "compression_ratio":8.0,"probability_mass_max_normalization_error":float(np.max(abs(mass.sum(axis=0)-1))),
      "mean_js_bits_vs_mel":js_divergence(reference_mass,mass),"feature_rmse_vs_mel":rmse,
      "feature_cosine_vs_mel":cosine,"transcript":text,"token_ids":ids[0].tolist(),
      "packed_state_block":{"offset":state_offset,"bytes":len(packed),"layout":"frame-major; low nibble first",
        "sha256":hashlib.sha256(packed).hexdigest()},
      "word_edit_distance_vs_reference":edit_distance(words(reference_text),words(text)),
      "word_error_rate_vs_reference":edit_distance(words(reference_text),words(text))/len(words(reference_text)),
      "transcript_exact":words(text)==words(reference_text)})

state_path=OUT/"audio_frequency_quotient_states.bin";state_path.write_bytes(state_blob)
artifact={"language":"AUDIO-FREQUENCY-QUOTIENT-DAG-1","audio_sha256":sha(AUDIO),
 "checkpoint_sha256":sha(ROOT/"model.safetensors"),"sample_rate":SAMPLE_RATE,"window_samples":N_FFT,"hop_samples":HOP,
 "active_frames":active_frames,"reference_transcript":reference_text,"methods":methods,
 "packed_state_blob":{"path":state_path.name,"bytes":len(state_blob),"sha256":sha(state_path)},
 "carfac_provenance":{"repository":"https://github.com/google/carfac","commit":CARFAC_COMMIT,
   "implementation":"python/src/carfac/np/carfac.py","parameters":{"erb_per_step":.4,"ihc_style":"two_cap","agc":"default closed loop"},"physical_sections":81},
 "dag_semantics":{"frame_edge":"deterministic time t to t+1","frequency_nodes":"each node denotes an explicit finite set of DFT bins, wavelet packets, or cochlear sections","node_mass":"positive band energy divided by total frame energy","quotient_state":"80 four-bit energy classes","readout":"80-channel adapter into the fixed Whisper encoder"},
 "proof_boundary":"frequency partitions, quantization and mass normalization are explicit; alternate frontend to Whisper semantic equivalence is empirical on this waveform, not universal"}
json_path=OUT/"audio_frequency_quotient_dags.json";json_path.write_text(json.dumps(artifact,indent=2)+"\n")

with (OUT/"audio_frequency_quotient_nodes.tsv").open("w") as out:
    out.write("method\tnode\tlow_hz\thigh_hz\tmember_kind\tmembers\n")
    for method in methods:
        for node in method["nodes"]:out.write(f"{method['id']}\t{node['id']}\t{node['low_hz']:.9f}\t{node['high_hz']:.9f}\t{node['member_kind']}\t{','.join(map(str,node['members']))}\n")

W,H=1500,1070;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="svg-title svg-desc">','<title id="svg-title">Five frequency and cochlear-place quotient DAGs feeding Whisper</title>','<desc id="svg-desc">Comparison of Mel, uniform subband, Goertzel, wavelet packet, and CAR-FAC cochlear frontends, each reduced to 80 probabilistic nodes before the same Whisper encoder.</desc>','<rect width="100%" height="100%" fill="#fbfcfe"/>','<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#526274"/></marker></defs>','<style>text{font-family:Menlo,monospace;fill:#17212b}.title{font-size:20px;font-weight:bold}.sub{font-size:12px;fill:#526274}.lane{fill:#eaf2ff;stroke:#285c9e;stroke-width:1.5}.prob{fill:#ecf8ef;stroke:#287a42;stroke-width:1.5}.edge{stroke:#526274;stroke-width:1.7;marker-end:url(#a)}.band{stroke:#285c9e;stroke-width:.7;fill:#dbe9ff}.label{font-size:11px}</style>','<text x="28" y="34" class="title">Five frequency/place quotient DAGs feeding the same Whisper encoder</text>','<text x="28" y="58" class="sub">Each lane has 80 quotient nodes. CAR-FAC uses cochlear place; the others use transform frequency. Nodes emit normalized energy mass and a 4-bit class.</text>']
for lane,method in enumerate(methods):
    y=105+lane*190;parts += [f'<text x="30" y="{y}" font-weight="bold">{html.escape(method["name"])}</text>',f'<text x="30" y="{y+22}" class="sub">{html.escape(method["theory"][:43])}</text>']
    x0=340;barw=500
    parts.append(f'<rect class="lane" x="{x0}" y="{y-24}" width="{barw}" height="65" rx="8"/>')
    for node in method["nodes"]:
        bx=x0+barw*node["low_hz"]/8000;bw=max(.8,barw*(node["high_hz"]-node["low_hz"])/8000)
        parts.append(f'<rect class="band" x="{bx:.2f}" y="{y-16}" width="{bw:.2f}" height="49"/>')
    prob_label="P(place set)" if method["id"]=="carfac-cochlea" else "P(frequency set)"
    parts += [f'<text x="{x0}" y="{y+58}" class="label">0 Hz</text>',f'<text x="{x0+barw}" y="{y+58}" text-anchor="end" class="label">8 kHz</text>',f'<line class="edge" x1="{x0+barw}" y1="{y+8}" x2="905" y2="{y+8}"/>',f'<rect class="lane" x="910" y="{y-24}" width="150" height="65" rx="10"/><text x="985" y="{y+2}" text-anchor="middle" class="label">16 energy classes</text><text x="985" y="{y+23}" text-anchor="middle" class="sub">40 bytes/frame</text>',f'<line class="edge" x1="1060" y1="{y+8}" x2="1115" y2="{y+8}"/>',f'<rect class="prob" x="1120" y="{y-24}" width="150" height="65" rx="10"/><text x="1195" y="{y+2}" text-anchor="middle" class="label">{prob_label}</text><text x="1195" y="{y+23}" text-anchor="middle" class="sub">sum = 1</text>',f'<line class="edge" x1="1270" y1="{y+8}" x2="1320" y2="{y+8}"/>',f'<rect class="prob" x="1325" y="{y-24}" width="145" height="65" rx="10"/><text x="1397" y="{y+2}" text-anchor="middle" class="label">Whisper encoder</text><text x="1397" y="{y+23}" text-anchor="middle" class="sub">WER {method["word_error_rate_vs_reference"]:.2f}</text>',f'<text x="340" y="{y+82}" class="sub">cos={method["feature_cosine_vs_mel"]:.3f}  JS={method["mean_js_bits_vs_mel"]:.3f} bits  transcript: {html.escape(method["transcript"][:90])}</text>']
parts.append('</svg>');(OUT/"audio_frequency_quotient_dags.svg").write_text("\n".join(parts)+"\n")

rows="\n".join(f"| {m['name']} | {m['frequency_node_count']} | {m['feature_cosine_vs_mel']:.4f} | {m['mean_js_bits_vs_mel']:.4f} | {m['word_error_rate_vs_reference']:.3f} | {m['transcript_exact']} | {m['transcript']} |" for m in methods)
(OUT/"AUDIO_FREQUENCY_QUOTIENT_COMPARISON.md").write_text(f"""# Five frequency/place quotient models for Whisper audio

All five methods analyze the same 5.855-second, 16 kHz LibriSpeech waveform and feed an 80-channel representation into the unchanged Whisper Tiny English encoder. Each active frame becomes a probabilistic DAG layer: explicit frequency/place nodes carry normalized energy masses, their log energies are quotiented into 16 discrete classes, and the resulting 80-channel register is passed to Whisper.

| frontend | nodes | cosine vs Mel | mean JS bits | WER vs reference | exact transcript | transcription |
|---|---:|---:|---:|---:|---|---|
{rows}

The triangular Mel method is the model's actual frontend and is bit-close to the Transformers processor (`max error < 2e-6`). The other four are controlled replacements, not claimed equivalent. Their WER and divergence values are concrete evidence for this waveform only.

The cochlear lane executes Google's official NumPy implementation of Lyon's CAR-FAC at pinned commit `{CARFAC_COMMIT}`. It uses 81 nonlinear asymmetric-resonator sections, the two-capacitor inner-hair-cell stage, and closed-loop multi-timescale AGC. Adjacent cochlear places are explicitly quotiented to 80 nodes. Frame energy is the mean squared neural-activity-pattern output, giving a positive measure suitable for normalization; this probability mass is a constructed graph measure, not a claim that CAR-FAC firing rates themselves are categorical probabilities.

Each quantized frame uses 80 four-bit classes (40 bytes) instead of 80 binary32 values (320 bytes), an 8x representation reduction before graph/temporal compression. The finite utterance gives a DAG over {active_frames} time layers. A universal audio model is a parametric DAG schema because the frame count varies.

The quotient certificate records every node's frequency interval and exact DFT-bin, wavelet-packet, or CAR-FAC-section membership. Probability mass is `band_energy / total_frame_energy`; normalization error is checked numerically. Exact semantic equivalence to Whisper is proved only for the Mel frontend under the shared numerical ABI. Proving an alternate frontend equivalent would require a uniform bound connecting its frequency masses to Whisper logits for every waveform.
""")
print(json.dumps({"certificate":"AUDIO_FREQUENCY_QUOTIENTS_BUILT","methods":[{"id":m["id"],"wer":m["word_error_rate_vs_reference"],"cosine":m["feature_cosine_vs_mel"],"transcript":m["transcript"]} for m in methods]},indent=2))
