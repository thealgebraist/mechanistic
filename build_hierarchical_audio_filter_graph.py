#!/usr/bin/env python3
"""Build a graph-of-graphs for actual Whisper and CAR-FAC audio detectors."""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import html
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperProcessor

OUT=Path("outputs")
WHISPER=Path("work/whisper_tiny_en")
CARFAC=Path("work/google_carfac")
CARFAC_COMMIT="c74663cc7d05713ae2f2308765eb040530a81c7f"
N=400;BINS=201;BANDS=80;FS=16000;HOP=160;ANALYSIS_FRAMES=3001;OUTPUT_FRAMES=3000

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

class Blob:
    def __init__(self):self.data=bytearray();self.blocks=[]
    def add(self,name:str,array,semantic:str):
        a=np.asarray(array,dtype="<f4")
        payload=a.tobytes(order="C")
        block={"name":name,"offset":len(self.data),"bytes":len(payload),"dtype":"F32BitsLE",
          "shape":list(a.shape),"sha256":hashlib.sha256(payload).hexdigest(),"semantic":semantic,
          "finite":bool(np.isfinite(a).all())}
        self.data.extend(payload);self.blocks.append(block);return block

processor=WhisperProcessor.from_pretrained(WHISPER,local_files_only=True)
hann=torch.hann_window(N,dtype=torch.float32).numpy()
mel=np.asarray(processor.feature_extractor.mel_filters,dtype=np.float32).T
assert mel.shape==(BANDS,BINS)
n=np.arange(N,dtype=np.float64)[None,:]
k=np.arange(BINS,dtype=np.float64)[:,None]
dft_cos=np.cos(2*math.pi*k*n/N).astype(np.float32)
dft_sin=(-np.sin(2*math.pi*k*n/N)).astype(np.float32)

actual_commit=subprocess.check_output(["git","-C",str(CARFAC),"rev-parse","HEAD"],text=True).strip()
assert actual_commit==CARFAC_COMMIT
sys.path.insert(0,str(CARFAC/"python/src"))
from carfac.np import carfac
cfp=carfac.carfac_init(carfac.design_carfac(fs=FS,car_params=carfac.CarParams(erb_per_step=.4),ihc_style="two_cap"))
assert cfp.n_ch==81

blob=Blob()
blob.add("whisper.hann_window",hann,"periodic 400-sample analysis window used by torch.stft")
blob.add("whisper.dft_cos",dft_cos,"direct real DFT detector coefficients for bins 0..200")
blob.add("whisper.dft_minus_sin",dft_sin,"direct imaginary DFT detector coefficients for bins 0..200")
blob.add("whisper.mel_weights",mel,"actual 80 by 201 Whisper feature-extractor Mel matrix")
blob.add("carfac.pole_freqs",cfp.pole_freqs,"81 physical cochlear-section pole frequencies")

scalars={}
def collect(prefix,obj):
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):collect(f"{prefix}.{field.name}",getattr(obj,field.name))
    elif isinstance(obj,np.ndarray):
        if obj.dtype.kind in "biufc":blob.add(prefix,obj,f"official CAR-FAC designed coefficient/state-independent array {prefix}")
    elif isinstance(obj,(list,tuple)):
        for i,value in enumerate(obj):collect(f"{prefix}.{i}",value)
    elif isinstance(obj,(bool,int,float,str)) or obj is None:
        scalars[prefix]=obj

collect("carfac.car",cfp.ears[0].car_coeffs)
collect("carfac.ihc",cfp.ears[0].ihc_coeffs)
collect("carfac.agc",cfp.ears[0].agc_coeffs)

coeff_path=OUT/"audio_filter_coefficients.bin";coeff_path.write_bytes(blob.data)
with (OUT/"audio_filter_coefficient_blocks.tsv").open("w") as out:
    out.write("name\toffset\tbytes\tdtype\tshape\tsha256\n")
    for b in blob.blocks:out.write(f"{b['name']}\t{b['offset']}\t{b['bytes']}\t{b['dtype']}\t{'x'.join(map(str,b['shape']))}\t{b['sha256']}\n")

base_path=OUT/"whisper_tiny_en_probabilistic_graph.json"
base=json.loads(base_path.read_text())
quotient_path=OUT/"audio_frequency_quotient_dags.json"
quotient=json.loads(quotient_path.read_text())
state_path=OUT/quotient["packed_state_blob"]["path"]

templates={
 "OrderedF32Sum":{"parameters":["length:Nat"],"nodes":["input:F32Bits[length]","accumulator:F32Bits","output:F32Bits"],
  "edges":[["input","accumulator","ORDERED_REDUCE"],["accumulator","accumulator","STATE_DELAY"],["accumulator","output","DATA"]]},
 "DFTBinDetector":{"parameters":["bin:Fin201"],"nodes":["windowed:F32Bits[400]","cos:F32Bits[400]","minus_sin:F32Bits[400]","real_products","imag_products","real_sum","imag_sum","power"],
  "contains":[{"template":"OrderedF32Sum","instances":2,"length":400}],
  "edges":[["windowed","real_products","TAP_PRODUCT"],["cos","real_products","PARAMETER_BITS"],["windowed","imag_products","TAP_PRODUCT"],["minus_sin","imag_products","PARAMETER_BITS"],["real_products","real_sum","GRAPH_INSTANCE"],["imag_products","imag_sum","GRAPH_INSTANCE"],["real_sum","power","SQUARE_ADD"],["imag_sum","power","SQUARE_ADD"]]},
 "MelBandDetector":{"parameters":["band:Fin80"],"nodes":["dft_power:F32Bits[201]","mel_row:F32Bits[201]","weighted_bins","band_energy"],
  "contains":[{"template":"OrderedF32Sum","instances":1,"length":"nonzero(mel_row)"}],
  "edges":[["dft_power","weighted_bins","TAP_PRODUCT"],["mel_row","weighted_bins","PARAMETER_BITS"],["weighted_bins","band_energy","GRAPH_INSTANCE"]]},
 "WhisperFrameFrontend":{"parameters":["frame:Fin3001"],"nodes":["center_reflect_pad","frame_slice:Fin65536[400]","f32_samples","windowed","dft_bank","mel_energy80"],
  "contains":[{"template":"DFTBinDetector","instances":201},{"template":"MelBandDetector","instances":80}],
  "edges":[["center_reflect_pad","frame_slice","FRAME_SLICE"],["frame_slice","f32_samples","DISCRETE_CAST"],["f32_samples","windowed","TAP_PRODUCT"],["windowed","dft_bank","GRAPH_INSTANCE"],["dft_bank","mel_energy80","GRAPH_INSTANCE"]]},
 "WhisperWindowFrontend":{"parameters":["window:480000 samples"],"nodes":["pcm_q15","center_reflect_pad","mel_energy80x3001","drop_last_frame","log10","utterance_global_max","eight_decibel_floor","affine_normalize","LogMel80x3000"],
  "contains":[{"template":"WhisperFrameFrontend","instances":3001}],
  "edges":[["pcm_q15","center_reflect_pad","REFLECT_PAD"],["center_reflect_pad","mel_energy80x3001","GRAPH_INSTANCE"],["mel_energy80x3001","drop_last_frame","DROP_LAST"],["drop_last_frame","log10","NONLINEAR"],["log10","utterance_global_max","UTTERANCE_GLOBAL"],["utterance_global_max","eight_decibel_floor","UTTERANCE_GLOBAL"],["log10","eight_decibel_floor","DATA"],["eight_decibel_floor","affine_normalize","AFFINE"],["affine_normalize","LogMel80x3000","MODEL_INTERFACE"]]},
 "CARFACSection":{"parameters":["place:Fin81"],"state":["z1:F32Bits","z2:F32Bits","za:F32Bits","zb:F32Bits","g:F32Bits"],
  "nodes":["cascade_in","interpolate_g_zb","velocity","ohc_nonlinearity","radius","rotate_state","asymmetric_zero","cascade_out"],
  "edges":[["cascade_in","rotate_state","CASCADE_PLACE"],["z1","rotate_state","STATE_DELAY"],["z2","velocity","STATE_DELAY"],["za","velocity","STATE_DELAY"],["velocity","ohc_nonlinearity","NONLINEAR"],["ohc_nonlinearity","radius","OHC_FEEDBACK"],["radius","rotate_state","PARAMETER_MODULATION"],["rotate_state","asymmetric_zero","DATA"],["asymmetric_zero","cascade_out","CASCADE_PLACE"]]},
 "InnerHairCell":{"parameters":["place:Fin81"],"state":["ac_coupler","cap1","cap2","lpf1","lpf2"],
  "nodes":["basilar_motion","ac_difference","conductance_detector","capacitor_update","lowpass","neural_activity"],
  "edges":[["basilar_motion","ac_difference","DATA"],["ac_coupler","ac_difference","STATE_DELAY"],["ac_difference","conductance_detector","NONLINEAR"],["conductance_detector","capacitor_update","STATE_DELAY"],["capacitor_update","lowpass","STATE_DELAY"],["lowpass","neural_activity","DATA"]]},
 "AGCStage":{"parameters":["stage:Fin4","place:Fin81"],"state":["decimation_phase","input_accum","agc_memory"],
  "nodes":["neural_activity","decimate","temporal_iir","spatial_fir","damping_feedback"],
  "edges":[["neural_activity","decimate","AGC_FEEDBACK"],["decimate","temporal_iir","STATE_DELAY"],["temporal_iir","spatial_fir","NEIGHBOR_PLACE"],["spatial_fir","damping_feedback","AGC_FEEDBACK"]]},
 "CARFACSampleFrontend":{"parameters":["sample:Nat"],"nodes":["pcm_sample","car_cascade","ihc_bank","agc_bank","NAP81"],
  "contains":[{"template":"CARFACSection","instances":81},{"template":"InnerHairCell","instances":81},{"template":"AGCStage","instances":324}],
  "edges":[["pcm_sample","car_cascade","SAMPLE_FLOW"],["car_cascade","ihc_bank","GRAPH_INSTANCE"],["ihc_bank","agc_bank","GRAPH_INSTANCE"],["agc_bank","car_cascade","AGC_FEEDBACK"],["ihc_bank","NAP81","DATA"]]},
 "CARFACFrameEnergy":{"parameters":["frame:Nat","place:Fin81"],"nodes":["NAP160","squared_activity","ordered_sum","mean_energy"],
  "contains":[{"template":"OrderedF32Sum","instances":1,"length":160}],
  "edges":[["NAP160","squared_activity","SQUARE"],["squared_activity","ordered_sum","GRAPH_INSTANCE"],["ordered_sum","mean_energy","AFFINE"]]},
 "CARFACWindowFrontend":{"parameters":["samples:Nat"],"nodes":["pcm_q15","NAP_stream81","frame_energy81","zero_pad_to_3000"],
  "contains":[{"template":"CARFACSampleFrontend","instances":"samples"},{"template":"CARFACFrameEnergy","instances":"ceil(samples/160) x 81"}],
  "edges":[["pcm_q15","NAP_stream81","GRAPH_INSTANCE"],["NAP_stream81","frame_energy81","GRAPH_INSTANCE"],["frame_energy81","zero_pad_to_3000","PAD_FRAMES"]]},
 "ProbabilityQuotient":{"parameters":["frame:Nat","method:Fin5"],"nodes":["fine_activity","membership_relation","positive_energy","total_energy","mass80","class80"],
  "edges":[["fine_activity","positive_energy","POSITIVE_MEASURE"],["membership_relation","positive_energy","QUOTIENT_MEMBERSHIP"],["positive_energy","total_energy","ORDERED_REDUCE"],["total_energy","mass80","PROBABILITY_NORMALIZE"],["mass80","class80","QUANTIZE"]]}
}

edge_kinds={
 "SAMPLE_FLOW":"same-time audio sample propagation","STATE_DELAY":"state from time t to t+1",
 "PARAMETER_BITS":"immutable finite coefficient bit vector","GRAPH_INSTANCE":"invoke a nested graph template",
 "CASCADE_PLACE":"same-time propagation from basal section j to apical section j+1",
 "NEIGHBOR_PLACE":"spatial coupling between adjacent cochlear places","OHC_FEEDBACK":"fast local outer-hair-cell feedback",
 "AGC_FEEDBACK":"delayed spatial/temporal gain-control feedback","QUOTIENT_MEMBERSHIP":"fine detector belongs to an explicit quotient set",
 "POSITIVE_MEASURE":"convert detector activity to nonnegative energy","PROBABILITY_NORMALIZE":"divide node energy by frame total",
 "MODEL_INTERFACE":"feed the fixed 80-channel neural suffix","ORDERED_REDUCE":"fixed left-to-right binary32 reduction",
 "TAP_PRODUCT":"coefficient-weighted sample or bin","DISCRETE_CAST":"finite Q15 code to binary32 bit pattern",
 "NONLINEAR":"named deterministic finite-bit operation","FRAME_GLOBAL":"operation depending on all values in a frame",
 "AFFINE":"fixed finite-bit affine map","SQUARE_ADD":"squared magnitude","DATA":"ordinary deterministic data edge",
 "PARAMETER_MODULATION":"state-dependent coefficient selection","QUANTIZE":"map to Fin16","REFLECT_PAD":"center padding by reflected samples",
 "FRAME_SLICE":"select 400 samples at a 160-sample hop","DROP_LAST":"remove centered STFT frame 3000","UTTERANCE_GLOBAL":"dependency on all 80x3000 values",
 "SQUARE":"finite-bit activity square","PAD_FRAMES":"zero-pad active frame energies to 3000","MODEL_INTERFACE_CANDIDATE_UNPROVED":"alternate unproved 80-channel interface"}

mel_nnz=int(np.count_nonzero(mel))
dft_ops_per_bin=2*N+2*(N-1)+3
flat_mel_frame=N+BINS*dft_ops_per_bin+(2*mel_nnz-BANDS)+5*BANDS
active_frames=int(quotient["active_frames"])
hierarchy={
 "language":"HIERARCHICAL-DISCRETE-AUDIO-PROBABILISTIC-GRAPH-1",
 "value_types":{"SampleQ15":"Fin 65536","F32Bits":"BitVec 32","EnergyClass":"Fin 16","FrequencyNode":"Fin 80","CochlearPlace":"Fin 81","Token":"Fin 51864"},
 "edge_kinds":edge_kinds,"templates":templates,
 "coefficient_blob":{"path":coeff_path.name,"bytes":len(blob.data),"sha256":sha(coeff_path),"blocks":blob.blocks,"scalar_parameters":scalars},
 "sample_state_blob":{"path":state_path.name,"bytes":state_path.stat().st_size,"sha256":sha(state_path),"layout":quotient["packed_state_blob"]},
 "outer_graph":{"nodes":[
   {"id":"PCM","type":"Stream SampleQ15"},
   {"id":"WhisperMel","type":"HyperNode","contains_graph":"WhisperWindowFrontend","instances":1},
   {"id":"CARFAC","type":"HyperNode","contains_graph":"CARFACWindowFrontend","instances":1},
   {"id":"MelQuotient","type":"HyperNode","contains_graph":"ProbabilityQuotient","instances":"frames"},
   {"id":"CochlearQuotient","type":"HyperNode","contains_graph":"ProbabilityQuotient","instances":"frames"},
   {"id":"WhisperNeuralSuffix","type":"HyperNode","contains_graph":"BaseWhisper74Ops","instances":1},
   {"id":"TokenLaw","type":"Categorical Token"}],
  "edges":[["PCM","WhisperMel","SAMPLE_FLOW"],["PCM","CARFAC","SAMPLE_FLOW"],["WhisperMel","MelQuotient","GRAPH_INSTANCE"],["CARFAC","CochlearQuotient","GRAPH_INSTANCE"],["MelQuotient","WhisperNeuralSuffix","MODEL_INTERFACE"],["CochlearQuotient","WhisperNeuralSuffix","MODEL_INTERFACE_CANDIDATE_UNPROVED"],["WhisperNeuralSuffix","TokenLaw","PROBABILITY_NORMALIZE"]]},
 "base_whisper":{"path":base_path.name,"sha256":sha(base_path),"checkpoint_sha256":base["checkpoint_sha256"],"opcode_count":base["opcode_count"],"tensor_count":base["tensor_count"],"ops":base["ops"]},
 "instance_families":{"stft_analysis_frames":ANALYSIS_FRAMES,"output_frames":OUTPUT_FRAMES,"dft_bins_per_frame":BINS,"mel_bands_per_frame":BANDS,"carfac_sections_per_sample":81,"ihc_detectors_per_sample":81,"agc_stage_places_per_sample":324,"active_frames":active_frames,"padded_samples":480000},
 "flattening_estimates":{"mel_scalar_nodes_per_frame":flat_mel_frame,"mel_scalar_nodes_active_sample":flat_mel_frame*active_frames,"mel_scalar_nodes_30_seconds":flat_mel_frame*ANALYSIS_FRAMES,
   "carfac_scalar_nodes_per_sample_estimate":81*35+81*15+324*8,"carfac_scalar_nodes_30_seconds_estimate":(81*35+81*15+324*8)*480000,
   "note":"counts expose flattening cost but are not exact backend instruction counts"},
 "semantic_status":{"coefficient_bits":"explicit and finite","mel_filter_membership":"actual Whisper frontend coefficients","dft":"explicit ordered direct-DFT graph; backend bit equality to torch FFT remains unproved","carfac":"actual designed coefficient arrays and source-shaped update graph; source/backend equivalence remains unproved","probability":"positive frame measures and Fin16 sample traces are explicit","whisper_suffix":"74-op graph references original checkpoint tensors"}
}
graph_path=OUT/"hierarchical_audio_filter_probabilistic_graph.json"
graph_path.write_text(json.dumps(hierarchy,indent=2)+"\n")

W,H=1500,930
parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
 '<title id="t">Hierarchical discrete audio filters inside the Whisper probabilistic graph</title>',
 '<desc id="d">Nested graph templates for DFT, Mel, CAR-FAC, inner hair cells, AGC, quotient probabilities, and the Whisper neural suffix, with typed special edges.</desc>',
 '<rect width="100%" height="100%" fill="#fbfcfe"/><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#536274"/></marker></defs>',
 '<style>text{font-family:Menlo,monospace;fill:#16212c}.title{font-size:20px;font-weight:bold}.h{font-size:14px;font-weight:bold}.s{font-size:11px;fill:#536274}.box{fill:#edf4ff;stroke:#275d9e;stroke-width:1.5}.inner{fill:#fff;stroke:#6d86a8;stroke-width:1}.prob{fill:#ecf8ef;stroke:#287a42;stroke-width:1.5}.edge{fill:none;stroke:#536274;stroke-width:1.6;marker-end:url(#arr)}.delay{stroke-dasharray:7 5}.feedback{stroke:#a45a24;stroke-dasharray:3 4}.param{stroke:#7955a3;stroke-dasharray:1 4}</style>',
 '<text x="28" y="34" class="title">Discrete graph-of-graphs: filters are executable nested nodes</text>',
 '<text x="28" y="57" class="s">All samples, coefficients, states, classes, and tokens have finite types. Templates are shared; instance edges expand them only when required.</text>']

def box(x,y,w,h,title,sub,klass="box"):
    parts.extend([f'<rect class="{klass}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>',f'<text x="{x+14}" y="{y+24}" class="h">{html.escape(title)}</text>',f'<text x="{x+14}" y="{y+44}" class="s">{html.escape(sub)}</text>'])
def edge(x1,y1,x2,y2,label,klass="edge"):
    parts.append(f'<path class="{klass}" d="M{x1} {y1} L{x2} {y2}"/>')
    if label:parts.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-6}" text-anchor="middle" class="s">{html.escape(label)}</text>')

box(30,105,150,90,"PCM stream","SampleQ15 = Fin 65536")
box(230,88,475,300,"WhisperWindowFrontend","contains 3,001 shared frame instances")
box(255,145,125,72,"Window taps","400 coefficient bits","inner")
box(405,145,125,72,"DFT detector","201 graph instances","inner")
box(555,145,125,72,"Mel detector","80 graph instances","inner")
box(330,260,250,75,"Log/floor/affine","80 finite binary32 channels","inner")
edge(180,150,230,150,"SAMPLE_FLOW")
edge(380,181,405,181,"")
edge(530,181,555,181,"")
edge(618,217,530,260,"global max after drop-last")

box(230,455,475,330,"CARFACWindowFrontend","contains shared sample + frame templates")
box(255,515,125,72,"CAR section","81 place instances","inner")
box(405,515,125,72,"IHC detector","81 state machines","inner")
box(555,515,125,72,"AGC stage","4 x 81 instances","inner")
box(330,665,250,72,"NAP81 -> quotient","81 sections to 80 place sets","inner")
edge(180,150,230,510,"SAMPLE_FLOW")
edge(380,551,405,551,"")
edge(530,551,555,551,"")
parts.append('<path class="edge feedback" d="M617 587 C617 642 320 642 320 587"/><text x="470" y="637" text-anchor="middle" class="s">AGC feedback</text>')
edge(468,587,455,665,"")

box(790,120,265,100,"Coefficient bit blocks",f"{len(blob.blocks)} blocks / {len(blob.data)} bytes")
edge(790,170,705,170,"","edge param")
edge(790,200,705,520,"","edge param")
box(790,330,265,115,"ProbabilityQuotient","energy -> mass80 -> Fin16","prob")
edge(705,300,790,365,"")
edge(705,700,790,410,"")
box(1135,300,320,150,"Whisper neural suffix","74 op graph / 167 tensor refs")
edge(1055,382,1135,382,"MODEL_INTERFACE")
box(1135,510,320,100,"Token law","Categorical over Fin 51864","prob")
edge(1295,450,1295,510,"PROBABILITY_NORMALIZE")
parts.extend(['<text x="790" y="690" class="h">Special edge semantics</text>',
 '<line class="edge" x1="790" y1="725" x2="895" y2="725"/><text x="910" y="729" class="s">ordinary / graph instance</text>',
 '<line class="edge delay" x1="790" y1="760" x2="895" y2="760"/><text x="910" y="764" class="s">state delay t -> t+1</text>',
 '<line class="edge feedback" x1="790" y1="795" x2="895" y2="795"/><text x="910" y="799" class="s">biological feedback</text>',
 '<line class="edge param" x1="790" y1="830" x2="895" y2="830"/><text x="910" y="834" class="s">immutable parameter bits</text>',
 f'<text x="28" y="900" class="s">Flat 30 s estimates: Mel {flat_mel_frame*ANALYSIS_FRAMES:,} scalar nodes; CAR-FAC {(81*35+81*15+324*8)*480000:,}. Hierarchical templates remain finite and shared.</text>',
 '</svg>'])
(OUT/"hierarchical_audio_filter_probabilistic_graph.svg").write_text("\n".join(parts)+"\n")

graph_bytes=graph_path.stat().st_size
checkpoint_bytes=(WHISPER/"model.safetensors").stat().st_size
aux_raw=graph_bytes+len(blob.data)+state_path.stat().st_size
aux_gzip=len(gzip.compress(graph_path.read_bytes(),compresslevel=9))+len(gzip.compress(bytes(blob.data),compresslevel=9))+len(gzip.compress(state_path.read_bytes(),compresslevel=9))
(OUT/"HIERARCHICAL_DISCRETE_AUDIO_FILTER_GRAPH.md").write_text(f"""# Hierarchical discrete audio filter graph

## Construction

The expanded representation is a graph-of-graphs. A hypernode contains a reusable detector template and an indexed instance family instead of copying its scalar arithmetic for every bin, band, cochlear place, sample, or frame.

The finite ADTs are `SampleQ15 = Fin 65536`, `F32Bits = BitVec 32`, `EnergyClass = Fin 16`, `FrequencyNode = Fin 80`, `CochlearPlace = Fin 81`, and `Token = Fin 51864`. The coefficient blob contains the actual 400-point Hann window, the actual 80x201 Whisper Mel matrix, explicit 201x400 direct-DFT real and imaginary tables, and every numeric coefficient array designed by the pinned official CAR-FAC implementation.

## Relation to established visual calculi

The proposed notation is a **typed hierarchical probabilistic signal-flow hypergraph**. “Filter cirquent” is a convenient project nickname, not an established mathematical term.

- Classical signal-flow graphs contribute gain, sum, delay, and feedback nodes. Mason's original formulation relates graphs directly to systems of equations.
- Categorical signal-flow/string diagrams contribute typed open boxes, composition by wiring, copying/merging generators, feedback as trace, and equational graph rewriting. Interacting Hopf-algebra calculi give a complete graphical language for important classes of linear relations and signal-flow systems.
- Forney normal/factor graphs contribute variables as edges, local constraints as vertices, and exact sum-product inference on cycle-free graphs.
- Cirquent calculus contributes the key resource intuition: graph-shaped expressions can explicitly share a subexpression or resource instead of duplicating a proof-tree branch.

Our additions are finite bit-vector types, nested template instances, coefficient-bit resource nodes, quotient maps, positive-measure nodes, and a distinction between deterministic data edges and stochastic-kernel edges. Thus the graph can be read both operationally as a filter program and probabilistically as a factorization.

## Special edges

| edge | meaning |
|---|---|
| `GRAPH_INSTANCE` | invoke a nested graph template with an index parameter |
| `PARAMETER_BITS` | immutable finite coefficient bits |
| `STATE_DELAY` | recurrent state from sample `t` to `t+1` |
| `CASCADE_PLACE` | same-sample basal-to-apical cochlear propagation |
| `AGC_FEEDBACK` / `OHC_FEEDBACK` | biological gain-control dependencies |
| `QUOTIENT_MEMBERSHIP` | merge fine bins, packets, or sections into an interpretable set |
| `POSITIVE_MEASURE` | map activity to nonnegative energy |
| `PROBABILITY_NORMALIZE` | normalize energies to a categorical mass |
| `MODEL_INTERFACE` | pass 80 channels to the fixed Whisper suffix |

## Why nesting matters

The direct Mel graph has about {flat_mel_frame:,} scalar primitive nodes per analyzed frame, {flat_mel_frame*active_frames:,} for this active utterance, and {flat_mel_frame*ANALYSIS_FRAMES:,} for the padded 30-second window. Center padding produces 3,001 analyzed frames; the final frame is explicitly dropped to produce Whisper's 3,000 output frames. The log-energy floor depends on the maximum over the complete 80x3,000 window. The CAR-FAC estimate is {(81*35+81*15+324*8)*480000:,} scalar nodes for 30 seconds. These counts are useful flattening estimates, not backend instruction counts.

The hierarchical JSON is {graph_bytes:,} bytes and its coefficient blob is {len(blob.data):,} bytes, compared with the {checkpoint_bytes:,}-byte Whisper checkpoint. The representation stays small because templates and indexed instance families share repeated structure. It still references the checkpoint for the 74-op neural suffix, so the standalone exact package remains checkpoint-sized.

Including the {state_path.stat().st_size:,}-byte trajectory blob, the hierarchical addition is {aux_raw:,} raw bytes ({100*aux_raw/checkpoint_bytes:.3f}% of the checkpoint) or {aux_gzip:,} bytes when its three components are gzip-compressed independently. This is plausible overhead for an explicit frontend graph; it is not a replacement for the checkpoint parameters.

The five finite sample trajectories are now genuinely serialized in `{state_path.name}`: {state_path.stat().st_size:,} bytes, exactly 586 frames x 80 four-bit classes x five methods.

## Proof boundary

- Coefficient blocks, offsets, hashes, finite types, hierarchy, memberships, and packed sample states are explicit and mechanically checked.
- Representative direct-DFT and Mel detector instances are replayed from the serialized coefficient bits against PyTorch, and all 586 stored Mel `Fin16` frame states are reconstructed from the actual processor output.
- The direct DFT graph defines an ordered finite-bit implementation, but bit-for-bit equality with PyTorch's FFT kernel is not proved.
- CAR-FAC coefficient arrays are extracted from the pinned official implementation and the graph mirrors its update dependencies, but an opcode-level source equivalence proof remains open.
- The Mel path is the intended Whisper interface. The CAR-FAC-to-Whisper edge is marked `MODEL_INTERFACE_CANDIDATE_UNPROVED`; the observed transcription already refutes naïve equivalence.

## Visual-calculus sources

- S. J. Mason, “Feedback Theory—Some Properties of Signal Flow Graphs,” 1953, DOI `10.1109/JRPROC.1953.274449`.
- F. Bonchi, P. Sobociński, and F. Zanasi, “Interacting Hopf Algebras,” [arXiv:1403.7048](https://arxiv.org/abs/1403.7048), and “The Calculus of Signal Flow Diagrams I,” [author repository](https://eprints.soton.ac.uk/396532/).
- G. D. Forney Jr., “Codes on Graphs: Normal Realizations,” 2001, DOI `10.1109/18.910573`.
- G. Japaridze and B. Lamichhane, “Cirquent Calculus in a Nutshell,” [arXiv:2108.12552](https://arxiv.org/abs/2108.12552).
""")
print(json.dumps({"certificate":"HIERARCHICAL_AUDIO_FILTER_GRAPH_BUILT","templates":len(templates),"coefficient_blocks":len(blob.blocks),"coefficient_bytes":len(blob.data),"graph_bytes":graph_bytes,"mel_flat_30s":flat_mel_frame*ANALYSIS_FRAMES},indent=2))
