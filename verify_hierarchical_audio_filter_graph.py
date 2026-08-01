#!/usr/bin/env python3
import hashlib,json,math
from pathlib import Path

OUT=Path("outputs")
g=json.loads((OUT/"hierarchical_audio_filter_probabilistic_graph.json").read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert g["language"]=="HIERARCHICAL-DISCRETE-AUDIO-PROBABILISTIC-GRAPH-1"
assert set(g["value_types"])=={"SampleQ15","F32Bits","EnergyClass","FrequencyNode","CochlearPlace","Token"}
required_templates={"OrderedF32Sum","DFTBinDetector","MelBandDetector","WhisperFrameFrontend","WhisperWindowFrontend","CARFACSection","InnerHairCell","AGCStage","CARFACSampleFrontend","CARFACFrameEnergy","CARFACWindowFrontend","ProbabilityQuotient"}
assert set(g["templates"])==required_templates
required_edges={"GRAPH_INSTANCE","PARAMETER_BITS","STATE_DELAY","CASCADE_PLACE","AGC_FEEDBACK","OHC_FEEDBACK","QUOTIENT_MEMBERSHIP","POSITIVE_MEASURE","PROBABILITY_NORMALIZE","MODEL_INTERFACE"}
assert required_edges<=set(g["edge_kinds"])
blob=OUT/g["coefficient_blob"]["path"];data=blob.read_bytes()
assert len(data)==g["coefficient_blob"]["bytes"] and sha(blob)==g["coefficient_blob"]["sha256"]
cursor=0;names=set()
for b in g["coefficient_blob"]["blocks"]:
    assert b["offset"]==cursor and b["bytes"]==4*math.prod(b["shape"]) and b["finite"]
    payload=data[cursor:cursor+b["bytes"]]
    assert hashlib.sha256(payload).hexdigest()==b["sha256"]
    cursor+=b["bytes"];names.add(b["name"])
assert cursor==len(data)
assert {"whisper.hann_window","whisper.dft_cos","whisper.dft_minus_sin","whisper.mel_weights","carfac.pole_freqs"}<=names
assert next(b for b in g["coefficient_blob"]["blocks"] if b["name"]=="whisper.hann_window")["shape"]==[400]
assert next(b for b in g["coefficient_blob"]["blocks"] if b["name"]=="whisper.mel_weights")["shape"]==[80,201]
assert next(b for b in g["coefficient_blob"]["blocks"] if b["name"]=="carfac.pole_freqs")["shape"]==[81]
state=OUT/g["sample_state_blob"]["path"]
assert state.stat().st_size==586*80//2*5 and sha(state)==g["sample_state_blob"]["sha256"]
base=OUT/g["base_whisper"]["path"]
assert sha(base)==g["base_whisper"]["sha256"] and g["base_whisper"]["opcode_count"]==74
assert len(g["base_whisper"]["ops"])==74
assert g["flattening_estimates"]["mel_scalar_nodes_30_seconds"]>900_000_000
assert g["flattening_estimates"]["carfac_scalar_nodes_30_seconds_estimate"]>1_000_000_000
assert g["instance_families"]["stft_analysis_frames"]==3001 and g["instance_families"]["output_frames"]==3000
assert ["mel_energy80x3001","drop_last_frame","DROP_LAST"] in g["templates"]["WhisperWindowFrontend"]["edges"]
assert ["log10","utterance_global_max","UTTERANCE_GLOBAL"] in g["templates"]["WhisperWindowFrontend"]["edges"]
candidate=[e for e in g["outer_graph"]["edges"] if e[0]=="CochlearQuotient" and e[1]=="WhisperNeuralSuffix"]
assert candidate==[["CochlearQuotient","WhisperNeuralSuffix","MODEL_INTERFACE_CANDIDATE_UNPROVED"]]
print(f"HIERARCHICAL_AUDIO_FILTER_GRAPH_OK templates={len(required_templates)} blocks={len(names)} bytes={len(data)}")
