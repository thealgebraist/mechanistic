#!/usr/bin/env python3
"""Independent structural and checkpoint verifier for the Whisper graph."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from safetensors import safe_open

root=Path("work/whisper_tiny_en")
graph_path=Path("outputs/whisper_tiny_en_probabilistic_graph.json")
trace_path=Path("outputs/whisper_tiny_en_trace.json")
graph=json.loads(graph_path.read_text());trace=json.loads(trace_path.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert graph["language"]=="WHISPER-PROBABILISTIC-REGISTER-GRAPH-1"
assert graph["checkpoint_sha256"]==sha(root/"model.safetensors")
assert graph["config_sha256"]==sha(root/"config.json")
assert graph["generation_config_sha256"]==sha(root/"generation_config.json")
assert graph["preprocessor_config_sha256"]==sha(root/"preprocessor_config.json")
assert graph["opcode_count"]==len(graph["ops"])==74
assert [op["index"] for op in graph["ops"]]==list(range(74))
with safe_open(root/"model.safetensors",framework="pt",device="cpu") as handle:
    names=list(handle.keys())
    assert len(names)==graph["tensor_count"]==167
    for name in names:
        assert graph["tensor_metadata"][name]["shape"]==list(handle.get_slice(name).get_shape())
        assert graph["tensor_metadata"][name]["dtype"]==handle.get_slice(name).get_dtype()
used={weight for op in graph["ops"] for weight in op["weights"]}
assert used==set(names)
counts=Counter(op["opcode"] for op in graph["ops"])
assert counts["SELF_ATTENTION"]==4 and counts["CACHED_SELF_ATTENTION"]==4
assert counts["CROSS_ATTENTION"]==4 and counts["MLP_GELU"]==8
assert counts["SOFTMAX"]==counts["SAMPLE_OR_ARGMAX"]==1
assert trace["graph_sha256"]==sha(graph_path)
assert trace["checkpoint_sha256"]==graph["checkpoint_sha256"]
assert trace["audio_samples"]==93680 and trace["feature_shape"]==[1,80,3000]
assert trace["greedy_positions_matching_processed_argmax"]==trace["greedy_positions"]==22
assert "Quilter is the apostle" in trace["transcript"]
assert trace["module_output_shapes"]["encoder.conv2"]==[[1,384,1500]]
assert [1,23,51864] in trace["module_output_shapes"]["lm_head"]
assert graph["semantic_status"].endswith("CONDITIONAL_ON_SHARED_PRIMITIVE_ABI")
print(json.dumps({"certificate":"WHISPER_PROBABILISTIC_GRAPH_OK","opcodes":74,"tensors":167,"greedy_argmax":"22/22","universal_scope":True,"portable_equivalence_complete":False},indent=2))
