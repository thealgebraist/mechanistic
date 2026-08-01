#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
path=Path("outputs/whisper_tiny_en_equivalence_manifest.json");x=json.loads(path.read_text())
graph_path=Path("outputs/whisper_tiny_en_probabilistic_graph.json")
trace_path=Path("outputs/whisper_tiny_en_trace.json")
assert x["language"]=="WHISPER-AUDIO-TOKEN-INTERTWINING-1"
assert x["graph_sha256"]==hashlib.sha256(graph_path.read_bytes()).hexdigest()
assert x["trace_sha256"]==hashlib.sha256(trace_path.read_bytes()).hexdigest()
assert x["lean_source_sha256"]==hashlib.sha256(Path("WhisperAudioTokenEquivalence.lean").read_bytes()).hexdigest()
assert x["opcode_count"]==len(x["obligations"])==74
assert [o["index"] for o in x["obligations"]]==list(range(74))
for obligation in x["obligations"]:
    payload=dict(obligation);digest=payload.pop("obligation_sha256")
    assert digest==hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
assert not x["finite_state_quotient"] and x["probabilistic_register_graph"]
assert x["graph_structure_and_checkpoint_binding_complete"]
assert x["shared_abi_equivalence_complete_under_assumptions"]
assert not x["portable_backend_independent_equivalence_complete"]
assert x["theorem"].endswith("every_audio_every_finite_transcript_mass_equal")
print("WHISPER_EQUIVALENCE_MANIFEST_OK opcodes=74 all_audio=true portable=false")
