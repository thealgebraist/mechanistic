#!/usr/bin/env python3
"""Verify provenance and total routing of the whole-model hybrid translation."""
import hashlib, json
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

path = Path("outputs/flan_whole_model_hybrid.json")
m = json.loads(path.read_text()); payload = dict(m); cert = payload.pop("manifest_sha256")
assert cert == hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
model = m["model"]; register = m["whole_model_register_program"]; patch = m["certified_graph_patch"]
assert model["checkpoint_sha256"] == sha(model["checkpoint"])
assert model["tokenizer_sha256"] == sha(model["tokenizer"])
assert register["sha256"] == sha(register["artifact"])
assert patch["sha256"] == sha(patch["artifact"])
assert m["backend_contract"]["sha256"] == sha(m["backend_contract"]["artifact"])
full = json.loads(Path(register["artifact"]).read_text())
tower = json.loads(Path(patch["artifact"]).read_text())
assert full["checkpoint_sha256"] == model["checkpoint_sha256"]
assert register["opcodes"] == len(full["ops"]) == 129
assert len(patch["levels"]) == len(tower["levels"])
for declared, actual in zip(patch["levels"], tower["levels"]):
    assert declared["certificate_sha256"] == actual["certificate_sha256"]
    assert declared["states"] == actual["states"]
    assert declared["horizon_tv_bound"] == actual["horizon_bound"]
    assert declared["neural_horizon_tv_bound"] == actual["neural_horizon_bound"]
    assert declared["direct_trace_max_tv"] == actual["direct_trace_max_tv"]
    assert declared["direct_neural_trace_bound"] == actual["direct_neural_trace_bound"]
    assert declared["status"] == actual["status"]
assert m["router"]["total"] is True
assert m["router"]["priority"][-1] == "whole_model_register_program"
assert m["semantic_contract"]["whole_input_space_coverage"] == "graph patch union register residual"
print(json.dumps({"certificate": "WHOLE_MODEL_HYBRID_OK", "opcodes": 129,
                  "checkpoint_bound": True, "tokenizer_bound": True,
                  "graph_patch_levels": len(tower["levels"]), "router_total": True,
                  "fallback": "complete register program"}, indent=2))
