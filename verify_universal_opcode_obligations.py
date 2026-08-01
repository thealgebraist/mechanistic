#!/usr/bin/env python3
"""Verify one-to-one coverage and integrity of universal opcode obligations."""
import hashlib, importlib.util, json
from pathlib import Path

ledger = json.loads(Path("outputs/flan_universal_opcode_obligations.json").read_text())
graph_path = Path("outputs/flan_full_graph.json"); graph = json.loads(graph_path.read_text())
kernel_path = Path("outputs/flan_kernel_refinement_ledger.json"); kernel = json.loads(kernel_path.read_text())
assert ledger["full_graph_sha256"] == hashlib.sha256(graph_path.read_bytes()).hexdigest()
assert ledger["checkpoint_sha256"] == graph["checkpoint_sha256"]
assert ledger["kernel_refinement_ledger_sha256"] == hashlib.sha256(kernel_path.read_bytes()).hexdigest()
spec = importlib.util.find_spec("transformers")
assert spec is not None and spec.origin is not None
t5_source = Path(spec.origin).parent / "models/t5/modeling_t5.py"
assert ledger["transformers_t5_source_sha256"] == hashlib.sha256(t5_source.read_bytes()).hexdigest()
assert ledger["opcode_count"] == len(graph["ops"]) == len(ledger["obligations"]) == 129
for index, (op, obligation) in enumerate(zip(graph["ops"], ledger["obligations"])):
    assert obligation["index"] == index and obligation["opcode"] == op["op"]
    payload = dict(obligation); cert = payload.pop("obligation_sha256")
    assert cert == hashlib.sha256(json.dumps(payload, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    expected_refs = [v for k, v in op.items()
                     if k in {"weight", "q", "k", "v", "o", "wi_0", "wi_1", "wo"}]
    assert obligation["tensor_references"] == expected_refs
pending = [x for x in ledger["obligations"] if "pending" in x["semantic_status"]]
lowered = [x for x in ledger["obligations"] if "definitionally" in x["semantic_status"]]
proved = [x for x in ledger["obligations"] if "machine-checked" in x["semantic_status"]]
assert len(pending) + len(lowered) + len(proved) == 129
assert len(proved) == 84
assert sum(x["proof_artifact"] == "RMSNormLowering.lean" for x in proved) == 42
assert sum(x["proof_artifact"] == "GatedMLPLowering.lean" for x in proved) == 16
assert sum(x["proof_artifact"] == "AttentionLowering.lean" for x in proved) == 24
assert sum(x["proof_artifact"] == "SoftmaxLowering.lean" for x in proved) == 1
assert sum(x["proof_artifact"] == "CacheSemantics.lean" for x in proved) == 1
assert not pending
assert ledger["schema_composition_complete"] and not ledger["pinned_backend_refinement_complete"]
assert ledger["open_backend_micro_occurrences"] == kernel["open_occurrences"] == 343
assert sum(x["open_backend_micro_occurrences"] for x in ledger["obligations"]) == 343
assert ledger["status"] == "opcode schemas and concrete 129-opcode assembly discharged under primitive relations; pinned backend refinement remains"
print(json.dumps({"certificate": "UNIVERSAL_OBLIGATION_LEDGER_OK",
                  "opcodes": 129, "pending_semantic_proofs": len(pending),
                  "definitionally_lowered_schemas": len(lowered),
                  "machine_checked_complex_obligations": len(proved),
                  "open_backend_micro_occurrences": 343,
                  "universal_equivalence_complete": False}, indent=2))
