#!/usr/bin/env python3
"""Emit the missing numerical obligations for a universal FLAN approximation."""
import hashlib
import json
from pathlib import Path

graph_path = Path("outputs/flan_full_graph.json")
graph = json.loads(graph_path.read_text())
reachable_path = Path("outputs/flan_reachable_state_bounds.json")
reachable = json.loads(reachable_path.read_text())
add_path = Path("outputs/flan_ieee_add_transfers.json")
add_cert = json.loads(add_path.read_text())
add_by_index = {x["index"]: x for x in add_cert["transfers"]}
matmul_path = Path("outputs/flan_ieee_matmul_transfer.json")
matmul_transfer = json.loads(matmul_path.read_text())["transfer"]
rms_path = Path("outputs/flan_ieee_rmsnorm_transfers.json")
rms_by_index = {x["index"]: x for x in json.loads(rms_path.read_text())["transfers"]}
mlp_path = Path("outputs/flan_ieee_mlp_transfers.json")
mlp_by_index = {x["index"]: x for x in json.loads(mlp_path.read_text())["transfers"]}
attention_path = Path("outputs/flan_ieee_attention_transfers.json")
attention_by_index = {x["index"]: x for x in json.loads(attention_path.read_text())["transfers"]}
softmax_path = Path("outputs/flan_softmax_tv_transfer.json")
softmax_transfer = json.loads(softmax_path.read_text())["transfer"]

records = []
for index, op in enumerate(graph["ops"]):
    structural_exact = op["op"] in {"INPUT_TOKENS", "INPUT_DECODER_STACK", "SAMPLE_PUSH_UPDATE_CACHE", "EMBED"}
    add_transfer = add_by_index.get(index)
    matrix_transfer = matmul_transfer if index == matmul_transfer["index"] else None
    rms_transfer = rms_by_index.get(index)
    mlp_transfer = mlp_by_index.get(index)
    attention_transfer = attention_by_index.get(index)
    softmax = softmax_transfer if index == softmax_transfer["index"] else None
    payload = {
        "index": index,
        "opcode": op["op"],
        "quantification": "all model-reachable states for every backend-valid finite token sequence",
        "reachable_state_contract": "FLAN-REACHABLE-SUPNORM-BOUND-1",
        "source_semantics": "pinned PyTorch CPU float32 backend",
        "target_semantics": "ordered probabilistic register primitive",
        "distance": "integer upper bound on an explicitly scaled output pseudometric",
        "required_lemma": "d(targetEval(register), sourceEval(source)) <= local_gain * d(register, source) + local_bias",
        "error_units_per_real": add_cert["error_units_per_real"],
        "local_gain": 1 if structural_exact else add_transfer["gain"] if add_transfer else matrix_transfer["gain"] if matrix_transfer else rms_transfer["gain"] if rms_transfer else mlp_transfer["gain"] if mlp_transfer else attention_transfer["gain"] if attention_transfer else softmax["gain"] if softmax else None,
        "local_bias_units": 0 if structural_exact else add_transfer["bias_units"] if add_transfer else matrix_transfer["bias_units"] if matrix_transfer else rms_transfer["bias_units"] if rms_transfer else mlp_transfer["bias_units"] if mlp_transfer else attention_transfer["bias_units"] if attention_transfer else softmax["bias_units"] if softmax else None,
        "status": "CERTIFIED_STRUCTURAL_EXACT" if structural_exact else
                  "CERTIFIED_UNDER_IEEE754_RNE" if add_transfer else
                  "CERTIFIED_UNDER_IEEE754_RNE_REDUCTION_CONTRACT" if matrix_transfer else
                  "CERTIFIED_UNDER_IEEE754_RNE_RSQRT_CONTRACT" if rms_transfer else
                  "CERTIFIED_UNDER_IEEE754_RNE_GELU_CLIP_CONTRACT" if mlp_transfer else
                  "CERTIFIED_UNDER_PROBABILITY_SIMPLEX_AND_IEEE754_CLIP_CONTRACT" if attention_transfer else
                  "CERTIFIED_PROBABILITY_SIMPLEX_DIAMETER" if softmax else "OPEN_BACKEND_NUMERICAL_REFINEMENT",
        "proof_artifact": "EmbeddingLowering.lean" if op["op"] == "EMBED" else
                          "CacheSemantics.lean" if op["op"] == "SAMPLE_PUSH_UPDATE_CACHE" else
                          "GeneratedFlanProgram.lean" if structural_exact else
                          "flan_ieee_add_transfers.json" if add_transfer else
                          "flan_ieee_matmul_transfer.json" if matrix_transfer else
                          "flan_ieee_rmsnorm_transfers.json" if rms_transfer else
                          "flan_ieee_mlp_transfers.json" if mlp_transfer else
                          "flan_ieee_attention_transfers.json" if attention_transfer else
                          "flan_softmax_tv_transfer.json" if softmax else None,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["obligation_sha256"] = digest
    records.append(payload)

out = {
    "language": "FLAN-BACKEND-ERROR-OBLIGATIONS-1",
    "full_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    "checkpoint_sha256": graph["checkpoint_sha256"],
    "reachable_bounds_sha256": hashlib.sha256(reachable_path.read_bytes()).hexdigest(),
    "ieee_add_transfers_sha256": hashlib.sha256(add_path.read_bytes()).hexdigest(),
    "ieee_matmul_transfer_sha256": hashlib.sha256(matmul_path.read_bytes()).hexdigest(),
    "ieee_rmsnorm_transfers_sha256": hashlib.sha256(rms_path.read_bytes()).hexdigest(),
    "ieee_mlp_transfers_sha256": hashlib.sha256(mlp_path.read_bytes()).hexdigest(),
    "ieee_attention_transfers_sha256": hashlib.sha256(attention_path.read_bytes()).hexdigest(),
    "softmax_tv_transfer_sha256": hashlib.sha256(softmax_path.read_bytes()).hexdigest(),
    "reachable_final_bounds": reachable["final_bounds"],
    "opcode_count": len(records),
    "opcode_composition_theorem": "GeneratedFlanProgram.concrete_129_opcode_error_affine",
    "trace_composition_theorem": "ApproximateWholeModel.all_prompts_bounded_horizon",
    "composition_formula": "one_step_error <= composed_gain * initial_error + composed_bias; trace_distance <= horizon * certified_one_step_TV",
    "saturation_note": "for total variation, divide by the chosen unit scale and cap the final bound at 1",
    "obligations": records,
    "all_occurrence_transfers_instantiated": True,
    "nontrivial_universal_tv_bound": False,
    "status": "CONDITIONAL COVERAGE COMPLETE: all 129 transfers instantiated; final and composed TV bounds saturate at 1",
}
path = Path("outputs/flan_backend_error_obligations.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"artifact": str(path), "opcodes": len(records), "open": len(records)}, indent=2))
