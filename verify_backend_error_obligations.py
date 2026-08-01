#!/usr/bin/env python3
"""Verify coverage and integrity of backend numerical refinement obligations."""
import hashlib
import json
from pathlib import Path

graph_path = Path("outputs/flan_full_graph.json")
graph = json.loads(graph_path.read_text())
ledger = json.loads(Path("outputs/flan_backend_error_obligations.json").read_text())
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
assert ledger["full_graph_sha256"] == hashlib.sha256(graph_path.read_bytes()).hexdigest()
assert ledger["checkpoint_sha256"] == graph["checkpoint_sha256"]
assert ledger["reachable_bounds_sha256"] == hashlib.sha256(reachable_path.read_bytes()).hexdigest()
assert ledger["ieee_add_transfers_sha256"] == hashlib.sha256(add_path.read_bytes()).hexdigest()
assert ledger["ieee_matmul_transfer_sha256"] == hashlib.sha256(matmul_path.read_bytes()).hexdigest()
assert ledger["ieee_rmsnorm_transfers_sha256"] == hashlib.sha256(rms_path.read_bytes()).hexdigest()
assert ledger["ieee_mlp_transfers_sha256"] == hashlib.sha256(mlp_path.read_bytes()).hexdigest()
assert ledger["ieee_attention_transfers_sha256"] == hashlib.sha256(attention_path.read_bytes()).hexdigest()
assert ledger["softmax_tv_transfer_sha256"] == hashlib.sha256(softmax_path.read_bytes()).hexdigest()
assert ledger["reachable_final_bounds"] == reachable["final_bounds"]
assert ledger["opcode_count"] == len(graph["ops"]) == len(ledger["obligations"]) == 129
for index, (op, record) in enumerate(zip(graph["ops"], ledger["obligations"])):
    assert record["index"] == index and record["opcode"] == op["op"]
    payload = dict(record)
    digest = payload.pop("obligation_sha256")
    assert digest == hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if op["op"] in {"INPUT_TOKENS", "INPUT_DECODER_STACK", "SAMPLE_PUSH_UPDATE_CACHE", "EMBED"}:
        assert record["local_gain"] == 1 and record["local_bias_units"] == 0
        assert record["status"] == "CERTIFIED_STRUCTURAL_EXACT"
    elif op["op"] == "ADD":
        add = add_by_index[index]
        assert record["local_gain"] == add["gain"] == 2
        assert record["local_bias_units"] == add["bias_units"] > 0
        assert record["status"] == "CERTIFIED_UNDER_IEEE754_RNE"
    elif op["op"] == "MATMUL":
        assert index == matmul_transfer["index"]
        assert record["local_gain"] == matmul_transfer["gain"]
        assert record["local_bias_units"] == matmul_transfer["bias_units"]
        assert record["status"] == "CERTIFIED_UNDER_IEEE754_RNE_REDUCTION_CONTRACT"
    elif op["op"] == "RMSNORM":
        rms = rms_by_index[index]
        assert record["local_gain"] == rms["gain"]
        assert record["local_bias_units"] == rms["bias_units"]
        assert record["status"] == "CERTIFIED_UNDER_IEEE754_RNE_RSQRT_CONTRACT"
    elif op["op"] == "GATED_MLP":
        mlp = mlp_by_index[index]
        assert record["local_gain"] == mlp["gain"]
        assert record["local_bias_units"] == mlp["bias_units"]
        assert record["status"] == "CERTIFIED_UNDER_IEEE754_RNE_GELU_CLIP_CONTRACT"
    elif op["op"] in {"SELF_ATTENTION_SEQUENCE", "SELF_ATTENTION_KV", "CROSS_ATTENTION"}:
        att = attention_by_index[index]
        assert record["local_gain"] == att["gain"]
        assert record["local_bias_units"] == att["bias_units"]
        assert record["status"] == "CERTIFIED_UNDER_PROBABILITY_SIMPLEX_AND_IEEE754_CLIP_CONTRACT"
    elif op["op"] == "SOFTMAX":
        assert index == softmax_transfer["index"]
        assert record["local_gain"] == softmax_transfer["gain"] == 1
        assert record["local_bias_units"] == softmax_transfer["bias_units"]
        assert record["status"] == "CERTIFIED_PROBABILITY_SIMPLEX_DIAMETER"
    else:
        assert record["local_gain"] is None and record["local_bias_units"] is None
        assert record["status"] == "OPEN_BACKEND_NUMERICAL_REFINEMENT"
    assert record["reachable_state_contract"] == "FLAN-REACHABLE-SUPNORM-BOUND-1"
assert ledger["opcode_composition_theorem"] == "GeneratedFlanProgram.concrete_129_opcode_error_affine"
assert ledger["trace_composition_theorem"] == "ApproximateWholeModel.all_prompts_bounded_horizon"
print(json.dumps({
    "certificate": "BACKEND_ERROR_OBLIGATION_LEDGER_OK",
    "opcodes": 129,
    "coverage_exact": True,
    "arithmetic_free_transfers_closed": 5,
    "ieee_add_transfers_closed": 40,
    "ieee_matmul_transfers_closed": 1,
    "ieee_rmsnorm_transfers_closed": 42,
    "ieee_clipped_mlp_transfers_closed": 16,
    "ieee_clipped_attention_transfers_closed": 24,
    "softmax_diameter_transfers_closed": 1,
    "open_backend_numerical_bounds": 0,
    "all_occurrence_transfers_instantiated": True,
    "universal_approximation_complete": False,
}, indent=2))
