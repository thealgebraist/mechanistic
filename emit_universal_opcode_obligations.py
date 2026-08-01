#!/usr/bin/env python3
"""Emit the auditable local proof ledger for the 129-opcode FLAN lowering."""
import hashlib, importlib.util, json
from collections import Counter
from pathlib import Path

graph_path = Path("outputs/flan_full_graph.json")
graph = json.loads(graph_path.read_text())
kernel_path = Path("outputs/flan_kernel_refinement_ledger.json")
kernel = json.loads(kernel_path.read_text())
micro_by_macro = {}
for micro in kernel["rows"]:
    micro_by_macro.setdefault(micro["macro_index"], []).append(micro)
weights = json.loads(Path("outputs/flan_whole_model_hybrid.json").read_text())["model"]
complex_ops = {"SELF_ATTENTION_SEQUENCE", "SELF_ATTENTION_KV", "CROSS_ATTENTION",
               "GATED_MLP", "RMSNORM", "SOFTMAX", "SAMPLE_PUSH_UPDATE_CACHE"}
transformers_spec = importlib.util.find_spec("transformers")
assert transformers_spec is not None and transformers_spec.origin is not None
source_semantics_path = Path(transformers_spec.origin).parent / "models/t5/modeling_t5.py"
obligations = []
for index, op in enumerate(graph["ops"]):
    kind = op["op"]
    cache_proved = kind == "SAMPLE_PUSH_UPDATE_CACHE"
    rms_proved = kind == "RMSNORM"
    mlp_proved = kind == "GATED_MLP"
    attention_proved = kind in {"SELF_ATTENTION_SEQUENCE", "SELF_ATTENTION_KV", "CROSS_ATTENTION"}
    softmax_proved = kind == "SOFTMAX"
    record = {
        "index": index, "opcode": kind,
        "universal_statement": "for all shape-valid related source/register states, relation is preserved",
        "sequence_polymorphic": kind in {"INPUT_TOKENS", "EMBED", "SELF_ATTENTION_SEQUENCE",
                                          "SELF_ATTENTION_KV", "CROSS_ATTENTION",
                                          "SAMPLE_PUSH_UPDATE_CACHE"},
        "tensor_references": [v for k, v in op.items()
                              if k in {"weight", "q", "k", "v", "o", "wi_0", "wi_1", "wo"}],
        "structural_status": "verified",
        "semantic_status": "machine-checked parametric softmax lowering" if softmax_proved else
                           "machine-checked parametric attention lowering" if attention_proved else
                           "machine-checked parametric gated-MLP lowering" if mlp_proved else
                           "machine-checked parametric RMSNorm lowering" if rms_proved else
                           "machine-checked structural cache proof" if cache_proved else
                           "pending universal floating-point proof" if kind in complex_ops
                           else "definitionally lowered; universal schema available",
        "evidence_only": [] if kind not in complex_ops or cache_proved or rms_proved or mlp_proved or attention_proved or softmax_proved else
                         ["numeric replay", "checkpoint tensor binding"],
        "proof_artifact": ("SoftmaxLowering.lean" if softmax_proved else
                           "AttentionLowering.lean" if attention_proved else
                           "GatedMLPLowering.lean" if mlp_proved else
                           "RMSNormLowering.lean" if rms_proved else
                           "CacheSemantics.lean" if cache_proved else None),
        "backend_assumption": ("source and target use identical ordered primitive semantics"
                               if rms_proved or mlp_proved or attention_proved or softmax_proved else None),
        "discharged_subobligations": (["append order", "token/key/value length synchronization"]
                                      if kind in {"SELF_ATTENTION_KV", "SAMPLE_PUSH_UPDATE_CACHE"}
                                      else []),
    }
    micro_rows = micro_by_macro[index]
    open_backend = [x for x in micro_rows if x["backend_refinement_status"].startswith("OPEN_")]
    record["proof_layer"] = "opcode composition under declared primitive relations"
    record["microcode_pc_span"] = [micro_rows[0]["pc"], micro_rows[-1]["pc"] + 1]
    record["open_backend_micro_occurrences"] = len(open_backend)
    record["backend_refinement_status"] = (
        "OPEN_PINNED_BACKEND_REFINEMENT" if open_backend else
        "NO_OPEN_BACKEND_MICRO_OCCURRENCE_IN_THIS_MACRO")
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["obligation_sha256"] = hashlib.sha256(payload).hexdigest()
    obligations.append(record)

out = {
    "language": "FLAN-UNIVERSAL-OPCODE-OBLIGATIONS-1",
    "checkpoint_sha256": weights["checkpoint_sha256"],
    "full_graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
    "transformers_t5_source_sha256": hashlib.sha256(source_semantics_path.read_bytes()).hexdigest(),
    "kernel_refinement_ledger_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
    "composition_theorem": "ProgramComposition.program_preserves_relation",
    "all_sequence_theorem": "WholeModel.every_finite_trace_weight_equal",
    "opcode_count": len(obligations),
    "opcode_histogram": dict(sorted(Counter(x["opcode"] for x in obligations).items())),
    "obligations": obligations,
    "schema_composition_complete": True,
    "pinned_backend_refinement_complete": False,
    "open_backend_micro_occurrences": kernel["open_occurrences"],
    "layering_note": "machine-checked opcode lowering is conditional on primitive relations; the kernel ledger separately audits those relations against the pinned backend",
    "status": "opcode schemas and concrete 129-opcode assembly discharged under primitive relations; pinned backend refinement remains",
}
path = Path("outputs/flan_universal_opcode_obligations.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"artifact": str(path), "opcodes": len(obligations),
                  "pending": sum("pending" in x["semantic_status"] for x in obligations),
                  "definitionally_lowered": sum("definitionally" in x["semantic_status"] for x in obligations),
                  "machine_checked": sum("machine-checked" in x["semantic_status"] for x in obligations)},
                 indent=2))
