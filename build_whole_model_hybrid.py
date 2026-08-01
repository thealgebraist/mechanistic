#!/usr/bin/env python3
"""Bind the whole FLAN register program to certified quotient patches."""
import hashlib, json
from pathlib import Path

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

full_path = "outputs/flan_full_graph.json"
tower_path = "outputs/flan_domain32_refinement_tower.json"
full = json.loads(Path(full_path).read_text())
tower = json.loads(Path(tower_path).read_text())
checkpoint = "work/google_flan/model.safetensors"
tokenizer = "work/google_flan/spiece.model"
backend = "outputs/flan_backend_contract.json"
assert full["checkpoint_sha256"] == sha(checkpoint)

manifest = {
    "language": "PRSL-WHOLE-MODEL-HYBRID-1",
    "model": {
        "name": "google/flan-t5-small",
        "checkpoint": checkpoint, "checkpoint_sha256": sha(checkpoint),
        "tokenizer": tokenizer, "tokenizer_sha256": sha(tokenizer),
        "numerical_semantics": "float32 PyTorch-compatible T5 opcodes",
    },
    "whole_model_register_program": {
        "artifact": full_path, "sha256": sha(full_path),
        "opcodes": len(full["ops"]), "coverage": "all tokenized inputs within runtime resource limits",
        "state": ["encoder_memory", "decoder_hidden", "self_KV_cache", "fuel", "token_stack"],
        "verification": ["FULL_GRAPH_TENSOR_REFERENCES_OK", "FULL_GRAPH_NUMERIC_REPLAY",
                         "REGISTER_KV_CACHE_REPLAY"],
    },
    "backend_contract": {"artifact": backend, "sha256": sha(backend),
                         "status": "reproducible hashed semantics; universal IEEE-754 proof pending"},
    "certified_graph_patch": {
        "artifact": tower_path, "sha256": sha(tower_path),
        "guard": {"prompt_set": len(tower["levels"][0]["roots"]),
                  "prompt_identity": "exact serialized prompt string", "max_decoder_horizon": 3},
        "levels": [{"level": x["level"], "states": x["states"], "epsilon": x["epsilon"],
                    "horizon_tv_bound": x["horizon_bound"],
                    "neural_horizon_tv_bound": x["neural_horizon_bound"],
                    "direct_trace_max_tv": x["direct_trace_max_tv"],
                    "direct_neural_trace_bound": x["direct_neural_trace_bound"],
                    "status": x["status"],
                    "certificate_sha256": x["certificate_sha256"]} for x in tower["levels"]],
        "verification": "REFINEMENT_TOWER_OK",
    },
    "router": {
        "priority": ["certified_graph_patch_when_guard_holds", "whole_model_register_program"],
        "total": True,
        "graph_residual_overlap": "graph takes priority only under exact guard",
        "outside_patch": "register program; never reported as graph-certified",
    },
    "semantic_contract": {
        "inside_patch": "TV(neural projected output traces) <= selected level neural_horizon_tv_bound",
        "outside_patch": "execute complete register program",
        "whole_input_space_coverage": "graph patch union register residual",
        "unproven": ["universal floating-point equivalence for every possible input",
                     "finite quotient of unrestricted FLAN behavior",
                     "resource-unbounded execution"],
    },
}
payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
out = Path("outputs/flan_whole_model_hybrid.json")
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"artifact": str(out), "opcodes": len(full["ops"]),
                  "patch_levels": len(tower["levels"]), "router_total": True,
                  "sha256": manifest["manifest_sha256"]}, indent=2))
