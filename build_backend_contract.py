#!/usr/bin/env python3
"""Pin the executable backend semantics used by source and register replay."""
import hashlib, importlib.util, json, platform, sys
from pathlib import Path
import torch, transformers

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision("highest")
tspec = importlib.util.find_spec("transformers")
assert tspec and tspec.origin
troot = Path(tspec.origin).parent
files = {
    "modeling_t5": troot / "models/t5/modeling_t5.py",
    "activations": troot / "activations.py",
    "register_interpreter": Path("verify_kv_cache_register.py"),
    "full_graph": Path("outputs/flan_full_graph.json"),
    "checkpoint": Path("work/google_flan/model.safetensors"),
    "tokenizer": Path("work/google_flan/spiece.model"),
    "source_correspondence": Path("outputs/flan_backend_source_correspondence.json"),
}
contract = {
    "language": "FLAN-BACKEND-CONTRACT-1",
    "platform": {"python": sys.version.split()[0], "system": platform.system(),
                 "machine": platform.machine()},
    "libraries": {"torch": torch.__version__, "transformers": transformers.__version__},
    "execution": {"device": "cpu", "dtype": "float32",
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "float32_matmul_precision": torch.get_float32_matmul_precision(),
                  "training": False, "dropout": False},
    "primitive_order": {
        "RMSNORM": ["square", "mean_last", "add_epsilon", "rsqrt", "scale", "weight_multiply"],
        "GATED_MLP": ["wi_0", "gelu_tanh", "wi_1", "hadamard", "wo"],
        "ATTENTION": ["qkv_project", "split_heads", "qk_scores", "bias", "mask",
                      "softmax_float32", "weighted_values", "merge_heads", "output_project"],
        "CACHE": ["append_key", "append_value", "append_token"],
        "READOUT": ["final_rmsnorm", "lm_head", "softmax_float32"],
    },
    "artifact_hashes": {name: sha(path) for name, path in files.items()},
    "status": "reproducible hashed backend contract; IEEE-754 universal proof not supplied",
}
payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
contract["contract_sha256"] = hashlib.sha256(payload).hexdigest()
out = Path("outputs/flan_backend_contract.json")
out.write_text(json.dumps(contract, indent=2) + "\n")
print(json.dumps({"artifact": str(out), "contract_sha256": contract["contract_sha256"],
                  "torch": torch.__version__, "transformers": transformers.__version__,
                  "deterministic": True}, indent=2))
