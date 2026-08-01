#!/usr/bin/env python3
"""Recompute every environment and artifact assertion in the backend contract."""
import hashlib, importlib.util, json, platform, sys
from pathlib import Path
import torch, transformers

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

path = Path("outputs/flan_backend_contract.json"); c = json.loads(path.read_text())
payload = dict(c); cert = payload.pop("contract_sha256")
assert cert == hashlib.sha256(json.dumps(payload, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
assert c["platform"] == {"python": sys.version.split()[0], "system": platform.system(),
                         "machine": platform.machine()}
assert c["libraries"] == {"torch": torch.__version__, "transformers": transformers.__version__}
spec = importlib.util.find_spec("transformers"); assert spec and spec.origin
root = Path(spec.origin).parent
paths = {"modeling_t5": root / "models/t5/modeling_t5.py", "activations": root / "activations.py",
         "register_interpreter": Path("verify_kv_cache_register.py"),
         "full_graph": Path("outputs/flan_full_graph.json"),
         "checkpoint": Path("work/google_flan/model.safetensors"),
         "tokenizer": Path("work/google_flan/spiece.model"),
         "source_correspondence": Path("outputs/flan_backend_source_correspondence.json")}
assert c["artifact_hashes"] == {name: sha(p) for name, p in paths.items()}
assert c["execution"]["device"] == "cpu" and c["execution"]["dtype"] == "float32"
assert c["execution"]["deterministic_algorithms"] is True
assert c["execution"]["float32_matmul_precision"] == "highest"
print(json.dumps({"certificate": "BACKEND_CONTRACT_OK", "hashes": len(paths),
                  "deterministic": True, "device": "cpu", "dtype": "float32",
                  "universal_ieee754_proof": False}, indent=2))
