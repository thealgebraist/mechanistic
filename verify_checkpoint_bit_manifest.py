#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
x=json.loads(Path("outputs/flan_checkpoint_bit_manifest.json").read_text());m=Path("work/google_flan/model.safetensors")
assert x["language"]=="FLAN-SAFETENSORS-BINARY32-BITS-1" and x["tensor_count"]==190
assert x["checkpoint_sha256"]==hashlib.sha256(m.read_bytes()).hexdigest()
assert x["all_tensors_f32"] and x["all_values_finite"] and x["graph_unique_weight_references"]==188
assert all(t["byte_length"]==4*t["elements"] for t in x["tensors"])
assert sum(bool(t["graph_references"]) for t in x["tensors"])==188
print("FLAN_CHECKPOINT_BIT_MANIFEST_OK tensors=190 referenced=188 all_f32_finite=true")
