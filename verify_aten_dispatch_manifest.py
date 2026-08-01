#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
x=json.loads(Path("outputs/flan_aten_dispatch_manifest.json").read_text())
assert x["language"]=="FLAN-SHARED-ATEN-DISPATCH-1" and x["unique_schemas"]>=40
families={r["family"] for r in x["schemas"]}
for required in ["aten::mm","aten::bmm","aten::_softmax","aten::rsqrt","aten::tanh","aten::embedding","aten::cat"]:assert required in families
for r in x["schemas"]:assert r["schema_sha256"]==hashlib.sha256(r["schema"].encode()).hexdigest() and r["full_calls"]+r["cached_calls"]>0
assert not x["portable_backend_independent"] and x["full_dispatch_calls"]>x["unique_schemas"] and x["cached_dispatch_calls"]>0
print(f"FLAN_ATEN_DISPATCH_MANIFEST_OK schemas={x['unique_schemas']} full={x['full_dispatch_calls']} cached={x['cached_dispatch_calls']}")
