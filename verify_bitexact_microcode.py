#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
x=json.loads(Path("outputs/flan_binary32_microcode.json").read_text())
assert x["language"]=="PRSL-BINARY32-MICROCODE-1" and x["macro_opcode_count"]==129
assert len(x["spans"])==129 and len(x["microcode"])==x["micro_opcode_count"]
assert [m["pc"] for m in x["microcode"]]==list(range(x["micro_opcode_count"]))
for m in x["microcode"]:
 p=dict(m);h=p.pop("instruction_sha256");assert h==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 assert m["micro_opcode"] in x["scalar_semantics"]
for i,s in enumerate(x["spans"]):
 assert s["macro_index"]==i and s["pc_begin"]<s["pc_end_exclusive"]
 assert all(m["macro_index"]==i for m in x["microcode"][s["pc_begin"]:s["pc_end_exclusive"]])
assert any(m["micro_opcode"]=="CATEGORICAL_INVERSE_CDF" for m in x["microcode"])
print(f"FLAN_BINARY32_MICROCODE_OK macros=129 micro_ops={x['micro_opcode_count']}")
