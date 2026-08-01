#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
x=json.loads(Path("outputs/flan_fused_rms_attention_transfers.json").read_text())
assert x["language"]=="FLAN-FUSED-RMSRADIAL-ATTENTION-1" and len(x["transfers"])==24
for r in x["transfers"]:
 p=dict(r);h=p.pop("transfer_sha256");assert h==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 assert r["index"]==r["rms_index"]+1 and r["bias_units"]>0 and r["error_constructor"]=="RMSRadial"
print("FLAN_FUSED_RMSRADIAL_ATTENTION_OK occurrences=24")
