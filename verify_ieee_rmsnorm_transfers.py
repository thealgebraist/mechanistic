#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
gpath=Path("outputs/flan_full_graph.json");g=json.loads(gpath.read_text());c=json.loads(Path("outputs/flan_ieee_rmsnorm_transfers.json").read_text())
ops=[(i,o) for i,o in enumerate(g["ops"]) if o["op"]=="RMSNORM"];assert len(ops)==len(c["transfers"])==42
for (i,o),r in zip(ops,c["transfers"]):
 assert r["index"]==i and r["weight"]==o["weight"] and r["input"]==o["input"]
 p=dict(r);d=p.pop("transfer_sha256");assert d==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
 assert r["gain"]>0 and r["bias_units"]>0 and r["status"]=="CERTIFIED_UNDER_IEEE754_RNE_RSQRT_CONTRACT"
assert c["graph_sha256"]==hashlib.sha256(gpath.read_bytes()).hexdigest()
tsv=Path("outputs/flan_ieee_rmsnorm_transfers.tsv");assert c["tsv_sha256"]==hashlib.sha256(tsv.read_bytes()).hexdigest()
print(json.dumps({"certificate":"IEEE_RMSNORM_TRANSFERS_OK","occurrences":42,"coverage_exact":True,"rsqrt_contract":"correctly-rounded"},indent=2))
