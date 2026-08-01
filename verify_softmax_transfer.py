#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text());c=json.loads(Path("outputs/flan_softmax_tv_transfer.json").read_text());r=c["transfer"]
ops=[(i,o) for i,o in enumerate(g["ops"]) if o["op"]=="SOFTMAX"];assert len(ops)==1 and r["index"]==ops[0][0]
p=dict(r);d=p.pop("transfer_sha256");assert d==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest();assert r["bias_units"]==r["tv_diameter_units"] and not c["nontrivial"]
print(json.dumps({"certificate":"SOFTMAX_TV_DIAMETER_TRANSFER_OK","occurrences":1,"coverage_exact":True,"nontrivial":False},indent=2))
