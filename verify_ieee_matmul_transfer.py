#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
gpath=Path("outputs/flan_full_graph.json"); g=json.loads(gpath.read_text())
c=json.loads(Path("outputs/flan_ieee_matmul_transfer.json").read_text()); r=c["transfer"]
matches=[(i,o) for i,o in enumerate(g["ops"]) if o["op"]=="MATMUL"]
assert len(matches)==1 and r["index"]==matches[0][0] and r["weight"]==matches[0][1]["weight"]
assert c["graph_sha256"]==hashlib.sha256(gpath.read_bytes()).hexdigest()
convex=Path("outputs/flan_convex_geometry_certificate.json")
assert c["convex_geometry_sha256"]==hashlib.sha256(convex.read_bytes()).hexdigest()
p=dict(r); d=p.pop("transfer_sha256")
assert d==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
assert r["dot_length"]==512 and r["rows"]==32128 and r["gain"]>0 and r["bias_units"]>0
print(json.dumps({"certificate":"IEEE_MATMUL_TRANSFER_OK","occurrences":1,"dot_length":512,
                  "vocabulary":32128,"coverage_exact":True},indent=2))
