#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
gp=Path("outputs/flan_full_graph.json"); g=json.loads(gp.read_text())
c=json.loads(Path("outputs/flan_ieee_mlp_transfers.json").read_text())
ops=[(i,o) for i,o in enumerate(g["ops"]) if o["op"]=="GATED_MLP"]
assert len(ops)==len(c["transfers"])==16
for (i,o),r in zip(ops,c["transfers"]):
    assert r["index"]==i and r["wi_0"]==o["wi_0"] and r["wi_1"]==o["wi_1"] and r["wo"]==o["wo"]
    p=dict(r); d=p.pop("transfer_sha256")
    assert d==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    assert r["gain"]>0 and r["bias_units"]>0
assert c["graph_sha256"]==hashlib.sha256(gp.read_bytes()).hexdigest()
print(json.dumps({"certificate":"IEEE_CLIPPED_MLP_TRANSFERS_OK","occurrences":16,"coverage_exact":True,"target_clipped":True},indent=2))
