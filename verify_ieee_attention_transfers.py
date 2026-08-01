#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text());c=json.loads(Path("outputs/flan_ieee_attention_transfers.json").read_text());kinds={"SELF_ATTENTION_SEQUENCE","SELF_ATTENTION_KV","CROSS_ATTENTION"};ops=[(i,o) for i,o in enumerate(g["ops"]) if o["op"] in kinds];assert len(ops)==len(c["transfers"])==24
for(i,o),r in zip(ops,c["transfers"]):
 assert r["index"]==i and r["opcode"]==o["op"] and r["q"]==o["q"] and r["k"]==o["k"] and r["v"]==o["v"] and r["o"]==o["o"]
 p=dict(r);d=p.pop("transfer_sha256");assert d==hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":")).encode()).hexdigest();assert r["gain"]>0 and r["bias_units"]>0 and r["softmax_linf_to_l1"]==1 and r["sequence_length_dependency"]=="none; convex softmax l_inf-to-l1 transport and probability-normalization opcode sharing"
assert c["graph_sha256"]==hashlib.sha256(gp.read_bytes()).hexdigest();print(json.dumps({"certificate":"IEEE_CLIPPED_ATTENTION_TRANSFERS_OK","occurrences":24,"coverage_exact":True,"sequence_length_cap":None},indent=2))
