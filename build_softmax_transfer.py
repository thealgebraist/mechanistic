#!/usr/bin/env python3
"""Universal but diameter-level transfer for the final vocabulary softmax."""
import hashlib,json
from pathlib import Path
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text())
scale=json.loads(Path("outputs/flan_ieee_add_transfers.json").read_text())["error_units_per_real"]
ops=[(i,o) for i,o in enumerate(g["ops"]) if o["op"]=="SOFTMAX"];assert len(ops)==1;i,o=ops[0]
r={"index":i,"opcode":"SOFTMAX","input":o["input"],"output":o["output"],"error_units_per_real_and_tv":scale,
   "gain":1,"bias_units":scale,"tv_diameter_units":scale,
   "theorem":"TV(source_float_softmax(x), exact_softmax(y)) <= min(1, ||x-y||_inf + 1)",
   "source_contract":"softmax output denotes a probability law over the 32128-token vocabulary",
   "target_contract":"exact-real softmax probability law",
   "quality":"TRIVIAL_DIAMETER_BOUND; occurrence coverage only, not a nontrivial approximation guarantee",
   "status":"CERTIFIED_PROBABILITY_SIMPLEX_DIAMETER"}
p=json.dumps(r,sort_keys=True,separators=(",",":")).encode();r["transfer_sha256"]=hashlib.sha256(p).hexdigest()
out={"language":"FLAN-SOFTMAX-TV-TRANSFER-1","graph_sha256":hashlib.sha256(gp.read_bytes()).hexdigest(),"transfer":r,"nontrivial":False}
path=Path("outputs/flan_softmax_tv_transfer.json");path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"artifact":str(path),"index":i,"gain":1,"bias_units":scale,"nontrivial":False},indent=2))
