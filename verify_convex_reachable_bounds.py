#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
p=Path("outputs/flan_convex_reachable_bounds.json");c=json.loads(p.read_text());m=Path("work/google_flan/model.safetensors");g=Path("outputs/flan_full_graph.json")
assert c["checkpoint_sha256"]==hashlib.sha256(m.read_bytes()).hexdigest() and c["graph_sha256"]==hashlib.sha256(g.read_bytes()).hexdigest()
assert c["sequence_length_cap"] is None and len(c["events"])==19
assert len(c["op_bounds"])==126 and [x["index"] for x in c["op_bounds"]]==sorted(x["index"] for x in c["op_bounds"])
assert all(v>1 for v in c["improvement"].values()) and c["final_bounds"]["logit_abs"]<c["previous_box_bounds"]["logit_abs"]
assert [x["register"] for x in c["events"] if ".layer" in x["register"]]==[f"enc_h.layer{i}" for i in range(8)]+[f"dec_h.layer{i}" for i in range(8)]
print(json.dumps({"certificate":"CONVEX_WEIGHTED_REACHABLE_BOUNDS_OK","events":19,"decoder_improvement":c["improvement"]["decoder_hidden"],"sequence_cap":None},indent=2))
