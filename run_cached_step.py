"""Replay the explicit-KV-cache decoder-step register program."""
import json, torch
from pathlib import Path
def T(x): return torch.tensor(x,dtype=torch.float32)
p=json.loads(Path("outputs/flan_cached_step.json").read_text()); f=json.loads(Path("outputs/flan_cached_step_fixture.json").read_text()); r={}
for o in p["ops"]:
 n=o["op"]
 if n.startswith("INPUT_"): r[o["name"]]=T(f[o["name"]])
 elif n=="RMSNORM":
  x=r[o["input"]]; r[o["output"]]=T(o["weight"])*x/torch.sqrt(torch.mean(x*x)+o["epsilon"])
 elif n=="ADD": r[o["output"]]=r[o["left"]]+r[o["right"]]
 elif n=="MATMUL": r[o["output"]]=T(o["weights"])@r[o["input"]]
 elif n=="SELF_ATTENTION_CACHE":
  q=(T(o["q"])@r[o["input"]]).reshape(6,64); w=torch.softmax(torch.einsum("hd,hld->hl",q,r[o["k_cache"]])+r[o["bias"]],dim=-1); z=torch.einsum("hl,hld->hd",w,r[o["v_cache"]]); r[o["output"]]=T(o["o"])@z.reshape(-1)
 elif n=="CROSS_ATTENTION":
  x,m=r[o["input"]],r[o["memory"]]; h,d=o["heads"],o["head_width"]; q=(T(o["q"])@x).reshape(h,d); k=(m@T(o["k"]).T).reshape(m.shape[0],h,d).permute(1,0,2); v=(m@T(o["v"]).T).reshape(m.shape[0],h,d).permute(1,0,2); w=torch.softmax(torch.einsum("hd,hld->hl",q,k),dim=-1); r[o["output"]]=T(o["o"])@torch.einsum("hl,hld->hd",w,v).reshape(-1)
 elif n=="GELU_GATE": r[o["output"]]=torch.nn.functional.gelu(r[o["left"]],approximate="tanh")*r[o["right"]]
 elif n=="HALT": pass
 else: raise ValueError(n)
err=float((r["y"]-T(p["target"])).abs().max()); print(json.dumps({"certificate":"CACHED_STEP_REPLAY_OK","max_replay_error":err,"program_bytes":Path("outputs/flan_cached_step.json").stat().st_size},indent=2)); assert err<2e-4
