"""Fixed-point-style replay of the cached FLAN register program.

Every input, weight, and intermediate register is rounded to Q fractional
bits. This is an arithmetic experiment; the reported bound is not inferred
from the experiment and must be supplied by a separate interval proof.
"""
import json, math, torch
from pathlib import Path

P=json.loads(Path("outputs/flan_cached_step.json").read_text()); F=json.loads(Path("outputs/flan_cached_step_fixture.json").read_text())
def run(bits):
 scale=2.0**bits
 def Q(x): return torch.round(x*scale)/scale
 def T(x): return Q(torch.tensor(x,dtype=torch.float64))
 r={}
 for o in P["ops"]:
  n=o["op"]
  if n.startswith("INPUT_"): r[o["name"]]=T(F[o["name"]])
  elif n=="RMSNORM":
   x=r[o["input"]]; w=T(o["weight"]); r[o["output"]]=Q(w*x/torch.sqrt(torch.mean(x*x,dim=-1,keepdim=True)+o["epsilon"]))
  elif n=="ADD": r[o["output"]]=Q(r[o["left"]]+r[o["right"]])
  elif n=="MATMUL": r[o["output"]]=Q(T(o["weights"])@r[o["input"]])
  elif n=="SELF_ATTENTION_CACHE":
   q=Q(T(o["q"])@r[o["input"]]).reshape(6,64); score=Q(torch.einsum("hd,hld->hl",q,r[o["k_cache"]])+r[o["bias"]]); w=Q(torch.softmax(score,dim=-1)); z=Q(torch.einsum("hl,hld->hd",w,r[o["v_cache"]])); r[o["output"]]=Q(T(o["o"])@z.reshape(-1))
  elif n=="CROSS_ATTENTION":
   x,m=r[o["input"]],r[o["memory"]]; h,d=o["heads"],o["head_width"]; q=Q(T(o["q"])@x).reshape(h,d); k=Q((m@T(o["k"]).T).reshape(m.shape[0],h,d).permute(1,0,2)); v=Q((m@T(o["v"]).T).reshape(m.shape[0],h,d).permute(1,0,2)); w=Q(torch.softmax(Q(torch.einsum("hd,hld->hl",q,k)),dim=-1)); r[o["output"]]=Q(T(o["o"])@Q(torch.einsum("hl,hld->hd",w,v).reshape(-1)))
  elif n=="GELU_GATE": r[o["output"]]=Q(torch.nn.functional.gelu(r[o["left"]].float(),approximate="tanh")*r[o["right"]])
  elif n=="HALT": pass
 err=float((r["y"].float()-torch.tensor(P["target"])).abs().max())
 return err
results={str(b):run(b) for b in (6,8,10,12,14,16)}
print(json.dumps({"program":"flan_cached_step.json","quantization":"round_to_2^-bits_after_each_opcode","max_abs_output_error":results},indent=2))
