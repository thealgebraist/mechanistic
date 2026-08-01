#!/usr/bin/env python3
"""All-sequence reachable envelope using exact weighted-RMS support functions."""
import hashlib,json,math
from pathlib import Path
import torch
from safetensors.torch import load_file
model=Path("work/google_flan/model.safetensors");gp=Path("outputs/flan_full_graph.json")
W=load_file(str(model),device="cpu");g=json.loads(gp.read_text());d=g["config"]["d_model"];sd=math.sqrt(d)
def support(weight,rms_weight):
 return sd*torch.linalg.vector_norm(W[weight].double()*W[rms_weight].double().unsqueeze(0),ord=2,dim=1)
def attention(v,o,rms_weight):
 vs=support(v,rms_weight); out=torch.mv(W[o].double().abs(),vs);return float(out.max().item()),float(vs.max().item())
def mlp(w0,w1,wo,rms_weight):
 u=support(w0,rms_weight);v=support(w1,rms_weight);out=torch.mv(W[wo].double().abs(),u*v);return float(out.max().item()),float(u.max().item()),float(v.max().item())
embed=float(W["shared.weight"].double().abs().max().item());events=[];enc=embed
events.append({"register":"enc_h.embed","bound":enc})
for i in range(8):
 p=f"encoder.block.{i}";rn0=p+".layer.0.layer_norm.weight";a,v=attention(p+".layer.0.SelfAttention.v.weight",p+".layer.0.SelfAttention.o.weight",rn0);r=enc+a
 rn1=p+".layer.1.layer_norm.weight";m,u0,u1=mlp(p+".layer.1.DenseReluDense.wi_0.weight",p+".layer.1.DenseReluDense.wi_1.weight",p+".layer.1.DenseReluDense.wo.weight",rn1);enc=r+m
 events.append({"register":f"enc_h.layer{i}","attention":a,"v_coordinate":v,"mlp":m,"gate_coordinate":u0,"linear_coordinate":u1,"bound":enc})
enc_final="encoder.final_layer_norm.weight";memory_sup=sd*float(W[enc_final].double().abs().max().item())
events.append({"register":"encoder_memory","bound":memory_sup,"ellipsoid_weight":enc_final})
dec=embed;events.append({"register":"dec_h.embed","bound":dec})
for i in range(8):
 p=f"decoder.block.{i}";rn0=p+".layer.0.layer_norm.weight";a,v=attention(p+".layer.0.SelfAttention.v.weight",p+".layer.0.SelfAttention.o.weight",rn0);r=dec+a
 rn1=p+".layer.1.layer_norm.weight";ca,cv=attention(p+".layer.1.EncDecAttention.v.weight",p+".layer.1.EncDecAttention.o.weight",enc_final);cr=r+ca
 rn2=p+".layer.2.layer_norm.weight";m,u0,u1=mlp(p+".layer.2.DenseReluDense.wi_0.weight",p+".layer.2.DenseReluDense.wi_1.weight",p+".layer.2.DenseReluDense.wo.weight",rn2);dec=cr+m
 events.append({"register":f"dec_h.layer{i}","self_attention":a,"self_v_coordinate":v,"cross_attention":ca,"cross_v_coordinate":cv,"mlp":m,"gate_coordinate":u0,"linear_coordinate":u1,"bound":dec})
finalw="decoder.final_layer_norm.weight";lm_support=support("lm_head.weight",finalw);logit=float(lm_support.max().item());readout=sd*float(W[finalw].double().abs().max().item())
old=json.loads(Path("outputs/flan_reachable_state_bounds.json").read_text())["final_bounds"]
# Replay the JSON program with occurrence-indexed bounds; repeated names such
# as enc_h and dec_h are intentionally overwritten only after each occurrence.
env={};ellipsoid={};op_bounds=[]
for index,op in enumerate(g["ops"]):
 kind=op["op"];out_name=op.get("output");bound=None;geometry=None
 if kind=="EMBED":bound=embed
 elif kind=="RMSNORM":
  bound=sd*float(W[op["weight"]].double().abs().max().item());ellipsoid[out_name]=op["weight"]
 elif kind in {"SELF_ATTENTION_SEQUENCE","SELF_ATTENTION_KV"}:
  rw=ellipsoid[op["input"]];bound,_=attention(op["v"],op["o"],rw);geometry={"q_max":float(support(op["q"],rw).max()),"k_max":float(support(op["k"],rw).max()),"v_max":float(support(op["v"],rw).max())}
 elif kind=="CROSS_ATTENTION":
  qw=ellipsoid[op["input"]];mw=ellipsoid[op["memory"]];bound,_=attention(op["v"],op["o"],mw);geometry={"q_max":float(support(op["q"],qw).max()),"k_max":float(support(op["k"],mw).max()),"v_max":float(support(op["v"],mw).max())}
 elif kind=="GATED_MLP":
  rw=ellipsoid[op["input"]];bound,u0,u1=mlp(op["wi_0"],op["wi_1"],op["wo"],rw);geometry={"wi_0_max":u0,"wi_1_max":u1}
 elif kind=="ADD":bound=env[op["left"]]+env[op["right"]]
 elif kind=="MATMUL":bound=float(support(op["weight"],ellipsoid[op["input"]]).max().item())
 elif kind=="SOFTMAX":bound=1.0
 if out_name is not None and bound is not None:
  inputs={k:env[v] for k,v in op.items() if k in {"input","left","right","memory"} and v in env}
  env[out_name]=bound;op_bounds.append({"index":index,"opcode":kind,"output":out_name,"input_bounds":inputs,"abs_bound":bound,"ellipsoid_weight":ellipsoid.get(out_name),"projection_supports":geometry})
assert abs(env["logits"]-logit)<1e-9
out={"language":"FLAN-CONVEX-WEIGHTED-REACHABLE-1","checkpoint_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),"graph_sha256":hashlib.sha256(gp.read_bytes()).hexdigest(),
 "scope":"every backend-valid nonempty finite encoder sequence and every finite decoder continuation","sequence_length_cap":None,
 "method":"weighted RMS ellipsoid support -> attention convex hull / coordinate support -> gated-product support -> residual triangle inequality",
 "events":events,"op_bounds":op_bounds,"final_bounds":{"encoder_hidden":enc,"encoder_memory":memory_sup,"decoder_hidden":dec,"readout_hidden":readout,"logit_abs":logit},
 "previous_box_bounds":old,"improvement":{"encoder_hidden":old["encoder_hidden"]/enc,"decoder_hidden":old["decoder_hidden"]/dec,"logit_abs":old["logit_abs"]/logit},
 "status":"checkpoint-derived universal convex envelope; no sampled prompts used"}
path=Path("outputs/flan_convex_reachable_bounds.json");path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"artifact":str(path),"final_bounds":out["final_bounds"],"improvement":out["improvement"]},indent=2))
