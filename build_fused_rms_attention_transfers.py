#!/usr/bin/env python3
"""Structured RMSRadial -> attention zero-input-error transfer certificates."""
import hashlib,json
from decimal import Decimal,ROUND_CEILING,getcontext
from pathlib import Path
import torch
from safetensors.torch import load_file
getcontext().prec=100
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text())
cp=Path("outputs/flan_convex_reachable_bounds.json");convex=json.loads(cp.read_text());cb={x["index"]:x for x in convex["op_bounds"]}
rp=Path("outputs/flan_ieee_rmsnorm_transfers.json");rms={x["index"]:x for x in json.loads(rp.read_text())["transfers"]}
ap=Path("outputs/flan_ieee_attention_transfers.json");attention={x["index"]:x for x in json.loads(ap.read_text())["transfers"]}
W=load_file("work/google_flan/model.safetensors",device="cpu")
u=Decimal(2)**-24;eta=Decimal(2)**-150;scale=2**40
def gamma(k):return Decimal(k)*u/(Decimal(1)-Decimal(k)*u)
def support(name,rw):return (512.0**.5)*torch.linalg.vector_norm(W[name].double()*W[rw].double().unsqueeze(0),2,1)
def td(x):return torch.tensor(float(x),dtype=torch.float64)
records=[]
for i,o in enumerate(g["ops"]):
 if o["op"] not in {"SELF_ATTENTION_SEQUENCE","SELF_ATTENTION_KV","CROSS_ATTENTION"}:continue
 prev=g["ops"][i-1];assert prev["op"]=="RMSNORM" and prev["output"]==o["input"]
 rr=rms[i-1];rw=prev["weight"]
 en=Decimal(rr["normalized_error_bound"]);rt=Decimal(rr["rsqrt_relative_transport"])
 rmsw=W[rw].double().abs()
 # Error = radial_coeff * exact normalized output + additive box.  Keeping
 # the first term radial lets a following matrix use its ellipsoid support
 # instead of an absolute row sum.
 norm_round=u*Decimal(512).sqrt()*(Decimal(1)+u)/(Decimal(1)-Decimal(rr["relative_shifted_mean_error"])).sqrt()+eta
 additive=rmsw*float(norm_round)+float(u)*rmsw*(512.0**.5+float(en))+float(eta)
 mrw="encoder.final_layer_norm.weight" if o["op"]=="CROSS_ATTENTION" else rw
 qs=support(o["q"],rw).reshape(6,64);ks=support(o["k"],mrw).reshape(6,64);vs=support(o["v"],mrw).reshape(6,64)
 # Query always consumes this RMS output. Self-attention K/V do too; cross
 # attention K/V consume independently certified encoder memory.
 qextra=(float(rt)*qs+(W[o["q"]].double().abs()@additive).reshape(6,64))
 if o["op"]=="CROSS_ATTENTION": kextra=torch.zeros_like(ks);vextra=torch.zeros_like(vs)
 else:
  kextra=float(rt)*ks+(W[o["k"]].double().abs()@additive).reshape(6,64)
  vextra=float(rt)*vs+(W[o["v"]].double().abs()@additive).reshape(6,64)
 rq=qextra+float(gamma(1024))*(qs+qextra)+float(Decimal(1024)*eta)
 rk=kextra+float(gamma(1024))*(ks+kextra)+float(Decimal(1024)*eta)
 rv=vextra+float(gamma(1024))*(vs+vextra)+float(Decimal(1024)*eta)
 se0=(rq*ks+(qs+rq)*rk).sum(1)+float(gamma(128))*((qs+rq)*(ks+rk)).sum(1)+float(Decimal(128)*eta)
 rb=Decimal(attention[i]["relative_bias_abs_bound"])
 se=se0+float(u)*((qs*ks).sum(1)+se0+float(rb))+float(eta)
 pre=vs*se.unsqueeze(1)+rv
 O=W[o["o"]].double().abs();weighted=O@pre.reshape(-1);ro=float(gamma(768))*(O@(vs+pre).reshape(-1))+float(Decimal(768)*eta)
 err=(Decimal(str(float(weighted.max())))+Decimal(str(float(ro.max()))))*Decimal("1.000000000000001")
 bias=int((err*scale).to_integral_value(rounding=ROUND_CEILING))
 rec={"index":i,"rms_index":i-1,"opcode":"FUSED_RMS_ATTENTION","attention_opcode":o["op"],"rms_weight":rw,
      "error_constructor":"RMSRadial","radial_coefficient":str(rt),"additive_coordinate_error_max":str(float(additive.max())),"score_error_max":str(float(se.max())),
      "weighted_output_error_max":str(float(weighted.max())),"o_rounding_error_max":str(float(ro.max())),
      "bias_units":bias,"bias_real_upper":str(Decimal(bias)/scale),"error_units_per_real":scale,
      "applicability":"incoming error at RMS input is zero; cross-attention encoder memory error is zero",
      "status":"CERTIFIED_CHECKPOINT_DERIVED_STRUCTURED_ZERO_INPUT_TRANSFER"}
 payload=json.dumps(rec,sort_keys=True,separators=(",",":")).encode();rec["transfer_sha256"]=hashlib.sha256(payload).hexdigest();records.append(rec)
out={"language":"FLAN-FUSED-RMSRADIAL-ATTENTION-1","graph_sha256":hashlib.sha256(gp.read_bytes()).hexdigest(),
 "rms_transfers_sha256":hashlib.sha256(rp.read_bytes()).hexdigest(),"attention_transfers_sha256":hashlib.sha256(ap.read_bytes()).hexdigest(),
 "transfers":records,"scope":"all 24 syntactic RMSNorm-to-attention pairs; zero-input-error applicability is checked by the DAG audit",
 "status":"structured occurrence certificates; no prompt samples"}
path=Path("outputs/flan_fused_rms_attention_transfers.json");path.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"artifact":str(path),"occurrences":len(records),"min_bias_real":min(float(r["bias_real_upper"]) for r in records),"max_bias_real":max(float(r["bias_real_upper"]) for r in records)},indent=2))
