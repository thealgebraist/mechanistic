#!/usr/bin/env python3
"""Sequence-independent clipped transfers for all 24 FLAN attentions."""
import hashlib,json
from decimal import Decimal,ROUND_CEILING,getcontext
from pathlib import Path
from safetensors.torch import load_file
import torch
getcontext().prec=100
gp=Path("outputs/flan_full_graph.json");g=json.loads(gp.read_text());rp=Path("outputs/flan_reachable_state_bounds.json")
weights=load_file("work/google_flan/model.safetensors",device="cpu")
abstract=json.loads(Path("outputs/flan_ieee_add_transfers.json").read_text());env={x["output"]:Decimal(x["abs_bound"]) for x in abstract["op_output_bounds"]}
convex=json.loads(Path("outputs/flan_convex_reachable_bounds.json").read_text()); convex_by_index={x["index"]:x for x in convex["op_bounds"]}
norm={}
for l in Path("outputs/flan_reachable_norm_manifest.tsv").read_text().splitlines()[1:]:f=l.split("\t");norm[f[0]]=Decimal(f[6])
u=Decimal(2)**-24;eta=Decimal(2)**-150;scale=int(abstract["error_units_per_real"]);dk=Decimal(64)
def gamma(k):return Decimal(k)*u/(Decimal(1)-Decimal(k)*u)
def support_vec(weight,rms_weight):
 return (512.0**0.5)*torch.linalg.vector_norm(weights[weight].double()*weights[rms_weight].double().unsqueeze(0),ord=2,dim=1)
records=[]
for i,o in enumerate(g["ops"]):
 if o["op"] not in {"SELF_ATTENTION_SEQUENCE","SELF_ATTENTION_KV","CROSS_ATTENTION"}:continue
 localrec=convex_by_index[i];local=localrec["input_bounds"]; Bq=Decimal(str(local["input"]));Bm=Decimal(str(local["memory"])) if o["op"]=="CROSS_ATTENTION" else Bq
 Q=norm[o["q"]];K=norm[o["k"]];V=norm[o["v"]];O=norm[o["o"]]
 Qb=Decimal(str(localrec["projection_supports"]["q_max"]));Kb=Decimal(str(localrec["projection_supports"]["k_max"]));Vb=Decimal(str(localrec["projection_supports"]["v_max"]))
 qrw=g["ops"][i-1]["weight"]
 mrw="encoder.final_layer_norm.weight" if o["op"]=="CROSS_ATTENTION" else qrw
 qsup=support_vec(o["q"],qrw).reshape(6,64);ksup=support_vec(o["k"],mrw).reshape(6,64);vsup=support_vec(o["v"],mrw).reshape(6,64)
 if o["op"]=="SELF_ATTENTION_SEQUENCE":
  rbname=("encoder" if o["q"].startswith("encoder.") else "decoder")+".block.0.layer.0.SelfAttention.relative_attention_bias.weight"
  Rb=Decimal(str(float(weights[rbname].double().abs().max().item())))
 else: Rb=Decimal(0)
 score_gain=dk*(Q*Kb+Qb*K)
 # The Jacobian of softmax has induced l_inf-to-l1 norm <= 1.  The DSL uses
 # the same pinned probability-normalization opcode as the source backend.
 gain_real=O*(score_gain*Vb+V)
 qg=weights[o["q"]].double().abs().sum(1).reshape(6,64);kg=weights[o["k"]].double().abs().sum(1).reshape(6,64);vg=weights[o["v"]].double().abs().sum(1).reshape(6,64)
 score_gain_heads=(qg*ksup+qsup*kg).sum(1)
 pre_gain_vec=vsup*score_gain_heads.unsqueeze(1)+vg
 tensor_gain_real=Decimal(str(float((weights[o["o"]].double().abs()@pre_gain_vec.reshape(-1)).max().item())))
 gain=int(tensor_gain_real.to_integral_value(rounding=ROUND_CEILING))
 rq=gamma(1024)*Qb+Decimal(1024)*eta
 rk=gamma(1024)*Kb+Decimal(1024)*eta
 rv=gamma(1024)*Vb+Decimal(1024)*eta
 score_dot_error=dk*(rq*Kb+(Qb+rq)*rk)+gamma(128)*dk*(Qb+rq)*(Kb+rk)+Decimal(128)*eta
 score_error=score_dot_error+u*(dk*Qb*Kb+score_dot_error+Rb)+eta
 probability_l1_error=score_error
 # Tensor-level convex transport avoids repeating coordinate maxima 64 times.
 gt=lambda z: torch.tensor(float(z),dtype=torch.float64)
 rqv=gt(gamma(1024))*qsup+gt(Decimal(1024)*eta);rkv=gt(gamma(1024))*ksup+gt(Decimal(1024)*eta);rvv=gt(gamma(1024))*vsup+gt(Decimal(1024)*eta)
 se0=((rqv*ksup+(qsup+rqv)*rkv).sum(1)+gt(gamma(128))*((qsup+rqv)*(ksup+rkv)).sum(1)+gt(Decimal(128)*eta))
 semag=(qsup*ksup).sum(1)
 sev=se0+gt(u)*(semag+se0+gt(Rb))+gt(eta)
 prevec=vsup*sev.unsqueeze(1)+rvv
 Oabs=weights[o["o"]].double().abs(); weighted=Oabs@prevec.reshape(-1)
 base=Oabs@(vsup+prevec).reshape(-1)
 rov=gt(gamma(768))*base+gt(Decimal(768)*eta)
 output_vec=weighted+rov
 score_error_tensor=Decimal(str(float(sev.max().item())))
 pre_o_error_tensor=Decimal(str(float(prevec.max().item())))
 weighted_error_tensor=Decimal(str(float(weighted.max().item())))
 ro_tensor=Decimal(str(float(rov.max().item())))
 output_err=(weighted_error_tensor+ro_tensor)*Decimal("1.000000000000001")
 bias=int((output_err*scale).to_integral_value(rounding=ROUND_CEILING))
 rec={"index":i,"opcode":o["op"],"input":o["input"],"memory":o.get("memory"),"q":o["q"],"k":o["k"],"v":o["v"],"o":o["o"],
 "query_clip_bound":str(Bq),"memory_clip_bound":str(Bm),"q_norm":str(Q),"k_norm":str(K),"v_norm":str(V),"o_norm":str(O),"d_k":64,
 "q_convex_support":str(Qb),"k_convex_support":str(Kb),"v_convex_support":str(Vb),"relative_bias_abs_bound":str(Rb),
 "score_lipschitz":str(score_gain),"softmax_linf_to_l1":1,"error_units_per_real":scale,"gain":gain,"gain_real_bound":str(tensor_gain_real),"scalar_collapsed_gain_bound":str(gain_real),"bias_units":bias,"bias_real_upper":str(Decimal(bias)/scale),"q_projection_error_max":str(rq),"k_projection_error_max":str(rk),"v_projection_error_max":str(rv),"score_error_max":str(score_error_tensor),"probability_l1_error_max":str(score_error_tensor),"o_projection_error_max":str(ro_tensor),
 "source_contract":"binary32 RNE Q/K/V/O and score reductions; pinned probability-normalization opcode; no overflow; gradual underflow",
 "target_contract":"exact-real clipped projections and scores with the same pinned probability-normalization opcode, mask, and relative-position bias",
 "sequence_length_dependency":"none; convex softmax l_inf-to-l1 transport and probability-normalization opcode sharing",
 "tensor_convex_score_error":str(score_error_tensor),"tensor_convex_pre_o_error_max":str(pre_o_error_tensor),"tensor_convex_weighted_error_max":str(weighted_error_tensor),"tensor_convex_o_rounding_error_max":str(ro_tensor),
 "status":"CERTIFIED_UNDER_PROBABILITY_SIMPLEX_AND_IEEE754_CLIP_CONTRACT"}
 p=json.dumps(rec,sort_keys=True,separators=(",",":")).encode();rec["transfer_sha256"]=hashlib.sha256(p).hexdigest();records.append(rec)
assert len(records)==24
tsv=Path("outputs/flan_ieee_attention_transfers.tsv");tsv.write_text("index\tkind\tBq\tBm\tQ\tK\tV\tO\tQb\tKb\tVb\tRb\tscale\tgain\tbias\tweighted\to_round\ttensor_gain\n"+"".join(f"{r['index']}\t{r['opcode']}\t{r['query_clip_bound']}\t{r['memory_clip_bound']}\t{r['q_norm']}\t{r['k_norm']}\t{r['v_norm']}\t{r['o_norm']}\t{r['q_convex_support']}\t{r['k_convex_support']}\t{r['v_convex_support']}\t{r['relative_bias_abs_bound']}\t{scale}\t{r['gain']}\t{r['bias_units']}\t{r['tensor_convex_weighted_error_max']}\t{r['tensor_convex_o_rounding_error_max']}\t{r['gain_real_bound']}\n" for r in records))
out={"language":"FLAN-IEEE754-CLIPPED-ATTENTION-TRANSFER-1","checkpoint_sha256":g["checkpoint_sha256"],"graph_sha256":hashlib.sha256(gp.read_bytes()).hexdigest(),"reachable_bounds_sha256":hashlib.sha256(rp.read_bytes()).hexdigest(),"transfers":records,"tsv_sha256":hashlib.sha256(tsv.read_bytes()).hexdigest(),"status":"24/24 attention occurrences certified without a sequence-length cap"}
path=Path("outputs/flan_ieee_attention_transfers.json");path.write_text(json.dumps(out,indent=2)+"\n");print(json.dumps({"artifact":str(path),"occurrences":24,"max_gain":max(r['gain'] for r in records),"max_bias_units":max(r['bias_units'] for r in records)},indent=2))
